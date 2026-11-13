"""Stairs evaluation of a blind Go2 policy on the training stair tiles.

Robots are pinned to the stair tiles of Unitree-Go2-Blind (4 step-height bands from
5 to 18 cm x up/down x narrow/wide treads, play config). Two tests:
  traverse : forward command from the tile center outwards; success = the base gets
             past the last step (Chebyshev distance >= 3.6 m) without terminating.
  stop     : walk outwards, zero the command once the base is mid-flight (distance
             >= 2.2 m) and hold it for 4 s; success = no fall and < 5 cm drift over
             the last 2 s.

  python scripts/go2_blind/eval_stairs.py logs/rsl_rl/go2_blind/<run>/model_N.pt --vx 0.5
"""

import argparse
import os
from collections import Counter
from dataclasses import asdict

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.velocity.blind_env_cfg import STAIR_TERRAIN_NAMES

NUM_ROWS = 4
STEP_RANGE = (0.05, 0.18)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--task", default="Unitree-Go2-Blind-Scratch", help="Blind task whose play config matches the policy.")
  parser.add_argument("checkpoint")
  parser.add_argument("--vx", type=float, default=0.5)
  parser.add_argument("--wz", type=float, default=0.0)
  parser.add_argument("--envs-per-tile", type=int, default=6)
  parser.add_argument("--seed", type=int, default=0)
  # 2026-11-12 (doc/redesign.md, the unified acceptance): opt-in, without them the check runs as before
  parser.add_argument("--payload", default=None, help="kg,x,y,z: a fixed point mass on the base (payload_opt.py)")
  parser.add_argument("--metrics", action="store_true",
                      help="also count kicks / slips / placements with src/metrics (tinynav's counters) while crossing")
  parser.add_argument("--causes", action="store_true",
                      help="also break the falls down (2026-11-13, go2-depth): the termination terms, the bodies in illegal "
                           "contact, the base's pitch / roll and its move along the flight over the last 1 s, taken just "
                           "before the reset; by run phase and by band (a reading: lines start with 'causes')")
  args = parser.parse_args()

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.seed = args.seed
  gen = env_cfg.scene.terrain.terrain_generator
  gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k in STAIR_TERRAIN_NAMES}
  for v in gen.sub_terrains.values():
    v.proportion = 1.0
  gen.num_rows = NUM_ROWS
  gen.num_cols = len(gen.sub_terrains) * 2
  env_cfg.events.pop("randomize_terrain")
  num_envs = gen.num_rows * gen.num_cols * args.envs_per_tile
  env_cfg.scene.num_envs = num_envs
  if args.payload:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blind", "eval"))
    import payload_opt  # noqa: E402

    payload_opt.apply(env_cfg, args.payload)
  if args.metrics:
    from src.metrics import mj as metrics_mj

    metrics_mj.add_sensor(env_cfg)

  agent_cfg = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(args.task)(venv, asdict(agent_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")

  dev, dt = env.device, env.step_dt
  terrain = env.scene.terrain
  robot = env.scene["robot"]
  feet_sensor = env.scene["feet_ground_contact"]
  leg_sensor = env.scene["leg_ground_touch"]
  cmd_term = env.command_manager.get_term("twist")

  # Column -> sub-terrain name (same rule as the terrain generator).
  names = list(gen.sub_terrains.keys())
  cum = np.cumsum(np.full(len(names), 1.0 / len(names)))
  col_name = [names[np.min(np.where(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]

  tile = torch.arange(num_envs, device=dev) // args.envs_per_tile
  levels = tile % gen.num_rows
  types = tile // gen.num_rows
  env_level = levels.cpu().numpy()
  env_dir = np.array(["down" if "down" in col_name[t] else "up" for t in types.cpu().numpy()])
  env_tread = np.array(["narrow" if "narrow" in col_name[t] else "wide" for t in types.cpu().numpy()])

  def pin_and_reset():
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = types
    terrain.env_origins[:] = terrain.terrain_origins[levels, types]
    env.reset()
    return venv.get_observations()

  def set_cmd(c, ids=None):
    t = torch.tensor(c, device=dev, dtype=torch.float32)
    if ids is None:
      cmd_term.vel_command_b[:] = t
    else:
      cmd_term.vel_command_b[ids] = t
    cmd_term.time_left[:] = 1e9

  def cheb_dist():
    d = (robot.data.root_link_pos_w[:, :2] - terrain.env_origins[:, :2]).abs()
    return d.max(dim=1).values

  def band(level):
    lo, hi = STEP_RANGE
    return f"{100 * (lo + (hi - lo) * level / NUM_ROWS):.0f}-{100 * (lo + (hi - lo) * (level + 1) / NUM_ROWS):.0f}cm"

  # --causes: env.step resets a terminated robot inside the step (ManagerBasedRlEnv._reset_idx), so its state is taken
  # in a wrapper of _reset_idx, before the reset: which termination terms fired, which bodies touched (the
  # illegal_contact term's sensor, as src/tasks/unified/diagnostics.py reads it), the base's pitch / roll and its
  # Chebyshev distance from the tile centre (the flight's direction: + outwards); the distance 1 s before from a ring
  hist_len = max(1, int(round(1.0 / dt)))
  cap = None
  falls_rec = []
  if args.causes:
    tm = env.termination_manager
    cause_names = [n for n in tm.active_terms if not tm.get_term_cfg(n).time_out]
    ic_sensor, ic_thr, ic_bodies = None, 10.0, []
    if "illegal_contact" in cause_names:
      try:
        prm = tm.get_term_cfg("illegal_contact").params
        ic_sensor = env.scene[prm["sensor_name"]]
        ic_thr = float(prm.get("force_threshold", 10.0))
        ic_bodies = [sl.primary_name for sl in ic_sensor._slots][:62]
      except Exception:
        ic_sensor, ic_bodies = None, []
    shifts = torch.arange(max(len(ic_bodies), 1), device=dev)
    cap = {"cause": torch.zeros(num_envs, dtype=torch.long, device=dev),
           "body": torch.zeros(num_envs, dtype=torch.long, device=dev),
           "pitch": torch.zeros(num_envs, device=dev), "roll": torch.zeros(num_envs, device=dev),
           "d": torch.zeros(num_envs, device=dev), "ring": torch.zeros(hist_len, num_envs, device=dev)}
    orig_reset_idx = env._reset_idx

    def reset_capture(env_ids=None):
      try:
        if env_ids is not None and len(env_ids) > 0 and env.common_step_counter > 0:
          ids = env_ids
          c = torch.zeros(len(ids), dtype=torch.long, device=dev)
          for j, n in enumerate(cause_names):
            c |= tm.get_term(n)[ids].long() << j
          cap["cause"][ids] = c
          if ic_sensor is not None and ic_bodies:
            dd = ic_sensor.data
            fh = getattr(dd, "force_history", None)
            hit = (torch.norm(fh[ids], dim=-1) > ic_thr).any(dim=-1) if fh is not None else dd.found[ids] > 0
            hit = hit.reshape(len(ids), -1)[:, :len(ic_bodies)]
            cap["body"][ids] = (hit.long() << shifts[:hit.shape[1]]).sum(dim=1)
          g = robot.data.projected_gravity_b[ids]
          cap["pitch"][ids] = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))   # + nose down
          cap["roll"][ids] = torch.rad2deg(torch.atan2(g[:, 1], -g[:, 2]))
          cap["d"][ids] = cheb_dist()[ids]
      except Exception:
        pass
      return orig_reset_idx(env_ids)

    env._reset_idx = reset_capture

  def note_falls(mask, phase, k):
    """The robots of mask fell in this step (their state taken before the reset): one record each."""
    if cap is None or not bool(mask.any()):
      return
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    back = cap["d"][ids] - cap["ring"][(k + 1) % hist_len, ids] if k + 1 >= hist_len else cap["d"][ids] - cap["ring"][0, ids]
    for i, e in enumerate(ids.tolist()):
      falls_rec.append({"phase": phase, "band": band(int(env_level[e])), "dir": str(env_dir[e]),
                        "cause": int(cap["cause"][e]), "body": int(cap["body"][e]), "pitch": float(cap["pitch"][e]),
                        "roll": float(cap["roll"][e]), "move": float(back[i])})

  def note_ring(k):
    if cap is not None:
      cap["ring"][k % hist_len] = cheb_dist()

  # ---- Traverse ----
  obs = pin_and_reset()
  set_cmd((args.vx, 0.0, args.wz))
  alive = torch.ones(num_envs, dtype=torch.bool, device=dev)
  success = torch.zeros_like(alive)
  meter = None
  if args.metrics:
    meter = metrics_mj.StairsMeter(env, torch.arange(num_envs, device=dev), wheeled="wheel_vel" in env_cfg.actions,
                                   ema_steps=1e12, min_steps=1.0)
    # the counts per tile too (2026-11-12, go2-depth: which steps drive a reading): band x direction x tread
    tile_groups = {f"{band(lvl)} {d} {tr}": (env_level == lvl) & (env_dir == d) & (env_tread == tr)
                   for lvl in range(gen.num_rows) for d in ("up", "down") for tr in ("wide", "narrow")}
    meter.groups = lambda env_, ids_: tile_groups
  leg_hits = torch.zeros(num_envs, device=dev)
  steps_alive = torch.zeros(num_envs, device=dev)
  max_dist = torch.zeros(num_envs, device=dev)
  # the speed on the flight (2026-11-12, doc/redesign.md 2.2: the typical payload's stairs speed against the unloaded):
  # when the base passes the 2 m central platform's edge (1.0 m) and the last step (3.6 m)
  t_enter = torch.full((num_envs,), -1.0, device=dev)
  t_reach = torch.full((num_envs,), -1.0, device=dev)
  # heading off the spawn yaw on the flight (2026-11-13, go2-depth; the user: "heading需要验收"): as tinynav's
  # accept_stairs.py -- |yaw - spawn yaw| in degrees, averaged over the steps the base is within 0.5 m of the steps
  # (Chebyshev distance 0.5-4.1 m: the platform edge at 1.0 m, the last step at 3.6 m), until the robot gets past
  yaw0 = robot.data.heading_w.clone()
  hd_sum = torch.zeros(num_envs, device=dev)
  hd_n = torch.zeros(num_envs, device=dev)
  for k in range(int(round(3.6 / max(abs(args.vx), 0.1) * 2.0 / dt))):
    note_ring(k)
    with torch.no_grad():
      obs, _, _, _ = venv.step(policy(obs))
    if meter is not None:
      meter.step(env)
    term = env.termination_manager.terminated
    active = alive & ~success
    leg_hits += (active & (leg_sensor.data.found > 0).flatten(1).any(1)).float()
    steps_alive += active.float()
    note_falls(term & active, "traverse", k)
    alive &= ~(term & active)
    max_dist = torch.where(alive, torch.maximum(max_dist, cheb_dist()), max_dist)
    d_now = cheb_dist()
    on_fl = alive & ~success & (d_now >= 0.5) & (d_now <= 4.1)
    dyaw = torch.atan2(torch.sin(robot.data.heading_w - yaw0), torch.cos(robot.data.heading_w - yaw0)).abs() * 57.29578
    hd_sum += torch.where(on_fl, dyaw, torch.zeros_like(dyaw))
    hd_n += on_fl.float()
    t_enter = torch.where(alive & (t_enter < 0) & (d_now >= 1.0), torch.full_like(t_enter, (k + 1) * dt), t_enter)
    t_reach = torch.where(alive & ~success & (d_now >= 3.6), torch.full_like(t_reach, (k + 1) * dt), t_reach)
    success |= alive & (d_now >= 3.6)
  trav_ok = success.cpu().numpy()
  trav_fell = (~alive).cpu().numpy()
  te, tr = t_enter.cpu().numpy(), t_reach.cpu().numpy()
  flight_speed = np.where(trav_ok & (te >= 0) & (tr > te), (3.6 - 1.0) / np.maximum(tr - te, 1e-6), np.nan)
  leg_frac = (leg_hits / steps_alive.clamp(min=1)).cpu().numpy()
  max_dist = max_dist.cpu().numpy()
  hd_s, hd_c = hd_sum.cpu().numpy(), hd_n.cpu().numpy()

  # ---- Stop mid-flight ----
  obs = pin_and_reset()
  set_cmd((args.vx, 0.0, args.wz))
  stopped_at = torch.full((num_envs,), -1, dtype=torch.long, device=dev)
  alive = torch.ones(num_envs, dtype=torch.bool, device=dev)
  fall_after = torch.full((num_envs,), -2, dtype=torch.long, device=dev)   # -2: did not fall
  pos_hist, contact_hist = [], []
  hold = int(4.0 / dt)
  for t in range(int(10.0 / dt) + hold):
    newly = (stopped_at < 0) & (cheb_dist() >= 2.2) & alive
    if newly.any():
      set_cmd((0.0, 0.0, 0.0), newly)
      stopped_at[newly] = t
    note_ring(t)
    with torch.no_grad():
      obs, _, _, _ = venv.step(policy(obs))
    term = env.termination_manager.terminated
    in_hold = (stopped_at >= 0) & (t - stopped_at < hold)
    fell_now = term & ((stopped_at < 0) | in_hold) & alive
    if cap is not None:
      since = t - stopped_at
      note_falls(fell_now & (stopped_at < 0), "stop: approach", t)
      note_falls(fell_now & (stopped_at >= 0) & (since < int(round(2.0 / dt))), "stop: 0-2 s", t)
      note_falls(fell_now & (stopped_at >= 0) & (since >= int(round(2.0 / dt))), "stop: 2-4 s", t)
    # when (2026-11-13, go2-depth): the steps since the stop command at the fall (-1: before it, on the approach)
    fall_after = torch.where(fell_now, torch.where(stopped_at >= 0, t - stopped_at, torch.full_like(stopped_at, -1)),
                             fall_after)
    alive &= ~fell_now
    pos_hist.append(robot.data.root_link_pos_w[:, :2].clone())
    contact_hist.append((feet_sensor.data.found > 0).flatten(1).all(1).clone())
    if ((stopped_at >= 0) & (t - stopped_at >= hold) | ~alive).all():
      break
  pos_hist = torch.stack(pos_hist).cpu().numpy()
  contact_hist = torch.stack(contact_hist).cpu().numpy()
  stopped_np, alive_np = stopped_at.cpu().numpy(), alive.cpu().numpy()
  drift = np.full(num_envs, np.nan)
  feet_down = np.full(num_envs, np.nan)
  for b in range(num_envs):
    s = stopped_np[b]
    if s >= 0 and alive_np[b] and s + hold <= len(pos_hist):
      a, e = s + hold // 2, s + hold - 1
      drift[b] = np.linalg.norm(pos_hist[e, b] - pos_hist[a, b])
      feet_down[b] = contact_hist[a : e + 1, b].mean()
  reached = stopped_np >= 0
  stop_ok = reached & alive_np & (np.nan_to_num(drift, nan=1.0) < 0.05)

  # ---- Report ----
  print(f"\ncheckpoint: {args.checkpoint}\ncommand vx={args.vx} wz={args.wz}, "
        f"{num_envs // (NUM_ROWS * 2)} robots per (band, direction)")
  print(f"{'band':>9} {'dir':>4} | {'traverse ok':>11} {'fell':>5} {'leg touch':>9} {'reach m':>7} | "
        f"{'stop ok':>7} {'fell':>5} {'drift cm':>8} {'4 feet':>6}")
  for lvl in range(NUM_ROWS):
    for d in ("up", "down"):
      m = (env_level == lvl) & (env_dir == d)
      mr = m & reached
      drift_cm = 100 * np.nanmean(drift[mr]) if np.isfinite(drift[mr]).any() else float("nan")
      feet_frac = np.nanmean(feet_down[mr]) if np.isfinite(feet_down[mr]).any() else float("nan")
      print(f"{band(lvl):>9} {d:>4} | {trav_ok[m].mean():11.0%} {trav_fell[m].mean():5.0%} {leg_frac[m].mean():9.1%} "
            f"{np.median(max_dist[m]):7.2f} | {stop_ok[m].mean():7.0%} {(~alive_np[m]).mean():5.0%} {drift_cm:8.1f} {feet_frac:6.2f}")
  # (these lines start with "speed": the table's parsers -- accept_tables, accept_judge -- read lines starting with a band)
  print("speed on the flight (m/s, the platform edge at 1.0 m to the last step at 3.6 m; the robots that got past it):")
  for lvl in range(NUM_ROWS):
    for d in ("up", "down"):
      m = (env_level == lvl) & (env_dir == d) & np.isfinite(flight_speed)
      sp = float(np.mean(flight_speed[m])) if m.any() else float("nan")
      print(f"speed {band(lvl):>9} {d:>4} | {sp:.3f} m/s x{sp / max(abs(args.vx), 1e-6):.2f} of the command, n {int(m.sum())}")
  for tread in ("narrow", "wide"):
    m = env_tread == tread
    print(f"tread {tread:>6}: traverse ok {trav_ok[m].mean():.0%}, stop ok {stop_ok[m].mean():.0%}")
  print(f"overall: traverse ok {trav_ok.mean():.0%}, fell {trav_fell.mean():.0%}; stop ok {stop_ok.mean():.0%}")
  # exact counts (2026-11-12, doc/redesign.md 2.2's "0 falls": the table's whole percents read 1-2 robots of 72 as 0%)
  print(f"falls: traverse {int(trav_fell.sum())} of {num_envs}, stop {int((~alive_np).sum())} of {num_envs}")
  # the stop run's falls by when (a reading; "falls:" above is what the judges read): on the approach (walking, before
  # the stop command at 2.2 m), within 2 s of the command, later in the 4 s hold
  fa = fall_after.cpu().numpy()
  two = int(round(2.0 / dt))
  print(f"stop-run falls by when: approach {int((fa == -1).sum())}, 0-2 s after the stop {int(((fa >= 0) & (fa < two)).sum())}, "
        f"2-4 s after {int((fa >= two).sum())}")
  # the traverse's robots that neither fell nor got past the last step in time (2026-11-13, go2-depth: As2 stopped at
  # the first riser -- not a fall, so the falls miss it): a reading, its own line (accept_tables parses the band rows)
  stalled = ~trav_ok & ~trav_fell
  st_parts = []
  for lvl in range(NUM_ROWS):
    for d in ("up", "down"):
      m = (env_level == lvl) & (env_dir == d)
      st_parts.append(f"{band(lvl)} {d} {stalled[m].mean():.0%}")
  print(f"stalled on the traverse (neither fell nor got past the last step): {stalled.mean():.1%} of {num_envs}; "
        + ", ".join(st_parts))
  # heading on the flight (tinynav's quantity, pooled over the robot-steps as its mean): all, then per band and direction
  hd_all = hd_s.sum() / max(hd_c.sum(), 1.0)
  hd_parts = []
  for lvl in range(NUM_ROWS):
    for d in ("up", "down"):
      m = (env_level == lvl) & (env_dir == d)
      hd_parts.append(f"{band(lvl)} {d} {hd_s[m].sum() / max(hd_c[m].sum(), 1.0):.1f}")
  print(f"heading on the flight: {hd_all:.2f} deg mean |yaw - spawn yaw| within 0.5 m of the steps "
        f"({hd_c.sum():.0f} robot-steps); " + ", ".join(hd_parts))
  if meter is not None:
    s = meter.sums
    if meter.wheeled:
      ups = s.get("ups", 0.0)
      print(f"metrics (src/metrics): step-ups {ups:.0f}, slide-back {meter.rate('slide_back') or 0.0:.3f} of step-ups, "
            f"idle start {meter.rate('idle_start') or 0.0:.3f} / end {meter.rate('idle_end') or 0.0:.3f} per step-up")
    else:
      steps = (s.get("ups", 0.0) + s.get("downs", 0.0)) / meter.feet
      print(f"metrics (src/metrics): steps {steps:.0f}, kicks {meter.rate('kicks') or 0.0:.3f} / step, "
            f"slips {meter.rate('slips') or 0.0:.4f} / step, mid {meter.rate('mid') or 0.0:.3f}, "
            f"edge {meter.rate('edge') or 0.0:.3f} of {s.get('placed', 0.0):.0f} placements")
    # the same counts by tile (lines start with "tile": the table parsers read lines starting with a band)
    for g in tile_groups:
      gs = meter.group_sums.get(g, {})
      if meter.wheeled:
        if gs.get("ups", 0.0) > 0:
          r = meter.rate("slide_back", g)
          print(f"tile {g}: step-ups {gs['ups']:.0f}, slide-back {'-' if r is None else f'{r:.3f}'}")
      else:
        st = (gs.get("ups", 0.0) + gs.get("downs", 0.0)) / meter.feet
        if st > 0:
          v = {k: meter.rate(k, g) for k in ("kicks", "slips", "mid", "edge")}
          print(f"tile {g}: steps {st:.0f}, " + ", ".join(f"{k} {'-' if x is None else f'{x:.3f}'}" for k, x in v.items()))
  if cap is not None:
    # the falls broken down (--causes; lines start with "causes": the table parsers read lines starting with a band)
    def summary(recs):
      n = len(recs)
      terms = Counter(nm for r in recs for j, nm in enumerate(cause_names) if r["cause"] >> j & 1)
      bodies = Counter(nm for r in recs for j, nm in enumerate(ic_bodies) if r["body"] >> j & 1)
      back = sum(1 for r in recs if r["move"] <= -0.05)
      pit = np.array([r["pitch"] for r in recs])
      rol = np.abs(np.array([r["roll"] for r in recs]))
      return (f"n {n} | terms " + ", ".join(f"{k} {v / n:.0%}" for k, v in terms.most_common()) + " | bodies "
              + (", ".join(f"{k} {v / n:.0%}" for k, v in bodies.most_common(5)) or "-")
              + f" | moved back >= 5 cm in the last 1 s {back / n:.0%} | pitch p50 {np.median(pit):+.0f} deg (+ nose down, "
              f"p10 {np.percentile(pit, 10):+.0f}, p90 {np.percentile(pit, 90):+.0f}) | |roll| p50 {np.median(rol):.0f} deg")
    print(f"causes of the falls (--causes, taken before the reset): {len(falls_rec)} falls; termination terms {cause_names}")
    for ph in ("traverse", "stop: approach", "stop: 0-2 s", "stop: 2-4 s"):
      recs = [r for r in falls_rec if r["phase"] == ph]
      if recs:
        print(f"causes {ph:>15}: " + summary(recs))
    for lvl in range(NUM_ROWS):
      for d in ("up", "down"):
        recs = [r for r in falls_rec if r["band"] == band(lvl) and r["dir"] == d]
        if recs:
          print(f"causes {band(lvl)} {d}: " + summary(recs))
  env.close()


if __name__ == "__main__":
  main()
