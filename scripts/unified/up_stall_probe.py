#!/usr/bin/env python3
"""up_stall_probe.py TASK CHECKPOINT [--num-envs N] [--steps S] [--reward-rows 3,4,5] (go2-home 2026-11-13, written
on the As2 teacher that parked before the first riser; taken into the repo by go2-depth): why the climbing row falls
without falls, and whether the rewards pay for climbing. Built like scripts/unified/fail_probe.py (the training env
of TASK, the checkpoint restored as a resume does, the mean action, no evaluation robots), but every robot stays on
the climbing columns (stairs_up_narrow / stairs_up_wide / walled_up, drawn as the training draws columns) on a row
spread evenly over all rows, for the whole run: after each reset's curriculum the robot is put back on its column and
row, and the curriculum's rows (env.u_rows) are put back to the checkpoint's so the progress p does not move with the
spread rows (the clock still moves it: about +0.03 a minute).
Measures, on those columns:
  1. the episodes that did not fail (no fall, no illegal contact): how far from the spawn point (the terrain origin,
     as the curriculum measures) they ended, p10/p50/p90; the share past 3 m (the curriculum's promotion) and the share
     the curriculum demotes for a short walk (< 0.5 x the commanded distance); by row and by payload;
  2. while the forward command is above 0.3 m/s: the forward speed (base frame) over the commanded one, and the share
     of time the planar speed is below 0.1 m/s, by phase -- before the first riser (no foot over a step yet), at the
     first riser (a foot over a step, a rear foot still on the bottom), on the flight (both rear feet over steps);
  3. where those stops are: the phase and the distance from the spawn, and where the not-promoted episodes ended;
  4. the reward account: on rows --reward-rows, the same settled forward-command steps in three classes -- A stopped
     before the first riser (|v| < 0.1), C walking before it (|v| >= 0.1), B climbing (both rear feet over steps) --
     and each reward term's mean per second (the reward manager's weighted rate, Episode_Reward's names) per class,
     sorted by |A - B|, with each class's total and the verdict "climbing minus stopped" (it has to be above zero:
     a policy that earns more standing before the riser than climbing parks there). A step's class is read before
     the step, its reward is the step's (envs that end in the step left out). The cost of the first step itself
     (a foot on a step, the rear feet not yet) is in no class.
A foot's ground is the terrain_scan hit nearest its site (hits above the foot excluded); "over a step" is more than
2.5 cm above the episode's spawn ground (the scan's hits near the base at the episode's first step).
Section 4 reads RewardManager._step_reward (a private field of mjlab). Run with UNIFIED_TASKS=1 from the repo root
(the file also runs from logs/probe/ of a box's tree)."""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("UNIFIED_TASKS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["U_EVAL_ENVS"] = "0"

import numpy as np  # noqa: E402
import torch  # noqa: E402

UP = ("stairs_up_narrow", "stairs_up_wide", "walled_up")
STEP_UP = 0.025
MIN_CLASS_STEPS = 500   # fewer steps in a class than this: its means are not read as a verdict


def column_names(env) -> list[str]:
  g = env.scene.terrain.cfg.terrain_generator
  names = list(g.sub_terrains.keys())
  prop = np.array([c.proportion for c in g.sub_terrains.values()], dtype=float)
  cum = np.cumsum(prop / prop.sum())
  return [names[int(np.min(np.where(i / g.num_cols + 0.001 < cum)[0]))] for i in range(g.num_cols)]


def q3(v) -> str:
  v = np.asarray(v, dtype=float)
  return "-" if v.size == 0 else "/".join(f"{x:.2f}" for x in np.percentile(v, (10, 50, 90)))


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("task")
  ap.add_argument("checkpoint")
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--steps", type=int, default=3000)
  ap.add_argument("--reward-rows", default="3,4,5", help="rows for the per-term reward split (4.)")
  ap.add_argument("--top", type=int, default=25)
  ap.add_argument("--reward-payload", default="", help="LO,HI (shares of the full payload): the reward split (4.) "
                  "over the robots carrying that much only, e.g. 0.75,1 -- the heavy ones")
  a = ap.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401

  cfg = load_env_cfg(a.task)
  cfg.scene.num_envs = a.num_envs
  failed_now: list[np.ndarray] = []
  fail_why: dict[int, list] = {}   # robot -> [the terminations, the bodies in contact] at this step (5.)

  def wrap(name):
    term = cfg.terminations.get(name)
    if term is None:
      return
    orig = term.func

    def f(env, **kw):
      out = orig(env, **kw)
      ids = out.nonzero().flatten()
      if len(ids):
        failed_now.append(ids.cpu().numpy())
        bodies = [()] * len(ids)
        if name == "illegal_contact":   # (read as fail_probe.py reads them)
          sn = env.scene[kw["sensor_name"]]
          prim = list(dict.fromkeys(sl.primary_name for sl in sn._slots))
          d = sn.data
          if d.force_history is not None:
            hit = (torch.norm(d.force_history[ids], dim=-1) > kw.get("force_threshold", 10.0)).any(dim=-1)
          else:
            hit = d.found[ids] > 0
          bodies = [tuple(prim[j] for j in np.nonzero(h)[0]) for h in hit.cpu().numpy()]
        for i, b in zip(ids.tolist(), bodies):
          w = fail_why.setdefault(i, [set(), set()])
          w[0].add(name)
          w[1].update(b)
      return out
    term.func = f
  wrap("illegal_contact")
  wrap("fell_over")
  tl = cfg.curriculum.get("terrain_levels")
  print(f"terrain_levels curriculum: {getattr(tl, 'func', None).__name__ if tl else None} {getattr(tl, 'params', None)}")
  print(f"episode_length_s {cfg.episode_length_s}")

  agent = load_rl_cfg(a.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(a.task)(venv, asdict(agent), log_dir=None, device="cuda:0")
  runner.load(a.checkpoint, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")

  cols = column_names(env)
  terrain = env.scene.terrain
  dev, dt = terrain.terrain_levels.device, terrain.terrain_levels.dtype
  up_cols = torch.tensor([i for i, c in enumerate(cols) if c in UP], device=dev)
  n_rows = int(terrain.max_terrain_level)
  N = env.num_envs
  my_col = up_cols[torch.randint(len(up_cols), (N,), device=dev)]
  my_row = (torch.randperm(N, device=dev) % n_rows).to(dt)
  saved_rows = {k: v.clone() for k, v in (getattr(env, "u_rows", None) or {}).items()}

  def place():
    terrain.terrain_types[:] = my_col
    terrain.terrain_levels[:] = my_row
    terrain.env_origins[:] = terrain.terrain_origins[my_row, my_col]
    if saved_rows:
      for k, v in saved_rows.items():
        env.u_rows[k][:] = v

  cm = env.curriculum_manager
  compute = cm.compute

  def compute_and_place(*args, **kwargs):
    out = compute(*args, **kwargs)
    place()
    return out
  place()
  cm.compute = lambda *a_, **k_: None
  try:
    env.reset()
  finally:
    cm.compute = compute_and_place
  print(f"{a.task} {a.checkpoint}: {N} robots x {a.steps} steps, mean action, p {getattr(env, 'u_progress', 0.0):.3f}, "
        f"payload full {getattr(env, 'u_payload_full', None)}, "
        f"columns {dict(Counter(cols[int(c)] for c in my_col.tolist()))}")

  robot = env.scene["robot"]
  sites = [s for s in ("FL", "FR", "RL", "RR") if s in robot.site_names]
  if len(sites) != 4:
    print(f"foot sites not found: {robot.site_names}"); return 1
  site_ids = [robot.site_names.index(s) for s in sites]
  scan = env.scene["terrain_scan"]
  twist = env.command_manager.get_term("twist")
  limit = float(getattr(env, "u_payload_full", 0.0) or 0.0)

  ground0 = torch.full((N,), float("nan"), device=dev)
  max_phase = torch.zeros(N, dtype=torch.long, device=dev)
  ep_fwd_cmd_steps = torch.zeros(N, device=dev)
  fwd_run = torch.zeros(N, device=dev)   # consecutive steps with the forward command above 0.3 m/s
  SETTLE = int(round(1.0 / env.step_dt))   # "settled": that command held 1 s and the episode 1 s old

  def foot_ground(hit, valid, foot):
    d = torch.norm(hit[:, None, :, :2] - foot[:, :, None, :2], dim=-1)   # B, 4, R
    ok = valid[:, None, :] & (hit[:, None, :, 2] <= foot[:, :, None, 2] + 0.05)
    d = torch.where(ok, d, torch.inf)
    j = d.argmin(dim=-1)
    z = torch.gather(hit[..., 2][:, None, :].expand(-1, 4, -1), 2, j[..., None])[..., 0]
    return torch.where(torch.isfinite(d.min(dim=-1).values), z, torch.nan)

  episodes = []   # (row, col, distance, cmd_distance, failed, kg, max_phase, end_phase, secs, fwd_cmd_share)
  speed = {m: defaultdict(list) for m in ("all", "settled")}   # phase -> ratios (cmd vx > 0.3)
  slow = {m: Counter() for m in speed}; tot = {m: Counter() for m in speed}
  stop_where = {m: Counter() for m in speed}   # (phase, distance bin)
  bins = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 99.0)
  binname = lambda x: next(f"{lo:.1f}-{hi:.1f}" if hi < 99 else f">{lo:.1f}" for lo, hi in zip((0.0,) + bins, bins) if x < hi)   # noqa: E731
  rm = env.reward_manager
  terms = list(rm.active_terms)
  rw_rows = torch.tensor([int(x) for x in a.reward_rows.split(",") if x != ""], device=dev)
  rw_on = torch.isin(my_row.long(), rw_rows.long())
  pay = [float(x) for x in a.reward_payload.split(",")] if a.reward_payload else None
  rsum = {k: torch.zeros(len(terms), device=dev) for k in "ABCD"}
  rcnt = {k: 0 for k in "ABCD"}
  fails = []   # (row, kg, terminations, bodies, phase, forward speed, forward command) of each failed episode (5.)
  cls_edges = torch.tensor([0.25, 2.0 / 3.0], device=dev)   # light / mid / heavy, of the full payload
  row_edges = torch.tensor([3.5, 6.5], device=dev)          # rows 0-3 / 4-6 / 7 and up
  fl_sum, fl_cnt = torch.zeros(9, device=dev), torch.zeros(9, device=dev)   # speed / commanded on the flight
  # (6.) the trunk clearance cost and the "commanded up" mask, read whatever the task's switches
  from src.tasks.unified import constraints as ucons
  bc = bc_sensor = None
  if "illegal_contact" in cfg.terminations:
    bc, bc_sensor = ucons.body_clear(None, env), cfg.terminations["illegal_contact"].params["sensor_name"]
  up_sensor = cfg.rewards["stair_speed_deficit"].params["sensor_name"] if "stair_speed_deficit" in cfg.rewards else None
  BC_W = int(round(0.5 / env.step_dt))
  bc_hist = torch.zeros(N, BC_W, device=dev)   # the cost over the last 0.5 s, the newest last
  bc_n, bc_on, bc_sum, up_on = (torch.zeros(3, device=dev) for _ in range(4))   # by phase, settled forward command
  bc_fail = []   # (terminations, bodies, the cost at the last step before the fail, its max over 0.1 s, over 0.5 s)
  obs = venv.get_observations()
  for _ in range(a.steps):
    hit = scan.data.hit_pos_w
    valid = scan.data.distances >= 0.0
    base = robot.data.root_link_pos_w
    # the episode's spawn ground: hits within 0.3 m of the base at its first step
    # (taken at the episode's third step: the scan may still hold the old place right after the reset)
    fresh = env.episode_length_buf == 2
    if bool(fresh.any()):
      near = valid & (torch.norm(hit[..., :2] - base[:, None, :2], dim=-1) < 0.3)
      g = torch.where(near, hit[..., 2], torch.nan).nanmedian(dim=-1).values
      ground0 = torch.where(fresh, g, ground0)
    foot = robot.data.site_pos_w[:, site_ids]
    fg = foot_ground(hit, valid, foot)   # B, 4 (FL FR RL RR)
    over = (fg - ground0[:, None]) > STEP_UP
    rear_up = over[:, 2] & over[:, 3]
    phase = torch.where(rear_up, 2, torch.where(over.any(dim=1), 1, 0))
    max_phase = torch.maximum(max_phase, phase)
    cmd = env.command_manager.get_command("twist")
    v_b = robot.data.root_link_lin_vel_b
    v_w = robot.data.root_link_lin_vel_w
    rel = (base - terrain.env_origins)
    dist = torch.norm(rel[:, :2], dim=1)
    fwd = cmd[:, 0] > 0.3
    ep_fwd_cmd_steps += fwd.float()
    fwd_run = torch.where(fwd, fwd_run + 1, 0.0)
    settled = (fwd_run >= SETTLE) & (env.episode_length_buf >= SETTLE)
    ratio = v_b[:, 0] / cmd[:, 0].clamp(min=1e-3)
    planar = torch.norm(v_w[:, :2], dim=1)
    ph_np, fwd_np = phase.cpu().numpy(), fwd.cpu().numpy()
    r_np, sl_np, d_np = ratio.cpu().numpy(), (planar < 0.1).cpu().numpy(), dist.cpu().numpy()
    st_np = settled.cpu().numpy()
    slow_t = planar < 0.1
    acc = rw_on
    if pay is not None:   # (the payload is drawn at a reset: read each step)
      kg = env.payload_kg.reshape(-1)
      acc = acc & (kg >= pay[0] * limit) & (kg <= pay[1] * limit)
    cls = {"A": acc & fwd & settled & (phase == 0) & slow_t, "C": acc & fwd & settled & (phase == 0) & ~slow_t,
           "B": acc & fwd & settled & (phase == 2), "D": acc & fwd & settled & (phase == 2) & slow_t}
    for i in np.nonzero(fwd_np)[0]:
      for m in (("all", "settled") if st_np[i] else ("all",)):
        speed[m][ph_np[i]].append(r_np[i]); tot[m][ph_np[i]] += 1
        if sl_np[i]:
          slow[m][ph_np[i]] += 1; stop_where[m][ph_np[i], binname(d_np[i])] += 1
    pre = dict(dist=d_np, cmdd=twist.cmd_distance.cpu().numpy().copy(), mp=max_phase.cpu().numpy().copy(), ph=ph_np,
               secs=env.episode_length_buf.cpu().numpy() * env.step_dt, fw=ep_fwd_cmd_steps.cpu().numpy().copy(),
               kg=(env.payload_kg.cpu().numpy().reshape(-1).copy() if getattr(env, "payload_kg", None) is not None
                   else np.zeros(N)))
    if limit:   # (5.) the speed on the flight by payload class and rows
      on = fwd & settled & (phase == 2)
      idx = (torch.bucketize(env.payload_kg.reshape(-1).float() / limit, cls_edges, right=True) * 3
             + torch.bucketize(my_row.float(), row_edges))
      fl_sum.index_add_(0, idx[on], ratio[on].float())
      fl_cnt.index_add_(0, idx[on], torch.ones_like(ratio[on].float()))
    pre["vx"], pre["cx"] = v_b[:, 0].cpu().numpy().copy(), cmd[:, 0].cpu().numpy().copy()
    if bc is not None:
      val = bc(env, sensor_name=bc_sensor)
      bc_hist = torch.roll(bc_hist, -1, dims=1)
      bc_hist[:, -1] = val
      ph1 = torch.nn.functional.one_hot(phase, 3).float() * (fwd & settled)[:, None]
      bc_n += ph1.sum(dim=0)
      bc_on += (ph1 * (val > 0)[:, None]).sum(dim=0)
      bc_sum += (ph1 * val[:, None]).sum(dim=0)
      if up_sensor is not None:
        up_on += (ph1 * ucons.climbing_mask(env, up_sensor)[:, None]).sum(dim=0)
      pre["bc"] = bc_hist.cpu().numpy().copy()
    failed_now.clear()
    fail_why.clear()
    with torch.inference_mode():
      obs, _, dones, _ = venv.step(policy(obs))
    done = dones.cpu().numpy().astype(bool)
    alive = ~dones.to(torch.bool)
    for k, m in cls.items():
      m = m & alive
      rsum[k] += rm._step_reward[m].sum(dim=0)
      rcnt[k] += int(m.sum())
    bad = np.zeros(N, dtype=bool)
    for ids in failed_now:
      bad[ids] = True
    for i in np.nonzero(done)[0]:
      steps_ep = max(pre["secs"][i] / env.step_dt, 1.0)
      episodes.append((int(my_row[i]), cols[int(my_col[i])], float(pre["dist"][i]), float(pre["cmdd"][i]), bool(bad[i]),
                       float(pre["kg"][i]), int(pre["mp"][i]), int(pre["ph"][i]), float(pre["secs"][i]),
                       float(pre["fw"][i]) / steps_ep))
      if bad[i]:
        w = fail_why.get(int(i), [set(), set()])
        fails.append((int(my_row[i]), float(pre["kg"][i]), tuple(sorted(w[0])), tuple(sorted(w[1])), int(pre["ph"][i]),
                      float(pre["vx"][i]), float(pre["cx"][i])))
        if bc is not None:
          h = pre["bc"][i]
          bc_fail.append((tuple(sorted(w[0])), tuple(sorted(w[1])), float(h[-1]), float(h[-5:].max()), float(h.max())))
    dn = torch.as_tensor(done, device=dev)
    ground0 = torch.where(dn, torch.nan, ground0)
    max_phase = torch.where(dn, 0, max_phase)
    ep_fwd_cmd_steps = torch.where(dn, 0.0, ep_fwd_cmd_steps)
    fwd_run = torch.where(dn, 0.0, fwd_run)
    bc_hist[dn] = 0.0

  PH = ("before the first riser", "at the first riser (a foot on a step, a rear foot still on the bottom)",
        "on the flight (both rear feet on steps)")
  E = episodes
  print(f"p at the end {getattr(env, 'u_progress', 0.0):.3f}; episodes ended {len(E)}, failed {sum(e[4] for e in E)}")
  ok = [e for e in E if not e[4]]

  def summary(rows, label):
    if not rows:
      print(f"  {label}: no episodes"); return
    d = np.array([e[2] for e in rows]); c = np.array([e[3] for e in rows])
    prom = d > 3.0; dem = (d < 0.5 * c) & ~prom
    print(f"  {label}: {len(rows)} episodes; end distance p10/p50/p90 {q3(d)} m; past 3 m (promoted) {100 * prom.mean():.0f}%; "
          f"short walk (demoted, < 0.5 x commanded {q3(c)} m) {100 * dem.mean():.0f}%; neither {100 * (~prom & ~dem).mean():.0f}%; "
          f"time {q3([e[8] for e in rows])} s")
  print("1. episodes without a fall or illegal contact on the climbing columns")
  summary(ok, "all")
  for col in UP:
    summary([e for e in ok if e[1] == col], col)
  fw = [e for e in ok if e[9] >= 0.5]
  summary(fw, "forward command > 0.3 m/s for at least half the episode")
  for r in range(n_rows):
    rr = [e for e in ok if e[0] == r]
    fr = [e for e in E if e[0] == r]
    if fr:
      summary(rr, f"row {r} (fail {100 * sum(e[4] for e in fr) / len(fr):.0f}% of {len(fr)})")
  if limit:
    summary([e for e in ok if e[5] <= 0.25 * limit], "light payload (<= 1/4 of the full)")
    summary([e for e in ok if e[5] > 0.25 * limit], "loaded")
  for m, label in (("settled", "forward command held >= 1 s, episode >= 1 s old"), ("all", "every step with the forward command")):
    print(f"2. [{label}] forward command > 0.3 m/s: forward speed / commanded (p10/p50/p90, mean) and the time below 0.1 m/s")
    for k in (0, 1, 2):
      v = np.array(speed[m][k])
      if v.size:
        print(f"  {PH[k]}: {v.size} steps; ratio {q3(v)} mean {v.mean():.2f}; |v| < 0.1 m/s {100 * slow[m][k] / tot[m][k]:.1f}%")
    allt = sum(tot[m].values()); alls = sum(slow[m].values())
    print(f"  all phases: |v| < 0.1 m/s {100 * alls / max(allt, 1):.1f}% of {allt} steps")
    print(f"3. [{label}] where the stops are (|v| < 0.1): share of the stopped steps by phase / distance from the spawn")
    for k in (0, 1, 2):
      n = sum(v for (p, _), v in stop_where[m].items() if p == k)
      if n:
        print(f"  {PH[k]} {100 * n / max(alls, 1):.0f}%: " + ", ".join(
          f"{b} m {100 * stop_where[m][k, b] / max(alls, 1):.0f}%" for b in [binname(x - 0.01) for x in bins] if stop_where[m][k, b]))
  notp = [e for e in ok if e[2] <= 3.0]
  if notp:
    print(f"  not promoted ({len(notp)}): end phase " + ", ".join(
      f"{PH[k]} {100 * sum(e[7] == k for e in notp) / len(notp):.0f}%" for k in (0, 1, 2))
          + "; furthest phase reached " + ", ".join(
      f"{PH[k]} {100 * sum(e[6] == k for e in notp) / len(notp):.0f}%" for k in (0, 1, 2)))
    print("  their end distance: " + ", ".join(
      f"{b} m {100 * sum(binname(e[2]) == b for e in notp) / len(notp):.0f}%" for b in [binname(x - 0.01) for x in bins]))
  print(f"4. reward terms per second (weighted), rows {a.reward_rows}, settled forward command > 0.3 m/s: "
        f"A stopped before the first riser ({rcnt['A']} steps), C walking before it ({rcnt['C']}), "
        f"B both rear feet on the steps ({rcnt['B']})")
  mean = {k: (rsum[k] / max(rcnt[k], 1)).cpu().numpy() for k in "ABCD"}
  order = sorted(range(len(terms)), key=lambda j: -abs(mean["A"][j] - mean["B"][j]))
  watch = ("feet_edge", "stance_leg_tilt", "leg_ground_contact", "foot_terrain_clearance", "foot_stumble",
           "foot_step_touch", "action_rate_l2", "track_linear_velocity", "foothold_height_gain", "stair_speed_deficit")
  shown = order[:a.top] + [j for j in range(len(terms)) if terms[j] in watch and j not in order[:a.top]]
  far = sorted(range(len(terms)), key=lambda j: -abs(mean["D"][j] - mean["B"][j]))[:8]   # what a stop on the flight costs
  shown += [j for j in far if j not in shown]
  print(f"  {'term':32s} {'A stopped':>10s} {'C walking':>10s} {'B climbing':>10s} {'A-B':>9s} {'D on flight':>11s} {'D-B':>9s}")
  for j in shown:
    print(f"  {terms[j]:32s} {mean['A'][j]:10.4f} {mean['C'][j]:10.4f} {mean['B'][j]:10.4f} {mean['A'][j] - mean['B'][j]:9.4f}"
          f" {mean['D'][j]:11.4f} {mean['D'][j] - mean['B'][j]:9.4f}" + ("  *" if terms[j] in watch else ""))
  print(f"  {'total (all terms)':32s} {mean['A'].sum():10.4f} {mean['C'].sum():10.4f} {mean['B'].sum():10.4f} "
        f"{mean['A'].sum() - mean['B'].sum():9.4f} {mean['D'].sum():11.4f} {mean['D'].sum() - mean['B'].sum():9.4f}")
  # (2026-11-13, the loaded falls: is a stop on the flight worth more than a fall?)
  w_end = rm.get_term_cfg("is_terminated").weight if "is_terminated" in terms else float("nan")
  print(f"  D stopped on the flight (of B, |v| < 0.1; {rcnt['D']} steps): total {mean['D'].sum():.4f} per second"
        + "".join(f", {t} {mean['D'][terms.index(t)]:.3f}" for t in ("track_linear_velocity", "stair_speed_deficit")
                  if t in terms)
        + f"; an episode ended by a fall or an illegal contact: is_terminated's weight now {w_end:.1f}"
        + (f"; robots carrying {pay[0]:g}-{pay[1]:g} of the full payload ({limit:g} kg) only" if pay else ""))
  missing = [w for w in watch if w not in terms]
  if missing:
    print(f"  (not in this task's reward terms: {', '.join(missing)})")
  share_a = rcnt["A"] / max(rcnt["A"] + rcnt["B"] + rcnt["C"], 1)
  if rcnt["B"] < MIN_CLASS_STEPS:
    print(f"  REWARD ACCOUNT: no verdict, only {rcnt['B']} climbing steps on rows {a.reward_rows} "
          f"(stopped before the riser {100 * share_a:.0f}% of the classed steps)")
  elif rcnt["A"] < MIN_CLASS_STEPS:
    print(f"  REWARD ACCOUNT: the policy does not stop before the riser on rows {a.reward_rows} "
          f"({rcnt['A']} stopped steps, {100 * share_a:.1f}% of the classed ones); climbing {mean['B'].sum():+.3f} per second, "
          f"walking up to it {mean['C'].sum():+.3f}")
  else:
    gain = float(mean["B"].sum() - mean["A"].sum())
    print(f"  REWARD ACCOUNT: climbing minus stopped {gain:+.3f} per second on rows {a.reward_rows} "
          f"(stopped {100 * share_a:.0f}% of the classed steps) -> "
          + ("climbing pays" if gain > 0 else "STOPPING PAYS: the policy earns more parked before the first riser"))
  if limit:   # (2026-11-13, the loaded falls: which payload, which rows, what touches, how fast)
    print("5. the failed episodes by payload class and rows: fail% of the class's episodes there; of its fails: "
          "fell_over / illegal_contact, the bodies in contact (share of the fails), the phase at the fail (before the "
          "first riser / at it / on the flight), the forward speed of the commanded at the fail (p50); and the speed "
          "/ commanded on the flight (settled forward command, mean)")
    names = ("light (< 1/4)", "mid", "heavy (>= 2/3)")
    cut = (0.0, 0.25, 2.0 / 3.0, 9.9)
    rows_of = ((0, 3), (4, 6), (7, n_rows - 1))
    for ci, label in enumerate(names):
      for gi, (r0, r1) in enumerate(rows_of):
        inside = lambda row, kg: r0 <= row <= r1 and cut[ci] * limit <= kg < cut[ci + 1] * limit   # noqa: E731
        ep = [e for e in E if inside(e[0], e[5])]
        fl = [f for f in fails if inside(f[0], f[1])]
        if not ep:
          continue
        k = ci * 3 + gi
        line = (f"  {label:15s} rows {r0}-{r1}: fail {100 * len(fl) / len(ep):3.0f}% of {len(ep):4d}; on the flight "
                f"x{float(fl_sum[k] / fl_cnt[k].clamp(min=1)):.2f} ({int(fl_cnt[k])} steps)")
        if fl:
          n = len(fl)
          parts = Counter(b for f in fl for b in f[3])
          line += (f"; fell_over {100 * sum('fell_over' in f[2] for f in fl) / n:.0f}%, illegal_contact "
                   f"{100 * sum('illegal_contact' in f[2] for f in fl) / n:.0f}%: "
                   + ", ".join(f"{b} {100 * c / n:.0f}%" for b, c in parts.most_common(4))
                   + "; phase " + " / ".join(f"{100 * sum(f[4] == ph for f in fl) / n:.0f}%" for ph in (0, 1, 2))
                   + f"; speed {np.median([f[5] for f in fl]):.2f} of {np.median([f[6] for f in fl]):.2f} m/s")
        print(line)
  if bc is not None:   # (2026-11-13: would a cost on the trunk's clearance have seen the touch coming?)
    print("6. the trunk clearance cost (constraints.body_clear at its defaults, the bodies of the illegal-contact "
          "termination; read here, whatever the task's switches): the settled forward-command steps with the cost "
          "on and its mean, and the share of them commanded up (constraints.climbing_mask), by phase; then the "
          "cost before the fails -- on at the last step, within 0.1 s, within 0.5 s, its value at the last step")
    for k in (0, 1, 2):
      n = float(bc_n[k])
      if n:
        print(f"  {PH[k]}: cost on {100 * float(bc_on[k]) / n:.1f}% of {int(n)} steps, mean {float(bc_sum[k]) / n:.3f}"
              + (f"; commanded up {100 * float(up_on[k]) / n:.0f}%" if up_sensor is not None else ""))

    def before(rows, label):
      if not rows:
        return
      v = np.array([[r[2], r[3], r[4]] for r in rows])
      print(f"  {label} ({len(rows)}): on at the last step {100 * (v[:, 0] > 0).mean():.0f}%, within 0.1 s "
            f"{100 * (v[:, 1] > 0).mean():.0f}%, within 0.5 s {100 * (v[:, 2] > 0).mean():.0f}%; value at the last "
            f"step p10/p50/p90 {q3(v[:, 0])}, max over 0.5 s {q3(v[:, 2])}")
    touch = [r for r in bc_fail if "illegal_contact" in r[0]]
    before(touch, "fails with an illegal contact")
    for b, _ in Counter(b for r in touch for b in r[1]).most_common(5):
      before([r for r in touch if b in r[1]], f"  with {b}")
    before([r for r in bc_fail if "illegal_contact" not in r[0]], "fails without one (fell over)")
  env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
