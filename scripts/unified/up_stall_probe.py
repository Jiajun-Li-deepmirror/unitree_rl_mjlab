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
  a = ap.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401

  cfg = load_env_cfg(a.task)
  cfg.scene.num_envs = a.num_envs
  failed_now: list[np.ndarray] = []

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
  rsum = {k: torch.zeros(len(terms), device=dev) for k in "ABC"}
  rcnt = {k: 0 for k in "ABC"}
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
    cls = {"A": rw_on & fwd & settled & (phase == 0) & slow_t, "C": rw_on & fwd & settled & (phase == 0) & ~slow_t,
           "B": rw_on & fwd & settled & (phase == 2)}
    for i in np.nonzero(fwd_np)[0]:
      for m in (("all", "settled") if st_np[i] else ("all",)):
        speed[m][ph_np[i]].append(r_np[i]); tot[m][ph_np[i]] += 1
        if sl_np[i]:
          slow[m][ph_np[i]] += 1; stop_where[m][ph_np[i], binname(d_np[i])] += 1
    pre = dict(dist=d_np, cmdd=twist.cmd_distance.cpu().numpy().copy(), mp=max_phase.cpu().numpy().copy(), ph=ph_np,
               secs=env.episode_length_buf.cpu().numpy() * env.step_dt, fw=ep_fwd_cmd_steps.cpu().numpy().copy(),
               kg=(env.payload_kg.cpu().numpy().reshape(-1).copy() if getattr(env, "payload_kg", None) is not None
                   else np.zeros(N)))
    failed_now.clear()
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
    dn = torch.as_tensor(done, device=dev)
    ground0 = torch.where(dn, torch.nan, ground0)
    max_phase = torch.where(dn, 0, max_phase)
    ep_fwd_cmd_steps = torch.where(dn, 0.0, ep_fwd_cmd_steps)
    fwd_run = torch.where(dn, 0.0, fwd_run)

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
  mean = {k: (rsum[k] / max(rcnt[k], 1)).cpu().numpy() for k in "ABC"}
  order = sorted(range(len(terms)), key=lambda j: -abs(mean["A"][j] - mean["B"][j]))
  watch = ("feet_edge", "stance_leg_tilt", "leg_ground_contact", "foot_terrain_clearance", "foot_stumble",
           "foot_step_touch", "action_rate_l2", "track_linear_velocity", "foothold_height_gain", "stair_speed_deficit")
  shown = order[:a.top] + [j for j in range(len(terms)) if terms[j] in watch and j not in order[:a.top]]
  print(f"  {'term':32s} {'A stopped':>10s} {'C walking':>10s} {'B climbing':>10s} {'A-B':>9s}")
  for j in shown:
    print(f"  {terms[j]:32s} {mean['A'][j]:10.4f} {mean['C'][j]:10.4f} {mean['B'][j]:10.4f} {mean['A'][j] - mean['B'][j]:9.4f}"
          + ("  *" if terms[j] in watch else ""))
  print(f"  {'total (all terms)':32s} {mean['A'].sum():10.4f} {mean['C'].sum():10.4f} {mean['B'].sum():10.4f} "
        f"{mean['A'].sum() - mean['B'].sum():9.4f}")
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
  env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
