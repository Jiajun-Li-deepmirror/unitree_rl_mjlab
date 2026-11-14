#!/usr/bin/env python3
"""stop_feet_probe.py TASK CHECKPOINT [--envs-per-tile 18] [--vx 0.5] [--bands 2,3] (go2-depth 2026-11-14): what a
stairs gate's "four feet at a stop" share is made of. The gate (scripts/go2_blind/eval_stairs.py's stop test) reads
one number per band and direction: over the 2-4 s after the zero command, the share of steps with all four feet in
contact, averaged over the robots. A robot still stepping loses a part of it, a robot holding one foot in the air all
of it -- the number does not tell the two apart, nor which foot. The Go2 legged candidate (2026-11-14, G13 / G14):
12-15 cm down 0.78 (bar 0.82), 15-18 cm down 0.63 (bar 0.66), unchanged by 600 more iterations and by three times
the weight on the missing-feet cost (U_ITEM_FLOOR=stop_feet_down:3).

The run is the gate's stop test (the Stairs check task of TASK's robot, play build, the stair tiles 4 bands x up /
down x narrow / wide, --envs-per-tile robots each; walk outwards at --vx, zero command 2.2 m from the tile centre,
hold 4 s; U_SCAN / U_PAYLOAD from the environment as the gate sets them). Per robot over the 2-4 s window, per foot:
the share of steps off the ground and the number of lift-offs; then one class per robot:

  four down     all four in contact on >= 95 % of the steps
  one foot off  one foot off on >= 50 % of the steps, each of the others on <= 10 %   (named by the foot)
  stepping      two or more feet off on > 10 % of the steps each
  short lifts   the rest (one foot, less than half the window)

and per band x direction: each class's share of the robots and of the missing share (1 - the gate's number), the
planar speed (p50), the base's displacement over the window, the share of steps with a leg (not a foot) on the ground,
and for the "one foot off" classes the foot's height over the ground under it (terrain_scan's hit nearest the foot,
p50 -- about a step's height: the foot hangs past a nosing; about 0: it rests unloaded) and the base's pitch.
Run from the repo root (or from logs/probe/ of a box's tree) with UNIFIED_TASKS=1."""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("UNIFIED_TASKS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

NUM_ROWS = 4
STEP_RANGE = (0.05, 0.18)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("task")
  ap.add_argument("checkpoint")
  ap.add_argument("--vx", type=float, default=0.5)
  ap.add_argument("--envs-per-tile", type=int, default=18)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--bands", default="0,1,2,3", help="the rows printed in detail (0 = 5-8 cm ... 3 = 15-18 cm)")
  a = ap.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.velocity.blind_env_cfg import STAIR_TERRAIN_NAMES

  # the gate's build (eval_stairs.py), line for line
  cfg = load_env_cfg(a.task, play=True)
  cfg.seed = a.seed
  gen = cfg.scene.terrain.terrain_generator
  gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k in STAIR_TERRAIN_NAMES}
  for v in gen.sub_terrains.values():
    v.proportion = 1.0
  gen.num_rows = NUM_ROWS
  gen.num_cols = len(gen.sub_terrains) * 2
  cfg.events.pop("randomize_terrain")
  N = gen.num_rows * gen.num_cols * a.envs_per_tile
  cfg.scene.num_envs = N
  agent = load_rl_cfg(a.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(a.task)(venv, asdict(agent), device="cuda:0")
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")

  dev, dt = env.device, env.step_dt
  terrain = env.scene.terrain
  robot = env.scene["robot"]
  feet_sensor = env.scene["feet_ground_contact"]
  leg_sensor = env.scene["leg_ground_touch"]
  cmd_term = env.command_manager.get_term("twist")
  feet = list(dict.fromkeys(sl.primary_name for sl in feet_sensor._slots))
  found0 = (feet_sensor.data.found > 0).flatten(1)
  if found0.shape[1] != 4 or len(feet) != 4:
    print(f"feet_ground_contact: {found0.shape[1]} columns, slots {feet} -- four feet expected"); return 1
  short = [f[:2] for f in feet]
  # each foot's site (for its height over the ground under it): the site named by the slot's first two letters
  site_ids = [robot.site_names.index(s) if s in robot.site_names else -1 for s in short]
  try:
    scan = env.scene["terrain_scan"]
  except KeyError:
    scan = None
  clear_ok = scan is not None and min(site_ids) >= 0
  print(f"{a.task} {a.checkpoint}: {N} robots, U_SCAN {os.environ.get('U_SCAN', '-')}, U_PAYLOAD "
        f"{os.environ.get('U_PAYLOAD', '-')}, feet {feet}, foot clearance {'read' if clear_ok else 'not read'}")

  names = list(gen.sub_terrains.keys())
  cum = np.cumsum(np.full(len(names), 1.0 / len(names)))
  col_name = [names[np.min(np.where(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
  tile = torch.arange(N, device=dev) // a.envs_per_tile
  levels = tile % gen.num_rows
  types = tile // gen.num_rows
  env_level = levels.cpu().numpy()
  env_dir = np.array(["down" if "down" in col_name[t] else "up" for t in types.cpu().numpy()])

  def band(level):
    lo, hi = STEP_RANGE
    return f"{100 * (lo + (hi - lo) * level / NUM_ROWS):.0f}-{100 * (lo + (hi - lo) * (level + 1) / NUM_ROWS):.0f}cm"

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

  def foot_clear():
    hit = scan.data.hit_pos_w
    valid = scan.data.distances >= 0.0
    foot = robot.data.site_pos_w[:, site_ids]
    d = torch.norm(hit[:, None, :, :2] - foot[:, :, None, :2], dim=-1)   # N, 4, R
    ok = valid[:, None, :] & (hit[:, None, :, 2] <= foot[:, :, None, 2] + 0.05)
    d = torch.where(ok, d, torch.inf)
    j = d.argmin(dim=-1)
    z = torch.gather(hit[..., 2][:, None, :].expand(-1, 4, -1), 2, j[..., None])[..., 0]
    return torch.where(torch.isfinite(d.min(dim=-1).values), foot[..., 2] - z, torch.nan)

  terrain.terrain_levels[:] = levels
  terrain.terrain_types[:] = types
  terrain.env_origins[:] = terrain.terrain_origins[levels, types]
  env.reset()
  obs = venv.get_observations()
  set_cmd((a.vx, 0.0, 0.0))
  stopped_at = torch.full((N,), -1, dtype=torch.long, device=dev)
  alive = torch.ones(N, dtype=torch.bool, device=dev)
  hold = int(4.0 / dt)
  H = {k: [] for k in ("pos", "con", "leg", "spd", "clr", "pitch")}
  for t in range(int(10.0 / dt) + hold):
    newly = (stopped_at < 0) & (cheb_dist() >= 2.2) & alive
    if newly.any():
      set_cmd((0.0, 0.0, 0.0), newly)
      stopped_at[newly] = t
    with torch.no_grad():
      obs, _, _, _ = venv.step(policy(obs))
    term = env.termination_manager.terminated
    in_hold = (stopped_at >= 0) & (t - stopped_at < hold)
    alive &= ~(term & ((stopped_at < 0) | in_hold) & alive)
    H["pos"].append(robot.data.root_link_pos_w[:, :2].clone())
    H["con"].append((feet_sensor.data.found > 0).flatten(1).clone())
    H["leg"].append((leg_sensor.data.found > 0).flatten(1).any(1).clone())
    H["spd"].append(torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1).clone())
    H["pitch"].append(torch.asin(robot.data.projected_gravity_b[:, 0].clamp(-1, 1)).clone())
    if clear_ok:
      H["clr"].append(foot_clear())
    if ((stopped_at >= 0) & (t - stopped_at >= hold) | ~alive).all():
      break
  T = len(H["pos"])
  pos = torch.stack(H["pos"]).cpu().numpy()
  con = torch.stack(H["con"]).cpu().numpy()            # T, N, 4
  leg = torch.stack(H["leg"]).cpu().numpy()
  spd = torch.stack(H["spd"]).cpu().numpy()
  pitch = np.degrees(torch.stack(H["pitch"]).cpu().numpy())
  clr = torch.stack(H["clr"]).cpu().numpy() if clear_ok else None
  s_np, alive_np = stopped_at.cpu().numpy(), alive.cpu().numpy()

  rows = []   # per robot: level, dir, all4, class, off (4), lifts (4), speed, drift, leg share, clearance of the off foot, pitch
  for b in range(N):
    s = s_np[b]
    if not (s >= 0 and alive_np[b] and s + hold <= T):
      continue
    w0, w1 = s + hold // 2, s + hold
    c = con[w0:w1, b]
    all4 = float(c.all(1).mean())
    off = 1.0 - c.mean(0)
    lifts = (c[:-1] & ~c[1:]).sum(0)
    k = int(off.argmax())
    others = np.delete(off, k)
    if all4 >= 0.95:
      cls = "four down"
    elif off[k] >= 0.5 and others.max() <= 0.1:
      cls = f"{short[k]} off"
    elif (off > 0.1).sum() >= 2:
      cls = "stepping"
    else:
      cls = "short lifts"
    cl = float(np.nanmedian(clr[w0:w1, b, k][~c[:, k]])) if clr is not None and (~c[:, k]).any() else float("nan")
    rows.append((env_level[b], env_dir[b], all4, cls, off, lifts, float(np.median(spd[w0:w1, b])),
                 float(np.linalg.norm(pos[w1 - 1, b] - pos[w0, b])), float(leg[w0:w1, b].mean()), cl,
                 float(np.median(pitch[w0:w1, b]))))

  def med(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")

  print(f"\nband  dir | robots  four feet (the gate's) | off the ground, share of the window: {' '.join(short)} | lift-offs / robot")
  for lvl in range(NUM_ROWS):
    for d in ("up", "down"):
      r = [x for x in rows if x[0] == lvl and x[1] == d]
      if not r:
        continue
      off = np.mean([x[4] for x in r], axis=0)
      print(f"{band(lvl):>8} {d:>4} | {len(r):6d}  {np.mean([x[2] for x in r]):9.2f}               | "
            f"{' '.join(f'{o:.2f}' for o in off)} | {np.mean([x[5].sum() for x in r]):.1f}")
  want = {int(x) for x in a.bands.split(",") if x != ""}
  order = ["four down"] + [f"{s} off" for s in short] + ["stepping", "short lifts"]
  for lvl in sorted(want):
    for d in ("up", "down"):
      r = [x for x in rows if x[0] == lvl and x[1] == d]
      if not r:
        continue
      miss = sum(1.0 - x[2] for x in r)
      print(f"\n{band(lvl)} {d}: {len(r)} robots, four feet {np.mean([x[2] for x in r]):.2f}")
      print("  class        | robots  of the missing share | four feet | speed p50 m/s | moved in 2 s, p50 m | leg on the ground | off foot over its ground p50 m | pitch p50 deg")
      by = defaultdict(list)
      for x in r:
        by[x[3]].append(x)
      for cls in order:
        g = by.get(cls)
        if not g:
          continue
        print(f"  {cls:12s} | {len(g) / len(r):6.0%}  {sum(1.0 - x[2] for x in g) / max(miss, 1e-9):20.0%} | "
              f"{np.mean([x[2] for x in g]):9.2f} | {med([x[6] for x in g]):13.3f} | {med([x[7] for x in g]):19.3f} | "
              f"{np.mean([x[8] for x in g]):17.2f} | {med([x[9] for x in g]):30.3f} | {med([x[10] for x in g]):6.1f}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
