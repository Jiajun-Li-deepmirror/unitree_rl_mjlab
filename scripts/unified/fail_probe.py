#!/usr/bin/env python3
"""fail_probe.py TASK CHECKPOINT [--num-envs N] [--steps S] (go2-depth 2026-11-12): where a unified teacher's training
episodes fail. Builds the training env of TASK (e.g. Unitree-Go2W-U-Teacher), restores the checkpoint as a resume
does (policy, progress, constraints, terrain levels), runs the policy for S control steps with the training's
curriculum and commands, and prints per sub-terrain: the share of robot-steps, episodes ended, falls (fell_over),
illegal contacts and which bodies made them, the fail rate and the mean terrain level; plus the fail rate by payload
(light / heavy) and by command (standing, straight, turning, sideways); then the constraints' measurements over the
run beside the checkpoint's (the training's own reading). --sample acts with the training's exploration noise (the
sampled action) instead of the policy's mean: the two runs show how much of a reading is the noise's. --spread-rows
starts the robots on difficulty rows spread evenly over all of them (both families) instead of the checkpoint's, and
the report's "fail% by row" (stairs up / stairs down / other ground, by the row each episode was walked on) then
shows up to which row the policy gets through; "where the fails end" gives, per family, how far from the spawn point,
how long into the episode, at which nose-down pitch and base height (over the spawn ground) the failed episodes
ended, and how fast they were told to go. --meter-split counts the shared stairs metrics (src/metrics.mj.StairsMeter,
the training's "lib:" constraints and the acceptance's eval_stairs --metrics) over every robot of this run, split by
the conditions the two readings differ in: the commanded speed, forward / backward, the payload class, pyramid /
walled flights, up / down and the row's step height -- "as the acceptance" is forward at <= 0.6 m/s, light, on the
pyramid stairs (2026-11-12: the Go2 teacher's training reading of kicks and slips was 2.4-3.2 times the acceptance's
on one checkpoint). Run with UNIFIED_TASKS=1 from the repo root."""

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
os.environ["U_EVAL_ENVS"] = "0"   # every robot acts alike here (--sample or not) and all of them are measured

import numpy as np  # noqa: E402
import torch  # noqa: E402


def column_names(env) -> list[str]:
  """Each curriculum column's sub-terrain name (terrain_utils.sub_terrain_column_mask's allocation)."""
  g = env.scene.terrain.cfg.terrain_generator
  names = list(g.sub_terrains.keys())
  prop = np.array([c.proportion for c in g.sub_terrains.values()], dtype=float)
  cum = np.cumsum(prop / prop.sum())
  return [names[int(np.min(np.where(i / g.num_cols + 0.001 < cum)[0]))] for i in range(g.num_cols)]


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("task")
  ap.add_argument("checkpoint")
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--steps", type=int, default=1500)
  ap.add_argument("--sample", action="store_true", help="act with the exploration noise, as the training does")
  ap.add_argument("--spread-rows", action="store_true", help="start on rows spread evenly over all of them")
  ap.add_argument("--meter-split", action="store_true", help="the stairs metrics split by command, payload, terrain")
  a = ap.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401

  cfg = load_env_cfg(a.task)
  cfg.scene.num_envs = a.num_envs
  rec: dict[str, list] = defaultdict(list)

  def wrap(name):
    term = cfg.terminations.get(name)
    if term is None:
      return
    orig = term.func

    def f(env, **kw):
      out = orig(env, **kw)
      ids = out.nonzero().flatten()
      if len(ids):
        bodies = [None] * len(ids)
        if name == "illegal_contact":
          s = env.scene[kw["sensor_name"]]
          prim = list(dict.fromkeys(sl.primary_name for sl in s._slots))
          d = s.data
          if d.force_history is not None:
            hit = (torch.norm(d.force_history[ids], dim=-1) > kw.get("force_threshold", 10.0)).any(dim=-1)
          else:
            hit = d.found[ids] > 0
          bodies = [tuple(prim[j] for j in np.nonzero(h)[0]) for h in hit.cpu().numpy()]
        rec[name].append((ids.cpu().numpy(), bodies))
      return out

    term.func = f

  wrap("illegal_contact")
  wrap("fell_over")
  agent = load_rl_cfg(a.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(a.task)(venv, asdict(agent), log_dir=None, device="cuda:0")
  runner.load(a.checkpoint, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  if a.sample:
    actor = runner.alg.actor
    policy = lambda o: actor(o, stochastic_output=True)   # noqa: E731
  cons = getattr(env, "u_constraints", None)
  trained = dict(cons.value) if cons is not None else {}   # the checkpoint's measurements (the training's reading)
  cols = column_names(env)
  from src.tasks.velocity.blind_env_cfg import MAP3_STAIR_TERRAIN_NAMES
  stairs_col = [c in MAP3_STAIR_TERRAIN_NAMES for c in cols]
  terrain = env.scene.terrain
  twist = env.command_manager.get_term("twist")
  if a.spread_rows:
    n_rows = int(terrain.max_terrain_level)
    dev, dt = terrain.terrain_levels.device, terrain.terrain_levels.dtype
    spread = {k: (torch.randperm(env.num_envs, device=dev) % n_rows).to(dt) for k in ("up", "down", "other")}
    if getattr(env, "u_rows", None) is not None:
      env.u_rows = spread
    on = torch.tensor(stairs_col, device=dev)[terrain.terrain_types]
    up = torch.tensor([s and "up" in c for s, c in zip(stairs_col, cols)], device=dev)[terrain.terrain_types]
    terrain.terrain_levels[:] = torch.where(up, spread["up"], torch.where(on, spread["down"], spread["other"]))
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    cm = env.curriculum_manager   # the robots onto their new origins; the curriculum held off (progress._restore_terrain)
    compute = cm.compute
    cm.compute = lambda *a_, **k_: None
    try:
      env.reset()
    finally:
      cm.compute = compute

  limit = float(getattr(env, "u_payload_full", 0.0) or 0.0)
  split = None
  if a.meter_split:
    from src.metrics.mj import StairsMeter
    from src.tasks.velocity.blind_env_cfg import WALLED_TERRAIN_NAMES

    n_rows = int(terrain.max_terrain_level)
    gen = terrain.cfg.terrain_generator
    lo_d, hi_d = gen.difficulty_range
    hs = [s_.step_height_range for k_, s_ in gen.sub_terrains.items()
          if k_ in MAP3_STAIR_TERRAIN_NAMES and getattr(s_, "step_height_range", None) is not None]
    # a row's step heights (cm): its tiles draw their difficulty in [r, r + 1) / n_rows of the range
    row_cm = [(100 * min(x + (y - x) * (lo_d + (hi_d - lo_d) * r / n_rows) for x, y in hs),
               100 * max(x + (y - x) * (lo_d + (hi_d - lo_d) * (r + 1) / n_rows) for x, y in hs)) if hs else (0.0, 0.0)
              for r in range(n_rows)]
    thirds = [range(n_rows)[i * n_rows // 3:(i + 1) * n_rows // 3] for i in range(3)]
    row_names = [f"rows {t[0]}-{t[-1]} ({row_cm[t[0]][0]:.0f}-{row_cm[t[-1]][1]:.0f} cm)" for t in thirds]
    col_up = np.array(["up" in c for c in cols])
    col_walled = np.array([c in WALLED_TERRAIN_NAMES for c in cols])

    def split_groups(env_, ids):
      c = env_.command_manager.get_command("twist")[ids].cpu().numpy()
      kg_ = getattr(env_, "payload_kg", None)
      kg_ = kg_[ids].cpu().numpy().reshape(-1) if kg_ is not None else np.zeros(len(ids))
      ty = terrain.terrain_types[ids].cpu().numpy()
      lv_ = terrain.terrain_levels[ids].cpu().numpy()
      v = np.abs(c[:, 0])
      fwd = c[:, 0] > 0
      light = kg_ <= 0.25 * limit if limit else np.ones(len(ids), dtype=bool)
      heavy = kg_ >= 2 / 3 * limit if limit else np.zeros(len(ids), dtype=bool)
      up_, walled = col_up[ty], col_walled[ty]
      g = {"all": np.ones(len(ids), dtype=bool),
           "speed <= 0.6": v <= 0.6, "speed 0.6-1.0": (v > 0.6) & (v <= 1.0), "speed > 1.0": v > 1.0,
           "forward": fwd, "backward": ~fwd,
           "payload light": light, "payload mid": ~light & ~heavy, "payload heavy": heavy,
           "pyramid": ~walled, "walled": walled, "up": up_, "down": ~up_}
      for name, t in zip(row_names, thirds):
        g[name] = (lv_ >= t[0]) & (lv_ <= t[-1])
      like = fwd & (v <= 0.6) & light & ~walled
      g["as the acceptance"] = like
      g["as the acceptance, up"] = like & up_
      g["as the acceptance, down"] = like & ~up_
      for name, t in zip(row_names, thirds):
        g["as the acceptance, " + name] = like & (lv_ >= t[0]) & (lv_ <= t[-1])
      g["forward fast (> 1.0), light, pyramid"] = fwd & (v > 1.0) & light & ~walled
      g["backward, light, pyramid"] = ~fwd & light & ~walled
      return g

    wheeled_ = "wheel_vel" in cfg.actions
    # straight commands only, as the training's reading (constraints.lib_meter); every robot of this run
    split = StairsMeter(env, torch.arange(env.num_envs, device="cuda:0"), wheeled=wheeled_, ema_steps=1e12,
                        min_steps=20.0, straight=True)
    split.groups = split_groups

  def cmd_class(c):
    vx, vy, wz = abs(c[0]), abs(c[1]), abs(c[2])
    if max(vx, vy, wz) < 0.1:
      return "stand"
    if vy > 0.15:
      return "sideways"
    if wz > 0.3:
      return "turning"
    return "straight"

  steps_on = Counter()
  ended = Counter()
  fails = Counter()
  ill = Counter()
  fell = Counter()
  bodies = Counter()
  levels = defaultdict(list)
  pay_end = Counter()
  pay_fail = Counter()
  cmd_end = Counter()
  cmd_fail = Counter()
  row_end = Counter()
  row_fail = Counter()
  where_fail = defaultdict(list)   # family -> (distance from the spawn point, s, nose-down pitch deg, base z, |cmd v|)
  robot = env.scene["robot"]

  def family(c):
    return "other" if c not in MAP3_STAIR_TERRAIN_NAMES else ("stairs up" if "up" in c else "stairs down")
  obs = venv.get_observations()
  for _ in range(a.steps):
    # the state the step's terminations are judged in: the terrain column, level, payload and command before the step
    types = terrain.terrain_types.cpu().numpy().copy()
    lv = terrain.terrain_levels.cpu().numpy().copy()
    kg = getattr(env, "payload_kg", None)
    kg = kg.cpu().numpy().reshape(-1).copy() if kg is not None else np.zeros(env.num_envs)
    cmd = env.command_manager.get_command("twist").cpu().numpy().copy()
    rel = (robot.data.root_link_pos_w - terrain.env_origins).cpu().numpy()
    q = robot.data.root_link_quat_w.cpu().numpy()   # w, x, y, z
    pitch = np.degrees(np.arcsin(np.clip(2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1, 1)))
    secs = env.episode_length_buf.cpu().numpy() * env.step_dt
    for t, n in zip(*np.unique(types, return_counts=True)):
      steps_on[cols[t]] += int(n)
    for k in rec:
      rec[k].clear()
    with torch.inference_mode():
      obs, _, dones, _ = venv.step(policy(obs))
    if split is not None:
      split.step(env)
    done = dones.cpu().numpy().astype(bool)
    failed = np.zeros(env.num_envs, dtype=bool)
    for ids, bs in rec.get("illegal_contact", []):
      failed[ids] = True
      for i, b in zip(ids, bs):
        ill[cols[types[i]]] += 1
        bodies[(cols[types[i]], b)] += 1
    for ids, _ in rec.get("fell_over", []):
      failed[ids] = True
      for i in ids:
        fell[cols[types[i]]] += 1
    for i in np.nonzero(done)[0]:
      c = cols[types[i]]
      ended[c] += 1
      levels[c].append(lv[i])
      heavy = "heavy" if limit and kg[i] >= 2 / 3 * limit else ("light" if limit and kg[i] <= 0.25 * limit else "mid")
      where = "stairs" if stairs_col[types[i]] else "other"
      pay_end[heavy, where] += 1
      cmd_end[cmd_class(cmd[i])] += 1
      row_end[family(c), int(lv[i])] += 1
      if failed[i]:
        row_fail[family(c), int(lv[i])] += 1
        where_fail[family(c)].append((float(np.hypot(rel[i, 0], rel[i, 1])), float(secs[i]), float(pitch[i]),
                                      float(rel[i, 2]), float(np.hypot(cmd[i, 0], cmd[i, 1]))))
        fails[c] += 1
        pay_fail[heavy, where] += 1
        cmd_fail[cmd_class(cmd[i])] += 1
  tot = sum(steps_on.values())
  print(f"{a.task} {a.checkpoint}: {a.num_envs} robots x {a.steps} steps, {'sampled' if a.sample else 'mean'} actions, progress {getattr(env, 'u_progress', 0.0):.3f}, "
        f"stairs share {float(twist.is_stair_env.float().mean()):.2f}")
  print(f"{'sub-terrain':22s} {'steps%':>7s} {'ended':>6s} {'fails':>6s} {'fail%':>6s} {'fell':>5s} {'illegal':>7s} {'level':>6s}")
  for c in sorted(steps_on, key=lambda c: -fails[c]):
    e = ended[c]
    print(f"{c:22s} {100 * steps_on[c] / tot:7.1f} {e:6d} {fails[c]:6d} {100 * fails[c] / max(e, 1):6.1f} {fell[c]:5d} "
          f"{ill[c]:7d} {np.mean(levels[c]) if levels[c] else float('nan'):6.2f}")
  e, f = sum(ended.values()), sum(fails.values())
  print(f"{'all':22s} {100.0:7.1f} {e:6d} {f:6d} {100 * f / max(e, 1):6.1f} {sum(fell.values()):5d} {sum(ill.values()):7d}")
  rows = getattr(env, "u_rows", None)
  if rows is not None:
    print("difficulty rows at the end: " + ", ".join(f"{k} {float(v.float().mean()):.2f}" for k, v in rows.items())
          + f"; stairs quality {getattr(env, 'u_quality', 0.0):.3f}")
  print("illegal-contact bodies (sub-terrain, bodies): " + "; ".join(f"{c} {'+'.join(b) or '?'} {n}"
                                                                     for (c, b), n in bodies.most_common(10)))
  for w in ("stairs", "other"):
    print(f"fail% by payload on {w}: " + ", ".join(f"{k} {100 * pay_fail[k, w] / max(pay_end[k, w], 1):.1f} "
                                                   f"({pay_end[k, w]})" for k in ("light", "mid", "heavy")))
  print("fail% by command: " + ", ".join(f"{k} {100 * cmd_fail[k] / max(cmd_end[k], 1):.1f} ({cmd_end[k]})"
                                         for k in ("stand", "straight", "turning", "sideways")))
  for fam in ("stairs up", "stairs down", "other"):
    print(f"fail% by row on {fam}: " + ", ".join(f"{r} {100 * row_fail[fam, r] / row_end[fam, r]:.0f} ({row_end[fam, r]})"
                                                for r in range(int(terrain.max_terrain_level)) if row_end[fam, r]))
  for fam, w in where_fail.items():
    w = np.array(w)
    q3 = lambda j: "/".join(f"{v:.2f}" for v in np.percentile(w[:, j], (10, 50, 90)))   # noqa: E731
    print(f"where the fails end on {fam} (p10/p50/p90 of {len(w)}): from the spawn {q3(0)} m, after {q3(1)} s, nose-down "
          f"pitch {q3(2)} deg, base z {q3(3)} m, commanded {q3(4)} m/s")
  if split is not None:
    names = ("slide_back", "idle_start", "idle_end") if split.wheeled else ("kicks", "slips", "mid", "edge")
    unit = "step-ups" if split.wheeled else "steps"
    print(f"stairs metrics by condition (straight commands only, as the training counts; {unit} = the sample):")
    print(f"  {'condition':44s} {unit:>9s} " + " ".join(f"{n:>10s}" for n in names))
    for g, sums in split.group_sums.items():
      n = sums.get("ups", 0.0) if split.wheeled else (sums.get("ups", 0.0) + sums.get("downs", 0.0)) / split.feet
      vals = [split.rate(k, g) for k in names]
      print(f"  {g:44s} {n:9.0f} " + " ".join(f"{'-':>10s}" if v is None else f"{v:10.3f}" for v in vals))
  if cons is not None and cons.values:
    cur = env.curriculum_manager.get_term_cfg("u_constraints").params["items"]
    p = getattr(env, "u_progress", 0.0)
    # per scan mode (U_SCAN=on / off: one of them only); "training": the checkpoint's reading that drove the
    # multiplier (the mode further from the target); "target": the teacher's own, "bar": the students'
    print(f"constraints ({'sampled' if a.sample else 'mean'} actions here; the checkpoint's reading is the training's):")
    print(f"  {'name':16s} {'seen':>8s} {'blind':>8s} {'training':>8s} {'target':>8s} {'bar':>8s} {'sense':>5s} {'scale':>6s}")
    f = lambda v: f"{'-':>8s}" if v is None else f"{v:8.4f}"   # noqa: E731
    for c in cur:
      n = c["name"]
      seen, blind = cons.values.get(f"{n}@seen", cons.values.get(n)), cons.values.get(f"{n}@blind")
      if seen is None and blind is None:
        continue
      print(f"  {n:16s} {f(seen)} {f(blind)} {f(trained.get(n))} {f(cons.target(c, p))} {f(c['target'])} "
            f"{c['sense']:>5s} {cons.scale.get(n, 1.0):6.2f}")
  env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
