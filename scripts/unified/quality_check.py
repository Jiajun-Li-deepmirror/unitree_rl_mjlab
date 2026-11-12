#!/usr/bin/env python3
"""quality_check.py ROBOT [CHECKPOINT] [--num-envs N] -- the stairs quality stage against the recipe, term by term
(go2-depth 2026-11-12; src/tasks/unified/curriculum.py's q).

Builds the unified teacher's training environment and compares the live reward terms with the recipe's own
environments (robots.base_env_cfg with the M and with the P keys):
  q = 0     every weight and parameter the M keys set must be the M value; a term only the P keys have must sit at
            Q_EPS x its weight; the stairs-quality constraints must read "off"
  q = 0.25  half-way through the reward blend: printed
  q = 1     every weight and parameter must be the P value (x the constraint's multiplier where one scales it)
Terms the recipe ramps by iteration and the flat constraints' terms are the same under both key sets and are compared
too (their ramp value at the first steps). With CHECKPOINT: the same after the runner restores it (its progress,
rows and multipliers), plus the restored quality and rows. Exit 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import torch


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("robot")
  ap.add_argument("checkpoint", nargs="?")
  ap.add_argument("--num-envs", type=int, default=64)
  a = ap.parse_args()
  os.environ["UNIFIED_TASKS"] = "1"
  sys.path.insert(0, os.getcwd())
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.unified.robots import ROBOTS, base_env_cfg
  from src.tasks.unified.teacher_env import Q_EPS, _m_default, _same

  robot = ROBOTS[a.robot]
  task = f"Unitree-{a.robot}-U-Teacher"
  ref = {k: base_env_cfg(robot, keys=k).rewards for k in ("M", "P")}
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = a.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  if a.checkpoint:
    agent = load_rl_cfg(task)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(task)(venv, asdict(agent), log_dir=None, device="cuda:0")
    runner.load(a.checkpoint, map_location="cuda:0")
    rows = getattr(env, "u_rows", None)
    print(f"restored: progress {env.u_progress:.3f}, quality {env.u_quality:.3f}, rows "
          + ", ".join(f"{k} {float(v.float().mean()):.2f}" for k, v in (rows or {}).items()))
  else:
    env.reset()
  cons = env.u_constraints
  items = cfg.curriculum["u_constraints"].params["items"]
  scaled = {t: c for c in items for t in c["terms"]}
  ramped = {r["name"] for r in cfg.curriculum["ramps"].params["ramps"] if r["kind"].startswith("reward")}
  rm = env.reward_manager
  action = torch.zeros(a.num_envs, env.action_manager.total_action_dim, device=env.device)
  bad = 0

  def run(q: float, steps: int = 3) -> None:
    os.environ["U_QUALITY"] = str(q)
    all_ids = torch.arange(a.num_envs, device=env.device)
    for _ in range(steps):
      env.step(action)
      env.curriculum_manager.compute(env_ids=all_ids[:1])   # the terms run whether or not a robot reset this step

  def value(key: str, t: str, k: str):
    r = ref[key][t]
    return r.params[k] if k in r.params else _m_default(ref["P"][t].func, k)

  def check(q: float, key: str) -> None:
    nonlocal bad
    run(q)
    p = env.u_progress
    n = 0
    for t in rm.active_terms:
      if t.startswith("u_") or t not in ref["P"]:
        continue
      live = rm.get_term_cfg(t)
      r = ref[key].get(t)
      # the weight
      if r is None:
        want = Q_EPS * ref["P"][t].weight
      else:
        want = r.weight
      c = scaled.get(t)
      if c is not None:
        on = cons.on(c, p, q)
        mult = max(cons.scale[x["name"]] for x in items if t in x["terms"]) if on else 1.0
        if on or c["by"] == "p":
          want = ref["P"][t].weight * (mult if on else c.get("start_scale", 1.0))
      if t in ramped:
        n += 1   # by iteration, the same under both key sets: not judged here
      elif abs(live.weight - want) > 1e-9 * max(1.0, abs(want)):
        bad += 1
        print(f"  MISMATCH q {q}: {t}.weight {live.weight:g}, the {key} value {want:g}")
      else:
        n += 1
      if r is None:
        continue
      for k in ref["P"][t].params:
        w = value(key, t, k)
        got = live.params[k]
        if hasattr(w, "name") and hasattr(got, "name"):   # scene entities: the live one is resolved
          same = (w.name, getattr(w, "site_names", None), getattr(w, "body_names", None)) == \
                 (got.name, getattr(got, "site_names", None), getattr(got, "body_names", None))
        else:
          same = _same(w, got) or (isinstance(w, float) and isinstance(got, float) and abs(w - got) < 1e-9)
        if any(rr["kind"] == "reward_param" and rr["name"] == t and rr["key"] == k
               for rr in cfg.curriculum["ramps"].params["ramps"]):
          continue
        if not same:
          bad += 1
          print(f"  MISMATCH q {q}: {t}.{k} {str(got)[:60]}, the {key} value {str(w)[:60]}")
        else:
          n += 1
    off = [c["name"] for c in items if c["by"] == "q" and not cons.on(c, p, q)]
    print(f"q {q}: {n} weights / parameters match the {key} keys' environment; stairs constraints off: {len(off)} "
          f"of {sum(c['by'] == 'q' for c in items)}; progress {p:.3f}")

  check(0.0, "M")
  run(0.25)
  qt = cfg.curriculum.get("u_quality")
  if qt is not None:
    print("q 0.25: " + ", ".join(f"{t} {rm.get_term_cfg(t).weight:g}" for t in qt.params["weights"]) + "; "
          + ", ".join(f"{t}.{k} {str(rm.get_term_cfg(t).params[k])[:24]}" for t, ks in qt.params["params"].items() for k in ks))
  print("q 0.25: " + ", ".join(f"{t} {rm.get_term_cfg(t).weight:g}" for t, c in scaled.items() if c["by"] == "q"))
  check(1.0, "P")
  check(0.0, "M")   # and back (U_QUALITY only: in training q never goes back)
  print("FAIL" if bad else "OK")
  return 1 if bad else 0


if __name__ == "__main__":
  sys.exit(main())
