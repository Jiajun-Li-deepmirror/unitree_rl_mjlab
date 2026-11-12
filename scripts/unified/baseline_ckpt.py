#!/usr/bin/env python3
"""baseline_ckpt.py R SRC_CHECKPOINT OUT_DIR [K] (unitree-rl-mjlab-62 2026-11-12, the review's item 12; doc/redesign.md
step 5 "compare with the two-teacher baseline"): another blind policy -- the deployed go2_blind_v4
(logs/rsl_rl/go2_blind/pkg_go2_fe_polish_10000/model_10000.pt on the boxes) -- as a unified blind student's stage-2
checkpoint, so jobs/unified/accept.sh reads it under exactly the unified acceptance's tests (the check tasks
Unitree-<R>-U-Student-Blind-Flat / -Stairs: the aligned robot model, the walking payload, the shared metrics).

The blind deploy contract is shared (src/tasks/unified/students.py: the student's "actor" group is the deployed blind
policies' input), so the source's actor_state_dict (normalizer, MLP, action std) loads into the student's actor as
it is; only the student's priv_head (never used to act) stays at its initial weights. Anything else that does not
fit (a different input size, layer sizes) stops it. The critic and teacher are the fresh runner's (the checks do not
use them). Writes OUT_DIR/model_K.pt (K default: the source's iteration) and OUT_DIR/source.txt.
Needs a GPU (it builds the student's environment at 16 robots); run from the repo root.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path


def main(argv: list[str]) -> int:
  if len(argv) < 3 or argv[0] in ("-h", "--help"):
    print(__doc__)
    return 0 if argv and argv[0] in ("-h", "--help") else 2
  robot, src, out = argv[0], argv[1], Path(argv[2])
  os.environ["UNIFIED_TASKS"] = "1"
  sys.path.insert(0, os.getcwd())
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401

  task = f"Unitree-{robot}-U-Student-Blind-RL"
  loaded = torch.load(src, map_location="cpu", weights_only=False)
  sd = loaded.get("actor_state_dict")
  if sd is None:
    print(f"baseline_ckpt: {src} has no actor_state_dict (not a PPO checkpoint)", file=sys.stderr)
    return 2
  k = int(argv[3]) if len(argv) > 3 else int(loaded.get("iter", 0))
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  agent = load_rl_cfg(task)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(task)(venv, asdict(agent), log_dir=None, device="cuda:0")
  actor = runner.alg.actor
  try:
    missing, unexpected = actor.load_state_dict({n: v.to("cuda:0") for n, v in sd.items()}, strict=False)
  except RuntimeError as e:   # a size mismatch: not the same contract
    print(f"baseline_ckpt: {src} does not fit {task}'s actor: {e}", file=sys.stderr)
    return 2
  bad = list(unexpected) + [n for n in missing if not n.startswith("priv_head.")]
  if bad:
    print(f"baseline_ckpt: {src} does not fit {task}'s actor: {bad}", file=sys.stderr)
    return 2
  names = env.observation_manager.active_terms["actor"]
  out.mkdir(parents=True, exist_ok=True)
  runner.current_learning_iteration = k
  runner.save(str(out / f"model_{k}.pt"), infos={"baseline_source": src})
  (out / "source.txt").write_text(f"{src}\nactor input terms: {', '.join(names)}\n")
  print(f"wrote {out / f'model_{k}.pt'}: {src}'s actor in {task} (input {actor.obs_dim}: {', '.join(names)})")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
