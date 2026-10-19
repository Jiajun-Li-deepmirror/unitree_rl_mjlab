"""Pre-deploy check (go2-blind): does the policy emit huge actions right after a reset?

go2-blind's Go2 run diverged after ~19100 iterations: a few % of robots reset
with a moving command emitted max|a| of 130-400 on the first step and were
thrown over, which on hardware is a violent jerk when the policy is switched
on. A healthy checkpoint stays below ~5. Runs the task's play config (random
commands) for 1 s after a reset and reports the share of robots with
max|a| > 20 on the first step and within 1 s, and the falls within 1 s.
Teachers and depth students alike (told apart by the checkpoint keys).

  PYTHONPATH=. python scripts/go2_depth/check_action_spike.py \\
    --task Unitree-Go2-Depth-Teacher-Flat <run>/model_N.pt [more.pt]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("checkpoints", nargs="+")
  ap.add_argument("--task", required=True)
  ap.add_argument("--num-envs", type=int, default=2048)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import src.tasks  # noqa: F401
  from eval_flat import load_checkpoint_policy, step_policy

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  bad = False
  for ckpt in args.checkpoints:
    torch.manual_seed(args.seed)
    # The env state becomes inference tensors once stepped in inference mode,
    # so the next checkpoint's reset must run in it too.
    with torch.inference_mode():
      raw, _ = env.reset()
      obs = TensorDict(raw, batch_size=[env.num_envs])
      policy = load_checkpoint_policy(ckpt, obs, env.device)
      amax = []
      fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
      for _ in range(int(round(1.0 / env.step_dt))):
        a = policy(obs)
        amax.append(a.abs().max(-1).values)
        raw, terminated = step_policy(env, lambda _o, _a=a: _a, obs)
        obs = TensorDict(raw, batch_size=[env.num_envs])
        fell |= terminated
    A = torch.stack(amax)
    first, within = (A[0] > 20).float().mean(), (A > 20).any(0).float().mean()
    bad |= bool(first > 0)
    print(
      f"[spike] {ckpt}: first step max|a| > 20 {first:.1%}, within 1 s {within:.1%}; "
      f"first step max|a| p50/p99/max {A[0].median():.1f}/{A[0].quantile(0.99):.1f}/{A[0].max():.0f}; "
      f"fell within 1 s {fell.float().mean():.1%}"
    )
  env.close()
  return 1 if bad else 0


if __name__ == "__main__":
  sys.exit(main())
