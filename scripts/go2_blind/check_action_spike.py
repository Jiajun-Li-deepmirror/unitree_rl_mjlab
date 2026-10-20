"""Pre-deploy check: does the policy emit huge actions right after a reset?

The end of the stage C run (from ~19100 iterations) diverged: a few % of robots
reset with a moving command produced max|a| of 130-400 on the first step and were
thrown over, which on hardware means a violent jerk when the policy is switched
on. A healthy checkpoint stays below ~5. Runs the flat play config (2048 robots,
random commands) for 1 s after a reset and reports the share of robots with
max|a| > 20 on the first step and within 1 s, and the falls within 1 s.

  python scripts/go2_blind/check_action_spike.py logs/rsl_rl/go2_blind/<run>/model_N.pt [more.pt]
"""

import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls



def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--task", default="Unitree-Go2-Blind-Flat", help="Blind task whose play config matches the policy.")
  parser.add_argument("checkpoints", nargs="+")
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--obs-noise", action="store_true", help="Keep the training observation noise.")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  env_cfg.observations["actor"].enable_corruption = args.obs_noise
  agent_cfg = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(args.task)(venv, asdict(agent_cfg), device="cuda:0")
  for ckpt in args.checkpoints:
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    torch.manual_seed(args.seed)
    env.reset()
    obs = venv.get_observations()
    amax = []
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(int(round(1.0 / env.step_dt))):
      with torch.no_grad():
        a = policy(obs)
        obs, _, dones, _ = venv.step(a)
      amax.append(a.abs().max(-1).values)
      fell |= dones.bool()
    A = torch.stack(amax)
    print(
      f"{ckpt}: first step max|a| > 20 {(A[0] > 20).float().mean():.1%}, within 1 s {(A > 20).any(0).float().mean():.1%}; "
      f"first step max|a| p50/p99/max {A[0].median():.1f}/{A[0].quantile(0.99):.1f}/{A[0].max():.0f}; "
      f"fell within 1 s {fell.float().mean():.1%}"
    )
  env.close()


if __name__ == "__main__":
  main()
