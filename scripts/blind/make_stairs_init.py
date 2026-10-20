"""Make the starting checkpoint of a stairs fine-tune from a flat policy.

A flat run's observation normalizers have seen some 5e8 flat samples, so their
statistics barely move any more, and the critic's height-scan statistics are
those of flat ground: stair scans come out at tens of sigma. This copies the flat
checkpoint and
- replaces the critic normalizer's mean and variance by statistics of the stairs
  task, measured by rolling out the flat policy on its training terrain (every
  difficulty row) for --steps control steps of --num-envs robots;
- lowers both normalizer counts to --count, so they keep adapting to stairs data
  over a few hundred iterations.
The actor's statistics stay the flat ones, so the first stairs iteration acts
exactly like the flat policy.

  python scripts/blind/make_stairs_init.py --task Unitree-B2-Blind-R1-Map3 \
    logs/rsl_rl/b2_blind/<flat run>/model_4999.pt

writes model_4999_stairsinit.pt next to the checkpoint; resume the stairs task from
it with --agent.load-checkpoint='^model_4999_stairsinit\\.pt$'.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

EPS = 1e-2  # rsl_rl EmpiricalNormalization: (x - mean) / (std + eps)

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, help="stairs task of the fine-tune")
p.add_argument("checkpoint", help="flat checkpoint")
p.add_argument("--out", default=None, help="default: <checkpoint>_stairsinit.pt")
p.add_argument("--num-envs", type=int, default=1024)
p.add_argument("--steps", type=int, default=250)
p.add_argument("--count", type=float, default=2.4e7)
args = p.parse_args()
src_path = Path(args.checkpoint)
out = Path(args.out) if args.out else src_path.with_name(f"{src_path.stem}_stairsinit.pt")

cfg = load_env_cfg(args.task)
cfg.scene.num_envs = args.num_envs
cfg.scene.terrain.max_init_terrain_level = None  # every difficulty row
agent_cfg = load_rl_cfg(args.task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = load_runner_cls(args.task)(venv, asdict(agent_cfg), device="cuda:0")
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
# Loading restored the step counter of the flat run; resetting again applies the
# curricula at that counter (final command ranges), as the resumed run will.
print(f"step counter {env.common_step_counter} (iteration {runner.current_learning_iteration})")
env.reset()

obs = venv.get_observations()
total = torch.zeros(obs["critic"].shape[1], dtype=torch.float64, device=env.device)
total_sq = torch.zeros_like(total)
n = 0
for _ in range(args.steps):
  x = obs["critic"].double()
  total += x.sum(0)
  total_sq += (x * x).sum(0)
  n += x.shape[0]
  with torch.inference_mode():
    obs, _, _, _ = venv.step(policy(obs))
mean = total / n
var = (total_sq / n - mean * mean).clamp(min=0.0)

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
critic = ck["critic_state_dict"]
old_mean = critic["obs_normalizer._mean"].double().squeeze(0)
old_std = critic["obs_normalizer._std"].double().squeeze(0)
assert old_mean.shape == mean.shape, (old_mean.shape, mean.shape)
new_mean, new_std = mean.cpu(), var.sqrt().cpu()

# Per critic term: how far the stairs data sits from the flat statistics.
print(f"{n} critic samples ({args.num_envs} robots x {args.steps} steps) on {args.task}")
print(f"{'term':>28} | dims | std new/flat median  max | |mean shift| in flat sigma max")
start = 0
manager = env.observation_manager
for name, dims in zip(manager.active_terms["critic"], manager.group_obs_term_dim["critic"], strict=True):
  size = int(torch.tensor(dims).prod())
  sl = slice(start, start + size)
  ratio = (new_std[sl] + EPS) / (old_std[sl] + EPS)
  shift = (new_mean[sl] - old_mean[sl]).abs() / (old_std[sl] + EPS)
  print(f"{name:>28} | {size:4d} | {ratio.median():8.2f} {ratio.max():8.2f} | {shift.max():8.2f}")
  start += size

critic["obs_normalizer._mean"] = new_mean.float().unsqueeze(0)
critic["obs_normalizer._var"] = var.cpu().float().unsqueeze(0)
critic["obs_normalizer._std"] = new_std.float().unsqueeze(0)
for part in ("actor_state_dict", "critic_state_dict"):
  old = int(ck[part]["obs_normalizer.count"])
  ck[part]["obs_normalizer.count"] = torch.tensor(int(args.count), dtype=torch.long)
  print(f"{part}: normalizer count {old:.3g} -> {args.count:.3g}")
torch.save(ck, out)
print(f"wrote {out} (iter {ck['iter']})")
env.close()
