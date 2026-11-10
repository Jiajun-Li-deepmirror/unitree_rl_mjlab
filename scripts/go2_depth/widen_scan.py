"""Widen a blind checkpoint into a depth teacher's (depth_env_cfg.apply_depth_teacher):
the privileged scan appended to the actor and the critic.

Like scripts/blind/widen_obs.py the new first-layer columns and their Adam moments
are zero, so the widened policy acts exactly like the blind one until training uses
the scan. The scan's normalizer statistics are measured instead of guessed: the
blind policy walks the task's training terrain (`--num-envs` robots, `--steps`
control steps after `--warmup`), and the new inputs get the mean and variance seen
there. The normalizers' counts stay (~1e9 samples), so those statistics hardly move
in training: with mean 0 / std 0.2 the scan (floor ~0.05, walls -0.2 after scaling)
would stay mis-normalized.

  BLIND_PILOT=<the blind model's pilot> python scripts/go2_depth/widen_scan.py \
    --task Unitree-Go2-Depth-Polish logs/rsl_rl/go2_blind/<run>/model_8000.pt --out <path>

BLIND_PILOT must be the blind model's (blind_pilot.txt): the tasks read it when they
register.
"""

import argparse
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, help="a depth teacher task (actor and critic with the scan last)")
p.add_argument("checkpoint", help="the blind checkpoint")
p.add_argument("--out", required=True)
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--steps", type=int, default=400)
args = p.parse_args()

cfg = load_env_cfg(args.task)
cfg.scene.num_envs = args.num_envs
agent_cfg = load_rl_cfg(args.task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
dims = {g: int(env.observation_manager.group_obs_dim[g][0]) for g in ("actor", "critic")}

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
old = {g: ck[f"{g}_state_dict"]["mlp.0.weight"].shape[1] for g in ("actor", "critic")}
for g in ("actor", "critic"):
  if dims[g] <= old[g]:
    raise ValueError(f"{g}: the task has {dims[g]} inputs, the checkpoint {old[g]}: nothing to widen.")
  print(f"{g}: inputs {old[g]} -> {dims[g]}")

# The blind actor (rsl_rl MLPModel: normalizer, ELU MLP, mean action), on the first inputs.
sd = {k: v.to("cuda:0") for k, v in ck["actor_state_dict"].items()}
layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})


def blind_actor(x: torch.Tensor) -> torch.Tensor:
  x = (x - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + 1e-2)  # rsl_rl's eps
  for i, n in enumerate(layers):
    x = x @ sd[f"mlp.{n}.weight"].T + sd[f"mlp.{n}.bias"]
    if i < len(layers) - 1:
      x = torch.nn.functional.elu(x)
  return x


samples = {g: [] for g in ("actor", "critic")}
with torch.inference_mode():
  env.reset()
  obs = venv.get_observations()
  for step in range(args.warmup + args.steps):
    if step >= args.warmup:
      for g in samples:
        samples[g].append(obs[g][:, old[g]:].float().cpu())
    obs, _, _, _ = venv.step(blind_actor(obs["actor"][:, : old["actor"]]))
env.close()

widened = {}  # old first-layer weight shape -> new input width
for g in ("actor", "critic"):
  x = torch.cat(samples[g])
  mean, var = x.mean(0, keepdim=True), x.var(0, keepdim=True).clamp_min(1e-4)
  print(f"{g} scan ({x.shape[1]} inputs, {x.shape[0]} samples): mean {mean.mean():+.3f} "
        f"[{mean.min():+.3f}, {mean.max():+.3f}], std {var.sqrt().mean():.3f} [{var.sqrt().min():.3f}, {var.sqrt().max():.3f}]")
  state = ck[f"{g}_state_dict"]
  weight = state["mlp.0.weight"]
  extra = dims[g] - old[g]
  state["mlp.0.weight"] = torch.cat([weight, weight.new_zeros(weight.shape[0], extra)], dim=1)
  for key, new in (("_mean", mean), ("_var", var), ("_std", var.sqrt())):
    stat = state[f"obs_normalizer.{key}"]
    state[f"obs_normalizer.{key}"] = torch.cat([stat, new.to(stat.dtype)], dim=1)
  widened[tuple(weight.shape)] = dims[g]

for entry in ck["optimizer_state_dict"]["state"].values():
  for key in ("exp_avg", "exp_avg_sq"):
    t = entry.get(key)
    if t is not None and tuple(t.shape) in widened:
      entry[key] = torch.cat([t, t.new_zeros(t.shape[0], widened[tuple(t.shape)] - t.shape[1])], dim=1)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(ck, out)
print(f"wrote {out} (iter {ck['iter']})")
