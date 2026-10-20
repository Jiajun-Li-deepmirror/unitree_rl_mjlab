"""Widen a checkpoint for observation terms appended at the end of the actor and
critic groups (e.g. apply_blind_heading's heading_error).

The new first-layer weight columns and their Adam moments are zero, so the
widened policy acts exactly like the old one until training uses the new inputs.
The observation normalizers get mean 0 and std --new-std for the new inputs (their
counts stay, so they hardly move: pick a std of the input's typical size).

  python scripts/blind/widen_obs.py --task Unitree-As2-Blind-Map3-Heading \
    logs/rsl_rl/as2_blind/<run>/model_8500.pt

writes model_8500_widened.pt next to the checkpoint.
"""

import argparse
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, help="task with the appended observation terms")
p.add_argument("checkpoint")
p.add_argument("--out", default=None, help="default: <checkpoint>_widened.pt")
p.add_argument("--new-std", type=float, default=0.2)
args = p.parse_args()
src_path = Path(args.checkpoint)
out = Path(args.out) if args.out else src_path.with_name(f"{src_path.stem}_widened.pt")

cfg = load_env_cfg(args.task)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
dims = {g: int(env.observation_manager.group_obs_dim[g][0]) for g in ("actor", "critic")}
env.close()

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
widened = {}  # old first-layer weight shape -> new input width
for group in ("actor", "critic"):
  sd = ck[f"{group}_state_dict"]
  weight = sd["mlp.0.weight"]
  old, new = weight.shape[1], dims[group]
  if new < old:
    raise ValueError(f"{group}: the task has {new} inputs, the checkpoint {old}.")
  extra = new - old
  print(f"{group}: inputs {old} -> {new}")
  if extra == 0:
    continue
  sd["mlp.0.weight"] = torch.cat([weight, weight.new_zeros(weight.shape[0], extra)], dim=1)
  for key, fill in (("_mean", 0.0), ("_var", args.new_std**2), ("_std", args.new_std)):
    stat = sd[f"obs_normalizer.{key}"]
    sd[f"obs_normalizer.{key}"] = torch.cat([stat, stat.new_full((stat.shape[0], extra), fill)], dim=1)
  widened[tuple(weight.shape)] = new

state = ck["optimizer_state_dict"]["state"]
for entry in state.values():
  for key in ("exp_avg", "exp_avg_sq"):
    t = entry.get(key)
    if t is not None and tuple(t.shape) in widened:
      extra = widened[tuple(t.shape)] - t.shape[1]
      entry[key] = torch.cat([t, t.new_zeros(t.shape[0], extra)], dim=1)
torch.save(ck, out)
print(f"wrote {out} (iter {ck['iter']})")
