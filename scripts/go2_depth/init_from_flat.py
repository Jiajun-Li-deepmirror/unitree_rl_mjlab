"""Start a full-terrain teacher from its flat-ground checkpoint (go2-blind).

go2-blind trained its wheeled robots' stairs from the flat model: from
scratch the stairs came but the flat omnidirectional skills did not (28/44
commands off), from the flat model +500 it they stayed (6/44). The flat
checkpoint's observation normalizers saw only flat ground, though: the
height-scan spread there is a small fraction of the stairs' (go2-blind:
1/26), so stairs would read as tens of sigmas. This rolls the flat policy
out on the full-terrain task, measures the observation statistics there and
writes them over the dimensions whose flat spread is too narrow, in the
actor (our teacher's actor sees the scan too) and the critic, with the
sample count lowered so they keep adapting during the fine-tune.

  PYTHONPATH=. python scripts/go2_depth/init_from_flat.py \\
    --task Unitree-As2W-Depth-Teacher --checkpoint <flat run>/model_2999.pt \\
    --out <full_init>/model_2999.pt
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
  ap.add_argument("--task", required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--num-envs", type=int, default=1024)
  ap.add_argument("--steps", type=int, default=250)
  ap.add_argument("--ratio", type=float, default=2.0, help="replace dims whose std grows by more")
  ap.add_argument("--count", type=float, default=2.4e7, help="normalizer sample count afterwards")
  args = ap.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import src.tasks  # noqa: F401
  from eval_map3 import load_policy

  cfg = load_env_cfg(args.task, play=False)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  raw, _ = env.reset()
  obs = TensorDict(raw, batch_size=[env.num_envs])
  policy = load_policy("teacher", args.checkpoint, obs, env.device)
  groups = ("actor", "critic")
  s1 = {g: torch.zeros(obs[g].shape[-1], dtype=torch.float64, device=env.device) for g in groups}
  s2 = {g: torch.zeros_like(s1[g]) for g in groups}
  n = 0
  with torch.inference_mode():
    for _ in range(args.steps):
      raw, *_ = env.step(policy(obs))
      obs = TensorDict(raw, batch_size=[env.num_envs])
      for g in groups:
        x = obs[g].double()
        s1[g] += x.sum(0)
        s2[g] += (x * x).sum(0)
      n += env.num_envs
  env.close()

  ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  for g, key in zip(groups, ("actor_state_dict", "critic_state_dict")):
    mean = (s1[g] / n).float().cpu()
    var = (s2[g] / n - (s1[g] / n) ** 2).clamp(min=0).float().cpu()
    state = ck[key]
    old_std = state["obs_normalizer._std"].squeeze(0).clone()
    new_std = var.sqrt()
    wide = new_std > args.ratio * old_std.clamp(min=1e-6)
    state["obs_normalizer._mean"][0, wide] = mean[wide]
    state["obs_normalizer._var"][0, wide] = var[wide]
    state["obs_normalizer._std"][0, wide] = new_std[wide]
    state["obs_normalizer.count"] = torch.tensor(args.count, dtype=state["obs_normalizer.count"].dtype)
    print(f"[init_from_flat] {g}: {int(wide.sum())}/{wide.numel()} dims re-normalized "
          f"(median std x{float((new_std[wide] / old_std[wide].clamp(min=1e-6)).median()) if wide.any() else 0:.1f})")
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  torch.save(ck, args.out)
  print(f"[init_from_flat] {args.checkpoint} -> {args.out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
