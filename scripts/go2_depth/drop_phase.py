"""Remove the `phase` observation from a wheeled teacher or student checkpoint.

The wheeled tasks dropped the gait clock (go2-blind's wheeled recipe), so a
checkpoint trained with it no longer matches the observation layout. This
deletes the phase columns from the observation normalizers, from the first
layer that reads the observations (teacher actor and critic MLPs, the
student's GRU input) and from that layer's Adam moments, so the policy can
be fine-tuned or warm-started on the new task.

  PYTHONPATH=. python scripts/go2_depth/drop_phase.py \\
    --task Unitree-Go2W-Depth-Teacher --checkpoint old.pt --out new.pt [--std-scale 1.2]

The old layout is the task's layout with the 2-D phase put back after
`command`, with the history length of `command` (term-major history).
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import torch


def group_layouts(task: str) -> dict[str, list[tuple[str, int]]]:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import src.tasks  # noqa: F401

  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  om = env.observation_manager
  layouts = {}
  for group, names in om.active_terms.items():
    dims = om.group_obs_term_dim[group]
    if any(len(d) != 1 for d in dims):
      continue  # depth images etc.
    layouts[group] = [(n, int(d[0])) for n, d in zip(names, dims)]
  env.close()
  return layouts


def kept_columns(layout: list[tuple[str, int]]) -> tuple[torch.Tensor, int]:
  """Old-layout column indices of the new layout's columns, and the old width."""
  if any(n == "phase" for n, _ in layout):
    raise SystemExit("The task still has a phase observation.")
  keep, col = [], 0
  for name, size in layout:
    keep.extend(range(col, col + size))
    col += size
    if name == "command":
      col += 2 * (size // 3)  # the phase block that followed it
  return torch.tensor(keep), col


def concat(layouts: dict, groups: list[str]) -> tuple[torch.Tensor, int]:
  keep, width = [], 0
  for g in groups:
    k, w = kept_columns(layouts[g])
    keep.append(k + width)
    width += w
  return torch.cat(keep), width


def slice_cols(t: torch.Tensor, keep: torch.Tensor, old_width: int) -> torch.Tensor:
  assert t.shape[-1] >= old_width, (tuple(t.shape), old_width)
  rest = t[..., old_width:]  # e.g. the GRU input's depth latent
  return torch.cat([t[..., keep], rest], dim=-1)


def fix_model(state: dict, first_layer: str, keep: torch.Tensor, width: int, olds: dict) -> None:
  for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
    state[key] = slice_cols(state[key], keep, width)
  olds[tuple(state[first_layer].shape)] = (keep, width)
  state[first_layer] = slice_cols(state[first_layer], keep, width)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--task", required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--std-scale", type=float, default=1.0, help="teacher action std factor")
  args = ap.parse_args()

  layouts = group_layouts(args.task)
  ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  olds: dict[tuple[int, ...], tuple[torch.Tensor, int]] = {}
  if "student_state_dict" in ck:
    keep, width = concat(layouts, ["proprio", "proprio_history"])
    fix_model(ck["student_state_dict"], "rnn.rnn.weight_ih_l0", keep, width, olds)
    keep, width = concat(layouts, ["actor"])
    fix_model(ck["teacher_state_dict"], "mlp.0.weight", keep, width, {})
  else:
    for group, key in (("actor", "actor_state_dict"), ("critic", "critic_state_dict")):
      keep, width = concat(layouts, [group])
      fix_model(ck[key], "mlp.0.weight", keep, width, olds)
    if args.std_scale != 1.0:
      std = ck["actor_state_dict"]["distribution.std_param"]
      ck["actor_state_dict"]["distribution.std_param"] = std * args.std_scale
  # Adam moments of the sliced layers, found by their (unique) old shapes.
  matched = set()
  for st in ck.get("optimizer_state_dict", {}).get("state", {}).values():
    for name in ("exp_avg", "exp_avg_sq"):
      t = st.get(name)
      if t is not None and tuple(t.shape) in olds:
        keep, width = olds[tuple(t.shape)]
        matched.add(tuple(t.shape))
        st[name] = slice_cols(t, keep, width)
  if "optimizer_state_dict" in ck and matched != set(olds):
    raise SystemExit(f"Adam moments not found for {set(olds) - matched}")
  torch.save(ck, args.out)
  print(f"[drop_phase] {args.checkpoint} -> {args.out}: first layers {sorted(olds)} lost the phase columns")
  return 0


if __name__ == "__main__":
  sys.exit(main())
