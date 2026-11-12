#!/usr/bin/env python3
"""avg_ckpt.py OUT CKPT ... (unitree-rl-mjlab-62 2026-11-12, the review's item 9): one acceptance candidate made of
several checkpoints of one unified student run (stage 2, a PPO checkpoint): every floating-point tensor of the actor
(actor_state_dict -- what the checks, the export and the videos load) and of its distillation copy
(student_state_dict) is the mean over the checkpoints; everything else (critic, teacher, optimizer, iter, infos, the
normalizers' integer counts) comes from the last one, and infos["averaged"] lists the sources.

Why (doc/blind_phased_training.md 8.1): single checkpoints near the bars flip from one save to the next; the blind
chains' averaged last 8 were stable. jobs/avg_student.py does the same for a distillation checkpoint's student only.
Exit 2 when the checkpoints do not have the same tensors.
"""

from __future__ import annotations

import sys

import torch

KEYS = ("actor_state_dict", "student_state_dict")


def average(paths: list[str]) -> tuple[dict, int]:
  cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
  out = cks[-1]
  n = 0
  for key in KEYS:
    if key not in out:
      continue
    sd = out[key]
    for c, p in zip(cks, paths):
      if key not in c or set(c[key]) != set(sd):
        raise ValueError(f"{p}: {key} differs from {paths[-1]}'s")
    for name, value in sd.items():
      if torch.is_floating_point(value):
        sd[name] = torch.stack([c[key][name].float() for c in cks]).mean(0).to(value.dtype)
        n += 1
  out.setdefault("infos", {})
  out["infos"] = {**(out["infos"] or {}), "averaged": list(paths)}
  return out, n


def main(argv: list[str]) -> int:
  if len(argv) < 2 or argv[0] in ("-h", "--help"):
    print(__doc__)
    return 0 if argv and argv[0] in ("-h", "--help") else 2
  out, paths = argv[0], argv[1:]
  try:
    avg, n = average(paths)
  except ValueError as e:
    print(f"avg_ckpt: {e}", file=sys.stderr)
    return 2
  torch.save(avg, out)
  print(f"wrote {out}: {n} tensors averaged over {len(paths)} checkpoints ({paths[0]} .. {paths[-1]}), iter {avg.get('iter')}")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
