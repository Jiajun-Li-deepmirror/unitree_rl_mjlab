"""Copy a checkpoint with its action noise std scaled (start of stage C of the release recipe).

Stage C resumed from model_9000 with the std multiplied by 1.5 (0.40-0.55 ->
0.60-0.83), so that the new foothold_height_gain reward had higher swings to work
with. Everything else in the checkpoint, iteration counter included, is kept.

  python scripts/go2_blind/scale_action_std.py logs/rsl_rl/go2_blind/<run>/model_9000.pt \
    logs/rsl_rl/go2_blind/<run>/model_9000_explore.pt --scale 1.5
"""

import argparse

import torch


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("src")
  parser.add_argument("dst")
  parser.add_argument("--scale", type=float, default=1.5)
  args = parser.parse_args()

  ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
  key = "distribution.std_param"
  std = ckpt["actor_state_dict"][key]
  ckpt["actor_state_dict"][key] = std * args.scale
  torch.save(ckpt, args.dst)
  print(f"{args.dst}: action std x{args.scale}: {[round(float(s), 3) for s in std]} -> "
        f"{[round(float(s), 3) for s in ckpt['actor_state_dict'][key]]}")


if __name__ == "__main__":
  main()
