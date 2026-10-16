"""Per-sub-terrain breakdown of a teacher checkpoint on the training env.

Runs one episode with the training command distribution and reports, per
sub-terrain type: falls, velocity tracking error, how often the robot stalls
under a non-trivial command, displacement from spawn and terrain level.

  PYTHONPATH=. python scripts/go2_depth/diag_terrain.py \
    --checkpoint logs/rsl_rl/go2_depth_teacher/<run>/model_N.pt
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from tensordict import TensorDict


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--task", default="Unitree-Go2-Depth-Teacher")
  parser.add_argument("--num-envs", type=int, default=2000)
  parser.add_argument("--seconds", type=float, default=20.0)
  parser.add_argument(
    "--levels", type=int, nargs=2, default=None, metavar=("LO", "HI"),
    help="Place every env on a random terrain row in [LO, HI] (default: training init).",
  )
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from rsl_rl.models import MLPModel

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  cfg.curriculum.pop("terrain_levels", None)
  cfg.episode_length_s = args.seconds + 5.0
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  if args.levels is not None:
    terrain = env.scene.terrain
    lo, hi = args.levels
    terrain.terrain_levels[:] = torch.randint_like(terrain.terrain_levels, lo, hi + 1)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
  raw, _ = env.reset()
  obs = TensorDict(raw, batch_size=[env.num_envs])
  model = MLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    12,
    hidden_dims=(512, 256, 128),
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  model.load_state_dict(ck["actor_state_dict"])
  model = model.to(env.device).eval()

  gen = cfg.scene.terrain.terrain_generator
  names = list(gen.sub_terrains)
  props = np.array([c.proportion for c in gen.sub_terrains.values()])
  cum = np.cumsum(props / props.sum())
  col_type = [int(np.min(np.where(i / gen.num_cols + 0.001 < cum)[0])) for i in range(gen.num_cols)]
  terrain = env.scene.terrain
  env_type = torch.tensor(col_type, device=env.device)[terrain.terrain_types]
  levels = terrain.terrain_levels.clone()

  robot = env.scene["robot"]
  cmd_term = env.command_manager.get_term("twist")
  n = env.num_envs
  fell = torch.zeros(n, dtype=torch.bool, device=env.device)
  err = torch.zeros(n, device=env.device)
  active = torch.zeros(n, device=env.device)
  stalled = torch.zeros(n, device=env.device)
  term_names = [t for t in env.termination_manager.active_terms if t != "time_out"]
  term_first = {t: torch.zeros(n, dtype=torch.bool, device=env.device) for t in term_names}
  leg_touch = torch.zeros(n, device=env.device)
  leg_sensor = env.scene["leg_ground_touch"]
  steps = int(args.seconds / env.step_dt)
  for _ in range(steps):
    with torch.inference_mode():
      raw, _, terminated, _, _ = env.step(model(obs))
    obs = TensorDict(raw, batch_size=[n])
    for t in term_names:
      term_first[t] |= env.termination_manager.get_term(t) & ~fell
    leg_touch += (leg_sensor.data.found.reshape(n, -1) > 0).any(dim=1).float() * (~fell).float()
    fell |= terminated
    cmd = cmd_term.vel_command_b[:, :2]
    v = robot.data.root_link_lin_vel_b[:, :2]
    moving = (torch.norm(cmd, dim=1) > 0.15) & ~fell
    err += torch.where(moving, torch.norm(cmd - v, dim=1), 0.0)
    active += moving.float()
    stalled += (moving & (torch.norm(v, dim=1) < 0.05)).float()
  disp = torch.norm(robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)

  print(f"\n[diag] {args.checkpoint}  ({n} envs, {args.seconds:.0f}s, levels 0-{int(levels.max())})")
  print(
    f"  {'terrain':16s} {'envs':>5s} {'fall':>6s} {'err_xy':>7s} {'stall':>6s} {'disp_m':>7s} {'level':>6s} "
    + " ".join(f"{t[:12]:>12s}" for t in term_names) + f" {'leg_touch':>9s}"
  )
  for t, name in enumerate(names):
    m = env_type == t
    if not m.any():
      continue
    ok = m & ~fell
    a = active[ok].sum().clamp(min=1)
    print(
      f"  {name:16s} {int(m.sum()):5d} {float(fell[m].float().mean()):6.2f} "
      f"{float(err[ok].sum() / a):7.3f} {float(stalled[ok].sum() / a):6.2f} "
      f"{float(disp[ok].mean()):7.2f} {float(levels[m].float().mean()):6.2f} "
      + " ".join(f"{float(term_first[t][m].float().mean()):12.2f}" for t in term_names)
      + f" {float(leg_touch[m].sum() / (steps * m.sum())):9.2f}"
    )
  env.close()


if __name__ == "__main__":
  main()
