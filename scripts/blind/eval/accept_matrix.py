"""Acceptance matrix: one command set for every robot, flagged against a tolerance.

Robots on flat ground (play config: no pushes, no observation noise) hold one
command each for 8 s; over the last 4 s, per command: actual / commanded body
velocity on each commanded axis (fallen robots left out), and falls. A command
fails when any commanded axis is off by more than --tol (default 20%) or a robot
fell. Uncommanded axes are not judged.

Usage: python accept_matrix.py --task <task> <checkpoint> [--tol 0.2]
"""

import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

CMDS = [
  # straight
  (0.3, 0, 0), (0.5, 0, 0), (1.0, 0, 0), (1.5, 0, 0), (-0.5, 0, 0), (-1.0, 0, 0),
  # sideways and diagonal
  (0, 0.3, 0), (0, -0.3, 0), (0, 0.6, 0), (0, -0.6, 0), (0.6, 0.4, 0), (0.6, -0.4, 0),
  # in place
  (0, 0, 0.2), (0, 0, -0.2), (0, 0, 0.5), (0, 0, -0.5), (0, 0, 1.0), (0, 0, -1.0), (0, 0, 1.5), (0, 0, -1.5),
  # arcs
  (0.5, 0, 0.15), (0.5, 0, -0.15), (1.0, 0, 0.15), (1.0, 0, -0.15), (1.0, 0, 0.2), (1.0, 0, -0.2),
  (1.5, 0, 0.15), (1.5, 0, -0.15), (1.5, 0, 0.2), (1.5, 0, -0.2), (1.5, 0, 0.3), (1.5, 0, -0.3),
  (-0.5, 0, 0.2), (-0.5, 0, -0.2), (-1.0, 0, 0.2), (-1.0, 0, -0.2), (0.8, 0, 0.8), (0.8, 0, -0.8),
  # combined and small sticks
  (0.3, 0.3, 0.2), (0, 0.4, -0.3), (0.15, 0, 0.3), (0.2, 0, 0), (0, 0.15, 0), (1.0, 0, 0.1),
]  # fmt: skip

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--tol", type=float, default=0.2)
p.add_argument("--envs-per-cmd", type=int, default=32)
args = p.parse_args()

n = args.envs_per_cmd
cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = len(CMDS) * n
cfg.episode_length_s = 1000.0
agent_cfg = load_rl_cfg(args.task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = load_runner_cls(args.task)(venv, asdict(agent_cfg), device="cuda:0")
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
dev, dt = env.device, env.step_dt
robot = env.scene["robot"]
cmd = env.command_manager.get_term("twist")
C = torch.tensor(CMDS, device=dev, dtype=torch.float32).repeat_interleave(n, 0)
steps, start = int(round(8.0 / dt)), int(round(4.0 / dt))
vel_sum = torch.zeros(len(C), 3, device=dev)
fell = torch.zeros(len(C), dtype=torch.bool, device=dev)
obs = venv.get_observations()
for k in range(steps):
  cmd.vel_command_b[:] = C
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, dones, _ = venv.step(policy(obs))
  fell |= dones.bool()
  if k >= start:
    v = robot.data.root_link_lin_vel_b[:, :2]
    w = robot.data.root_link_ang_vel_b[:, 2:3]
    vel_sum += torch.cat([v, w], dim=1)
mean_vel = vel_sum / (steps - start)

print(f"{args.checkpoint}\n{args.task}: {n} robots per command, last 4 s of 8 s, tolerance {args.tol:.0%}")
print(f"{'command (vx, vy, wz)':>22} | {'actual / command':>22} | falls | verdict")
fails = 0
for ci, c in enumerate(CMDS):
  sl = slice(ci * n, (ci + 1) * n)
  ok = ~fell[sl]
  m = mean_vel[sl][ok].mean(0) if ok.any() else torch.full((3,), float("nan"), device=dev)
  parts, bad = [], int((~ok).sum()) > 0
  for axis, name in enumerate("xyw"):
    if abs(c[axis]) >= 0.05:
      r = (m[axis] / c[axis]).item()
      parts.append(f"{name}{r * 100:.0f}%")
      bad |= not (abs(r - 1.0) <= args.tol)
  fails += bad
  print(f"{str(c):>22} | {' '.join(parts):>22} | {int((~ok).sum()):2d}/{n} | {'FAIL' if bad else 'ok'}")
print(f"failed {fails} of {len(CMDS)} commands")
env.close()
