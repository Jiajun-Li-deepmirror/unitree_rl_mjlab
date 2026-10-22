"""Gait numbers of a wheeled policy on flat ground for go2-depth's questions.

Play config (no pushes, no observation noise, wheel brake as trained). Each
command is held by --envs-per-cmd robots for 8 s; over the last 4 s (fallen robots
left out): tracking ratio on the commanded axes, share of steps with all four
wheels on the ground, and touchdowns (off -> on ground) per second for each foot
(mean over robots), with the min and max foot.

Usage: python gait_numbers.py --task Unitree-As2W-Blind-Flat <checkpoint> [--envs-per-cmd 32]
"""

import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

CMDS = [  # forward / backward only (2026-10-21: wheeled stairs models stepped there)
  (0.2, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0), (-0.5, 0.0, 0.0), (-1.0, 0.0, 0.0),
]

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
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
feet = env.scene["feet_ground_contact"]
names = [s.primary_name.replace("_foot_collision", "") for s in feet._slots if s.field_name == "found"]
term = env.command_manager.get_term("twist")
C = torch.tensor(CMDS, device=dev, dtype=torch.float32).repeat_interleave(n, 0)
steps, start = int(round(8.0 / dt)), int(round(4.0 / dt))
vel_sum = torch.zeros(len(C), 3, device=dev)
touch = torch.zeros(len(C), 4, device=dev)
all_down = torch.zeros(len(C), device=dev)
air = torch.zeros(len(C), 4, device=dev)
fell = torch.zeros(len(C), dtype=torch.bool, device=dev)
obs = venv.get_observations()
prev = None
for k in range(steps):
  term.vel_command_b[:] = C
  term.time_left[:] = 1e9
  with torch.inference_mode():
    obs, _, dones, _ = venv.step(policy(obs))
  fell |= dones.bool()
  on = feet.data.found > 0
  if k >= start and prev is not None:
    v = torch.cat([robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]], dim=1)
    vel_sum += v
    touch += (on & ~prev).float()
    all_down += on.all(dim=1).float()
    air += (~on).float()
  prev = on.clone()
window = (steps - start) * dt
print(f"{args.checkpoint} on {args.task}: {n} robots per command, last 4 s of 8 s; feet {names}")
print(f"{'command':>16} | {'ratio':>16} | {'all down':>8} | touchdowns/s per foot ({', '.join(names)}) | min  max | falls | air share per foot")
for i, c in enumerate(CMDS):
  sl = slice(i * n, (i + 1) * n)
  ok = ~fell[sl]
  m = ok.sum().clamp(min=1)
  vel = vel_sum[sl][ok].sum(0) / m / (steps - start)
  ratio = " ".join(f"{a}{vel[j] / c[j] * 100:.0f}%" for j, a in enumerate("xyw") if c[j] != 0.0)
  per_foot = touch[sl][ok].sum(0) / m / window
  down = all_down[sl][ok].sum() / m / (steps - start) * 100
  air_share = air[sl][ok].sum(0) / m / (steps - start) * 100
  print(f"{str(c):>16} | {ratio:>16} | {down:7.1f}% | {' '.join(f'{x:4.2f}' for x in per_foot.tolist())} | "
        f"{per_foot.min():4.2f} {per_foot.max():4.2f} | {int(fell[sl].sum())}/{n} | "
        f"{' '.join(f'{x:3.0f}%' for x in air_share.tolist())}")
env.close()
