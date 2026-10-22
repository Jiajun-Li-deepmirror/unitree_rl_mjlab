"""Standing still on flat ground: from rest and after letting go of the stick.

Groups of robots (play config: no pushes, no observation noise) hold a command for
3 s, then the zero command for 6 s ("rest" holds zero throughout). Per group,
robots that fell left out:
  stop s   : time from release until the base speed stays under 0.05 m/s,
  stop cm  : distance travelled from release to that moment,
  creep cm : distance travelled over the last 4 s,
  yaw deg  : heading change over the last 4 s,
  air %    : share of foot/wheel-time off the ground over the last 4 s,
  flips/s  : contact changes per second per foot over the last 4 s,
plus the mean value per second (= per-step value) over the zero-command phase of
standing_position_hold / standing_speed / standing_feet_air, for weights.

Usage: python eval_standing.py --task <flat task> <checkpoint> [--envs-per-group 16]
"""

import argparse
import math
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

GROUPS = [
  ("rest", (0.0, 0.0, 0.0)),
  ("fwd 1.0", (1.0, 0.0, 0.0)),
  ("back 1.0", (-1.0, 0.0, 0.0)),
  ("side 0.6", (0.0, 0.6, 0.0)),
  ("turn 1.0", (0.0, 0.0, 1.0)),
  ("arc 1.0+0.3", (1.0, 0.0, 0.3)),
  ("fwd 1.5", (1.5, 0.0, 0.0)),
]

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--envs-per-group", type=int, default=16)
args = p.parse_args()

n = args.envs_per_group
cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = len(GROUPS) * n
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
cmd = env.command_manager.get_term("twist")
B = len(GROUPS) * n
C = torch.tensor([c for _, c in GROUPS], device=dev).repeat_interleave(n, 0)
moving_group = C.abs().sum(1) > 0

release, total, window = int(3.0 / dt), int(9.0 / dt), int(4.0 / dt)
fell = torch.zeros(B, dtype=torch.bool, device=dev)
stop_step = torch.full((B,), -1, dtype=torch.long, device=dev)
pos_release = torch.zeros(B, 2, device=dev)
stop_dist = torch.zeros(B, device=dev)
pos_w0 = yaw_w0 = None
air = torch.zeros(B, device=dev)
flips = torch.zeros(B, device=dev)
prev_contact = None
anchor_pos = torch.zeros(B, 2, device=dev)
anchor_yaw = torch.zeros(B, device=dev)
hold_sum = torch.zeros(B, device=dev)
speed_sum = torch.zeros(B, device=dev)
air_term_sum = torch.zeros(B, device=dev)
zero_steps = 0
obs = venv.get_observations()
for k in range(total):
  standing = (k >= release) | ~moving_group
  cmd.vel_command_b[:] = torch.where(standing.unsqueeze(1), 0.0, C)
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, dones, _ = venv.step(policy(obs))
  fell |= dones.bool()
  pos = robot.data.root_link_pos_w[:, :2]
  yaw = robot.data.heading_w
  v = torch.norm(robot.data.root_link_lin_vel_b[:, :2], dim=1)
  w = robot.data.root_link_ang_vel_b[:, 2].abs()
  contact = feet.data.found > 0
  if k == release - 1:
    pos_release = pos.clone()
  if k >= release:
    # Stopped: speed under 0.05 m/s from here on (reset if it speeds up again).
    slow = v < 0.05
    newly = slow & (stop_step < 0)
    stop_step = torch.where(newly, k, stop_step)
    stop_dist = torch.where(newly, torch.norm(pos - pos_release, dim=1), stop_dist)
    stop_step = torch.where(~slow, -1, stop_step)
    # Term values over the zero-command phase.
    follow = (v > 0.1) | (w > 0.2) | (k == release)
    anchor_pos = torch.where(follow.unsqueeze(1), pos, anchor_pos)
    anchor_yaw = torch.where(follow, yaw, anchor_yaw)
    turn = torch.atan2(torch.sin(yaw - anchor_yaw), torch.cos(yaw - anchor_yaw)).abs()
    hold_sum += torch.norm(pos - anchor_pos, dim=1).clamp(max=0.3) + 0.3 * turn
    speed_sum += v.clamp(max=1.0) + 0.3 * w.clamp(max=1.0)
    calm = torch.exp(-(v**2 + w**2) / 0.1)
    air_term_sum += (~contact).float().mean(1) * calm
    zero_steps += 1
  if k == total - window:
    pos_w0, yaw_w0 = pos.clone(), yaw.clone()
  if k > total - window:
    air += (~contact).float().mean(1)
    flips += (contact != prev_contact).float().mean(1)
  prev_contact = contact

creep = torch.norm(robot.data.root_link_pos_w[:, :2] - pos_w0, dim=1) * 100
dyaw = torch.rad2deg(torch.atan2(torch.sin(robot.data.heading_w - yaw_w0), torch.cos(robot.data.heading_w - yaw_w0)).abs())
stop_t = torch.where(stop_step >= 0, (stop_step - release + 1).float() * dt, float("nan"))
ok = ~fell

print(f"{args.checkpoint}\n{args.task}: {n} robots per group; command 3 s, then zero 6 s; last 4 s for creep")
print(f"{'group':>12} | {'stop s':>6} {'stop cm':>7} | {'creep cm':>8} {'p90':>5} {'yaw deg':>7} | {'air %':>5} {'flips/s':>7} | {'hold/s':>6} {'speed/s':>7} {'air/s':>6} | falls")
for gi, (name, _) in enumerate(GROUPS):
  s = slice(gi * n, (gi + 1) * n)
  m = ok[s]
  def mean(x):
    x = x[s][m]
    x = x[~torch.isnan(x)]
    return float(x.mean()) if len(x) else math.nan
  cp = creep[s][m]
  p90 = float(torch.quantile(cp, 0.9)) if len(cp) else math.nan
  print(
    f"{name:>12} | {mean(stop_t):6.2f} {mean(stop_dist) * 100:7.1f} | {mean(creep):8.2f} {p90:5.2f} {mean(dyaw):7.2f} | "
    f"{mean(air / (window - 1)) * 100:5.1f} {mean(flips / (window - 1) / dt):7.2f} | "
    f"{mean(hold_sum / zero_steps):6.3f} {mean(speed_sum / zero_steps):7.3f} {mean(air_term_sum / zero_steps):6.3f} | {int((~m).sum())}/{n}"
  )
env.close()
