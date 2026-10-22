"""Drift on flat ground with a pure command and no operator: straight, sideways and
turning in place.

Play config of a flat task (no pushes, no observation noise), N robots per command,
all commands in one simulation (each robot holds its own command for T s from a
standstill); per command, over the robots that did not fall:
- heading change (deg, + = left) and the offsets along / across the start heading
  (m, + = forward / left), p10 p50 p90;
- distance of the base from its start point at the end and the largest distance
  during the run (m, p50 p90).
Driving straight the drift is the heading change and the sideways offset; walking
sideways it is the heading change and the forward offset; turning in place it is
how far the base wanders from its spot.

Usage: python drift_all.py --task <flat task> <checkpoint> [--num-envs 64] [--seconds 10]
"""
import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--num-envs", type=int, default=64)
p.add_argument("--seconds", type=float, default=10.0)
args = p.parse_args()
CMDS = [
  (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0),
  (0.0, 0.3, 0.0), (0.0, -0.3, 0.0), (0.0, 0.6, 0.0), (0.0, -0.6, 0.0),
  (0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.5), (0.0, 0.0, -1.5),
]  # fmt: skip

cfg = load_env_cfg(args.task, play=True)
n = args.num_envs
cfg.scene.num_envs = n * len(CMDS)
cfg.episode_length_s = args.seconds + 5.0
cfg.events.pop("push_robot", None)
gen = cfg.scene.terrain.terrain_generator
gen.size = (120.0, 120.0)
gen.num_rows = gen.num_cols = 1
agent = load_rl_cfg(args.task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(args.task)(venv, asdict(agent), device="cuda:0")
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
robot = env.scene["robot"]
cmd = env.command_manager.get_term("twist")
q = torch.tensor((0.1, 0.5, 0.9), device=env.device)
f3 = lambda x: " ".join(f"{v:6.2f}" for v in x.quantile(q).tolist())
f2 = lambda x: " ".join(f"{v:5.2f}" for v in x.quantile(q[1:]).tolist())
print(f"{args.checkpoint}\n{args.task}: {n} robots per command, {args.seconds:.0f} s per command, all commands at once")
print(f"{'command':>18} | heading deg p10 p50 p90 | along m p10 p50 p90 | across m p10 p50 p90 | end dist m p50 p90 | max dist m p50 p90 | fell")
env.reset()
obs = venv.get_observations()
C = torch.tensor(CMDS, device=env.device).repeat_interleave(n, 0)
yaw0 = robot.data.heading_w.clone()
p0 = robot.data.root_link_pos_w[:, :2].clone()
fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
far = torch.zeros(env.num_envs, device=env.device)
for _ in range(int(args.seconds / env.step_dt)):
  cmd.vel_command_b[:] = C
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, dones, extras = venv.step(policy(obs))
  fell |= dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
  far = torch.maximum(far, (robot.data.root_link_pos_w[:, :2] - p0).norm(dim=1))
d_all = robot.data.heading_w - yaw0
d_all = torch.rad2deg(torch.atan2(torch.sin(d_all), torch.cos(d_all)))
rel_all = robot.data.root_link_pos_w[:, :2] - p0
for i, c in enumerate(CMDS):
  g = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  g[i * n : (i + 1) * n] = True
  ok = g & ~fell
  d = d_all[ok]
  rel = rel_all[ok]
  ax = torch.stack([torch.cos(yaw0[ok]), torch.sin(yaw0[ok])], 1)
  across = -ax[:, 1] * rel[:, 0] + ax[:, 0] * rel[:, 1]
  along = (ax * rel).sum(1)
  print(f"{str(c):>18} | {f3(d)}    | {f3(along)}  | {f3(across)}   | {f2(rel.norm(dim=1))}        | {f2(far[ok])}        | {int((fell & g).sum())}/{n}")
env.close()
