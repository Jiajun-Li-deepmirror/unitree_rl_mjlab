"""Cumulative drift turning in place left and right in turn (go2-depth 2026-11-06, the user: "验收里添加左右交替转多次后的
累积位移测试").

Play config of a flat task (no pushes, no observation noise), N robots per sequence, all sequences in one simulation.
Each sequence turns in place at +-wz for T s per turn, starting left, TURNS turns (left and right alternating, so the
heading comes back), then stands for 2 s at the zero command. Per sequence, over the robots that did not fall: the
base's distance from its start point at the end (after the stand) and the largest during the run (m, p50 p90), the
heading at the end against the start (|deg|, p50 p90), falls. drift_all.py's single spins measure one direction for
10 s; this one adds the reversals (each reversal is a stop and a start, where a stepping robot shuffles).

--yaw-accel A (rad/s^2): the yaw command slews at most A per second, through the reversals too (the B2 / B2W / As2-W
controllers' max_yaw_accel 3.0), as drift_all.py.

Usage: python spin_alt.py --task <flat task> <checkpoint> [--num-envs 256] [--turns 8] [--yaw-accel A]
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
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--turns", type=int, default=8)
p.add_argument("--yaw-accel", type=float, default=0.0)
p.add_argument("--stand", type=float, default=2.0)
args = p.parse_args()
SEQS = [(0.5, 3.0), (1.0, 2.0), (1.5, 1.5)]  # (wz rad/s, seconds per turn)

cfg = load_env_cfg(args.task, play=True)
n = args.num_envs
cfg.scene.num_envs = n * len(SEQS)
T = max(w for _, w in SEQS) * args.turns + args.stand
cfg.episode_length_s = T + 5.0
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
dev = env.device
q = torch.tensor((0.5, 0.9), device=dev)
f2 = lambda x: " ".join(f"{v:5.2f}" for v in x.quantile(q).tolist()) if len(x) else "  nan   nan"
ramp = f", yaw command slewed at {args.yaw_accel:g} rad/s^2" if args.yaw_accel > 0 else ""
print(f"{args.checkpoint}\n{args.task}: {n} robots per sequence, {args.turns} turns left / right in turn from a standstill, "
      f"then {args.stand:g} s standing{ramp}")
print(f"{'wz x s per turn':>16} | end dist m p50 p90 | max dist m p50 p90 | heading |deg| p50 p90 | fell")
env.reset()
obs = venv.get_observations()
wz = torch.tensor([w for w, _ in SEQS], device=dev).repeat_interleave(n)
per = torch.tensor([s for _, s in SEQS], device=dev).repeat_interleave(n)
end_t = per * args.turns
yaw0 = robot.data.heading_w.clone()
p0 = robot.data.root_link_pos_w[:, :2].clone()
fell = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
far = torch.zeros(env.num_envs, device=dev)
cur = torch.zeros(env.num_envs, device=dev)
steps = int(round(T / env.step_dt))
for k in range(steps):
  t = k * env.step_dt
  turn = torch.floor(t / per)
  target = torch.where(t < end_t, torch.where(turn % 2 == 0, wz, -wz), torch.zeros_like(wz))
  if args.yaw_accel > 0:
    a = args.yaw_accel * env.step_dt
    cur = cur + (target - cur).clamp(-a, a)
  else:
    cur = target
  cmd.vel_command_b[:] = 0.0
  cmd.vel_command_b[:, 2] = cur
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, dones, extras = venv.step(policy(obs))
  fell |= dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
  far = torch.maximum(far, (robot.data.root_link_pos_w[:, :2] - p0).norm(dim=1))
end = (robot.data.root_link_pos_w[:, :2] - p0).norm(dim=1)
dh = robot.data.heading_w - yaw0
dh = torch.rad2deg(torch.atan2(torch.sin(dh), torch.cos(dh))).abs()
for i, (w, s) in enumerate(SEQS):
  g = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
  g[i * n : (i + 1) * n] = True
  ok = g & ~fell
  print(f"{f'+-{w:g} x {s:g}':>16} | {f2(end[ok])}        | {f2(far[ok])}        | {f2(dh[ok])}             | "
        f"{int((fell & g).sum())}/{n}")
env.close()
