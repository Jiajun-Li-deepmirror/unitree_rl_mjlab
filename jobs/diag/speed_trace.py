# jobs/diag/speed_trace.py (go2-depth 11-06): base speed against wheel rim speed for forward commands.
"""Go2W blind g12 10500 drives (0.2, 0, 0) at 133% while (0.3, 0, 0) is 115% and go2-home's 11300 101%: are the
wheels turning that fast (the policy's wheel command) or do the legs push the base past them? Flat play task as the
acceptance (its own DR), N robots per command, T s from a standstill. Per command over the last 4 s: base vx /
command, wheel rim speed (mean wheel joint velocity x radius) / command, base / rim, mean |leg joint velocity|,
and a 0.5 s trace of base and rim speed for the first command.

Usage: python speed_trace.py --task <flat task> <checkpoint> [--num-envs 64] [--seconds 8] [--radius 0.086]"""
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
p.add_argument("--seconds", type=float, default=8.0)
p.add_argument("--radius", type=float, default=0.086)
args = p.parse_args()
CMDS = [(0.2, 0.0, 0.0), (0.3, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0)]

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
names = list(robot.joint_names)
wheel = [i for i, j in enumerate(names) if "foot" in j or "wheel" in j]
legs = [i for i in range(len(names)) if i not in wheel]
print("wheel joints:", [names[i] for i in wheel])
cmd = env.command_manager.get_term("twist")
env.reset()
obs = venv.get_observations()
C = torch.tensor(CMDS, device=env.device).repeat_interleave(n, 0)
steps = int(args.seconds / env.step_dt)
t0 = steps - int(4.0 / env.step_dt)
base = torch.zeros(env.num_envs, device=env.device)
rim = torch.zeros(env.num_envs, device=env.device)
legv = torch.zeros(env.num_envs, device=env.device)
tr_b, tr_r, acc = [], [], [0.0, 0.0, 0]
for k in range(steps):
  cmd.vel_command_b[:] = C
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, _, _ = venv.step(policy(obs))
  vx = robot.data.root_link_lin_vel_b[:, 0]
  wv = robot.data.joint_vel[:, wheel].mean(dim=1) * args.radius
  if k >= t0:
    base += vx
    rim += wv
    legv += robot.data.joint_vel[:, legs].abs().mean(dim=1)
  acc[0] += float(vx[:n].mean()); acc[1] += float(wv[:n].mean()); acc[2] += 1
  if acc[2] == int(0.5 / env.step_dt):
    tr_b.append(acc[0] / acc[2]); tr_r.append(acc[1] / acc[2]); acc = [0.0, 0.0, 0]
m = steps - t0
print(f"{args.checkpoint}\n{args.task}: {n} robots per command, last 4 s of {args.seconds:g} s, wheel radius {args.radius}")
print(f"{'command':>18} | base/cmd | rim/cmd | base/rim | |leg qd| rad/s")
for i, c in enumerate(CMDS):
  sl = slice(i * n, (i + 1) * n)
  b, r = base[sl].mean() / m, rim[sl].mean() / m
  print(f"{str(c):>18} | {100 * b / c[0]:7.0f}% | {100 * r / c[0]:6.0f}% | {b / r:8.2f} | {float(legv[sl].mean() / m):.3f}")
print(f"trace {CMDS[0]} every 0.5 s, base m/s: " + " ".join(f"{x:.3f}" for x in tr_b))
print(f"trace {CMDS[0]} every 0.5 s, rim  m/s: " + " ".join(f"{x:.3f}" for x in tr_r))
env.close()
