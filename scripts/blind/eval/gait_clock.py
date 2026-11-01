"""Gait clock: do the feet follow the gait clock while the robot moves?

Robots on flat ground (play config) hold one command each of the acceptance
matrix (accept_matrix.py) for 8 s; over the last 4 s, per command: the share of
foot-steps whose contact state matches the clock of the task's foot_gait reward
(stance while the leg's phase is below its threshold, swing after), touchdowns per
second per foot, and falls. Judged commands: every one for legged robots; for
wheeled robots (foot_gait with stepping_only) those the wheels cannot drive, as
the reward gates them (sideways or turning beyond its limits); forward or back
beyond --max-speed (default 1 m/s) only reported (user 2026-10-23: the fixed 0.6 s
clock is too slow there, the deployed Go2 matched it 55-65% at 1.5 m/s stepping
2.1-2.6 times a second). A judged command fails below --min-match (default 80%;
contact by chance matches ~50%) or when a robot fell. Marching in place is judged by march_eval.py. The wheeled evaluation
tasks have no foot_gait reward: --clock-task names the training task to take it
from (the clock observation is the same).

Usage: python gait_clock.py --task <flat eval task> <checkpoint> [--clock-task <training task>]
  [--min-match 0.8]
"""

import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.velocity import mdp

# The acceptance matrix's commands (accept_matrix.py).
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
p.add_argument("--clock-task", help="task whose foot_gait reward defines the clock (default --task)")
p.add_argument("--min-match", type=float, default=0.8)
p.add_argument("--max-speed", type=float, default=1.0, help="judge |vx| up to this")
p.add_argument("--envs-per-cmd", type=int, default=16)
args = p.parse_args()

n = args.envs_per_cmd
cfg = load_env_cfg(args.task, play=True)
clock_task = args.clock_task or args.task
gait = (cfg if clock_task == args.task else load_env_cfg(clock_task)).rewards.get("foot_gait")
assert gait is not None, f"{clock_task} has no foot_gait reward (no gait clock)"
gp = gait.params
cfg.scene.num_envs = len(CMDS) * n
cfg.episode_length_s = 1000.0
agent_cfg = load_rl_cfg(args.task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = load_runner_cls(args.task)(venv, asdict(agent_cfg), device="cuda:0")
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
dev, dt = env.device, env.step_dt
sensor = env.scene[gp["sensor_name"]]
cmd = env.command_manager.get_term("twist")
offsets = torch.tensor(gp["offset"], device=dev, dtype=torch.float32).view(1, -1)
C = torch.tensor(CMDS, device=dev, dtype=torch.float32).repeat_interleave(n, 0)
steps, start = int(round(8.0 / dt)), int(round(4.0 / dt))
match = torch.zeros(len(C), device=dev)
touchdowns = torch.zeros(len(C), device=dev)
fell = torch.zeros(len(C), dtype=torch.bool, device=dev)
obs = venv.get_observations()
prev = None
for k in range(steps):
  cmd.vel_command_b[:] = C
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, dones, _ = venv.step(policy(obs))
  fell |= dones.bool()
  contact = sensor.data.current_contact_time > 0
  if k >= start:
    # The training clock (speed-dependent with ref_speed, see mdp.gait_clock).
    phase = mdp.gait_clock(env, gp["period"], gp["command_name"], gp.get("ref_speed"), gp.get("foot_xy")).unsqueeze(1)
    stance = (phase + offsets) % 1.0 < gp["threshold"]
    match += (stance == contact).float().mean(dim=1)
    touchdowns += (contact & ~prev).float().mean(dim=1)
  prev = contact
match /= steps - start
touchdowns /= (steps - start) * dt

stepping_only = gp.get("stepping_only", False)
max_vy, max_wz = gp.get("max_lin_vel_y", 0.1), gp.get("max_ang_vel_z", 0.1)
print(f"{args.checkpoint}\n{args.task}: {n} robots per command, last 4 s of 8 s, clock period "
      f"{gp['period']} s" + (f" (shorter above {gp['ref_speed']:.2f} m/s" + (", fastest foot" if gp.get("foot_xy") else "") + ")" if gp.get("ref_speed") else "") + f", offsets {gp['offset']}, stance < {gp['threshold']}; min match {args.min_match:.0%}"
      + (f"; judged only beyond |vy| {max_vy} or |wz| {max_wz} (wheels)" if stepping_only else "")
      + f"; |vx| above {args.max_speed} m/s reported only")
print(f"{'command (vx, vy, wz)':>22} | clock match | touchdowns/s | falls | verdict")
fails = judged = 0
for ci, c in enumerate(CMDS):
  sl = slice(ci * n, (ci + 1) * n)
  ok = ~fell[sl]
  m = match[sl][ok].mean().item() if ok.any() else float("nan")
  td = touchdowns[sl][ok].mean().item() if ok.any() else float("nan")
  judge = (not stepping_only or abs(c[1]) > max_vy or abs(c[2]) > max_wz) and abs(c[0]) <= args.max_speed
  verdict = "-"
  if judge:
    judged += 1
    bad = int((~ok).sum()) > 0 or not (m >= args.min_match)
    fails += bad
    verdict = "FAIL" if bad else "ok"
  print(f"{str(c):>22} | {m * 100:10.0f}% | {td:12.2f} | {int((~ok).sum()):2d}/{n} | {verdict}")
print(f"gait clock: failed {fails} of {judged} judged commands")
env.close()
