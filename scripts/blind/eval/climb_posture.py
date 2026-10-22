"""Climbing posture on map3's L0 flight (up, 0.5 m/s, operator off) vs. standing on
flat ground: per leg the hip / thigh / calf angles relative to the default pose,
the foot (wheel) position relative to the base in the base frame (sideways y,
fore-aft x), base pitch and roll, and the speed ratio. Mean over 16 robots and the
middle of the flight (2.5-7.5 s after the start of walking).

Usage: python climb_posture.py --task <stairs eval task> <checkpoint> [--num-envs 16]
"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("scripts/go2_blind").resolve()))
import eval_map3  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--num-envs", type=int, default=16)
args = p.parse_args()

boxes = eval_map3.load_map3_boxes()
scen = next(s for s in eval_map3.SCENARIOS if s.name == "L0_up")
env = eval_map3.ManagerBasedRlEnv(cfg=eval_map3.make_env_cfg(scen, boxes, args.num_envs, 0, args.task), device="cuda:0")
venv, policy = eval_map3.load_policy(env, args.checkpoint, args.task)
env.reset()
obs = venv.get_observations()
robot = env.scene["robot"]
cmd = env.command_manager.get_term("twist")
names = list(robot.joint_names)
legs = [i for i, n in enumerate(names) if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]
sites, site_names = robot.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
default = robot.data.default_joint_pos[0]
dt = env.step_dt


def frame_stats(acc):
  q = robot.data.joint_pos[:, legs] - default[legs]
  quat = robot.data.root_link_quat_w
  w, x, y, z = quat.unbind(1)
  pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
  roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  rel = robot.data.site_pos_w[:, sites] - robot.data.root_link_pos_w[:, None]
  # into the base frame (yaw only, the posture is about the legs)
  yaw = robot.data.heading_w
  c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
  fx = c * rel[..., 0] + s * rel[..., 1]
  fy = -s * rel[..., 0] + c * rel[..., 1]
  for k, v in (("q", q), ("pitch", pitch), ("roll", roll), ("fx", fx), ("fy", fy), ("fz", rel[..., 2]),
               ("vx", robot.data.root_link_lin_vel_b[:, 0])):
    acc[k] = acc.get(k, 0) + v.mean(0)
  acc["n"] = acc.get("n", 0) + 1


def report(label, acc):
  n = acc["n"]
  q = (acc["q"] / n).tolist()
  print(f"{label}: pitch {math.degrees(acc['pitch'] / n):+.1f} deg, roll {math.degrees(acc['roll'] / n):+.1f} deg, vx {float(acc['vx'] / n):.2f} m/s")
  leg_names = [names[i] for i in legs]
  for leg in ("FL", "FR", "RL", "RR"):
    idx = [j for j, nm in enumerate(leg_names) if nm.startswith(leg)]
    k = site_names.index(leg) if leg in site_names else [i for i, s in enumerate(site_names) if s.startswith(leg)][0]
    print(f"  {leg}: hip/thigh/calf - default {', '.join(f'{math.degrees(q[j]):+5.1f}' for j in idx)} deg | "
          f"foot x {float(acc['fx'][k] / n):+.2f} y {float(acc['fy'][k] / n):+.2f} z {float(acc['fz'][k] / n):+.2f} m")


# Standing still first (1 s), then walking up.
stand, climb = {}, {}
for k in range(int(9.0 / dt)):
  t = k * dt
  cmd.vel_command_b[:] = torch.tensor([0.0 if t < 1.0 else 0.5, 0.0, 0.0], device=env.device)
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, _, _ = venv.step(policy(obs))
  if 0.5 <= t < 1.0:
    frame_stats(stand)
  if 3.5 <= t < 8.5:
    frame_stats(climb)
print(args.checkpoint)
report("standing (flat landing)", stand)
report("climbing L0 (0.5 m/s)", climb)
env.close()
