"""Zero command for 30 s, standing (gait clock stopped) and marching in place (clock
running, the controller's march flag): over the last 20 s per mode, touchdowns per
second per foot, share of time each foot is in the air, base drift (cm) and yaw
change (deg), hip angle outward from the default (deg) and hip abduction RMS
torque (Nm) front / rear, and falls."""
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
p.add_argument("--envs", type=int, default=32)
args = p.parse_args()

for march in (False, True):
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.envs
  cfg.episode_length_s = 1000.0
  cfg.events.pop("push_robot", None)
  agent_cfg = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(args.task)(venv, asdict(agent_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("twist")
  sensor = env.scene["feet_ground_contact"]
  hip_ids, hip_names = robot.find_joints(".*_hip_joint")
  side = torch.tensor([1.0 if n[1] == "L" else -1.0 for n in hip_names], device=env.device)
  front = torch.tensor([n[0] == "F" for n in hip_names], device=env.device)
  act = [list(robot.actuator_names).index(n) for n in hip_names]
  default = robot.data.default_joint_pos[:, hip_ids]
  obs = venv.get_observations()
  fell = torch.zeros(args.envs, dtype=torch.bool, device=env.device)
  start = int(10.0 / env.step_dt)
  steps = int(30.0 / env.step_dt)
  touch = air = ang = tau2 = 0
  n = 0
  for k in range(steps):
    cmd.vel_command_b[:] = 0.0
    cmd.time_left[:] = 1e9
    cmd.march[:] = march
    with torch.no_grad():
      obs, _, dones, _ = venv.step(policy(obs))
    fell |= dones.bool()
    if k == start:
      p0 = robot.data.root_link_pos_w[:, :2].clone()
      y0 = robot.data.heading_w.clone()
    if k >= start:
      contact = sensor.data.found > 0
      touch = touch + sensor.compute_first_contact(dt=env.step_dt).float().mean(0)
      air = air + (~contact).float().mean(0)
      ang = ang + ((robot.data.joint_pos[:, hip_ids] - default) * side).mean(0)
      tau2 = tau2 + robot.data.actuator_force[:, act].square().mean(0)
      n += 1
  T = n * env.step_dt
  drift = (robot.data.root_link_pos_w[:, :2] - p0).norm(dim=1).mean() * 100
  yaw = torch.rad2deg(torch.atan2(torch.sin(robot.data.heading_w - y0), torch.cos(robot.data.heading_w - y0))).abs().mean()
  ang = torch.rad2deg(ang / n)
  rms = (tau2 / n).sqrt()
  f = lambda x: f"{float(x[front].mean()):+.2f} / {float(x[~front].mean()):+.2f}"
  print(f"{'march ' if march else 'stand '}| touchdowns/s per foot {' '.join(f'{v:.2f}' for v in (touch / T).tolist())} "
        f"| air % {' '.join(f'{100 * v:.0f}' for v in (air / n).tolist())} | drift {drift:.1f} cm, yaw {yaw:.1f} deg "
        f"| hip outward {f(ang)} deg | rms tau {f(rms)} Nm | fell {int(fell.sum())}/{args.envs}")
  env.close()
