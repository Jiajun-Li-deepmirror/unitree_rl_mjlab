"""slow_wobble.py --task <flat task> <checkpoint> (go2-depth 2026-11-07, the user: Go2W "慢速前进的时候腿会前后晃动"): N robots
drive straight at vx 0.1-1.0 for 8 s on flat ground (play config, no pushes); from 3 s on, the std of each thigh / calf
angle (deg, the swing about the mean pose) and the mean |joint speed|."""
import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
import src.tasks.velocity.mdp as mdp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--num-envs", type=int, default=128)
args = p.parse_args()

cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 30.0
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
names = robot.joint_names
thigh = [i for i, n in enumerate(names) if "thigh" in n]
calf = [i for i, n in enumerate(names) if "calf" in n]
wheel = [i for i, n in enumerate(names) if "wheel" in n]
print(args.checkpoint)
for v in (0.1, 0.2, 0.3, 0.5, 1.0):
  env.reset()
  obs = venv.get_observations()
  qs, vs, ws = [], [], []
  for k in range(int(8.0 / env.step_dt)):
    cmd.vel_command_b[:] = 0.0
    cmd.vel_command_b[:, 0] = v
    cmd.time_left[:] = 1e9
    if hasattr(cmd, "march"):
      cmd.march[:] = False
    with torch.no_grad():
      obs, _, _, _ = venv.step(policy(obs))
    if k * env.step_dt >= 3.0:
      qs.append(robot.data.joint_pos.clone())
      vs.append(robot.data.root_link_lin_vel_b[:, 0].clone())
      ws.append(robot.data.joint_vel[:, wheel].clone())
  q = torch.rad2deg(torch.stack(qs))   # T x N x J
  sd = q.std(0)                         # N x J
  qd = torch.rad2deg(torch.stack(qs).diff(dim=0) / env.step_dt).abs().mean(0)
  leg = torch.cat([sd[:, thigh], sd[:, calf]], 1).mean(1)
  wsd = torch.stack(ws).std(0).mean(1) if wheel else torch.zeros(1, device=env.device)
  qq = torch.tensor((0.5, 0.9, 1.0), device=env.device)
  print(f"vx {v:.1f}: per robot leg angle std deg p50 p90 max {' '.join(f'{x:.2f}' for x in leg.quantile(qq).tolist())} | "
        f"wheel speed std rad/s p50 p90 max {' '.join(f'{x:.2f}' for x in wsd.quantile(qq).tolist())}")
  print(f"vx {v:.1f}: actual {torch.stack(vs).mean():.2f} m/s | joint angle std deg (mean over robots) thigh "
        f"{sd[:, thigh].mean():.2f} calf {sd[:, calf].mean():.2f} | mean |joint speed| deg/s thigh {qd[:, thigh].mean():.1f} "
        f"calf {qd[:, calf].mean():.1f}")
env.close()
