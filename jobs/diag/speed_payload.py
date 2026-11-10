# jobs/diag/speed_payload.py (go2-depth 11-06): flat speed ratio and wheels-down share binned by payload.
"""go2-home (11-06): our Go2W blind overspeeds small commands; bin by payload to see whether the heavy robots
are the steppers. Flat play task, the pilot's point payload injected (every robot uniform 0..MAX kg, same
position ranges as training), each robot holds one command for T s from a standstill; over the last 4 s per robot:
the speed along each commanded axis / the command (accept_matrix's ratio) and the share of steps with all four
feet / wheels in contact. Printed per command and payload bin.

Usage: python speed_payload.py --task <flat task> <checkpoint> [--num-envs 128] [--seconds 8] [--max 12]"""
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
p.add_argument("--seconds", type=float, default=8.0)
p.add_argument("--max", type=float, default=12.0)
args = p.parse_args()
CMDS = [(0.2, 0.0, 0.0), (0.3, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0),
        (0.0, 0.15, 0.0), (0.0, 0.3, 0.0), (0.0, -0.3, 0.0), (0.0, 0.6, 0.0)]  # fmt: skip
BINS = [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.01)]

cfg = load_env_cfg(args.task, play=True)
n = args.num_envs
cfg.scene.num_envs = n * len(CMDS)
cfg.episode_length_s = args.seconds + 5.0
cfg.events.pop("push_robot", None)
cfg.events.pop("base_com", None)
cfg.events["base_payload"] = EventTermCfg(
  mode="reset",
  func=mdp.base_payload_point,
  params={"asset_cfg": SceneEntityCfg("robot"), "light_range": (0.0, 0.0), "heavy_max": args.max, "heavy_share": 1.0,
          "pos_ranges": ((-0.15, 0.15), (-0.08, 0.08), (0.0, 0.20)), "base_com_range": 0.05, "ramp_steps": 0},
)
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
sensor = env.scene["feet_ground_contact"]
env.reset()
obs = venv.get_observations()
kg = env.payload_kg.clone()
C = torch.tensor(CMDS, device=env.device).repeat_interleave(n, 0)
steps = int(args.seconds / env.step_dt)
t0 = steps - int(4.0 / env.step_dt)
vsum = torch.zeros(env.num_envs, 2, device=env.device)
down = torch.zeros(env.num_envs, device=env.device)
fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
for k in range(steps):
  cmd.vel_command_b[:] = C
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, dones, extras = venv.step(policy(obs))
  fell |= dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
  if k >= t0:
    vsum += robot.data.root_link_lin_vel_b[:, :2]
    down += (sensor.data.found > 0).all(dim=1).float()
m = steps - t0
v = vsum / m
print(f"{args.checkpoint}\n{args.task}: {n} robots per command, payload uniform 0..{args.max:g} kg, last 4 s of {args.seconds:g} s")
print(f"{'command':>18} | " + " | ".join(f"{lo:.0f}-{hi:.0f} kg: ratio  4down  n" for lo, hi in BINS))
for i, c in enumerate(CMDS):
  sl = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  sl[i * n:(i + 1) * n] = True
  ax = 0 if c[0] != 0.0 else 1
  ratio = v[:, ax] / c[ax]
  cells = []
  for lo, hi in BINS:
    g = sl & ~fell & (kg >= lo) & (kg < hi)
    if int(g.sum()) == 0:
      cells.append("      -      -   0")
      continue
    cells.append(f"{100 * ratio[g].mean():6.0f}% {100 * down[g].mean() / m:5.0f}% {int(g.sum()):3d}")
  print(f"{str(c):>18} | " + " | ".join(cells) + f" | fell {int((fell & sl).sum())}")
env.close()
