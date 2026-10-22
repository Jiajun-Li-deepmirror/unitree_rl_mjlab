"""Hip abduction load standing on stairs: robots pinned to the stair tiles (as in
scripts/go2_blind/eval_stairs.py: 4 step-height bands from 5 to 18 cm x up/down x
narrow/wide treads, play config) walk outwards at --vx, get the zero command once
the base is mid-flight (Chebyshev distance >= 2.2 m) and hold it for --hold-s;
over the last half of the hold, hip abduction RMS torque front / rear (Nm), per
band and direction and overall, over the robots that reached mid-flight and did
not fall, plus the share that fell.

Usage: python stairs_hip.py --task <stairs eval task> <checkpoint> [--hold-s 20]
"""
import argparse
from dataclasses import asdict

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.velocity.blind_env_cfg import STAIR_TERRAIN_NAMES

NUM_ROWS = 4
STEP_RANGE = (0.05, 0.18)

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--vx", type=float, default=0.5)
p.add_argument("--hold-s", type=float, default=20.0)
p.add_argument("--envs-per-tile", type=int, default=4)
args = p.parse_args()

cfg = load_env_cfg(args.task, play=True)
cfg.seed = 0
gen = cfg.scene.terrain.terrain_generator
gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k in STAIR_TERRAIN_NAMES}
for v in gen.sub_terrains.values():
  v.proportion = 1.0
gen.num_rows = NUM_ROWS
gen.num_cols = len(gen.sub_terrains) * 2
cfg.events.pop("randomize_terrain")
cfg.events.pop("push_robot", None)
cfg.episode_length_s = 1000.0
num_envs = gen.num_rows * gen.num_cols * args.envs_per_tile
cfg.scene.num_envs = num_envs
agent = load_rl_cfg(args.task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(args.task)(venv, asdict(agent), device="cuda:0")
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
dev, dt = env.device, env.step_dt
terrain = env.scene.terrain
robot = env.scene["robot"]
cmd = env.command_manager.get_term("twist")
hip_ids, hip_names = robot.find_joints(".*_hip_joint")
act = [list(robot.actuator_names).index(n) for n in hip_names]
front = torch.tensor([n[0] == "F" for n in hip_names], device=dev)

names = list(gen.sub_terrains.keys())
cum = np.cumsum(np.full(len(names), 1.0 / len(names)))
col_name = [names[np.min(np.where(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
tile = torch.arange(num_envs, device=dev) // args.envs_per_tile
levels, types = tile % gen.num_rows, tile // gen.num_rows
env_level = levels.cpu().numpy()
env_dir = np.array(["down" if "down" in col_name[t] else "up" for t in types.cpu().numpy()])
terrain.terrain_levels[:] = levels
terrain.terrain_types[:] = types
terrain.env_origins[:] = terrain.terrain_origins[levels, types]
env.reset()
obs = venv.get_observations()

go = torch.tensor([args.vx, 0.0, 0.0], device=dev)
stopped_at = torch.full((num_envs,), -1, dtype=torch.long, device=dev)
alive = torch.ones(num_envs, dtype=torch.bool, device=dev)
hold = int(args.hold_s / dt)
tau2 = torch.zeros(num_envs, len(hip_names), device=dev)
count = torch.zeros(num_envs, device=dev)
for t in range(int(10.0 / dt) + hold):
  dist = (robot.data.root_link_pos_w[:, :2] - terrain.env_origins[:, :2]).abs().max(dim=1).values
  newly = (stopped_at < 0) & (dist >= 2.2) & alive
  stopped_at[newly] = t
  cmd.vel_command_b[:] = torch.where((stopped_at >= 0).unsqueeze(1), 0.0, go)
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, _, _ = venv.step(policy(obs))
  alive &= ~env.termination_manager.terminated
  late = (stopped_at >= 0) & (t - stopped_at >= hold // 2) & (t - stopped_at < hold) & alive
  tau2 += late.float().unsqueeze(1) * robot.data.actuator_force[:, act].square()
  count += late.float()
  if (((stopped_at >= 0) & (t - stopped_at >= hold)) | ~alive).all():
    break

ok = (count > 0) & alive
rms = (tau2 / count.clamp(min=1).unsqueeze(1)).sqrt()
fr = lambda m: (float(rms[m][:, front].mean()), float(rms[m][:, ~front].mean())) if m.any() else (float("nan"),) * 2


def band(level):
  lo, hi = STEP_RANGE
  return f"{100 * (lo + (hi - lo) * level / NUM_ROWS):.0f}-{100 * (lo + (hi - lo) * (level + 1) / NUM_ROWS):.0f}cm"


print(f"{args.checkpoint}\n{args.task}: {num_envs // (NUM_ROWS * 2)} robots per (band, direction); zero command held {args.hold_s:.0f} s mid-flight, hip RMS over its last half")
print(f"{'band':>9} {'dir':>4} | hip rms front / rear Nm | fell")
okn = torch.as_tensor(ok, device=dev)
for lvl in range(NUM_ROWS):
  for d in ("up", "down"):
    m = torch.as_tensor((env_level == lvl) & (env_dir == d), device=dev)
    f, r = fr(m & okn)
    print(f"{band(lvl):>9} {d:>4} | {f:8.2f} / {r:6.2f}         | {float((~alive[m]).float().mean()):.0%}")
for d in ("up", "down"):
  m = torch.as_tensor(env_dir == d, device=dev)
  f, r = fr(m & okn)
  print(f"all {d:>4}: hip rms front {f:.2f} rear {r:.2f} Nm, fell {float((~alive[m]).float().mean()):.0%}")
f, r = fr(okn)
print(f"overall: hip rms front {f:.2f} rear {r:.2f} Nm, fell {float((~alive).float().mean()):.0%}")
env.close()
