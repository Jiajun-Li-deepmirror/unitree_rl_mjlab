"""stairs_term.py TASK CKPT: eval_stairs.py's traverse, reporting which termination term ends each robot and where
(go2-depth 2026-11-05 diag: the Go2W blind mixed avg6500 fell 100% on every up band at ~0.97 m)."""
import sys
from dataclasses import asdict
import numpy as np
import torch
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.velocity.blind_env_cfg import STAIR_TERRAIN_NAMES

task, ck = sys.argv[1:3]
env_cfg = load_env_cfg(task, play=True)
gen = env_cfg.scene.terrain.terrain_generator
print("stair sub-terrains:", {k: type(v).__name__ for k, v in gen.sub_terrains.items() if k in STAIR_TERRAIN_NAMES})
gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k in STAIR_TERRAIN_NAMES}
for v in gen.sub_terrains.values():
  v.proportion = 1.0
gen.num_rows, gen.num_cols = 4, len(gen.sub_terrains) * 2
env_cfg.events.pop("randomize_terrain")
E = 6
n = gen.num_rows * gen.num_cols * E
env_cfg.scene.num_envs = n
print("terminations:", list(env_cfg.terminations.keys()))
print("events:", list(env_cfg.events.keys()))
agent_cfg = load_rl_cfg(task)
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = load_runner_cls(task)(venv, asdict(agent_cfg), device="cuda:0")
runner.load(ck, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
terrain, robot, cmd = env.scene.terrain, env.scene["robot"], env.command_manager.get_term("twist")
names = list(gen.sub_terrains.keys())
cum = np.cumsum(np.full(len(names), 1.0 / len(names)))
col_name = [names[np.min(np.where(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
tile = torch.arange(n, device="cuda:0") // E
levels, types = tile % gen.num_rows, tile // gen.num_rows
tn = [col_name[t] for t in types.cpu().numpy()]
terrain.terrain_levels[:] = levels
terrain.terrain_types[:] = types
terrain.env_origins[:] = terrain.terrain_origins[levels, types]
env.reset()
obs = venv.get_observations()
tm = env.termination_manager
alive = torch.ones(n, dtype=torch.bool, device="cuda:0")
first = {}
for step in range(int(3.6 / 0.5 * 2.0 / env.step_dt)):
  cmd.vel_command_b[:] = torch.tensor((0.5, 0.0, 0.0), device="cuda:0")
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, _, _ = venv.step(policy(obs))
  term = tm.terminated
  d = (robot.data.root_link_pos_w[:, :2] - terrain.env_origins[:, :2]).abs().max(1).values
  new = term & alive
  for i in new.nonzero().flatten().tolist():
    which = [k for k in tm.active_terms if bool(tm.get_term(k)[i])]
    first[i] = (step, which, float(d[i]), float(robot.data.root_link_pos_w[i, 2] - terrain.env_origins[i, 2]))
  alive &= ~term
  if step == 25:
    print("cmd after 25 steps (first 3 robots):", cmd.vel_command_b[:3].tolist())
from collections import Counter
for kind in sorted(set(tn)):
  ids = [i for i in range(n) if tn[i] == kind]
  dead = [first[i] for i in ids if i in first]
  c = Counter(tuple(w) for _, w, _, _ in dead)
  st = [s for s, _, _, _ in dead]
  med = lambda xs: f"{np.median(xs):.2f}" if xs else "nan"
  print(f"{kind:>22}: {len(dead)}/{len(ids)} ended; terms {dict(c)}; step p50 {med(st)}; "
        f"dist p50 {med([x for _, _, x, _ in dead])}; z p50 {med([z for _, _, _, z in dead])}")
