"""Stairs evaluation of a blind Go2 policy on the training stair tiles.

Robots are pinned to the stair tiles of Unitree-Go2-Blind (4 step-height bands from
5 to 18 cm x up/down x narrow/wide treads, play config). Two tests:
  traverse : forward command from the tile center outwards; success = the base gets
             past the last step (Chebyshev distance >= 3.6 m) without terminating.
  stop     : walk outwards, zero the command once the base is mid-flight (distance
             >= 2.2 m) and hold it for 4 s; success = no fall and < 5 cm drift over
             the last 2 s.

  python scripts/go2_blind/eval_stairs.py logs/rsl_rl/go2_blind/<run>/model_N.pt --vx 0.5
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


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--task", default="Unitree-Go2-Blind", help="Blind task whose play config matches the policy.")
  parser.add_argument("checkpoint")
  parser.add_argument("--vx", type=float, default=0.5)
  parser.add_argument("--wz", type=float, default=0.0)
  parser.add_argument("--envs-per-tile", type=int, default=6)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.seed = args.seed
  gen = env_cfg.scene.terrain.terrain_generator
  gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k in STAIR_TERRAIN_NAMES}
  for v in gen.sub_terrains.values():
    v.proportion = 1.0
  gen.num_rows = NUM_ROWS
  gen.num_cols = len(gen.sub_terrains) * 2
  env_cfg.events.pop("randomize_terrain")
  num_envs = gen.num_rows * gen.num_cols * args.envs_per_tile
  env_cfg.scene.num_envs = num_envs

  agent_cfg = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(args.task)(venv, asdict(agent_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")

  dev, dt = env.device, env.step_dt
  terrain = env.scene.terrain
  robot = env.scene["robot"]
  feet_sensor = env.scene["feet_ground_contact"]
  leg_sensor = env.scene["leg_ground_touch"]
  cmd_term = env.command_manager.get_term("twist")

  # Column -> sub-terrain name (same rule as the terrain generator).
  names = list(gen.sub_terrains.keys())
  cum = np.cumsum(np.full(len(names), 1.0 / len(names)))
  col_name = [names[np.min(np.where(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]

  tile = torch.arange(num_envs, device=dev) // args.envs_per_tile
  levels = tile % gen.num_rows
  types = tile // gen.num_rows
  env_level = levels.cpu().numpy()
  env_dir = np.array(["down" if "down" in col_name[t] else "up" for t in types.cpu().numpy()])
  env_tread = np.array(["narrow" if "narrow" in col_name[t] else "wide" for t in types.cpu().numpy()])

  def pin_and_reset():
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = types
    terrain.env_origins[:] = terrain.terrain_origins[levels, types]
    env.reset()
    return venv.get_observations()

  def set_cmd(c, ids=None):
    t = torch.tensor(c, device=dev, dtype=torch.float32)
    if ids is None:
      cmd_term.vel_command_b[:] = t
    else:
      cmd_term.vel_command_b[ids] = t
    cmd_term.time_left[:] = 1e9

  def cheb_dist():
    d = (robot.data.root_link_pos_w[:, :2] - terrain.env_origins[:, :2]).abs()
    return d.max(dim=1).values

  def band(level):
    lo, hi = STEP_RANGE
    return f"{100 * (lo + (hi - lo) * level / NUM_ROWS):.0f}-{100 * (lo + (hi - lo) * (level + 1) / NUM_ROWS):.0f}cm"

  # ---- Traverse ----
  obs = pin_and_reset()
  set_cmd((args.vx, 0.0, args.wz))
  alive = torch.ones(num_envs, dtype=torch.bool, device=dev)
  success = torch.zeros_like(alive)
  leg_hits = torch.zeros(num_envs, device=dev)
  steps_alive = torch.zeros(num_envs, device=dev)
  max_dist = torch.zeros(num_envs, device=dev)
  for _ in range(int(round(3.6 / max(abs(args.vx), 0.1) * 2.0 / dt))):
    with torch.no_grad():
      obs, _, _, _ = venv.step(policy(obs))
    term = env.termination_manager.terminated
    active = alive & ~success
    leg_hits += (active & (leg_sensor.data.found > 0).flatten(1).any(1)).float()
    steps_alive += active.float()
    alive &= ~(term & active)
    max_dist = torch.where(alive, torch.maximum(max_dist, cheb_dist()), max_dist)
    success |= alive & (cheb_dist() >= 3.6)
  trav_ok = success.cpu().numpy()
  trav_fell = (~alive).cpu().numpy()
  leg_frac = (leg_hits / steps_alive.clamp(min=1)).cpu().numpy()
  max_dist = max_dist.cpu().numpy()

  # ---- Stop mid-flight ----
  obs = pin_and_reset()
  set_cmd((args.vx, 0.0, args.wz))
  stopped_at = torch.full((num_envs,), -1, dtype=torch.long, device=dev)
  alive = torch.ones(num_envs, dtype=torch.bool, device=dev)
  pos_hist, contact_hist = [], []
  hold = int(4.0 / dt)
  for t in range(int(10.0 / dt) + hold):
    newly = (stopped_at < 0) & (cheb_dist() >= 2.2) & alive
    if newly.any():
      set_cmd((0.0, 0.0, 0.0), newly)
      stopped_at[newly] = t
    with torch.no_grad():
      obs, _, _, _ = venv.step(policy(obs))
    term = env.termination_manager.terminated
    in_hold = (stopped_at >= 0) & (t - stopped_at < hold)
    alive &= ~(term & ((stopped_at < 0) | in_hold))
    pos_hist.append(robot.data.root_link_pos_w[:, :2].clone())
    contact_hist.append((feet_sensor.data.found > 0).flatten(1).all(1).clone())
    if ((stopped_at >= 0) & (t - stopped_at >= hold) | ~alive).all():
      break
  pos_hist = torch.stack(pos_hist).cpu().numpy()
  contact_hist = torch.stack(contact_hist).cpu().numpy()
  stopped_np, alive_np = stopped_at.cpu().numpy(), alive.cpu().numpy()
  drift = np.full(num_envs, np.nan)
  feet_down = np.full(num_envs, np.nan)
  for b in range(num_envs):
    s = stopped_np[b]
    if s >= 0 and alive_np[b] and s + hold <= len(pos_hist):
      a, e = s + hold // 2, s + hold - 1
      drift[b] = np.linalg.norm(pos_hist[e, b] - pos_hist[a, b])
      feet_down[b] = contact_hist[a : e + 1, b].mean()
  reached = stopped_np >= 0
  stop_ok = reached & alive_np & (np.nan_to_num(drift, nan=1.0) < 0.05)

  # ---- Report ----
  print(f"\ncheckpoint: {args.checkpoint}\ncommand vx={args.vx} wz={args.wz}, "
        f"{num_envs // (NUM_ROWS * 2)} robots per (band, direction)")
  print(f"{'band':>9} {'dir':>4} | {'traverse ok':>11} {'fell':>5} {'leg touch':>9} {'reach m':>7} | "
        f"{'stop ok':>7} {'fell':>5} {'drift cm':>8} {'4 feet':>6}")
  for lvl in range(NUM_ROWS):
    for d in ("up", "down"):
      m = (env_level == lvl) & (env_dir == d)
      mr = m & reached
      drift_cm = 100 * np.nanmean(drift[mr]) if np.isfinite(drift[mr]).any() else float("nan")
      feet_frac = np.nanmean(feet_down[mr]) if np.isfinite(feet_down[mr]).any() else float("nan")
      print(f"{band(lvl):>9} {d:>4} | {trav_ok[m].mean():11.0%} {trav_fell[m].mean():5.0%} {leg_frac[m].mean():9.1%} "
            f"{np.median(max_dist[m]):7.2f} | {stop_ok[m].mean():7.0%} {(~alive_np[m]).mean():5.0%} {drift_cm:8.1f} {feet_frac:6.2f}")
  for tread in ("narrow", "wide"):
    m = env_tread == tread
    print(f"tread {tread:>6}: traverse ok {trav_ok[m].mean():.0%}, stop ok {stop_ok[m].mean():.0%}")
  print(f"overall: traverse ok {trav_ok.mean():.0%}, fell {trav_fell.mean():.0%}; stop ok {stop_ok.mean():.0%}")
  env.close()


if __name__ == "__main__":
  main()
