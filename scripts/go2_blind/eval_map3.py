"""Evaluate a blind Go2 policy on the tinynav-sim map3 stairwell.

map3 (src/assets/terrains/map3: 123 collision boxes plus guard rails) is loaded as
a single terrain tile; it is never part of the training terrain. Each scenario
spawns all robots at one map3 spot with a fixed heading and runs a fixed command
script on the Unitree-Go2-Blind play config. Stair scenarios count a robot as
`reached` when it gains (climb) or loses (descend) the target height before any
termination; `reset_before_goal` is the share terminated before that. With
--operator (default) the command also holds the flight heading and steers back
to the flight centerline, like an operator with a joystick.

--time-scale stretches the moving segments of the stair scenarios: at 1.0 a
policy must track the commanded speed to finish in time, at 1.6 slower robots
can still finish.

  python scripts/go2_blind/eval_map3.py logs/rsl_rl/go2_blind/<run>/model_N.pt --num-envs 512
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)
from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

TASK = "Unitree-Go2-Blind"
MAP3_DIR = Path(__file__).resolve().parents[2] / "src" / "assets" / "terrains" / "map3"
# Shift map3 world coordinates into a positive tile.
MAP3_OFFSET = np.array([13.0, 1.0, 0.0])
MAP3_TILE_SIZE = (14.0, 10.0)
_BOX_RE = re.compile(r'<geom name="([^"]+)"\s+type="box"\s+[^>]*?size="([^"]+)"\s+pos="([^"]+)"')


@dataclass
class Scenario:
  name: str
  spawn: tuple[float, float, float]
  """map3 world position on the walking surface (x, y, top z)."""
  yaw_deg: float
  segments: list[tuple[float, float, float, float]]
  """(duration_s, vx, vy, wz) command script."""
  success: str
  """'climb:<dz>' (gain at least dz), 'descend:<dz>' (lose at least dz) or
  'track' (no fall, follow commands)."""


SCENARIOS = [
  Scenario("L0_up", (-10.34, 5.75, 0.431), -90.0, [(1.0, 0, 0, 0), (9.0, 0.5, 0, 0)], "climb:2.0"),
  Scenario("L0_down", (-10.34, 1.05, 2.659), 90.0, [(1.0, 0, 0, 0), (9.0, 0.4, 0, 0)], "descend:2.0"),
  Scenario("R1_up", (-8.80, 1.05, 2.659), 90.0, [(1.0, 0, 0, 0), (9.0, 0.5, 0, 0)], "climb:1.7"),
  Scenario("R2b_down", (-8.75, 5.70, 11.802), -90.0, [(1.0, 0, 0, 0), (7.0, 0.4, 0, 0)], "descend:1.0"),
  Scenario(
    "L0_stop_midway",
    (-10.34, 5.75, 0.431),
    -90.0,
    [(1.0, 0, 0, 0), (4.0, 0.5, 0, 0), (4.0, 0, 0, 0), (5.0, 0.5, 0, 0)],
    "climb:2.0",
  ),
  Scenario(
    "F1_omni",
    (-5.30, 3.45, 0.0),
    180.0,
    [
      (1.0, 0, 0, 0),
      (2.0, 0.6, 0, 0),
      (2.0, -0.4, 0, 0),
      (2.0, 0, 0.4, 0),
      (2.0, 0, -0.4, 0),
      (2.0, 0, 0, 1.0),
      (2.0, 0, 0, -1.0),
      (2.0, 0.5, 0.2, 0.5),
      (2.0, 0, 0, 0),
    ],
    "track",
  ),
]


def load_map3_boxes(map3_dir: Path = MAP3_DIR) -> list[tuple[str, np.ndarray, np.ndarray]]:
  text = (map3_dir / "map3_collision.xml").read_text() + (map3_dir / "scene.xml").read_text()
  boxes = []
  for name, size, pos in _BOX_RE.findall(text):
    if name.startswith(("step", "landing", "connector", "floor", "roof", "rail")):
      boxes.append((name, np.fromstring(size, sep=" "), np.fromstring(pos, sep=" ")))
  if len(boxes) < 123:
    raise RuntimeError(f"Expected >= 123 map3 boxes, parsed {len(boxes)}.")
  return boxes


@dataclass(kw_only=True)
class Map3TerrainCfg(SubTerrainCfg):
  spawn: tuple[float, float, float] = (0.0, 0.0, 0.0)
  size: tuple[float, float] = MAP3_TILE_SIZE
  boxes: list = field(default_factory=list)

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    geometries = []
    for name, size, pos in self.boxes:
      geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(size), pos=tuple(pos + MAP3_OFFSET))
      rgba = (0.85, 0.3, 0.3, 1.0) if name.startswith("rail") else (0.78, 0.72, 0.6, 1.0)
      geometries.append(TerrainGeometry(geom=geom, color=rgba))
    return TerrainOutput(origin=np.asarray(self.spawn) + MAP3_OFFSET, geometries=geometries)


def make_env_cfg(scenario: Scenario, boxes, num_envs: int, seed: int):
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=MAP3_TILE_SIZE,
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    curriculum=False,
    sub_terrains={"map3": Map3TerrainCfg(spawn=scenario.spawn, boxes=boxes)},
    add_lights=True,
  )
  cfg.scene.terrain.max_init_terrain_level = None
  cfg.events.pop("randomize_terrain", None)
  cfg.events.pop("push_robot", None)
  yaw = math.radians(scenario.yaw_deg)
  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (yaw - 0.05, yaw + 0.05)}
  reset_base.params["stair_terrain_names"] = ()
  cfg.curriculum = {}
  cfg.episode_length_s = sum(s[0] for s in scenario.segments) + 1.0
  twist = cfg.commands["twist"]
  twist.stair_terrain_names = ()
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.zero_threshold = 0.0
  return cfg


def set_command(env, vx: float, vy: float, wz: float) -> None:
  """Make the command term resample exactly (vx, vy, wz) on the next step.

  Going through the term (instead of writing the command tensor) keeps every
  observation group, history included, consistent; like a joystick, the new
  command reaches the policy one control step later.
  """
  term = env.command_manager.get_term("twist")
  fixed = UniformVelocityCommandCfg.Ranges(lin_vel_x=(vx, vx), lin_vel_y=(vy, vy), ang_vel_z=(wz, wz))
  term.cfg.ranges = fixed
  term.cfg.stair_ranges = fixed
  term.cfg.flat_mode_probs = (0.0, 0.0, 0.0, 0.0, 1.0)
  term.cfg.stair_standing_prob = 0.0
  term.time_left[:] = 0.0


def load_policy(env, checkpoint: str):
  agent_cfg = load_rl_cfg(TASK)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK)(venv, asdict(agent_cfg), device="cuda:0")
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return venv, runner.get_inference_policy(device="cuda:0")


def run_scenario(args, scenario: Scenario, boxes) -> dict:
  env = ManagerBasedRlEnv(cfg=make_env_cfg(scenario, boxes, args.num_envs, args.seed), device="cuda:0")
  venv, policy = load_policy(env, args.checkpoint)
  env.reset()
  obs = venv.get_observations()

  robot = env.scene["robot"]
  z0 = robot.data.root_link_pos_w[:, 2].clone()
  zmin, zmax = z0.clone(), z0.clone()
  fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  err_xy, err_yaw, n = 0.0, 0.0, 0
  yaw_target = math.radians(scenario.yaw_deg)
  axis = torch.tensor((math.cos(yaw_target), math.sin(yaw_target)), device=env.device)
  term = env.command_manager.get_term("twist")
  kind, _, thr = scenario.success.partition(":")
  reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def goal_met():
    if kind == "climb":
      return zmax - z0 >= float(thr)
    if kind == "descend":
      return z0 - zmin >= float(thr)
    return torch.zeros_like(reached)

  for duration, vx, vy, wz in scenario.segments:
    set_command(env, vx, vy, wz)
    for step in range(int(round(duration / env.step_dt))):
      if args.operator and kind != "track" and vx != 0.0 and step > 0:
        heading_err = torch.atan2(
          torch.sin(yaw_target - robot.data.heading_w), torch.cos(yaw_target - robot.data.heading_w)
        )
        rel = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
        lateral = -axis[1] * rel[:, 0] + axis[0] * rel[:, 1]
        term.vel_command_b[:, 1] = (-lateral).clamp(-0.2, 0.2)
        term.vel_command_b[:, 2] = heading_err.clamp(-0.5, 0.5)
      with torch.no_grad():
        obs, _, dones, extras = venv.step(policy(obs))
      terminated = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
      # Goal reached before any termination (wall hits after the end of a flight,
      # where the command keeps walking forward, do not undo it).
      reached |= goal_met() & ~fell
      fell |= terminated
      z = robot.data.root_link_pos_w[:, 2]
      zmin = torch.where(fell, zmin, torch.minimum(zmin, z))
      zmax = torch.where(fell, zmax, torch.maximum(zmax, z))
      alive = ~fell
      if kind == "track" and alive.any():
        v = robot.data.root_link_lin_vel_b[alive]
        w = robot.data.root_link_ang_vel_b[alive, 2]
        err_xy += float(torch.norm(v[:, :2] - torch.tensor((vx, vy), device=env.device), dim=1).mean())
        err_yaw += float((w - wz).abs().mean())
        n += 1
  reached |= goal_met() & ~fell
  result = {"scenario": scenario.name}
  if kind == "track":
    result |= {"success": float((~fell).float().mean()), "err_vxy": err_xy / n, "err_wz": err_yaw / n}
  else:
    result |= {"reached": float(reached.float().mean()), "reset_before_goal": float((fell & ~reached).float().mean())}
  env.close()
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("checkpoint")
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--time-scale", type=float, default=1.0)
  parser.add_argument("--scenarios", nargs="*", default=[s.name for s in SCENARIOS])
  parser.add_argument("--operator", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  boxes = load_map3_boxes()
  results = []
  for s in SCENARIOS:
    if s.name not in args.scenarios:
      continue
    if s.success != "track":
      s.segments = [(d * args.time_scale if vx else d, vx, vy, wz) for d, vx, vy, wz in s.segments]
    results.append(run_scenario(args, s, boxes))
  print(f"\n[map3] {args.checkpoint}: {args.num_envs} robots per scenario, time scale {args.time_scale}")
  for r in results:
    print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
  main()
