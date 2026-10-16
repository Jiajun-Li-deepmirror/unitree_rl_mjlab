"""Evaluate a Go2 depth teacher or student on the tinynav-sim map3 stairwell.

map3 (123 collision boxes + the sim guard rails, copied into
src/assets/terrains/map3) is loaded as a single terrain tile; it is never used
for training. Each
scenario spawns all envs at one map3 spot with a fixed heading and runs a
fixed command script, then reports per-scenario success, falls and height
change.

  PYTHONPATH=. python scripts/go2_depth/eval_map3.py \
    --policy student --checkpoint logs/rsl_rl/go2_depth_student/<run>/model_N.pt
  PYTHONPATH=. python scripts/go2_depth/eval_map3.py \
    --policy teacher --checkpoint logs/rsl_rl/go2_depth_teacher/<run>/model_N.pt
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

# Shift map3 world coordinates into a positive tile.
MAP3_OFFSET = np.array([13.0, 1.0, 0.0])
MAP3_TILE_SIZE = (14.0, 10.0)
_BOX_RE = re.compile(
  r'<geom name="([^"]+)"\s+type="box"\s+[^>]*?size="([^"]+)"\s+pos="([^"]+)"'
)


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
  keep_below: float = 1.0e9
  """Teacher only: drop map3 boxes whose top is above this map3 z (rails are
  kept). The teacher's height scan casts from 8 m above the robot and was
  trained without overhangs, so the floors and flights of the stories above
  would read as walls everywhere. The depth student sees them normally."""


SCENARIOS = [
  Scenario("L0_up", (-10.34, 5.75, 0.431), -90.0, [(1.0, 0, 0, 0), (9.0, 0.5, 0, 0)], "climb:2.0", 3.3),
  Scenario("L0_down", (-10.34, 1.05, 2.659), 90.0, [(1.0, 0, 0, 0), (9.0, 0.4, 0, 0)], "descend:2.0", 3.7),
  Scenario("R1_up", (-8.80, 1.05, 2.659), 90.0, [(1.0, 0, 0, 0), (9.0, 0.5, 0, 0)], "climb:1.7", 4.8),
  Scenario("R2b_down", (-8.75, 5.70, 11.802), -90.0, [(1.0, 0, 0, 0), (7.0, 0.4, 0, 0)], "descend:1.0", 12.8),
  Scenario(
    "L0_stop_midway",
    (-10.34, 5.75, 0.431),
    -90.0,
    [(1.0, 0, 0, 0), (4.0, 0.5, 0, 0), (4.0, 0, 0, 0), (5.0, 0.5, 0, 0)],
    "climb:2.0",
    3.3,
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
    1.0,
  ),
]


MAP3_DIR = Path(__file__).resolve().parents[2] / "src" / "assets" / "terrains" / "map3"


def _load_map3_boxes(map3_dir: Path = MAP3_DIR) -> list[tuple[str, np.ndarray, np.ndarray]]:
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

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    geometries = []
    for name, size, pos in self.boxes:
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(size), pos=tuple(pos + MAP3_OFFSET)
      )
      rgba = (0.85, 0.3, 0.3, 1.0) if name.startswith("rail") else (0.78, 0.72, 0.6, 1.0)
      geometries.append(TerrainGeometry(geom=geom, color=rgba))
    origin = np.asarray(self.spawn) + MAP3_OFFSET
    return TerrainOutput(origin=origin, geometries=geometries)


def make_env_cfg(task: str, scenario: Scenario, boxes, num_envs: int):
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
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
  reset_base.params["pose_range"] = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "yaw": (yaw - 0.05, yaw + 0.05),
  }
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
  from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

  term = env.command_manager.get_term("twist")
  fixed = UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(vx, vx), lin_vel_y=(vy, vy), ang_vel_z=(wz, wz)
  )
  term.cfg.ranges = fixed
  term.cfg.stair_ranges = fixed
  term.cfg.flat_mode_probs = (0.0, 0.0, 0.0, 0.0, 1.0)
  term.cfg.stair_standing_prob = 0.0
  term.time_left[:] = 0.0


def load_policy(kind: str, checkpoint: str, obs: TensorDict, device: str):
  from rsl_rl.models import MLPModel

  from src.tasks.velocity.rl.distillation import DepthStudentModel

  ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
  if kind == "teacher":
    state = ck.get("actor_state_dict") or ck["teacher_state_dict"]
    model = MLPModel(
      obs,
      {"actor": ["actor"]},
      "actor",
      12,
      hidden_dims=(512, 256, 128),
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
  else:
    state = ck["student_state_dict"]
    model = DepthStudentModel(
      obs, {"student": ["proprio", "proprio_history", "depth"]}, "student", 12
    )
  model.load_state_dict(state)
  return model.to(device).eval()


def run_scenario(args, scenario: Scenario, boxes) -> dict:
  from mjlab.envs import ManagerBasedRlEnv

  task = "Unitree-Go2-Depth-Student" if args.policy == "student" else "Unitree-Go2-Depth-Teacher"
  if args.policy == "teacher":
    boxes = [
      (n, size, pos) for n, size, pos in boxes
      if n.startswith("rail") or pos[2] + size[2] <= scenario.keep_below
    ]
  if args.solid_gaps:
    # Diagnostic: fill the open stairwell well and the east slot so the side
    # of every flight is a solid wall, as in the training terrain.
    for name, (x0, x1) in (("rail_fill_well", (-9.86, -9.30)), ("rail_fill_slot", (-8.34, -8.12))):
      lo = np.array([x0, 0.40, -0.2]); hi = np.array([x1, 6.50, 19.0])
      boxes = boxes + [(name, (hi - lo) / 2, (hi + lo) / 2)]
  env = ManagerBasedRlEnv(cfg=make_env_cfg(task, scenario, boxes, args.num_envs), device="cuda:0")
  obs, _ = env.reset()
  obs = TensorDict(obs, batch_size=[env.num_envs])
  policy = load_policy(args.policy, args.checkpoint, obs, env.device)
  if getattr(policy, "is_recurrent", False):
    policy.reset()

  robot = env.scene["robot"]
  z0 = robot.data.root_link_pos_w[:, 2].clone()
  zmin, zmax = z0.clone(), z0.clone()
  fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  err_xy, err_yaw, n = 0.0, 0.0, 0
  fall_along = torch.full((env.num_envs,), float("nan"), device=env.device)
  yaw_target = math.radians(scenario.yaw_deg)
  axis = torch.tensor((math.cos(yaw_target), math.sin(yaw_target)), device=env.device)
  term = env.command_manager.get_term("twist")
  reasons = {
    t: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for t in env.termination_manager.active_terms
    if t != "time_out"
  }
  for duration, vx, vy, wz in scenario.segments:
    set_command(env, vx, vy, wz)
    for step in range(int(round(duration / env.step_dt))):
      if args.operator and scenario.success != "track" and vx != 0.0 and step > 0:
        # Operator keeping the robot on the flight, like the training
        # corridor hold: hold the flight heading, steer back to its centerline.
        heading_err = torch.atan2(
          torch.sin(yaw_target - robot.data.heading_w), torch.cos(yaw_target - robot.data.heading_w)
        )
        rel = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
        lateral = -axis[1] * rel[:, 0] + axis[0] * rel[:, 1]
        term.vel_command_b[:, 1] = (-lateral).clamp(-0.2, 0.2)
        term.vel_command_b[:, 2] = heading_err.clamp(-0.5, 0.5)
      # Terminated envs are reset inside step(), so remember the pose before it.
      pre_along = ((robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]) * axis).sum(-1)
      with torch.inference_mode():
        actions = policy(obs)
        raw, _, terminated, time_outs, _ = env.step(actions)
        if getattr(policy, "is_recurrent", False):
          policy.reset((terminated | time_outs).float())
      obs = TensorDict(raw, batch_size=[env.num_envs])
      for t in reasons:
        reasons[t] |= env.termination_manager.get_term(t) & ~fell
      new_fall = terminated & ~fell
      fall_along = torch.where(new_fall, pre_along, fall_along)
      fell |= terminated
      alive = ~fell
      z = robot.data.root_link_pos_w[:, 2]
      zmin = torch.where(alive, torch.minimum(zmin, z), zmin)
      zmax = torch.where(alive, torch.maximum(zmax, z), zmax)
      if scenario.success == "track" and alive.any():
        v = robot.data.root_link_lin_vel_b[alive]
        w = robot.data.root_link_ang_vel_b[alive, 2]
        target = torch.tensor((vx, vy), device=env.device)
        err_xy += float(torch.norm(v[:, :2] - target, dim=1).mean())
        err_yaw += float((w - wz).abs().mean())
        n += 1
  kind, _, thr = scenario.success.partition(":")
  if kind == "climb":
    ok = (zmax - z0 >= float(thr)) & ~fell
  elif kind == "descend":
    ok = (z0 - zmin >= float(thr)) & ~fell
  else:
    ok = ~fell
  result = {
    "scenario": scenario.name,
    "success": float(ok.float().mean()),
    "falls": float(fell.float().mean()),
    "dz_up": float((zmax - z0).mean()),
    "dz_down": float((z0 - zmin).mean()),
    **{f"by_{t}": float(v.float().mean()) for t, v in reasons.items()},
  }
  if fell.any():
    result["fall_at_m"] = sorted(round(float(x), 2) for x in fall_along[fell].tolist())
  if n:
    result |= {"err_vxy": err_xy / n, "err_wz": err_yaw / n}
  env.close()
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy", choices=("teacher", "student"), required=True)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument(
    "--map3-dir", type=Path, default=MAP3_DIR,
    help="Folder with map3_collision.xml and scene.xml (default: the copy in this repo).",
  )
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--scenarios", nargs="*", default=[s.name for s in SCENARIOS])
  parser.add_argument("--solid-gaps", action="store_true")
  parser.add_argument(
    "--operator", action=argparse.BooleanOptionalAction, default=True,
    help="On stair scenarios, hold the flight heading and centerline like an operator.",
  )
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  boxes = _load_map3_boxes(args.map3_dir.expanduser())
  results = [run_scenario(args, s, boxes) for s in SCENARIOS if s.name in args.scenarios]
  print("\n[map3 eval]", args.policy, args.checkpoint)
  for r in results:
    print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
  main()
