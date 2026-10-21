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
import gc
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
from mjlab.utils.lab_api.math import quat_apply

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


ROBOT_TASK_NAMES = {"go2": "Go2", "go2w": "Go2W", "as2": "As2", "b2": "B2", "b2w": "B2W", "as2w": "As2W"}

SCENARIOS = [
  Scenario("L0_up", (-10.34, 5.75, 0.431), -90.0, [(1.0, 0, 0, 0), (9.0, 0.5, 0, 0)], "climb:2.0", 3.3),
  # The 3.65 m flight takes ~9.1 s at 0.4 m/s; then stop on the landing like an
  # operator would (the landing ends in a wall ~1.3 m further).
  Scenario(
    "L0_down", (-10.34, 1.05, 2.659), 90.0,
    [(1.0, 0, 0, 0), (10.0, 0.4, 0, 0), (2.0, 0, 0, 0)], "descend:2.0", 3.7,
  ),
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


# --set flights (go2-blind d9ed8ba): up and down every map3 flight (step_<flight>_NN boxes).
APPROACH = 0.65
"""Spawn distance before the first riser (up) or the top edge (down), m."""
GOAL_SHARE = 0.85
"""Share of the flight's rise that counts as done."""
TIME_SLACK = 1.25
"""Walking time = this x (approach + goal share of the flight) / commanded speed."""
BACK_CLEARANCE = 0.65
"""Room kept behind a spawn, m. F1_entry_X_down's top landing leaves 0.51 m to
the outer rail: the B2's tail spawned inside it and was thrown down the flight
(go2-blind found this; 14-19% reached -> 98-100% moved forward)."""


def _room_behind(boxes, x: float, y: float, z: float, yaw_deg: float, reach: float) -> float:
  """Free distance behind (x, y) at body height above the surface z, up to `reach`."""
  bx, by = -math.cos(math.radians(yaw_deg)), -math.sin(math.radians(yaw_deg))
  for i in range(1, int(round(reach / 0.01)) + 1):
    px, py = x + bx * i * 0.01, y + by * i * 0.01
    for name, s, p in boxes:
      if (
        not name.startswith("roof")
        and abs(px - p[0]) <= s[0]
        and abs(py - p[1]) <= s[1]
        and p[2] + s[2] > z + 0.1
        and p[2] - s[2] < z + 0.6
      ):
        return i * 0.01
  return reach


def _surface_z(boxes, x: float, y: float, below: float | None = None, above: float | None = None) -> float:
  """Top of the highest box under (x, y) up to `below`, or of the lowest one from `above`."""
  tops = [
    p[2] + s[2]
    for name, s, p in boxes
    if not name.startswith(("rail", "roof")) and abs(x - p[0]) <= s[0] and abs(y - p[1]) <= s[1]
  ]
  if below is not None:
    return max(t for t in tops if t <= below)
  return min(t for t in tops if t >= above)


def flight_scenarios(boxes, up_vx: float = 0.5, down_vx: float = 0.4) -> list[Scenario]:
  """Up and down every map3 flight, spawned on its landings (go2-blind's set)."""
  flights: dict[str, list] = {}
  for name, size, pos in boxes:
    m = re.match(r"step_(.+)_\d+$", name)
    if m:
      flights.setdefault(m.group(1), []).append((size, pos))
  scenarios = []
  for flight, steps in sorted(flights.items()):
    steps.sort(key=lambda b: b[1][2] + b[0][2])  # by tread height
    (s_lo, p_lo), (s_hi, p_hi) = steps[0], steps[-1]
    d = p_hi[:2] - p_lo[:2]
    axis = int(abs(d[1]) > abs(d[0]))  # ascent along x (0) or y (1)
    sign = 1.0 if d[axis] > 0 else -1.0
    other = 1 - axis
    lateral = float(np.mean([p[other] for _, p in steps]))
    length = max(p[axis] + s[axis] for s, p in steps) - min(p[axis] - s[axis] for s, p in steps)
    first_riser = p_lo[axis] - sign * s_lo[axis]
    top_edge = p_hi[axis] + sign * s_hi[axis]

    def point(u: float, axis=axis, other=other, lateral=lateral) -> tuple[float, float]:
      xy = [0.0, 0.0]
      xy[axis], xy[other] = u, lateral
      return xy[0], xy[1]

    bottom = point(first_riser - sign * APPROACH)
    top = point(top_edge + sign * APPROACH)
    z_bottom = _surface_z(boxes, *bottom, below=p_lo[2] + s_lo[2] - 0.02)
    z_top = _surface_z(boxes, *top, above=p_hi[2] + s_hi[2] + 0.02)
    rise = z_top - z_bottom
    yaw_up = math.degrees(math.atan2(sign * axis, sign * (1 - axis)))
    for direction, (x, y), z, yaw, vx, kind in (
      ("up", bottom, z_bottom, yaw_up, up_vx, "climb"),
      ("down", top, z_top, yaw_up + 180.0, down_vx, "descend"),
    ):
      shift = BACK_CLEARANCE - _room_behind(boxes, x, y, z, yaw, BACK_CLEARANCE)
      x, y = x + shift * math.cos(math.radians(yaw)), y + shift * math.sin(math.radians(yaw))
      walk = round(float(TIME_SLACK * (APPROACH + GOAL_SHARE * length) / vx), 1)
      scenarios.append(
        Scenario(
          f"{flight}_{direction}",
          (float(x), float(y), float(z)),
          (yaw + 180.0) % 360.0 - 180.0,
          [(1.0, 0, 0, 0), (walk, vx, 0, 0), (2.0, 0, 0, 0)],
          f"{kind}:{GOAL_SHARE * rise:.2f}",
          # Teacher: the stories above the flight's top landing are dropped.
          z_top + 1.0,
        )
      )
  return scenarios


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


def solid_gap_boxes(keep_landings: bool = False) -> list[tuple[str, np.ndarray, np.ndarray]]:
  """Diagnostic: fill the open stairwell well and the east slot so the side of
  every flight is a solid wall, as in the training terrain (the teacher's
  height scan reads the open well as a cliff).

  The full fill (y 0.40-6.50) also cuts the F2 landing between L0 and R1,
  which no eval scenario crosses; `keep_landings` fills only between the
  landings (y 1.73-5.08) so a tour can still walk from one flight to the other.
  """
  y0, y1 = (1.73, 5.08) if keep_landings else (0.40, 6.50)
  boxes = []
  for name, (x0, x1) in (("rail_fill_well", (-9.86, -9.30)), ("rail_fill_slot", (-8.34, -8.12))):
    lo = np.array([x0, y0, -0.2]); hi = np.array([x1, y1, 19.0])
    boxes.append((name, (hi - lo) / 2, (hi + lo) / 2))
  return boxes


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
  # Sampling options that would replace the command (slow linear range,
  # steering arcs; map3 counts as flat terrain).
  term.cfg.flat_slow_prob = 0.0
  term.cfg.flat_steer_prob = 0.0
  term.time_left[:] = 0.0


def load_policy(kind: str, checkpoint: str, obs: TensorDict, device: str):
  from rsl_rl.models import MLPModel

  from src.tasks.velocity.rl.distillation import DepthStudentModel

  ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
  if kind == "teacher":
    state = ck.get("actor_state_dict") or ck["teacher_state_dict"]
  else:
    state = ck["student_state_dict"]
  # Action count from the last MLP layer (12 legs, + 4 wheels on wheeled robots).
  weights = [k for k in state if k.startswith("mlp.") and k.endswith(".weight")]
  num_actions = state[max(weights, key=lambda k: int(k.split(".")[1]))].shape[0]
  if kind == "teacher":
    model = MLPModel(
      obs,
      {"actor": ["actor"]},
      "actor",
      num_actions,
      hidden_dims=(512, 256, 128),
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
  else:
    model = DepthStudentModel(
      obs, {"student": ["proprio", "proprio_history", "depth"]}, "student", num_actions
    )
  # The student's terrain reconstruction head is training-only; checkpoints
  # from before it existed load without it.
  missing, unexpected = model.load_state_dict(state, strict=False)
  if unexpected or any(not k.startswith("aux_head.") for k in missing):
    raise RuntimeError(f"Checkpoint mismatch: {missing=} {unexpected=}")
  return model.to(device).eval()


def run_scenario(args, scenario: Scenario, boxes) -> dict:
  from mjlab.envs import ManagerBasedRlEnv

  task = f"Unitree-{ROBOT_TASK_NAMES[args.robot]}-Depth-{args.policy.capitalize()}"
  if args.heading:
    task += "-Heading"
  if args.policy == "teacher" or args.drop_overhead:
    boxes = [
      (n, size, pos) for n, size, pos in boxes
      if n.startswith("rail") or pos[2] + size[2] <= scenario.keep_below
    ]
  if args.solid_gaps:
    boxes = boxes + solid_gap_boxes()
  if args.policy == "teacher":
    # map3 rails/walls are up to 19 m tall, above the teacher's scan origins
    # (8 m over the robot), so every ray at a wall would start inside it and
    # pass through: no walls at all. Cut them to just above the scenario's
    # height range, as in training where wall tops are below the origins.
    clipped = []
    for n, size, pos in boxes:
      if n.startswith("rail"):
        lo_z, hi_z = pos[2] - size[2], min(pos[2] + size[2], scenario.keep_below + 1.5)
        if hi_z <= lo_z:
          continue
        size = np.array([size[0], size[1], 0.5 * (hi_z - lo_z)])
        pos = np.array([pos[0], pos[1], 0.5 * (hi_z + lo_z)])
      clipped.append((n, size, pos))
    boxes = clipped
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
  # Flight speed (go2-blind): time and position along the flight when the
  # height change passes 0.15 m and the target minus 0.15 m, to tell a slow
  # robot from a short scripted time.
  kind, _, thr = scenario.success.partition(":")
  sign = {"climb": 1.0, "descend": -1.0}.get(kind)
  marks = {k: torch.full((env.num_envs,), float("nan"), device=env.device)
           for k in ("t_a", "x_a", "t_b", "x_b")}
  elapsed = 0.0
  # Stance-leg splay (go2-blind): how far the leg plane of a leg on the ground
  # tilts outward beyond the default hip angle (the stance_leg_tilt reward's
  # quantity, signed outward), in degrees, on the flight (between the two
  # marks); front and rear legs apart.
  splay = None
  if sign is not None and "stance_leg_tilt" in env.reward_manager.active_terms:
    params = env.reward_manager.get_term_cfg("stance_leg_tilt").params
    legs = params["asset_cfg"]
    names = [robot.body_names[i] for i in legs.body_ids]
    splay = {
      "cfg": legs,
      "contact": env.scene[params["sensor_name"]],
      "out": torch.tensor([1.0 if n.startswith(("FL", "RL")) else -1.0 for n in names], device=env.device),
      "front": torch.tensor([n.startswith("F") for n in names], device=env.device),
      "sum": torch.zeros(len(names), device=env.device),
      "n": torch.zeros(len(names), device=env.device),
    }
  reasons = {
    t: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for t in env.termination_manager.active_terms
    if t != "time_out"
  }
  # go2-blind's goal: reached before any termination (a wall hit after the end
  # of a flight, where the command keeps walking forward, does not undo it),
  # with the heading error (deg, + = turned left) and the offset from the spawn
  # line (m, + = left) at that moment: the drift without the operator.
  reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  yaw_at = torch.full((env.num_envs,), float("nan"), device=env.device)
  lat_at = yaw_at.clone()
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
      if sign is not None:
        z_now = robot.data.root_link_pos_w[:, 2]
        met = sign * (z_now - z0) >= float(thr)
        new = met & ~fell & ~terminated & ~reached
        if new.any():
          err = robot.data.heading_w - yaw_target
          err = torch.rad2deg(torch.atan2(torch.sin(err), torch.cos(err)))
          rel = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
          yaw_at = torch.where(new, err, yaw_at)
          lat_at = torch.where(new, -axis[1] * rel[:, 0] + axis[0] * rel[:, 1], lat_at)
        reached |= met & ~fell & ~terminated
      new_fall = terminated & ~fell
      fall_along = torch.where(new_fall, pre_along, fall_along)
      fell |= terminated
      alive = ~fell
      z = robot.data.root_link_pos_w[:, 2]
      zmin = torch.where(alive, torch.minimum(zmin, z), zmin)
      zmax = torch.where(alive, torch.maximum(zmax, z), zmax)
      elapsed += env.step_dt
      if sign is not None:
        dz = sign * (z - z0)
        along = ((robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]) * axis).sum(-1)
        for tag, level in (("a", 0.15), ("b", float(thr) - 0.15)):
          hit = alive & marks[f"t_{tag}"].isnan() & (dz >= level)
          marks[f"t_{tag}"] = torch.where(hit, elapsed, marks[f"t_{tag}"])
          marks[f"x_{tag}"] = torch.where(hit, along, marks[f"x_{tag}"])
        if splay is not None:
          on_flight = alive & ~marks["t_a"].isnan() & marks["t_b"].isnan()
          quat = robot.data.body_link_quat_w[:, splay["cfg"].body_ids]
          lateral = torch.zeros(*quat.shape[:-1], 3, device=env.device)
          lateral[..., 1] = 1.0
          tilt = torch.asin(quat_apply(quat, lateral)[..., 2].clamp(-1.0, 1.0))
          out = (tilt - robot.data.default_joint_pos[:, splay["cfg"].joint_ids]) * splay["out"]
          w = (splay["contact"].data.found > 0).float() * on_flight.float().unsqueeze(1)
          splay["sum"] += (out * w).sum(0)
          splay["n"] += w.sum(0)
      if args.trace and alive.any() and (step + 1) % int(round(1.0 / env.step_dt)) == 0:
        along = ((robot.data.root_link_pos_w[alive, :2] - env.scene.env_origins[alive, :2]) * axis).sum(-1)
        q = torch.quantile(along, torch.tensor((0.1, 0.5, 0.9), device=env.device)).tolist()
        print(f"  [trace] cmd_vx={vx:+.1f} along p10/p50/p90 = {q[0]:5.2f} {q[1]:5.2f} {q[2]:5.2f}", flush=True)
      if scenario.success == "track" and alive.any():
        v = robot.data.root_link_lin_vel_b[alive]
        w = robot.data.root_link_ang_vel_b[alive, 2]
        target = torch.tensor((vx, vy), device=env.device)
        err_xy += float(torch.norm(v[:, :2] - target, dim=1).mean())
        err_yaw += float((w - wz).abs().mean())
        n += 1
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
  if sign is not None:
    result["reached"] = float(reached.float().mean())
    ok_at = ~torch.isnan(yaw_at)
    if ok_at.any():
      qs = torch.tensor((0.1, 0.5, 0.9), device=env.device)
      yq, lq = yaw_at[ok_at].quantile(qs).tolist(), lat_at[ok_at].quantile(qs).tolist()
      result |= {"yaw_deg_p10": yq[0], "yaw_deg_p90": yq[2], "lat_m_p10": lq[0], "lat_m_p90": lq[2]}
  if fell.any():
    result["fall_at_m"] = sorted(round(float(x), 2) for x in fall_along[fell].tolist())
  moving = [abs(vx) for _, vx, _, _ in scenario.segments if vx]
  timed = ~marks["t_b"].isnan() & ~fell
  if sign is not None and len(set(moving)) == 1 and timed.sum() > 0 and not any(
    vx == 0 for _, vx, _, _ in scenario.segments[1:-1]
  ):
    speed = (marks["x_b"] - marks["x_a"]).abs() / (marks["t_b"] - marks["t_a"]).clamp(min=1e-3)
    r = speed[timed] / moving[0]
    q = torch.quantile(r, torch.tensor((0.1, 0.5, 0.9), device=r.device)).tolist()
    result |= {"speed_ratio": q[1], "speed_ratio_p10": q[0], "speed_ratio_p90": q[2]}
  if splay is not None and splay["n"].sum() > 0:
    for tag, legs in (("front", splay["front"]), ("rear", ~splay["front"])):
      result[f"splay_{tag}_deg"] = math.degrees(float(splay["sum"][legs].sum() / splay["n"][legs].sum().clamp(min=1)))
  if n:
    result |= {"err_vxy": err_xy / n, "err_wz": err_yaw / n}
  env.close()
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy", choices=("teacher", "student"), required=True)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--robot", choices=tuple(ROBOT_TASK_NAMES), default="go2")
  parser.add_argument(
    "--map3-dir", type=Path, default=MAP3_DIR,
    help="Folder with map3_collision.xml and scene.xml (default: the copy in this repo).",
  )
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument(
    "--set", choices=("default", "flights"), default="default",
    help="default: the scenarios above; flights: up and down every map3 flight (go2-blind's 18).",
  )
  parser.add_argument("--up-vx", type=float, default=0.5, help="--set flights: climbing command (m/s).")
  parser.add_argument("--down-vx", type=float, default=0.4, help="--set flights: descending command (m/s).")
  parser.add_argument("--scenarios", nargs="*", default=None, help="Names to run (default: the whole set).")
  parser.add_argument("--list", action="store_true", help="Print the scenarios and exit.")
  parser.add_argument(
    "--heading", action="store_true",
    help="Policy from a heading task (go2-blind d9ed8ba recipe, heading-error input).",
  )
  parser.add_argument("--solid-gaps", action="store_true")
  parser.add_argument("--trace", action="store_true", help="Print progress along the flight every second.")
  parser.add_argument(
    "--drop-overhead", action="store_true",
    help="Diagnostic: drop the stories above the scenario for the student too.",
  )
  parser.add_argument(
    "--operator", action=argparse.BooleanOptionalAction, default=True,
    help="On stair scenarios, hold the flight heading and centerline like an operator.",
  )
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  boxes = _load_map3_boxes(args.map3_dir.expanduser())
  scenarios = SCENARIOS if args.set == "default" else flight_scenarios(boxes, args.up_vx, args.down_vx)
  if args.list:
    for s in scenarios:
      print(f"{s.name:16s} spawn ({s.spawn[0]:7.2f}, {s.spawn[1]:5.2f}, {s.spawn[2]:6.3f}) "
            f"yaw {s.yaw_deg:6.1f}  {s.success:14s} segments {s.segments}")
    return
  results = []
  for s in scenarios:
    if args.scenarios is None or s.name in args.scenarios:
      results.append(run_scenario(args, s, boxes))
      # A closed env holds its GPU arrays until the garbage collector breaks its
      # reference cycles; without this the next env's CUDA graph capture failed
      # (2nd scenario on a 6 GB GPU, ~11th on go2-blind's 24 GB one).
      gc.collect()
      torch.cuda.empty_cache()
  print("\n[map3 eval]", args.policy, args.checkpoint, f"set={args.set}", f"operator={args.operator}")
  for r in results:
    print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
  if args.set == "flights" and results:
    print(f"  flights reached mean {np.mean([r['reached'] for r in results]):.3f} over {len(results)}")


if __name__ == "__main__":
  main()
