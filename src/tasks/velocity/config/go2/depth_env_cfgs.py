"""Unitree Go2 depth-camera velocity configurations (teacher-student).

The teacher is the go2-blind task with a privileged height scan in the actor
and map3-like walled stairs in the terrain mix. The student sees what the
tinynav-sim robot sees: proprioception (with a 10-step history) and two
10 Hz frames from the front depth camera. It is distilled from the teacher.

Rewards, events, domain randomization, collisions and terminations are the
go2-blind ones; only terrain, commands and observations change.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.config.go2.blind_env_cfgs import (
  ACTOR_HISTORY_LENGTH,
  NUM_STEPS_PER_ENV,
  unitree_go2_blind_env_cfg,
)
from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

# Proprioceptive actor terms, in the order the deployed policy receives them.
PROPRIO_TERM_NAMES = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "phase",
  "joint_pos",
  "joint_vel",
  "actions",
)

# Sub-terrains, in generator (column) order.
PYRAMID_STAIRS = ("pyramid_down", "pyramid_up")
CROSS_STAIRS = ("walled_down", "walled_up")
UTURN_STAIRS = ("stairwell_up", "stairwell_down")
STAIR_TERRAIN_NAMES = PYRAMID_STAIRS + CROSS_STAIRS + UTURN_STAIRS
# Corridors ~1 m wide: spawn with little x/y noise.
NARROW_TERRAIN_NAMES = CROSS_STAIRS + UTURN_STAIRS

# tinynav-sim command envelope (keyboard and HIL /cmd_vel clip).
FINAL_VEL_RANGES = {
  "lin_vel_x": (-0.5, 1.0),
  "lin_vel_y": (-0.5, 0.5),
  "ang_vel_z": (-1.2, 1.2),
}
INITIAL_VEL_RANGES = {
  "lin_vel_x": (-0.3, 0.6),
  "lin_vel_y": (-0.3, 0.3),
  "ang_vel_z": (-0.8, 0.8),
}
STAIR_VEL_RANGES = {
  "lin_vel_x": (0.2, 0.6),
  "lin_vel_y": (-0.1, 0.1),
  "ang_vel_z": (-0.3, 0.3),
}

# map3 rises reach ~19 cm; the walled tiles add +-1.5 cm per step on top.
STEP_HEIGHT_RANGE = (0.05, 0.20)
# Foothold progress is paid in full up to 0.05 m/s over the commanded speed and
# not at all from 0.25 m/s over (v9 ran down map3 at 0.6 m/s for 0.4).
FOOTHOLD_OVERSPEED_RANGE = (0.05, 0.25)

DEPTH_SENSOR_NAME = "front_depth"

SCAN_Z_OFFSET = 8.0
FOOT_SCAN_NAME = "foot_scan"
WALL_SCAN_NAME = "wall_scan"
FOOT_SITE_NAMES = ("FR", "FL", "RR", "RL")
SCAN_WIDTH = 1.2
SCAN_CLIP = (-1.0, 2.0)
SCAN_SCALE = 1.0 / 5.0  # go2-blind's 1 / terrain_scan.max_distance.


def make_go2_depth_terrains_cfg(stairs: bool) -> TerrainGeneratorCfg:
  """go2-blind flat/rough/pyramid mix plus walled map3-like stairs."""
  sub_terrains: dict[str, terrain_gen.SubTerrainCfg] = {
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.20),
  }
  if stairs:
    pyramid = dict(
      proportion=0.10,
      step_height_range=STEP_HEIGHT_RANGE,
      step_width=0.28,
      platform_width=2.0,
      border_width=0.5,
    )
    walled = dict(step_height_range=STEP_HEIGHT_RANGE)
    sub_terrains |= {
      "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.10,
        noise_range=(0.02, 0.06),
        noise_step=0.01,
        border_width=0.25,
      ),
      # Spawn on the top platform and walk down / in the pit and walk up.
      "pyramid_down": terrain_gen.BoxPyramidStairsTerrainCfg(**pyramid),
      "pyramid_up": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(**pyramid),
      "walled_down": mdp.WalledCrossStairsTerrainCfg(proportion=0.15, **walled),
      "walled_up": mdp.WalledCrossStairsTerrainCfg(
        proportion=0.15, inverted=True, **walled
      ),
      "stairwell_up": mdp.StairwellUTurnTerrainCfg(proportion=0.10, **walled),
      "stairwell_down": mdp.StairwellUTurnTerrainCfg(
        proportion=0.10, descend=True, **walled
      ),
    }
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains=sub_terrains,
    add_lights=True,
  )


def unitree_go2_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  """go2-blind with a height scan in the actor and the map3-like terrain mix.

  The actor keeps the blind 10-step proprio history and adds the current
  height scan (no history), so its first 470 dims match the student's
  `proprio_history` layout.
  """
  cfg = unitree_go2_blind_env_cfg(play=play, stairs=stairs)

  ##
  # Terrain.
  ##

  assert cfg.scene.terrain is not None
  terrain_generator = make_go2_depth_terrains_cfg(stairs)
  if play:
    terrain_generator.num_rows = 5
    terrain_generator.num_cols = 10
    terrain_generator.border_width = 10.0
  cfg.scene.terrain.terrain_generator = terrain_generator

  ##
  # Sensors.
  ##

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      pattern = sensor.pattern
      assert isinstance(pattern, GridPatternCfg)
      # Cast from above the walls so the teacher can see them (wall tops reach
      # ~6 m above the floor on the hardest stairwells).
      # 1.2 m wide (go2-blind: 1.0) so both walls of a ~1 m flight are in view.
      sensor.pattern = mdp.ElevatedGridPatternCfg(
        size=(pattern.size[0], SCAN_WIDTH),
        resolution=pattern.resolution,
        z_offset=SCAN_Z_OFFSET,
      )
      sensor.max_distance = SCAN_Z_OFFSET + 5.0
      # Fine scan around the feet, for the terrain clearance reward only.
      foot_scan = deepcopy(sensor)
      foot_scan.name = FOOT_SCAN_NAME
      foot_scan.pattern = mdp.ElevatedGridPatternCfg(
        size=(0.8, 0.7), resolution=0.04, z_offset=SCAN_Z_OFFSET
      )
      foot_scan.debug_vis = False
  # Level ray fan to both sides at base height, for the wall proximity reward.
  wall_scan = deepcopy(foot_scan)
  wall_scan.name = WALL_SCAN_NAME
  wall_scan.pattern = mdp.HorizontalFanPatternCfg()
  wall_scan.max_distance = 1.0
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (foot_scan, wall_scan)

  ##
  # Rewards (stair climbing, from the later go2-blind experiments).
  ##

  standing = {"command_name": "twist", "command_threshold": 0.1}
  feet = SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)
  # Lifting higher over stairs is fine; only too-low swings are penalized there.
  cfg.rewards["foot_swing_clearance"].params["scan_sensor_name"] = "terrain_scan"
  cfg.rewards["foot_terrain_clearance"] = RewardTermCfg(
    func=mdp.feet_terrain_clearance,
    weight=-40.0,
    params={
      "sensor_name": "feet_ground_contact",
      "scan_sensor_name": FOOT_SCAN_NAME,
      "asset_cfg": feet,
    },
  )
  # Positive progress signal: without it robots stuck at a step back off,
  # since stalling and slowly reversing score about the same.
  cfg.rewards["foothold_height_gain"] = RewardTermCfg(
    func=mdp.feet_foothold_height_gain,
    weight=2.0,
    params={
      "sensor_name": "feet_ground_contact",
      **standing,
      "asset_cfg": deepcopy(feet),
      "overspeed_range": FOOTHOLD_OVERSPEED_RANGE,
    },
  )
  # The policy keeps itself off the walls; corridor commands only hold heading.
  cfg.rewards["wall_proximity"] = RewardTermCfg(
    func=mdp.wall_proximity,
    weight=-20.0,
    params={"sensor_name": WALL_SCAN_NAME, "safe_distance": 0.35},
  )

  # Same progress signal for going down; high narrow descents stalled without it.
  cfg.rewards["foothold_height_drop"] = RewardTermCfg(
    func=mdp.feet_foothold_height_gain,
    weight=2.0,
    params={
      "sensor_name": "feet_ground_contact",
      **standing,
      "asset_cfg": deepcopy(feet),
      "descend": True,
      "overspeed_range": FOOTHOLD_OVERSPEED_RANGE,
    },
  )

  ##
  # Observations.
  ##

  # Walls read as large negative heights; clip them (meters, before scaling)
  # and keep go2-blind's scaling for the floor.
  for group in ("actor", "critic"):
    term = cfg.observations[group].terms.get("height_scan")
    if term is not None:
      term.clip = SCAN_CLIP
      term.scale = SCAN_SCALE

  actor = cfg.observations["actor"]
  actor.history_length = None
  for name in PROPRIO_TERM_NAMES:
    actor.terms[name].history_length = ACTOR_HISTORY_LENGTH
  height_scan = deepcopy(cfg.observations["critic"].terms["height_scan"])
  height_scan.noise = Unoise(n_min=-0.05, n_max=0.05)
  height_scan.history_length = 0
  actor.terms["height_scan"] = height_scan

  ##
  # Commands.
  ##

  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.TerrainAwareVelocityCommandCfg)
  vel_ranges = FINAL_VEL_RANGES if play else INITIAL_VEL_RANGES
  twist.ranges = UniformVelocityCommandCfg.Ranges(**vel_ranges)
  twist.stair_ranges = UniformVelocityCommandCfg.Ranges(**STAIR_VEL_RANGES)
  twist.stair_terrain_names = STAIR_TERRAIN_NAMES
  twist.corridor_terrain_names = NARROW_TERRAIN_NAMES
  # The operator only holds the flight heading; staying centered (no rail
  # strikes in map3's 0.9 m flights under keyboard control) is the policy's job.
  twist.corridor_lateral_gain = 0.0

  ##
  # Events.
  ##

  reset_base = cfg.events["reset_base"]
  reset_base.params["stair_terrain_names"] = STAIR_TERRAIN_NAMES
  reset_base.params["fixed_yaw_terrain_names"] = UTURN_STAIRS
  reset_base.params["narrow_terrain_names"] = NARROW_TERRAIN_NAMES

  ##
  # Curriculum.
  ##

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, **INITIAL_VEL_RANGES},
      {"step": 1500 * NUM_STEPS_PER_ENV, **FINAL_VEL_RANGES},
    ]

  return cfg


def unitree_go2_depth_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Teacher env plus the front depth camera and the student observation groups.

  Groups: `actor` (teacher input, unchanged), `critic` (unused), `proprio`
  (47), `proprio_history` (47 x 10, term-major, oldest first) and `depth`
  (2 x 60 x 86). The last three are the deployed ONNX inputs of the same name.
  """
  cfg = unitree_go2_depth_teacher_env_cfg(play=play, stairs=True)

  robot = cfg.scene.entities["robot"]
  cfg.scene.entities["robot"] = replace(
    robot, cameras=robot.cameras + (mdp.front_depth_camera_spec_cfg(),)
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    mdp.front_depth_camera_cfg(DEPTH_SENSOR_NAME),
  )

  if not play:
    # Mounting tolerance. Also covers tinynav's native GL depth, which reads
    # ~0.5 px (~0.5 deg pitch) lower than the ray-cast training render.
    camera = SceneEntityCfg("robot", camera_names=(mdp.FRONT_DEPTH_CAMERA,))
    cfg.events["depth_camera_pose"] = EventTermCfg(
      mode="reset",
      func=dr.cam_quat,
      params={
        "roll_range": (-math.radians(1.0), math.radians(1.0)),
        "pitch_range": (-math.radians(2.0), math.radians(2.0)),
        "yaw_range": (-math.radians(1.0), math.radians(1.0)),
        "asset_cfg": camera,
      },
    )
    cfg.events["depth_camera_pos"] = EventTermCfg(
      mode="reset",
      func=dr.cam_pos,
      params={
        "ranges": (-0.01, 0.01),
        "operation": "add",
        "asset_cfg": deepcopy(camera),
      },
    )

  actor = cfg.observations["actor"]
  proprio_terms = {}
  for name in PROPRIO_TERM_NAMES:
    term = deepcopy(actor.terms[name])
    term.history_length = 0
    proprio_terms[name] = term
  cfg.observations["proprio"] = ObservationGroupCfg(
    terms=proprio_terms,
    concatenate_terms=True,
    enable_corruption=not play,
  )
  cfg.observations["proprio_history"] = ObservationGroupCfg(
    terms=deepcopy(proprio_terms),
    concatenate_terms=True,
    enable_corruption=not play,
    history_length=ACTOR_HISTORY_LENGTH,
  )
  depth_params = {"sensor_name": DEPTH_SENSOR_NAME}
  if not play:
    depth_params |= {"stale_prob": 0.2, "noise_std": 0.01, "dropout_prob": 0.01}
  cfg.observations["depth"] = ObservationGroupCfg(
    terms={
      "depth": ObservationTermCfg(func=mdp.depth_history, params=depth_params),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  return cfg
