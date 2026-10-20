"""Depth-camera velocity configurations (teacher-student), any LeggedTaskSpec.

The teacher is the blind task with a privileged height scan in the actor
and map3-like walled stairs in the terrain mix. The student sees what the
tinynav-sim robot sees: proprioception (with a 10-step history) and two
10 Hz frames from the front depth camera. It is distilled from the teacher.

Rewards, events, domain randomization, collisions and terminations are the
go2-blind ones; only terrain, commands and observations change. Developed on
the Go2 (see config/go2); other robots reuse it through their spec.
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
from src.tasks.velocity.config.legged.blind import (
  ACTOR_HISTORY_LENGTH,
  NUM_STEPS_PER_ENV,
  make_blind_env_cfg,
)
from src.tasks.velocity.config.legged.blind import (
  FINAL_VEL_RANGES as BLIND_FINAL_VEL_RANGES,
)
from src.tasks.velocity.config.legged.blind import (
  INITIAL_VEL_RANGES as BLIND_INITIAL_VEL_RANGES,
)
from src.tasks.velocity.config.legged.spec import LeggedTaskSpec
from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

# Proprioceptive actor terms, in the order the deployed policy receives them
# (wheeled robots on go2-blind's recipe have no `phase`; only the heading tasks
# have `heading_error`).
PROPRIO_TERM_NAMES = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "phase",
  "joint_pos",
  "joint_vel",
  "actions",
  "heading_error",
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
# go2-blind's stairs commands (the heading tasks).
BLIND_STAIR_VEL_RANGES = {
  "lin_vel_x": (0.2, 0.8),
  "lin_vel_y": (-0.2, 0.2),
  "ang_vel_z": (-0.3, 0.3),
}

DEPTH_SENSOR_NAME = "front_depth"
TERRAIN_TARGET_GROUP = "terrain_target"
TERRAIN_TARGET_CLIP = (-0.5, 1.5)

SCAN_Z_OFFSET = 8.0
FOOT_SCAN_NAME = "foot_scan"
WALL_SCAN_NAME = "wall_scan"
SCAN_WIDTH = 1.2
SCAN_CLIP = (-1.0, 2.0)
SCAN_SCALE = 1.0 / 5.0  # go2-blind's 1 / terrain_scan.max_distance.


def make_depth_terrains_cfg(spec: LeggedTaskSpec, stairs: bool) -> TerrainGeneratorCfg:
  """go2-blind flat/rough/pyramid mix plus walled map3-like stairs."""
  sub_terrains: dict[str, terrain_gen.SubTerrainCfg] = {
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.20),
  }
  if stairs:
    pyramid = dict(
      proportion=0.10,
      step_height_range=spec.step_height_range,
      step_width=spec.pyramid_step_width,
      platform_width=2.0,
      border_width=0.5,
    )
    walled = dict(step_height_range=spec.step_height_range)
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


def make_depth_teacher_env_cfg(
  spec: LeggedTaskSpec, play: bool = False, stairs: bool = True, heading: bool = False
) -> ManagerBasedRlEnvCfg:
  """The blind task with a height scan in the actor and the map3-like terrains.

  The actor keeps the blind 10-step proprio history and adds the current
  height scan (no history), so its first dims match the student's
  `proprio_history` layout. With `heading`, go2-blind's straight walking and
  heading-error input (see apply_depth_heading).
  """
  cfg = make_blind_env_cfg(spec, play=play, stairs=stairs)

  ##
  # Terrain.
  ##

  assert cfg.scene.terrain is not None
  terrain_generator = make_depth_terrains_cfg(spec, stairs)
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
  feet = SceneEntityCfg("robot", site_names=spec.foot_site_names)
  # Lifting higher over stairs is fine; only too-low swings are penalized there.
  cfg.rewards["foot_swing_clearance"].params["scan_sensor_name"] = "terrain_scan"
  cfg.rewards["foot_terrain_clearance"] = RewardTermCfg(
    func=mdp.feet_terrain_clearance,
    weight=spec.terrain_clearance_weight,
    params={
      "sensor_name": "feet_ground_contact",
      "scan_sensor_name": FOOT_SCAN_NAME,
      "foot_radius": spec.foot_radius,
      "margin": spec.terrain_clearance_margin,
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
      "overspeed_range": spec.foothold_overspeed_range,
    },
  )
  # The policy keeps itself off the walls; corridor commands only hold heading.
  cfg.rewards["wall_proximity"] = RewardTermCfg(
    func=mdp.wall_proximity,
    weight=-20.0,
    params={"sensor_name": WALL_SCAN_NAME, "safe_distance": spec.wall_safe_distance},
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
      "overspeed_range": spec.foothold_overspeed_range,
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
    if name in actor.terms:
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
  # go2-blind's wheeled robots train on its wider blind ranges (back to -1.0,
  # forward to 1.5 m/s). On the tinynav-sim envelope the AS2-W, B2-W and Go2-W
  # stepped when backing up at -0.4 and the first two at 1.0 m/s, both near
  # its edges and seldom drawn.
  if spec.wheel_blind_recipe:
    initial_ranges, final_ranges = BLIND_INITIAL_VEL_RANGES, BLIND_FINAL_VEL_RANGES
  else:
    initial_ranges, final_ranges = INITIAL_VEL_RANGES, FINAL_VEL_RANGES
  vel_ranges = final_ranges if play else initial_ranges
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
      {"step": 0, **initial_ranges},
      {"step": 1500 * NUM_STEPS_PER_ENV, **final_ranges},
    ]

  if heading:
    apply_depth_heading(cfg, play=play, stairs=stairs)
    if stairs and spec.wheel_blind_recipe:
      apply_depth_wheel_stairs(cfg, spec)
  return cfg


def apply_depth_wheel_stairs(cfg: ManagerBasedRlEnvCfg, spec: LeggedTaskSpec) -> None:
  """go2-blind's wheeled stairs terms (make_wheeled_blind_env_cfg with stairs,
  d9ed8ba) on a wheeled heading teacher; its Go2W / As2W walked 85 / 78% of
  map3's flights without an operator, where our wheeled teachers climbed
  slowly (B2-W 0.83x) or not at all:
  - wheel_riser_lift (0.5): pays for lifting a wheel that a riser blocks; its
    first Go2W stairs pilot without it stalled at the first riser;
  - linear tracking whose std shrinks with the commanded speed (min_std 0.2),
    so stalling costs most of it at slow stairs commands too;
  - wheel_air_spin (-0.01): lifted wheels spinning;
  - foot friction 0.4-1.0: higher friction let a spinning wheel climb a riser
    face, a simulator exploit;
  - no foot_stumble (a wheel finds a riser by bumping into it) and no wheel
    edge-clearance rule (it penalized partly lifted wheels, not unlifted ones);
  - the climbing progress reward without the overspeed gate, as there;
  - stair_speed_deficit (-1, its ea6aae8): going slower than 0.9x the command
    on stairs; its Go2W / As2W / B2W went down map3's flights at 0.57 / 0.66 /
    0.84x the command, 0.80 / 0.80 / 1.02x after 300 iterations with it.
  The descent progress reward stays too, with its gate: go2-blind's wheeled
  robots have none, ours ungated rushed down at 1.35-5x and gated went 1.09x
  (B2-W full_v5ft); above 0.9x the command only it still pays for going down.
  """
  rewards = cfg.rewards
  rewards.pop("foot_stumble", None)
  rewards.pop("foot_terrain_clearance", None)
  rewards["track_linear_velocity"].params["min_std"] = 0.2
  rewards["wheel_riser_lift"] = RewardTermCfg(
    func=mdp.wheel_riser_lift,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "scan_sensor_name": "terrain_scan",
      "wheel_radius": spec.wheel_radius,
      "radius": spec.wheel_radius + 0.1,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=spec.foot_site_names),
    },
  )
  rewards["wheel_air_spin"] = RewardTermCfg(
    func=mdp.wheel_air_spin,
    weight=-0.01,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", joint_names=(spec.wheel_joints,)),
    },
  )
  rewards["stair_speed_deficit"] = RewardTermCfg(
    func=mdp.stair_speed_deficit,
    weight=-1.0,
    params={"command_name": "twist", "sensor_name": "terrain_scan"},
  )
  rewards["foothold_height_gain"].params["overspeed_range"] = None
  cfg.events["foot_friction"].params["ranges"] = (0.4, 1.0)


def apply_depth_heading(cfg: ManagerBasedRlEnvCfg, play: bool, stairs: bool) -> None:
  """go2-blind's straight walking without an operator and heading-error input
  (its d9ed8ba, 2026-10-20), on a depth teacher config.

  Its blind policies drifted ~11 deg per 10 s walking straight and, without
  the operator, hit map3's flight walls; with these its Go2 / As2 / B2 walked
  the 18 map3 flights without an operator 87 / 85 / 93% of the time and
  drifted +-2-5 deg per 10 s at 0.5 m/s on flat ground. Changes:
  - go2-blind's command ranges for every robot (back to -1.0, forward to 1.5
    m/s; tinynav-sim's keyboard is a share of the policy's range) and its
    stairs ranges;
  - half of the walled-flight commands walk straight with no operator, lined
    up with the flight at the spawn (yaw +-3 instead of 15 deg); half of the
    stairs commands walk straight (vy = wz = 0); 0.1 more of the flat commands
    are vx only;
  - straight_drift (-1.0): heading change since the start of a forward or
    backward command without turning, up to 1.0 m/s;
  - heading_error (target heading = integral of the commanded yaw rate, minus
    the base heading, +-0.5 rad) last in the actor (with the proprio history,
    so in the student's proprio groups too) and critic, so a trained teacher
    widens with zero weights (scripts/go2_depth/widen_obs.py). On the robot it
    comes from the IMU yaw (noise +-0.02 rad here).
  Train it with MirrorVelocityOnPolicyRunner (left-right mirror loss).
  """
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.TerrainAwareVelocityCommandCfg)
  vel_ranges = BLIND_FINAL_VEL_RANGES if play else BLIND_INITIAL_VEL_RANGES
  twist.ranges = UniformVelocityCommandCfg.Ranges(**vel_ranges)
  twist.stair_ranges = UniformVelocityCommandCfg.Ranges(**BLIND_STAIR_VEL_RANGES)
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, **BLIND_INITIAL_VEL_RANGES},
      {"step": 1500 * NUM_STEPS_PER_ENV, **BLIND_FINAL_VEL_RANGES},
    ]
  stand, vx, vy, wz, combined = twist.flat_mode_probs
  twist.flat_mode_probs = (stand, vx + 0.1, vy, wz, combined - 0.1)
  if stairs:
    twist.stair_straight_prob = 0.5
    twist.corridor_operator_prob = 0.5
    cfg.events["reset_base"].params["narrow_yaw_noise"] = math.radians(3.0)
  cfg.rewards["straight_drift"] = RewardTermCfg(
    func=mdp.straight_drift,
    weight=-1.0,
    params={
      "command_name": "twist",
      "lateral_scale": None,
      "max_speed": 1.0,
      "forward_only": True,
    },
  )
  params = {"command_name": "twist"}
  cfg.observations["actor"].terms["heading_error"] = ObservationTermCfg(
    func=mdp.heading_error,
    params=dict(params),
    noise=Unoise(n_min=-0.02, n_max=0.02),
    history_length=ACTOR_HISTORY_LENGTH,
  )
  cfg.observations["critic"].terms["heading_error"] = ObservationTermCfg(
    func=mdp.heading_error, params=dict(params)
  )


def make_depth_student_env_cfg(
  spec: LeggedTaskSpec, play: bool = False, heading: bool = False
) -> ManagerBasedRlEnvCfg:
  """Teacher env plus the front depth camera and the student observation groups.

  Groups: `actor` (teacher input, unchanged), `critic` (unused), `proprio`
  (47 for 12 joints, 48 with the heading error), `proprio_history` (x 10, term-major, oldest first) and
  `depth` (2 x 60 x 86). The last three are the deployed ONNX inputs of the
  same name.
  """
  cfg = make_depth_teacher_env_cfg(spec, play=play, stairs=True, heading=heading)

  robot = cfg.scene.entities["robot"]
  camera_cfg = mdp.front_depth_camera_spec_cfg(
    pos=spec.camera_pos, quat=spec.camera_quat, body=spec.base_body
  )
  cfg.scene.entities["robot"] = replace(robot, cameras=robot.cameras + (camera_cfg,))
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
    if name not in actor.terms:
      continue
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
  # Training target for the student's terrain reconstruction head (not a
  # policy input): the teacher's height scan without noise, in meters, with
  # walls clipped closer so they don't dominate the loss over the steps.
  terrain_target = deepcopy(cfg.observations["critic"].terms["height_scan"])
  terrain_target.noise = None
  terrain_target.history_length = 0
  terrain_target.clip = TERRAIN_TARGET_CLIP
  terrain_target.scale = None
  cfg.observations[TERRAIN_TARGET_GROUP] = ObservationGroupCfg(
    terms={"height_scan": terrain_target},
    concatenate_terms=True,
    enable_corruption=False,
  )

  return cfg
