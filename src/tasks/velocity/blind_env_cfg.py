"""Blind (proprioception-only) velocity task for Unitree quadrupeds.

The actor sees no height scan, only proprioception with a short history. The
critic keeps privileged terrain information. Commands mimic a joystick on flat
ground and are forward-biased on stairs. Robot-specific inputs (model, collision
names, swing height, payload range) come in a BlindRobotCfg; the robot's base
velocity config must follow the repo conventions: base body `base_link`, foot
sites FR/FL/RR/RL, foot collision geoms `{leg}_foot_collision`.
"""

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, fields, replace

import mjlab.terrains as terrain_gen
from mjlab.actuator import (
  ActuatorCfg,
  BuiltinVelocityActuatorCfg,
  DelayedActuatorCfg,
  IdealPdActuatorCfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.spec_config import CollisionCfg

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.mdp.velocity_command import (
  TerrainAwareVelocityCommandCfg,
  UniformVelocityCommandCfg,
)

FOOT_SITE_NAMES = ("FR", "FL", "RR", "RL")

STAIR_TERRAIN_NAMES = (
  "stairs_down_narrow",
  "stairs_down_wide",
  "stairs_up_narrow",
  "stairs_up_wide",
)

# Joystick full-stick velocities (must match deploy.yaml command ranges).
FINAL_VEL_RANGES = {
  "lin_vel_x": (-1.0, 1.5),
  "lin_vel_y": (-0.6, 0.6),
  "ang_vel_z": (-1.5, 1.5),
}
INITIAL_VEL_RANGES = {
  "lin_vel_x": (-0.6, 1.0),
  "lin_vel_y": (-0.4, 0.4),
  "ang_vel_z": (-1.0, 1.0),
}

ACTOR_HISTORY_LENGTH = 10
NUM_STEPS_PER_ENV = 24  # Must match rl_cfg; converts iterations to env steps.


def blind_collision_cfg(foot: str, calf: str) -> CollisionCfg:
  """All collision geoms on, like the robots' FULL_COLLISION, but calves get
  friction (condim 3) so shins resting on a stair nose don't slide off
  frictionlessly. `foot`/`calf` are regexes of the foot and calf geoms."""
  return CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={foot: 3, calf: 3, ".*_collision": 1},
    priority={foot: 1},
    friction={foot: (0.6,)},
    solimp={foot: (0.9, 0.95, 0.023)},
    contype=1,
    conaffinity=0,
  )


@dataclass(frozen=True)
class BlindRobotCfg:
  """Robot-specific inputs of the blind velocity task."""

  base_env_cfg: Callable[[bool], ManagerBasedRlEnvCfg]
  """The robot's rough-terrain velocity config, called with `play`."""
  robot_cfg: Callable[[], EntityCfg]
  collision: CollisionCfg
  body_contact_patterns: tuple[str, ...]
  """Collision geoms whose ground contact ends the episode (base, hips)."""
  leg_contact_patterns: tuple[str, ...]
  """Collision geoms whose ground contact is penalized (thighs, calves)."""
  swing_height: float = 0.10
  """Target swing-foot clearance above the liftoff/touchdown height (m)."""
  payload_range: tuple[float, float] = (-1.0, 2.0)
  """Mass added to base_link at startup (kg)."""
  nconmax: int | None = None
  """Per-env contact capacity if the base config's is too small. The model is
  checked at qpos0 (joints at zero, legs straight) before the first reset, where
  long legs reach into the ground with several boxes per calf."""
  stair_widths: tuple[float, float] = (0.25, 0.32)
  """Tread widths of the narrow and wide stairs (m)."""
  angular_momentum_weight: float | None = None
  """Override of the whole-body angular momentum penalty (default -0.025). It
  grows with the squared yaw inertia, so heavy robots scale it by (I_Go2 / I)^2
  (Go2 0.53 kg m^2) to keep turning and stepping as cheap as for Go2."""
  swing_weight_stages: tuple[tuple[int, float], ...] | None = None
  """(iteration, weight) stages of the swing clearance penalty (default -3.0
  throughout). A heavy robot's first steps barely lift, and at -3 each such
  landing can cost more than a step of tracking earns."""
  stance_tilt_weight: float | None = None
  """Weight of stance_leg_tilt on the stairs task (off if None): legs on the
  ground keep the default stance's sideways tilt on uneven ground."""


def _stairs(cls, step_width: float) -> terrain_gen.SubTerrainCfg:
  return cls(
    proportion=0.14,
    step_height_range=(0.05, 0.18),
    step_width=step_width,
    platform_width=2.0,
    border_width=0.5,
  )


def make_blind_terrains_cfg(
  stairs: bool, stair_widths: tuple[float, float] = (0.25, 0.32)
) -> TerrainGeneratorCfg:
  """Flat (+ light roughness and up/down pyramid stairs when `stairs`), with
  narrow and wide stairs of the given tread widths."""
  narrow, wide = stair_widths
  sub_terrains: dict[str, terrain_gen.SubTerrainCfg] = {
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.30),
  }
  if stairs:
    sub_terrains |= {
      "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.14,
        noise_range=(0.02, 0.06),
        noise_step=0.01,
        border_width=0.25,
      ),
      # Spawn on the top platform and walk down.
      "stairs_down_narrow": _stairs(terrain_gen.BoxPyramidStairsTerrainCfg, narrow),
      "stairs_down_wide": _stairs(terrain_gen.BoxPyramidStairsTerrainCfg, wide),
      # Spawn in the pit and walk up.
      "stairs_up_narrow": _stairs(terrain_gen.BoxInvertedPyramidStairsTerrainCfg, narrow),
      "stairs_up_wide": _stairs(terrain_gen.BoxInvertedPyramidStairsTerrainCfg, wide),
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


def _velocity_controlled(actuator: ActuatorCfg) -> bool:
  """Wheels: a velocity actuator, or a PD with zero stiffness."""
  if isinstance(actuator, BuiltinVelocityActuatorCfg):
    return True
  return isinstance(actuator, IdealPdActuatorCfg) and actuator.stiffness == 0.0


def _blind_robot_entity(robot: BlindRobotCfg) -> EntityCfg:
  robot_cfg = robot.robot_cfg()
  articulation = robot_cfg.articulation
  assert articulation is not None
  # Per-env lag (0-20 ms) is set at reset by dr.sync_actuator_delays and held
  # for the episode, identical across the actuator groups.
  delayed_actuators = tuple(
    DelayedActuatorCfg(
      base_cfg=actuator,
      delay_target="velocity" if _velocity_controlled(actuator) else "position",
      delay_min_lag=0,
      delay_max_lag=4,
      delay_hold_prob=1.0,
    )
    for actuator in articulation.actuators
  )
  return replace(
    robot_cfg,
    collisions=(robot.collision,),
    articulation=replace(articulation, actuators=delayed_actuators),
  )


def _robot_contact_sensor(name: str, patterns: tuple[str, ...]) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", entity="robot", pattern=patterns),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


def make_blind_env_cfg(
  robot: BlindRobotCfg,
  play: bool = False,
  stairs: bool = True,
  terrain_clearance: bool = True,
  foothold_gain: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create the blind velocity configuration for a quadruped.

  Args:
    robot: Robot-specific inputs.
    play: Evaluation settings (no noise or pushes, random terrain per reset).
    stairs: Include stairs and rough patches; otherwise flat ground only. Both
      variants share observation shapes, sensors and rewards.
    terrain_clearance: Add foot_terrain_clearance and make the swing clearance
      one-sided on uneven ground.
    foothold_gain: Add foothold_height_gain.
    The released Go2 policy was trained with both off up to iteration 6000, with
    terrain_clearance from there and with both from 9000 (doc/go2_blind.md).
  """
  cfg = robot.base_env_cfg(play)

  cfg.scene.entities = {"robot": _blind_robot_entity(robot)}

  # The rough defaults (ccd_iterations=500, njmax=1500) run out of memory at
  # 4096 envs on a 16 GB GPU. Measured peak on the hardest stairs (random
  # actions) is ~56 constraints per env (Go2); collision primitives are simple.
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.njmax = 300
  cfg.sim.contact_sensor_maxmatch = 128
  if robot.nconmax is not None:
    cfg.sim.nconmax = robot.nconmax

  ##
  # Terrain.
  ##

  assert cfg.scene.terrain is not None
  terrain_generator = make_blind_terrains_cfg(stairs, robot.stair_widths)
  if play:
    terrain_generator.num_rows = 5
    terrain_generator.num_cols = 10
    terrain_generator.border_width = 10.0
  cfg.scene.terrain.terrain_generator = terrain_generator
  cfg.scene.terrain.max_init_terrain_level = 3

  ##
  # Sensors.
  ##

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      # Terrain geoms only; the robot's visual meshes (group 2) are under the base.
      sensor.include_geom_groups = (0,)

  # Falls: base or hip touching the ground. Thighs/calves brushing stair edges
  # are penalized instead of terminating.
  body_ground_cfg = _robot_contact_sensor("body_ground_touch", robot.body_contact_patterns)
  leg_ground_cfg = _robot_contact_sensor("leg_ground_touch", robot.leg_contact_patterns)
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "nonfoot_ground_touch"
  ) + (body_ground_cfg, leg_ground_cfg)
  cfg.terminations["illegal_contact"].params["sensor_name"] = body_ground_cfg.name

  ##
  # Observations.
  ##

  actor = cfg.observations["actor"]
  actor.terms.pop("height_scan", None)
  actor.history_length = ACTOR_HISTORY_LENGTH

  critic = cfg.observations["critic"]
  critic.terms["foot_height"] = ObservationTermCfg(
    func=mdp.foot_height_rel_base,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)},
  )

  ##
  # Commands.
  ##

  vel_ranges = FINAL_VEL_RANGES if play else INITIAL_VEL_RANGES
  cfg.commands["twist"] = TerrainAwareVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(2.0, 5.0),
    heading_command=False,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(**vel_ranges),
    stair_terrain_names=STAIR_TERRAIN_NAMES,
  )

  ##
  # Events.
  ##

  cfg.events["reset_base"] = EventTermCfg(
    func=mdp.reset_root_state_terrain_aware,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.0),
        "yaw": (-math.pi, math.pi),
      },
      "stair_terrain_names": STAIR_TERRAIN_NAMES,
      "velocity_range": {},
    },
  )
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["base_payload"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "operation": "add",
      "ranges": robot.payload_range,
    },
  )
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "kp_range": (0.85, 1.15),
      "kd_range": (0.85, 1.15),
      "operation": "scale",
    },
  )
  cfg.events["actuator_delay"] = EventTermCfg(
    mode="reset",
    func=dr.sync_actuator_delays,
    params={"lag_range": (0, 4)},
  )
  if play:
    # Reset events run in insertion order; pick the new terrain tile before
    # placing the robot on it.
    cfg.events = {
      "randomize_terrain": cfg.events.pop("randomize_terrain"),
      **cfg.events,
    }

  ##
  # Rewards.
  ##

  # SceneEntityCfg is resolved in place, so every term gets its own instance.
  scan = {"sensor_name": "terrain_scan"}
  standing = {"command_name": "twist", "command_threshold": 0.1}

  def base_link() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=("base_link",))

  def feet() -> SceneEntityCfg:
    return SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)

  def all_joints() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=(".*",))

  rewards = cfg.rewards
  rewards["track_linear_velocity"].params["z_weight"] = 0.0
  rewards["pose"].func = mdp.variable_posture_terrain_aware
  rewards["pose"].params |= scan
  # Side-stepping swings the hips out; with the Go2 default 0.15 rad tolerance
  # the posture term outweighed tracking and vy reached only ~70-80%.
  for key in ("std_walking", "std_running"):
    rewards["pose"].params[key] = {
      **rewards["pose"].params[key],
      r".*(FR|FL|RR|RL)_hip_joint.*": 0.3,
    }
  rewards["foot_slip"].weight = -0.5
  if robot.angular_momentum_weight is not None:
    rewards["angular_momentum"].weight = robot.angular_momentum_weight

  for name in ("body_orientation_l2", "foot_clearance", "stand_still"):
    rewards.pop(name)

  rewards["body_orientation"] = RewardTermCfg(
    func=mdp.body_orientation_terrain_aware,
    weight=-1.0,
    params={**scan, "uneven_scale": 0.3, "asset_cfg": base_link()},
  )
  rewards["lin_vel_z_even"] = RewardTermCfg(
    func=mdp.lin_vel_z_even, weight=-2.0, params=dict(scan)
  )
  # At -0.5 this term was ~3% of the tracking reward and feet lifted only 3-6 cm.
  swing_params = {
    "sensor_name": "feet_ground_contact",
    "target_height": robot.swing_height,
    **standing,
    "asset_cfg": feet(),
  }
  if terrain_clearance:
    swing_params["scan_sensor_name"] = "terrain_scan"
  rewards["foot_swing_clearance"] = RewardTermCfg(
    func=mdp.feet_swing_clearance, weight=-3.0, params=swing_params
  )
  if robot.swing_weight_stages is not None:
    rewards["foot_swing_clearance"].weight = robot.swing_weight_stages[0][1]
  if terrain_clearance:
    # Without this, robots stalled at up-risers >= 12 cm while trotting in place
    # with the flat-ground 9 cm swing: nothing rewarded lifting a blocked foot.
    rewards["foot_terrain_clearance"] = RewardTermCfg(
      func=mdp.feet_terrain_clearance,
      weight=-40.0,
      params={
        "sensor_name": "feet_ground_contact",
        "scan_sensor_name": "terrain_scan",
        "asset_cfg": feet(),
      },
    )
  if foothold_gain:
    # With only foot_terrain_clearance, robots stalled at >= 12 cm risers backed
    # away instead of retrying: at tracking std 0.35, stalling and retreating
    # score almost the same. This pays for each foot reaching a higher foothold.
    rewards["foothold_height_gain"] = RewardTermCfg(
      func=mdp.feet_foothold_height_gain,
      weight=2.0,
      params={"sensor_name": "feet_ground_contact", **standing, "asset_cfg": feet()},
    )
  rewards["foot_stumble"] = RewardTermCfg(
    func=mdp.feet_stumble,
    weight=-0.5,
    params={"sensor_name": "feet_ground_contact"},
  )
  rewards["leg_ground_contact"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": leg_ground_cfg.name, "force_threshold": 5.0},
  )
  if stairs and robot.stance_tilt_weight is not None:
    # Climbing, the Go2 release opened its front hips 7 deg past the default
    # stance once the feet were down (As2 at 1000 iterations: 12 deg, and its
    # rear hips 10 deg going down); swinging legs stay free to open.
    rewards["stance_leg_tilt"] = RewardTermCfg(
      func=mdp.stance_leg_tilt,
      weight=robot.stance_tilt_weight,
      params={
        "sensor_name": "feet_ground_contact",
        "scan_sensor_name": "terrain_scan",
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=tuple(f"{n}_hip_joint" for n in FOOT_SITE_NAMES),
          body_names=tuple(f"{n}_thigh" for n in FOOT_SITE_NAMES),
        ),
      },
    )
  rewards["stand_still_even"] = RewardTermCfg(
    func=mdp.stand_still_even,
    weight=-1.0,
    params={**standing, **scan, "asset_cfg": all_joints()},
  )
  rewards["standing_joint_vel"] = RewardTermCfg(
    func=mdp.standing_joint_vel,
    weight=-0.02,
    params={**standing, "asset_cfg": all_joints()},
  )
  rewards["standing_feet_contact"] = RewardTermCfg(
    func=mdp.standing_feet_contact,
    weight=0.5,
    params={**standing, "sensor_name": "feet_ground_contact"},
  )

  ##
  # Curriculum.
  ##

  if play:
    cfg.curriculum = {}
  else:
    cfg.curriculum = {
      "command_vel": CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
          "command_name": "twist",
          "velocity_stages": [
            {"step": 0, **INITIAL_VEL_RANGES},
            {"step": 1500 * NUM_STEPS_PER_ENV, **FINAL_VEL_RANGES},
          ],
        },
      ),
      "track_lin_vel_std": CurriculumTermCfg(
        func=mdp.reward_params,
        params={
          "reward_name": "track_linear_velocity",
          "param_stages": [{"step": 2000 * NUM_STEPS_PER_ENV, "params": {"std": 0.35}}],
        },
      ),
      "track_ang_vel_std": CurriculumTermCfg(
        func=mdp.reward_params,
        params={
          "reward_name": "track_angular_velocity",
          "param_stages": [{"step": 2000 * NUM_STEPS_PER_ENV, "params": {"std": 0.45}}],
        },
      ),
    }
    if robot.swing_weight_stages is not None:
      cfg.curriculum["swing_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
          "reward_name": "foot_swing_clearance",
          "weight_stages": [
            {"step": it * NUM_STEPS_PER_ENV, "weight": w} for it, w in robot.swing_weight_stages
          ],
        },
      )
    if stairs:
      cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=mdp.terrain_levels_cmd_distance,
        params={"command_name": "twist"},
      )

  return cfg


##
# Wheel-legged robots (Go2W, B2W, As2W): wheels at the feet.
##


@dataclass(frozen=True)
class WheeledBlindRobotCfg(BlindRobotCfg):
  """A BlindRobotCfg whose feet are velocity-controlled wheels.

  The wheel collision geoms are the robot's `{leg}_foot_collision` and the foot
  sites sit at the wheel centers.
  """

  leg_joints: str = r".*_(hip|thigh|calf)_joint"
  wheel_joints: str = r".*_wheel_joint"
  wheel_bodies: str = r"[FR][LR]_wheel(_link)?"
  wheel_radius: float = 0.086
  """Radius of the wheel collision sphere (m)."""
  wheel_action_scale: float = 5.0
  """Wheel velocity target per unit of action (rad/s)."""
  drivable_air_weight: float = -1.0
  """Weight of wheel_air_drivable (lifted wheels while driving straight; on the
  stairs tasks on even ground only)."""


def make_wheeled_blind_env_cfg(
  robot: WheeledBlindRobotCfg, play: bool = False, stairs: bool = False
) -> ManagerBasedRlEnvCfg:
  """Blind velocity configuration for a wheel-legged quadruped.

  Legs take position targets (scale and offset as for the legged robots) and
  wheels velocity targets. The policy sees joints in the order legs (FL, FR, RL,
  RR x hip, thigh, calf) then wheels (FL, FR, RL, RR), which is also the action
  order; wheel angles are left out of the observation (unbounded). There is no
  gait clock: wheels roll on even ground and the legs step only when they must,
  so the gait and foot-slip terms of the legged task are off (foot sites are the
  wheel centers, which move while rolling). With nothing else in their place,
  the first flat policies (2026-10-18) moved sideways and turned in place with a
  shuffle: wheels lifted ~1 cm and grounded wheels slid sideways. So the legged
  swing-clearance term stays (a rolling wheel has no landings), and wheel_slip
  penalizes grounded wheels skidding, the way around it. With these, the B2W
  fine-tune went on to step while driving straight too (a 10 cm step costs
  nothing), so wheel_air_drivable penalizes lifted wheels while the command is
  to drive (nearly) straight; turns are free to step.

  With `stairs`, a wheel can roll over an edge lower than about half its radius
  but must be lifted over a higher one. There is still no gait clock, so the
  legs step only where they must, driven by:
  - wheel_riser_lift: pays for lifting a wheel that a riser blocks. The first
    pilot without it (2026-10-18) went down every flight and up none, stalling
    at the first riser even at 5 cm.
  - Tracking whose std shrinks with the commanded speed, so stalling costs
    most of the tracking reward at slow stair commands too.
  - More up-stair than down-stair columns.
  The legged foot_terrain_clearance (it penalized partly lifted wheels, not
  unlifted ones) and foot_stumble (a blind wheel finds a riser by bumping into
  it) are off. foothold_height_gain stays; wheel_slip / wheel_air_spin stop
  wheels from spinning against risers or in the air. foot_swing_clearance and
  wheel_air_drivable apply on even ground only (a wheel rolling off a down-stair
  edge lands after a zero-clearance swing): without them the second stairs pilot
  (2026-10-19) climbed every flight but turned in place at 44% of 0.5 rad/s on
  flat ground.
  """
  cfg = make_blind_env_cfg(
    robot, play=play, stairs=stairs, terrain_clearance=False, foothold_gain=stairs
  )
  if stairs:
    # 7 up and 4 down of 20 columns (was 5 and 6).
    sub_terrains = cfg.scene.terrain.terrain_generator.sub_terrains
    for name in ("stairs_up_narrow", "stairs_up_wide"):
      sub_terrains[name].proportion = 0.18
    for name in ("stairs_down_narrow", "stairs_down_wide"):
      sub_terrains[name].proportion = 0.10

  def legs() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=(robot.leg_joints,))

  def legs_then_wheels() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot", joint_names=(robot.leg_joints, robot.wheel_joints), preserve_order=True
    )

  ##
  # Actions.
  ##

  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  joint_pos.actuator_names = (robot.leg_joints,)
  # Zero wheel targets at the zero command once the wheels roll slower than 0.1 m/s
  # (a parking brake, also on the deploy side; see BrakedJointVelocityAction).
  cfg.actions["wheel_vel"] = mdp.BrakedJointVelocityActionCfg(
    entity_name="robot",
    actuator_names=(robot.wheel_joints,),
    scale=robot.wheel_action_scale,
    command_name="twist",
    brake_speed=0.1 / robot.wheel_radius,
  )

  ##
  # Observations.
  ##

  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms.pop("phase", None)
    terms["joint_pos"].params["asset_cfg"] = legs()
    terms["joint_vel"].params["asset_cfg"] = legs_then_wheels()

  ##
  # Rewards.
  ##

  rewards = cfg.rewards
  for name in ("foot_gait", "foot_slip"):
    rewards.pop(name)
  # Wheel angles have no default or limits; posture terms cover the legs.
  rewards["pose"].params["asset_cfg"] = legs()
  rewards["joint_pos_limits"].params["asset_cfg"] = legs()
  rewards["stand_still_even"].params["asset_cfg"] = legs()
  if stairs:
    rewards["foot_swing_clearance"].params |= {
      "scan_sensor_name": "terrain_scan",
      "even_only": True,
    }
    rewards.pop("foot_stumble")
    rewards["track_linear_velocity"].params["min_std"] = 0.2
    # Weight 0.5: at most 0.5 per wheel and riser, 2 per step climbed. The foot
    # site is the wheel center; the scan radius reaches the next tread from a
    # wheel against its riser (scan spacing 0.1 m).
    rewards["wheel_riser_lift"] = RewardTermCfg(
      func=mdp.wheel_riser_lift,
      weight=0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "scan_sensor_name": "terrain_scan",
        "wheel_radius": robot.wheel_radius,
        "radius": robot.wheel_radius + 0.1,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
        "max_speed_ratio": 1.1,
      },
    )
  # For the flat shuffle this cost ~0.04 per second, but dragging all wheels
  # sideways at 0.6 m/s costs ~0.7: it keeps foot_swing_clearance from being
  # dodged by not lifting at all.
  rewards["wheel_slip"] = RewardTermCfg(
    func=mdp.wheel_slip,
    weight=-0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "wheel_radius": robot.wheel_radius,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=FOOT_SITE_NAMES, body_names=(robot.wheel_bodies,)
      ),
    },
  )
  if stairs:
    rewards["wheel_air_spin"] = RewardTermCfg(
      func=mdp.wheel_air_spin,
      weight=-0.01,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", joint_names=(robot.wheel_joints,)),
      },
    )
  # The B2W fine-tune stepped at 0.5-1.5 m/s straight ahead with its wheels
  # down 58-60% of the time; that trot costs ~0.4 per second here. Turning
  # counts as drivable only up to 0.1 rad/s: wheels cannot steer, and with
  # lifts penalized up to 0.5 rad/s B2W barely turned below 0.3 rad/s (10% of
  # 0.2 rad/s at 1 m/s) instead of skidding or stepping.
  drivable_params = {
    "sensor_name": "feet_ground_contact",
    "command_name": "twist",
    "max_ang_vel_z": 0.1,
  }
  if stairs:
    drivable_params["scan_sensor_name"] = "terrain_scan"
  rewards["wheel_air_drivable"] = RewardTermCfg(
    func=mdp.wheel_air_drivable, weight=robot.drivable_air_weight, params=drivable_params
  )
  # Turning while driving takes steps. At std 0.45 not turning at 0.15 rad/s
  # cost 0.1 per second, and the fine-tunes often turned at under 40% of
  # 0.15-0.2 rad/s at 1-1.5 m/s, at other commands per checkpoint; at std 0.2
  # it costs 0.43.
  rewards["track_angular_velocity"].params["min_std"] = 0.2
  # Standing still: velocity-controlled wheels hold no position at zero speed,
  # and the deployed flat policies (2026-10-19) crept 2-3 cm (Go2W, As2W) and
  # 5-6 cm (B2W, also shuffling) per 4 s at the zero command, where the squared
  # standing terms cost ~0.0005 per second. At these weights that creep costs
  # 0.3 (Go2W) to 0.6 (B2W) per second.
  standing = {"command_name": "twist", "command_threshold": 0.1}
  rewards["standing_position_hold"] = RewardTermCfg(
    func=mdp.standing_position_hold, weight=-10.0, params={**standing}
  )
  rewards["standing_speed"] = RewardTermCfg(
    func=mdp.standing_speed, weight=-2.0, params={**standing}
  )
  rewards["standing_feet_air"] = RewardTermCfg(
    func=mdp.standing_feet_air,
    weight=-1.0,
    params={**standing, "sensor_name": "feet_ground_contact"},
  )
  # The creep came mostly from the policy's own wheel targets (Go2W: 0.63 rad/s
  # mean |target| per wheel at the zero command, right wheels opposed): at this
  # weight that costs 0.25 per second.
  rewards["standing_wheel_target"] = RewardTermCfg(
    func=mdp.standing_wheel_target, weight=-0.1, params={**standing}
  )

  ##
  # Commands.
  ##

  # Driving with a gentle turn, the joystick's most common use, was 0.4% of
  # the flat commands (the combined mode draws all three axes uniformly), and
  # the deployed policies turned 16-80% of such commands, worse to one side.
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.TerrainAwareVelocityCommandCfg)
  twist.steer_prob = 0.2

  ##
  # Events.
  ##

  if stairs:
    # High wheel friction lets a spinning wheel climb a riser face, a simulator
    # exploit (Chamorro et al. 2024); rubber on floors is ~0.4-1.0.
    cfg.events["foot_friction"].params["ranges"] = (0.4, 1.0)

  ##
  # Terminations.
  ##

  # Safety net: an env whose physics state turns NaN/Inf is reset instead of
  # stopping the run (rsl_rl raises on NaN observations). Logged as
  # Episode_Termination/nan; it should stay at 0.
  cfg.terminations["nan"] = TerminationTermCfg(func=mdp.nan_detection)

  return cfg


##
# map3 fine-tune: walled flights like the tinynav-sim map3 stairwell.
##

WALLED_TERRAIN_NAMES = ("walled_down", "walled_up")
MAP3_STAIR_TERRAIN_NAMES = STAIR_TERRAIN_NAMES + WALLED_TERRAIN_NAMES


def make_blind_map3_terrains_cfg() -> TerrainGeneratorCfg:
  """The blind mix with less pyramid area plus walled map3-like flights.

  Walled flights are 0.85-1.5 m wide (narrower with difficulty), with treads
  of 0.18-0.32 m and per-step rise noise; map3 has 0.19 m treads and
  16-19 cm rises where the blind policy got stuck. Proportions are multiples
  of 1/20 so each type gets whole columns. Column order puts flat ground next
  to the walled tiles: robots leaving a pyramid tile walk straight on and hit
  the (to them invisible) walls of the next column.
  """
  cfg = make_blind_terrains_cfg(stairs=True)
  subs = cfg.sub_terrains
  proportions = {
    "stairs_down_narrow": 0.05,
    "stairs_down_wide": 0.05,
    "stairs_up_narrow": 0.10,
    "stairs_up_wide": 0.05,
    "random_rough": 0.10,
    "flat": 0.25,
  }
  cfg.sub_terrains = {name: replace(subs[name], proportion=p) for name, p in proportions.items()}
  cfg.sub_terrains["walled_down"] = mdp.WalledCrossStairsTerrainCfg(proportion=0.20)
  cfg.sub_terrains["walled_up"] = mdp.WalledCrossStairsTerrainCfg(
    proportion=0.20, inverted=True
  )
  return cfg


def apply_blind_map3(
  cfg: ManagerBasedRlEnvCfg, play: bool, legged_terms: bool = True
) -> ManagerBasedRlEnvCfg:
  """Turn a blind stairs config into its map3 fine-tune (same observations and
  actions).

  Changes: walled flights in the terrain (with an operator-style corridor
  command and little spawn x/y noise there); without `legged_terms` (wheeled) a
  penalty for going slower than 0.9x the command on stairs (stair_speed_deficit),
  and with `legged_terms` a tighter
  linear tracking std on even ground plus more slow flat commands (flat
  low-speed under-tracking) and a progress reward for going down stairs. Built
  for the Go2 (release map3_ft). The wheeled stairs tasks keep their own
  tracking and commands (legged_terms=False): the go2-depth branch's wheeled
  robots with an ungated descent progress reward rushed down the map3 flights
  at 1.35-5x the command (2026-10-20)."""
  assert cfg.scene.terrain is not None
  terrain_generator = make_blind_map3_terrains_cfg()
  if play:
    terrain_generator.num_rows = 5
    terrain_generator.num_cols = 20
    terrain_generator.border_width = 10.0
  cfg.scene.terrain.terrain_generator = terrain_generator

  twist = cfg.commands["twist"]
  assert isinstance(twist, TerrainAwareVelocityCommandCfg)
  twist.stair_terrain_names = MAP3_STAIR_TERRAIN_NAMES
  twist.corridor_terrain_names = WALLED_TERRAIN_NAMES

  reset_base = cfg.events["reset_base"]
  reset_base.params["stair_terrain_names"] = MAP3_STAIR_TERRAIN_NAMES
  reset_base.params["narrow_terrain_names"] = WALLED_TERRAIN_NAMES
  if not legged_terms:
    # Go2W, As2W and B2W (2026-10-20) went down the map3 flights at 0.5-0.7x the
    # command; the shortfall below 0.9x costs up to 1 per step on stairs.
    cfg.rewards["stair_speed_deficit"] = RewardTermCfg(
      func=mdp.stair_speed_deficit,
      weight=-1.0,
      params={"command_name": "twist", "sensor_name": "terrain_scan"},
    )
    return cfg

  twist.flat_slow_prob = 0.3
  rewards = cfg.rewards
  # At std 0.35 a 30% shortfall at 0.3 m/s cost only 6% of the tracking reward.
  rewards["track_linear_velocity"].params |= {
    "even_std": 0.25,
    "scan_sensor_name": "terrain_scan",
  }
  # Descents tracked ~75% of the command, too slow for map3 flights.
  rewards["foothold_height_drop"] = RewardTermCfg(
    func=mdp.feet_foothold_height_gain,
    weight=2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "descend": True,
    },
  )
  return cfg


##
# Straight walking without an operator (2026-10-20).
##


def apply_blind_straight(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Make a map3 stairs config walk straight under a command without turning,
  with no operator; the corridor operator stays an optional correction.

  The blind policies (2026-10-20) drifted ~11 deg per 10 s walking straight on
  flat ground and, without the operator, hit the walls on 41-69% (As2) and
  20-50% (Go2) of the runs on map3's flights: the operator was in every walled
  training command, open-stairs commands turn at random, and the tracking
  kernels hardly see a small steady yaw rate. Changes:
  - half of the walled-flight commands walk straight with no operator, and
    walled-flight spawns are lined up with the flight (yaw +-3 instead of 15 deg);
  - half of the open-stairs commands walk straight (vy = wz = 0), and the flat
    vx-only mode takes 0.1 more of the flat commands from the combined mode;
  - straight_drift (-1.0): heading change since the start of a forward or
    backward command without turning, up to 1.0 m/s (stairs and corridor
    speeds). Its sideways-offset term is left out: that offset grows with the
    distance walked, and with it (weight -0.5) As2 capped its speed at ~0.9 m/s
    (1.5 m/s tracked 62%). Applied at all speeds and to sideways commands too,
    Go2 and As2 still slowed to ~76% at 1.5 m/s and ~65-70% sideways on diagonals.
  Train it with MirrorVelocityOnPolicyRunner (left-right mirror loss).
  """
  twist = cfg.commands["twist"]
  assert isinstance(twist, TerrainAwareVelocityCommandCfg)
  assert twist.corridor_terrain_names, "apply_blind_map3 first"
  twist.corridor_operator_prob = 0.5
  twist.stair_straight_prob = 0.5
  stand, vx, vy, wz, combined = twist.flat_mode_probs
  twist.flat_mode_probs = (stand, vx + 0.1, vy, wz, combined - 0.1)
  cfg.events["reset_base"].params["narrow_yaw_noise"] = math.radians(3.0)
  cfg.rewards["straight_drift"] = RewardTermCfg(
    func=mdp.straight_drift,
    weight=-1.0,
    params={"command_name": "twist", "lateral_scale": None, "max_speed": 1.0, "forward_only": True},
  )
  return cfg


def apply_blind_heading(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Add the heading error (target heading = integral of the commanded yaw rate,
  minus the base heading; see mdp.heading_error) to the actor and critic
  observations, appended last so a trained policy widens with zero weights.

  The blind policies cannot tell how far they have turned in total and drifted
  +-8-14 deg over a map3 flight even after straight_drift and the mirror loss
  (As2, 2026-10-20); 0.9 m flights tolerate about +-5 deg. On the robot the
  same quantity comes from the IMU yaw (noise +-0.02 rad here).
  """
  from mjlab.utils.noise import UniformNoiseCfg

  params = {"command_name": "twist"}
  cfg.observations["actor"].terms["heading_error"] = ObservationTermCfg(
    func=mdp.heading_error, params=dict(params), noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02)
  )
  cfg.observations["critic"].terms["heading_error"] = ObservationTermCfg(
    func=mdp.heading_error, params=dict(params)
  )
  return cfg


##
# Second fix round (2026-10-22).
##


@dataclass(frozen=True, kw_only=True)
class BlindFixCfg:
  """Per-robot changes of the second fix round; the defaults change nothing.

  From the diagnostics of the six final models (2026-10-22):
  - Stopped mid-flight on stairs, the Go2W and B2W rolled away on their wheels
    (damping-only brake; see BrakedJointVelocityAction.hold_stiffness).
  - Small commands were under-tracked by the wheeled robots (Go2W turned 56% of
    (1.0, 0, 0.1), went sideways 77% of (0, 0.15, 0)): at the 0.2 std floor that
    shortfall costs ~5% of the tracking reward.
  - Sideways and diagonal commands: As2W 67-80%, Go2 and As2 diagonals 74-80%.
  - Descents ran at 0.76-0.80x the command (Go2W, As2W) and 1.21-1.24x (As2, B2).
  - Go2 hit the first riser of the four map3 flights that start with a 20 cm one
    (18 cm at most in training) with its body: 72-84% with the operator.
  """

  brake_hold_stiffness: float = 0.0
  """Wheel brake hold (Nm/rad; see BrakedJointVelocityAction); wheeled only."""
  min_std: float | None = None
  """Floor of the speed-scaled linear and yaw tracking std where the task has one
  (wheeled stairs tasks: 0.2)."""
  stair_deficit_weight: float | None = None
  """Weight of stair_speed_deficit (wheeled map3 tasks: -1)."""
  descent_max_speed_ratio: float | None = None
  """max_speed_ratio of foothold_height_drop (legged map3 tasks: none)."""
  vy_mode_boost: float = 0.0
  """Share of flat commands moved from the combined to the sideways-only mode."""
  diag_prob: float = 0.0
  """Share of diagonal flat commands (TerrainAwareVelocityCommandCfg.diag_prob)."""
  max_step_height: float | None = None
  """Highest mean rise of the pyramid and walled stairs (0.18 m)."""


def apply_blind_fix(
  cfg: ManagerBasedRlEnvCfg, fix: BlindFixCfg, training: bool = True
) -> ManagerBasedRlEnvCfg:
  """Apply `fix` to a blind config. Without `training` (evaluation tasks of the
  same policy) only the wheel brake changes: it is part of the controller, also
  on the robot, while the other changes only shape training."""
  if fix.brake_hold_stiffness > 0.0:
    wheel_vel = cfg.actions["wheel_vel"]
    assert isinstance(wheel_vel, mdp.BrakedJointVelocityActionCfg)
    wheel_vel.hold_stiffness = fix.brake_hold_stiffness
  if not training:
    return cfg
  rewards = cfg.rewards
  if fix.min_std is not None:
    for name in ("track_linear_velocity", "track_angular_velocity"):
      if rewards[name].params.get("min_std") is not None:
        rewards[name].params["min_std"] = fix.min_std
  if fix.stair_deficit_weight is not None:
    rewards["stair_speed_deficit"].weight = fix.stair_deficit_weight
  if fix.descent_max_speed_ratio is not None:
    rewards["foothold_height_drop"].params["max_speed_ratio"] = fix.descent_max_speed_ratio
  twist = cfg.commands["twist"]
  assert isinstance(twist, TerrainAwareVelocityCommandCfg)
  if fix.vy_mode_boost > 0.0:
    stand, vx, vy, wz, combined = twist.flat_mode_probs
    twist.flat_mode_probs = (stand, vx, vy + fix.vy_mode_boost, wz, combined - fix.vy_mode_boost)
  if fix.diag_prob > 0.0:
    twist.diag_prob = fix.diag_prob
  if fix.max_step_height is not None:
    assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
    for sub in cfg.scene.terrain.terrain_generator.sub_terrains.values():
      if hasattr(sub, "step_height_range"):
        sub.step_height_range = (sub.step_height_range[0], fix.max_step_height)
  return cfg


##
# Hip load (third round, 2026-10-22).
##


@dataclass(frozen=True, kw_only=True)
class BlindHipCfg:
  """Hip-load changes of the third round (user 2026-10-22); see apply_blind_hip.

  Standing and driving on wheels load the hip abduction motors without a break:
  the feet sit outboard of the hip axes, so the body weight keeps a moment on
  them, and Unitree's Go2-W controller overheated them to a thermal shutdown
  (arXiv 2606.21387; the user saw the same on its other robots). Statically
  (2026-10-22) the hip torque falls about linearly as the hips close in, to near
  zero with the feet under the hip axes, while the lateral support (half the
  foot spread over the base height) narrows.
  """

  hip_default: float
  """Outward-positive hip angle of the default stance (rad, mirrored left/right):
  the most closed stance whose static half spread / base height stays >= 0.35
  (0.42 for the narrowest stance in use, Go2 and B2W)."""
  torque_ref: float
  """Static hip torque of the previous default stance (Nm); the torque penalty is
  (tau / torque_ref)^2 averaged over the hips."""
  torque_weight: float = -0.2
  splay_weight: float = -20.0
  splay_deadband: float = 0.05
  """Outward hip abduction (rad) beyond the default stance that costs nothing."""
  splay_inward_deadband: float | None = None
  """Inward hip adduction (rad) inside the default stance that costs nothing;
  beyond it the splay penalty applies too (None: tucking in is free)."""
  wheel_free_splay: float | None = None
  """Wheeled: lifting a wheel is free while its hip is splayed this far (rad)."""
  reset_splay_prob: float = 0.0
  """Share of resets that start with the hips splayed by reset_splay_range (rad)."""
  reset_splay_range: tuple[float, float] = (0.04, 0.15)


def apply_blind_hip(
  cfg: ManagerBasedRlEnvCfg, hip: BlindHipCfg, training: bool = True
) -> ManagerBasedRlEnvCfg:
  """Apply `hip` to a blind config. The default stance changes for evaluation too:
  it sets the policy's joint coordinates (observation, action offset) and the
  deploy default pose; terms keyed to the default stance follow it, e.g.
  stance_leg_tilt, whose stairs target is the default hip angle. The penalties,
  the wheel-lift waiver and the splayed resets only shape training."""
  robot = cfg.scene.entities["robot"]
  joint_pos = {k: v for k, v in robot.init_state.joint_pos.items() if "hip_joint" not in k}
  joint_pos[".*L_hip_joint"] = hip.hip_default
  joint_pos[".*R_hip_joint"] = -hip.hip_default
  robot.init_state = replace(robot.init_state, joint_pos=joint_pos)
  if not training:
    return cfg
  def hips() -> SceneEntityCfg:  # one per term: a resolved config cannot be reused
    return SceneEntityCfg("robot", joint_names=(".*_hip_joint",))

  rewards = cfg.rewards
  rewards["hip_torque"] = RewardTermCfg(
    func=mdp.hip_torque_sq,
    weight=hip.torque_weight,
    params={"actuator_names": (".*_hip_joint",), "torque_ref": hip.torque_ref},
  )
  rewards["hip_splay"] = RewardTermCfg(
    func=mdp.hip_splay,
    weight=hip.splay_weight,
    params={"command_name": "twist", "hip_cfg": hips(), "deadband": hip.splay_deadband},
  )
  if hip.splay_inward_deadband is not None:
    rewards["hip_splay"].params["inward_deadband"] = hip.splay_inward_deadband
  if hip.wheel_free_splay is not None:
    rewards["wheel_air_drivable"].params |= {"free_splay": hip.wheel_free_splay, "hip_cfg": hips()}
  if hip.reset_splay_prob > 0.0:
    # After reset_robot_joints (reset events run in insertion order).
    cfg.events["reset_hip_splay"] = EventTermCfg(
      mode="reset",
      func=mdp.reset_hip_splay,
      params={"prob": hip.reset_splay_prob, "splay_range": hip.reset_splay_range, "asset_cfg": hips()},
    )
  return cfg


##
# Fourth round (2026-10-22).
##


@dataclass(frozen=True, kw_only=True)
class BlindR4Cfg:
  """Fourth-round changes (user 2026-10-22 ~07:30, on the third round's results);
  see apply_blind_r4. With the stance and splay changes in the robots' BlindHipCfg.

  - Descents: the 1.1 cap of the second round, judged on the base speed at the
    instant a foot lands (it lurches forward on every step down), slowed Go2 and
    As2 to 0.63-0.67x the command; judged on each foot's mean speed since its
    previous record instead (feet_foothold_height_gain `step_average`).
  - Wheel brake: a policy holding its wheels just above the brake speed on a riser
    kept releasing the hold (B2W, see BrakedJointVelocityAction).
  """

  descent_step_average: bool = False
  """Legged map3 tasks with a descent cap: judge it on the per-foot step mean."""
  descent_min_gain: float = 0.02
  """With descent_step_average: least drop (m) that makes a new foothold record."""
  brake_after_s: float | None = None
  """Wheeled: BrakedJointVelocityActionCfg.brake_after_s."""
  brake_release_ratio: float | None = None
  """Wheeled: BrakedJointVelocityActionCfg.release_speed_ratio."""


def apply_blind_r4(
  cfg: ManagerBasedRlEnvCfg, r4: BlindR4Cfg, training: bool = True
) -> ManagerBasedRlEnvCfg:
  """Apply `r4` to a blind config. The brake changes for evaluation too (it is part
  of the controller, also on the robot); the descent change only shapes training."""
  if r4.brake_after_s is not None or r4.brake_release_ratio is not None:
    wheel_vel = cfg.actions["wheel_vel"]
    assert isinstance(wheel_vel, mdp.BrakedJointVelocityActionCfg)
    wheel_vel.brake_after_s = r4.brake_after_s
    wheel_vel.release_speed_ratio = r4.brake_release_ratio
  if training and r4.descent_step_average:
    drop = cfg.rewards["foothold_height_drop"]
    assert drop.params.get("max_speed_ratio") is not None, "no descent cap to judge"
    drop.params |= {"step_average": True, "min_gain": r4.descent_min_gain}
  return cfg


##
# Fifth round (2026-10-22).
##


@dataclass(frozen=True, kw_only=True)
class BlindR5Cfg:
  """Fifth-round changes (user 2026-10-22 ~13:30: the hip motors overheat to a
  collapse mostly while the robot stands): a rest stance; see apply_blind_r5.

  Standing, the hip torque falls about linearly as the feet come in under the hip
  axes (step1 stance sweep), and the hip motors cannot be relaxed instead: with
  their stiffness at zero the body weight pushes the feet apart (Go2W 0.18 rad in
  5 s). Walking and driving need the wider stance of the fourth round (narrower,
  As2 and B2 drifted without an operator and the Go2W rolled over backward), but
  standing still does not: on even ground the robot stands with its hips at
  `hip_rest`, the stance whose static half foot spread / base height is 0.30 (the
  walking stances 0.35-0.44), which statically halves the standing hip torque
  again, and the hip torque costs `standing_torque_scale` times more while the
  command is zero.
  """

  hip_rest: float
  """Outward-positive hip angle of the rest stance (rad, mirrored left/right)."""
  standing_torque_scale: float = 3.0
  march_prob: float = 0.0
  """Legged (gait clock): share of zero commands off the stairs that march in place
  instead (user 2026-10-22 ~13:35: step in place when idle or the motors run hot,
  as the Go2-W of arXiv 2606.21387 resets its splayed hips by lifting its feet);
  the robot's controller decides when to march by running the clock."""


def apply_blind_r5(
  cfg: ManagerBasedRlEnvCfg, r5: BlindR5Cfg, training: bool = True
) -> ManagerBasedRlEnvCfg:
  """Apply `r5` to a blind config with the hip-load terms (apply_blind_hip). Only
  shapes training: the policy takes the rest stance itself (the default stance,
  the policy's joint coordinates and the deploy pose stay the walking stance)."""
  if not training:
    return cfg
  rewards = cfg.rewards
  rewards["pose"].params["rest_hip"] = r5.hip_rest
  rewards["stand_still_even"].params["rest_hip"] = r5.hip_rest
  rewards["hip_torque"].params |= {"command_name": "twist", "standing_scale": r5.standing_torque_scale}
  if r5.march_prob > 0.0:
    assert "phase" in cfg.observations["actor"].terms, "marching needs the gait clock"
    cfg.commands["twist"].march_prob = r5.march_prob
  return cfg


##
# Sixth round (2026-10-22).
##


@dataclass(frozen=True, kw_only=True)
class BlindR6Cfg:
  """Sixth-round changes of the legged robots (user 2026-10-22 ~17:30, on the fifth
  round's results); see apply_blind_r6. The wheeled robots' change is their stance:
  the fifth round's rest stance becomes their default (the robots' *_HIP6).

  - Marching in place, the fifth-round policies drifted 41-56 cm in 20 s (Go2,
    As2, B2; the fourth-round ones as much): a position hold while marching.
  - Descents: the step-average cap of 1.1 settled Go2 and As2 at 0.69-0.74x the
    command; go2-depth's As2 / B2 teachers with the same cap at 1.4 descend at
    0.9-1.0x.
  """

  descent_max_speed_ratio: float | None = None
  """New max_speed_ratio of foothold_height_drop (legged map3 tasks with a cap)."""
  march_hold_weight: float = 0.0
  """Weight of march_position_hold (standing_position_hold with `march`)."""
  march_moving_speed: float = 0.25
  """Its moving_speed (m/s): above the sway of the base marching in place (fifth
  round: at most 0.15-0.16 m/s, Go2 and As2; B2 0.09)."""
  march_moving_yaw_rate: float = 0.3
  """Its moving_yaw_rate (rad/s): above the yaw sway marching in place (at most
  0.12-0.14 rad/s)."""


def apply_blind_r6(
  cfg: ManagerBasedRlEnvCfg, r6: BlindR6Cfg, training: bool = True
) -> ManagerBasedRlEnvCfg:
  """Apply `r6` to a blind config with the fifth round's terms (apply_blind_r5).
  Only shapes training."""
  if not training:
    return cfg
  rewards = cfg.rewards
  if r6.descent_max_speed_ratio is not None:
    drop = rewards["foothold_height_drop"]
    assert drop.params.get("max_speed_ratio") is not None, "no descent cap to change"
    drop.params["max_speed_ratio"] = r6.descent_max_speed_ratio
  if r6.march_hold_weight != 0.0:
    assert cfg.commands["twist"].march_prob > 0.0, "no marching to hold"
    rewards["march_position_hold"] = RewardTermCfg(
      func=mdp.standing_position_hold,
      weight=r6.march_hold_weight,
      params={
        "command_name": "twist",
        "march": True,
        "moving_speed": r6.march_moving_speed,
        "moving_yaw_rate": r6.march_moving_yaw_rate,
      },
    )
  return cfg


##
# Seventh round (2026-10-22).
##


@dataclass(frozen=True, kw_only=True)
class BlindR7Cfg:
  """Seventh-round changes (user 2026-10-22 ~20:40 "六项全做", on the sixth round's
  results); see apply_blind_r7.

  - Marching in place drifted as much as before (41-62 cm per 20 s): the sixth
    round's position hold cost ~0.01-0.03 per episode beside ~0.67 for tracking,
    in 2-5 s march commands that drift only a few cm. Five times the weight, and
    march commands held 10-20 s.
  - Standing mid-flight loads the hips about twice as much as on flat ground
    (B2 49-51 Nm front facing down): more zero commands on the stairs and the
    standing hip torque cost doubled there.
  - Descents (Go2, As2) still at 0.77-0.84x the command with the cap at 1.4.
  - Wheeled robots march in place too (user ~20:35 "把轮式踏步列进第七轮"): the
    gait clock appended to their observations (widen the checkpoint, see
    scripts/blind/widen_obs.py), lifting one wheel at a time while told to march,
    and walking sideways costs heading change like walking straight (Go2W and B2W
    turned up to 15-16 deg per 10 s sideways, p90).
  """

  march_hold_weight: float = -50.0
  march_resampling_time_range: tuple[float, float] = (10.0, 20.0)
  stair_standing_prob: float = 0.4
  stairs_torque_scale: float = 2.0
  """hip_torque_sq uneven_scale: standing on uneven ground costs this much more."""
  descent_max_speed_ratio: float | None = None
  wheel_march: bool = False
  """Wheeled: gait clock observation, march_prob 0.5 and a march-only gait reward."""
  wheel_gait_period: float = 1.0
  wheel_gait_offset: tuple[float, ...] = (0.0, 0.5, 0.25, 0.75)
  """One wheel up at a time (a four-beat walk, 25% swing)."""
  wheel_gait_threshold: float = 0.75
  wheel_gait_weight: float = 0.5
  side_drift_weight: float = 0.0
  """Wheeled: straight_drift's heading term for pure sideways commands."""


def apply_blind_r7(
  cfg: ManagerBasedRlEnvCfg, r7: BlindR7Cfg, training: bool = True
) -> ManagerBasedRlEnvCfg:
  """Apply `r7` to a blind config with the fifth round's terms (apply_blind_r5; the
  sixth round's for the legged robots). The wheeled gait clock changes the
  policy's observations, so it applies for evaluation too; the rest only shapes
  training."""
  if r7.wheel_march:
    for group in ("actor", "critic"):
      terms = cfg.observations[group].terms
      assert "phase" not in terms, "the legged robots have their gait clock already"
      terms["phase"] = ObservationTermCfg(
        func=mdp.phase, params={"period": r7.wheel_gait_period, "command_name": "twist"}
      )
  if not training:
    return cfg
  rewards = cfg.rewards
  twist = cfg.commands["twist"]
  assert isinstance(twist, TerrainAwareVelocityCommandCfg)
  if r7.wheel_march:
    twist.march_prob = 0.5
    rewards["foot_gait"] = RewardTermCfg(
      func=mdp.feet_gait,
      weight=r7.wheel_gait_weight,
      params={
        "period": r7.wheel_gait_period,
        "offset": list(r7.wheel_gait_offset),
        "threshold": r7.wheel_gait_threshold,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
        "march_only": True,
      },
    )
  assert twist.march_prob > 0.0, "no marching to hold"
  twist.march_resampling_time_range = r7.march_resampling_time_range
  rewards["march_position_hold"] = RewardTermCfg(
    func=mdp.standing_position_hold,
    weight=r7.march_hold_weight,
    params={"command_name": "twist", "march": True, "moving_speed": 0.25, "moving_yaw_rate": 0.3},
  )
  twist.stair_standing_prob = r7.stair_standing_prob
  rewards["hip_torque"].params |= {"uneven_scale": r7.stairs_torque_scale, "sensor_name": "terrain_scan"}
  if r7.descent_max_speed_ratio is not None:
    drop = rewards["foothold_height_drop"]
    assert drop.params.get("max_speed_ratio") is not None, "no descent cap to change"
    drop.params["max_speed_ratio"] = r7.descent_max_speed_ratio
  if r7.side_drift_weight != 0.0:
    rewards["side_drift"] = RewardTermCfg(
      func=mdp.straight_drift,
      weight=r7.side_drift_weight,
      params={"command_name": "twist", "lateral_scale": None, "sideways_only": True},
    )
  return cfg


def _staged_ramps(
  cfg: ManagerBasedRlEnvCfg,
  drift: tuple[int, int],
  hip: tuple[int, int],
  rest: tuple[int, int],
  march: tuple[int, int],
) -> list[dict]:
  """The phased teacher's ramps over the given (start, end) iterations: drift
  costs 30% -> 100%, hip torque / splay costs 10% -> 100%, the rest stance and the
  standing hip torque scale, marching 5% -> its share of zero commands with its
  position hold 10% -> 100%. Also drops the stairs-standing hip torque scale."""
  rewards = cfg.rewards
  twist = cfg.commands["twist"]
  assert isinstance(twist, TerrainAwareVelocityCommandCfg)
  it = lambda i: i * NUM_STEPS_PER_ENV  # noqa: E731
  ramps: list[dict] = []

  def weight(name: str, frac: float, a: int, b: int) -> None:
    w = rewards[name].weight
    ramps.append({"kind": "reward_weight", "name": name, "start": frac * w, "end": w, "start_step": it(a), "end_step": it(b)})

  def param(name: str, key: str, start: float, a: int, b: int) -> None:
    end = rewards[name].params[key]
    ramps.append({"kind": "reward_param", "name": name, "key": key, "start": start, "end": end, "start_step": it(a), "end_step": it(b)})

  for name in ("straight_drift", "side_drift"):
    if name in rewards:
      weight(name, 0.3, *drift)
  if "hip_torque" in rewards:  # Not with apply_no_hip_load.
    weight("hip_torque", 0.1, *hip)
    weight("hip_splay", 0.1, *hip)
    param("hip_torque", "standing_scale", 1.0, *rest)
    # No extra hip torque cost standing on stairs (user 2026-10-22 ~21:35: robots do not
    # stand long on stairs); stopping mid-flight stays (stair_standing_prob).
    rewards["hip_torque"].params["uneven_scale"] = 1.0
  for name in ("pose", "stand_still_even"):
    if rewards[name].params.get("rest_hip") is not None:  # Not with apply_no_hip_load.
      param(name, "rest_hip", cfg.scene.entities["robot"].init_state.joint_pos[".*L_hip_joint"], *rest)
  ramps.append({"kind": "command", "name": "twist", "key": "march_prob", "start": 0.05, "end": twist.march_prob,
                "start_step": it(march[0]), "end_step": it(march[1])})
  weight("march_position_hold", 0.1, *march)
  if "march_hip_default" in rewards:  # Only with "marchreset".
    weight("march_hip_default", 0.1, *march)
  return ramps


def _stairs_share(cfg: ManagerBasedRlEnvCfg, start: float, end: float, a: int, b: int) -> None:
  """Stairs for `start` -> `end` of the robots over iterations a-b (see mdp.stairs_share);
  after terrain_levels (it judges the last episode on the old origins)."""
  cfg.curriculum["stairs_share"] = CurriculumTermCfg(
    func=mdp.stairs_share,
    params={"stair_terrain_names": MAP3_STAIR_TERRAIN_NAMES, "start": start, "end": end,
            "start_step": a * NUM_STEPS_PER_ENV, "end_step": b * NUM_STEPS_PER_ENV},
  )


# Phased training from scratch (user 2026-10-23 ~02:00, "直接按你的方法来"): the
# teacher's cap, the mixed phase's stairs ramp (iterations on the gated clock from
# where it starts, BLIND_PHASE_START_IT) and cap, and the polish's end.
PHASE_MIXED_START = 5000
PHASE_MIXED_RAMP = 3000
PHASE_FINAL = 13000
TEACHER_STAIRS_SHARE = 0.08
FINAL_STAIRS_SHARE = 0.65
# B2W polish constraint targets ("b2wcons"), training-time values from tools/cons_calib.py (2026-11-01).
# Teacher avg 5000 / mixed avg 7000 / 8000: sideways back-drift p90 0.054 / 0.060 / 0.079 (acceptance 3.1 / 5.6 /
# 6.7%, bar 5%), descents 0.889 / 0.942 / 0.897 of the command (acceptance - / 0.801 / 0.755, bar 0.8).
B2W_BACK_DRIFT = 0.055
B2W_DESCENT_RATIO = 0.96
# "b2wcons2" (2026-11-01 ~16:20): the b2wcons polish's averages 8000 / 9000 still drifted back 5.4% / 5.1% of the
# distance at (0, 0.6, 0) (seeds 0-2 / seed 0) while its training-time back_drift sat at the 0.055 target. Measured
# from the start of the command, a sideways command after a moving one carries the robot forward at first and hides
# the bias; from 1 s into it (back_drift "settled", |vy| >= 0.45; tools/side_calib.py) the median tracks the
# acceptance: +0.005 / +0.026 / +0.041 at teacher avg 5000 / polish avg 8000 / mixed avg 8000, acceptance 3.1 / 5.4 /
# 6.7% (~2.6% + the median). 0.015: ~4.1%. Descents with margin (the average 8000 went down at 0.81-0.83 of the
# command at ~1.0 training-time, bar 0.80).
B2W_FAST_BACK_DRIFT = 0.015
B2W_DESCENT_RATIO2 = 1.0
# The legged teacher's schedule starts once the robots walk (see mdp.staged_clock):
# planar tracking error at most this share of the commanded distance.
TEACHER_WALK_RATIO = 0.7
# Raw policy actions are clipped to this in every phase (2026-10-23): the last action
# is an observation, so one large output fed back on itself; the phased B2 (v1, ~2600)
# and As2 (v2, ~3950) teachers blew up that way (action_rate_l2 per episode -0.3 ->
# -1e3, value loss 0.05 -> 1e10 within 20 iterations). Normal outputs stay within a
# few units (joint actions are scaled by 0.25 rad, the wheels' by 10 rad/s).
PHASE_CLIP_ACTIONS = 10.0


def phase_entropy_coef(phase: str) -> float:
  """PPO entropy coefficient of a phase: 0.01, the polish half of it. With "lowent" in
  BLIND_PILOT (legged v7, 2026-10-24 ~15:50) half of that again: the teachers' action std grew
  with the costs added (Go2 v1 0.41 -> v5 / v6 0.60-0.64, rising as the cost ramps run) and
  the policies learned to offset the drift their own exploration noise causes. Go2 v6 2000
  marching in place without randomisation: 20.7 cm per 20 s acting on its mean, as the
  acceptance and the robot do, 7.1 cm sampling its actions as in training, so training
  barely saw the bias the march hold should remove. Read when the tasks register: set
  BLIND_PILOT before the import."""
  coef = 0.005 if phase == "polish" else 0.01
  if "lowent" in os.environ.get("BLIND_PILOT", "").split(","):
    coef *= 0.5
  return coef


def apply_leg_hold(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """With "leghold" in BLIND_PILOT (B2W, 2026-11-01 ~10:00), the legs' position targets hold still once the robot
  stands still at the zero command (mdp.HeldJointPositionAction, deploy JointPositionAction hold_after_s):
  after 1.5 s ("leghold10" / "leghold20": 1.0 / 2.0 s) and with every leg joint slower than 0.1 rad/s; they
  follow the policy again on a command, a march or a joint faster than 1 rad/s. The B2W teachers crept 1.3-6.5 cm
  per 4 s after stops (bar 1) bringing their body back over the braked wheels, their own targets frozen 2 s after
  the release 0.12-0.30, and no reward or observation change taught the hold (see HeldJointPositionAction).
  Training and evaluation tasks call it alike."""
  pilot = os.environ.get("BLIND_PILOT", "").split(",")
  after = [s for t, s in (("leghold", 1.5), ("leghold10", 1.0), ("leghold20", 2.0)) if t in pilot]
  if after:
    old = cfg.actions["joint_pos"]
    keep = {f.name: getattr(old, f.name) for f in fields(old) if f.init}
    cfg.actions["joint_pos"] = mdp.HeldJointPositionActionCfg(**keep, command_name="twist", hold_after_s=after[0])
  return cfg


def apply_stand_clock(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """With "standclock" in BLIND_PILOT, the actor and critic see mdp.stand_time (how long the command has been
  zero, over 2 s), appended to both groups: the training phases (apply_blind_phased) and the evaluation tasks of
  the same policies call it (see "standclock" there)."""
  if "standclock" in os.environ.get("BLIND_PILOT", "").split(","):
    for group in ("actor", "critic"):
      cfg.observations[group].terms["stand_time"] = ObservationTermCfg(
        func=mdp.stand_time, params={"command_name": "twist", "horizon_s": 2.0}
      )
  return cfg


def apply_blind_phased(cfg: ManagerBasedRlEnvCfg, phase: str, swing_weight: float = -3.0) -> ManagerBasedRlEnvCfg:
  """The seventh round's recipe trained from scratch in three phases (user
  2026-10-23 ~02:00; replaced the staged recipe's single ramped run):

  - "teacher" (0-5000): flat and rough ground with 8% stairs (kept so the critic's
    height-scan statistics have seen stairs and stairs are never absent), every
    flat rule at full weight early: drift and hip costs ramp over 500-2000, the
    rest stance and marching over 1000-3000 / 1000-2500. Gate before going on:
    the flat acceptance (44 commands, drift, marching, stop, hip torque).
  - "mixed" (from the teacher checkpoint that passed, iteration BLIND_PHASE_START_IT
    in the environment, else 5000): every weight final, stairs 8% -> 65% over 3000
    iterations of the gated clock (user 2026-10-23 ~02:40: progress by how training
    goes; see mdp.staged_clock), which runs at a quarter of the pace while more than
    30% of episodes end in a fall or illegal contact (the staged runs ended 14-23% of
    their episodes so with stairs on 20-50% of the robots; terrain demotions are
    expected here and do not pause it); the runner anchors flat ground to the teacher
    (see PhasedAnchoredRunner). Polish starts from the first checkpoint that passes
    the stairs and flat acceptance (judged by hand), at 11000 at the latest.
  - "polish" (2000 iterations): stairs at 65%, lower entropy (rl cfg), still anchored.

  Every phase also trains what the flat acceptance judges of walking straight and
  sideways (user 2026-10-23 ~02:50): the offset from the line as a share of the
  distance walked (straight_drift offset_ratio_scale 0.1: costs 0.5 at the 5%
  limit), and the sideways heading cost for legged robots too (the wheeled ones
  have it since the seventh round).

  One recipe for every robot (user 2026-10-23 ~03:05): the fixes once made for
  single robots apply to all of them, or to every robot of the kind they concern:
  10% diagonal flat commands and 5% more sideways-only ones (were Go2 / As2 and As2);
  rises up to 20 cm (were Go2: train above map3's highest, 18.6 cm); the stair speed
  deficit cost at -2 (was Go2W; legged robots descend slowly too); wheeled: tracking
  std floor 0.1 (As2W had 0.2), wheel lift cost -4 (now -12, below) and the inward splay deadband (B2W
  had -1 and none); legged: descent pay capped at 1.6x (was Go2 / As2), the standing
  position hold and speed costs of the wheeled robots (legged ones crept), and the
  foot swing cost -3, ramped from -0.5 over 500-3000 in the teacher (B2 had its own
  -0.5 -> -1.5 stages). What stays per robot is computed by one rule from its size
  (hip torque reference, stances, payload, wheel and swing heights).

  Every robot also pays the speed deficit on even ground (2026-10-23: the Go2 and
  As2 teachers went from 70-83% of 1.5 m/s at 2000 to marching in place at 3000,
  as their costs reached full weight; the staged Go2 slid 83 -> 79% too. The
  wheeled teachers stepped sideways at 47-67% (Go2W, As2W) and 0% (B2W) at
  2000-3000: at 0.6 m/s B2W standing earned 0.13/s more than the seventh-round
  B2W stepping at 0.59).

  The legged teacher's schedule waits until the robots walk (2026-10-23: the B2
  teacher stood still for 2400 iterations, error_vel_xy 1.1 throughout; its costs
  rose from iteration 500 and B2 first walks at ~550 with them light; Go2 and As2
  walk by ~400, so their schedule does not move).

  The march hold now charges the drift (2026-10-23): its anchor followed the base
  whenever it turned faster than 0.3 rad/s, and the trot's yaw sway marching in
  place is above that on half of the steps (Go2 teacher 4000: p50 0.27, p99 0.51
  rad/s), so it charged ~0.1 cm of a 27 cm drift per 20 s (the legged and wheeled
  robots drifted 26-51 cm, every earlier round 30-64 cm). It follows above 1 rad/s,
  at the standing hold's weight (-10, not the seventh round's -50 that never
  charged).

  Wheeled robots drive straight on all four wheels (2026-10-23): from scratch Go2W
  and As2W trotted while driving straight from iteration ~200 on (all four wheels
  down 9-27% and 2-6% of the time at 2000 and 3000, the staged runs 5-27%; their
  lift cost stayed flat for 3000 iterations, also with the stepping reward ramped
  in late). On the teacher's terms at 3000 (scratchpad wdiag/terms2.py), the
  rolling seventh-round Go2W paid 1.0 / 3.3 per second of outward splay at 1.0 /
  1.5 m/s (the legs splay under load while rolling, partly the simulator's contact
  creep, see hip_splay) and lost to trotting there by 0.0 / 2.8 per second; the
  rolling As2W beat trotting by 1.5-2.0 per second at every speed and still
  trotted (an early local optimum). So the outward splay counts while standing
  only (hip_torque still charges the load while driving; the inward part stays,
  it keeps backing up upright) and lifting a wheel costs 3x (-12): trotting then
  pays ~4.5 per second driving straight. In the teacher the lift cost ramps from -4
  to -12 with the hip costs (500-2000): at -12 from the first iteration the random
  early policy paid so much for lifted wheels that As2W fell after ~190 steps (was
  ~950) and Go2W hit the ground with its body. With the ramp too the wheeled returns
  fell to -20 to -40 by 850-1700 and all three v7 teachers ended their episodes on
  purpose (body contacts, -4 each, cost less than living on), so living pays +4 per
  second for wheeled robots (is_alive). Robot-specific (user 2026-10-23
  ~02:40: allowed, but labelled): `swing_weight`, the legged foot swing cost the
  teacher ramps to (B2 -1.5, see its config).
  """
  rewards = cfg.rewards
  twist = cfg.commands["twist"]
  assert isinstance(twist, TerrainAwareVelocityCommandCfg)
  twist.diag_prob = 0.1
  twist.flat_mode_probs = (0.2, 0.25, 0.2, 0.15, 0.2)
  for sub_terrain in cfg.scene.terrain.terrain_generator.sub_terrains.values():
    if hasattr(sub_terrain, "step_height_range"):
      sub_terrain.step_height_range = (sub_terrain.step_height_range[0], 0.20)
  for name in ("track_linear_velocity", "track_angular_velocity"):
    if "min_std" in rewards[name].params:
      rewards[name].params = {**rewards[name].params, "min_std": 0.1}
  rewards["stair_speed_deficit"] = RewardTermCfg(
    func=mdp.stair_speed_deficit, weight=-2.0, params={"command_name": "twist", "sensor_name": "terrain_scan"}
  )
  wheeled = "wheel_air_drivable" in rewards
  if wheeled:
    rewards["wheel_air_drivable"].weight = -12.0
    if "hip_splay" in rewards:  # Not with apply_no_hip_load.
      rewards["hip_splay"].params = {**rewards["hip_splay"].params, "inward_deadband": 0.1, "outward_standing_only": True}
    # Living pays (wheeled v8, 2026-10-23 ~19:15): the wheeled returns run negative (-8 to
    # -12 by 600 in every version: action rate ~3x the legged, wheel lift and swing costs)
    # and the lift ramp to -12 drives them to -20 to -40, where ending the episode (-4 for
    # is_terminated) costs less than living: all three v7 teachers took to body contacts
    # (illegal contact 0.3 -> 3.5 per episode, episodes 950 -> 520 steps) at 850-1700.
    # +4/s keeps a step alive worth more than the costs (the legged run at +1.3-2/s).
    rewards["is_alive"] = RewardTermCfg(func=mdp.is_alive, weight=4.0)
  else:
    drop = rewards["foothold_height_drop"]
    drop.params = {**drop.params, "max_speed_ratio": 1.6, "step_average": True, "min_gain": 0.02}
    rewards["standing_position_hold"] = RewardTermCfg(
      func=mdp.standing_position_hold, weight=-10.0, params={"command_name": "twist", "command_threshold": 0.1}
    )
    rewards["standing_speed"] = RewardTermCfg(
      func=mdp.standing_speed, weight=-2.0, params={"command_name": "twist", "command_threshold": 0.1}
    )
    rewards["foot_swing_clearance"].weight = swing_weight
    cfg.curriculum.pop("swing_weight", None)
  rewards["flat_speed_deficit"] = RewardTermCfg(
    func=mdp.stair_speed_deficit,
    weight=-2.0,
    params={"command_name": "twist", "sensor_name": "terrain_scan", "even": True},
  )
  if "side_drift" not in rewards:
    rewards["side_drift"] = RewardTermCfg(
      func=mdp.straight_drift,
      weight=-1.0,
      params={"command_name": "twist", "lateral_scale": None, "sideways_only": True},
    )
  for name in ("straight_drift", "side_drift"):
    rewards[name].params = {**rewards[name].params, "offset_ratio_scale": 0.1}
  hold = rewards["march_position_hold"]
  hold.weight = -10.0
  hold.params = {**hold.params, "moving_yaw_rate": 1.0}
  # Pilot changes, only with BLIND_PILOT (comma-separated) in the environment (2026-10-23
  # ~14:15, fine-tunes of the 5000 teachers; the running teachers do not have them):
  # "critic" shows the critic the anchor / line state the hold and drift costs depend on
  # (mdp.hold_offsets, mdp.drift_offsets, appended: widen the checkpoint with
  # scripts/blind/widen_obs.py).
  pilot = set(filter(None, os.environ.get("BLIND_PILOT", "").split(",")))
  if "critic" in pilot:
    critic = cfg.observations["critic"].terms
    critic["hold_offsets"] = ObservationTermCfg(
      func=mdp.hold_offsets, params={"term_names": ("standing_position_hold", "march_position_hold")}
    )
    critic["drift_offsets"] = ObservationTermCfg(
      func=mdp.drift_offsets, params={"term_names": ("straight_drift", "side_drift")}
    )
  if "drcritic" in pilot and wheeled:
    # The critic sees each robot's base CoM offset, payload and foot friction (wheeled group, v13wk,
    # 2026-10-30; mdp.dr_params). As2W v13wh failed its gate on a different item at 3000 / 4000 / 5000:
    # sideways drift (p10 5.9%), 1.5 m/s wheels down (96.9%), the slow arc (0.15, 0, 0.3) at x77%. Over
    # 128 robots each, the drift and the slow arc followed the base CoM offset (+-5 cm): the worst 10%
    # drifting sideways had it 1.5-3.5 cm further forward, the slowest arcs ~3.5 cm further back (corr
    # +0.7), and the arcs slowed from x87% to x76% (p50) from 3000 to 5000. Braking the wheels while
    # stepping did not help (the drift is in the steps). Go2W, same recipe, passed at 3000.
    cfg.observations["critic"].terms["dr_params"] = ObservationTermCfg(
      func=mdp.dr_params, params={"friction_cfg": cfg.events["foot_friction"].params["asset_cfg"]}
    )
  if "marchw40" in pilot:
    # The anchor still follows a pushed robot (user 2026-10-23 ~17:15: no walking back
    # after a push, the cost cap stays): pushes every 5-6 s leave ~5.5 s of drift to
    # charge where the acceptance measures 20 s, so the hold weighs 4x.
    hold.weight = -40.0
  if "marchsettle" in pilot:
    # The march anchor follows pushes, not the base's speed: at -40 the speed rule (moving_speed
    # 0.25) let the Go2/B2 pilots walk off at ~0.35 m/s, 7-8 m per 20 s of marching (2026-10-23
    # ~18:50). It follows until the robot marches steadily again, at most 2 s (user ~19:35: a push
    # is not drift, the drift counts once it marches steadily).
    if "push_robot" in cfg.events:  # Not in play configs.
      cfg.events["push_robot"].func = mdp.push_by_setting_velocity_marked
    hold.params = {**hold.params, "push_window": 2.0, "settle": True}
  if "marchsmooth" in pilot and "marchsettle" in pilot:
    # The march hold on the mean pose over one gait period, locking once the robot marches
    # steadily (mean speed <= 3 cm/s, yaw rate <= 0.1 rad/s over the period; legged v4
    # candidate, 2026-10-24 ~04:35). v3 5000 marched off 10.6 (Go2) / 16.6 cm (As2) per 20 s at a
    # steady 0.6 / 1.0 cm/s from the first second, pushed or not: a learned backward bias (4.5 /
    # 8.0 cm with no randomisation) plus ~9 cm of per-robot spread. The instant rule locked the
    # anchor 0.2 s after marching began at <= 0.25 m/s, mostly while the robot still slid on from
    # walking forward, so the hold paid for stepping back.
    gait = rewards.get("foot_gait")
    hold.params = {**hold.params, "smooth_s": gait.params["period"] if gait is not None else 0.6}
  if "marchquiet" in pilot and "push_robot" in cfg.events:
    # A marching robot skips 3 of 4 pushes (legged v5 candidate, 2026-10-24 ~08:30). A steady drift r
    # costs the hold about r x locked share x mean time since the lock: with a push every 5-6 s that
    # was 1.7 s (smoothed anchor; 2.4 s instant) on the v4 2000 teachers sampling their actions, and
    # they marched off 25-35 cm per 20 s by 2000, nearly all of it a learned bias (25 / 35 cm with no
    # randomisation). Marches last 10-20 s, so most now run undisturbed as the acceptance's 20 s.
    # Pushes that do come still move the anchor (the user's rule: a push is not drift).
    push = cfg.events["push_robot"]
    push.func = mdp.push_by_setting_velocity_marked
    push.params = {**push.params, "march_skip_prob": 0.75}
  if "marchreset" in pilot:
    # Marching in place resets the stance (all robots, v8; user 2026-10-24 ~23:20: "原地踏步的作用是将
    # 模型的站立姿态从内收或者外翻重新定义回零度"): half the robots start with every hip turned in or out by
    # one angle up to 0.15 rad (mirrored left / right, after the joint reset), and while marching the hips
    # pay their squared distance from the default (hipzero: legs vertical), -10 so 0.15 rad on all four
    # costs ~0.9 per second; ramps with the march. Standing still cannot move a planted foot back; the
    # deploy marches when a standing hip is off its default.
    hip_joints = lambda: SceneEntityCfg("robot", joint_names=(".*_hip_joint",))  # noqa: E731
    cfg.events["reset_hip_splay"] = EventTermCfg(
      mode="reset",
      func=mdp.reset_hip_splay,
      params={"prob": 0.5, "splay_range": (-0.15, 0.15), "asset_cfg": hip_joints()},
    )
    rewards["march_hip_default"] = RewardTermCfg(
      func=mdp.march_joint_deviation, weight=-10.0, params={"command_name": "twist", "asset_cfg": hip_joints()}
    )
  if "lowfric" in pilot and "foot_friction" in cfg.events:
    # Slippery floors twice as often in training (legged v8 candidate, 2026-10-24 ~20:40): the foot
    # friction drawn log-uniform over the same 0.3-1.6 (17% of the robots at 0.4 or less, uniform
    # 7.7%; median 0.69). Go2 v7 3000 held 1.5 m/s with a turn for 8 s (the acceptance's time) and lost
    # 6 of 3584 robots, all on friction 0.33-0.40; the acceptance keeps the uniform draw.
    friction = cfg.events["foot_friction"]
    friction.params = {**friction.params, "distribution": "log_uniform"}
  if "standsettle" in pilot:
    # The standing anchor too (2026-10-23 ~21:00): with the anchors in the critic (pilot pcj)
    # the stop creep rose 0.7 -> 1.1-1.2 cm per 4 s, as the march drift had under the speed
    # rule: brief moves over moving_speed (0.1) reset the anchor. Only a push or a stop
    # (braking, up to 2 s) unlocks it now.
    if "push_robot" in cfg.events:  # Not in play configs.
      cfg.events["push_robot"].func = mdp.push_by_setting_velocity_marked
    stand = rewards["standing_position_hold"]
    stand.params = {**stand.params, "push_window": 2.0, "settle": True}
    if "standsmooth" in pilot:
      # As "marchsmooth" for the standing anchor (legged v4 candidate, 2026-10-24 ~04:35): it locks
      # once the robot has come to rest (mean speed <= 3 cm/s over 0.6 s), not at 0.1 m/s 0.2 s
      # after the stop while it still slides on; B2 v3 crept 0.5-1.3 -> 1.2-1.9 cm per 4 s
      # from 2000 to 3000.
      stand.params = {**stand.params, "smooth_s": 0.6}
  if "standlong" in pilot and not wheeled:
    # Zero commands off the stairs hold 4-8 s, not 2-5 (legged group, B2 v8g, 2026-10-26 ~13:10; the march got
    # 10-20 s for its 20 s acceptance the same way): the stop acceptance watches 6 s at zero, and the anchor locks
    # ~1 s after the stop, so a slow settle was charged for 1-4 s at most. B2 v8f: after a stop from 1 m/s the
    # 7000 policy settled within 1 s (creep 0.43 cm per 4 s); polish 12000 bent its calves 2-3 deg over 5 s and
    # the base slid back 3.5 cm over its planted feet (creep 2.02); 8000-11000 drew 0.9-1.3 between them
    # (scratchpad b2fix/creep_trace.py). A steady creep's charge grows with the square of the time standing.
    cfg.commands["twist"].stand_resampling_time_range = (4.0, 8.0)
  if "standlongw" in pilot and wheeled:
    # The same for a wheeled robot (B2W only, v13wb, 2026-10-27 ~11:10): the B2W v13w teacher failed its 3000
    # gate on creep 1.5-2.5 cm per 4 s after stops (Go2W / As2W v13w 0.3-0.5) with B2's pattern: after a stop
    # from 1 m/s its wheels stood still within 1.5 s while the base moved on over them 6.5 -> 11 cm over 4 s
    # (scratchpad w13/b2w_trace.py).
    cfg.commands["twist"].stand_resampling_time_range = (4.0, 8.0)
  if "standw4w" in pilot and wheeled:
    # The standing hold costs 4x, as the march hold under "marchw40" (B2W only, v13wc, 2026-10-27 ~16:40): the
    # B2W v13wb teacher failed its 3000 / 4000 gates on stop creep (4000: 2.5-3.6 cm per 4 s after forward, back
    # and arc stops, rest 0.9; bar 1) with march and sideways heading fixed. Per robot (scratchpad
    # w13/stand_tail.py) the base slides 6-20 cm back over braked wheels (0.01 rad/s) as the thighs turn back
    # 6-14 deg: the legs settle toward the default pose after the stop (also after the random reset pose
    # standing from the start) and the parking brake leaves the base to move instead. Not friction, payload,
    # kp, COM, encoder bias, delay or observation noise (all |r| < 0.15; noise on: same creep). At -10 a
    # creeping stand pays ~0.8 per second, which the posture terms outbid.
    rewards["standing_position_hold"].weight = 4.0 * rewards["standing_position_hold"].weight
  if "standtgt" in pilot and wheeled:
    # Once at rest at the zero command the legs hold their targets (wheeled group, 2026-10-29 ~20:00; see
    # mdp.standing_target_hold): on braked wheels every leg move moves the base. The B2W final changed its
    # targets 3-12 deg (summed) over the last 4 s of the stop acceptance and crept 0.75-2.22 cm; frozen at kp 240
    # ("legkp15") 0.21-0.91. The anchor locks as the base hold's. At -10 a 0.05 rad drift costs 0.5 per second;
    # the action noise low-passed over 1 s leaves ~0.03 rad.
    hold = rewards["standing_position_hold"].params
    keys = ("command_name", "command_threshold", "push_window", "settle", "smooth_s")
    rewards["standing_target_hold"] = RewardTermCfg(
      func=mdp.standing_target_hold,
      weight=-10.0,
      params={
        **{k: hold[k] for k in keys if k in hold},
        "tau_s": 1.0,
        "asset_cfg": SceneEntityCfg("robot", joint_names=rewards["pose"].params["asset_cfg"].joint_names),
      },
    )
  for k in (4, 10):
    if f"standtgtx{k}" in pilot and "standing_target_hold" in rewards:
      # The leg-target hold at k times its weight (B2W pilots, 2026-11-01): the v13wn teacher (legkp15 + standtgt)
      # failed its gate at 3000-5000 on the stop creep alone (fwd 1.0 2.88 cm, bar 1); with its actions frozen 2 s
      # after the release 0.14-0.32 cm in every group, running 2.84 with its targets moving 12 deg (summed) in the
      # window (scratchpad w17/creep_split.py): the pulls toward the default pose still outbid the hold at -10.
      rewards["standing_target_hold"].weight *= k
  if "restlock" in pilot and wheeled and "standing_position_hold" in rewards:
    # Once the standing hold has its anchor (settled, or 2 s after the stop at most) the posture terms keep the
    # pose the robot settled in instead of pulling it to the default (wheeled group, B2W, 2026-11-01): the v13wn
    # teacher's stop creep (2-3 cm after forward stops, bar 1) was the policy re-posing toward the default on braked
    # wheels (frozen actions 0.14-0.32 cm); the target hold at x4 / x10 cut it after forward stops but the back
    # stop slowed to 2.1-2.6 s and crept 2.3-3.3 cm. The march resets the posture (user 2026-10-24).
    for name in ("pose", "stand_still_even"):
      if name in rewards:
        rewards[name].params = {**rewards[name].params, "hold_term": "standing_position_hold"}
  if "lockfix" in pilot and wheeled and "standing_target_hold" in rewards:
    # The leg-target hold's anchor is the mean target over the first 0.5 s after the lock, not the 1 s low-passed
    # target at the lock, and the critic sees the lock and both posture offsets (wheeled group, B2W v13wp,
    # 2026-11-01 ~07:50; mdp.standing_target_hold anchor_s, mdp.standing_lock_offsets). The v13wo teacher (restlock
    # from scratch) still crept 1.3-3.0 cm per 4 s after stops at 4000 (bar 1) with its legs turning 6-15 deg
    # (frozen 2 s after the release: 0.1-0.3 cm), though under the training rewards freezing paid 0.25-0.65 per
    # second more (scratchpad w20/rew_attr2.py; rew_attr.py had used the evaluation task's rewards, which have no
    # pilot switches). Holding still paid the lagged anchor's 0.8-1.1 per second (3.6 standing from the start),
    # which the critic could not see, nor the restlock pose the posture terms keep.
    rewards["standing_target_hold"].params = {**rewards["standing_target_hold"].params, "anchor_s": 0.5}
    if "restlock" in pilot and "pose" in rewards:
      cfg.observations["critic"].terms["standing_lock_offsets"] = ObservationTermCfg(
        func=mdp.standing_lock_offsets,
        params={
          "pose_term": "pose",
          "target_term": "standing_target_hold",
          "hold_term": "standing_position_hold",
          "asset_cfg": SceneEntityCfg("robot", joint_names=rewards["pose"].params["asset_cfg"].joint_names),
        },
      )
  if "standclock" in pilot and wheeled:
    # The actor (and critic) see how long the command has been zero, over 2 s (wheeled group, B2W v13wq, 2026-11-01
    # ~08:50; mdp.stand_time, deploy observation stand_time): its 0.2 s history cannot tell a stop's settling from
    # standing on. The v13wo teacher 5000 reversing at 1 m/s stopped its wheels within 0.5 s but leaned 5-8 deg on
    # its rear legs and brought its body back over the braked wheels for 5 s (-17 -> -29 cm; creep 6.5 cm in the
    # acceptance's last 4 s, bar 1) while freezing its actions 2 s after the release left 0.3 and paid 0.25-0.65 per
    # second more under its training rewards (scratchpad w20/back_trace.py, rew_attr2.py). Appended: widen a
    # checkpoint with scripts/blind/widen_obs.py. The evaluation tasks add it too (apply_stand_clock).
    apply_stand_clock(cfg)
  if "drivepose" in pilot and wheeled:
    # Driving on its wheels on even ground the legs keep the standing posture tolerance (thigh 0.1, calf
    # 0.15 rad), not the walking one (0.35 / 0.5) (wheeled group from v13we, 2026-10-28 ~00:30): the B2W
    # v13w / v13wb / v13wd teachers all failed their last gate on stop creep (v13wd 5000: 1.3-1.7 cm per 4 s,
    # bar 1) and Go2W v13wd 4000 crept 3 cm after forward stops (its v13w 4000 0.3). Both drove leaning their
    # legs, free under the walking tolerance (Go2W front thighs -12.6 deg at 1 m/s, v13w -3.3; B2W rear thighs
    # +13 deg backing), and once stopped the standing tolerance pulled them back over 2-6 s while the parking
    # brake held the wheels, so the base slid over them (scratchpad w13/b2w_trace2.py, stand_tail.py).
    drivable = rewards["wheel_air_drivable"].params
    pose = rewards["pose"]
    pose.params = {
      **pose.params,
      "drive_max_lin_vel_y": drivable.get("max_lin_vel_y", 0.1),
      "drive_max_ang_vel_z": drivable.get("max_ang_vel_z", 0.1),
    }
  if "slowarc" in pilot and wheeled:
    # The steering commands start at 0.1 m/s, not 0.3 (wheeled group, 2026-10-28 ~03:30): the wheels drive
    # straight commands, so a wheeled robot steps forward only on arcs (|wz| > 0.03) or with a sideways part,
    # and slow arcs (0.1-0.25 m/s, |vy| <= 0.1) were 0.24% of the flat commands against 2.6% at 0.25-0.5 m/s
    # (1.64% with this). Stepping arcs lagged ~0.03 m/s at every speed (Go2W v13we 2000 (0.15 / 0.3 / 0.5, 0,
    # 0.3): x 81 / 93 / 95%, w13/arc_diag.py), which only the slowest fails; Go2W v13wd 5000 failed its gate on
    # (0.15, 0, 0.3) at x 71%, As2W 81-85%, B2W 84-97%, the legged robots 98-107% (all their forward is steps).
    twist.steer_vx_range = (0.1, twist.steer_vx_range[1])
  if "feetedge" in pilot and not wheeled:
    # Feet standing at stair edges pay (legged group, 2026-10-30, the user's "feet_edge"; see mdp.feet_edge): a
    # 3 x 3 grid of downward rays fixed in the world around each foot site (+-4 cm) tells an edge under the foot.
    sensors = list(cfg.scene.sensors or ())
    names = []
    for foot in ("FL", "FR", "RL", "RR"):
      names.append(f"edge_{foot}")
      sensors.append(RayCastSensorCfg(
        name=names[-1],
        frame=ObjRef(type="site", name=foot, entity="robot"),
        pattern=GridPatternCfg(size=(0.08, 0.08), resolution=0.04),
        ray_alignment="world",
        max_distance=1.0,
        include_geom_groups=(0,),
      ))
    cfg.scene.sensors = tuple(sensors)
    rewards["feet_edge"] = RewardTermCfg(
      func=mdp.feet_edge,
      weight=-1.0,
      params={"command_name": "twist", "contact_sensor_name": "feet_ground_contact", "edge_sensor_names": tuple(names)},
    )
    if "feetedgew3" in pilot:
      # Three times the cost (2026-10-30 pilot): at -1 a 1000-iteration Go2 polish cut feet at an edge on the map3
      # flights only 7.5 -> 6.6% of the stance time (per foot; the final it was polished against) and 21.7 -> 20.7% of
      # the touchdowns on the eval stairs.
      rewards["feet_edge"].weight = -3.0
  if "legcons" in pilot and not wheeled:
    # Acceptance items as constraints on reward weights (legged group, 2026-10-30, the user's OK to try constrained
    # RL; mdp.adaptive_constraints): each term's weight grows while its item misses a target inside the bar and eases
    # back once met, up to max_scale x. Sideways / diagonal gait clock (the Go2 final re-checked at 80% on (0, -0.6, 0),
    # the feetedge pilot 72-79% on (0, 0.6) / (0.6, +-0.4); bar 80% per command), feet at stair edges (Go2 polish on
    # the map3 flights 6.6-7.5% of the feet down; feet_edge at -1 hardly moved it), legs on the ground on the stairs
    # (the legged finals 1.5-6% of the time descending 12-18 cm).
    assert "feet_edge" in rewards, "legcons needs feetedge"
    # Targets in training-time terms (pushes, noise, rough ground; tools/cons_calib.py, 2026-10-30): the Go2 final
    # measured 0.831 sideways / diagonal clock (|vy| >= 0.5; its acceptance 80% on (0, -0.6)), its feetedge polish
    # 0.800 (acceptance 75%), so 0.89 is ~85% in the acceptance; feet at edges 12.3 / 11.6% of the feet down on the
    # stairs (feet_edge at -3 cut it but robots fell, so the cost may at most double); legs on the ground 2.1%.
    cfg.curriculum["constraints"] = CurriculumTermCfg(
      func=mdp.adaptive_constraints,
      params={"constraints": [
        {"name": "clock_side", "metric": "gait_match", "params": {"min_abs_vy": 0.5}, "target": 0.89, "sense": "min",
         "term": "foot_gait", "lr": 0.2, "max_scale": 3.0},
        {"name": "side_drift", "metric": "drift_share", "stat": "p90", "params": {"term": "side_drift"},
         "target": 0.06, "sense": "max", "term": "side_drift", "lr": 1.0, "max_scale": 3.0},
        {"name": "feet_edge", "metric": "feet_edge_share", "target": 0.10, "sense": "max", "term": "feet_edge",
         "lr": 0.5, "max_scale": 2.0},
        {"name": "leg_contact", "metric": "leg_contact", "target": 0.015, "sense": "max", "term": "leg_ground_contact",
         "lr": 2.0, "max_scale": 3.0},
      ]},
    )
  if "wheelcons" in pilot and wheeled:
    # Acceptance items as constraints on reward weights (wheeled group, 2026-10-30; as "legcons", polish only). The
    # Go2W v13wh polish candidate (average 10600..11300) failed seeds 0 / 2 on the sideways clock (0, -0.6) at 79.x%
    # (bar 80%), the As2W teachers swung on the slow arc (x72-86%) and sideways drift. Training-time values
    # (tools/cons_calib.py, 1024 robots): Go2W candidate clock |vy| >= 0.5 0.827 (polish start 0.841, passed),
    # sideways drift p90 0.064, slow arc x0.94, all wheels down 0.913 (pushes and rough ground; the acceptance's
    # flat drive 98-100%); As2W slow arc 0.836 / 0.874 / 0.883 = acceptance x76 / x80 / x86. The legged pilot moved
    # the clock 0.80 -> 0.89 in 1000 iterations with foot_gait up to 3x.
    # All wheels down is a guard (the Go2W candidate passed it), at 0.75 (2026-10-31): cons_calib.py acts on the
    # policy's mean, training samples its actions, and the Go2W polish from 9300 measured 0.77 in training (0.913
    # calibrated); at 0.90 it sat out of reach and wheel_air_drivable would stay at 2x. The other metrics agreed
    # (clock 0.836 / 0.841, slow arc 0.91-0.93 / 0.90).
    # Milder push (2026-10-31 ~13:10): with the clock at 0.89 / 3x and sideways drift at 0.06 / 3x the Go2W polish from
    # 9300 held both at 3x all along (training clock 0.85-0.88, drift p90 0.085-0.095) and overshot: the sideways
    # clock went 79 -> 89-92% (bar 80), but walking sideways at 0.3 m/s ran y122-127% (bar 120; 103-116% before) and
    # all wheels down at 1.5 m/s fell to 97.7% (bar 98) at 11300. Now the clock aims at 0.855 with foot_gait at most
    # 2x, and sideways drift is a guard at the training level (0.095).
    cfg.curriculum["constraints"] = CurriculumTermCfg(
      func=mdp.adaptive_constraints,
      params={"constraints": [
        {"name": "clock_side", "metric": "gait_match", "params": {"min_abs_vy": 0.5}, "target": 0.855, "sense": "min",
         "term": "foot_gait", "lr": 0.2, "max_scale": 2.0},
        {"name": "side_drift", "metric": "drift_share", "stat": "p90", "params": {"term": "side_drift"},
         "target": 0.095, "sense": "max", "term": "side_drift", "lr": 1.0, "max_scale": 3.0},
        {"name": "slow_arc", "metric": "arc_speed", "target": 0.90, "sense": "min", "term": "track_linear_velocity",
         "lr": 0.5, "max_scale": 1.5},
        {"name": "wheels_down", "metric": "wheels_down", "target": 0.75, "sense": "min", "term": "wheel_air_drivable",
         "lr": 2.0, "max_scale": 2.0},
      ]},
    )
  if "b2wcons" in pilot and wheeled:
    # B2W only (2026-11-01, polish; robot-specific targets): the v13wq mixed (gated teacher avg 5000 with leghold20)
    # passed every stairs-phase item but two that slid as it trained. Walking sideways at 0.6 m/s the robots drift
    # back: acceptance p10 3.1% of the distance (teacher avg 5000) -> 5.6% (mixed avg 7000) -> 6.2-7.2% (8000; bar
    # 5%), mostly the steps landing further back each time (tools/flat_items_dr.py: step part p50 -0.8 -> -3.5%,
    # rolling -0.9 -> -1.6%), uncorrelated with the randomization (|r| <= 0.5). Map3 descents at 0.4 m/s with the
    # operator: 0.801 -> 0.755 of the command (bar 0.80). Targets in training terms (tools/cons_calib.py on the
    # three checkpoints): the signed back-drift walking sideways ("back_drift" p90; the |offset| p90 of "wheelcons"
    # stayed at 0.11-0.12 on all three) at the teacher's level, descents ("stair_speed") just above the mixed 7000's.
    cfg.curriculum["constraints"] = CurriculumTermCfg(
      func=mdp.adaptive_constraints,
      params={"constraints": [
        {"name": "back_drift", "metric": "back_drift", "stat": "p90", "params": {"term": "side_drift"},
         "target": B2W_BACK_DRIFT, "sense": "max", "term": "side_drift", "lr": 1.0, "max_scale": 3.0},
        {"name": "descent", "metric": "stair_speed", "params": {"descend": True}, "target": B2W_DESCENT_RATIO,
         "sense": "min", "term": "stair_speed_deficit", "lr": 0.5, "max_scale": 2.0},
      ]},
    )
  if "b2wcons2" in pilot and wheeled:
    # B2W only (2026-11-01 ~16:20, polish from the mixed avg 7000 again; see B2W_FAST_BACK_DRIFT): b2wcons with the
    # back-drift measured as the acceptance sees it (walking fast sideways, past the start; its median, smaller than
    # the p90, hence the higher rate) and a higher descent target.
    cfg.curriculum["constraints"] = CurriculumTermCfg(
      func=mdp.adaptive_constraints,
      params={"constraints": [
        {"name": "back_drift", "metric": "back_drift", "stat": "p50",
         "params": {"term": "side_drift", "min_abs_vy": 0.45, "settled": True}, "target": B2W_FAST_BACK_DRIFT,
         "sense": "max", "term": "side_drift", "lr": 2.0, "max_scale": 3.0},
        {"name": "descent", "metric": "stair_speed", "params": {"descend": True}, "target": B2W_DESCENT_RATIO2,
         "sense": "min", "term": "stair_speed_deficit", "lr": 0.5, "max_scale": 2.0},
      ]},
    )
  if "slowarc2" in pilot and wheeled:
    # A quarter of the steering commands are slow arcs, 0.1-0.3 m/s (wheeled group, v13wm, 2026-10-30): with
    # "slowarc" they were ~1.6% of the flat commands, and the As2W v13wl teacher's slow arc (0.15, 0, 0.3) fell from
    # x90 (2000) to x80 / x76 (3000 / 4000; their weight averages x72 / x73, bar x80) while every other flat item
    # passed with margin; the robots step these arcs (|wz| > 0.03) and stepping slowly was rarely trained.
    twist.steer_slow_prob = 0.25
  if "drivequad" in pilot and wheeled:
    # Driving, the legs pay their squared deviation from the default (wheeled group, 2026-10-28 ~09:20; with
    # "drivepose"): the posture term's exp kernel flattens far from the default, and halving its tolerance
    # (the dropped pilot "drivepose2") cut its pull on a crouch ~10x instead of raising it (B2W at 1 m/s,
    # calves -10 / -20 deg: exp(-4.9) = 0.007, d/dq 0.08 per rad, under drivepose 0.75). B2W v13wf drove
    # deeper than v13we (fwd 1.5 calves -13 / -21 deg against -9 / -12) and after the stop straightened
    # over 5 s with the wheels braked, the base moving on 5.8 cm (creep 3.9 cm per 4 s, gate 3000; at 2000
    # it stood up within 1 s while its wheels still rolled, 0.97; w13/creep_target_b2w_v13wf.out).
    # At -2 that crouch costs 0.63 per second, and the pull does not fade with the depth.
    assert "drive_max_lin_vel_y" in rewards["pose"].params, "drivequad needs drivepose"
    pose = rewards["pose"].params
    rewards["drive_still_even"] = RewardTermCfg(
      func=mdp.drive_still_even,
      weight=-2.0,
      params={
        "command_name": "twist",
        "sensor_name": pose["sensor_name"],
        "max_lin_vel_y": pose["drive_max_lin_vel_y"],
        "max_ang_vel_z": pose["drive_max_ang_vel_z"],
        # Its own copy: the reward manager resolves each term's cfg in place.
        "asset_cfg": SceneEntityCfg("robot", joint_names=pose["asset_cfg"].joint_names),
      },
    )
  if "drivelp" in pilot and wheeled:
    # drivequad on the deviation low-passed over 0.5 s (wheeled group, 2026-10-28 ~13:50; replaces
    # "drivequad"). Charged per step, B2W v13wg cancelled its own action noise through the last-action input:
    # thigh / calf loop gains 4-36 from 2200 (v13wf below 2; Go2W v13wg 2400 ~1), and right after resets the
    # deterministic actions ran away x3 per step, 0.6 -> 1e6, robots sitting down: 86 falls at the 3000 gate,
    # 7-14% of resets at 2400-3200 (w13/fall_traj.py, kick_probe.py). drivequad had cut its forward creep
    # 3.9 -> 0.8 cm; a held crouch still pays the same here.
    assert "drive_max_lin_vel_y" in rewards["pose"].params, "drivelp needs drivepose"
    assert "drive_still_even" not in rewards, "drivelp replaces drivequad"
    pose = rewards["pose"].params
    rewards["drive_still_even_lp"] = RewardTermCfg(
      func=mdp.drive_still_even_lp,
      weight=-2.0,
      params={
        "command_name": "twist",
        "sensor_name": pose["sensor_name"],
        "max_lin_vel_y": pose["drive_max_lin_vel_y"],
        "max_ang_vel_z": pose["drive_max_ang_vel_z"],
        "tau_s": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=pose["asset_cfg"].joint_names),
      },
    )
  if "marchgait3" in pilot and wheeled:
    # The gait clock pays three times while marching in place (B2W only, v13wb, 2026-10-27 ~11:10): the B2W
    # v13w teacher never stepped marching (0.00 touchdowns/s at 2000 and 3000; Go2W / As2W 1.2-1.75 from
    # 2000) but crouched on four wheels, thighs +11-14 deg. Under the training rewards (w13/b2w_trace.py
    # --train, per second) marching paid Go2W 4000 +0.89 on the clock against -1.95 march hold, -0.45
    # tracking at zero, -0.16 wheel targets, -0.11 swing clearance; standing still matches the trot's 56%
    # stance share for free (B2W +0.53), so the heavier robot stayed put.
    gait = rewards["foot_gait"]
    gait.params = {**gait.params, "march_scale": 3.0}
  if "stopfast" in pilot and not wheeled:
    # Half the flat commands after a fast one (a part >= 75% of its range limit) are stands (legged group, B2 v8h,
    # 2026-10-27 ~00:30): B2 v8g 9000-polish 12000 failed the stop acceptance on falls after stopping from 0.6 m/s
    # sideways (5-9 of 64): with the stop moved through one gait period (scratchpad b2fix/stop_phase.py) they fall at
    # one or two phases (8-11 of 96 there; 0.45 m/s and below never), the acceptance stopping all robots at one
    # phase. Stopping from the top of the range was ~1 in 300 flat commands (fast sideways, then a stand: 10%).
    cfg.commands["twist"].stop_after_fast_prob = 0.5
  if "stopside" in pilot and not wheeled:
    # 15% of the flat commands walk sideways at 75-100% of the vy limit (legged group, B2 feetedge mixed + polish,
    # 2026-10-31): every B2 average from mixed 9000 on (polish or mixed) fell after the acceptance's stop from 0.6 m/s
    # to its left, 1-5 of 256 (stop_trace.py: 6-11 of 512 at that phase, none to the right; stuck on both left feet
    # with both right feet up, rolling over 1-2 s later). With stopfast that stop was ~1 flat command in 60; this
    # makes it ~1 in 12 (scratchpad b2fix/cmd_mix.py; to the left 1 in 131 -> 1 in 24; the stand after a fast command
    # keeps its 50%).
    cfg.commands["twist"].fast_side_prob = 0.15
  if "spinup" in pilot and not wheeled:
    # A fifth of the flat commands after a stand turn in place at 75-100% of the wz limit (legged group, B2 feetedge,
    # 2026-10-31 ~16:20): with stopside the B2 side stop held (0 of 256 at polish 10000, mixed 5000), but most checks
    # from 9000 on fell 1-2 of 256 spinning up to 1.5 rad/s from a standstill (spin_diag.py: illegal contact at 1.72 s,
    # foot friction 0.31; mixed avg 8000 0 of 512). Spin-ups from a stand: 1 in 89 flat commands, 1 in 8 at 0.33
    # (scratchpad b2fix/cmd_mix2.py), ~1 in 13 here.
    cfg.commands["twist"].fast_spin_prob = 0.2
  if "yawslew" in pilot and not wheeled:
    # The yaw rate is tracked against the command rising at most 3 rad/s^2 (legged group, B2 stopside+spinup mixed,
    # 2026-11-01 ~03:30): spinup did not stop the spin-up falls (mixed averages 8000 / 9000 / 10000: 3 / 7 / 2 falls
    # over seeds 0-2 at +-1.5 rad/s). spin_trace.py (scratchpad w19, avg 9000, 2048 per direction): the policy reaches
    # 80% of 1.5 rad/s in 0.12 s from a standstill, falls 7 of 4096 with one foot down and 16-26 deg of roll at
    # 0.1-0.3 s; under a command ramped over 0.5 s it fell 0 (tilt p90 9.2 -> 5.2 deg). The acceptance steps the
    # command, so the policy must ramp itself: the tracking no longer pays for a faster spin-up.
    rewards["track_angular_velocity"].params["yaw_accel"] = 3.0
  if "sidew3" in pilot:
    # Sideways drift costs 3x (legged v4 candidate, 2026-10-24 ~02:30): at -1 it came to ~0.08
    # per second (march hold 0.46, tracking ~0.8) and the v3 teachers drifted 7-10% stepping
    # right at 3000 (Go2: a learned bias, 5.6% with no randomisation; As2: mostly the
    # per-robot randomisation spread).
    rewards["side_drift"].weight = -3.0
  if "sidew6" in pilot and wheeled:
    # Sideways drift costs 6x (wheeled group, v10w, 2026-10-26 ~15:00): the Go2W v9w teacher failed its flat gate
    # at 3000-5000 stepping sideways, the base going back 1.5-2.2 cm/s whichever way it stepped (median offset
    # -0.15..-0.22 m over 3-5 m, p10 5.4-7.7%, bar 5%), As2W v9w 3000 alike (-0.10..-0.17 m), mostly in the
    # stepping phases, not rolling (scratchpad wheel8/sidestep_diag.py). At -3 a 5% share costs 1.5 per second.
    rewards["side_drift"].weight = -6.0
  if "marchreset2" in pilot and wheeled and "march_hip_default" in rewards:
    # The march resets the hips at twice the cost (wheeled group, v10w, 2026-10-26 ~15:00): from an 8 deg splay the
    # Go2W v9w teacher's 5 s march left 1.1 deg (3000) -> 1.7 -> 2.9 deg (5000; bar 2.0, worst robot 5.3), the plain
    # march within ~1 deg; squared at -10, 3 deg on all four hips costs ~0.1 per second.
    rewards["march_hip_default"].weight = 2.0 * rewards["march_hip_default"].weight
  if "steptarget" in pilot and wheeled:
    # The wheel targets, not the wheel speed, go to zero while stepping without a forward command (wheeled group, v12w,
    # 2026-10-26 ~22:50; see mdp.stepping_wheel_target): the v11w pilot charged the measured speed, which the
    # policy does not set alone; it became the largest cost and the 2000 checks regressed everywhere. -0.1 as
    # standing_wheel_target at the zero command: the v10w targets (~0.6 rad/s a wheel) cost ~0.24 per s.
    rewards["stepping_wheel_target"] = RewardTermCfg(
      func=mdp.stepping_wheel_target,
      weight=-0.1,
      params={"command_name": "twist", "scan_sensor_name": "terrain_scan"},
    )
  if "straightw3" in pilot:
    # Walking-straight drift costs 3x too (legged v4 candidate, 2026-10-24 ~04:50): with
    # "sidew3" alone the pilot pcl's sideways drift fell to 2.7-3.5% (Go2 6000), but Go2 took
    # to slipping left walking forward at 0.5 m/s (median offset -0.06 -> +0.13 m from v3 4000
    # to pcl 6000, p90 7.2%). The term stops at 1.0 m/s (max_speed), so top speed is unaffected.
    rewards["straight_drift"].weight = -3.0
  if "speedboth" in pilot:
    # The original instant flat speed deficit, two-sided and for every command again (legged v6, user
    # 2026-10-24 ~12:10: "之前的做法但是双向 ... 说不定这样就已经解决了问题，如果不行再用这个方法"): also charged
    # above 110% of the command. On Go2 v5 2000's own speed traces, the one-sided term is lowest at 127-130% of a
    # 0.15-0.3 m/s command (the v3 overspeed), the two-sided one at 96-99%; the stride swing still costs ~0.09 at
    # 0.15 m/s at any average speed.
    deficit = rewards["flat_speed_deficit"]
    deficit.func = mdp.stair_speed_deficit
    deficit.params = {**deficit.params, "ratio": 0.9, "ratio_high": 1.1}
  if "diagaxis" in pilot and wheeled:
    # The flat speed deficit judges diagonals per axis (wheeled group, v13w, 2026-10-27 ~04:00; see
    # mdp.stair_speed_deficit axis_min): the Go2W v12w teacher failed its 3000 / 4000 gates walking (0.6, +-0.4)
    # at x 103-108%, y 67-78% (bar 80%), the speed along the command 0.96x, inside the 0.9-1.1 band; only the
    # tracking kernel (~14%) saw it, and v9w-v11w swung between 57 and 85% on it. 0.2 m/s keeps the stride
    # swing of small parts out (the diagonal commands draw |vx| >= 0.3, |vy| >= 0.2).
    deficit = rewards["flat_speed_deficit"]
    deficit.params = {**deficit.params, "axis_min": 0.2}
  if "stairboth" in pilot:
    # The stairs speed two-sided too (v9, user 2026-10-25 ~04:10: "既然实际速度 / 指令速度的中位数在 0.8–1.2 倍之间，
    # 是不是应该使用双向"): the stair speed deficit also charges above 1.1x the command, as "speedboth" on even
    # ground, and no height term pays for a step taken faster than 1.2x, the acceptance's top (the mean speed
    # since that foot's last record): the descent pay was capped at 1.6x and the climb pay not at all, both
    # paying per metre, so going faster than the band earned more. The 0.9x floor of the deficit keeps the
    # descents from the 0.7x a 1.1x cap gave them without it (round 5).
    deficit = rewards["stair_speed_deficit"]
    deficit.params = {**deficit.params, "ratio_high": 1.1}
    for name in ("foothold_height_gain", "foothold_height_drop"):
      if name in rewards:
        term = rewards[name]
        term.params = {**term.params, "max_speed_ratio": 1.2, "step_average": True, "min_gain": 0.02}
  if "stairpay16" in pilot:
    # The height pay cut off at 1.6x again, the two-sided deficit kept (legged v8e, 2026-10-25 ~17:25): with
    # the 1.2x cut-off the descents settle well under the band, Go2 v8c mixed 5000 / 6000 at 0.85 / 0.77x and
    # every map3 flight at 0.75-0.79x (v8b 0.87 / 0.81x, the na50 pilot 0.80x). The mean speed since a foot's
    # last record swings far above the flight's (a 1.11x descent had 28% of its records over 1.1x, round 4),
    # so a cut-off c leaves the flights at ~0.65c (1.1 -> 0.66-0.74x, 1.2 -> 0.77x); without one they ran
    # 1.21-1.24x (2026-10-21), which the deficit's 1.1x top now charges.
    for name in ("foothold_height_gain", "foothold_height_drop"):
      if name in rewards:
        rewards[name].params = {**rewards[name].params, "max_speed_ratio": 1.6}
  if "stairline" in pilot:
    # The line costs judge flat ground only (legged v8c candidate, 2026-10-25 ~09:30): on
    # stairs only the heading counts, at -1 (the seventh round's heading-only line on stairs).
    # With "straightw3", the offset share and the sideways offset at -3 were the largest cost
    # on Go2 v8b's up-stairs (scratchpad climb_attrib.py, mixed 5000 at vx 0.5: ~1.1 per second
    # climbing 12-16 cm risers, ~0.6 stalled at the first riser, tracking ~0.5), and from mixed
    # 5000 to 7000 the robots took to stalling there: 12-15 cm up 62 -> 8%, map3 climbs 28 -> 2%.
    for name in ("straight_drift", "side_drift"):
      if name in rewards:
        term = rewards[name]
        term.params = {**term.params, "uneven_sensor_name": "terrain_scan", "uneven_scale": 1.0 / abs(term.weight)}
  if "stairshare" in pilot:
    # On stairs the line cost counts the sideways offset as a share of the distance walked again, next to the
    # heading, at the stairs' -1 (As2 / B2 pilot, 2026-10-26 ~01:00): with "stairline" alone the wide robots slid
    # off map3's flights without an operator, B2 v8e noop descents 89 -> 69 -> 74% at 5000-7000 (op 100%; 23-38%
    # ended against the rails, the rest within -0.2..+0.2 m), As2 v8e 88 -> 88 -> 83%; Go2 drifts as far but fits.
    # The share, not the offset: the offset grows with the distance walked, which made Go2 v8b stall at the risers.
    # Wheeled group from v13wd (2026-10-27 ~18:30): As2W v13w mixed 6000 / 7000 noop 77 / 78% (op 99.7%), ~20% of
    # each flight swinging 25-35 deg at the first up riser or 70-100 deg on the first down steps into the rails
    # (illegal contact; scratchpad w13/noop_term.py).
    for name in ("straight_drift", "side_drift"):
      if name in rewards and "uneven_sensor_name" in rewards[name].params:
        rewards[name].params = {**rewards[name].params, "uneven_share": True}
  if "stairhead" in pilot and "heading_error" in cfg.observations["critic"].terms:
    # On stairs the line cost's heading part is the observed heading error, clamped at 3 (0.51 rad, past the
    # observation's own 0.5) instead of 1 (wheeled group from v13we, 2026-10-27 ~23:10): As2W v13wd mixed 6000
    # noop 77.9% with stairshare (v13w 77 / 78%), op 99.1%. The same 10 of 64 robots (env ids) ended early in all
    # four flights traced (scratchpad w13/noop_draws.py; lag 1 40-47%, lag 0 / 2 none), turning 0.3-0.5 rad/s
    # while climbing the first risers to 30 deg into the rails or carrying 16-20 deg from marching in place
    # before the start (noop_trace.py), never turning back: past 0.17 rad (10 deg) the heading part was
    # saturated, so turning back earned nothing and cost yaw tracking, and it measured from the heading at the
    # command's start, so a turn made standing never counted.
    drift = rewards["straight_drift"]
    if "uneven_sensor_name" in drift.params:
      drift.params = {**drift.params, "uneven_heading_obs": "heading_error", "uneven_turn_max": 3.0}
  if "gaitw1" in pilot and "foot_gait" in rewards:
    # The gait clock pays twice (B2 only, v8d, 2026-10-25 ~11:05): the B2 v8 teacher failed its flat gate at
    # 5000 on the clock walking back at 1.0 m/s alone (match 87% at 3000 -> 79 -> 77%, the bar 80%): every foot
    # landed 0.11-0.16 of a period early and lifted at 0.48-0.49 instead of 0.56 (scratchpad b2fix/clockdiag.py),
    # stepping at the clock's rate; B2 v7 missed the same commands at 4000 (73-77%). Go2 / As2 match 37 of 37.
    rewards["foot_gait"].weight = 2.0 * rewards["foot_gait"].weight
  if "riserprox" in pilot and wheeled and "wheel_riser_lift" in rewards:
    # A wheel near a riser counts as blocked whatever its contact force (Go2W only, mixed / polish v13wc,
    # 2026-10-27 ~15:20): the trigger wanted |F_xy| > 1.2 |F_z|, and a wheel that carries its share of the weight
    # never gets there. Go2W v13w mixed 6000 (anchored and not) climbed no riser in any band: its front wheels
    # stood 0-10 cm from the first riser pushing 20 N against 45 N of weight (scratchpad w13/riser_diag.py),
    # 1-3% of them triggered, and the lift pay stayed at ~0.003 per episode from 4000 to 6250 (with it on, 45%
    # of the stalled wheels count). As2W under the same recipe climbs every band at 5800 (0.5 m/s, 4-7% of its
    # front wheels triggering), as the from-scratch wheeled pilots (2026-10-18/19) did by spinning their
    # wheels against the risers. The riser must still stand 4 cm above the wheel's best foothold within the
    # scan radius, so flat ground and descents pay nothing; the pay per riser and wheel is unchanged.
    # From the teacher on for the wheeled group from v13wd (2026-10-27 ~18:30, B2W first): B2W's v13wb teacher 4000
    # stood 0.8 m in at the first riser pushing 65 N against 250 N, 0.1-0.3% triggering, 0% up, and the Go2W / B2W
    # teachers (8% stairs) gained ~0.002 per second of foothold height against As2W's 0.006-0.036; the Go2W mixed
    # with it from such a teacher rose slowly and still climbed nothing at 6000.
    lift = rewards["wheel_riser_lift"]
    lift.params = {**lift.params, "ratio": 0.0, "min_force": 0.0, "tolerance": 0.04}
  if "liftflat" in pilot and wheeled:
    # Wheeled v10 (2026-10-24 ~01:40): the wheel lift cost stays at -4, no ramp to -12. The -12
    # fought trotting the gait clock taught (fixed at the root by wheelclock: v8 rolled with
    # 98.7-100% of the wheels down at 400 under the ramp's -4), and lifts are noisy contact
    # events: with the ramp the value loss rose 0.4 -> 1.0, the thigh / calf / wheel action std
    # grew 0.8 -> 1.3-1.5 (v1 at -4: flat) and Go2W, As2W v8 collapsed at ~1900 / ~1800.
    rewards["wheel_air_drivable"].weight = -4.0
  if phase == "teacher":
    ramps = _staged_ramps(cfg, drift=(500, 2000), hip=(500, 2000), rest=(1000, 3000), march=(1000, 2500))
    if wheeled:
      # Without "liftflat": -4 -> the final weight. -12 from the start (As2W v8b, 2026-10-23 ~20:00)
      # never learned to drive (action std 1.0 -> 1.39, ~1.5 m/s whatever the command, falls) by 400.
      if "liftflat" not in pilot:
        ramps.append({"kind": "reward_weight", "name": "wheel_air_drivable", "start": -4.0,
                      "end": rewards["wheel_air_drivable"].weight,
                      "start_step": 500 * NUM_STEPS_PER_ENV, "end_step": 2000 * NUM_STEPS_PER_ENV})
    else:
      ramps.append({"kind": "reward_weight", "name": "foot_swing_clearance", "start": -0.5, "end": swing_weight,
                    "start_step": 500 * NUM_STEPS_PER_ENV, "end_step": 3000 * NUM_STEPS_PER_ENV})
      # Before the ramps that read it; falls do not pause it here.
      cfg.curriculum["staged_clock"] = CurriculumTermCfg(
        func=mdp.staged_clock, params={"max_fall_rate": 1.0, "level_drop": None, "walk_ratio": TEACHER_WALK_RATIO}
      )
    cfg.curriculum["ramps"] = CurriculumTermCfg(func=mdp.ramp_terms, params={"ramps": ramps})
    _stairs_share(cfg, TEACHER_STAIRS_SHARE, TEACHER_STAIRS_SHARE, 0, 0)
  elif phase in ("mixed", "polish"):
    if "hip_torque" in cfg.rewards:
      cfg.rewards["hip_torque"].params["uneven_scale"] = 1.0  # Final weights otherwise.
    # Pilots of the mixed phase without the flat anchor (2026-10-25 ~12:30; unanchored, Go2 climbed but
    # drifted walking sideways and lost the diagonal gait clock): "stairs50" keeps half the robots on
    # flat ground (the stairs share ends at 50%, not 65%).
    final = 0.5 if "stairs50" in pilot else FINAL_STAIRS_SHARE
    if "stairs35" in pilot:
      # B2W v13wq (2026-11-01): its mixed slid on flat items as the share rose, the averages at 7000 (stairs ~0.27-0.36
      # of the robots over 6300..7000) drifting back 5.6% walking sideways and descending at 0.801x, at 8000
      # (0.41-0.50) 6.2-7.2% and 0.755x (bars 5% / 0.8x), climbing 99-100% at both: the polish holds 0.35.
      final = 0.35
    if phase == "mixed":
      start = int(os.environ.get("BLIND_PHASE_START_IT", PHASE_MIXED_START))
      cfg.curriculum["staged_clock"] = CurriculumTermCfg(
        func=mdp.staged_clock, params={"max_fall_rate": 0.3, "level_drop": None}
      )
      _stairs_share(cfg, TEACHER_STAIRS_SHARE, final, start, start + PHASE_MIXED_RAMP)
    else:
      _stairs_share(cfg, final, final, 0, 0)
  else:
    raise ValueError(f"Unknown phase {phase!r}.")
  return cfg


# The trot clock keeps a stride of this many leg lengths (thigh + calf) per period at
# speed (see apply_speed_clock): Go2 0.48 m, which its phased teachers walked at the
# 0.6 s clock up to 0.8 m/s.
GAIT_STRIDE_PER_LEG = 1.13


def apply_speed_clock(
  cfg: ManagerBasedRlEnvCfg,
  leg_length: float,
  foot_xy: tuple[tuple[float, float], ...] | None = None,
  foot_xy_hip0: tuple[tuple[float, float], ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """The trot clock quickens at speed (user 2026-10-23 ~14:25, for the gait clock
  failures at 1.0 m/s: "步频随速度变化……同样要保证原地旋转和低速运动的时候不会出现扭动身体的情况").
  Above ref_speed = GAIT_STRIDE_PER_LEG x `leg_length` / period (Go2 / As2 0.8 m/s,
  B2 1.31 m/s) the period of the phase observation and of foot_gait shortens so the
  stride stays GAIT_STRIDE_PER_LEG leg lengths (mdp.gait_clock): Go2 steps 3.1 times
  per second per foot at 1.5 m/s, where its phased teachers stepped 2.5-3.2 times at
  1.0-1.5 m/s against the 0.6 s clock's 1.67. Slower commands, turning in place,
  marching and standing keep the period, so the robots still step there. The deploy
  side's gait_phase takes the same ref_speed. Only with "speedclock" in BLIND_PILOT
  until the next round starts (the running teachers and their evaluations have the
  fixed clock)."""
  pilot = os.environ.get("BLIND_PILOT", "").split(",")
  if "speedclock" not in pilot:
    return cfg
  if "hipzero" in pilot and foot_xy_hip0 is not None:
    foot_xy = foot_xy_hip0  # The feet at the default stance with the hips at 0 (see apply_no_hip_load).
  terms = [cfg.observations[g].terms.get("phase") for g in ("actor", "critic")] + [cfg.rewards.get("foot_gait")]
  for term in terms:
    if term is not None:
      term.params = {**term.params, "ref_speed": GAIT_STRIDE_PER_LEG * leg_length / term.params["period"]}
      if "arcclock" in pilot and foot_xy is not None:
        # The clock follows the fastest foot's speed (legged v7, 2026-10-24 ~16:20; see mdp.gait_clock):
        # the v4 and v6 teachers failed the clock only on the 0.8 m/s, 0.8 rad/s arcs (67-78%), stepping
        # 1.85-1.90 (Go2) / 1.92-2.00 (As2) per second where the fastest foot's speed gives 1.89 / 1.93.
        # Turning in place, slow commands and marching keep the period (the feet stay below ref_speed).
        term.params = {**term.params, "foot_xy": foot_xy}
  return cfg


def apply_grip(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Firmer contact friction (2026-10-24): an elliptic friction cone at impratio 10 instead of
  MuJoCo's pyramidal cone at impratio 1, under which wheels and feet slid apart under a steady
  sideways load well inside the friction limit, as static friction would not let them on the
  robot (Go2W model_400 standing 12 s: hips +1.8 -> +4.8 deg out, +0.5 with this; the hip torque
  grew with it, and once its cost ramped up the wheeled policies lifted wheels to set them back).
  Only with "grip" in BLIND_PILOT: it is physics, so a run's training and evaluation tasks must
  both have it or both not."""
  if "grip" not in os.environ.get("BLIND_PILOT", "").split(","):
    return cfg
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.mujoco.impratio = 10.0
  return cfg


BRAKE_MAX_FORWARD = 0.05
"""Forward command (m/s) up to which "brakeside" brakes the wheels (the deploy's brake_max_forward)."""


def apply_side_brake(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """The wheel brake also holds while the forward command is at most BRAKE_MAX_FORWARD
  (wheeled group, v13wl, 2026-10-30; see BrakedJointVelocityAction.brake_max_forward):
  stepping sideways or turning in place on braked wheels, as on feet. Only with
  "brakeside" in BLIND_PILOT: it is part of the controller, so a run's training and
  evaluation tasks must both have it or both not, and so must the deploy."""
  if "brakeside" not in os.environ.get("BLIND_PILOT", "").split(","):
    return cfg
  wheel_vel = cfg.actions.get("wheel_vel")
  if isinstance(wheel_vel, mdp.BrakedJointVelocityActionCfg):
    wheel_vel.brake_max_forward = BRAKE_MAX_FORWARD
  return cfg


def apply_leg_gains(cfg: ManagerBasedRlEnvCfg, kp_scale: float = 1.5, kd_scale: float = 1.2) -> ManagerBasedRlEnvCfg:
  """Stiffer leg PD gains (B2W only; kp 160 -> 240, kd 5 -> 6), only with "legkp15" in BLIND_PILOT: it is
  physics, so a run's training and evaluation tasks must both have it or both not, and the deploy
  (export_deploy_yaml.py) takes the gains from the config. B2W's stop creep (1-5 cm per 4 s in every v13w* run,
  bar 1) comes from a stance that does not hold itself at kp 160: the B2W final (polish 7000) crept 0.75-2.22 cm
  with its wheels turning <= 0.06 cm, the base moving over them as the legs moved; with its actions frozen
  1.5 s after the stop 0.68-13.9 cm at kp 160, 0.21-0.91 at these gains (2026-10-29, scratchpad
  w17/creep_split.py). Alone (v13wi, 2026-10-29 ~01:00) it left the policy re-posing after stops (creep
  0.18-3.28 at 1000-2000); "standtgt" holds the targets."""
  if "legkp15" not in os.environ.get("BLIND_PILOT", "").split(","):
    return cfg
  entity = cfg.scene.entities["robot"]
  articulation = entity.articulation
  assert articulation is not None
  actuators = []
  for actuator in articulation.actuators:
    base = actuator.base_cfg if isinstance(actuator, DelayedActuatorCfg) else actuator
    if not _velocity_controlled(base):
      base = replace(base, stiffness=kp_scale * base.stiffness, damping=kd_scale * base.damping)
    actuators.append(replace(actuator, base_cfg=base) if isinstance(actuator, DelayedActuatorCfg) else base)
  cfg.scene.entities["robot"] = replace(entity, articulation=replace(articulation, actuators=tuple(actuators)))
  return cfg


def apply_no_hip_load(cfg: ManagerBasedRlEnvCfg, robot: EntityCfg) -> ManagerBasedRlEnvCfg:
  """No hip-load shaping (user 2026-10-24 ~15:45: "我觉得可以从配方中取消掉这个力矩相关的内容
  ... 可以退回到之前走楼梯垂直于地面或者是和默认姿态一样。因为之后散热可以通过原地踏步来解决"):
  the default stance back to the robot model's own (`robot`'s hip angles; the third round had
  turned the hips inward, the wheeled sixth round to the rest stance), no hip torque or splay
  cost, no rest stance (the standing pose terms target the default stance) and no splayed
  resets. Hot hips are the deploy's ThermalGuard's to cool by marching in place; on stairs
  stance_leg_tilt still holds the stance legs at the default stance's sideways tilt. The
  hip-load rounds 3-7 left the legged policies standing wider than their default and never
  in the rest stance, As2W with crossed legs (default -0.38 rad) and Go2W splaying out
  driving, and their costs ran up to -0.7 per second (tracking ~0.75). Only with
  "nohipload" in BLIND_PILOT: the default stance sets the policy's joint coordinates, so a
  run's training and evaluation tasks must both have it or both not.

  With "hipzero" too (all robots, v8; user 2026-10-24 ~23:20, "统一默认都改成零"): every hip's
  default is 0, the legs vertical seen from the front, as the deploy's FixStand ends and as
  the stance legs stand on stairs; the robot models' own defaults turned the hips 5.7 deg in
  (Go2, As2, As2W: unitree_rl_mjlab) or out (B2: unitree_rl_lab). Thighs and calves keep the
  model's angles."""
  pilot = os.environ.get("BLIND_PILOT", "").split(",")
  if "nohipload" not in pilot:
    return cfg
  entity = cfg.scene.entities["robot"]
  joint_pos = {k: v for k, v in entity.init_state.joint_pos.items() if "hip_joint" not in k}
  if "hipzero" in pilot:
    joint_pos[".*_hip_joint"] = 0.0
  else:
    joint_pos |= {k: v for k, v in robot.init_state.joint_pos.items() if "hip_joint" in k}
  entity.init_state = replace(entity.init_state, joint_pos=joint_pos)
  for name in ("hip_torque", "hip_splay"):
    cfg.rewards.pop(name, None)
  for name in ("pose", "stand_still_even"):
    term = cfg.rewards.get(name)
    if term is not None and "rest_hip" in term.params:
      term.params = {k: v for k, v in term.params.items() if k != "rest_hip"}
  cfg.events.pop("reset_hip_splay", None)
  return cfg


def apply_wheel_stepping_clock(cfg: ManagerBasedRlEnvCfg, training: bool = True) -> ManagerBasedRlEnvCfg:
  """Wheeled robots of the staged run step to the gait clock wherever they step on
  flat ground (user 2026-10-22 ~21:30: turning in place and other lifts in the clock's
  rhythm): the clock runs at the legged robots' 0.6 s, and the gait reward is the
  legged trot (diagonal pairs, half a period apart left to right) gated to marching
  and to commands the wheels cannot drive (feet_gait stepping_only), on even ground;
  stair edges are still lifted over freely. The seventh round's four-beat 1 s march
  could not step sideways at 0.6 m/s (the stance legs would sweep ~0.45 m)."""
  for group in ("actor", "critic"):
    cfg.observations[group].terms["phase"].params["period"] = 0.6
  pilot = os.environ.get("BLIND_PILOT", "").split(",")
  # "arcstep" (wheeled v9w, 2026-10-26 ~08:55): the wheels drive turns only up to 0.03 rad/s, not
  # 0.1; above it the robot steps to the clock as on other arcs. The As2W v8w teacher failed its
  # 3000 / 4000 gates on (1.0, 0, 0.1), turning at 19-56%: at 4000 (scratchpad wheel8/wz_diag.py)
  # it held all four wheels down and turned 56-88% of 0.05-0.1 rad/s at 1 m/s, 14-53% at 0.5 m/s
  # (wheels cannot steer, so a rolling turn skids), but 81-94% of 0.02-0.03 rad/s at 0.3-1 m/s,
  # and stepping from 0.12 rad/s up it turned 100%.
  turn_limit = 0.03 if "arcstep" in pilot else 0.1
  if "wheelclock" in pilot:
    # The clock runs only where the robot steps: zero under commands the wheels drive
    # (the limits of wheel_air_drivable and feet_gait stepping_only; see mdp.phase).
    # Only with "wheelclock" in BLIND_PILOT until the next round starts.
    for group in ("actor", "critic"):
      terms = cfg.observations[group].terms
      terms["phase"].params = {
        **terms["phase"].params, "drivable_max_lin_vel_y": 0.1, "drivable_max_ang_vel_z": turn_limit
      }
  if not training:
    return cfg
  gait = cfg.rewards["foot_gait"]
  drivable = cfg.rewards["wheel_air_drivable"].params
  if "arcstep" in pilot:
    drivable["max_ang_vel_z"] = turn_limit
  gait.params = {
    **gait.params,
    "period": 0.6,
    "offset": [0.0, 0.5, 0.5, 0.0],
    "threshold": 0.56,
    "march_only": False,
    "stepping_only": True,
    "max_lin_vel_y": drivable.get("max_lin_vel_y", 0.1),
    "max_ang_vel_z": drivable.get("max_ang_vel_z", 0.1),
    "scan_sensor_name": "terrain_scan",
  }
  return cfg
