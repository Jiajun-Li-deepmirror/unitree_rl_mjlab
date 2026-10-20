"""Blind (proprioception-only) velocity task for Unitree quadrupeds.

The actor sees no height scan, only proprioception with a short history. The
critic keeps privileged terrain information. Commands mimic a joystick on flat
ground and are forward-biased on stairs. Robot-specific inputs (model, collision
names, swing height, payload range) come in a BlindRobotCfg; the robot's base
velocity config must follow the repo conventions: base body `base_link`, foot
sites FR/FL/RR/RL, foot collision geoms `{leg}_foot_collision`.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

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
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
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
  gait_clock: bool = True
  """Reward a trot that follows a clock, fed to the policy as the phase
  observation. Without it the policy finds its own gait."""
  gait_period: float | None = None
  """Period of the trot clock if not the base config's 0.6 s (Go2-sized legs)."""
  hip_action_scale: float | None = None
  """Action scale of the hip joints if not the base config's 0.25."""
  body_contact_termination: bool = True
  """End the episode when a body_contact_patterns geom touches the ground."""
  termination_weight: float | None = None
  """Override of the is_terminated penalty (default -200)."""
  tracking_weights: tuple[float, float] | None = None
  """Override of the linear and angular velocity tracking weights (default 1.0 each)."""
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
  if not robot.body_contact_termination:
    cfg.terminations.pop("illegal_contact")

  ##
  # Actions.
  ##

  if robot.hip_action_scale is not None:
    joint_pos = cfg.actions["joint_pos"]
    assert isinstance(joint_pos, JointPositionActionCfg)
    assert isinstance(joint_pos.scale, float)
    joint_pos.scale = {
      r".*_hip_joint": robot.hip_action_scale,
      r".*_(thigh|calf)_joint": joint_pos.scale,
    }

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
  for group in (actor, critic):
    if "phase" not in group.terms:
      continue
    if not robot.gait_clock:
      group.terms.pop("phase")
    elif robot.gait_period is not None:
      group.terms["phase"].params["period"] = robot.gait_period

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
  if robot.termination_weight is not None:
    rewards["is_terminated"].weight = robot.termination_weight
  if robot.tracking_weights is not None:
    lin_weight, ang_weight = robot.tracking_weights
    rewards["track_linear_velocity"].weight = lin_weight
    rewards["track_angular_velocity"].weight = ang_weight
  if not robot.gait_clock:
    rewards.pop("foot_gait")
  elif robot.gait_period is not None:
    rewards["foot_gait"].params["period"] = robot.gait_period

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
    func=mdp.wheel_air_drivable, weight=-1.0, params=drivable_params
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
  command and little spawn x/y noise there), and with `legged_terms` a tighter
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
