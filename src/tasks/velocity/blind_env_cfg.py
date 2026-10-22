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


##
# Staged training from scratch (2026-10-22).
##


def apply_blind_staged(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Train the seventh round's recipe from scratch, the same way for every robot
  (user 2026-10-22 ~21:10: the fine-tune chains started from different models; stages
  by weights, every term there from the first step; stairs always there, fewer
  at first; no map3 in training, which it never had: the walled flights are
  generated with map3-like parameters and map3 itself is only evaluated).

  Ramps (iterations): stairs 15% of the robots -> the terrain's 65% over 0-6000;
  drift costs 30% -> 100% and hip torque / splay costs 10% -> 100% over 2000-8000;
  the rest stance (default -> rest hip angle) and the standing hip torque scale
  (1 -> x3) over 4000-10000 (no extra cost standing on stairs); marching 5% -> 50% of
  zero commands off the stairs, its position hold 10% -> 100%, over 6000-12000.
  Inputs, the default stance and the controller (brake, mirror loss) are final
  from the start. Training task only: evaluation uses the seventh round's tasks.
  """
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
      weight(name, 0.3, 2000, 8000)
  weight("hip_torque", 0.1, 2000, 8000)
  weight("hip_splay", 0.1, 2000, 8000)
  param("hip_torque", "standing_scale", 1.0, 4000, 10000)
  # No extra hip torque cost standing on stairs (user 2026-10-22 ~21:35: robots do not
  # stand long on stairs); stopping mid-flight stays (stair_standing_prob).
  rewards["hip_torque"].params["uneven_scale"] = 1.0
  hip_default = cfg.scene.entities["robot"].init_state.joint_pos[".*L_hip_joint"]
  for name in ("pose", "stand_still_even"):
    if rewards[name].params.get("rest_hip") is not None:
      param(name, "rest_hip", hip_default, 4000, 10000)
  ramps.append({"kind": "command", "name": "twist", "key": "march_prob", "start": 0.05, "end": twist.march_prob,
                "start_step": it(6000), "end_step": it(12000)})
  weight("march_position_hold", 0.1, 6000, 12000)
  cfg.curriculum["ramps"] = CurriculumTermCfg(func=mdp.ramp_terms, params={"ramps": ramps})
  # After terrain_levels (it judges the last episode on the old origins).
  cfg.curriculum["stairs_share"] = CurriculumTermCfg(
    func=mdp.stairs_share,
    params={"stair_terrain_names": MAP3_STAIR_TERRAIN_NAMES, "start": 0.15, "end": 0.65, "start_step": 0, "end_step": it(6000)},
  )
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
  if not training:
    return cfg
  gait = cfg.rewards["foot_gait"]
  drivable = cfg.rewards["wheel_air_drivable"].params
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
