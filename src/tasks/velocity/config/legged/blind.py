"""Blind (proprioception-only) velocity configurations (go2-blind).

The actor sees no height scan, only proprioception with a short history. The
critic keeps privileged terrain information. Commands mimic a joystick on flat
ground and are forward-biased on stairs. Robot specifics come from a
LeggedTaskSpec.
"""

import math
from dataclasses import replace

import mjlab.terrains as terrain_gen
from mjlab.actuator import BuiltinVelocityActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.spec_config import CollisionCfg

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.config.legged.spec import LeggedTaskSpec
from src.tasks.velocity.mdp.velocity_command import (
  TerrainAwareVelocityCommandCfg,
  UniformVelocityCommandCfg,
)

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


def blind_collision(spec: LeggedTaskSpec) -> CollisionCfg:
  """The robots' FULL_COLLISION, but calves get friction (condim 3) so shins
  resting on a stair nose don't slide off frictionlessly."""
  foot, calf = spec.foot_collision, spec.calf_collision
  return CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={foot: 3, calf: 3, ".*_collision": 1},
    priority={foot: 1},
    friction={foot: (0.6,)},
    solimp={foot: (0.9, 0.95, 0.023)},
    contype=1,
    conaffinity=0,
  )


def _stairs(cls, step_width: float) -> terrain_gen.SubTerrainCfg:
  return cls(
    proportion=0.14,
    step_height_range=(0.05, 0.18),
    step_width=step_width,
    platform_width=2.0,
    border_width=0.5,
  )


def make_blind_terrains_cfg(stairs: bool) -> TerrainGeneratorCfg:
  """Flat (+ light roughness and up/down pyramid stairs when `stairs`)."""
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
      "stairs_down_narrow": _stairs(terrain_gen.BoxPyramidStairsTerrainCfg, 0.25),
      "stairs_down_wide": _stairs(terrain_gen.BoxPyramidStairsTerrainCfg, 0.32),
      # Spawn in the pit and walk up.
      "stairs_up_narrow": _stairs(terrain_gen.BoxInvertedPyramidStairsTerrainCfg, 0.25),
      "stairs_up_wide": _stairs(terrain_gen.BoxInvertedPyramidStairsTerrainCfg, 0.32),
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


def blind_robot_cfg(spec: LeggedTaskSpec) -> EntityCfg:
  robot_cfg = spec.robot_cfg()
  articulation = robot_cfg.articulation
  assert articulation is not None
  # Per-env lag (0-20 ms) is set at reset by dr.sync_actuator_delays and held
  # for the episode, identical across the hip/thigh/calf groups.
  delayed_actuators = tuple(
    DelayedActuatorCfg(
      base_cfg=actuator,
      delay_target=(
        "velocity" if isinstance(actuator, BuiltinVelocityActuatorCfg) else "position"
      ),
      delay_min_lag=0,
      delay_max_lag=4,
      delay_hold_prob=1.0,
    )
    for actuator in articulation.actuators
  )
  return replace(
    robot_cfg,
    collisions=(blind_collision(spec),),
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
  spec: LeggedTaskSpec, play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  """Create the blind velocity configuration of `spec`'s robot.

  Args:
    play: Evaluation settings (no noise or pushes, random terrain per reset).
    stairs: Include stairs and rough patches; otherwise flat ground only. Both
      variants share observation shapes, sensors and rewards.
  """
  cfg = spec.rough_env_cfg(play=play)

  cfg.scene.entities = {"robot": blind_robot_cfg(spec)}

  # The rough defaults (ccd_iterations=500, njmax=1500) run out of memory at
  # 4096 envs on a 16 GB GPU. Measured peak on the hardest stairs (random
  # actions) is ~56 constraints per env; collision primitives are simple.
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.njmax = 300
  cfg.sim.contact_sensor_maxmatch = 128
  if spec.nconmax is not None:
    cfg.sim.nconmax = spec.nconmax

  ##
  # Terrain.
  ##

  assert cfg.scene.terrain is not None
  terrain_generator = make_blind_terrains_cfg(stairs)
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
  body_ground_cfg = _robot_contact_sensor("body_ground_touch", spec.body_ground_patterns)
  leg_ground_cfg = _robot_contact_sensor("leg_ground_touch", spec.leg_ground_patterns)
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
    params={"asset_cfg": SceneEntityCfg("robot", site_names=spec.foot_site_names)},
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
      "asset_cfg": SceneEntityCfg("robot", body_names=(spec.base_body,)),
      "operation": "add",
      "ranges": spec.base_payload_range,
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
    return SceneEntityCfg("robot", body_names=(spec.base_body,))

  def feet() -> SceneEntityCfg:
    return SceneEntityCfg("robot", site_names=spec.foot_site_names)

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
      r".*(FR|FL|RR|RL)_hip_joint.*": spec.hip_walking_std,
    }
  rewards["foot_slip"].weight = -0.5

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
  rewards["foot_swing_clearance"] = RewardTermCfg(
    func=mdp.feet_swing_clearance,
    weight=-3.0,
    params={
      "sensor_name": "feet_ground_contact",
      "target_height": spec.swing_target_height,
      **standing,
      "asset_cfg": feet(),
    },
  )
  if spec.swing_weight_stages:
    rewards["foot_swing_clearance"].weight = spec.swing_weight_stages[0][1]
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

  if spec.wheel_joints is not None:
    _add_wheels(cfg, spec)

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
    if spec.swing_weight_stages:
      cfg.curriculum["swing_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
          "reward_name": "foot_swing_clearance",
          "weight_stages": [
            {"step": it * NUM_STEPS_PER_ENV, "weight": w} for it, w in spec.swing_weight_stages
          ],
        },
      )
    if stairs:
      cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=mdp.terrain_levels_cmd_distance,
        params={"command_name": "twist"},
      )

  return cfg


def _add_wheels(cfg: ManagerBasedRlEnvCfg, spec: LeggedTaskSpec) -> None:
  """Wheel-legged robots: velocity-controlled wheels as feet (mdp/wheels.py).

  Legs keep the legged actions, posture terms and gait; wheel angles are left
  out of the observations and posture terms (they are unbounded). The trot
  gait is only rewarded in step mode, and foot slip is measured at the wheel
  contact point, so rolling is free; in roll mode the legs should stay still
  with all wheels down.
  """
  assert spec.wheel_joints is not None

  def legs() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=(spec.leg_joints,))

  leg_action = cfg.actions["joint_pos"]
  assert isinstance(leg_action, JointPositionActionCfg)
  leg_action.actuator_names = (spec.leg_joints,)
  cfg.actions["wheel_vel"] = JointVelocityActionCfg(
    entity_name="robot",
    actuator_names=(spec.wheel_joints,),
    scale=spec.wheel_action_scale,
    use_default_offset=False,
  )

  for group in ("actor", "critic"):
    cfg.observations[group].terms["joint_pos"].params["asset_cfg"] = legs()

  # dr.pd_gains handles position actuators only. It indexes the entity's
  # actuator groups (one per articulation actuator cfg) by actuator_ids.
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  position_groups = [
    i
    for i, a in enumerate(articulation.actuators)
    if not isinstance(getattr(a, "base_cfg", a), BuiltinVelocityActuatorCfg)
  ]
  cfg.events["pd_gains"].params["asset_cfg"] = SceneEntityCfg(
    "robot", actuator_ids=position_groups
  )

  # A rare NaN physics state (1 env in ~10M steps on the first Go2-W run,
  # likely a wheel cylinder contact, since replaced by spheres) crashed
  # training; reset such envs.
  cfg.terminations["nan_state"] = TerminationTermCfg(func=mdp.nan_detection)

  rewards = cfg.rewards
  for name in ("pose", "stand_still_even", "joint_acc_l2", "joint_pos_limits"):
    rewards[name].params["asset_cfg"] = legs()

  scan = {"scan_sensor_name": "terrain_scan"}
  standing = {"command_name": "twist", "command_threshold": 0.1}
  gait = rewards["foot_gait"]
  gait.func = mdp.feet_gait_stepping
  gait.params |= scan

  rewards.pop("foot_slip")
  rewards["wheel_slip"] = RewardTermCfg(
    func=mdp.wheel_slip,
    weight=-0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "wheel_radius": spec.wheel_radius,
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot",
        site_names=spec.foot_site_names,
        body_names=tuple(f"{s}_wheel" for s in spec.foot_site_names),
        preserve_order=True,
      ),
    },
  )
  # Roll mode. At -0.002 / +0.5 (legs moving / wheels down) alone the first
  # Go2-W trotted on flat ground anyway (feet touching down at ~1.8 Hz):
  # also hold the default leg pose and reward wheel speeds that drive the
  # command like a differential drive.
  rewards["roll_leg_motion"] = RewardTermCfg(
    func=mdp.roll_leg_motion,
    weight=-0.01,
    params={**standing, **scan, "asset_cfg": legs()},
  )
  rewards["roll_leg_deviation"] = RewardTermCfg(
    func=mdp.roll_leg_deviation,
    weight=-2.0,
    params={**standing, **scan, "asset_cfg": legs()},
  )
  rewards["roll_wheel_contact"] = RewardTermCfg(
    func=mdp.roll_wheel_contact,
    weight=2.0,
    params={**standing, **scan, "sensor_name": "feet_ground_contact"},
  )
  rewards["wheel_roll_tracking"] = RewardTermCfg(
    func=mdp.wheel_roll_tracking,
    weight=1.0,
    params={
      **standing,
      **scan,
      "wheel_radius": spec.wheel_radius,
      "std": 3.0,
      # Wheel joints listed in the foot site order (FR, FL, RR, RL).
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=tuple(f"{s}_wheel_joint" for s in spec.foot_site_names),
        site_names=spec.foot_site_names,
        preserve_order=True,
      ),
    },
  )
