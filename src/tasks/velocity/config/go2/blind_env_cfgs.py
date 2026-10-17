"""Unitree Go2 blind (proprioception-only) velocity configurations.

The actor sees no height scan, only proprioception with a short history. The
critic keeps privileged terrain information. Commands mimic a joystick on flat
ground and are forward-biased on stairs.
"""

import math
from dataclasses import replace

import mjlab.terrains as terrain_gen
from mjlab.actuator import DelayedActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.spec_config import CollisionCfg

import src.tasks.velocity.mdp as mdp
from src.assets.robots import get_go2_robot_cfg
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_env_cfg
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

_FOOT_COLLISION = r"^[FR][LR]_foot_collision$"
_CALF_COLLISION = r"^[FR][LR]_calf\d_collision$"

# Same as go2_constants.FULL_COLLISION, but calves get friction (condim 3) so
# shins resting on a stair nose don't slide off frictionlessly.
BLIND_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_FOOT_COLLISION: 3, _CALF_COLLISION: 3, ".*_collision": 1},
  priority={_FOOT_COLLISION: 1},
  friction={_FOOT_COLLISION: (0.6,)},
  solimp={_FOOT_COLLISION: (0.9, 0.95, 0.023)},
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


def make_go2_blind_terrains_cfg(stairs: bool) -> TerrainGeneratorCfg:
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


def _go2_blind_robot_cfg() -> EntityCfg:
  robot_cfg = get_go2_robot_cfg()
  articulation = robot_cfg.articulation
  assert articulation is not None
  # Per-env lag (0-20 ms) is set at reset by dr.sync_actuator_delays and held
  # for the episode, identical across the hip/thigh/calf groups.
  delayed_actuators = tuple(
    DelayedActuatorCfg(
      base_cfg=actuator,
      delay_target="position",
      delay_min_lag=0,
      delay_max_lag=4,
      delay_hold_prob=1.0,
    )
    for actuator in articulation.actuators
  )
  return replace(
    robot_cfg,
    collisions=(BLIND_COLLISION,),
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


def unitree_go2_blind_env_cfg(
  play: bool = False,
  stairs: bool = True,
  terrain_clearance: bool = True,
  foothold_gain: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go2 blind velocity configuration.

  Args:
    play: Evaluation settings (no noise or pushes, random terrain per reset).
    stairs: Include stairs and rough patches; otherwise flat ground only. Both
      variants share observation shapes, sensors and rewards.
    terrain_clearance: Add foot_terrain_clearance and make the swing clearance
      one-sided on uneven ground.
    foothold_gain: Add foothold_height_gain.
    The released policy was trained with both off up to iteration 6000, with
    terrain_clearance from there and with both from 9000 (doc/go2_blind.md).
  """
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.scene.entities = {"robot": _go2_blind_robot_cfg()}

  # The rough defaults (ccd_iterations=500, njmax=1500) run out of memory at
  # 4096 envs on a 16 GB GPU. Measured peak on the hardest stairs (random
  # actions) is ~56 constraints per env; collision primitives are simple.
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.njmax = 300
  cfg.sim.contact_sensor_maxmatch = 128

  ##
  # Terrain.
  ##

  assert cfg.scene.terrain is not None
  terrain_generator = make_go2_blind_terrains_cfg(stairs)
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
  body_ground_cfg = _robot_contact_sensor(
    "body_ground_touch", (r"base\d_collision", r"[FR][LR]_hip_collision")
  )
  leg_ground_cfg = _robot_contact_sensor(
    "leg_ground_touch", (r"[FR][LR]_thigh_collision", r"[FR][LR]_calf\d_collision")
  )
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
      "ranges": (-1.0, 2.0),
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
    "target_height": 0.10,
    **standing,
    "asset_cfg": feet(),
  }
  if terrain_clearance:
    swing_params["scan_sensor_name"] = "terrain_scan"
  rewards["foot_swing_clearance"] = RewardTermCfg(
    func=mdp.feet_swing_clearance, weight=-3.0, params=swing_params
  )
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
    if stairs:
      cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=mdp.terrain_levels_cmd_distance,
        params={"command_name": "twist"},
      )

  return cfg


##
# map3 fine-tune: walled flights like the tinynav-sim map3 stairwell.
##

WALLED_TERRAIN_NAMES = ("walled_down", "walled_up")
MAP3_STAIR_TERRAIN_NAMES = STAIR_TERRAIN_NAMES + WALLED_TERRAIN_NAMES


def make_go2_blind_map3_terrains_cfg() -> TerrainGeneratorCfg:
  """The blind mix with less pyramid area plus walled map3-like flights.

  Walled flights are 0.85-1.5 m wide (narrower with difficulty), with treads
  of 0.18-0.32 m and per-step rise noise; map3 has 0.19 m treads and
  16-19 cm rises where the blind policy got stuck. Proportions are multiples
  of 1/20 so each type gets whole columns. Column order puts flat ground next
  to the walled tiles: robots leaving a pyramid tile walk straight on and hit
  the (to them invisible) walls of the next column.
  """
  cfg = make_go2_blind_terrains_cfg(stairs=True)
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


def unitree_go2_blind_map3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Unitree-Go2-Blind fine-tuned for map3: same observations and actions.

  Changes vs Unitree-Go2-Blind: walled flights in the terrain (with an
  operator-style corridor command and little spawn x/y noise there), a tighter
  linear tracking std on even ground plus more slow flat commands (flat
  low-speed under-tracking), and a progress reward for going down stairs.
  """
  cfg = unitree_go2_blind_env_cfg(play=play, stairs=True)

  assert cfg.scene.terrain is not None
  terrain_generator = make_go2_blind_map3_terrains_cfg()
  if play:
    terrain_generator.num_rows = 5
    terrain_generator.num_cols = 20
    terrain_generator.border_width = 10.0
  cfg.scene.terrain.terrain_generator = terrain_generator

  twist = cfg.commands["twist"]
  assert isinstance(twist, TerrainAwareVelocityCommandCfg)
  twist.stair_terrain_names = MAP3_STAIR_TERRAIN_NAMES
  twist.corridor_terrain_names = WALLED_TERRAIN_NAMES
  twist.flat_slow_prob = 0.3

  reset_base = cfg.events["reset_base"]
  reset_base.params["stair_terrain_names"] = MAP3_STAIR_TERRAIN_NAMES
  reset_base.params["narrow_terrain_names"] = WALLED_TERRAIN_NAMES

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
