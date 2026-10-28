"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # Read-only sensors are also present in baseline evaluation for matched comparisons.
  ground_sensors = tuple(ContactSensorCfg(
    name=f"{side}_foot_ground",
    primary=ContactMatch(mode="body", pattern=f"{side}_ankle_roll_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force", "pos"), reduce="maxforce", num_slots=1,
  ) for side in ("left", "right"))
  cfg.scene.sensors = (self_collision_cfg, *ground_sensors)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  # ankle_roll_link sits ~0.011m above the floor when the foot is flat, so
  # contact_height=0.05 also counts a moderately raised heel as planted.
  cfg.rewards["foot_contact_slip"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)

    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_leg_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Opt-in refinement with the baseline's observations, actuators and safety terms."""
  from mjlab.managers.reward_manager import RewardTermCfg
  from src.tasks.tracking import mdp
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)
  feet = ("left_ankle_roll_link", "right_ankle_roll_link")
  joints = tuple(f"{side}_{part}_joint" for side in ("left", "right")
                 for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
  cfg.rewards["motion_leg_joint_pos"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp, weight=1.0,
    params={"command_name": "motion", "std": .25, "joint_names": joints})
  cfg.rewards["motion_feet_pos"] = RewardTermCfg(
    func=mdp.motion_relative_body_position_error_exp, weight=1.0,
    params={"command_name": "motion", "std": .10, "body_names": feet})
  cfg.rewards["motion_swing_height"] = RewardTermCfg(
    func=mdp.motion_swing_height_error_exp, weight=.5,
    params={"command_name": "motion", "std": .05, "body_names": feet, "swing_height": .06})
  return cfg


def unitree_g1_balanced_leg_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Second experiment: leg tracking with explicit arms and measured-contact support."""
  from mjlab.managers.reward_manager import RewardTermCfg
  from src.tasks.tracking import mdp
  cfg = unitree_g1_leg_tracking_env_cfg(play=play)
  # At a 14-17 cm undershoot, the previous 5 cm exponential is nearly zero.
  # A broader learning signal still has its maximum at the exact target height.
  cfg.rewards["motion_swing_height"].params["std"] = .10
  arms = tuple(f"{side}_{part}_joint" for side in ("left", "right") for part in
               ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                "wrist_roll", "wrist_pitch", "wrist_yaw"))
  cfg.rewards["motion_arm_joint_pos"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp, weight=1.5,
    params={"command_name": "motion", "std": .20, "joint_names": arms})
  cfg.rewards["measured_foot_contact_slip"] = RewardTermCfg(
    func=mdp.motion_measured_contact_slip, weight=-1.0,
    params={"command_name": "motion"})
  return cfg


def unitree_g1_step_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Third experiment: Leg-Rewards policies drag 3-12 cm reference steps along the floor.

  A grounded foot under a 6 cm reference lift still earned 85% of the broad
  swing term, and slip cost under 0.5% of the reward. The tight term starts
  1.5 cm above the 0.035 m flat-foot ankle height and pays only when the lift
  is matched within a few centimetres; the broad term still guides large lifts.
  """
  from mjlab.managers.reward_manager import RewardTermCfg
  from src.tasks.tracking import mdp
  cfg = unitree_g1_balanced_leg_tracking_env_cfg(play=play)
  cfg.rewards["motion_swing_height_tight"] = RewardTermCfg(
    func=mdp.motion_swing_height_error_exp, weight=1.0,
    params={"command_name": "motion", "std": .03,
            "body_names": ("left_ankle_roll_link", "right_ankle_roll_link"), "swing_height": .05})
  cfg.rewards["measured_foot_contact_slip"].weight = -3.0
  return cfg


def unitree_g1_tight_leg_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Fourth experiment: Step-Rewards policies shrink the legs' motion.

  On the video-fit references (2026-10-28) the legs were the worst-tracked group,
  4.6 deg mean in dance2-3: the ankles rolled 64-71% as far as the reference and
  the hips pitched 71-81% as far. The leg joint term's 0.25 rad std still pays 85%
  at a 0.1 rad error, so closing it earns little. The tight term pays 37% there
  and 78% at 0.05 rad.
  """
  from mjlab.managers.reward_manager import RewardTermCfg
  from src.tasks.tracking import mdp
  cfg = unitree_g1_step_tracking_env_cfg(play=play)
  cfg.rewards["motion_leg_joint_pos_tight"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp, weight=1.0,
    params={"command_name": "motion", "std": .10,
            "joint_names": cfg.rewards["motion_leg_joint_pos"].params["joint_names"]})
  return cfg


def unitree_g1_tight_leg_latency_tracking_env_cfg(max_latency_ms: float, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Tight-Legs with one control latency per episode, as Step-Rewards-Latency."""
  from src.tasks.tracking.latency import with_episode_latency
  cfg = unitree_g1_tight_leg_tracking_env_cfg(play=play)
  return cfg if play else with_episode_latency(cfg, max_latency_ms)


def unitree_g1_step_latency_tracking_env_cfg(max_latency_ms: float, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Step-Rewards trained with one control latency per episode, drawn from [0, max_latency_ms].

  Step-Rewards policies tracked well without delay but fell at 20-40 ms in the
  pre-hardware robustness sweep (2026-10-22). Play mode stays undelayed: the
  evaluation adds its own fixed latency.
  """
  from src.tasks.tracking.latency import with_episode_latency
  cfg = unitree_g1_step_tracking_env_cfg(play=play)
  return cfg if play else with_episode_latency(cfg, max_latency_ms)
