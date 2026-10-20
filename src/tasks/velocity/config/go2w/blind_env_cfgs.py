"""Unitree Go2W blind (proprioception-only) velocity configuration.

The task is the wheel-legged variant in src/tasks/velocity/blind_env_cfg.py:
legs take position targets, wheels velocity targets.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_go2w_robot_cfg
from src.tasks.velocity.blind_env_cfg import (
  WheeledBlindRobotCfg,
  apply_blind_heading,
  apply_blind_map3,
  apply_blind_straight,
  blind_collision_cfg,
  make_wheeled_blind_env_cfg,
)
from src.tasks.velocity.quadruped_env_cfg import make_quadruped_rough_env_cfg

# Wheel kd is 0.5 Nm s/rad, so a 15 Nm wheel torque needs a 30 rad/s velocity
# error: the action scale leaves room for that.
GO2W_BLIND = WheeledBlindRobotCfg(
  base_env_cfg=lambda play: make_quadruped_rough_env_cfg(get_go2w_robot_cfg, play),
  robot_cfg=get_go2w_robot_cfg,
  collision=blind_collision_cfg(
    foot=r"^[FR][LR]_foot_collision$", calf=r"^[FR][LR]_calf\d_collision$"
  ),
  body_contact_patterns=(r"base\d_collision", r"[FR][LR]_hip_collision"),
  leg_contact_patterns=(r"[FR][LR]_thigh_collision", r"[FR][LR]_calf\d_collision"),
  payload_range=(-1.0, 2.0),
  wheel_radius=0.086,
  wheel_action_scale=10.0,
)


def unitree_go2w_blind_env_cfg(play: bool = False, stairs: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go2W blind velocity configuration (see make_wheeled_blind_env_cfg)."""
  return make_wheeled_blind_env_cfg(GO2W_BLIND, play=play, stairs=stairs)


# Stairs (2026-10-20): the stairs task plus the stance-leg tilt term of the legged
# stairs runs, fine-tuned from the v5 flat model (flat_v5_gbrake model_6997).
GO2W_BLIND_STAIRS = replace(GO2W_BLIND, stance_tilt_weight=-10.0)


def unitree_go2w_blind_map3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The Go2W stairs task with the map3 walled flights (terrain, corridor commands
  and spawns only; see apply_blind_map3)."""
  cfg = make_wheeled_blind_env_cfg(GO2W_BLIND_STAIRS, play=play, stairs=True)
  return apply_blind_map3(cfg, play, legged_terms=False)


def unitree_go2w_blind_straight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The Go2W map3 stairs task that walks straight without an operator (see
  apply_blind_straight); trained with the flat-ground anchor to the v5 flat
  model (rl.anchor), as the first map3 fine-tune lost the flat skills."""
  return apply_blind_straight(unitree_go2w_blind_map3_env_cfg(play=play))


def unitree_go2w_blind_heading_env_cfg(
  play: bool = False, terrain: str = "map3"
) -> ManagerBasedRlEnvCfg:
  """The Go2W stairs tasks with the heading-error input (see apply_blind_heading),
  fine-tuned from Unitree-Go2W-Blind-Map3-Straight widened by
  scripts/blind/widen_obs.py: `terrain` "map3" is the training task, "stairs"
  and "flat" the evaluation tasks of the same policy."""
  if terrain == "map3":
    cfg = unitree_go2w_blind_straight_env_cfg(play=play)
  else:
    cfg = make_wheeled_blind_env_cfg(GO2W_BLIND_STAIRS, play=play, stairs=terrain == "stairs")
  return apply_blind_heading(cfg)
