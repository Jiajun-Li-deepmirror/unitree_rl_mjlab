"""Unitree As2W blind (proprioception-only) velocity configuration.

The task is the wheel-legged variant in src/tasks/velocity/blind_env_cfg.py:
legs take position targets, wheels velocity targets.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_as2w_robot_cfg
from src.tasks.velocity.blind_env_cfg import (
  WheeledBlindRobotCfg,
  apply_blind_heading,
  apply_blind_map3,
  apply_blind_straight,
  blind_collision_cfg,
  make_wheeled_blind_env_cfg,
)
from src.tasks.velocity.quadruped_env_cfg import make_quadruped_rough_env_cfg

# Wheel kd 1.0 saturates the 15 Nm wheel torque at a 15 rad/s velocity error.
AS2W_BLIND = WheeledBlindRobotCfg(
  base_env_cfg=lambda play: make_quadruped_rough_env_cfg(get_as2w_robot_cfg, play),
  robot_cfg=get_as2w_robot_cfg,
  collision=blind_collision_cfg(
    foot=r"^[FR][LR]_foot_collision$", calf=r"^[FR][LR]_calf\d_collision$"
  ),
  body_contact_patterns=(r"base\d_collision", r"[FR][LR]_hip_collision"),
  leg_contact_patterns=(r"[FR][LR]_thigh\d_collision", r"[FR][LR]_calf\d_collision"),
  payload_range=(-1.0, 3.0),
  wheel_radius=0.086,
  wheel_action_scale=8.0,
)


def unitree_as2w_blind_env_cfg(play: bool = False, stairs: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree As2W blind velocity configuration (see make_wheeled_blind_env_cfg)."""
  return make_wheeled_blind_env_cfg(AS2W_BLIND, play=play, stairs=stairs)


# Stairs (2026-10-20), the recipe of the Go2W stairs runs: stance-leg tilt, map3
# walled flights, straight walking without an operator, the heading-error input,
# the mirror loss and the flat-ground anchor to the v5 flat model (flat_v5_gbrake model_5300).
# Lifted wheels while driving straight on even ground cost 4x (see GO2W_BLIND_STAIRS).
AS2W_BLIND_STAIRS = replace(AS2W_BLIND, stance_tilt_weight=-10.0, drivable_air_weight=-4.0)


def unitree_as2w_blind_heading_env_cfg(
  play: bool = False, terrain: str = "map3"
) -> ManagerBasedRlEnvCfg:
  """The As2W stairs tasks with the heading-error input: `terrain` "map3" is
  the training task (map3 walled flights with the wheeled terms, see
  apply_blind_map3; apply_blind_straight; apply_blind_heading), "stairs" and
  "flat" the evaluation tasks of the same policy. Start from the flat model
  widened by scripts/blind/widen_obs.py, then scripts/blind/make_stairs_init.py."""
  if terrain == "map3":
    cfg = make_wheeled_blind_env_cfg(AS2W_BLIND_STAIRS, play=play, stairs=True)
    cfg = apply_blind_straight(apply_blind_map3(cfg, play, legged_terms=False))
  else:
    cfg = make_wheeled_blind_env_cfg(AS2W_BLIND_STAIRS, play=play, stairs=terrain == "stairs")
  return apply_blind_heading(cfg)
