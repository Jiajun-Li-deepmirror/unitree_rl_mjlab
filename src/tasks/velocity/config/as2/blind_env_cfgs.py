"""Unitree As2 blind (proprioception-only) velocity configuration.

The task is src/tasks/velocity/blind_env_cfg.py. As2 legs are close to Go2's
(0.21 m thigh and calf, ~21 kg), so the Go2 swing height and payload range are
kept.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_as2_robot_cfg
from src.tasks.velocity.blind_env_cfg import (
  BlindRobotCfg,
  apply_blind_map3,
  blind_collision_cfg,
  make_blind_env_cfg,
)
from src.tasks.velocity.config.as2.env_cfgs import unitree_as2_rough_env_cfg

AS2_BLIND = BlindRobotCfg(
  base_env_cfg=unitree_as2_rough_env_cfg,
  robot_cfg=get_as2_robot_cfg,
  collision=blind_collision_cfg(
    foot=r"^[FR][LR]_foot_collision$", calf=r"^[FR][LR]_calf\d_collision$"
  ),
  # thigh1 is the motor cylinder at the thigh root, next to the hip.
  body_contact_patterns=(
    r"base\d_collision", r"[FR][LR]_hip_collision", r"[FR][LR]_thigh1_collision"
  ),
  leg_contact_patterns=(r"[FR][LR]_thigh2_collision", r"[FR][LR]_calf\d_collision"),
  # At 1000 stairs iterations, 0.07 rad^2 per step climbing (front hips 12 deg
  # open): -0.7 per second, against at most 1 per second for linear tracking.
  stance_tilt_weight=-10.0,
)


def unitree_as2_blind_env_cfg(play: bool = False, stairs: bool = True) -> ManagerBasedRlEnvCfg:
  """Create the Unitree As2 blind velocity configuration (see make_blind_env_cfg)."""
  return make_blind_env_cfg(AS2_BLIND, play=play, stairs=stairs)


def unitree_as2_blind_map3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The As2 stairs task with the Go2 map3 fine-tune changes (see apply_blind_map3).

  As2 is Go2-sized, so the Go2 walled flights fit. At 1700 stairs iterations
  (pyramid stairs only) it reached map3 goals in time 0-61% (Go2 release
  5-97%), with 16-28% resets on the L0 flight and its 0.188 m tread.
  """
  return apply_blind_map3(unitree_as2_blind_env_cfg(play=play, stairs=True), play)
