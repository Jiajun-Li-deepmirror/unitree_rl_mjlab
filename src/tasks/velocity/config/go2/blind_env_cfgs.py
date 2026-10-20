"""Unitree Go2 blind (proprioception-only) velocity configurations.

The task itself is src/tasks/velocity/blind_env_cfg.py; this module gives it the
Go2 model and adds the Go2 map3 fine-tune.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_go2_robot_cfg
from src.tasks.velocity.blind_env_cfg import (
  FINAL_VEL_RANGES as FINAL_VEL_RANGES,
)
from src.tasks.velocity.blind_env_cfg import (
  MAP3_STAIR_TERRAIN_NAMES as MAP3_STAIR_TERRAIN_NAMES,
)
from src.tasks.velocity.blind_env_cfg import (
  WALLED_TERRAIN_NAMES as WALLED_TERRAIN_NAMES,
)
from src.tasks.velocity.blind_env_cfg import (
  BlindRobotCfg,
  apply_blind_map3,
  blind_collision_cfg,
  make_blind_env_cfg,
)
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_env_cfg

GO2_BLIND = BlindRobotCfg(
  base_env_cfg=unitree_go2_rough_env_cfg,
  robot_cfg=get_go2_robot_cfg,
  collision=blind_collision_cfg(
    foot=r"^[FR][LR]_foot_collision$", calf=r"^[FR][LR]_calf\d_collision$"
  ),
  body_contact_patterns=(r"base\d_collision", r"[FR][LR]_hip_collision"),
  leg_contact_patterns=(r"[FR][LR]_thigh_collision", r"[FR][LR]_calf\d_collision"),
)


def unitree_go2_blind_env_cfg(
  play: bool = False,
  stairs: bool = True,
  terrain_clearance: bool = True,
  foothold_gain: bool = True,
  stance_tilt_weight: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go2 blind velocity configuration (see make_blind_env_cfg)."""
  return make_blind_env_cfg(
    replace(GO2_BLIND, stance_tilt_weight=stance_tilt_weight),
    play=play,
    stairs=stairs,
    terrain_clearance=terrain_clearance,
    foothold_gain=foothold_gain,
  )


def unitree_go2_blind_map3_env_cfg(
  play: bool = False, stance_tilt_weight: float | None = None
) -> ManagerBasedRlEnvCfg:
  """Unitree-Go2-Blind fine-tuned for map3 (see apply_blind_map3): same
  observations and actions. With `stance_tilt_weight`, stance legs on stairs
  keep the default stance's sideways tilt (stance_leg_tilt; the release opened
  its front hips 7 deg)."""
  cfg = unitree_go2_blind_env_cfg(play=play, stairs=True, stance_tilt_weight=stance_tilt_weight)
  return apply_blind_map3(cfg, play)
