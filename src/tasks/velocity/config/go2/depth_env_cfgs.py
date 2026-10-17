"""Unitree Go2 depth-camera velocity configurations (teacher-student).

The generic builders live in config/legged/depth.py.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.velocity.config.go2.blind_env_cfgs import GO2_SPEC
from src.tasks.velocity.config.legged.depth import (
  DEPTH_SENSOR_NAME,  # noqa: F401 (scripts/go2_depth)
  make_depth_student_env_cfg,
  make_depth_teacher_env_cfg,
)


def unitree_go2_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(GO2_SPEC, play=play, stairs=stairs)


def unitree_go2_depth_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(GO2_SPEC, play=play)
