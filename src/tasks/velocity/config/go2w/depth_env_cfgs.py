"""Unitree Go2-W depth-camera velocity configurations (teacher-student).

The Go2's tasks (config/legged) with the wheels as feet: legs position-
controlled, wheels velocity-controlled; it rolls on even ground and steps
on stairs or to move sideways (mdp/wheels.py). Same stairs, commands and
camera as the Go2 (same body and head).
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_go2w_robot_cfg
from src.assets.robots.unitree_go2w.go2w_constants import GO2W_WHEEL_RADIUS
from src.tasks.velocity.config.go2w.env_cfgs import unitree_go2w_rough_env_cfg
from src.tasks.velocity.config.legged.depth import (
  make_depth_student_env_cfg,
  make_depth_teacher_env_cfg,
)
from src.tasks.velocity.config.legged.spec import LeggedTaskSpec

GO2W_SPEC = LeggedTaskSpec(
  name="go2w",
  rough_env_cfg=unitree_go2w_rough_env_cfg,
  robot_cfg=get_go2w_robot_cfg,
  wheel_joints=r"^[FR][LR]_wheel_joint$",
  wheel_radius=GO2W_WHEEL_RADIUS,
  foot_radius=GO2W_WHEEL_RADIUS,
  # Wheel centre 0.164 m up = the Go2 foot's 0.078 m sole clearance.
  swing_target_height=GO2W_WHEEL_RADIUS + 0.078,
  # Wheel spheres reach 0.28 m each side (Go2 feet: 0.16 m).
  wall_safe_distance=0.40,
  onnx_contract="go2w_depth_v1",
)


def unitree_go2w_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(GO2W_SPEC, play=play, stairs=stairs)


def unitree_go2w_depth_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(GO2W_SPEC, play=play)
