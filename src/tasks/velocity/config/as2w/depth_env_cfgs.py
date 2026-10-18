"""Unitree As2-W depth-camera velocity configurations (teacher-student).

The As2's tasks with the wheels as feet, as for the Go2-W: legs position-
controlled, wheels velocity-controlled; it rolls on even ground and steps
on stairs or to move sideways (mdp/wheels.py).
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_as2w_robot_cfg
from src.assets.robots.unitree_as2w.as2w_constants import As2W_WHEEL_RADIUS
from src.tasks.velocity.config.as2.depth_env_cfgs import AS2_SPEC
from src.tasks.velocity.config.as2w.env_cfgs import unitree_as2w_rough_env_cfg
from src.tasks.velocity.config.legged.depth import (
  make_depth_student_env_cfg,
  make_depth_teacher_env_cfg,
)

AS2W_SPEC = replace(
  AS2_SPEC,
  name="as2w",
  rough_env_cfg=unitree_as2w_rough_env_cfg,
  robot_cfg=get_as2w_robot_cfg,
  wheel_joints=r"^[FR][LR]_wheel_joint$",
  wheel_radius=As2W_WHEEL_RADIUS,
  foot_radius=As2W_WHEEL_RADIUS,
  # Wheel centre 0.164 m up = the Go2 foot's 0.078 m sole clearance.
  swing_target_height=As2W_WHEEL_RADIUS + 0.078,
  # Wheel spheres reach 0.31 m each side.
  wall_safe_distance=0.42,
  # What the Go2-W (same size and wheel) and the B2-W needed: the wheel-sized
  # edge rule, looser hips for side steps, a strong tracking-gated step-mode
  # trot reward and a swing penalty that only applies once it steps.
  terrain_clearance_weight=-10.0,
  terrain_clearance_margin=0.02,
  hip_walking_std=0.45,
  step_gait_weight=2.0,
  swing_weight_stages=((0, 0.0), (1000, -1.0), (3000, -2.0)),
  onnx_contract="as2w_depth_v1",
)


def unitree_as2w_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(AS2W_SPEC, play=play, stairs=stairs)


def unitree_as2w_depth_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(AS2W_SPEC, play=play)
