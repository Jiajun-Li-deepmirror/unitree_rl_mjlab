"""Unitree B2-W depth-camera velocity configurations (teacher-student).

The B2's tasks with the wheels as feet, as for the Go2-W: legs position-
controlled, wheels velocity-controlled; it rolls on even ground and steps
on stairs or to move sideways (mdp/wheels.py).
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_b2w_robot_cfg
from src.assets.robots.unitree_b2w.b2w_constants import B2W_WHEEL_RADIUS
from src.tasks.velocity.config.b2.depth_env_cfgs import B2_SPEC
from src.tasks.velocity.config.b2w.env_cfgs import unitree_b2w_rough_env_cfg
from src.tasks.velocity.config.legged.depth import (
  make_depth_student_env_cfg,
  make_depth_teacher_env_cfg,
)

B2W_SPEC = replace(
  B2_SPEC,
  name="b2w",
  rough_env_cfg=unitree_b2w_rough_env_cfg,
  robot_cfg=get_b2w_robot_cfg,
  wheel_joints=r"^[FR][LR]_wheel_joint$",
  wheel_radius=B2W_WHEEL_RADIUS,
  foot_radius=B2W_WHEEL_RADIUS,
  # Wheel centre 0.201 m up = the B2 foot's 0.088 m sole clearance.
  swing_target_height=B2W_WHEEL_RADIUS + 0.088,
  # Wheel spheres reach 0.35 m each side (B2 feet: 0.23 m).
  wall_safe_distance=0.45,
  # The first B2-W flat teacher only rolled: it never lifted a wheel (no side
  # steps, vy 0; turning in place 20-36%). Reward the trot harder where it
  # has to step, and let it find stepping before the swing penalty applies.
  step_gait_weight=2.0,
  swing_weight_stages=((0, 0.0), (1000, -0.5), (3000, -1.5)),
  onnx_contract="b2w_depth_v1",
)


def unitree_b2w_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(B2W_SPEC, play=play, stairs=stairs)


def unitree_b2w_depth_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(B2W_SPEC, play=play)
