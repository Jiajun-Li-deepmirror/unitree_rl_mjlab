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
  # Swing clearance is the wheel centre's rise between footholds (see Go2-W).
  swing_target_height=0.10,
  # Wheel spheres reach 0.31 m each side.
  wall_safe_distance=0.42,
  # The wheel-sized edge rule and looser hips of the Go2-W (same size and
  # wheel), and go2-blind's wheeled recipe and tracking terms (see GO2W_SPEC).
  terrain_clearance_weight=-10.0,
  terrain_clearance_margin=0.02,
  hip_walking_std=0.45,
  wheel_blind_recipe=True,
  even_track_std=0.25,
  flat_slow_prob=0.3,
  foothold_overspeed_range=None,
  steer_prob=0.2,
  wheel_brake_speed=0.1,
  onnx_contract="as2w_depth_v2",
)


def unitree_as2w_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(AS2W_SPEC, play=play, stairs=stairs)


def unitree_as2w_depth_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(AS2W_SPEC, play=play)
