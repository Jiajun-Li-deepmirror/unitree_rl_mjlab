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
  # Wheel spheres reach 0.35 m each side (B2 feet: 0.23 m).
  wall_safe_distance=0.45,
  # The first B2-W flat teacher only rolled: it never lifted a wheel, most
  # likely because the swing target was the wheel centre at 0.201 m (radius
  # + the B2 sole's 0.088), so at -3 every lower step cost more than standing
  # on the wheels. flat_v2 / full_v1 then stepped with a tracking-gated trot
  # reward (2.0) and the swing penalty in stages, but stepped backward and
  # climbed no map3 flight (full_v1 model_6400). Now go2-blind's wheeled
  # recipe as for the Go2-W (its B2W steps sideways at 89-92% with -3 at
  # 0.10 from the start) and the Go2-W's wheel-sized edge rule.
  swing_target_height=0.10,
  swing_weight_stages=None,
  terrain_clearance_weight=-10.0,
  terrain_clearance_margin=0.02,
  wheel_blind_recipe=True,
  even_track_std=0.25,
  flat_slow_prob=0.3,
  # The Go2-W's foothold overspeed gate (see GO2W_SPEC).
  foothold_overspeed_range=(0.15, 0.4),
  steer_prob=0.2,
  wheel_brake_speed=0.1,
  onnx_contract="b2w_depth_v2",
)


def unitree_b2w_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(B2W_SPEC, play=play, stairs=stairs)


def unitree_b2w_depth_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(B2W_SPEC, play=play)
