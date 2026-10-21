"""Unitree As2 depth-camera velocity configurations (teacher-student).

Same tasks as the Go2's (config/legged); the As2 is Go2-sized (legs 0.21 +
0.21 m, 20.7 kg in go2-blind's unitree_mujoco model) with stiffer actuators,
so the Go2 stairs, commands and reward scales carry over. Only its geometry
differs.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_as2_robot_cfg
from src.tasks.velocity.config.as2.env_cfgs import unitree_as2_rough_env_cfg
from src.tasks.velocity.config.legged.depth import (
  make_depth_student_env_cfg,
  make_depth_teacher_env_cfg,
)
from src.tasks.velocity.config.legged.spec import LeggedTaskSpec

AS2_SPEC = LeggedTaskSpec(
  name="as2",
  rough_env_cfg=unitree_as2_rough_env_cfg,
  robot_cfg=get_as2_robot_cfg,
  # go2-blind's As2 contact split: thigh1 is the motor cylinder at the thigh
  # root, next to the hip.
  body_ground_patterns=(
    r"base\d_collision", r"[FR][LR]_hip_collision", r"[FR][LR]_thigh1_collision"
  ),
  leg_ground_patterns=(r"[FR][LR]_thigh2_collision", r"[FR][LR]_calf\d_collision"),
  nconmax=64,
  foot_radius=0.028,
  # Go2 camera orientation, 1.4 cm ahead of the As2 head (face at x 0.311 m,
  # z 0.04-0.11 m). No real extrinsics yet.
  camera_pos=(0.325, 0.0, 0.075),
  onnx_contract="as2_depth_v1",
  # go2-blind's legged tracking terms and no foothold overspeed gate, as the
  # Go2's (its accepted As2 / B2 stairs models train with them).
  even_track_std=0.25,
  flat_slow_prob=0.3,
  foothold_overspeed_range=None,
  # Seeing the steps, its heading teacher ran down map3 at 3-5x the command
  # (go2-blind's blind ones ~1.2x): its second stairs round caps foothold
  # progress at 1.1x.
  foothold_max_speed_ratio=1.1,
)


def unitree_as2_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True, heading: bool = False
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(AS2_SPEC, play=play, stairs=stairs, heading=heading)


def unitree_as2_depth_student_env_cfg(
  play: bool = False, heading: bool = False
) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(AS2_SPEC, play=play, heading=heading)
