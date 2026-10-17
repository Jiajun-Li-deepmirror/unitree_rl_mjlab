"""Unitree As2 depth-camera velocity configurations (teacher-student).

Same tasks as the Go2's (config/legged); the As2 is Go2-sized (legs 0.21 +
0.21 m, 17.6 kg) with stiffer actuators, so the Go2 stairs, commands and
reward scales carry over. Only its geometry differs.
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
  # No hip collision geoms; thigh1 is the knee motor housing at the hip.
  body_ground_patterns=(r"base\d_collision", r"[FR][LR]_thigh1_collision"),
  leg_ground_patterns=(r"[FR][LR]_thigh[23]_collision", r"[FR][LR]_calf\d_collision"),
  # 36 contacts already in the initial state (Go2: 35 is enough).
  nconmax=64,
  # Go2 camera orientation, 1.4 cm ahead of the As2 head (face at x 0.311 m,
  # z 0.04-0.11 m). No real extrinsics yet.
  camera_pos=(0.325, 0.0, 0.075),
  onnx_contract="as2_depth_v1",
)


def unitree_as2_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(AS2_SPEC, play=play, stairs=stairs)


def unitree_as2_depth_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(AS2_SPEC, play=play)
