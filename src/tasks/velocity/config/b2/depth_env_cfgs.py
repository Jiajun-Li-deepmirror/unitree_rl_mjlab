"""Unitree B2 depth-camera velocity configurations (teacher-student).

Same tasks as the Go2's (config/legged). The B2 is much bigger (legs 0.35 +
0.35 m, 83.5 kg, ~0.5 m wide over the hips), so map3's 14-19 cm rises are
easy for it while its ~0.9 m flights are tight: it keeps more side clearance
and lifts its feet higher. Stairs and commands stay the Go2's (map3, tinynav
command envelope).
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_b2_robot_cfg
from src.tasks.velocity.config.b2.env_cfgs import unitree_b2_rough_env_cfg
from src.tasks.velocity.config.legged.depth import (
  make_depth_student_env_cfg,
  make_depth_teacher_env_cfg,
)
from src.tasks.velocity.config.legged.spec import LeggedTaskSpec

B2_SPEC = LeggedTaskSpec(
  name="b2",
  rough_env_cfg=unitree_b2_rough_env_cfg,
  robot_cfg=get_b2_robot_cfg,
  nconmax=128,  # 61 contacts in the initial state.
  base_payload_range=(-5.0, 10.0),
  swing_target_height=0.12,
  foot_radius=0.032,
  wall_safe_distance=0.40,
  # Go2 camera orientation, 1.5 cm ahead of the B2's front panel (x 0.43 m at
  # z -0.06-0.10 m). No real extrinsics yet.
  camera_pos=(0.445, 0.0, 0.05),
  onnx_contract="b2_depth_v1",
)


def unitree_b2_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(B2_SPEC, play=play, stairs=stairs)


def unitree_b2_depth_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(B2_SPEC, play=play)
