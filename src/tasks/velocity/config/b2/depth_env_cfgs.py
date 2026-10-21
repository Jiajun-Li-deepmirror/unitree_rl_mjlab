"""Unitree B2 depth-camera velocity configurations (teacher-student).

Same tasks as the Go2's (config/legged). The B2 is much bigger (legs 0.35 +
0.35 m, 74.6 kg in go2-blind's unitree_mujoco model, ~0.5 m wide over the
hips), so map3's 14-19 cm rises are easy for it while its ~0.9 m flights are
tight: it keeps more side clearance and lifts its feet higher. Stairs and
commands stay the Go2's.
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
  # go2-blind's B2 model at qpos0 needs 344 rows on the stairs (376 the B2-W).
  njmax=512,
  base_payload_range=(-5.0, 10.0),
  swing_target_height=0.12,
  # Go2's -3 swing penalty froze it (stood still for any command); -1 from the
  # start broke free slowly. Start at -0.5 and tighten once it walks.
  swing_weight_stages=((0, -0.5), (3000, -1.5)),
  foot_radius=0.032,
  wall_safe_distance=0.40,
  # Go2 camera orientation, 1.5 cm ahead of the B2's front panel (x 0.43 m at
  # z -0.06-0.10 m). No real extrinsics yet.
  camera_pos=(0.445, 0.0, 0.05),
  onnx_contract="b2_depth_v1",
  # go2-blind's legged tracking terms and no foothold overspeed gate, as the
  # Go2's (its accepted As2 / B2 stairs models train with them).
  even_track_std=0.25,
  flat_slow_prob=0.3,
  foothold_overspeed_range=None,
  # Seeing the steps, its heading teacher ran down map3 at 3-5x the command
  # (go2-blind's blind ones ~1.2x): foothold progress capped at 1.1x, on the
  # speed averaged over each step.
  foothold_max_speed_ratio=1.1,
)


def unitree_b2_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True, heading: bool = False
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(B2_SPEC, play=play, stairs=stairs, heading=heading)


def unitree_b2_depth_student_env_cfg(
  play: bool = False, heading: bool = False
) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(B2_SPEC, play=play, heading=heading)
