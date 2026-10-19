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
  # Swing clearance is the wheel centre's rise between footholds, which is
  # the sole's: 0.10 as on go2-blind's wheeled robots (radius + 0.078 = 0.164
  # asked for 16 cm steps and made dragging the wheels nearly as cheap).
  swing_target_height=0.10,
  # Wheel spheres reach 0.28 m each side (Go2 feet: 0.16 m).
  wall_safe_distance=0.40,
  # Side steps reached only ~45% of vy at the Go2's 0.3.
  hip_walking_std=0.45,
  # The Go2's edge rule (sole 5 cm over any edge within 12 cm ahead, -40)
  # asks a 0.086 m wheel to lift its centre ~22 cm before an 8 cm step:
  # the first teacher never stepped up even 5 cm rises (descents were fine).
  terrain_clearance_weight=-10.0,
  terrain_clearance_margin=0.02,
  # full_v4ft model_12500 (roll/step modes, ungated 0.5 trot reward, swing
  # -1/-2 at 0.164) steps backward and reaches vy ~0.4; fine-tuning it with
  # the B2-W's tracking-gated 2.0 (full_v5ft) raised vy to 0.65 but it trotted
  # at every forward speed and climbed no map3 flight. Now go2-blind's wheeled
  # recipe (swing -3 at 0.10, nothing paying for steps, wheels down while the
  # command can be rolled) and its tracking terms; no foothold overspeed gate.
  wheel_blind_recipe=True,
  even_track_std=0.25,
  flat_slow_prob=0.3,
  foothold_overspeed_range=None,
  onnx_contract="go2w_depth_v1",
)


def unitree_go2w_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(GO2W_SPEC, play=play, stairs=stairs)


def unitree_go2w_depth_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(GO2W_SPEC, play=play)
