"""Unitree Go2-W depth-camera velocity configurations (teacher-student).

The Go2's tasks (config/legged) with the wheels as feet: legs position-
controlled, wheels velocity-controlled; it rolls on even ground and steps
on stairs or to move sideways (mdp/wheels.py). Same stairs, commands and
camera as the Go2 (same body and head).
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_go2w_robot_cfg
from src.tasks.velocity.config.go2w.env_cfgs import unitree_go2w_rough_env_cfg
from src.tasks.velocity.config.legged.depth import (
  make_depth_student_env_cfg,
  make_depth_teacher_env_cfg,
)
from src.tasks.velocity.config.legged.spec import LeggedTaskSpec

# Wheel collision spheres in go2w.xml (*_foot_collision).
GO2W_WHEEL_RADIUS = 0.086

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
  # The Go2's walking hip std (0.3) as go2-blind's Go2W: its swing penalty and
  # wheel_slip make the side steps (0.45 had been tried at the Go2's 45% vy).
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
  # command can be rolled).
  wheel_blind_recipe=True,
  # go2-blind's wheeled v5 (its flat models pass every flat gate) has neither
  # the even-ground tracking std nor the slow commands of its legged map3
  # stage, which these had taken over.
  even_track_std=None,
  flat_slow_prob=0.0,
  # A foothold overspeed gate, wider than the legged default: without one the
  # progress rewards' discounting paid for rolling down the flights (B2-W
  # full_v2ft descended map3 at 1.8x the command at 10500 and 3-5x by 12000,
  # Go2-W full_v6ft at 1.35-2.6x; the legged Go2 needs none: 1.1-1.2x). At
  # (0.05, 0.25), judged on the speed at the foothold, where the body lurches
  # forward, they went ~0.1 m/s under the command instead (B2-W climbs 0.8x
  # and stalled atop the L0 descent, Go2-W 0.88x; the legged Go2 v10 had
  # descended at 0.6-0.75x).
  foothold_overspeed_range=(0.15, 0.4),
  # go2-blind's Go2W v5: arcs with a small yaw rate in the commands, and the
  # low-speed brake that stopped the creep at a zero command (2.9 -> 0.01 cm).
  steer_prob=0.2,
  wheel_brake_speed=0.1,
  onnx_contract="go2w_depth_v2",
)


def unitree_go2w_depth_teacher_env_cfg(
  play: bool = False, stairs: bool = True, heading: bool = False
) -> ManagerBasedRlEnvCfg:
  return make_depth_teacher_env_cfg(GO2W_SPEC, play=play, stairs=stairs, heading=heading)


def unitree_go2w_depth_student_env_cfg(
  play: bool = False, heading: bool = False
) -> ManagerBasedRlEnvCfg:
  return make_depth_student_env_cfg(GO2W_SPEC, play=play, heading=heading)
