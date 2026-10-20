"""Unitree B2W blind (proprioception-only) velocity configuration.

The task is the wheel-legged variant in src/tasks/velocity/blind_env_cfg.py:
legs take position targets, wheels velocity targets.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_b2w_robot_cfg
from src.tasks.velocity.blind_env_cfg import (
  WheeledBlindRobotCfg,
  blind_collision_cfg,
  make_wheeled_blind_env_cfg,
)
from src.tasks.velocity.quadruped_env_cfg import make_quadruped_rough_env_cfg

# Wheel kd 2.0 saturates the 40 Nm wheel torque (unitree.com/b2-w) at a 20 rad/s
# velocity error.
# nconmax 64: at qpos0 (legs straight) two calf boxes per leg reach into the ground.
# Stairs: the 0.226 m wheel needs wider treads than the 0.25/0.32 m of the others.
B2W_BLIND = WheeledBlindRobotCfg(
  base_env_cfg=lambda play: make_quadruped_rough_env_cfg(get_b2w_robot_cfg, play),
  robot_cfg=get_b2w_robot_cfg,
  collision=blind_collision_cfg(
    foot=r"^[FR][LR]_foot_collision$", calf=r"^[FR][LR]_calf\d_collision$"
  ),
  body_contact_patterns=(r"base\d_collision", r"[FR][LR]_hip_collision"),
  leg_contact_patterns=(r"[FR][LR]_thigh_collision", r"[FR][LR]_calf\d_collision"),
  payload_range=(-5.0, 10.0),
  nconmax=64,
  stair_widths=(0.30, 0.40),
  wheel_radius=0.113,
  wheel_action_scale=6.0,
  # Yaw inertia 7.74 kg m^2: -0.025 * (0.53 / 7.74)^2. At -0.025 a 1.5 rad/s turn
  # cost 3.4 per step against at most 1.0 of tracking, and the first flat run
  # (2026-10-18) did not turn faster than ~0.4 rad/s.
  angular_momentum_weight=-1.2e-4,
)


def unitree_b2w_blind_env_cfg(play: bool = False, stairs: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree B2W blind velocity configuration (see make_wheeled_blind_env_cfg)."""
  return make_wheeled_blind_env_cfg(B2W_BLIND, play=play, stairs=stairs)
