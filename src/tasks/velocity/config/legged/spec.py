"""Per-robot parameters of the blind and depth-camera velocity tasks.

The blind task (go2-blind) and the depth teacher-student tasks were developed
on the Go2; everything robot-specific they need is collected here so other
12-DoF Unitree quadrupeds with the same naming (FR/FL/RR/RL foot sites,
`*_collision` geoms, `base_link`) reuse them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.velocity.mdp import depth as depth_mdp


@dataclass(frozen=True, kw_only=True)
class LeggedTaskSpec:
  name: str
  """Robot name: experiment names and the student's ONNX contract."""
  rough_env_cfg: Callable[..., ManagerBasedRlEnvCfg]
  """The robot's rough-terrain velocity config (`play=` keyword)."""
  robot_cfg: Callable[[], EntityCfg]

  # Collision geoms.
  foot_collision: str = r"^[FR][LR]_foot_collision$"
  calf_collision: str = r"^[FR][LR]_calf\d_collision$"
  body_ground_patterns: tuple[str, ...] = (r"base\d_collision", r"[FR][LR]_hip_collision")
  """Touching the ground with these ends the episode."""
  leg_ground_patterns: tuple[str, ...] = (
    r"[FR][LR]_thigh_collision",
    r"[FR][LR]_calf\d_collision",
  )
  """Touching the ground with these is penalized."""
  foot_site_names: tuple[str, ...] = ("FR", "FL", "RR", "RL")
  base_body: str = "base_link"

  nconmax: int | None = None
  """Contacts per world, if the rough config's default is too small."""

  # Domain randomization and reward scales.
  base_payload_range: tuple[float, float] = (-1.0, 2.0)
  """kg added to the base."""
  swing_target_height: float = 0.10
  foot_radius: float = 0.022

  # Stairs.
  step_height_range: tuple[float, float] = (0.05, 0.20)
  pyramid_step_width: float = 0.28
  wall_safe_distance: float = 0.35
  """Side clearance the policy keeps from corridor walls (base frame, m)."""

  # Wheel-legged robots: wheels are the feet (foot geoms/sites), velocity-
  # controlled; legs stay position-controlled. See mdp/wheels.py.
  wheel_joints: str | None = None
  """Regex of the wheel joints; None for legged robots."""
  leg_joints: str = r"^[FR][LR]_(hip|thigh|calf)_joint$"
  wheel_radius: float = 0.0
  wheel_action_scale: float = 10.0
  """rad/s per unit action (default offset 0: zero action brakes)."""

  # Front depth camera on `base_body` (tinynav PIE intrinsics for all robots).
  camera_pos: tuple[float, float, float] = depth_mdp.DEPTH_CAMERA_POS
  camera_quat: tuple[float, float, float, float] = depth_mdp.DEPTH_CAMERA_QUAT
  onnx_contract: str = ""
  """`contract` metadata of the exported student ONNX."""
