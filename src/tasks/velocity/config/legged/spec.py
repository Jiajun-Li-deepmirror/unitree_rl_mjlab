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
  hip_walking_std: float = 0.3
  terrain_clearance_weight: float = -40.0
  terrain_clearance_margin: float = 0.05
  """Swing soles must clear step edges ahead by this much (m)."""
  swing_weight_stages: tuple[tuple[int, float], ...] | None = None
  """(iteration, weight) stages of the swing clearance penalty (default -3
  throughout). At -3 the heavy B2 never lifted a foot: each low landing cost
  more than a step of tracking earned, so it stood still."""
  """Hip tolerance of the posture reward while moving (side steps swing the
  hips out; Go2 default 0.15 capped vy at ~70-80%)."""
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
  step_gait_weight: float = 0.5
  """Weight of the trot reward in step mode; above the legged 0.5 it is also
  scaled by command tracking (mdp.feet_gait_stepping)."""
  wheel_blind_recipe: bool = False
  """go2-blind's wheeled rewards instead of the roll/step modes: no roll-mode
  terms and nothing paying for steps; wheels off the ground cost while the
  command can be rolled (wheel_air_drivable, -1), swing clearance -3 from the
  start (judged on even ground only) and a yaw tracking std that shrinks with
  the command (min 0.2). Its wheeled robots step sideways 6-8 cm high (vy
  83-93%) and roll straight, back included, with all four wheels down."""

  # Tracking (go2-blind's map3 stage): at the 0.35 tracking std a 30% shortfall
  # at 0.3 m/s cost only 6% of the reward, and slow walking was mostly seen on
  # stairs.
  even_track_std: float | None = None
  """Linear tracking std on even ground (go2-blind: 0.25); None = one std."""
  flat_slow_prob: float = 0.0
  """Share of flat commands drawn from the slow range (go2-blind: 0.3)."""
  foothold_overspeed_range: tuple[float, float] | None = (0.05, 0.25)
  """Foothold progress is paid in full up to lo m/s over the commanded speed and
  not at all from hi (v9 ran down map3 at 0.6 m/s for 0.4). None = no gate:
  the gate made v10 descend at 0.25-0.3 m/s for 0.4, and the gate-free v8 had
  the best descents."""

  # Front depth camera on `base_body` (tinynav PIE intrinsics for all robots).
  camera_pos: tuple[float, float, float] = depth_mdp.DEPTH_CAMERA_POS
  camera_quat: tuple[float, float, float, float] = depth_mdp.DEPTH_CAMERA_QUAT
  onnx_contract: str = ""
  """`contract` metadata of the exported student ONNX."""
