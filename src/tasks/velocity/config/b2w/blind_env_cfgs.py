"""Unitree B2W blind (proprioception-only) velocity configuration.

The task is the wheel-legged variant in src/tasks/velocity/blind_env_cfg.py:
legs take position targets, wheels velocity targets.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_b2w_robot_cfg
from src.tasks.velocity.blind_env_cfg import (
  BlindFixCfg,
  BlindHipCfg,
  BlindR4Cfg,
  BlindR5Cfg,
  BlindR7Cfg,
  WheeledBlindRobotCfg,
  apply_blind_fix,
  apply_blind_heading,
  apply_blind_hip,
  apply_blind_map3,
  apply_blind_r4,
  apply_blind_r5,
  apply_blind_r7,
  apply_blind_phased,
  apply_grip,
  apply_leg_gains,
  apply_leg_hold,
  apply_no_hip_load,
  apply_side_brake,
  apply_speed_clock,
  apply_stand_clock,
  apply_wheel_stepping_clock,
  apply_blind_straight,
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


# Stairs (2026-10-20), the recipe of the Go2W stairs runs: stance-leg tilt, map3
# walled flights, straight walking without an operator, the heading-error input,
# the mirror loss and the flat-ground anchor to the v5 flat model (flat_v5_gbrake model_7898).
B2W_BLIND_STAIRS = replace(B2W_BLIND, stance_tilt_weight=-10.0)

# Second fix round (2026-10-22; see BlindFixCfg): brake hold and a lower tracking
# std floor ((0.15, 0, 0.3) went 77% forward).
B2W_FIX = BlindFixCfg(brake_hold_stiffness=40.0, min_std=0.1)


def unitree_b2w_blind_heading_env_cfg(
  play: bool = False, terrain: str = "map3"
) -> ManagerBasedRlEnvCfg:
  """The B2W stairs tasks with the heading-error input: `terrain` "map3" is
  the training task (map3 walled flights with the wheeled terms, see
  apply_blind_map3; apply_blind_straight; apply_blind_heading), "stairs" and
  "flat" the evaluation tasks of the same policy. Start from the flat model
  widened by scripts/blind/widen_obs.py, then (until 2026-10-29) the stairs
  normalizer reset of scripts/blind/make_stairs_init.py, since removed."""
  if terrain == "map3":
    cfg = make_wheeled_blind_env_cfg(B2W_BLIND_STAIRS, play=play, stairs=True)
    cfg = apply_blind_straight(apply_blind_map3(cfg, play, legged_terms=False))
  else:
    cfg = make_wheeled_blind_env_cfg(B2W_BLIND_STAIRS, play=play, stairs=terrain == "stairs")
  return apply_blind_heading(apply_blind_fix(cfg, B2W_FIX, training=terrain == "map3"))


# Fourth round (user 2026-10-22 ~07:30): no wheel-lift waiver, a five times stronger
# splay penalty (the third round cut the driving hip torque only 5-31%), and the
# brake engaging after 1 s at the zero command and releasing only above 3x the
# brake speed (stopped on its highest risers B2W kept its wheels above it).
B2W_HIP2 = BlindHipCfg(hip_default=-0.07, torque_ref=22.3, splay_weight=-100.0, reset_splay_prob=0.3)
B2W_R4 = BlindR4Cfg(brake_after_s=1.0, brake_release_ratio=3.0)


def unitree_b2w_blind_hip2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The fourth round: the heading tasks (see unitree_b2w_blind_heading_env_cfg)
  with B2W_HIP2 and B2W_R4 (see apply_blind_hip, apply_blind_r4): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  training = terrain == "map3"
  cfg = unitree_b2w_blind_heading_env_cfg(play=play, terrain=terrain)
  return apply_blind_r4(apply_blind_hip(cfg, B2W_HIP2, training=training), B2W_R4, training=training)


# Fifth round (user 2026-10-22 ~13:30): standing still on even ground, the rest
# stance (see BlindR5Cfg; static half foot spread / base height 0.30).
B2W_R5 = BlindR5Cfg(hip_rest=-0.12)


def unitree_b2w_blind_rest_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_b2w_blind_hip2_env_cfg with B2W_R5 (see apply_blind_r5): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_b2w_blind_hip2_env_cfg(play=play, terrain=terrain)
  return apply_blind_r5(cfg, B2W_R5, training=terrain == "map3")


# Sixth round (user 2026-10-22 ~17:30, "都做"): the rest stance becomes the default
# stance. A wheel cannot move sideways unless its leg lifts: standing, the fifth-round
# policies aimed their hips at the rest stance (Go2W and B2W 4-5 deg past it) but
# the load pushed the hips back out within ~10 s as the wheels slid apart, and on
# the robot static friction would keep the wheels where the robot stood up. Taken
# from the start (FixStand pose) the stance only has to be held, which the fourth
# round's splay penalty taught; the fifth round's rest terms then target the default.
B2W_HIP6 = replace(B2W_HIP2, hip_default=B2W_R5.hip_rest)


def unitree_b2w_blind_rest2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_b2w_blind_rest_env_cfg with B2W_HIP6 for B2W_HIP2 (see apply_blind_hip):
  `terrain` "map3" is the training task, "stairs" and "flat" the evaluation tasks of
  the same policy."""
  training = terrain == "map3"
  cfg = unitree_b2w_blind_heading_env_cfg(play=play, terrain=terrain)
  cfg = apply_blind_r4(apply_blind_hip(cfg, B2W_HIP6, training=training), B2W_R4, training=training)
  return apply_blind_r5(cfg, B2W_R5, training=training)


# Seventh round (user 2026-10-22 ~20:40, "六项全做"; see BlindR7Cfg): march position hold x5 in
# 10-20 s march commands, more stops on stairs with their standing hip torque cost
# doubled; marching in place (gait clock), sideways heading cost.
B2W_R7 = BlindR7Cfg(wheel_march=True, side_drift_weight=-1.0)


def unitree_b2w_blind_rest3_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_b2w_blind_rest2_env_cfg with B2W_R7 (see apply_blind_r7): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_b2w_blind_rest2_env_cfg(play=play, terrain=terrain)
  return apply_blind_r7(cfg, B2W_R7, training=terrain == "map3")


# Thigh plus calf (b2w.xml: thigh 0.35 + calf to wheel axle 0.35), for apply_speed_clock.
B2W_LEG_LENGTH = 0.7
# Feet (FL, FR, RL, RR) in the base frame at the model's own default stance (m), for the gait clock's
# fastest-foot speed on arcs (apply_speed_clock with "arcclock").
B2W_FOOT_XY = ((0.303, 0.237), (0.303, -0.237), (-0.354, 0.237), (-0.354, -0.237))


def unitree_b2w_blind_scratch_env_cfg(
  play: bool = False, terrain: str = "map3", phase: str = "teacher"
) -> ManagerBasedRlEnvCfg:
  """unitree_b2w_blind_rest3_env_cfg with the clocked stepping
  (apply_wheel_stepping_clock), the speed-dependent gait clock, the grip, the robot's own default
  stance and optionally stiffer legs (apply_speed_clock, apply_grip, apply_no_hip_load, apply_leg_gains): `terrain` "map3" (generated
  map3-like flights, not map3) trains `phase` of apply_blind_phased, "stairs" and "flat"
  are the evaluation tasks of the same policies."""
  training = terrain == "map3"
  cfg = apply_wheel_stepping_clock(unitree_b2w_blind_rest3_env_cfg(play=play, terrain=terrain), training=training)
  cfg = apply_no_hip_load(apply_grip(apply_speed_clock(cfg, B2W_LEG_LENGTH, B2W_FOOT_XY)), get_b2w_robot_cfg())
  cfg = apply_side_brake(cfg)
  cfg = apply_leg_hold(apply_leg_gains(cfg))
  if not training:
    return apply_stand_clock(cfg)
  return apply_blind_phased(cfg, phase)
