"""Unitree As2W blind (proprioception-only) velocity configuration.

The task is the wheel-legged variant in src/tasks/velocity/blind_env_cfg.py:
legs take position targets, wheels velocity targets.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_as2w_robot_cfg
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
  apply_blind_staged,
  apply_wheel_stepping_clock,
  apply_blind_straight,
  blind_collision_cfg,
  make_wheeled_blind_env_cfg,
)
from src.tasks.velocity.quadruped_env_cfg import make_quadruped_rough_env_cfg

# Wheel kd 1.0 saturates the 15 Nm wheel torque at a 15 rad/s velocity error.
AS2W_BLIND = WheeledBlindRobotCfg(
  base_env_cfg=lambda play: make_quadruped_rough_env_cfg(get_as2w_robot_cfg, play),
  robot_cfg=get_as2w_robot_cfg,
  collision=blind_collision_cfg(
    foot=r"^[FR][LR]_foot_collision$", calf=r"^[FR][LR]_calf\d_collision$"
  ),
  body_contact_patterns=(r"base\d_collision", r"[FR][LR]_hip_collision"),
  leg_contact_patterns=(r"[FR][LR]_thigh\d_collision", r"[FR][LR]_calf\d_collision"),
  payload_range=(-1.0, 3.0),
  wheel_radius=0.086,
  wheel_action_scale=8.0,
)


def unitree_as2w_blind_env_cfg(play: bool = False, stairs: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree As2W blind velocity configuration (see make_wheeled_blind_env_cfg)."""
  return make_wheeled_blind_env_cfg(AS2W_BLIND, play=play, stairs=stairs)


# Stairs (2026-10-20), the recipe of the Go2W stairs runs: stance-leg tilt, map3
# walled flights, straight walking without an operator, the heading-error input,
# the mirror loss and the flat-ground anchor to the v5 flat model (flat_v5_gbrake model_5300).
# Lifted wheels while driving straight on even ground cost 4x (see GO2W_BLIND_STAIRS).
AS2W_BLIND_STAIRS = replace(AS2W_BLIND, stance_tilt_weight=-10.0, drivable_air_weight=-4.0)

# Second fix round (2026-10-22; see BlindFixCfg): brake hold, a lower tracking std
# floor, descents at 0.76x charged twice as much, and more sideways and diagonal
# commands (all eight sideways acceptance commands under 80%).
AS2W_FIX = BlindFixCfg(
  brake_hold_stiffness=20.0, min_std=0.1, stair_deficit_weight=-2.0, vy_mode_boost=0.1, diag_prob=0.1
)


def unitree_as2w_blind_heading_env_cfg(
  play: bool = False, terrain: str = "map3", fix: BlindFixCfg = AS2W_FIX
) -> ManagerBasedRlEnvCfg:
  """The As2W stairs tasks with the heading-error input: `terrain` "map3" is
  the training task (map3 walled flights with the wheeled terms, see
  apply_blind_map3; apply_blind_straight; apply_blind_heading), "stairs" and
  "flat" the evaluation tasks of the same policy. Start from the flat model
  widened by scripts/blind/widen_obs.py, then scripts/blind/make_stairs_init.py."""
  if terrain == "map3":
    cfg = make_wheeled_blind_env_cfg(AS2W_BLIND_STAIRS, play=play, stairs=True)
    cfg = apply_blind_straight(apply_blind_map3(cfg, play, legged_terms=False))
  else:
    cfg = make_wheeled_blind_env_cfg(AS2W_BLIND_STAIRS, play=play, stairs=terrain == "stairs")
  return apply_blind_heading(apply_blind_fix(cfg, fix, training=terrain == "map3"))


# Hip load (third round, user 2026-10-22; see BlindHipCfg): default hips -0.30 rad
# outward (was -0.10), torque penalty against 5.75 Nm (the static hip torque of the
# old stance) and the splay penalty. Wheeled: lifting a wheel is free while its hip
# is splayed 0.08 rad, and 30% of the resets start splayed 0.04-0.15 rad.
AS2W_HIP = BlindHipCfg(hip_default=-0.3, torque_ref=5.75, wheel_free_splay=0.08, reset_splay_prob=0.3)


def unitree_as2w_blind_hip_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The heading tasks (see unitree_as2w_blind_heading_env_cfg) with the hip-load
  changes (see apply_blind_hip): `terrain` "map3" is the training task, "stairs"
  and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_as2w_blind_heading_env_cfg(play=play, terrain=terrain)
  return apply_blind_hip(cfg, AS2W_HIP, training=terrain == "map3")


# Fourth round (user 2026-10-22 ~07:30): from the deployed stairs_v4_air model_8200,
# as both later rounds drove with lifted wheels (9200: all four down 52% / 15% at
# 1 / 1.5 m/s; third round 0-8% through the wheel-lift waiver): only the brake hold
# of the second round (AS2W_FIX4), the third round's stance without the waiver and
# with a five times stronger splay penalty, and the brake engaging after 1 s at the
# zero command and releasing only above 3x the brake speed.
AS2W_FIX4 = BlindFixCfg(brake_hold_stiffness=20.0)
AS2W_HIP2 = BlindHipCfg(hip_default=-0.3, torque_ref=5.75, splay_weight=-100.0, reset_splay_prob=0.3)
AS2W_R4 = BlindR4Cfg(brake_after_s=1.0, brake_release_ratio=3.0)


def unitree_as2w_blind_hip2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The fourth round: the heading tasks (see unitree_as2w_blind_heading_env_cfg)
  with AS2W_HIP2 and AS2W_R4 (see apply_blind_hip, apply_blind_r4): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  training = terrain == "map3"
  cfg = unitree_as2w_blind_heading_env_cfg(play=play, terrain=terrain, fix=AS2W_FIX4)
  return apply_blind_r4(apply_blind_hip(cfg, AS2W_HIP2, training=training), AS2W_R4, training=training)


# Fifth round (user 2026-10-22 ~13:30): standing still on even ground, the rest
# stance (see BlindR5Cfg; static half foot spread / base height 0.30).
AS2W_R5 = BlindR5Cfg(hip_rest=-0.38)


def unitree_as2w_blind_rest_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_as2w_blind_hip2_env_cfg with AS2W_R5 (see apply_blind_r5): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_as2w_blind_hip2_env_cfg(play=play, terrain=terrain)
  return apply_blind_r5(cfg, AS2W_R5, training=terrain == "map3")


# Sixth round (user 2026-10-22 ~17:30, "都做"): the rest stance becomes the default
# stance. A wheel cannot move sideways unless its leg lifts: standing, the fifth-round
# policies aimed their hips at the rest stance (Go2W and B2W 4-5 deg past it) but
# the load pushed the hips back out within ~10 s as the wheels slid apart, and on
# the robot static friction would keep the wheels where the robot stood up. Taken
# from the start (FixStand pose) the stance only has to be held, which the fourth
# round's splay penalty taught; the fifth round's rest terms then target the default.
AS2W_HIP6 = replace(AS2W_HIP2, hip_default=AS2W_R5.hip_rest)


def unitree_as2w_blind_rest2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_as2w_blind_rest_env_cfg with AS2W_HIP6 for AS2W_HIP2 (see apply_blind_hip):
  `terrain` "map3" is the training task, "stairs" and "flat" the evaluation tasks of
  the same policy."""
  training = terrain == "map3"
  cfg = unitree_as2w_blind_heading_env_cfg(play=play, terrain=terrain, fix=AS2W_FIX4)
  cfg = apply_blind_r4(apply_blind_hip(cfg, AS2W_HIP6, training=training), AS2W_R4, training=training)
  return apply_blind_r5(cfg, AS2W_R5, training=training)


# Seventh round (user 2026-10-22 ~20:40, "六项全做"; see BlindR7Cfg): march position hold x5 in
# 10-20 s march commands, more stops on stairs with their standing hip torque cost
# doubled; marching in place (gait clock), sideways heading cost.
AS2W_R7 = BlindR7Cfg(wheel_march=True, side_drift_weight=-1.0)
# As2W only: the rest stance as the default made it tuck its legs in backing up and
# roll over (10700: 14 of 32 fell at -1 m/s, all four wheels down 58.5%), as the
# Go2W did in the fourth round; the inward splay cost fixed that there.
AS2W_HIP7 = replace(AS2W_HIP6, splay_inward_deadband=0.1)


def unitree_as2w_blind_rest3_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_as2w_blind_rest2_env_cfg with AS2W_HIP7 for AS2W_HIP6 and AS2W_R7 (see
  apply_blind_r7): `terrain` "map3" is the training task, "stairs" and "flat" the
  evaluation tasks of the same policy."""
  training = terrain == "map3"
  cfg = unitree_as2w_blind_heading_env_cfg(play=play, terrain=terrain, fix=AS2W_FIX4)
  cfg = apply_blind_r4(apply_blind_hip(cfg, AS2W_HIP7, training=training), AS2W_R4, training=training)
  return apply_blind_r7(apply_blind_r5(cfg, AS2W_R5, training=training), AS2W_R7, training=training)


def unitree_as2w_blind_scratch_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_as2w_blind_rest3_env_cfg with the clocked stepping (apply_wheel_stepping_clock)
  trained from scratch with ramped weights (see apply_blind_staged): `terrain` "map3" (generated map3-like flights, not map3) is the
  training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  training = terrain == "map3"
  cfg = apply_wheel_stepping_clock(unitree_as2w_blind_rest3_env_cfg(play=play, terrain=terrain), training=training)
  return apply_blind_staged(cfg) if training else cfg
