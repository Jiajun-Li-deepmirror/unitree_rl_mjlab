"""Unitree As2 blind (proprioception-only) velocity configuration.

The task is src/tasks/velocity/blind_env_cfg.py. As2 legs are close to Go2's
(0.21 m thigh and calf, ~21 kg), so the Go2 swing height and payload range are
kept.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_as2_robot_cfg
from src.tasks.velocity.blind_env_cfg import (
  BlindFixCfg,
  BlindHipCfg,
  BlindR4Cfg,
  BlindR5Cfg,
  BlindR6Cfg,
  BlindR7Cfg,
  BlindRobotCfg,
  apply_blind_fix,
  apply_blind_heading,
  apply_blind_hip,
  apply_blind_map3,
  apply_blind_r4,
  apply_blind_r5,
  apply_blind_r6,
  apply_blind_r7,
  apply_blind_staged,
  apply_blind_straight,
  blind_collision_cfg,
  make_blind_env_cfg,
)
from src.tasks.velocity.config.as2.env_cfgs import unitree_as2_rough_env_cfg

AS2_BLIND = BlindRobotCfg(
  base_env_cfg=unitree_as2_rough_env_cfg,
  robot_cfg=get_as2_robot_cfg,
  collision=blind_collision_cfg(
    foot=r"^[FR][LR]_foot_collision$", calf=r"^[FR][LR]_calf\d_collision$"
  ),
  # thigh1 is the motor cylinder at the thigh root, next to the hip.
  body_contact_patterns=(
    r"base\d_collision", r"[FR][LR]_hip_collision", r"[FR][LR]_thigh1_collision"
  ),
  leg_contact_patterns=(r"[FR][LR]_thigh2_collision", r"[FR][LR]_calf\d_collision"),
  # At 1000 stairs iterations, 0.07 rad^2 per step climbing (front hips 12 deg
  # open): -0.7 per second, against at most 1 per second for linear tracking.
  stance_tilt_weight=-10.0,
)

# Second fix round (2026-10-22; see BlindFixCfg): more sideways and diagonal commands
# (79-80% sideways), descent pay capped at 1.1x the command speed (went down at 1.24x).
AS2_FIX = BlindFixCfg(vy_mode_boost=0.05, diag_prob=0.1, descent_max_speed_ratio=1.1)


def unitree_as2_blind_env_cfg(play: bool = False, stairs: bool = True) -> ManagerBasedRlEnvCfg:
  """Create the Unitree As2 blind velocity configuration (see make_blind_env_cfg)."""
  return make_blind_env_cfg(AS2_BLIND, play=play, stairs=stairs)


def unitree_as2_blind_map3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The As2 stairs task with the Go2 map3 fine-tune changes (see apply_blind_map3).

  As2 is Go2-sized, so the Go2 walled flights fit. At 1700 stairs iterations
  (pyramid stairs only) it reached map3 goals in time 0-61% (Go2 release
  5-97%), with 16-28% resets on the L0 flight and its 0.188 m tread.
  """
  return apply_blind_map3(unitree_as2_blind_env_cfg(play=play, stairs=True), play)


def unitree_as2_blind_straight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The As2 map3 task that walks straight without an operator (see
  apply_blind_straight). The stairs model_7500 drifted into the walls of map3's
  flights on 41-69% of the runs without one."""
  return apply_blind_straight(unitree_as2_blind_map3_env_cfg(play=play))


def unitree_as2_blind_heading_env_cfg(
  play: bool = False, terrain: str = "map3"
) -> ManagerBasedRlEnvCfg:
  """The As2 tasks with the heading-error input (see apply_blind_heading):
  `terrain` "map3" is the straight-walking training task, "stairs" and "flat"
  the evaluation tasks of the same policy."""
  if terrain == "map3":
    cfg = unitree_as2_blind_straight_env_cfg(play=play)
  else:
    cfg = unitree_as2_blind_env_cfg(play=play, stairs=terrain == "stairs")
  return apply_blind_heading(apply_blind_fix(cfg, AS2_FIX, training=terrain == "map3"))


# Hip load (third round, user 2026-10-22; see BlindHipCfg): default hips -0.23 rad
# outward (was -0.10), torque penalty against 2.97 Nm (the static hip torque of the
# old stance) and the splay penalty.
AS2_HIP = BlindHipCfg(hip_default=-0.23, torque_ref=2.97)


def unitree_as2_blind_hip_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The heading tasks (see unitree_as2_blind_heading_env_cfg) with the hip-load
  changes (see apply_blind_hip): `terrain` "map3" is the training task, "stairs"
  and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_as2_blind_heading_env_cfg(play=play, terrain=terrain)
  return apply_blind_hip(cfg, AS2_HIP, training=terrain == "map3")


# Fourth round (user 2026-10-22 ~07:30, "减半"): half the third round's stance change
# (-0.165 rad instead of -0.23; -0.10 before): at -0.23 map3's flights without an
# operator fell from 85% to 75-76% while the hip torque fell 50-69%. The descent
# cap judged on each foot's mean speed per step (the instant-speed cap slowed
# descents to 0.66x).
AS2_HIP2 = BlindHipCfg(hip_default=-0.165, torque_ref=2.97)
AS2_R4 = BlindR4Cfg(descent_step_average=True)


def unitree_as2_blind_hip2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The fourth round: the heading tasks (see unitree_as2_blind_heading_env_cfg)
  with AS2_HIP2 and AS2_R4 (see apply_blind_hip, apply_blind_r4): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  training = terrain == "map3"
  cfg = unitree_as2_blind_heading_env_cfg(play=play, terrain=terrain)
  return apply_blind_r4(apply_blind_hip(cfg, AS2_HIP2, training=training), AS2_R4, training=training)


# Fifth round (user 2026-10-22 ~13:30): standing still on even ground, the rest
# stance (see BlindR5Cfg; static half foot spread / base height 0.30), and half
# the zero commands off the stairs march in place.
AS2_R5 = BlindR5Cfg(hip_rest=-0.27, march_prob=0.5)


def unitree_as2_blind_rest_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_as2_blind_hip2_env_cfg with AS2_R5 (see apply_blind_r5): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_as2_blind_hip2_env_cfg(play=play, terrain=terrain)
  return apply_blind_r5(cfg, AS2_R5, training=terrain == "map3")


# Sixth round (user 2026-10-22 ~17:30, "都做"): a position hold while marching in place
# (the fifth round drifted 46-54 cm per 20 s) and the descent cap at 1.4 (at 1.1 the
# fifth round descended at 0.69-0.74x the command).
AS2_R6 = BlindR6Cfg(descent_max_speed_ratio=1.4, march_hold_weight=-10.0)


def unitree_as2_blind_rest2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_as2_blind_rest_env_cfg with AS2_R6 (see apply_blind_r6): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_as2_blind_rest_env_cfg(play=play, terrain=terrain)
  return apply_blind_r6(cfg, AS2_R6, training=terrain == "map3")


# Seventh round (user 2026-10-22 ~20:40, "六项全做"; see BlindR7Cfg): march position hold x5 in
# 10-20 s march commands, more stops on stairs with their standing hip torque cost
# doubled; descent cap 1.6 (at 1.4 the sixth round descended at 0.79-0.84x).
AS2_R7 = BlindR7Cfg(descent_max_speed_ratio=1.6)


def unitree_as2_blind_rest3_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_as2_blind_rest2_env_cfg with AS2_R7 (see apply_blind_r7): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_as2_blind_rest2_env_cfg(play=play, terrain=terrain)
  return apply_blind_r7(cfg, AS2_R7, training=terrain == "map3")


def unitree_as2_blind_scratch_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_as2_blind_rest3_env_cfg trained from scratch with ramped weights (see
  apply_blind_staged): `terrain` "map3" (generated map3-like flights, not map3) is the
  training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_as2_blind_rest3_env_cfg(play=play, terrain=terrain)
  return apply_blind_staged(cfg) if terrain == "map3" else cfg
