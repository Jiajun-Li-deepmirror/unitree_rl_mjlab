"""Unitree Go2 blind (proprioception-only) velocity configurations.

The task itself is src/tasks/velocity/blind_env_cfg.py; this module gives it the
Go2 model and adds the Go2 map3 fine-tune.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_go2_robot_cfg
from src.tasks.velocity.blind_env_cfg import (
  BlindHipCfg,
  BlindR4Cfg,
  BlindR5Cfg,
  BlindR6Cfg,
  BlindR7Cfg,
  FINAL_VEL_RANGES as FINAL_VEL_RANGES,
  apply_blind_hip,
  apply_blind_r4,
  apply_blind_r5,
  apply_blind_r6,
  apply_blind_r7,
  apply_blind_staged,
)
from src.tasks.velocity.blind_env_cfg import (
  MAP3_STAIR_TERRAIN_NAMES as MAP3_STAIR_TERRAIN_NAMES,
)
from src.tasks.velocity.blind_env_cfg import (
  WALLED_TERRAIN_NAMES as WALLED_TERRAIN_NAMES,
)
from src.tasks.velocity.blind_env_cfg import (
  BlindFixCfg,
  BlindRobotCfg,
  apply_blind_fix,
  apply_blind_heading,
  apply_blind_map3,
  apply_blind_straight,
  blind_collision_cfg,
  make_blind_env_cfg,
)
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_env_cfg

GO2_BLIND = BlindRobotCfg(
  base_env_cfg=unitree_go2_rough_env_cfg,
  robot_cfg=get_go2_robot_cfg,
  collision=blind_collision_cfg(
    foot=r"^[FR][LR]_foot_collision$", calf=r"^[FR][LR]_calf\d_collision$"
  ),
  body_contact_patterns=(r"base\d_collision", r"[FR][LR]_hip_collision"),
  leg_contact_patterns=(r"[FR][LR]_thigh_collision", r"[FR][LR]_calf\d_collision"),
)

# Second fix round (2026-10-22; see BlindFixCfg): rises up to 20 cm (map3's flights
# that start with one stopped the Go2 72-84% of the time with the operator), more
# diagonal commands (74-80% sideways), descent pay capped at 1.1x the command speed.
GO2_FIX = BlindFixCfg(max_step_height=0.20, diag_prob=0.1, descent_max_speed_ratio=1.1)


def unitree_go2_blind_env_cfg(
  play: bool = False,
  stairs: bool = True,
  terrain_clearance: bool = True,
  foothold_gain: bool = True,
  stance_tilt_weight: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go2 blind velocity configuration (see make_blind_env_cfg)."""
  return make_blind_env_cfg(
    replace(GO2_BLIND, stance_tilt_weight=stance_tilt_weight),
    play=play,
    stairs=stairs,
    terrain_clearance=terrain_clearance,
    foothold_gain=foothold_gain,
  )


def unitree_go2_blind_map3_env_cfg(
  play: bool = False, stance_tilt_weight: float | None = None
) -> ManagerBasedRlEnvCfg:
  """Unitree-Go2-Blind fine-tuned for map3 (see apply_blind_map3): same
  observations and actions. With `stance_tilt_weight`, stance legs on stairs
  keep the default stance's sideways tilt (stance_leg_tilt; the release opened
  its front hips 7 deg)."""
  cfg = unitree_go2_blind_env_cfg(play=play, stairs=True, stance_tilt_weight=stance_tilt_weight)
  return apply_blind_map3(cfg, play)


def unitree_go2_blind_heading_env_cfg(
  play: bool = False, terrain: str = "map3"
) -> ManagerBasedRlEnvCfg:
  """The Go2 stance-tilt tasks with straight walking without an operator (see
  apply_blind_straight) and the heading-error input (see apply_blind_heading),
  fine-tuned from Unitree-Go2-Blind-Map3-Tilt widened by scripts/blind/widen_obs.py:
  `terrain` "map3" is the training task, "stairs" and "flat" the evaluation
  tasks of the same policy. The As2 run with both (2026-10-20) walked map3's
  flights without an operator 79% of the time (51% before) and drifted +-1.5 deg
  per 10 s at 0.5 m/s on flat ground (p50/p90 +11/+34 deg before)."""
  if terrain == "map3":
    cfg = apply_blind_straight(unitree_go2_blind_map3_env_cfg(play=play, stance_tilt_weight=-10.0))
  else:
    cfg = unitree_go2_blind_env_cfg(play=play, stairs=terrain == "stairs", stance_tilt_weight=-10.0)
  return apply_blind_heading(apply_blind_fix(cfg, GO2_FIX, training=terrain == "map3"))


# Hip load (third round, user 2026-10-22; see BlindHipCfg): default hips -0.16 rad
# outward (was -0.10), torque penalty against 1.8 Nm (the static hip torque of the
# old stance) and the splay penalty.
GO2_HIP = BlindHipCfg(hip_default=-0.16, torque_ref=1.8)


def unitree_go2_blind_hip_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The heading tasks (see unitree_go2_blind_heading_env_cfg) with the hip-load
  changes (see apply_blind_hip): `terrain` "map3" is the training task, "stairs"
  and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_go2_blind_heading_env_cfg(play=play, terrain=terrain)
  return apply_blind_hip(cfg, GO2_HIP, training=terrain == "map3")


# Fourth round (user 2026-10-22 ~07:30): the third-round stance and penalties, and
# the descent cap judged on each foot's mean speed per step (the instant-speed cap
# slowed descents to 0.63x).
GO2_HIP2 = GO2_HIP
GO2_R4 = BlindR4Cfg(descent_step_average=True)


def unitree_go2_blind_hip2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The fourth round: the heading tasks (see unitree_go2_blind_heading_env_cfg)
  with GO2_HIP2 and GO2_R4 (see apply_blind_hip, apply_blind_r4): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  training = terrain == "map3"
  cfg = unitree_go2_blind_heading_env_cfg(play=play, terrain=terrain)
  return apply_blind_r4(apply_blind_hip(cfg, GO2_HIP2, training=training), GO2_R4, training=training)


# Fifth round (user 2026-10-22 ~13:30): standing still on even ground, the rest
# stance (see BlindR5Cfg; static half foot spread / base height 0.30), and half
# the zero commands off the stairs march in place.
GO2_R5 = BlindR5Cfg(hip_rest=-0.2, march_prob=0.5)


def unitree_go2_blind_rest_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_go2_blind_hip2_env_cfg with GO2_R5 (see apply_blind_r5): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_go2_blind_hip2_env_cfg(play=play, terrain=terrain)
  return apply_blind_r5(cfg, GO2_R5, training=terrain == "map3")


# Sixth round (user 2026-10-22 ~17:30, "都做"): a position hold while marching in place
# (the fifth round drifted 41-51 cm per 20 s) and the descent cap at 1.4 (at 1.1 the
# fifth round descended at 0.72x the command).
GO2_R6 = BlindR6Cfg(descent_max_speed_ratio=1.4, march_hold_weight=-10.0)


def unitree_go2_blind_rest2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_go2_blind_rest_env_cfg with GO2_R6 (see apply_blind_r6): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_go2_blind_rest_env_cfg(play=play, terrain=terrain)
  return apply_blind_r6(cfg, GO2_R6, training=terrain == "map3")


# Seventh round (user 2026-10-22 ~20:40, "六项全做"; see BlindR7Cfg): march position hold x5 in
# 10-20 s march commands, more stops on stairs with their standing hip torque cost
# doubled; descent cap 1.6 (at 1.4 the sixth round descended at 0.77-0.83x).
GO2_R7 = BlindR7Cfg(descent_max_speed_ratio=1.6)


def unitree_go2_blind_rest3_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_go2_blind_rest2_env_cfg with GO2_R7 (see apply_blind_r7): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_go2_blind_rest2_env_cfg(play=play, terrain=terrain)
  return apply_blind_r7(cfg, GO2_R7, training=terrain == "map3")


def unitree_go2_blind_scratch_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_go2_blind_rest3_env_cfg trained from scratch with ramped weights (see
  apply_blind_staged): `terrain` "map3" (generated map3-like flights, not map3) is the
  training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_go2_blind_rest3_env_cfg(play=play, terrain=terrain)
  return apply_blind_staged(cfg) if terrain == "map3" else cfg
