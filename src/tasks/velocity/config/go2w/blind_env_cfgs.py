"""Unitree Go2W blind (proprioception-only) velocity configuration.

The task is the wheel-legged variant in src/tasks/velocity/blind_env_cfg.py:
legs take position targets, wheels velocity targets.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_go2w_robot_cfg
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
  apply_leg_hold,
  apply_no_hip_load,
  apply_side_brake,
  apply_speed_clock,
  apply_wheel_stepping_clock,
  apply_blind_straight,
  blind_collision_cfg,
  make_wheeled_blind_env_cfg,
)
from src.tasks.velocity.quadruped_env_cfg import make_quadruped_rough_env_cfg

# Wheel kd is 0.5 Nm s/rad, so a 15 Nm wheel torque needs a 30 rad/s velocity
# error: the action scale leaves room for that.
GO2W_BLIND = WheeledBlindRobotCfg(
  base_env_cfg=lambda play: make_quadruped_rough_env_cfg(get_go2w_robot_cfg, play),
  robot_cfg=get_go2w_robot_cfg,
  collision=blind_collision_cfg(
    foot=r"^[FR][LR]_foot_collision$", calf=r"^[FR][LR]_calf\d_collision$"
  ),
  body_contact_patterns=(r"base\d_collision", r"[FR][LR]_hip_collision"),
  leg_contact_patterns=(r"[FR][LR]_thigh_collision", r"[FR][LR]_calf\d_collision"),
  payload_range=(-1.0, 2.0),
  wheel_radius=0.086,
  wheel_action_scale=10.0,
)


# Stairs (2026-10-20): the stairs task plus the stance-leg tilt term of the legged
# stairs runs, fine-tuned from the v5 flat model (flat_v5_gbrake model_6997).
# Lifted wheels while driving straight on even ground cost 4x: blind, the Go2W and
# As2W stairs fine-tunes lifted wheels ahead of risers they could not see, and at
# the default weight -1 went on to step on flat ground at 0.5-1.5 m/s (all four
# wheels down 1-3% of the time 700 iterations into the first map3 run; 2026-10-21),
# which their flat models and B2W drove on four wheels. Near risers (uneven ground
# within 0.45 m of the base) lifting stays free.
GO2W_BLIND_STAIRS = replace(GO2W_BLIND, stance_tilt_weight=-10.0, drivable_air_weight=-4.0)

# Second fix round (2026-10-22; see BlindFixCfg): brake hold on stairs, a lower
# tracking std floor for small commands (Go2W turned 56% of (1.0, 0, 0.1)), and
# descents at 0.80x the command charged twice as much.
GO2W_FIX = BlindFixCfg(brake_hold_stiffness=20.0, min_std=0.1, stair_deficit_weight=-2.0)


def unitree_go2w_blind_map3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The Go2W stairs task with the map3 walled flights (terrain, corridor commands
  and spawns only; see apply_blind_map3)."""
  cfg = make_wheeled_blind_env_cfg(GO2W_BLIND_STAIRS, play=play, stairs=True)
  return apply_blind_map3(cfg, play, legged_terms=False)


def unitree_go2w_blind_straight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The Go2W map3 stairs task that walks straight without an operator (see
  apply_blind_straight); trained with the flat-ground anchor to the v5 flat
  model (rl.anchor), as the first map3 fine-tune lost the flat skills."""
  return apply_blind_straight(unitree_go2w_blind_map3_env_cfg(play=play))


def unitree_go2w_blind_heading_env_cfg(
  play: bool = False, terrain: str = "map3"
) -> ManagerBasedRlEnvCfg:
  """The Go2W stairs tasks with the heading-error input (see apply_blind_heading),
  fine-tuned from Unitree-Go2W-Blind-Map3-Straight widened by
  scripts/blind/widen_obs.py: `terrain` "map3" is the training task, "stairs"
  and "flat" the evaluation tasks of the same policy."""
  if terrain == "map3":
    cfg = unitree_go2w_blind_straight_env_cfg(play=play)
  else:
    cfg = make_wheeled_blind_env_cfg(GO2W_BLIND_STAIRS, play=play, stairs=terrain == "stairs")
  return apply_blind_heading(apply_blind_fix(cfg, GO2W_FIX, training=terrain == "map3"))


def unitree_go2w_blind_v10_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The heading tasks as commit 9628ce6 built them, before the second fix round
  (no apply_blind_fix): the recipe of go2w_blind stairs_v10_air model_9497, the
  model on the real Go2W (resumed from stairs_v9_fix model_8100 on Map3-Heading,
  4096 envs; go2-home, 2026-10-29). Its params/env.yaml equals this config except
  dump artefacts, and the code it reaches equals 9628ce6's with the later options
  at their defaults. `terrain` as for unitree_go2w_blind_heading_env_cfg."""
  if terrain == "map3":
    cfg = unitree_go2w_blind_straight_env_cfg(play=play)
  else:
    cfg = make_wheeled_blind_env_cfg(GO2W_BLIND_STAIRS, play=play, stairs=terrain == "stairs")
  return apply_blind_heading(cfg)


# Fourth round (user 2026-10-22 ~07:30): no wheel-lift waiver (As2W drove with its
# wheels lifted through it), a five times stronger splay penalty (driving, the hip PD
# yields 13-15 deg under the load although the policy aims inward, so the third
# round cut the driving hip torque only 6-17%), and the brake engaging after 1 s at
# the zero command and releasing only above 3x the brake speed.
GO2W_HIP2 = BlindHipCfg(hip_default=-0.17, torque_ref=3.7, splay_weight=-100.0, reset_splay_prob=0.3)
GO2W_R4 = BlindR4Cfg(brake_after_s=1.0, brake_release_ratio=3.0)


def unitree_go2w_blind_hip2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The fourth round: the heading tasks (see unitree_go2w_blind_heading_env_cfg)
  with GO2W_HIP2 and GO2W_R4 (see apply_blind_hip, apply_blind_r4): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  training = terrain == "map3"
  cfg = unitree_go2w_blind_heading_env_cfg(play=play, terrain=terrain)
  return apply_blind_r4(apply_blind_hip(cfg, GO2W_HIP2, training=training), GO2W_R4, training=training)


# Fourth round, Go2W follow-up (user 2026-10-22 ~11:00): driving backward at 1 m/s the
# fourth-round policy tucked its front hips in until the body weight folded them
# further and it rolled over (see mdp.hip_splay); the splay penalty now also costs
# tucking more than 0.1 rad inside the default stance.
GO2W_HIP3 = replace(GO2W_HIP2, splay_inward_deadband=0.1)


def unitree_go2w_blind_hip3_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_go2w_blind_hip2_env_cfg with GO2W_HIP3: `terrain` "map3" is the
  training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  training = terrain == "map3"
  cfg = unitree_go2w_blind_heading_env_cfg(play=play, terrain=terrain)
  return apply_blind_r4(apply_blind_hip(cfg, GO2W_HIP3, training=training), GO2W_R4, training=training)


# Fifth round (user 2026-10-22 ~13:30): standing still on even ground, the rest
# stance (see BlindR5Cfg; static half foot spread / base height 0.30).
GO2W_R5 = BlindR5Cfg(hip_rest=-0.22)


def unitree_go2w_blind_rest_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_go2w_blind_hip3_env_cfg with GO2W_R5 (see apply_blind_r5): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_go2w_blind_hip3_env_cfg(play=play, terrain=terrain)
  return apply_blind_r5(cfg, GO2W_R5, training=terrain == "map3")


# Sixth round (user 2026-10-22 ~17:30, "都做"): the rest stance becomes the default
# stance. A wheel cannot move sideways unless its leg lifts: standing, the fifth-round
# policies aimed their hips at the rest stance (Go2W and B2W 4-5 deg past it) but
# the load pushed the hips back out within ~10 s as the wheels slid apart, and on
# the robot static friction would keep the wheels where the robot stood up. Taken
# from the start (FixStand pose) the stance only has to be held, which the fourth
# round's splay penalty taught; the fifth round's rest terms then target the default.
GO2W_HIP6 = replace(GO2W_HIP3, hip_default=GO2W_R5.hip_rest)


def unitree_go2w_blind_rest2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_go2w_blind_rest_env_cfg with GO2W_HIP6 for GO2W_HIP3 (see apply_blind_hip):
  `terrain` "map3" is the training task, "stairs" and "flat" the evaluation tasks of
  the same policy."""
  training = terrain == "map3"
  cfg = unitree_go2w_blind_heading_env_cfg(play=play, terrain=terrain)
  cfg = apply_blind_r4(apply_blind_hip(cfg, GO2W_HIP6, training=training), GO2W_R4, training=training)
  return apply_blind_r5(cfg, GO2W_R5, training=training)


# Seventh round (user 2026-10-22 ~20:40, "六项全做"; see BlindR7Cfg): march position hold x5 in
# 10-20 s march commands, more stops on stairs with their standing hip torque cost
# doubled; marching in place (gait clock), sideways heading cost.
GO2W_R7 = BlindR7Cfg(wheel_march=True, side_drift_weight=-1.0)


def unitree_go2w_blind_rest3_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_go2w_blind_rest2_env_cfg with GO2W_R7 (see apply_blind_r7): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_go2w_blind_rest2_env_cfg(play=play, terrain=terrain)
  return apply_blind_r7(cfg, GO2W_R7, training=terrain == "map3")


# Thigh plus calf (go2w.xml: thigh 0.213 + calf to wheel axle 0.226), for apply_speed_clock.
GO2W_LEG_LENGTH = 0.439
# Feet (FL, FR, RL, RR) in the base frame at the model's own default stance (m), for the gait clock's
# fastest-foot speed on arcs (apply_speed_clock with "arcclock").
GO2W_FOOT_XY = ((0.186, 0.193), (0.186, -0.193), (-0.2, 0.193), (-0.2, -0.193))


def unitree_go2w_blind_scratch_env_cfg(
  play: bool = False, terrain: str = "map3", phase: str = "teacher"
) -> ManagerBasedRlEnvCfg:
  """unitree_go2w_blind_rest3_env_cfg with the clocked stepping
  (apply_wheel_stepping_clock), the speed-dependent gait clock, the grip and the robot's own default
  stance (apply_speed_clock, apply_grip, apply_no_hip_load): `terrain` "map3" (generated
  map3-like flights, not map3) trains `phase` of apply_blind_phased, "stairs" and "flat"
  are the evaluation tasks of the same policies."""
  training = terrain == "map3"
  cfg = apply_wheel_stepping_clock(unitree_go2w_blind_rest3_env_cfg(play=play, terrain=terrain), training=training)
  cfg = apply_no_hip_load(apply_grip(apply_speed_clock(cfg, GO2W_LEG_LENGTH, GO2W_FOOT_XY)), get_go2w_robot_cfg())
  cfg = apply_side_brake(cfg)
  # "leghold20" in BLIND_PILOT as the B2W's / As2W's (go2-depth 2026-11-07: the Go2W stop creep after reversing
  # sat on its 1 cm bar in every polish, 1.2-1.6 cm in g18 / g19); off without it.
  cfg = apply_leg_hold(cfg)
  if not training:
    return cfg
  return apply_blind_phased(cfg, phase)
