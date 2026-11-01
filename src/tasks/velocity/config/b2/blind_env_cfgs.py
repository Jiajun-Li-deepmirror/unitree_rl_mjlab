"""Unitree B2 blind (proprioception-only) velocity configuration.

The task is src/tasks/velocity/blind_env_cfg.py. B2 legs are 1.6x Go2's (0.35 m
thigh and calf), so the swing height is 0.15 m, and the payload range follows
the heavier body.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_b2_robot_cfg
from src.tasks.velocity.blind_env_cfg import (
  BlindHipCfg,
  BlindR4Cfg,
  BlindR5Cfg,
  BlindR6Cfg,
  BlindR7Cfg,
  BlindRobotCfg,
  apply_blind_heading,
  apply_blind_hip,
  apply_blind_map3,
  apply_blind_r4,
  apply_blind_r5,
  apply_blind_r6,
  apply_blind_r7,
  apply_blind_phased,
  apply_grip,
  apply_no_hip_load,
  apply_speed_clock,
  apply_blind_straight,
  blind_collision_cfg,
  make_blind_env_cfg,
)
from src.tasks.velocity.quadruped_env_cfg import make_quadruped_rough_env_cfg

B2_BLIND = BlindRobotCfg(
  base_env_cfg=lambda play: make_quadruped_rough_env_cfg(get_b2_robot_cfg, play),
  robot_cfg=get_b2_robot_cfg,
  collision=blind_collision_cfg(
    foot=r"^[FR][LR]_foot_collision$", calf=r"^[FR][LR]_calf\d_collision$"
  ),
  body_contact_patterns=(r"base\d_collision", r"[FR][LR]_hip_collision"),
  leg_contact_patterns=(r"[FR][LR]_thigh_collision", r"[FR][LR]_calf\d_collision"),
  swing_height=0.15,
  payload_range=(-5.0, 10.0),
  nconmax=64,
  # Yaw inertia 6.38 kg m^2: -0.025 * (0.53 / 6.38)^2. At -0.025 a 1.5 rad/s turn
  # cost 2.3 per step against at most 1.0 of tracking, and the first flat run
  # (2026-10-18) learned to stand still.
  angular_momentum_weight=-1.7e-4,
)


def unitree_b2_blind_env_cfg(
  play: bool = False, stairs: bool = True, robot: BlindRobotCfg = B2_BLIND
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree B2 blind velocity configuration (see make_blind_env_cfg)."""
  return make_blind_env_cfg(robot, play=play, stairs=stairs)


# The go2-depth branch's B2 (2026-10-17) also stood still on the Go2 recipe. There,
# with the swing clearance penalty at -1 instead of -3 it started stepping (800-it
# ablations): its first steps barely lift, and each such landing cost more than a
# step of tracking earned. Its fix: -0.5 until iteration 3000, then -1.5, with a
# 12 cm swing target.
B2_BLIND_R1 = replace(
  B2_BLIND, swing_height=0.12, swing_weight_stages=((0, -0.5), (3000, -1.5))
)

# Stairs (2026-10-20): the R1 recipe plus the stance-leg tilt term of the As2 and
# Go2 stairs runs, fine-tuned from the final R1 flat model (flat_v6_r1_cont
# model_4999) on the map3 terrain like Unitree-As2-Blind-Map3.
B2_BLIND_R1_STAIRS = replace(B2_BLIND_R1, stance_tilt_weight=-10.0)


def unitree_b2_blind_map3_env_cfg(
  play: bool = False, robot: BlindRobotCfg = B2_BLIND_R1_STAIRS
) -> ManagerBasedRlEnvCfg:
  """The B2 stairs task with the Go2 map3 fine-tune changes (see apply_blind_map3)."""
  return apply_blind_map3(unitree_b2_blind_env_cfg(play=play, stairs=True, robot=robot), play)


def unitree_b2_blind_straight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The B2 map3 stairs task that walks straight without an operator (see
  apply_blind_straight): the As2 and Go2 stairs models drifted into map3's walls
  without one (2026-10-20)."""
  return apply_blind_straight(unitree_b2_blind_map3_env_cfg(play=play))


def unitree_b2_blind_heading_env_cfg(
  play: bool = False, terrain: str = "map3"
) -> ManagerBasedRlEnvCfg:
  """The B2 R1 stairs tasks with the heading-error input (see apply_blind_heading),
  fine-tuned from Unitree-B2-Blind-R1-Map3-Straight widened by
  scripts/blind/widen_obs.py: `terrain` "map3" is the training task, "stairs"
  and "flat" the evaluation tasks of the same policy. Its model_6000 drifted
  +-13 deg (p10/p90) per 10 s at 0.5 m/s on flat ground (2026-10-20)."""
  if terrain == "map3":
    cfg = unitree_b2_blind_straight_env_cfg(play=play)
  else:
    cfg = unitree_b2_blind_env_cfg(play=play, stairs=terrain == "stairs", robot=B2_BLIND_R1_STAIRS)
  return apply_blind_heading(cfg)


# Fourth round (user 2026-10-22 ~07:30, "减半"): half the third round's stance change
# (+0.03 rad instead of -0.04; +0.10 before): at -0.04 map3's flights without an
# operator fell from 98% to 85% (robots drifting into the rails) while the hip
# torque fell 41-63%. B2 has no descent cap.
B2_HIP2 = BlindHipCfg(hip_default=0.03, torque_ref=30.0)
B2_R4 = BlindR4Cfg()


def unitree_b2_blind_hip2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The fourth round: the heading tasks (see unitree_b2_blind_heading_env_cfg)
  with B2_HIP2 and B2_R4 (see apply_blind_hip, apply_blind_r4): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  training = terrain == "map3"
  cfg = unitree_b2_blind_heading_env_cfg(play=play, terrain=terrain)
  return apply_blind_r4(apply_blind_hip(cfg, B2_HIP2, training=training), B2_R4, training=training)


# Fifth round (user 2026-10-22 ~13:30): standing still on even ground, the rest
# stance (see BlindR5Cfg; static half foot spread / base height 0.30), and half
# the zero commands off the stairs march in place.
B2_R5 = BlindR5Cfg(hip_rest=-0.08, march_prob=0.5)


def unitree_b2_blind_rest_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_b2_blind_hip2_env_cfg with B2_R5 (see apply_blind_r5): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_b2_blind_hip2_env_cfg(play=play, terrain=terrain)
  return apply_blind_r5(cfg, B2_R5, training=terrain == "map3")


# Sixth round (user 2026-10-22 ~17:30, "都做"): a position hold while marching in place
# (the fifth round drifted 49-56 cm per 20 s); B2 has no descent cap.
B2_R6 = BlindR6Cfg(march_hold_weight=-10.0)


def unitree_b2_blind_rest2_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_b2_blind_rest_env_cfg with B2_R6 (see apply_blind_r6): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_b2_blind_rest_env_cfg(play=play, terrain=terrain)
  return apply_blind_r6(cfg, B2_R6, training=terrain == "map3")


# Seventh round (user 2026-10-22 ~20:40, "六项全做"; see BlindR7Cfg): march position hold x5 in
# 10-20 s march commands, more stops on stairs with their standing hip torque cost
# doubled; no descent cap.
B2_R7 = BlindR7Cfg()


def unitree_b2_blind_rest3_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_b2_blind_rest2_env_cfg with B2_R7 (see apply_blind_r7): `terrain` "map3"
  is the training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_b2_blind_rest2_env_cfg(play=play, terrain=terrain)
  return apply_blind_r7(cfg, B2_R7, training=terrain == "map3")


# B2-specific in the phased recipe (user 2026-10-23 ~02:40: allowed, but labelled): the
# foot swing cost ends at -1.5, B2's own value since R1, not the legged -3 (B2 stood
# still at -3 from the start; with -0.5 -> -1.5 it walks).
B2_PHASED_SWING_WEIGHT = -1.5


# Thigh plus calf (b2.xml: thigh 0.35 + calf 0.35), for apply_speed_clock.
B2_LEG_LENGTH = 0.7
# Feet (FL, FR, RL, RR) in the base frame at the model's own default stance (m), for the gait clock's
# fastest-foot speed on arcs (apply_speed_clock with "arcclock").
B2_FOOT_XY = ((0.303, 0.242), (0.303, -0.242), (-0.455, 0.241), (-0.455, -0.241))
# The same with every hip at 0 ("hipzero"; see apply_no_hip_load).
B2_FOOT_XY_HIP0 = ((0.303, 0.192), (0.303, -0.192), (-0.455, 0.192), (-0.455, -0.192))


def unitree_b2_blind_scratch_env_cfg(
  play: bool = False, terrain: str = "map3", phase: str = "teacher"
) -> ManagerBasedRlEnvCfg:
  """unitree_b2_blind_rest3_env_cfg, the speed-dependent gait clock, the grip and the robot's own default
  stance (apply_speed_clock, apply_grip, apply_no_hip_load): `terrain` "map3" (generated
  map3-like flights, not map3) trains `phase` of apply_blind_phased, "stairs" and "flat"
  are the evaluation tasks of the same policies."""
  cfg = apply_no_hip_load(apply_grip(apply_speed_clock(unitree_b2_blind_rest3_env_cfg(play=play, terrain=terrain), B2_LEG_LENGTH, B2_FOOT_XY, B2_FOOT_XY_HIP0)), get_b2_robot_cfg())
  if terrain != "map3":
    return cfg
  return apply_blind_phased(cfg, phase, swing_weight=B2_PHASED_SWING_WEIGHT)
