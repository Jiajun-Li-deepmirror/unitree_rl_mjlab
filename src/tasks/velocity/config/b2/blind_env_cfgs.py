"""Unitree B2 blind (proprioception-only) velocity configuration.

The task is src/tasks/velocity/blind_env_cfg.py. B2 legs are 1.6x Go2's (0.35 m
thigh and calf), so the swing height is 0.15 m, and the payload range follows
the heavier body.
"""

from dataclasses import replace

from mjlab.entity import EntityCfg
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
  apply_blind_staged,
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


# Gait clock pilots (2026-10-18). With the Go2 trot clock (0.6 s) neither flat run
# moved (0-2% of the commanded speed); the second marched in place, feet down 80%
# of the time at 0.8 touchdowns/s instead of the clock's 1.67, legs not swinging
# fore and aft. B2 legs are 1.6x Go2's, so a dynamically similar period is
# ~0.76 s; robot_lab's B2 turns its gait reward off.
B2_BLIND_GAIT08 = replace(B2_BLIND, gait_period=0.8)
B2_BLIND_NO_CLOCK = replace(B2_BLIND, gait_clock=False)

# Both clock pilots failed too (2026-10-19): error_vel_xy stayed at ~1.17 from
# iteration 100 (As2 0.61 by 300). In the first iterations B2 fell 6-8x as often
# as As2. robot_lab's B2 neither ends episodes on body contact nor penalizes
# termination, halves the hip action scale and tracks with weights 3.0 / 1.5.
B2_BLIND_P1 = replace(
  B2_BLIND, hip_action_scale=0.125, body_contact_termination=False, termination_weight=0.0
)
B2_BLIND_P2 = replace(B2_BLIND, tracking_weights=(3.0, 1.5))
B2_BLIND_P3 = replace(B2_BLIND_P1, tracking_weights=(3.0, 1.5))


# These failed too (2026-10-19), and an open-loop trot moves B2 as well as Go2. But
# on two diagonal legs with zero action B2's stance hips splay 0.62 rad and the
# base sinks to 0.23 m (Go2 0.26 rad): its default pose puts the feet 0.1 rad
# outward. B2W, with the same legs, trains from hips 0 and all thighs 0.8.
def _b2_robot_cfg_b2w_stance() -> EntityCfg:
  cfg = get_b2_robot_cfg()
  return replace(
    cfg,
    init_state=replace(
      cfg.init_state,
      joint_pos={".*_hip_joint": 0.0, ".*_thigh_joint": 0.8, ".*_calf_joint": -1.5},
    ),
  )


B2_BLIND_Q1 = replace(B2_BLIND, robot_cfg=_b2_robot_cfg_b2w_stance)
B2_BLIND_Q2 = replace(B2_BLIND, hip_action_scale=0.5)

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


# Hip load (third round, user 2026-10-22; see BlindHipCfg): default hips -0.04 rad
# outward (was +0.10), torque penalty against 30.0 Nm (the static hip torque of the
# old stance) and the splay penalty.
B2_HIP = BlindHipCfg(hip_default=-0.04, torque_ref=30.0)


def unitree_b2_blind_hip_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The heading tasks (see unitree_b2_blind_heading_env_cfg) with the hip-load
  changes (see apply_blind_hip): `terrain` "map3" is the training task, "stairs"
  and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_b2_blind_heading_env_cfg(play=play, terrain=terrain)
  return apply_blind_hip(cfg, B2_HIP, training=terrain == "map3")


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


def unitree_b2_blind_scratch_env_cfg(play: bool = False, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """unitree_b2_blind_rest3_env_cfg trained from scratch with ramped weights (see
  apply_blind_staged): `terrain` "map3" (generated map3-like flights, not map3) is the
  training task, "stairs" and "flat" the evaluation tasks of the same policy."""
  cfg = unitree_b2_blind_rest3_env_cfg(play=play, terrain=terrain)
  return apply_blind_staged(cfg) if terrain == "map3" else cfg
