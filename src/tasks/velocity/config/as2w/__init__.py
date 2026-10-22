from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_rl_cfg import make_blind_ppo_runner_cfg
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner
from src.tasks.velocity.rl.anchor import AnchoredMirrorVelocityOnPolicyRunner

from .blind_env_cfgs import (
  unitree_as2w_blind_env_cfg,
  unitree_as2w_blind_heading_env_cfg,
  unitree_as2w_blind_hip_env_cfg,
  unitree_as2w_blind_hip2_env_cfg,
  unitree_as2w_blind_rest2_env_cfg,
  unitree_as2w_blind_rest3_env_cfg,
  unitree_as2w_blind_scratch_env_cfg,
  unitree_as2w_blind_rest_env_cfg,
)


class As2WFlatAnchoredHeadingRunner(AnchoredMirrorVelocityOnPolicyRunner):
  """Anchored on flat ground to the v5 flat model widened for the heading input,
  sideways commands included, straight forward / backward ones at 0.3."""

  ANCHOR_CHECKPOINT = "logs/rsl_rl/as2w_blind/2026-10-20_02-01-07_flat_v5_gbrake/model_5300_widened.pt"
  ANCHOR_SKIP_SIDEWAYS = False
  # Straight forward / backward too, lightly (see rl.anchor).
  ANCHOR_STRAIGHT_COEFF = 0.3


# Blind joystick policy (see make_wheeled_blind_env_cfg): flat ground first, then stairs.
register_mjlab_task(
  task_id="Unitree-As2W-Blind-Flat",
  env_cfg=unitree_as2w_blind_env_cfg(stairs=False),
  play_env_cfg=unitree_as2w_blind_env_cfg(play=True, stairs=False),
  rl_cfg=make_blind_ppo_runner_cfg("as2w_blind", max_iterations=3000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-As2W-Blind",
  env_cfg=unitree_as2w_blind_env_cfg(stairs=True),
  play_env_cfg=unitree_as2w_blind_env_cfg(play=True, stairs=True),
  rl_cfg=make_blind_ppo_runner_cfg("as2w_blind", max_iterations=20000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)

# Stairs with the heading-error input: training (map3, anchored) and evaluation
# (stairs, flat) tasks.
for task_id, terrain, runner in (
  ("Unitree-As2W-Blind-Map3-Heading", "map3", As2WFlatAnchoredHeadingRunner),
  ("Unitree-As2W-Blind-Heading", "stairs", MirrorVelocityOnPolicyRunner),
  ("Unitree-As2W-Blind-Flat-Heading", "flat", MirrorVelocityOnPolicyRunner),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2w_blind_heading_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2w_blind_heading_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("as2w_blind", max_iterations=20000),
    runner_cls=runner,
  )

# The heading tasks with the hip-load changes (third round, 2026-10-22; see
# BlindHipCfg; resume from a Heading model): training (map3) and evaluation
# (stairs, flat) tasks.
for task_id, terrain, runner in (
  ("Unitree-As2W-Blind-Map3-Hip", "map3", As2WFlatAnchoredHeadingRunner),
  ("Unitree-As2W-Blind-Hip", "stairs", MirrorVelocityOnPolicyRunner),
  ("Unitree-As2W-Blind-Flat-Hip", "flat", MirrorVelocityOnPolicyRunner),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2w_blind_hip_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2w_blind_hip_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("as2w_blind", max_iterations=20000),
    runner_cls=runner,
  )


# The fourth round (2026-10-22; see BlindR4Cfg and the robot's *_HIP2 / *_R4;
# resume from a third-round Hip model, As2W from its Heading model_8200):
# training (map3) and evaluation (stairs, flat) tasks.
for task_id, terrain, runner in (
  ("Unitree-As2W-Blind-Map3-Hip2", "map3", As2WFlatAnchoredHeadingRunner),
  ("Unitree-As2W-Blind-Hip2", "stairs", MirrorVelocityOnPolicyRunner),
  ("Unitree-As2W-Blind-Flat-Hip2", "flat", MirrorVelocityOnPolicyRunner),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2w_blind_hip2_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2w_blind_hip2_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("as2w_blind", max_iterations=20000),
    runner_cls=runner,
  )


# The fifth round (2026-10-22; the rest stance, see BlindR5Cfg and the robot's *_R5;
# resume from the fourth round's model).
for task_id, terrain, runner in (
  ("Unitree-As2W-Blind-Map3-Rest", "map3", As2WFlatAnchoredHeadingRunner),
  ("Unitree-As2W-Blind-Rest", "stairs", MirrorVelocityOnPolicyRunner),
  ("Unitree-As2W-Blind-Flat-Rest", "flat", MirrorVelocityOnPolicyRunner),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2w_blind_rest_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2w_blind_rest_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("as2w_blind", max_iterations=20000),
    runner_cls=runner,
  )

# The sixth round (2026-10-22; the rest stance as the default stance, see the robot's
# *_HIP6; resume from the fifth round's model, As2W's from the fourth round's).
for task_id, terrain, runner in (
  ("Unitree-As2W-Blind-Map3-Rest2", "map3", As2WFlatAnchoredHeadingRunner),
  ("Unitree-As2W-Blind-Rest2", "stairs", MirrorVelocityOnPolicyRunner),
  ("Unitree-As2W-Blind-Flat-Rest2", "flat", MirrorVelocityOnPolicyRunner),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2w_blind_rest2_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2w_blind_rest2_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("as2w_blind", max_iterations=20000),
    runner_cls=runner,
  )


class As2WFlatAnchoredMarchRunner(As2WFlatAnchoredHeadingRunner):
  """As2WFlatAnchoredHeadingRunner with its anchor widened for the gait clock (scripts/blind/widen_obs.py
  --new-std 0.7); samples marching in place are not anchored (see rl.anchor)."""

  ANCHOR_CHECKPOINT = "logs/rsl_rl/as2w_blind/2026-10-20_02-01-07_flat_v5_gbrake/model_5300_widened_phase.pt"

# The seventh round (2026-10-22; see BlindR7Cfg and the robot's *_R7; resume from the
# sixth round's model, widened for the gait clock with scripts/blind/widen_obs.py).
for task_id, terrain, runner in (
  ("Unitree-As2W-Blind-Map3-Rest3", "map3", As2WFlatAnchoredMarchRunner),
  ("Unitree-As2W-Blind-Rest3", "stairs", MirrorVelocityOnPolicyRunner),
  ("Unitree-As2W-Blind-Flat-Rest3", "flat", MirrorVelocityOnPolicyRunner),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2w_blind_rest3_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2w_blind_rest3_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("as2w_blind", max_iterations=20000),
    runner_cls=runner,
  )

# Staged training from scratch (2026-10-22; see apply_blind_staged): the same ramped
# schedule for every robot, no flat anchor.
for task_id, terrain in (
  ("Unitree-As2W-Blind-Stairs-Scratch", "map3"),
  ("Unitree-As2W-Blind-Scratch", "stairs"),
  ("Unitree-As2W-Blind-Flat-Scratch", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2w_blind_scratch_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2w_blind_scratch_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("as2w_blind", max_iterations=15000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )
