from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_rl_cfg import make_blind_ppo_runner_cfg
from src.tasks.velocity.blind_env_cfg import PHASE_CLIP_ACTIONS, PHASE_FINAL, phase_entropy_coef
from src.tasks.velocity.rl.anchor import PhasedAnchoredRunner
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner
from src.tasks.velocity.rl.anchor import AnchoredMirrorVelocityOnPolicyRunner

from .blind_env_cfgs import (
  unitree_b2w_blind_heading_env_cfg,
  unitree_b2w_blind_scratch_env_cfg,
)


class B2WFlatAnchoredHeadingRunner(AnchoredMirrorVelocityOnPolicyRunner):
  """Anchored on flat ground to the v5 flat model widened for the heading input,
  sideways commands included, straight forward / backward ones at 0.3."""

  ANCHOR_CHECKPOINT = "logs/rsl_rl/b2w_blind/2026-10-19_23-13-52_flat_v5_gbrake/model_7898_widened.pt"
  ANCHOR_SKIP_SIDEWAYS = False
  # Straight forward / backward too, lightly (see rl.anchor).
  ANCHOR_STRAIGHT_COEFF = 0.3

# Stairs with the heading-error input: training (map3, anchored) and evaluation
# (stairs, flat) tasks.
for task_id, terrain, runner in (
  ("Unitree-B2W-Blind-Map3-Heading", "map3", B2WFlatAnchoredHeadingRunner),
  ("Unitree-B2W-Blind-Heading", "stairs", MirrorVelocityOnPolicyRunner),
  ("Unitree-B2W-Blind-Flat-Heading", "flat", MirrorVelocityOnPolicyRunner),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2w_blind_heading_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2w_blind_heading_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2w_blind", max_iterations=20000),
    runner_cls=runner,
  )


# Evaluation tasks of the phased training below (stairs, flat; also the trainers' acceptance).
for task_id, terrain in (
  ("Unitree-B2W-Blind-Scratch", "stairs"),
  ("Unitree-B2W-Blind-Flat-Scratch", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2w_blind_scratch_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2w_blind_scratch_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2w_blind", max_iterations=15000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Phased training from scratch (user 2026-10-23 ~02:00; see apply_blind_phased): flat
# teacher, then mixed with the stairs and anchored to the teacher (BLIND_ANCHOR_CHECKPOINT),
# then a polish at half the entropy; evaluated with the Scratch tasks above.
for phase, task_id, runner in (
  ("teacher", "Unitree-B2W-Blind-Teacher", MirrorVelocityOnPolicyRunner),
  ("mixed", "Unitree-B2W-Blind-Mixed", PhasedAnchoredRunner),
  ("polish", "Unitree-B2W-Blind-Polish", PhasedAnchoredRunner),
):
  rl_cfg = make_blind_ppo_runner_cfg("b2w_blind", max_iterations=PHASE_FINAL)
  rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
  rl_cfg.algorithm.entropy_coef = phase_entropy_coef(phase)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2w_blind_scratch_env_cfg(terrain="map3", phase=phase),
    play_env_cfg=unitree_b2w_blind_scratch_env_cfg(play=True, terrain="map3", phase=phase),
    rl_cfg=rl_cfg,
    runner_cls=runner,
  )
