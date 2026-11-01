from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_rl_cfg import make_blind_ppo_runner_cfg
from src.tasks.velocity.blind_env_cfg import PHASE_CLIP_ACTIONS, PHASE_FINAL, phase_entropy_coef
from src.tasks.velocity.rl.anchor import PhasedAnchoredRunner
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner

from .blind_env_cfgs import (
  unitree_b2_blind_heading_env_cfg,
  unitree_b2_blind_scratch_env_cfg,
)

# With the heading-error input (resume from the Straight model widened by
# scripts/blind/widen_obs.py): training (map3) and evaluation (stairs, flat) tasks.
for task_id, terrain in (
  ("Unitree-B2-Blind-R1-Map3-Heading", "map3"),
  ("Unitree-B2-Blind-R1-Heading", "stairs"),
  ("Unitree-B2-Blind-Flat-R1-Heading", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2_blind_heading_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2_blind_heading_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Evaluation tasks of the phased training below (stairs, flat; also the trainers' acceptance).
for task_id, terrain in (
  ("Unitree-B2-Blind-Scratch", "stairs"),
  ("Unitree-B2-Blind-Flat-Scratch", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2_blind_scratch_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2_blind_scratch_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=15000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Phased training from scratch (user 2026-10-23 ~02:00; see apply_blind_phased): flat
# teacher, then mixed with the stairs and anchored to the teacher (BLIND_ANCHOR_CHECKPOINT),
# then a polish at half the entropy; evaluated with the Scratch tasks above.
for phase, task_id, runner in (
  ("teacher", "Unitree-B2-Blind-Teacher", MirrorVelocityOnPolicyRunner),
  ("mixed", "Unitree-B2-Blind-Mixed", PhasedAnchoredRunner),
  ("polish", "Unitree-B2-Blind-Polish", PhasedAnchoredRunner),
):
  rl_cfg = make_blind_ppo_runner_cfg("b2_blind", max_iterations=PHASE_FINAL)
  rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
  rl_cfg.algorithm.entropy_coef = phase_entropy_coef(phase)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2_blind_scratch_env_cfg(terrain="map3", phase=phase),
    play_env_cfg=unitree_b2_blind_scratch_env_cfg(play=True, terrain="map3", phase=phase),
    rl_cfg=rl_cfg,
    runner_cls=runner,
  )
