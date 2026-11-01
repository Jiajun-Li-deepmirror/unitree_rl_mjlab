from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_env_cfg import PHASE_CLIP_ACTIONS, PHASE_FINAL, phase_entropy_coef
from src.tasks.velocity.rl.anchor import PhasedAnchoredRunner
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner, VelocityOnPolicyRunner

from .blind_env_cfgs import (
  unitree_as2_blind_heading_env_cfg,
  unitree_as2_blind_scratch_env_cfg,
)
from .env_cfgs import (
  unitree_as2_flat_env_cfg,
  unitree_as2_rough_env_cfg,
)
from .rl_cfg import unitree_as2_blind_ppo_runner_cfg, unitree_as2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-As2-Rough",
  env_cfg=unitree_as2_rough_env_cfg(),
  play_env_cfg=unitree_as2_rough_env_cfg(play=True),
  rl_cfg=unitree_as2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-As2-Flat",
  env_cfg=unitree_as2_flat_env_cfg(),
  play_env_cfg=unitree_as2_flat_env_cfg(play=True),
  rl_cfg=unitree_as2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# With the heading-error input: training (map3) and evaluation (stairs, flat) tasks.
for task_id, terrain in (
  ("Unitree-As2-Blind-Map3-Heading", "map3"),
  ("Unitree-As2-Blind-Heading", "stairs"),
  ("Unitree-As2-Blind-Flat-Heading", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2_blind_heading_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2_blind_heading_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_as2_blind_ppo_runner_cfg(max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Evaluation tasks of the phased training below (stairs, flat; also the trainers' acceptance).
for task_id, terrain in (
  ("Unitree-As2-Blind-Scratch", "stairs"),
  ("Unitree-As2-Blind-Flat-Scratch", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2_blind_scratch_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2_blind_scratch_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_as2_blind_ppo_runner_cfg(max_iterations=15000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Phased training from scratch (user 2026-10-23 ~02:00; see apply_blind_phased): flat
# teacher, then mixed with the stairs and anchored to the teacher (BLIND_ANCHOR_CHECKPOINT),
# then a polish at half the entropy; evaluated with the Scratch tasks above.
for phase, task_id, runner in (
  ("teacher", "Unitree-As2-Blind-Teacher", MirrorVelocityOnPolicyRunner),
  ("mixed", "Unitree-As2-Blind-Mixed", PhasedAnchoredRunner),
  ("polish", "Unitree-As2-Blind-Polish", PhasedAnchoredRunner),
):
  rl_cfg = unitree_as2_blind_ppo_runner_cfg(max_iterations=PHASE_FINAL)
  rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
  rl_cfg.algorithm.entropy_coef = phase_entropy_coef(phase)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2_blind_scratch_env_cfg(terrain="map3", phase=phase),
    play_env_cfg=unitree_as2_blind_scratch_env_cfg(play=True, terrain="map3", phase=phase),
    rl_cfg=rl_cfg,
    runner_cls=runner,
  )
