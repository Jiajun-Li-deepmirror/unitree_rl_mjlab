from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_env_cfg import PHASE_CLIP_ACTIONS, PHASE_FINAL, phase_entropy_coef
from src.tasks.velocity.rl.anchor import PhasedAnchoredRunner
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner, VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .blind_env_cfgs import (
  unitree_go2_blind_env_cfg,
  unitree_go2_blind_heading_env_cfg,
  unitree_go2_blind_hip_env_cfg,
  unitree_go2_blind_hip2_env_cfg,
  unitree_go2_blind_rest2_env_cfg,
  unitree_go2_blind_rest3_env_cfg,
  unitree_go2_blind_scratch_env_cfg,
  unitree_go2_blind_rest_env_cfg,
  unitree_go2_blind_map3_env_cfg,
)
from .rl_cfg import unitree_go2_blind_ppo_runner_cfg, unitree_go2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Flat-only check of joystick feel, turning in place and standing.
register_mjlab_task(
  task_id="Unitree-Go2-Blind-Flat",
  env_cfg=unitree_go2_blind_env_cfg(stairs=False),
  play_env_cfg=unitree_go2_blind_env_cfg(play=True, stairs=False),
  rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=3000),
  runner_cls=VelocityOnPolicyRunner,
)

# Flat + stairs. The released policy (deploy blind_v0) was trained in stages that
# added rewards along the way: A (from scratch) -> B -> Unitree-Go2-Blind -> Map3.
# StageA/StageB are the earlier reward sets, for reproducing it with
# scripts/go2_blind/train_release.sh (see doc/go2_blind.md).
register_mjlab_task(
  task_id="Unitree-Go2-Blind-StageA",
  env_cfg=unitree_go2_blind_env_cfg(terrain_clearance=False, foothold_gain=False),
  play_env_cfg=unitree_go2_blind_env_cfg(play=True, terrain_clearance=False, foothold_gain=False),
  rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=20000),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Blind-StageB",
  env_cfg=unitree_go2_blind_env_cfg(foothold_gain=False),
  play_env_cfg=unitree_go2_blind_env_cfg(play=True, foothold_gain=False),
  rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=20000),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Blind",
  env_cfg=unitree_go2_blind_env_cfg(stairs=True),
  play_env_cfg=unitree_go2_blind_env_cfg(play=True, stairs=True),
  rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=20000),
  runner_cls=VelocityOnPolicyRunner,
)

# Fine-tune of Unitree-Go2-Blind for the tinynav-sim map3 stairwell (resume from it).
register_mjlab_task(
  task_id="Unitree-Go2-Blind-Map3",
  env_cfg=unitree_go2_blind_map3_env_cfg(),
  play_env_cfg=unitree_go2_blind_map3_env_cfg(play=True),
  rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=4000, entropy_coef=0.005),
  runner_cls=VelocityOnPolicyRunner,
)

# The map3 fine-tune with legs on the ground kept upright on stairs (resume from
# the release model_20000; see stance_leg_tilt).
register_mjlab_task(
  task_id="Unitree-Go2-Blind-Map3-Tilt",
  env_cfg=unitree_go2_blind_map3_env_cfg(stance_tilt_weight=-10.0),
  play_env_cfg=unitree_go2_blind_map3_env_cfg(play=True, stance_tilt_weight=-10.0),
  rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=1500, entropy_coef=0.005),
  runner_cls=VelocityOnPolicyRunner,
)

# Map3-Tilt walking straight without an operator, with the heading-error input and
# the left-right mirror loss (resume from its model widened by
# scripts/blind/widen_obs.py): training (map3) and evaluation (stairs, flat) tasks.
for task_id, terrain in (
  ("Unitree-Go2-Blind-Map3-Heading", "map3"),
  ("Unitree-Go2-Blind-Heading", "stairs"),
  ("Unitree-Go2-Blind-Flat-Heading", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2_blind_heading_env_cfg(terrain=terrain),
    play_env_cfg=unitree_go2_blind_heading_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=1500, entropy_coef=0.005),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# The heading tasks with the hip-load changes (third round, 2026-10-22; see
# BlindHipCfg; resume from a Heading model): training (map3) and evaluation
# (stairs, flat) tasks.
for task_id, terrain in (
  ("Unitree-Go2-Blind-Map3-Hip", "map3"),
  ("Unitree-Go2-Blind-Hip", "stairs"),
  ("Unitree-Go2-Blind-Flat-Hip", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2_blind_hip_env_cfg(terrain=terrain),
    play_env_cfg=unitree_go2_blind_hip_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=1500, entropy_coef=0.005),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )


# The fourth round (2026-10-22; see BlindR4Cfg and the robot's *_HIP2 / *_R4;
# resume from a third-round Hip model, As2W from its Heading model_8200):
# training (map3) and evaluation (stairs, flat) tasks.
for task_id, terrain in (
  ("Unitree-Go2-Blind-Map3-Hip2", "map3"),
  ("Unitree-Go2-Blind-Hip2", "stairs"),
  ("Unitree-Go2-Blind-Flat-Hip2", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2_blind_hip2_env_cfg(terrain=terrain),
    play_env_cfg=unitree_go2_blind_hip2_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=1500, entropy_coef=0.005),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )


# The fifth round (2026-10-22; the rest stance, see BlindR5Cfg and the robot's *_R5;
# resume from the fourth round's model).
for task_id, terrain in (
  ("Unitree-Go2-Blind-Map3-Rest", "map3"),
  ("Unitree-Go2-Blind-Rest", "stairs"),
  ("Unitree-Go2-Blind-Flat-Rest", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2_blind_rest_env_cfg(terrain=terrain),
    play_env_cfg=unitree_go2_blind_rest_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=1500, entropy_coef=0.005),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# The sixth round (2026-10-22; see BlindR6Cfg and the robot's *_R6; resume from the
# fifth round's model).
for task_id, terrain in (
  ("Unitree-Go2-Blind-Map3-Rest2", "map3"),
  ("Unitree-Go2-Blind-Rest2", "stairs"),
  ("Unitree-Go2-Blind-Flat-Rest2", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2_blind_rest2_env_cfg(terrain=terrain),
    play_env_cfg=unitree_go2_blind_rest2_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=1500, entropy_coef=0.005),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# The seventh round (2026-10-22; see BlindR7Cfg and the robot's *_R7; resume from the
# sixth round's model).
for task_id, terrain in (
  ("Unitree-Go2-Blind-Map3-Rest3", "map3"),
  ("Unitree-Go2-Blind-Rest3", "stairs"),
  ("Unitree-Go2-Blind-Flat-Rest3", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2_blind_rest3_env_cfg(terrain=terrain),
    play_env_cfg=unitree_go2_blind_rest3_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=1500, entropy_coef=0.005),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Staged training from scratch (2026-10-22; see apply_blind_staged): the same ramped
# schedule for every robot, no flat anchor.
for task_id, terrain in (
  ("Unitree-Go2-Blind-Stairs-Scratch", "map3"),
  ("Unitree-Go2-Blind-Scratch", "stairs"),
  ("Unitree-Go2-Blind-Flat-Scratch", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2_blind_scratch_env_cfg(terrain=terrain),
    play_env_cfg=unitree_go2_blind_scratch_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=15000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Staged training from scratch with the health-gated schedule (2026-10-23; see
# mdp.staged_clock): evaluated with the Scratch tasks above.
register_mjlab_task(
  task_id="Unitree-Go2-Blind-Stairs-Gated",
  env_cfg=unitree_go2_blind_scratch_env_cfg(terrain="map3", gated=True),
  play_env_cfg=unitree_go2_blind_scratch_env_cfg(play=True, terrain="map3", gated=True),
  rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=15000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)

# Phased training from scratch (user 2026-10-23 ~02:00; see apply_blind_phased): flat
# teacher, then mixed with the stairs and anchored to the teacher (BLIND_ANCHOR_CHECKPOINT),
# then a polish at half the entropy; evaluated with the Scratch tasks above.
for phase, task_id, runner in (
  ("teacher", "Unitree-Go2-Blind-Teacher", MirrorVelocityOnPolicyRunner),
  ("mixed", "Unitree-Go2-Blind-Mixed", PhasedAnchoredRunner),
  ("polish", "Unitree-Go2-Blind-Polish", PhasedAnchoredRunner),
):
  rl_cfg = unitree_go2_blind_ppo_runner_cfg(max_iterations=PHASE_FINAL)
  rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
  rl_cfg.algorithm.entropy_coef = phase_entropy_coef(phase)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2_blind_scratch_env_cfg(terrain="map3", phase=phase),
    play_env_cfg=unitree_go2_blind_scratch_env_cfg(play=True, terrain="map3", phase=phase),
    rl_cfg=rl_cfg,
    runner_cls=runner,
  )

# Depth camera (port of go2-depth, 2026-10-27; see depth_env_cfg.py). Teacher: the
# phased blind tasks above plus a privileged height scan cast from above the walls,
# same phases, runners and BLIND_PILOT; evaluated with the Depth-*-Scratch tasks.
# Student: go2-depth's depth camera and distillation from a teacher checkpoint
# (--agent.teacher-checkpoint=<model_N.pt>), on the polish-phase terrain and commands.
from src.tasks.velocity.depth_env_cfg import apply_depth_student, apply_depth_teacher  # noqa: E402
from src.tasks.velocity.rl.distillation import DepthDistillationRunner  # noqa: E402

from .rl_cfg import unitree_go2_depth_student_runner_cfg  # noqa: E402

for phase, task_id, runner in (
  ("teacher", "Unitree-Go2-Depth-Teacher", MirrorVelocityOnPolicyRunner),
  ("mixed", "Unitree-Go2-Depth-Mixed", PhasedAnchoredRunner),
  ("polish", "Unitree-Go2-Depth-Polish", PhasedAnchoredRunner),
):
  rl_cfg = unitree_go2_blind_ppo_runner_cfg(max_iterations=PHASE_FINAL)
  rl_cfg.experiment_name = "go2_depth_teacher"
  rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
  rl_cfg.algorithm.entropy_coef = phase_entropy_coef(phase)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(terrain="map3", phase=phase)),
    play_env_cfg=apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(play=True, terrain="map3", phase=phase)),
    rl_cfg=rl_cfg,
    runner_cls=runner,
  )

for task_id, terrain in (
  ("Unitree-Go2-Depth-Stairs-Scratch", "map3"),
  ("Unitree-Go2-Depth-Scratch", "stairs"),
  ("Unitree-Go2-Depth-Flat-Scratch", "flat"),
):
  rl_cfg = unitree_go2_blind_ppo_runner_cfg(max_iterations=PHASE_FINAL)
  rl_cfg.experiment_name = "go2_depth_teacher"
  register_mjlab_task(
    task_id=task_id,
    env_cfg=apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(terrain=terrain)),
    play_env_cfg=apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(play=True, terrain=terrain)),
    rl_cfg=rl_cfg,
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

register_mjlab_task(
  task_id="Unitree-Go2-Depth-Student",
  env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(terrain="map3", phase="polish"))
  ),
  play_env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(play=True, terrain="map3", phase="polish")),
    play=True,
  ),
  rl_cfg=unitree_go2_depth_student_runner_cfg(),
  runner_cls=DepthDistillationRunner,
)
