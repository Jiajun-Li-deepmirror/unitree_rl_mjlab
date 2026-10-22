from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_rl_cfg import make_blind_ppo_runner_cfg
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner, VelocityOnPolicyRunner

from .blind_env_cfgs import (
  B2_BLIND_GAIT08,
  B2_BLIND_NO_CLOCK,
  B2_BLIND_P1,
  B2_BLIND_P2,
  B2_BLIND_P3,
  B2_BLIND_Q1,
  B2_BLIND_Q2,
  B2_BLIND_R1,
  B2_BLIND_R1_STAIRS,
  unitree_b2_blind_env_cfg,
  unitree_b2_blind_heading_env_cfg,
  unitree_b2_blind_hip_env_cfg,
  unitree_b2_blind_hip2_env_cfg,
  unitree_b2_blind_rest2_env_cfg,
  unitree_b2_blind_rest3_env_cfg,
  unitree_b2_blind_scratch_env_cfg,
  unitree_b2_blind_rest_env_cfg,
  unitree_b2_blind_map3_env_cfg,
  unitree_b2_blind_straight_env_cfg,
)

# Blind joystick policy (see doc/go2_blind.md): flat ground first, then stairs.
register_mjlab_task(
  task_id="Unitree-B2-Blind-Flat",
  env_cfg=unitree_b2_blind_env_cfg(stairs=False),
  play_env_cfg=unitree_b2_blind_env_cfg(play=True, stairs=False),
  rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=3000),
  runner_cls=VelocityOnPolicyRunner,
)

# Flat-ground pilots (see blind_env_cfgs.py).
for task_id, robot in (
  ("Unitree-B2-Blind-Flat-Gait08", B2_BLIND_GAIT08),
  ("Unitree-B2-Blind-Flat-NoClock", B2_BLIND_NO_CLOCK),
  ("Unitree-B2-Blind-Flat-P1", B2_BLIND_P1),
  ("Unitree-B2-Blind-Flat-P2", B2_BLIND_P2),
  ("Unitree-B2-Blind-Flat-P3", B2_BLIND_P3),
  ("Unitree-B2-Blind-Flat-Q1", B2_BLIND_Q1),
  ("Unitree-B2-Blind-Flat-Q2", B2_BLIND_Q2),
  ("Unitree-B2-Blind-Flat-R1", B2_BLIND_R1),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2_blind_env_cfg(stairs=False, robot=robot),
    play_env_cfg=unitree_b2_blind_env_cfg(play=True, stairs=False, robot=robot),
    rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=3000),
    runner_cls=VelocityOnPolicyRunner,
  )

register_mjlab_task(
  task_id="Unitree-B2-Blind",
  env_cfg=unitree_b2_blind_env_cfg(stairs=True),
  play_env_cfg=unitree_b2_blind_env_cfg(play=True, stairs=True),
  rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=20000),
  runner_cls=VelocityOnPolicyRunner,
)

# Stairs on the R1 recipe: pyramid stairs (for the stairs evals) and the map3
# fine-tune that trains from the R1 flat model.
register_mjlab_task(
  task_id="Unitree-B2-Blind-R1",
  env_cfg=unitree_b2_blind_env_cfg(stairs=True, robot=B2_BLIND_R1_STAIRS),
  play_env_cfg=unitree_b2_blind_env_cfg(play=True, stairs=True, robot=B2_BLIND_R1_STAIRS),
  rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=20000),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2-Blind-R1-Map3",
  env_cfg=unitree_b2_blind_map3_env_cfg(),
  play_env_cfg=unitree_b2_blind_map3_env_cfg(play=True),
  rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=20000),
  runner_cls=VelocityOnPolicyRunner,
)

# The same, walking straight without an operator, with the left-right mirror loss.
register_mjlab_task(
  task_id="Unitree-B2-Blind-R1-Map3-Straight",
  env_cfg=unitree_b2_blind_straight_env_cfg(),
  play_env_cfg=unitree_b2_blind_straight_env_cfg(play=True),
  rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=20000),
  runner_cls=MirrorVelocityOnPolicyRunner,
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

# The heading tasks with the hip-load changes (third round, 2026-10-22; see
# BlindHipCfg; resume from a Heading model): training (map3) and evaluation
# (stairs, flat) tasks.
for task_id, terrain in (
  ("Unitree-B2-Blind-R1-Map3-Hip", "map3"),
  ("Unitree-B2-Blind-R1-Hip", "stairs"),
  ("Unitree-B2-Blind-Flat-R1-Hip", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2_blind_hip_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2_blind_hip_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )


# The fourth round (2026-10-22; see BlindR4Cfg and the robot's *_HIP2 / *_R4;
# resume from a third-round Hip model, As2W from its Heading model_8200):
# training (map3) and evaluation (stairs, flat) tasks.
for task_id, terrain in (
  ("Unitree-B2-Blind-R1-Map3-Hip2", "map3"),
  ("Unitree-B2-Blind-R1-Hip2", "stairs"),
  ("Unitree-B2-Blind-Flat-R1-Hip2", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2_blind_hip2_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2_blind_hip2_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )


# The fifth round (2026-10-22; the rest stance, see BlindR5Cfg and the robot's *_R5;
# resume from the fourth round's model).
for task_id, terrain in (
  ("Unitree-B2-Blind-R1-Map3-Rest", "map3"),
  ("Unitree-B2-Blind-R1-Rest", "stairs"),
  ("Unitree-B2-Blind-Flat-R1-Rest", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2_blind_rest_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2_blind_rest_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# The sixth round (2026-10-22; see BlindR6Cfg and the robot's *_R6; resume from the
# fifth round's model).
for task_id, terrain in (
  ("Unitree-B2-Blind-R1-Map3-Rest2", "map3"),
  ("Unitree-B2-Blind-R1-Rest2", "stairs"),
  ("Unitree-B2-Blind-Flat-R1-Rest2", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2_blind_rest2_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2_blind_rest2_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# The seventh round (2026-10-22; see BlindR7Cfg and the robot's *_R7; resume from the
# sixth round's model).
for task_id, terrain in (
  ("Unitree-B2-Blind-R1-Map3-Rest3", "map3"),
  ("Unitree-B2-Blind-R1-Rest3", "stairs"),
  ("Unitree-B2-Blind-Flat-R1-Rest3", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2_blind_rest3_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2_blind_rest3_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Staged training from scratch (2026-10-22; see apply_blind_staged): the same ramped
# schedule for every robot, no flat anchor.
for task_id, terrain in (
  ("Unitree-B2-Blind-Stairs-Scratch", "map3"),
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
