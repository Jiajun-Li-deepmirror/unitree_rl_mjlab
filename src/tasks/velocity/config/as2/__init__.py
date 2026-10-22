from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner, VelocityOnPolicyRunner

from .blind_env_cfgs import (
  unitree_as2_blind_env_cfg,
  unitree_as2_blind_heading_env_cfg,
  unitree_as2_blind_hip_env_cfg,
  unitree_as2_blind_hip2_env_cfg,
  unitree_as2_blind_rest2_env_cfg,
  unitree_as2_blind_rest3_env_cfg,
  unitree_as2_blind_scratch_env_cfg,
  unitree_as2_blind_rest_env_cfg,
  unitree_as2_blind_map3_env_cfg,
  unitree_as2_blind_straight_env_cfg,
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

# Blind joystick policy (see doc/go2_blind.md): flat ground first, then stairs.
register_mjlab_task(
  task_id="Unitree-As2-Blind-Flat",
  env_cfg=unitree_as2_blind_env_cfg(stairs=False),
  play_env_cfg=unitree_as2_blind_env_cfg(play=True, stairs=False),
  rl_cfg=unitree_as2_blind_ppo_runner_cfg(max_iterations=3000),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-As2-Blind",
  env_cfg=unitree_as2_blind_env_cfg(stairs=True),
  play_env_cfg=unitree_as2_blind_env_cfg(play=True, stairs=True),
  rl_cfg=unitree_as2_blind_ppo_runner_cfg(max_iterations=20000),
  runner_cls=VelocityOnPolicyRunner,
)

# Stairs with map3-like walled flights (resume from Unitree-As2-Blind).
register_mjlab_task(
  task_id="Unitree-As2-Blind-Map3",
  env_cfg=unitree_as2_blind_map3_env_cfg(),
  play_env_cfg=unitree_as2_blind_map3_env_cfg(play=True),
  rl_cfg=unitree_as2_blind_ppo_runner_cfg(max_iterations=20000),
  runner_cls=VelocityOnPolicyRunner,
)

# Walks straight without an operator (resume from Unitree-As2-Blind-Map3), with
# the left-right mirror loss.
register_mjlab_task(
  task_id="Unitree-As2-Blind-Map3-Straight",
  env_cfg=unitree_as2_blind_straight_env_cfg(),
  play_env_cfg=unitree_as2_blind_straight_env_cfg(play=True),
  rl_cfg=unitree_as2_blind_ppo_runner_cfg(max_iterations=20000),
  runner_cls=MirrorVelocityOnPolicyRunner,
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

# The heading tasks with the hip-load changes (third round, 2026-10-22; see
# BlindHipCfg; resume from a Heading model): training (map3) and evaluation
# (stairs, flat) tasks.
for task_id, terrain in (
  ("Unitree-As2-Blind-Map3-Hip", "map3"),
  ("Unitree-As2-Blind-Hip", "stairs"),
  ("Unitree-As2-Blind-Flat-Hip", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2_blind_hip_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2_blind_hip_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_as2_blind_ppo_runner_cfg(max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )


# The fourth round (2026-10-22; see BlindR4Cfg and the robot's *_HIP2 / *_R4;
# resume from a third-round Hip model, As2W from its Heading model_8200):
# training (map3) and evaluation (stairs, flat) tasks.
for task_id, terrain in (
  ("Unitree-As2-Blind-Map3-Hip2", "map3"),
  ("Unitree-As2-Blind-Hip2", "stairs"),
  ("Unitree-As2-Blind-Flat-Hip2", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2_blind_hip2_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2_blind_hip2_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_as2_blind_ppo_runner_cfg(max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )


# The fifth round (2026-10-22; the rest stance, see BlindR5Cfg and the robot's *_R5;
# resume from the fourth round's model).
for task_id, terrain in (
  ("Unitree-As2-Blind-Map3-Rest", "map3"),
  ("Unitree-As2-Blind-Rest", "stairs"),
  ("Unitree-As2-Blind-Flat-Rest", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2_blind_rest_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2_blind_rest_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_as2_blind_ppo_runner_cfg(max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# The sixth round (2026-10-22; see BlindR6Cfg and the robot's *_R6; resume from the
# fifth round's model).
for task_id, terrain in (
  ("Unitree-As2-Blind-Map3-Rest2", "map3"),
  ("Unitree-As2-Blind-Rest2", "stairs"),
  ("Unitree-As2-Blind-Flat-Rest2", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2_blind_rest2_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2_blind_rest2_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_as2_blind_ppo_runner_cfg(max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# The seventh round (2026-10-22; see BlindR7Cfg and the robot's *_R7; resume from the
# sixth round's model).
for task_id, terrain in (
  ("Unitree-As2-Blind-Map3-Rest3", "map3"),
  ("Unitree-As2-Blind-Rest3", "stairs"),
  ("Unitree-As2-Blind-Flat-Rest3", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_as2_blind_rest3_env_cfg(terrain=terrain),
    play_env_cfg=unitree_as2_blind_rest3_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_as2_blind_ppo_runner_cfg(max_iterations=20000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Staged training from scratch (2026-10-22; see apply_blind_staged): the same ramped
# schedule for every robot, no flat anchor.
for task_id, terrain in (
  ("Unitree-As2-Blind-Stairs-Scratch", "map3"),
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
