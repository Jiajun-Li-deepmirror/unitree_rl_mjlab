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
