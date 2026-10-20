from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner, VelocityOnPolicyRunner

from .blind_env_cfgs import (
  unitree_as2_blind_env_cfg,
  unitree_as2_blind_heading_env_cfg,
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
