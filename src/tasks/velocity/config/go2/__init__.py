from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .blind_env_cfgs import unitree_go2_blind_env_cfg
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

# Stage 1: flat-only check of joystick feel, turning in place and standing.
register_mjlab_task(
  task_id="Unitree-Go2-Blind-Flat",
  env_cfg=unitree_go2_blind_env_cfg(stairs=False),
  play_env_cfg=unitree_go2_blind_env_cfg(play=True, stairs=False),
  rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=3000),
  runner_cls=VelocityOnPolicyRunner,
)

# Stage 2: flat + stairs, trained from scratch.
register_mjlab_task(
  task_id="Unitree-Go2-Blind",
  env_cfg=unitree_go2_blind_env_cfg(stairs=True),
  play_env_cfg=unitree_go2_blind_env_cfg(play=True, stairs=True),
  rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=20000),
  runner_cls=VelocityOnPolicyRunner,
)
