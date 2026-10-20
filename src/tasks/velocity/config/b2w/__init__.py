from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_rl_cfg import make_blind_ppo_runner_cfg
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner

from .blind_env_cfgs import unitree_b2w_blind_env_cfg

# Blind joystick policy (see make_wheeled_blind_env_cfg): flat ground first, then stairs.
register_mjlab_task(
  task_id="Unitree-B2W-Blind-Flat",
  env_cfg=unitree_b2w_blind_env_cfg(stairs=False),
  play_env_cfg=unitree_b2w_blind_env_cfg(play=True, stairs=False),
  rl_cfg=make_blind_ppo_runner_cfg("b2w_blind", max_iterations=3000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Blind",
  env_cfg=unitree_b2w_blind_env_cfg(stairs=True),
  play_env_cfg=unitree_b2w_blind_env_cfg(play=True, stairs=True),
  rl_cfg=make_blind_ppo_runner_cfg("b2w_blind", max_iterations=20000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)
