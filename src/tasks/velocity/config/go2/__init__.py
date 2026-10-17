from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .blind_env_cfgs import unitree_go2_blind_env_cfg, unitree_go2_blind_map3_env_cfg
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
