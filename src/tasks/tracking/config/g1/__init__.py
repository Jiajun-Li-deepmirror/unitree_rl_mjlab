from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (unitree_g1_flat_tracking_env_cfg, unitree_g1_leg_tracking_env_cfg,
                      unitree_g1_balanced_leg_tracking_env_cfg, unitree_g1_step_tracking_env_cfg,
                      unitree_g1_step_latency_tracking_env_cfg, unitree_g1_tight_leg_tracking_env_cfg,
                      unitree_g1_tight_leg_latency_tracking_env_cfg)
from .rl_cfg import unitree_g1_tracking_ppo_runner_cfg, unitree_g1_leg_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-Tracking",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-Legs",
  env_cfg=unitree_g1_leg_tracking_env_cfg(),
  play_env_cfg=unitree_g1_leg_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_leg_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-Legs-Balanced",
  env_cfg=unitree_g1_balanced_leg_tracking_env_cfg(),
  play_env_cfg=unitree_g1_balanced_leg_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_leg_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Base training from scratch with the Legs-Balanced rewards and the baseline
# PPO schedule; observations match No-State-Estimation for eval and deploy.
register_mjlab_task(
  task_id="Unitree-G1-Tracking-Leg-Rewards",
  env_cfg=unitree_g1_balanced_leg_tracking_env_cfg(),
  play_env_cfg=unitree_g1_balanced_leg_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Leg-Rewards plus a tight small-lift term and a 3x measured slip penalty.
register_mjlab_task(
  task_id="Unitree-G1-Tracking-Step-Rewards",
  env_cfg=unitree_g1_step_tracking_env_cfg(),
  play_env_cfg=unitree_g1_step_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Step-Rewards fine-tuned under a per-episode command latency, warm-started from
# a Step-Rewards model: first up to 20 ms, then up to 40 ms.
for _ms in (20, 40):
  register_mjlab_task(
    task_id=f"Unitree-G1-Tracking-Step-Rewards-Latency{_ms}",
    env_cfg=unitree_g1_step_latency_tracking_env_cfg(_ms),
    play_env_cfg=unitree_g1_step_latency_tracking_env_cfg(_ms, play=True),
    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )

# Step-Rewards plus a tight leg joint term, and its latency fine-tuning stages.
register_mjlab_task(
  task_id="Unitree-G1-Tracking-Tight-Legs",
  env_cfg=unitree_g1_tight_leg_tracking_env_cfg(),
  play_env_cfg=unitree_g1_tight_leg_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
for _ms in (20, 40):
  register_mjlab_task(
    task_id=f"Unitree-G1-Tracking-Tight-Legs-Latency{_ms}",
    env_cfg=unitree_g1_tight_leg_latency_tracking_env_cfg(_ms),
    play_env_cfg=unitree_g1_tight_leg_latency_tracking_env_cfg(_ms, play=True),
    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )
