from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_rl_cfg import make_blind_ppo_runner_cfg
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner
from src.tasks.velocity.rl.anchor import AnchoredMirrorVelocityOnPolicyRunner

from .blind_env_cfgs import unitree_b2w_blind_env_cfg, unitree_b2w_blind_heading_env_cfg


class B2WFlatAnchoredHeadingRunner(AnchoredMirrorVelocityOnPolicyRunner):
  """Anchored on flat ground to the v5 flat model widened for the heading input,
  sideways commands included, straight forward / backward ones at 0.3."""

  ANCHOR_CHECKPOINT = "logs/rsl_rl/b2w_blind/2026-10-19_23-13-52_flat_v5_gbrake/model_7898_widened.pt"
  ANCHOR_SKIP_SIDEWAYS = False
  # Straight forward / backward too, lightly (see rl.anchor).
  ANCHOR_STRAIGHT_COEFF = 0.3


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

# Stairs with the heading-error input: training (map3, anchored) and evaluation
# (stairs, flat) tasks.
for task_id, terrain, runner in (
  ("Unitree-B2W-Blind-Map3-Heading", "map3", B2WFlatAnchoredHeadingRunner),
  ("Unitree-B2W-Blind-Heading", "stairs", MirrorVelocityOnPolicyRunner),
  ("Unitree-B2W-Blind-Flat-Heading", "flat", MirrorVelocityOnPolicyRunner),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2w_blind_heading_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2w_blind_heading_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2w_blind", max_iterations=20000),
    runner_cls=runner,
  )
