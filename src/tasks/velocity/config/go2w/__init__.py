from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_rl_cfg import make_blind_ppo_runner_cfg
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner
from src.tasks.velocity.rl.anchor import AnchoredMirrorVelocityOnPolicyRunner

from .blind_env_cfgs import (
  unitree_go2w_blind_env_cfg,
  unitree_go2w_blind_heading_env_cfg,
  unitree_go2w_blind_map3_env_cfg,
  unitree_go2w_blind_straight_env_cfg,
)


class Go2WFlatAnchoredRunner(AnchoredMirrorVelocityOnPolicyRunner):
  """Anchored on flat ground to the accepted v5 flat model."""

  ANCHOR_CHECKPOINT = "logs/rsl_rl/go2w_blind/2026-10-19_22-15-36_flat_v5_gbrake/model_6997.pt"


class Go2WFlatAnchoredHeadingRunner(AnchoredMirrorVelocityOnPolicyRunner):
  """Anchored on flat ground to the v5 flat model widened for the heading input,
  sideways commands included, straight forward / backward ones at 0.3."""

  ANCHOR_CHECKPOINT = "logs/rsl_rl/go2w_blind/2026-10-19_22-15-36_flat_v5_gbrake/model_6997_widened.pt"
  ANCHOR_SKIP_SIDEWAYS = False
  # Straight forward / backward too, lightly (see rl.anchor).
  ANCHOR_STRAIGHT_COEFF = 0.3

# Blind joystick policy (see make_wheeled_blind_env_cfg): flat ground first, then stairs.
register_mjlab_task(
  task_id="Unitree-Go2W-Blind-Flat",
  env_cfg=unitree_go2w_blind_env_cfg(stairs=False),
  play_env_cfg=unitree_go2w_blind_env_cfg(play=True, stairs=False),
  rl_cfg=make_blind_ppo_runner_cfg("go2w_blind", max_iterations=3000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-Blind",
  env_cfg=unitree_go2w_blind_env_cfg(stairs=True),
  play_env_cfg=unitree_go2w_blind_env_cfg(play=True, stairs=True),
  rl_cfg=make_blind_ppo_runner_cfg("go2w_blind", max_iterations=20000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)

# Stairs fine-tune from the flat model: stance-leg tilt term and map3 walled flights.
register_mjlab_task(
  task_id="Unitree-Go2W-Blind-Map3",
  env_cfg=unitree_go2w_blind_map3_env_cfg(),
  play_env_cfg=unitree_go2w_blind_map3_env_cfg(play=True),
  rl_cfg=make_blind_ppo_runner_cfg("go2w_blind", max_iterations=20000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)

# The same, walking straight without an operator, anchored to the v5 flat model
# on flat ground.
register_mjlab_task(
  task_id="Unitree-Go2W-Blind-Map3-Straight",
  env_cfg=unitree_go2w_blind_straight_env_cfg(),
  play_env_cfg=unitree_go2w_blind_straight_env_cfg(play=True),
  rl_cfg=make_blind_ppo_runner_cfg("go2w_blind", max_iterations=20000),
  runner_cls=Go2WFlatAnchoredRunner,
)

# With the heading-error input (resume from the Straight model widened by
# scripts/blind/widen_obs.py): training (map3, anchored) and evaluation (stairs,
# flat) tasks.
for task_id, terrain, runner in (
  ("Unitree-Go2W-Blind-Map3-Heading", "map3", Go2WFlatAnchoredHeadingRunner),
  ("Unitree-Go2W-Blind-Heading", "stairs", MirrorVelocityOnPolicyRunner),
  ("Unitree-Go2W-Blind-Flat-Heading", "flat", MirrorVelocityOnPolicyRunner),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2w_blind_heading_env_cfg(terrain=terrain),
    play_env_cfg=unitree_go2w_blind_heading_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("go2w_blind", max_iterations=20000),
    runner_cls=runner,
  )
