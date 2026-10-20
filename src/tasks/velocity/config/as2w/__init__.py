from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner
from src.tasks.velocity.rl.anchor import AnchoredMirrorVelocityOnPolicyRunner
from src.tasks.velocity.config.legged.rl_cfg import heading_student_runner_cfg
from src.tasks.velocity.rl.distillation import DepthDistillationRunner

from .depth_env_cfgs import (
  unitree_as2w_depth_student_env_cfg,
  unitree_as2w_depth_teacher_env_cfg,
)
from .rl_cfg import (
  unitree_as2w_depth_student_runner_cfg,
  unitree_as2w_depth_teacher_ppo_runner_cfg,
)

# Depth camera policy, teacher-student; the As2's recipe with wheels as feet.
# (No plain Rough/Flat tasks: their single position action can't drive the
# velocity-controlled wheels.)

register_mjlab_task(
  task_id="Unitree-As2W-Depth-Teacher-Flat",
  env_cfg=unitree_as2w_depth_teacher_env_cfg(stairs=False),
  play_env_cfg=unitree_as2w_depth_teacher_env_cfg(play=True, stairs=False),
  rl_cfg=unitree_as2w_depth_teacher_ppo_runner_cfg(max_iterations=3000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-As2W-Depth-Teacher",
  env_cfg=unitree_as2w_depth_teacher_env_cfg(stairs=True),
  play_env_cfg=unitree_as2w_depth_teacher_env_cfg(play=True, stairs=True),
  rl_cfg=unitree_as2w_depth_teacher_ppo_runner_cfg(max_iterations=20000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-As2W-Depth-Student",
  env_cfg=unitree_as2w_depth_student_env_cfg(),
  play_env_cfg=unitree_as2w_depth_student_env_cfg(play=True),
  rl_cfg=unitree_as2w_depth_student_runner_cfg(),
  runner_cls=DepthDistillationRunner,
)

# go2-blind's straight walking and heading-error input (d9ed8ba, see
# apply_depth_heading), with the left-right mirror loss: the stairs teacher
# (resumed from a model widened by scripts/go2_depth/widen_obs.py), its flat
# task and the student.
register_mjlab_task(
  task_id="Unitree-As2W-Depth-Teacher-Heading",
  env_cfg=unitree_as2w_depth_teacher_env_cfg(stairs=True, heading=True),
  play_env_cfg=unitree_as2w_depth_teacher_env_cfg(play=True, stairs=True, heading=True),
  rl_cfg=unitree_as2w_depth_teacher_ppo_runner_cfg(max_iterations=20000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-As2W-Depth-Teacher-Flat-Heading",
  env_cfg=unitree_as2w_depth_teacher_env_cfg(stairs=False, heading=True),
  play_env_cfg=unitree_as2w_depth_teacher_env_cfg(play=True, stairs=False, heading=True),
  rl_cfg=unitree_as2w_depth_teacher_ppo_runner_cfg(max_iterations=3000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-As2W-Depth-Student-Heading",
  env_cfg=unitree_as2w_depth_student_env_cfg(heading=True),
  play_env_cfg=unitree_as2w_depth_student_env_cfg(play=True, heading=True),
  rl_cfg=heading_student_runner_cfg(unitree_as2w_depth_student_runner_cfg()),
  runner_cls=DepthDistillationRunner,
)
# Its stairs training, anchored on flat ground to the flat teacher named by
# DEPTH_ANCHOR_CHECKPOINT (rl/anchor.py); evaluate with the task above.
register_mjlab_task(
  task_id="Unitree-As2W-Depth-Teacher-Heading-Anchored",
  env_cfg=unitree_as2w_depth_teacher_env_cfg(stairs=True, heading=True),
  play_env_cfg=unitree_as2w_depth_teacher_env_cfg(play=True, stairs=True, heading=True),
  rl_cfg=unitree_as2w_depth_teacher_ppo_runner_cfg(max_iterations=20000),
  runner_cls=AnchoredMirrorVelocityOnPolicyRunner,
)
