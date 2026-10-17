from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner
from src.tasks.velocity.rl.distillation import DepthDistillationRunner

from .depth_env_cfgs import (
  unitree_b2w_depth_student_env_cfg,
  unitree_b2w_depth_teacher_env_cfg,
)
from .rl_cfg import (
  unitree_b2w_depth_student_runner_cfg,
  unitree_b2w_depth_teacher_ppo_runner_cfg,
)

# Depth camera policy, teacher-student; the B2's recipe with wheels as feet.
# (No plain Rough/Flat tasks: their single position action can't drive the
# velocity-controlled wheels.)
register_mjlab_task(
  task_id="Unitree-B2W-Depth-Teacher-Flat",
  env_cfg=unitree_b2w_depth_teacher_env_cfg(stairs=False),
  play_env_cfg=unitree_b2w_depth_teacher_env_cfg(play=True, stairs=False),
  rl_cfg=unitree_b2w_depth_teacher_ppo_runner_cfg(max_iterations=3000),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Depth-Teacher",
  env_cfg=unitree_b2w_depth_teacher_env_cfg(stairs=True),
  play_env_cfg=unitree_b2w_depth_teacher_env_cfg(play=True, stairs=True),
  rl_cfg=unitree_b2w_depth_teacher_ppo_runner_cfg(max_iterations=20000),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Depth-Student",
  env_cfg=unitree_b2w_depth_student_env_cfg(),
  play_env_cfg=unitree_b2w_depth_student_env_cfg(play=True),
  rl_cfg=unitree_b2w_depth_student_runner_cfg(),
  runner_cls=DepthDistillationRunner,
)
