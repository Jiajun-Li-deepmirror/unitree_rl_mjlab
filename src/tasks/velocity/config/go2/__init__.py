from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner, VelocityOnPolicyRunner
from src.tasks.velocity.config.legged.rl_cfg import heading_student_runner_cfg
from src.tasks.velocity.rl.distillation import DepthDistillationRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .blind_env_cfgs import unitree_go2_blind_env_cfg
from .depth_env_cfgs import (
  unitree_go2_depth_student_env_cfg,
  unitree_go2_depth_teacher_env_cfg,
)
from .rl_cfg import (
  unitree_go2_blind_ppo_runner_cfg,
  unitree_go2_depth_student_runner_cfg,
  unitree_go2_depth_teacher_ppo_runner_cfg,
  unitree_go2_ppo_runner_cfg,
)

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

# Depth camera policy, teacher-student. Teacher: go2-blind + privileged height
# scan on map3-like walled stairs. Student: proprio + front depth camera,
# distilled from the teacher (--agent.teacher-checkpoint=<model_N.pt>).
register_mjlab_task(
  task_id="Unitree-Go2-Depth-Teacher-Flat",
  env_cfg=unitree_go2_depth_teacher_env_cfg(stairs=False),
  play_env_cfg=unitree_go2_depth_teacher_env_cfg(play=True, stairs=False),
  rl_cfg=unitree_go2_depth_teacher_ppo_runner_cfg(max_iterations=3000),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Depth-Teacher",
  env_cfg=unitree_go2_depth_teacher_env_cfg(stairs=True),
  play_env_cfg=unitree_go2_depth_teacher_env_cfg(play=True, stairs=True),
  rl_cfg=unitree_go2_depth_teacher_ppo_runner_cfg(max_iterations=20000),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Depth-Student",
  env_cfg=unitree_go2_depth_student_env_cfg(),
  play_env_cfg=unitree_go2_depth_student_env_cfg(play=True),
  rl_cfg=unitree_go2_depth_student_runner_cfg(),
  runner_cls=DepthDistillationRunner,
)

# go2-blind's straight walking and heading-error input (d9ed8ba, see
# apply_depth_heading), with the left-right mirror loss: the stairs teacher
# (resumed from a model widened by scripts/go2_depth/widen_obs.py), its flat
# task and the student.
register_mjlab_task(
  task_id="Unitree-Go2-Depth-Teacher-Heading",
  env_cfg=unitree_go2_depth_teacher_env_cfg(stairs=True, heading=True),
  play_env_cfg=unitree_go2_depth_teacher_env_cfg(play=True, stairs=True, heading=True),
  rl_cfg=unitree_go2_depth_teacher_ppo_runner_cfg(max_iterations=20000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-Go2-Depth-Teacher-Flat-Heading",
  env_cfg=unitree_go2_depth_teacher_env_cfg(stairs=False, heading=True),
  play_env_cfg=unitree_go2_depth_teacher_env_cfg(play=True, stairs=False, heading=True),
  rl_cfg=unitree_go2_depth_teacher_ppo_runner_cfg(max_iterations=3000),
  runner_cls=MirrorVelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-Go2-Depth-Student-Heading",
  env_cfg=unitree_go2_depth_student_env_cfg(heading=True),
  play_env_cfg=unitree_go2_depth_student_env_cfg(play=True, heading=True),
  rl_cfg=heading_student_runner_cfg(unitree_go2_depth_student_runner_cfg()),
  runner_cls=DepthDistillationRunner,
)
