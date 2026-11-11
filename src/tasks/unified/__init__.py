"""The unified pipeline (go2-depth 2026-11-11, the user's redesign: one privileged teacher distilled into a blind
and a depth student; the acceptance items as training constraints; several seeds per run). New code only: the
blind / depth tasks under src/tasks/velocity are untouched and keep training the rp6 / gp1 runs.

This package registers the teachers so far (the students come next):

  Unitree-<R>-U-Teacher        privileged teacher, the recipe's shaping terms kept   (R = Go2, Go2W)
  Unitree-<R>-U-Teacher-Lean   the same with only the core rewards and the constrained costs

Read teacher_env.py first (what the teacher is), then curriculum.py (one run, progress-driven) and constraints.py
(the acceptance items as constraints); robots.py has the physical parameters and the environment it builds on,
runner.py the checkpointed curriculum state, symmetry.py the mirror loss's map. Registered only with UNIFIED_TASKS=1.
Launch: jobs/unified/teacher.sh. Notes: doc/unified_teacher.md.
"""

import os

MAX_ITERATIONS = 30000   # a cap: the launch script stops a seed once its checks pass


def _register() -> None:
  from mjlab.tasks.registry import register_mjlab_task

  from src.tasks.velocity.blind_env_cfg import PHASE_CLIP_ACTIONS
  from src.tasks.velocity.blind_rl_cfg import make_blind_ppo_runner_cfg

  from .robots import ROBOTS
  from .runner import UnifiedTeacherRunner
  from .teacher_env import unified_teacher_env_cfg

  for robot in ROBOTS.values():
    for suffix, lean in (("", False), ("-Lean", True)):
      rl_cfg = make_blind_ppo_runner_cfg(f"{robot.key}_u_teacher", max_iterations=MAX_ITERATIONS)
      rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
      register_mjlab_task(
        task_id=f"Unitree-{robot.name}-U-Teacher{suffix}",
        env_cfg=unified_teacher_env_cfg(robot, lean=lean),
        play_env_cfg=unified_teacher_env_cfg(robot, play=True, lean=lean),
        rl_cfg=rl_cfg,
        runner_cls=UnifiedTeacherRunner,
      )



# Only with UNIFIED_TASKS=1 (jobs/unified/teacher.sh sets it): src.tasks imports every package, and the boxes' other
# processes (the rp6 / gp1 chains' trainings and checks) should neither build these environments nor depend on them.
if os.environ.get("UNIFIED_TASKS") == "1":
  _register()
