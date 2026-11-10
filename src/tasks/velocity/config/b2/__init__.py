from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_rl_cfg import make_blind_ppo_runner_cfg
from src.tasks.velocity.blind_env_cfg import PHASE_CLIP_ACTIONS, PHASE_FINAL, phase_entropy_coef
from src.tasks.velocity.rl.anchor import PhasedAnchoredRunner
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner

from .blind_env_cfgs import (
  unitree_b2_blind_scratch_env_cfg,
)

# Evaluation tasks of the phased training below (stairs, flat; also the trainers' acceptance).
for task_id, terrain in (
  ("Unitree-B2-Blind-Scratch", "stairs"),
  ("Unitree-B2-Blind-Flat-Scratch", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2_blind_scratch_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2_blind_scratch_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2_blind", max_iterations=15000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Phased training from scratch (user 2026-10-23 ~02:00; see apply_blind_phased): flat
# teacher, then mixed with the stairs and anchored to the teacher (BLIND_ANCHOR_CHECKPOINT),
# then a polish at half the entropy; evaluated with the Scratch tasks above.
for phase, task_id, runner in (
  ("teacher", "Unitree-B2-Blind-Teacher", MirrorVelocityOnPolicyRunner),
  ("mixed", "Unitree-B2-Blind-Mixed", PhasedAnchoredRunner),
  ("polish", "Unitree-B2-Blind-Polish", PhasedAnchoredRunner),
):
  rl_cfg = make_blind_ppo_runner_cfg("b2_blind", max_iterations=PHASE_FINAL)
  rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
  rl_cfg.algorithm.entropy_coef = phase_entropy_coef(phase)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2_blind_scratch_env_cfg(terrain="map3", phase=phase),
    play_env_cfg=unitree_b2_blind_scratch_env_cfg(play=True, terrain="map3", phase=phase),
    rl_cfg=rl_cfg,
    runner_cls=runner,
  )

# Depth camera, as the Go2's (see config/go2/__init__.py and depth_env_cfg.py): the
# phased tasks plus the teacher's privileged scan, the Depth-*-Scratch evaluation
# tasks, and the student on the navcore camera (its B2 mount) distilled from a
# teacher checkpoint.
from src.tasks.velocity.blind_rl_cfg import make_depth_student_runner_cfg  # noqa: E402
from src.tasks.velocity.depth_env_cfg import (  # noqa: E402
  NAVCORE_CAMERA_POS,
  NAVCORE_CAMERA_QUAT,
  apply_depth_student,
  apply_depth_teacher,
)
from src.tasks.velocity.mdp.depth import NAVCORE_DEPTH_CAMERA  # noqa: E402
from src.tasks.velocity.rl.distillation import DepthDistillationRunner  # noqa: E402

for phase, task_id, runner in (
  ("teacher", "Unitree-B2-Depth-Teacher", MirrorVelocityOnPolicyRunner),
  ("mixed", "Unitree-B2-Depth-Mixed", PhasedAnchoredRunner),
  ("polish", "Unitree-B2-Depth-Polish", PhasedAnchoredRunner),
):
  rl_cfg = make_blind_ppo_runner_cfg("b2_depth_teacher", max_iterations=PHASE_FINAL)
  rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
  rl_cfg.algorithm.entropy_coef = phase_entropy_coef(phase)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=apply_depth_teacher(unitree_b2_blind_scratch_env_cfg(terrain="map3", phase=phase)),
    play_env_cfg=apply_depth_teacher(unitree_b2_blind_scratch_env_cfg(play=True, terrain="map3", phase=phase)),
    rl_cfg=rl_cfg,
    runner_cls=runner,
  )

for task_id, terrain in (
  ("Unitree-B2-Depth-Scratch", "stairs"),
  ("Unitree-B2-Depth-Flat-Scratch", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=apply_depth_teacher(unitree_b2_blind_scratch_env_cfg(terrain=terrain)),
    play_env_cfg=apply_depth_teacher(unitree_b2_blind_scratch_env_cfg(play=True, terrain=terrain)),
    rl_cfg=make_blind_ppo_runner_cfg("b2_depth_teacher", max_iterations=PHASE_FINAL),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Stair-edge fine-tune (user 2026-10-30): the -Occ randomization plus the real camera's
# smoothing of depth edges (REAL_CAMERA_EDGE_BLUR); warm-started from the -Occ student.
register_mjlab_task(
  task_id="Unitree-B2-Depth-Student-Edge",
  env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2_blind_scratch_env_cfg(terrain="map3", phase="polish")),
    camera_pos=NAVCORE_CAMERA_POS["b2"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
    real_camera=True,
    occlusion=True,
    edge_blur=True,
  ),
  play_env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2_blind_scratch_env_cfg(play=True, terrain="map3", phase="polish")),
    play=True,
    camera_pos=NAVCORE_CAMERA_POS["b2"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
  ),
  rl_cfg=make_depth_student_runner_cfg("b2_depth_student", "b2_depth_v3"),
  runner_cls=DepthDistillationRunner,
)

# The -Edge student's PPO fine-tune (go2-depth 2026-11-02, the user's choice for B2's rare falls): the same task,
# the distilled student as the actor with its teacher's critic and a pull toward the teacher (rl/student_ppo.py).
from src.tasks.velocity.blind_rl_cfg import make_depth_student_ppo_runner_cfg  # noqa: E402
from src.tasks.velocity.rl.student_ppo import DepthStudentPpoRunner  # noqa: E402

register_mjlab_task(
  task_id="Unitree-B2-Depth-Student-Edge-RL",
  env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2_blind_scratch_env_cfg(terrain="map3", phase="polish")),
    camera_pos=NAVCORE_CAMERA_POS["b2"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
    real_camera=True,
    occlusion=True,
    edge_blur=True,
  ),
  play_env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2_blind_scratch_env_cfg(play=True, terrain="map3", phase="polish")),
    play=True,
    camera_pos=NAVCORE_CAMERA_POS["b2"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
  ),
  rl_cfg=make_depth_student_ppo_runner_cfg("b2_depth_student", "b2_depth_v3", clip_actions=PHASE_CLIP_ACTIONS),
  runner_cls=DepthStudentPpoRunner,
)
