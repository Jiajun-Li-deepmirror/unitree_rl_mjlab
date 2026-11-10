from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_env_cfg import PHASE_CLIP_ACTIONS, PHASE_FINAL, phase_entropy_coef
from src.tasks.velocity.rl.anchor import PhasedAnchoredRunner
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner, VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .blind_env_cfgs import (
  unitree_go2_blind_env_cfg,
  unitree_go2_blind_heading_env_cfg,
  unitree_go2_blind_scratch_env_cfg,
  unitree_go2_blind_map3_env_cfg,
)
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

# The map3 fine-tune with legs on the ground kept upright on stairs (resume from
# the release model_20000; see stance_leg_tilt).
register_mjlab_task(
  task_id="Unitree-Go2-Blind-Map3-Tilt",
  env_cfg=unitree_go2_blind_map3_env_cfg(stance_tilt_weight=-10.0),
  play_env_cfg=unitree_go2_blind_map3_env_cfg(play=True, stance_tilt_weight=-10.0),
  rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=1500, entropy_coef=0.005),
  runner_cls=VelocityOnPolicyRunner,
)

# Map3-Tilt walking straight without an operator, with the heading-error input and
# the left-right mirror loss (resume from its model widened by
# scripts/blind/widen_obs.py): training (map3) and evaluation (stairs, flat) tasks.
for task_id, terrain in (
  ("Unitree-Go2-Blind-Map3-Heading", "map3"),
  ("Unitree-Go2-Blind-Heading", "stairs"),
  ("Unitree-Go2-Blind-Flat-Heading", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2_blind_heading_env_cfg(terrain=terrain),
    play_env_cfg=unitree_go2_blind_heading_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=1500, entropy_coef=0.005),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Evaluation tasks of the phased training below (stairs, flat; also the trainers' acceptance).
for task_id, terrain in (
  ("Unitree-Go2-Blind-Scratch", "stairs"),
  ("Unitree-Go2-Blind-Flat-Scratch", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2_blind_scratch_env_cfg(terrain=terrain),
    play_env_cfg=unitree_go2_blind_scratch_env_cfg(play=True, terrain=terrain),
    rl_cfg=unitree_go2_blind_ppo_runner_cfg(max_iterations=15000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Phased training from scratch (user 2026-10-23 ~02:00; see apply_blind_phased): flat
# teacher, then mixed with the stairs and anchored to the teacher (BLIND_ANCHOR_CHECKPOINT),
# then a polish at half the entropy; evaluated with the Scratch tasks above.
for phase, task_id, runner in (
  ("teacher", "Unitree-Go2-Blind-Teacher", MirrorVelocityOnPolicyRunner),
  ("mixed", "Unitree-Go2-Blind-Mixed", PhasedAnchoredRunner),
  ("polish", "Unitree-Go2-Blind-Polish", PhasedAnchoredRunner),
):
  rl_cfg = unitree_go2_blind_ppo_runner_cfg(max_iterations=PHASE_FINAL)
  rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
  rl_cfg.algorithm.entropy_coef = phase_entropy_coef(phase)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_go2_blind_scratch_env_cfg(terrain="map3", phase=phase),
    play_env_cfg=unitree_go2_blind_scratch_env_cfg(play=True, terrain="map3", phase=phase),
    rl_cfg=rl_cfg,
    runner_cls=runner,
  )

# Depth camera (port of go2-depth, 2026-10-27; see depth_env_cfg.py). Teacher: the
# phased blind tasks above plus a privileged height scan cast from above the walls,
# same phases, runners and BLIND_PILOT; evaluated with the Depth-*-Scratch tasks.
# Student: go2-depth's depth camera and distillation from a teacher checkpoint
# (--agent.teacher-checkpoint=<model_N.pt>), on the polish-phase terrain and commands.
from src.tasks.velocity.depth_env_cfg import (  # noqa: E402
  NAVCORE_CAMERA_POS,
  NAVCORE_CAMERA_QUAT,
  apply_depth_student,
  apply_depth_teacher,
)
from src.tasks.velocity.mdp.depth import NAVCORE_DEPTH_CAMERA  # noqa: E402
from src.tasks.velocity.rl.distillation import DepthDistillationRunner  # noqa: E402

from .rl_cfg import unitree_go2_depth_student_runner_cfg  # noqa: E402

for phase, task_id, runner in (
  ("teacher", "Unitree-Go2-Depth-Teacher", MirrorVelocityOnPolicyRunner),
  ("mixed", "Unitree-Go2-Depth-Mixed", PhasedAnchoredRunner),
  ("polish", "Unitree-Go2-Depth-Polish", PhasedAnchoredRunner),
):
  rl_cfg = unitree_go2_blind_ppo_runner_cfg(max_iterations=PHASE_FINAL)
  rl_cfg.experiment_name = "go2_depth_teacher"
  rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
  rl_cfg.algorithm.entropy_coef = phase_entropy_coef(phase)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(terrain="map3", phase=phase)),
    play_env_cfg=apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(play=True, terrain="map3", phase=phase)),
    rl_cfg=rl_cfg,
    runner_cls=runner,
  )

for task_id, terrain in (
  ("Unitree-Go2-Depth-Stairs-Scratch", "map3"),
  ("Unitree-Go2-Depth-Scratch", "stairs"),
  ("Unitree-Go2-Depth-Flat-Scratch", "flat"),
):
  rl_cfg = unitree_go2_blind_ppo_runner_cfg(max_iterations=PHASE_FINAL)
  rl_cfg.experiment_name = "go2_depth_teacher"
  register_mjlab_task(
    task_id=task_id,
    env_cfg=apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(terrain=terrain)),
    play_env_cfg=apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(play=True, terrain=terrain)),
    rl_cfg=rl_cfg,
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# The student's camera is the navcore unit's depth (its left lens, the intrinsics of
# its bag; user 2026-10-27), trained on a real stereo camera's errors (REAL_CAMERA_*).
register_mjlab_task(
  task_id="Unitree-Go2-Depth-Student",
  env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(terrain="map3", phase="polish")),
    camera_pos=NAVCORE_CAMERA_POS["go2"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
    real_camera=True,
  ),
  play_env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(play=True, terrain="map3", phase="polish")),
    play=True,
    camera_pos=NAVCORE_CAMERA_POS["go2"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
  ),
  rl_cfg=unitree_go2_depth_student_runner_cfg(),
  runner_cls=DepthDistillationRunner,
)

# The student's robustness fine-tune (user 2026-10-29, step 2a): the same, plus things in
# front of the lens and a covered lens (REAL_CAMERA_OCCLUSION); warm-started from a student.
register_mjlab_task(
  task_id="Unitree-Go2-Depth-Student-Occ",
  env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(terrain="map3", phase="polish")),
    camera_pos=NAVCORE_CAMERA_POS["go2"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
    real_camera=True,
    occlusion=True,
  ),
  play_env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(play=True, terrain="map3", phase="polish")),
    play=True,
    camera_pos=NAVCORE_CAMERA_POS["go2"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
  ),
  rl_cfg=unitree_go2_depth_student_runner_cfg(),
  runner_cls=DepthDistillationRunner,
)

# Stair-edge fine-tune (user 2026-10-30): the -Occ randomization plus the real camera's
# smoothing of depth edges (REAL_CAMERA_EDGE_BLUR); warm-started from the -Occ student.
register_mjlab_task(
  task_id="Unitree-Go2-Depth-Student-Edge",
  env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(terrain="map3", phase="polish")),
    camera_pos=NAVCORE_CAMERA_POS["go2"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
    real_camera=True,
    occlusion=True,
    edge_blur=True,
  ),
  play_env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_go2_blind_scratch_env_cfg(play=True, terrain="map3", phase="polish")),
    play=True,
    camera_pos=NAVCORE_CAMERA_POS["go2"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
  ),
  rl_cfg=unitree_go2_depth_student_runner_cfg(),
  runner_cls=DepthDistillationRunner,
)
