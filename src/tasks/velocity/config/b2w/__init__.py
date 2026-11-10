from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.blind_rl_cfg import make_blind_ppo_runner_cfg
from src.tasks.velocity.blind_env_cfg import PHASE_CLIP_ACTIONS, PHASE_FINAL, phase_entropy_coef
from src.tasks.velocity.rl.anchor import PhasedAnchoredRunner
from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner
from src.tasks.velocity.rl.anchor import AnchoredMirrorVelocityOnPolicyRunner

from .blind_env_cfgs import (
  unitree_b2w_blind_heading_env_cfg,
  unitree_b2w_blind_scratch_env_cfg,
)


class B2WFlatAnchoredHeadingRunner(AnchoredMirrorVelocityOnPolicyRunner):
  """Anchored on flat ground to the v5 flat model widened for the heading input,
  sideways commands included, straight forward / backward ones at 0.3."""

  ANCHOR_CHECKPOINT = "logs/rsl_rl/b2w_blind/2026-10-19_23-13-52_flat_v5_gbrake/model_7898_widened.pt"
  ANCHOR_SKIP_SIDEWAYS = False
  # Straight forward / backward too, lightly (see rl.anchor).
  ANCHOR_STRAIGHT_COEFF = 0.3

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


# Evaluation tasks of the phased training below (stairs, flat; also the trainers' acceptance).
for task_id, terrain in (
  ("Unitree-B2W-Blind-Scratch", "stairs"),
  ("Unitree-B2W-Blind-Flat-Scratch", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2w_blind_scratch_env_cfg(terrain=terrain),
    play_env_cfg=unitree_b2w_blind_scratch_env_cfg(play=True, terrain=terrain),
    rl_cfg=make_blind_ppo_runner_cfg("b2w_blind", max_iterations=15000),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

# Phased training from scratch (user 2026-10-23 ~02:00; see apply_blind_phased): flat
# teacher, then mixed with the stairs and anchored to the teacher (BLIND_ANCHOR_CHECKPOINT),
# then a polish at half the entropy; evaluated with the Scratch tasks above.
for phase, task_id, runner in (
  ("teacher", "Unitree-B2W-Blind-Teacher", MirrorVelocityOnPolicyRunner),
  ("mixed", "Unitree-B2W-Blind-Mixed", PhasedAnchoredRunner),
  ("polish", "Unitree-B2W-Blind-Polish", PhasedAnchoredRunner),
):
  rl_cfg = make_blind_ppo_runner_cfg("b2w_blind", max_iterations=PHASE_FINAL)
  rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
  rl_cfg.algorithm.entropy_coef = phase_entropy_coef(phase)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_b2w_blind_scratch_env_cfg(terrain="map3", phase=phase),
    play_env_cfg=unitree_b2w_blind_scratch_env_cfg(play=True, terrain="map3", phase=phase),
    rl_cfg=rl_cfg,
    runner_cls=runner,
  )

# Depth camera on the phased wheeled recipe (user 2026-10-29: depth from go2-home's latest
# wheeled blind models, Go2W polish 12000 / B2W polish 7000 / As2W mixed 7000, with its
# recipe), as the legged robots' (see config/go2/__init__.py and depth_env_cfg.py): the
# phased tasks plus the teacher's privileged scan, the Depth-*-Scratch evaluation tasks,
# and the student on the navcore camera (B2's mount: the same trunk) distilled from a
# teacher checkpoint. Contract b2w_depth_v4: the legged *_depth_v3 inputs on a wheeled robot
# (the 56-D proprio with the speed clock after the command, as contract_v2 orders it; the
# wheeled *_depth_v2 have no clock) plus
# the parking brake's hold in the ONNX metadata.
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
  ("teacher", "Unitree-B2W-Depth-Teacher", MirrorVelocityOnPolicyRunner),
  ("mixed", "Unitree-B2W-Depth-Mixed", PhasedAnchoredRunner),
  ("polish", "Unitree-B2W-Depth-Polish", PhasedAnchoredRunner),
):
  rl_cfg = make_blind_ppo_runner_cfg("b2w_depth_teacher", max_iterations=PHASE_FINAL)
  rl_cfg.clip_actions = PHASE_CLIP_ACTIONS
  rl_cfg.algorithm.entropy_coef = phase_entropy_coef(phase)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(terrain="map3", phase=phase)),
    play_env_cfg=apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(play=True, terrain="map3", phase=phase)),
    rl_cfg=rl_cfg,
    runner_cls=runner,
  )

for task_id, terrain in (
  ("Unitree-B2W-Depth-Stairs-Scratch", "map3"),
  ("Unitree-B2W-Depth-Scratch", "stairs"),
  ("Unitree-B2W-Depth-Flat-Scratch", "flat"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(terrain=terrain)),
    play_env_cfg=apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(play=True, terrain=terrain)),
    rl_cfg=make_blind_ppo_runner_cfg("b2w_depth_teacher", max_iterations=PHASE_FINAL),
    runner_cls=MirrorVelocityOnPolicyRunner,
  )

register_mjlab_task(
  task_id="Unitree-B2W-Depth-Student",
  env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(terrain="map3", phase="polish")),
    camera_pos=NAVCORE_CAMERA_POS["b2w"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
    real_camera=True,
  ),
  play_env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(play=True, terrain="map3", phase="polish")),
    play=True,
    camera_pos=NAVCORE_CAMERA_POS["b2w"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
  ),
  rl_cfg=make_depth_student_runner_cfg("b2w_depth_student", "b2w_depth_v4"),
  runner_cls=DepthDistillationRunner,
)

# The student's robustness fine-tune (user 2026-10-29, step 2a): the same, plus things in
# front of the lens and a covered lens (REAL_CAMERA_OCCLUSION); warm-started from a student.
register_mjlab_task(
  task_id="Unitree-B2W-Depth-Student-Occ",
  env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(terrain="map3", phase="polish")),
    camera_pos=NAVCORE_CAMERA_POS["b2w"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
    real_camera=True,
    occlusion=True,
  ),
  play_env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(play=True, terrain="map3", phase="polish")),
    play=True,
    camera_pos=NAVCORE_CAMERA_POS["b2w"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
  ),
  rl_cfg=make_depth_student_runner_cfg("b2w_depth_student", "b2w_depth_v4"),
  runner_cls=DepthDistillationRunner,
)

# Stair-edge fine-tune (user 2026-10-30): the -Occ randomization plus the real camera's
# smoothing of depth edges (REAL_CAMERA_EDGE_BLUR); warm-started from the -Occ student.
register_mjlab_task(
  task_id="Unitree-B2W-Depth-Student-Edge",
  env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(terrain="map3", phase="polish")),
    camera_pos=NAVCORE_CAMERA_POS["b2w"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
    real_camera=True,
    occlusion=True,
    edge_blur=True,
  ),
  play_env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(play=True, terrain="map3", phase="polish")),
    play=True,
    camera_pos=NAVCORE_CAMERA_POS["b2w"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
  ),
  rl_cfg=make_depth_student_runner_cfg("b2w_depth_student", "b2w_depth_v4"),
  runner_cls=DepthDistillationRunner,
)

# The student's PPO fine-tune against its teacher (rl/student_ppo.py; 2026-11-03, as the As2-W's): the wheelstep
# round's distilled B2W student slid back off the nosings in 43% of its tinynav-sim step-ups (delivered 17%) with a
# teacher better on every stairs metric; its own PPO on the wheelstep rewards. Same environment as -Student-Edge.
from src.tasks.velocity.blind_rl_cfg import make_depth_student_ppo_runner_cfg  # noqa: E402
from src.tasks.velocity.rl.student_ppo import DepthStudentPpoRunner  # noqa: E402

register_mjlab_task(
  task_id="Unitree-B2W-Depth-Student-Edge-RL",
  env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(terrain="map3", phase="polish")),
    camera_pos=NAVCORE_CAMERA_POS["b2w"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
    real_camera=True,
    occlusion=True,
    edge_blur=True,
  ),
  play_env_cfg=apply_depth_student(
    apply_depth_teacher(unitree_b2w_blind_scratch_env_cfg(play=True, terrain="map3", phase="polish")),
    play=True,
    camera_pos=NAVCORE_CAMERA_POS["b2w"],
    camera_quat=NAVCORE_CAMERA_QUAT,
    camera=NAVCORE_DEPTH_CAMERA,
  ),
  rl_cfg=make_depth_student_ppo_runner_cfg("b2w_depth_student", "b2w_depth_v4", clip_actions=PHASE_CLIP_ACTIONS),
  runner_cls=DepthStudentPpoRunner,
)
