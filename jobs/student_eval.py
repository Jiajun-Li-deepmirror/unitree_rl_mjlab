"""student_eval.py SCRIPT ARGS... (2026-10-27): run one of the blind check scripts on a depth
student checkpoint, from the repo root.

The checks load a checkpoint with load_cfg={"actor": True}, which a distillation runner
ignores (the student would stay at its random start), and never reset the policy's memory
when a robot resets (the student has a GRU). Here both are fixed, and the student gets the
checks' terrains as tasks of its own (eval only, nothing is registered for training):
  Unitree-<R>-Depth-Student-Stairs, -Flat         like Depth-Scratch / Depth-Flat-Scratch,
                                                  clean depth (the play camera)
  Unitree-<R>-Depth-Student-Stairs-Real, -Flat-Real  the play scene with the training
                                                  camera's errors and mounting spread
                                                  (REAL_CAMERA_*) and proprio noise
Wheeled (2026-10-29, depth_wheeled): Go2W, B2W and As2W on their Scratch tasks (go2-home's
wheeled finals' recipe), from Unitree-<R>-Depth-Student.
  Unitree-<R>-Depth-Student-Stairs-RealEdge, -Flat-RealEdge  -Real plus the stereo camera's
                                                  smoothing of depth edges on every robot
                                                  (REAL_CAMERA_EDGE_BLUR at probability 1;
                                                  depth_fe, 2026-10-31)
map3_par.py runs its eval_map3.py shards through this file too.
"""
import os
import runpy
import subprocess
import sys

HERE = os.path.abspath(__file__)
script = sys.argv[1]
sys.argv = sys.argv[1:]
# As if SCRIPT had been run itself (its own directory first), plus the repo root.
sys.path[:1] = [os.path.dirname(os.path.abspath(script)), os.getcwd()]

if os.path.basename(script) == "map3_par.py":

  def _via_here(cmd):
    if isinstance(cmd, list) and len(cmd) > 1 and str(cmd[1]).endswith("eval_map3.py"):
      return [cmd[0], HERE, *cmd[1:]]
    return cmd

  _run, _Popen = subprocess.run, subprocess.Popen
  subprocess.run = lambda cmd, *a, **k: _run(_via_here(cmd), *a, **k)

  class _ViaHere(_Popen):
    def __init__(self, cmd, *a, **k):
      super().__init__(_via_here(cmd), *a, **k)

  subprocess.Popen = _ViaHere
else:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.velocity.config.as2 as as2
  import src.tasks.velocity.config.b2 as b2
  import src.tasks.velocity.config.go2 as go2
  import src.tasks.velocity.config.as2w as as2w
  import src.tasks.velocity.config.b2w as b2w
  import src.tasks.velocity.config.go2w as go2w
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls, register_mjlab_task
  from rsl_rl.algorithms import Distillation
  from rsl_rl.runners import OnPolicyRunner
  from src.tasks.velocity.depth_env_cfg import (
    NAVCORE_CAMERA_POS,
    NAVCORE_CAMERA_QUAT,
    apply_depth_student,
    apply_depth_teacher,
  )
  from src.tasks.velocity.mdp.depth import NAVCORE_DEPTH_CAMERA

  builders = {
    "Go2": go2.unitree_go2_blind_scratch_env_cfg,
    "As2": as2.unitree_as2_blind_scratch_env_cfg,
    "B2": b2.unitree_b2_blind_scratch_env_cfg,
    "Go2W": go2w.unitree_go2w_blind_scratch_env_cfg,
    "B2W": b2w.unitree_b2w_blind_scratch_env_cfg,
    "As2W": as2w.unitree_as2w_blind_scratch_env_cfg,
  }
  for robot, build in builders.items():
    student = f"Unitree-{robot}-Depth-Student"
    prefix = f"Unitree-{robot}-Depth-Student"
    camera = dict(
      camera_pos=NAVCORE_CAMERA_POS[robot.lower()], camera_quat=NAVCORE_CAMERA_QUAT, camera=NAVCORE_DEPTH_CAMERA
    )
    for name, terrain in (("Stairs", "stairs"), ("Flat", "flat")):
      for real in ("", "-Real", "-RealEdge"):
        play_cfg = apply_depth_student(
          apply_depth_teacher(build(play=True, terrain=terrain)),
          play=not real,
          real_camera=bool(real),
          edge_blur=real == "-RealEdge",
          **camera,
        )
        if real == "-RealEdge":
          play_cfg.observations["depth"].terms["depth"].params["edge_blur_prob"] = 1.0
        register_mjlab_task(
          task_id=f"{prefix}-{name}{real}",
          env_cfg=apply_depth_student(apply_depth_teacher(build(terrain=terrain)), real_camera=True, **camera),
          play_env_cfg=play_cfg,
          rl_cfg=load_rl_cfg(student),
          runner_cls=load_runner_cls(student),
        )

  _load = Distillation.load

  def load(self, loaded_dict, load_cfg, strict):
    if load_cfg == {"actor": True} and "student_state_dict" in loaded_dict:
      # The terrain head is training-only; a student from before it loads without it.
      missing, unexpected = self.student.load_state_dict(loaded_dict["student_state_dict"], strict=False)
      bad = [k for k in missing + unexpected if not k.startswith("aux_head.")]
      if bad:
        raise RuntimeError(f"student checkpoint mismatch: {bad}")
      print(f"[student_eval] loaded the student (iteration {loaded_dict.get('iter')})", flush=True)
      return False
    return _load(self, loaded_dict, load_cfg, strict)

  Distillation.load = load

  policies = []
  _inference = OnPolicyRunner.get_inference_policy

  def get_inference_policy(self, device=None):
    policy = _inference(self, device)
    if getattr(policy, "is_recurrent", False):
      policies.append(policy)
    return policy

  OnPolicyRunner.get_inference_policy = get_inference_policy

  _step = RslRlVecEnvWrapper.step

  def step(self, actions):
    out = _step(self, actions)
    dones = out[2]
    for policy in policies:
      hidden = policy.get_hidden_state()
      if hidden is not None and hidden.shape[-2] == dones.shape[0]:
        policy.reset(dones)
    return out

  RslRlVecEnvWrapper.step = step

runpy.run_path(script, run_name="__main__")
