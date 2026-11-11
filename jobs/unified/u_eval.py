"""u_eval.py SCRIPT ARGS... (go2-depth 2026-11-12): run one of the blind check scripts (scripts/blind/eval/*.py,
scripts/go2_blind/eval_stairs.py) on a unified student's stage-2 checkpoint, from the repo root, with UNIFIED_TASKS=1
(the -U-Student-<Kind>-Flat / -Stairs tasks). A stage-2 checkpoint is a PPO checkpoint (actor_state_dict = the
student), which the checks load as they load any policy; what they lack is resetting a recurrent policy's memory when
a robot resets (the depth student's GRU) -- added here, as jobs/student_eval.py does for the older students.
"""
import os
import runpy
import sys

os.environ.setdefault("UNIFIED_TASKS", "1")
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path[:1] = [os.path.dirname(os.path.abspath(script)), os.getcwd()]

if os.path.basename(script) == "map3_par.py":
  # its shards run eval_map3.py in subprocesses: through this file too (as jobs/student_eval.py)
  import subprocess

  HERE = os.path.abspath(__file__)

  def _via_here(cmd):
    if isinstance(cmd, list) and len(cmd) > 1 and str(cmd[1]).endswith("eval_map3.py"):
      return [cmd[0], HERE, *cmd[1:]]
    return cmd

  _run, _Popen = subprocess.run, subprocess.Popen
  subprocess.run = lambda cmd, *a, **k: _run(_via_here(cmd), *a, **k)

  class _P(_Popen):
    def __init__(self, cmd, *a, **k):
      super().__init__(_via_here(cmd), *a, **k)

  subprocess.Popen = _P

from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

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
