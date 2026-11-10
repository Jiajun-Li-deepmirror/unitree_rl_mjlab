"""seed_run.py SEED SCRIPT ARGS... (depth_fe, 2026-10-31): go2-home's tools/seed_run.py: run an eval script
unchanged, with the env config's seed set to SEED (the play configs use seed 0, so every acceptance check draws the
same randomization). Through jobs/student_eval.py for a student: student_eval.py seed_run.py SEED SCRIPT ARGS."""
import os
import runpy
import sys

seed, script = int(sys.argv[1]), sys.argv[2]
sys.argv = [script] + sys.argv[3:]
sys.path[:1] = [os.path.dirname(os.path.abspath(script)), os.getcwd()]
import mjlab.tasks.registry as reg  # noqa: E402

_load = reg.load_env_cfg


def load_env_cfg(*a, **k):
  cfg = _load(*a, **k)
  cfg.seed = seed
  return cfg


reg.load_env_cfg = load_env_cfg
runpy.run_path(script, run_name="__main__")
