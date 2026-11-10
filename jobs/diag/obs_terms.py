"""obs_terms.py TASK: the observation terms per group under the current BLIND_PILOT (no env built)."""
import sys
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg(sys.argv[1])
for g, grp in cfg.observations.items():
  terms = {k: (getattr(v, "func", None).__name__ if hasattr(getattr(v, "func", None), "__name__") else type(getattr(v, "func", None)).__name__, getattr(v, "history_length", None)) for k, v in grp.terms.items()}
  print(g, sorted(terms.items()))
print("sensors", sorted(s.name for s in cfg.scene.sensors or ()))
