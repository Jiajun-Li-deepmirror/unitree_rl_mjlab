#!/usr/bin/env python3
"""recipe_test.py [--dump FILE | --against FILE] [--robots Go2,Go2W,As2] -- the recipe in code builds what the
probes' command lines built (go2-depth 2026-11-13).

The unified teacher's recipe was a line of U_ switches on every probe's command line (the launch scripts) and in the
gate's environment (taken from the run's timeline); the formal scripts (jobs/unified/teacher.sh, stairs_gate.sh,
flat_gate.sh, students.sh, accept.sh) carry none. src/tasks/unified/recipe.py makes that line the code's default per
robot. This test proves the two are the same thing, on fingerprints of the built environment configurations (every
task the package registers for a robot: the teacher's training / play / lean, its flat and stairs checks, the blind
and the depth student on the three terrains; a fingerprint = the configuration with every dataclass, function,
closure and tensor written out) and on the two switches the runner reads (the entropy bonus, the std cap):

  no option      (1) a build is the same twice (the fingerprint is stable);
                 (2) clean environment == U_RECIPE=0 + the probes' line, PROBE_LINES below (the recipe written a
                     second time, as the command lines had it): the code's default is the probes' recipe;
                 (3) an explicit switch wins over the recipe's value, and U_RECIPE=0 alone builds the plain task.
  --dump FILE    the fingerprints of U_RECIPE=0 + the probes' line, to FILE (run it on the commit BEFORE recipe.py:
                 U_RECIPE means nothing there, the line alone decides)
  --against FILE (2) against that file too: this code with the line == that commit with the line, so the formal run
                 from the clean environment trains what the probes trained.

Local, no GPU: PYTHONPATH=$PWD python scripts/unified/recipe_test.py"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import sys
from contextlib import contextmanager

# the probes' command lines (S/launch/launch_branch.sh's arguments, the runs' timeline env_vars) -- NOT derived from
# recipe.py: the test compares the two. Since 2026-11-13 (the Go2's formal run from zero under the probes' line ended
# its episodes by body contacts) the legged lines carry U_ALIVE=4 and the Go2's U_STAIRS_P=0.0,0.15 as well: the line
# a probe from zero is started with
_PK = "U_Q_SCALE_MAX=1 U_TERMINATED_Q=10 U_ENTROPY_Q=0.002 U_METER_EMA=2400 U_FLAT_ONLY=1 U_PAYLOAD_STICKY=0.05"
_LEGGED = (f"{_PK} U_STOP_FEET_DIR=3 U_STAIR_SPEED_H=1 U_HEADING_OBS=1 U_HEADING_LAG=2 U_FLAT_DIAG_AXIS=0.2 "
           "U_FLAT_SPEED_W=4 U_STAIR_PARK=10 U_BODY_CLEAR=50 U_COM_SUPPORT=15 U_ALIVE=4")
PROBE_LINES = {
  "Go2": f"{_LEGGED} U_STAIRS_P=0.0,0.15",
  "Go2W": f"{_PK} U_STAIR_SPEED_H=1 U_HEADING_OBS=1 U_HEADING_LAG=2 U_STAIR_APPROACH=0 U_BODY_CLEAR=50,0.06,0.10 "
          "U_STAIRS_P=0.0,0.15",
  "As2": f"{_LEGGED} U_STAND_POSE_EVEN=1 U_STAIRS_P=0.0,0.15",   # go2-home's mAM timeline env_vars (11-13 15:35)
}


@contextmanager
def environ(line: str = "", **more: str):
  """The U_ switches of the environment replaced by `line` ("A=1 B=2") and `more` for the duration."""
  keep = ("U_ROBOTS",)
  old = {k: v for k, v in os.environ.items() if k.startswith("U_") and k not in keep}
  new = dict(kv.split("=", 1) for kv in line.split()) | more
  try:
    for k in old:
      del os.environ[k]
    os.environ.update(new)
    yield
  finally:
    for k in new:
      os.environ.pop(k, None)
    os.environ.update(old)


def norm(x, seen: tuple = ()):
  """x as plain data: dataclasses and objects by their fields, functions by name, defaults and closure, tensors and
  arrays by their values."""
  if isinstance(x, (str, int, bool, type(None))):
    return x
  if isinstance(x, float):
    return repr(x)
  if id(x) in seen:
    return "<cycle>"
  seen = seen + (id(x),)
  if isinstance(x, dict):
    return {str(k): norm(v, seen) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
  if isinstance(x, (list, tuple)):
    return [norm(v, seen) for v in x]
  if isinstance(x, (set, frozenset)):
    return sorted(json.dumps(norm(v, seen), sort_keys=True) for v in x)
  if hasattr(x, "tolist") and hasattr(x, "shape"):   # a tensor, an array
    return {"__array__": list(x.shape), "v": norm(x.tolist(), seen)}
  if isinstance(x, functools.partial):
    return {"__partial__": norm(x.func, seen), "args": norm(x.args, seen), "kw": norm(x.keywords, seen)}
  if isinstance(x, type):
    return f"<class {x.__module__}.{x.__qualname__}>"
  if dataclasses.is_dataclass(x):
    return {"__dc__": f"{type(x).__module__}.{type(x).__qualname__}",
            **{f.name: norm(getattr(x, f.name, None), seen) for f in dataclasses.fields(x)}}
  if callable(x) and hasattr(x, "__qualname__"):
    out = {"__fn__": f"{getattr(x, '__module__', '?')}.{x.__qualname__}"}
    if getattr(x, "__defaults__", None):
      out["defaults"] = norm(x.__defaults__, seen)
    if getattr(x, "__kwdefaults__", None):
      out["kwdefaults"] = norm(x.__kwdefaults__, seen)
    cells = []
    for c in getattr(x, "__closure__", None) or ():
      try:
        cells.append(norm(c.cell_contents, seen))
      except ValueError:
        cells.append("<empty>")
    if cells:
      out["closure"] = cells
    if getattr(x, "__self__", None) is not None and not isinstance(x.__self__, type(sys)):
      out["self"] = norm(x.__self__, seen)
    return out
  if hasattr(x, "__dict__"):
    return {"__obj__": f"{type(x).__module__}.{type(x).__qualname__}", **norm(vars(x), seen)}
  if hasattr(x, "__slots__"):
    return {"__obj__": f"{type(x).__module__}.{type(x).__qualname__}",
            **{s: norm(getattr(x, s, None), seen) for s in x.__slots__}}
  return f"<{type(x).__module__}.{type(x).__qualname__} {x!r}>"


def builds(robot) -> dict:
  """Every environment configuration the package registers for the robot (__init__._register), by a name."""
  from src.tasks.unified import students as us
  from src.tasks.unified.teacher_env import unified_teacher_env_cfg as teacher

  out = {"teacher": teacher(robot), "teacher_play": teacher(robot, play=True), "teacher_lean": teacher(robot, lean=True)}
  for terrain in ("flat", "stairs"):
    out[f"teacher_{terrain}"] = teacher(robot, terrain=terrain)
    out[f"teacher_{terrain}_play"] = teacher(robot, play=True, terrain=terrain)
  for kind in ("blind", "depth"):
    out[f"{kind}"] = us.student_env_cfg(robot, kind)
    out[f"{kind}_play"] = us.student_env_cfg(robot, kind, play=True)
    for terrain in ("flat", "stairs"):
      out[f"{kind}_{terrain}"] = us.student_env_cfg(robot, kind, terrain=terrain)
      out[f"{kind}_{terrain}_play"] = us.student_env_cfg(robot, kind, play=True, terrain=terrain)
  return out


def prints(robot) -> dict[str, str]:
  """The fingerprint (a hash of the normalized configuration) of each of the robot's builds, and "_full": the
  normalized training configuration itself (what a difference is shown on)."""
  out = {}
  for name, cfg in builds(robot).items():
    n = norm(cfg)
    out[name] = hashlib.sha256(json.dumps(n, sort_keys=True).encode()).hexdigest()[:16]
    if name == "teacher":
      out["_full"] = n
  return out


def diff(a, b, path: str = "", out: list | None = None, limit: int = 12) -> list[str]:
  """Where two normalized configurations differ (the first `limit` places)."""
  out = [] if out is None else out
  if len(out) >= limit:
    return out
  if isinstance(a, dict) and isinstance(b, dict):
    for k in sorted(set(a) | set(b)):
      if k not in a or k not in b:
        out.append(f"{path}/{k}: only in the {'second' if k not in a else 'first'}")
      else:
        diff(a[k], b[k], f"{path}/{k}", out, limit)
  elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
    for i, (x, y) in enumerate(zip(a, b)):
      diff(x, y, f"{path}[{i}]", out, limit)
  elif a != b:
    out.append(f"{path}: {str(a)[:80]} != {str(b)[:80]}")
  return out


def same(what: str, a: dict, b: dict) -> bool:
  bad = [k for k in a if k != "_full" and a[k] != b.get(k)]
  if not bad:
    print(f"  ok    {what} ({len(a) - 1} builds)")
    return True
  print(f"  FAIL  {what}: {' '.join(bad)}")
  for line in diff(a.get("_full"), b.get("_full")):
    print(f"        {line}")
  return False


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dump")
  ap.add_argument("--against")
  ap.add_argument("--robots", default="Go2,Go2W,As2")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)   # robots.ROBOTS (read at import)
  os.environ.pop("UNIFIED_TASKS", None)      # no registration: the builders are called here
  sys.path.insert(0, os.getcwd())
  from src.tasks.unified.robots import ROBOTS

  if a.dump:
    out = {}
    for n in names:
      with environ(PROBE_LINES[n], U_RECIPE="0"):
        out[n] = prints(ROBOTS[n])
      print(f"{n}: {len(out[n]) - 1} builds")
    with open(a.dump, "w") as f:
      json.dump(out, f)
    return 0

  from src.tasks.unified import recipe as urecipe

  before = json.load(open(a.against)) if a.against else None
  ok = True
  for n in names:
    robot = ROBOTS[n]
    print(f"{n}: {' '.join(f'{k}={v}' for k, v in urecipe.recipe(robot).items())}")
    with environ(PROBE_LINES[n], U_RECIPE="0"):
      line = prints(robot)
      again = prints(robot)
      line_run = urecipe.runner_switches(robot)
    ok &= same("a build is the same twice", line, again)
    with environ():
      clean = prints(robot)
      clean_run = urecipe.runner_switches(robot)
      assert not urecipe.explicit(robot), urecipe.explicit(robot)
    ok &= same("clean environment == U_RECIPE=0 + the probes' line", clean, line)
    if clean_run != line_run:
      ok = False
      print(f"  FAIL  the runner's switches: {clean_run} != {line_run}")
    else:
      print(f"  ok    the runner's switches: {clean_run}")
    if before is not None:
      ok &= same(f"U_RECIPE=0 + the probes' line == {a.against}", line, before[n])
    # an explicit switch wins; U_RECIPE=0 alone is the plain task
    with environ(U_PAYLOAD_STICKY="0.5"):
      one = prints(robot)
      assert urecipe.explicit(robot) == {"U_PAYLOAD_STICKY": "0.5"}, urecipe.explicit(robot)
    with environ(" ".join(kv for kv in PROBE_LINES[n].split() if not kv.startswith("U_PAYLOAD_STICKY=")),
                 U_PAYLOAD_STICKY="0.5", U_RECIPE="0"):
      ok &= same("an explicit switch wins over the recipe's value", one, prints(robot))
    if one["teacher"] == clean["teacher"]:
      ok = False
      print("  FAIL  the explicit switch changed nothing")
    with environ(U_RECIPE="0"):
      plain = prints(robot)
      assert urecipe.runner_switches(robot) == {}, urecipe.runner_switches(robot)
    if plain["teacher"] == clean["teacher"]:
      ok = False
      print("  FAIL  U_RECIPE=0 builds the recipe's task")
    else:
      print("  ok    U_RECIPE=0 alone builds the plain task")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
