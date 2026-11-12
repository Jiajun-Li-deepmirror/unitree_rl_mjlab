#!/usr/bin/env python3
"""teacher_gate.py RUN_DIR [--hold N] [--json] -- has the unified teacher of RUN_DIR passed (go2-depth 2026-11-11)?

The teacher is judged on its constraints in training (src/tasks/unified/constraints.py: every acceptance item as a
measurement per scan mode, a target and a multiplier, logged as Curriculum/u_constraints/<name>/{value_seen,
value_blind,target,bar,scale}); the acceptance proper is the students' (the user's redesign: "验收要以学生为准，老师需要
留出余量"). It passes at the first checkpoint K (a multiple of 100) where, over the last `hold` iterations (default
500):
  - the curriculum is through (Curriculum/u_progress/p = 1 throughout, and the stairs quality stage too:
    Curriculum/u_progress/q = 1 throughout -- a run from before that stage logs no q and is judged on p alone), and
  - every enforced constraint's value is on its bar's side (mean of the window; "min" items >= bar, "max" items
    <= bar) in every scan mode it applies to (the item's "modes": seen and blind, the tread placement items seen
    only -- the depth student copies the seen mode, the blind student the blind one), each with at least half the
    window measured. The bar is the students' (<name>/bar); the teacher's own target (<name>/target) sits a margin
    inside it and is not what is judged here. A gate-only item (enforced, no reward terms: the survival items,
    teacher_env.constraint_items) is judged at the bar in the code on disk, not at the one the training logged:
    its bar touches nothing in the training, so it can be set by a deploy without a restart.
Prints one line: "PASS K <items>" or "NOT YET <iteration> <the items that miss, value vs bar>" (an item reads
<name>_<mode>); --json the details. Which constraints are enforced, their senses and modes come from the teacher
task's items (registered with UNIFIED_TASKS=1). A run from before the modes or the margin (no value_<mode> / bar
tags) is judged on <name>/value against <name>/target, as it was.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def items_for(run: str) -> list[dict]:
  """The constraint items of the run's robot and kind (from the experiment directory: <robot>_u_teacher)."""
  sys.path.insert(0, os.getcwd())
  from src.tasks.unified.robots import ROBOTS
  from src.tasks.unified.teacher_env import constraint_items

  exp = os.path.basename(os.path.dirname(os.path.abspath(run)))
  robot = next(r for r in ROBOTS.values() if exp == f"{r.key}_u_teacher")
  return [c for c in constraint_items(robot.kind) if c.get("enforce", True)]


def series(acc: EventAccumulator, tag: str) -> dict[int, float]:
  try:
    return {s.step: s.value for s in acc.Scalars(tag)}
  except KeyError:
    return {}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("run")
  ap.add_argument("--hold", type=int, default=500)
  ap.add_argument("--every", type=int, default=100, help="the checkpoint interval (judged only there)")
  ap.add_argument("--json", action="store_true")
  a = ap.parse_args()
  files = sorted(glob.glob(os.path.join(a.run, "events.out.tfevents*")))
  if not files:
    print("NOT YET 0 no events")
    return 0
  acc = EventAccumulator(files[-1], size_guidance={"scalars": 0})
  acc.Reload()
  p = series(acc, "Curriculum/u_progress/p")
  q = series(acc, "Curriculum/u_progress/q")   # empty for a run from before the stairs quality stage
  items = items_for(a.run)
  vals, bars, sense = {}, {}, {}   # "<name>_<mode>" (or "<name>") -> the readings; the bar they are judged at
  for c in items:
    n = c["name"]
    tag = f"Curriculum/u_constraints/{n}/"
    bar = series(acc, tag + "bar") or series(acc, tag + "target")
    fixed = None if c["terms"] else c["target"]   # a gate-only item: the bar of the code on disk
    per = {f"{n}_{m}": series(acc, tag + f"value_{m}") for m in c.get("modes", ("seen", "blind"))}
    if not any(per.values()):
      per = {n: series(acc, tag + "value")}
    for key, v in per.items():
      vals[key], bars[key], sense[key] = v, ({i: fixed for i in v} if fixed is not None else bar), c["sense"]
  steps = sorted(p)
  if not steps:
    print("NOT YET 0 no progress logged")
    return 0
  last = steps[-1]
  report = {}
  last_miss = None
  for k in steps:
    if k % a.every:   # judged at the checkpoints (saved every 100 iterations): the pick must be one
      continue
    lo = k - a.hold + 1
    if lo < steps[0] or any(p.get(i, 0.0) < 1.0 for i in range(lo, k + 1) if i in p):
      continue
    if any(q[i] < 1.0 for i in range(lo, k + 1) if i in q):
      continue
    miss = []
    for n in vals:
      window = [i for i in range(lo, k + 1) if i in vals[n] and i in bars[n]]
      if len(window) < a.hold // 2:
        miss.append(f"{n} unmeasured")
        continue
      v = sum(vals[n][i] for i in window) / len(window)
      t = sum(bars[n][i] for i in window) / len(window)
      ok = v >= t if sense[n] == "min" else v <= t
      report[n] = {"value": round(v, 4), "bar": round(t, 4), "ok": ok}
      if not ok:
        miss.append(f"{n} {v:.4g} vs {t:.4g}")
    if not miss:
      print(f"PASS {k} {' '.join(sorted(report))}")
      if a.json:
        print(json.dumps(report))
      return 0
    last_miss = (k, miss)
  if last_miss is not None:
    k, miss = last_miss
    print(f"NOT YET {last} ({k}: {'; '.join(miss)})")
  else:
    print(f"NOT YET {last} (progress {p[last]:.3f}" + (f", quality {q[last]:.3f})" if last in q else ")"))
  if a.json:
    print(json.dumps(report))
  return 0


if __name__ == "__main__":
  sys.exit(main())
