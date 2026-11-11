#!/usr/bin/env python3
"""teacher_gate.py RUN_DIR [--hold N] [--json] -- has the unified teacher of RUN_DIR passed (go2-depth 2026-11-11)?

The teacher is judged on its constraints in training (src/tasks/unified/constraints.py: every acceptance item as a
measurement, a target and a multiplier, logged as Curriculum/u_constraints/<name>/{value,target,scale}); the
acceptance proper is the students' (the user's redesign: "验收要以学生为准，老师需要留出余量"). It passes at the first
checkpoint K (a multiple of 100) where, over the last `hold` iterations (default 500):
  - the curriculum is through (Curriculum/u_progress/p = 1 throughout), and
  - every enforced constraint's value is on its target's side (mean of the window; "min" items >= target, "max"
    items <= target), each with at least half the window measured.
Prints one line: "PASS K <items>" or "NOT YET <iteration> <the items that miss, value vs target>"; --json the details.
Which constraints are enforced and their senses come from the teacher task's items (registered with UNIFIED_TASKS=1).
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
  items = items_for(a.run)
  vals = {c["name"]: series(acc, f"Curriculum/u_constraints/{c['name']}/value") for c in items}
  tgts = {c["name"]: series(acc, f"Curriculum/u_constraints/{c['name']}/target") for c in items}
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
    miss = []
    for c in items:
      n = c["name"]
      window = [i for i in range(lo, k + 1) if i in vals[n] and i in tgts[n]]
      if len(window) < a.hold // 2:
        miss.append(f"{n} unmeasured")
        continue
      v = sum(vals[n][i] for i in window) / len(window)
      t = sum(tgts[n][i] for i in window) / len(window)
      ok = v >= t if c["sense"] == "min" else v <= t
      report[n] = {"value": round(v, 4), "target": round(t, 4), "ok": ok}
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
    print(f"NOT YET {last} (progress {p[last]:.3f})")
  if a.json:
    print(json.dumps(report))
  return 0


if __name__ == "__main__":
  sys.exit(main())
