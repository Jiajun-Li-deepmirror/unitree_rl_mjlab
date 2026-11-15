#!/usr/bin/env python3
"""student_floor_test.py -- U_STUDENT_FLOOR / U_STUDENT_LINE_FLOOR: the students' floors on the teacher's frozen
constraint multipliers (go2-depth 2026-11-15; src/tasks/unified/students.py item_floor, line_floor,
restore_teacher_curriculum). The Go2 formal blind student failed its cap gate on four feet down after a stop on a
descent with that item's multiplier frozen at the teacher's 1.15.

No simulator, no GPU: a made-up constraints object holding the multipliers.
  (1) neither switch: the multipliers as loaded;
  (2) U_STUDENT_FLOOR=stop_feet_down:3: that item 1.15 -> 3, every other as loaded;
  (3) two items, one of them already above its floor: the lower one raised, the higher one kept;
  (4) with U_STUDENT_LINE_FLOOR=2 too: the line items at 2 or above, an item both name at the larger floor;
  (5) an item the teacher has no multiplier of, a floor of 0.2 or 8, a name without a floor, a name twice: stops;
  (6) restore_teacher_curriculum on a checkpoint file: loads the teacher's multipliers, then applies both floors;
      a checkpoint without the unified state, or no checkpoint: nothing loaded, nothing raised.

Local: PYTHONPATH=$PWD python scripts/unified/student_floor_test.py"""

from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

SWITCHES = ("U_STUDENT_FLOOR", "U_STUDENT_LINE_FLOOR")
TEACHER = {"stop_feet_down": 1.153, "stop_feet_up": 0.5, "straight_drift": 0.5, "side_drift": 2.4, "creep": 3.0,
           "kicks": 1.0}


class Cons:
  def __init__(self, scale=None):
    self.scale = dict(scale or {})
    self.loaded = 0

  def load(self, state: dict) -> None:
    self.scale = dict(state["scale"])
    self.loaded += 1


def main() -> int:
  for s in SWITCHES:
    os.environ.pop(s, None)
  sys.path.insert(0, os.getcwd())
  import torch

  from src.tasks.unified import students

  ok = True

  def check(good: bool, text: str) -> None:
    nonlocal ok
    ok &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'}  {text}")

  def run(env: dict[str, str], scale=None) -> dict[str, float]:
    cons = Cons(TEACHER if scale is None else scale)
    for k, v in env.items():
      os.environ[k] = v
    try:
      students.line_floor(cons)
      students.item_floor(cons)
    finally:
      for k in env:
        os.environ.pop(k)
    return cons.scale

  def stops(value: str) -> bool:
    try:
      run({"U_STUDENT_FLOOR": value})
    except (AssertionError, ValueError):
      return True
    return False

  check(run({}) == TEACHER, "neither switch: the multipliers as loaded")
  got = run({"U_STUDENT_FLOOR": "stop_feet_down:3"})
  check(got == {**TEACHER, "stop_feet_down": 3.0}, "U_STUDENT_FLOOR=stop_feet_down:3: 1.15 -> 3, the others as loaded")
  got = run({"U_STUDENT_FLOOR": "stop_feet_up:2, creep:2"})
  check(got == {**TEACHER, "stop_feet_up": 2.0}, "stop_feet_up:2, creep:2: 0.5 -> 2, creep stays at its 3")
  got = run({"U_STUDENT_LINE_FLOOR": "2", "U_STUDENT_FLOOR": "straight_drift:3,stop_feet_down:2"})
  check(got == {**TEACHER, "straight_drift": 3.0, "side_drift": 2.4, "stop_feet_down": 2.0},
        "with U_STUDENT_LINE_FLOOR=2: straight_drift at the larger floor 3, side_drift stays at 2.4, stop_feet_down 2")
  got = run({"U_STUDENT_LINE_FLOOR": "2"})
  check(got == {**TEACHER, "straight_drift": 2.0}, "U_STUDENT_LINE_FLOOR=2 alone as before: straight_drift 0.5 -> 2")
  check(all(stops(v) for v in ("stop_feet:3", "stop_feet_down:0.2", "stop_feet_down:8", "stop_feet_down",
                               "stop_feet_down:", ":3", "stop_feet_down:3,stop_feet_down:2", "stop_feet_down:x")),
        "an unknown item, a floor of 0.2 or 8, no floor, no name, a name twice, a floor that is no number: stops")

  with tempfile.TemporaryDirectory() as d:
    full, bare = os.path.join(d, "teacher.pt"), os.path.join(d, "bare.pt")
    torch.save({"infos": {"unified": {"constraints": {"scale": dict(TEACHER)}}}}, full)
    torch.save({"infos": {}}, bare)

    def restore(path: str, env: dict[str, str]) -> Cons:
      cons = Cons({"stop_feet_down": 1.0})
      fake = SimpleNamespace(unwrapped=SimpleNamespace(u_constraints=cons))
      for k, v in env.items():
        os.environ[k] = v
      try:
        students.restore_teacher_curriculum(fake, path, "cpu")
      finally:
        for k in env:
          os.environ.pop(k)
      return cons

    both = {"U_STUDENT_LINE_FLOOR": "2", "U_STUDENT_FLOOR": "stop_feet_down:3"}
    cons = restore(full, both)
    check(cons.loaded == 1 and cons.scale == {**TEACHER, "straight_drift": 2.0, "stop_feet_down": 3.0},
          "restore_teacher_curriculum: the teacher's multipliers loaded, then both floors")
    cons = restore(full, {})
    check(cons.loaded == 1 and cons.scale == TEACHER, "... without the switches: the teacher's multipliers as they are")
    cons = restore(bare, both)
    check(cons.loaded == 0 and cons.scale == {"stop_feet_down": 1.0},
          "a checkpoint without the unified state: nothing loaded, nothing raised")
    cons = restore("", both)
    check(cons.loaded == 0 and cons.scale == {"stop_feet_down": 1.0}, "no checkpoint: nothing loaded, nothing raised")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
