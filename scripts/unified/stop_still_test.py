#!/usr/bin/env python3
"""stop_still_test.py -- U_STOP_STILL=1 adds the stop-on-a-flight readings and nothing else (go2-depth 2026-11-13).

On the built configurations (no GPU), per robot:
  (1) without the switch: no stop_still item, no u_stairs_stop term;
  (2) with it: the four items stop_still_{up,down}[_heavy], logged only (enforce False, no terms), their metrics in
      constraints.CUSTOM; the term u_stairs_stop after standing_position_hold (it reads that term's clock);
  (3) every other reward term and constraint item as without the switch.

Local: PYTHONPATH=$PWD python scripts/unified/stop_still_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W,As2")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  os.environ.pop("U_STOP_STILL", None)
  sys.path.insert(0, os.getcwd())
  from src.tasks.unified import constraints as ucons
  from src.tasks.unified import recipe as urecipe
  from src.tasks.unified import teacher_env
  from src.tasks.unified.robots import ROBOTS

  ok = True

  def check(good: bool, text: str) -> None:
    nonlocal ok
    ok &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'}  {text}")

  def build(robot):
    cfg = teacher_env.unified_teacher_env_cfg(robot)
    with urecipe.scope(robot):
      items = teacher_env.constraint_items(robot.kind)
    return cfg, items

  want = [f"stop_still_{d}{h}" for d in ("up", "down") for h in ("", "_heavy")]
  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    cfg0, items0 = build(robot)
    check("u_stairs_stop" not in cfg0.rewards and not [i for i in items0 if i["name"].startswith("stop_still")],
          "without the switch: no item, no term")
    os.environ["U_STOP_STILL"] = "1"
    try:
      cfg1, items1 = build(robot)
    finally:
      os.environ.pop("U_STOP_STILL")
    new = [i for i in items1 if i["name"].startswith("stop_still")]
    check(sorted(i["name"] for i in new) == sorted(want), f"with it: the items {[i['name'] for i in new]}")
    check(all(not i["enforce"] and not i["terms"] and i["metric"] in ucons.CUSTOM and i["sense"] == "min"
              and i["target"] == 0.95 for i in new), "logged only, no terms, the metrics known, bar 0.95 (min)")
    order = list(cfg1.rewards)
    check("u_stairs_stop" in order and order.index("u_stairs_stop") > order.index("standing_position_hold"),
          "the term u_stairs_stop after standing_position_hold")
    p = cfg1.rewards["u_stairs_stop"].params
    check(p["key"] == "u_stairs_stop" and p["settle_s"] + p["window_s"] == 4.0 and p["min_window_s"] < p["window_s"],
          f"its window ends 4 s after the stand began ({p})")
    rest0 = {k: (v.weight, repr(sorted(v.params))) for k, v in cfg0.rewards.items()}
    rest1 = {k: (v.weight, repr(sorted(v.params))) for k, v in cfg1.rewards.items() if k != "u_stairs_stop"}
    check(rest0 == rest1, f"the other {len(rest0)} reward terms unchanged")
    check([i for i in items1 if not i["name"].startswith("stop_still")] == items0,
          f"the other {len(items0)} constraint items unchanged")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
