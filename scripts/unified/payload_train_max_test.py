#!/usr/bin/env python3
"""payload_train_max_test.py -- U_PAYLOAD_TRAIN_MAX: the training payload's limit instead of the robot table's
train_payload max (go2-depth 2026-11-16, go2-home's single-factor arm A: "payload max 12 kg, the test stays 8";
teacher_env._curriculum).

On the built configurations (no GPU), per robot:
  (1) off (unset, and an empty value): the payload event's heavy_max and the u_payload curriculum term's full_max are
      the table's limit;
  (2) U_PAYLOAD_TRAIN_MAX=12: both 12; every reward term, every other event's parameters, every other curriculum term
      and the payload point's box (pos ranges, corner share) as with it off;
  (3) 0, -1, a value above 3x the table's limit, and a non-number stop the build;
  (4) the play build keeps the table's limit: it has no training curriculum, and the gates fix their own U_PAYLOAD.

Local: PYTHONPATH=$PWD python scripts/unified/payload_train_max_test.py [--robots Go2,Go2W]"""

from __future__ import annotations

import argparse
import os
import sys

SWITCH = "U_PAYLOAD_TRAIN_MAX"


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  os.environ.pop(SWITCH, None)
  sys.path.insert(0, os.getcwd())
  from src.tasks.unified import teacher_env
  from src.tasks.unified.robots import ROBOTS

  ok = True

  def check(good: bool, text: str) -> None:
    nonlocal ok
    ok &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'}  {text}")

  def build(value: str | None, **kw):
    if value is None:
      os.environ.pop(SWITCH, None)
    else:
      os.environ[SWITCH] = value
    try:
      return teacher_env.unified_teacher_env_cfg(robot, **kw)
    finally:
      os.environ.pop(SWITCH, None)

  def fails(value: str) -> bool:
    try:
      build(value)
    except (AssertionError, ValueError):
      return True
    return False

  def limit(cfg):
    return cfg.events["base_payload"].params["heavy_max"], cfg.curriculum["u_payload"].params["full_max"]

  def rest(cfg):
    ev = {k: repr(sorted((v.params or {}).items(), key=repr)) for k, v in cfg.events.items() if k != "base_payload"}
    pay = {k: repr(v) for k, v in cfg.events["base_payload"].params.items() if k != "heavy_max"}
    rw = {k: (v.weight, repr(sorted((v.params or {}).items(), key=repr))) for k, v in cfg.rewards.items()}
    cur = {k: repr(sorted((v.params or {}).items(), key=repr)) for k, v in cfg.curriculum.items() if k != "u_payload"}
    return ev, pay, rw, cur

  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    c0, ce = build(None), build("")
    table = float(c0.events["base_payload"].params["heavy_max"])
    check(limit(c0) == (table, table) and limit(ce) == (table, table),
          f"off (unset / empty): heavy_max and full_max the table's {table:g} kg")
    c1 = build("12")
    check(limit(c1) == (12.0, 12.0), f"{SWITCH}=12: heavy_max and full_max 12 kg (got {limit(c1)})")
    r0, r1 = rest(c0), rest(c1)
    check(r0[0] == r1[0], f"the other {len(r0[0])} events' parameters unchanged")
    check(r0[1] == r1[1], "the payload event's other parameters (the point's box, shares) unchanged")
    check(r0[2] == r1[2], f"the {len(r0[2])} reward terms unchanged")
    check(r0[3] == r1[3], f"the other {len(r0[3])} curriculum terms unchanged")
    check(all(fails(v) for v in ("0", "-1", str(3 * table + 0.5), "twelve")),
          f"0, -1, {3 * table + 0.5:g} and a non-number stop the build")
    cp = build("12", play=True)
    check(cp.events["base_payload"].params["heavy_max"] == table,
          "the play build keeps the table's limit (no training curriculum there; the gates set U_PAYLOAD)")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
