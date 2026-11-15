#!/usr/bin/env python3
"""cmd_expose_test.py -- U_STEER / U_FAST_SIDE: the training robots' exposure to the flat commands the gait clock fails
(go2-depth 2026-11-15; teacher_env._probe_switches). The Go2W formal's flat gate by hand, seed 1 4500: the clock alone
is left, on the fast gentle arcs (1.0, 0, +-0.1..0.2) and the strafes (0, +-0.6, 0), with its multiplier at the cap.

On the built configurations (no GPU), per robot:
  (1) with the switches off (U_STEER= U_FAST_SIDE= in the environment: an empty value wins over the recipe's; every
      build below starts from it): the command's steering share, |wz| range, slow share and fast-side share as the old
      flow's keys set them (printed); the plain build: the recipe's own fast-side share where the robot's recipe
      carries U_FAST_SIDE (the wheeled kind's, 0.1, since 2026-11-15), the old flow's where it does not; no recipe
      carries U_STEER;
  (2) U_STEER=0.3: the steering share 0.3, its |wz| range and slow share as before; U_STEER=0.3,0.25,0.1: the largest
      |wz| 0.25 (the smallest kept) and the slow share 0.1; U_FAST_SIDE=0.1: the fast-side share 0.1; both together:
      both;
  (3) every other field of the command, every reward term, constraint item and curriculum term as without them;
  (4) a share of 0 or 0.6, a |wz| below 0.1 or above 0.6, a slow share of 0.6, a fourth field, a fast-side share of 0
      or 0.5: the build stops;
  (5) the training alone: the play build keeps the command as it was; the students' builds carry it (they train on
      the teacher's commands).

Local: PYTHONPATH=$PWD python scripts/unified/cmd_expose_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

SWITCHES = ("U_STEER", "U_FAST_SIDE")
FIELDS = ("steer_prob", "steer_wz_range", "steer_slow_prob", "fast_side_prob")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W,As2")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  for s in SWITCHES:
    os.environ.pop(s, None)
  sys.path.insert(0, os.getcwd())
  from src.tasks.unified import recipe as urecipe
  from src.tasks.unified import teacher_env
  from src.tasks.unified.robots import ROBOTS

  ok = True

  def check(good: bool, text: str) -> None:
    nonlocal ok
    ok &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'}  {text}")

  def build(robot, env: dict[str, str] | None = None, plain: bool = False, **kw):
    env = dict(env or {}) if plain else {**{s: "" for s in SWITCHES}, **(env or {})}   # off unless set (or plain)
    for k, v in env.items():
      os.environ[k] = v
    try:
      cfg = teacher_env.unified_teacher_env_cfg(robot, **kw)
      with urecipe.scope(robot):
        items = teacher_env.constraint_items(robot.kind)
    finally:
      for k in env:
        os.environ.pop(k)
    return cfg, items

  def fails(env: dict[str, str], robot) -> bool:
    try:
      build(robot, env)
    except AssertionError:
      return True
    return False

  def twist(cfg, skip=()):
    t = cfg.commands["twist"]
    return {f.name: repr(getattr(t, f.name)) for f in dataclasses.fields(t) if f.name not in skip}

  def terms(group):
    return {k: (getattr(v, "weight", None), repr(sorted((v.params or {}).items(), key=repr))) for k, v in group.items()}

  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    cfg0, items0 = build(robot)
    t0 = cfg0.commands["twist"]
    check(all(hasattr(t0, f) for f in FIELDS),
          f"the switches off: steering {t0.steer_prob} of the flat commands, |wz| {t0.steer_wz_range}, "
          f"slow {t0.steer_slow_prob}; fast side {t0.fast_side_prob}")
    cfgr, itemsr = build(robot, plain=True)
    tr = cfgr.commands["twist"]
    table = urecipe.recipe(robot)
    own = table.get("U_FAST_SIDE")
    check(not table.get("U_STEER") and tr.fast_side_prob == (float(own) if own else t0.fast_side_prob)
          and all(getattr(tr, f) == getattr(t0, f) for f in FIELDS[:3])
          and twist(cfgr, skip=FIELDS) == twist(cfg0, skip=FIELDS) and itemsr == items0,
          f"the plain build: the recipe's own U_FAST_SIDE={own} -> fast side {tr.fast_side_prob}, no U_STEER; "
          "the rest as with the switches off" if own else
          f"the plain build: the recipe carries neither switch -> fast side {tr.fast_side_prob}, as with them off")
    cfg1, items1 = build(robot, {"U_STEER": "0.3"})
    t1 = cfg1.commands["twist"]
    check(t1.steer_prob == 0.3 and t1.steer_wz_range == t0.steer_wz_range and t1.steer_slow_prob == t0.steer_slow_prob
          and t1.fast_side_prob == t0.fast_side_prob, "U_STEER=0.3: the steering share 0.3, the rest as before")
    cfg2, items2 = build(robot, {"U_STEER": "0.3,0.25,0.1"})
    t2 = cfg2.commands["twist"]
    check(t2.steer_prob == 0.3 and t2.steer_wz_range == (t0.steer_wz_range[0], 0.25) and t2.steer_slow_prob == 0.1
          and t2.fast_side_prob == t0.fast_side_prob,
          f"U_STEER=0.3,0.25,0.1: |wz| {t2.steer_wz_range} (the smallest kept), slow 0.1")
    cfg3, items3 = build(robot, {"U_FAST_SIDE": "0.1"})
    t3 = cfg3.commands["twist"]
    check(t3.fast_side_prob == 0.1 and all(getattr(t3, f) == getattr(t0, f) for f in FIELDS[:3]),
          "U_FAST_SIDE=0.1: the fast-side share 0.1, the steering as before")
    cfg4, items4 = build(robot, {"U_STEER": "0.3,0.25,0.1", "U_FAST_SIDE": "0.1"})
    t4 = cfg4.commands["twist"]
    check(t4.steer_prob == 0.3 and t4.steer_wz_range[1] == 0.25 and t4.steer_slow_prob == 0.1
          and t4.fast_side_prob == 0.1, "both together: both")
    same = all(twist(c, skip=FIELDS) == twist(cfg0, skip=FIELDS) for c in (cfg1, cfg2, cfg3, cfg4))
    check(same, f"the command's other {len(twist(cfg0, skip=FIELDS))} fields unchanged")
    check(all(terms(c.rewards) == terms(cfg0.rewards) for c in (cfg1, cfg2, cfg3, cfg4)),
          f"the {len(cfg0.rewards)} reward terms unchanged")
    check(all(i == items0 for i in (items1, items2, items3, items4)), f"the {len(items0)} constraint items unchanged")
    check(all(terms(c.curriculum) == terms(cfg0.curriculum) for c in (cfg1, cfg2, cfg3, cfg4)),
          f"the {len(cfg0.curriculum)} curriculum terms unchanged")
    bad = [{"U_STEER": v} for v in ("0", "0.6", "0.3,0.04", "0.3,0.7", "0.3,0.25,0.6", "0.3,0.25,0.1,1")]
    bad += [{"U_FAST_SIDE": v} for v in ("0", "0.5")]
    check(all(fails(env, robot) for env in bad),
          "a share of 0 or 0.6, |wz| 0.04 or 0.7, a slow share of 0.6, a fourth field, a fast side of 0 or 0.5: "
          "the build stops")
    both = {"U_STEER": "0.3,0.25,0.1", "U_FAST_SIDE": "0.1"}
    cfgp0, _ = build(robot, play=True)
    cfgp, _ = build(robot, both, play=True)
    cfgpr, _ = build(robot, plain=True, play=True)
    check(twist(cfgp) == twist(cfgp0) and twist(cfgpr) == twist(cfgp0),
          "the play build keeps the command as it was (with the switches set, and with the recipe's own)")
    for kind in ("blind", "depth"):
      cfgs, _ = build(robot, both, student=kind)
      ts = cfgs.commands["twist"]
      check(ts.steer_prob == 0.3 and ts.steer_wz_range[1] == 0.25 and ts.steer_slow_prob == 0.1
            and ts.fast_side_prob == 0.1, f"the {kind} student's build carries both")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
