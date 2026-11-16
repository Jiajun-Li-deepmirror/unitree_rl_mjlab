#!/usr/bin/env python3
"""recipe_min_test.py -- U_RECIPE_MIN=1, the minimal Go2W teacher recipe (go2-depth 2026-11-16, go2-home's spec after
the user's "如果都从选配方开始了，为什么还要跑A-D"; teacher_env._min_recipe, MIN_TERMS, min_default).

On the built configurations (no GPU), per robot:
  (1) off (unset / 0): the plain build;
  (2) on: the reward terms are MIN_TERMS at today's weights (equal to the plain build's), the logging terms (the
      meters, weight <= 1e-6) and the logging sources at 1e-9 (printed); nothing else; u_quality gone; u_constraints
      without items; no reward ramp of a deleted term; every other curriculum term as in the plain build except
      u_payload (fixed) and ramps (no march ramp);
  (3) the training conditions: the payload limit 12 kg, march 0.5, the emergency stop 0.75 / fast strafes 0.3, the
      payload fixed from iteration 0; U_PAYLOAD_TRAIN_MAX= (empty) and U_HARD_FIXED= (empty) turn them off;
  (4) the commands (but the D fields), the events (but the payload's limit), the observations and the terminations
      as in the plain build;
  (5) a bad value stops the build; the play build is the plain play build;
  (6) a robot not in MIN_ROBOTS (the Go2: train.py registers every robot's tasks) keeps its plain build.
Prints the minimal recipe: terms, weights, curricula.

Local: PYTHONPATH=$PWD python scripts/unified/recipe_min_test.py [--robots Go2W,Go2]"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

SWITCHES = ("U_RECIPE_MIN", "U_PAYLOAD_TRAIN_MAX", "U_HARD_FIXED", "U_MARCH_FIXED", "U_FAST_STOP", "U_PAYLOAD_FULL0")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2W,Go2")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  for s in SWITCHES:
    os.environ.pop(s, None)
  sys.path.insert(0, os.getcwd())
  from src.tasks.unified import teacher_env
  from src.tasks.unified.robots import ROBOTS

  ok = True

  def check(good: bool, text: str) -> None:
    nonlocal ok
    ok &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'}  {text}")

  def build(robot, env: dict[str, str] | None = None, **kw):
    for k, v in (env or {}).items():
      os.environ[k] = v
    try:
      return teacher_env.unified_teacher_env_cfg(robot, **kw)
    finally:
      for k in env or {}:
        os.environ.pop(k)

  def fails(robot, env) -> bool:
    try:
      build(robot, env)
    except (AssertionError, ValueError):
      return True
    return False

  rep = lambda x: repr(sorted((x or {}).items(), key=repr))
  twist = lambda c: {f.name: repr(getattr(c.commands["twist"], f.name)) for f in dataclasses.fields(c.commands["twist"])}
  events = lambda c: {k: rep(v.params) for k, v in c.events.items()}
  obs = lambda c: {g: repr(sorted(getattr(c.observations[g], "terms", {}).keys())) for g in c.observations}
  terms = lambda c: {k: repr(v.func) + rep(v.params) for k, v in c.terminations.items()}

  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    c0 = build(robot)
    if n not in teacher_env.MIN_ROBOTS:   # (train.py registers every robot: a robot not in MIN_ROBOTS stays as it is)
      c1 = build(robot, {"U_RECIPE_MIN": "1"})
      check(set(c1.rewards) == set(c0.rewards) and twist(c1) == twist(c0) and events(c1) == events(c0)
            and set(c1.curriculum) == set(c0.curriculum), f"{n} is not in MIN_ROBOTS: U_RECIPE_MIN=1 leaves its build as it is")
      continue
    check(twist(build(robot, {"U_RECIPE_MIN": "0"})) == twist(c0) and set(build(robot, {"U_RECIPE_MIN": "0"}).rewards) == set(c0.rewards),
          "U_RECIPE_MIN=0: the plain build")
    cm = build(robot, {"U_RECIPE_MIN": "1"})
    mt = teacher_env.MIN_TERMS
    w0 = {k: c0.rewards[k].weight for k in mt}
    check(w0 == mt, f"MIN_TERMS are today's weights in the plain build ({len(mt)} terms)")
    kept = {k: v.weight for k, v in cm.rewards.items()}
    judged = {k: w for k, w in kept.items() if abs(w) > teacher_env.LOG_WEIGHT}
    logs = {k: w for k, w in kept.items() if abs(w) <= teacher_env.LOG_WEIGHT}
    check(judged == mt, f"the weighted terms are exactly MIN_TERMS ({len(judged)})")
    sources = {k for k, w in logs.items() if w == teacher_env.LOG_SOURCE_WEIGHT and abs(c0.rewards[k].weight) > teacher_env.LOG_WEIGHT}
    check(all(k.startswith("u_") or k in sources for k in logs),
          f"the rest: {len(logs) - len(sources)} logging terms and {len(sources)} sources at 1e-9 ({sorted(sources)})")
    gone = sorted(set(c0.rewards) - set(cm.rewards))
    check(len(gone) + len(kept) == len(c0.rewards), f"{len(gone)} terms deleted")
    check("u_quality" not in cm.curriculum and cm.curriculum["u_constraints"].params["items"] == [],
          "u_quality deleted, u_constraints without items")
    ramps = cm.curriculum["ramps"].params["ramps"] if "ramps" in cm.curriculum else []
    check(all(not r.get("kind", "").startswith("reward") or r.get("name") in cm.rewards for r in ramps)
          and not any(r.get("key") == "march_prob" for r in ramps), f"no ramp of a deleted term, no march ramp ({ramps})")
    other = set(c0.curriculum) - {"u_quality", "u_constraints", "u_payload", "ramps"}
    check(all(rep(c0.curriculum[k].params) == rep(cm.curriculum[k].params) for k in other),
          f"the other {len(other)} curriculum terms unchanged ({sorted(other)})")
    # (3)
    t = cm.commands["twist"]
    check(cm.events["base_payload"].params["heavy_max"] == 12.0 and cm.curriculum["u_payload"].params["full_max"] == 12.0
          and cm.curriculum["u_payload"].params.get("fixed") is True, "the payload: 12 kg, fixed from iteration 0")
    check(t.march_prob == 0.5 and t.stop_after_fast_prob == 0.75 and t.fast_side_prob == 0.3,
          "march 0.5, emergency stop 0.75, fast strafes 0.3")
    co = build(robot, {"U_RECIPE_MIN": "1", "U_PAYLOAD_TRAIN_MAX": "", "U_HARD_FIXED": ""})
    check(co.events["base_payload"].params["heavy_max"] == c0.events["base_payload"].params["heavy_max"]
          and co.commands["twist"].stop_after_fast_prob == c0.commands["twist"].stop_after_fast_prob
          and set(co.rewards) == set(cm.rewards), "U_PAYLOAD_TRAIN_MAX= and U_HARD_FIXED= turn the conditions off")
    # (4)
    t0, tm = twist(c0), twist(cm)
    check({k for k in t0 if t0[k] != tm[k]} <= {"march_prob", "stop_after_fast_prob", "fast_side_prob",
                                                "stand_resampling_time_range"},
          "the command unchanged but the D fields")
    e0, em = events(c0), events(cm)
    check({k for k in e0 if e0[k] != em[k]} == {"base_payload"}
          and {k: v for k, v in c0.events["base_payload"].params.items() if k != "heavy_max"}
          == {k: v for k, v in cm.events["base_payload"].params.items() if k != "heavy_max"},
          "the events unchanged but the payload's limit")
    check(obs(c0) == obs(cm) and terms(c0) == terms(cm), "the observations and the terminations unchanged")
    # (5)
    check(fails(robot, {"U_RECIPE_MIN": "2"}), "U_RECIPE_MIN=2 stops the build")
    p0, pm = build(robot, play=True), build(robot, {"U_RECIPE_MIN": "1"}, play=True)
    check(set(p0.rewards) == set(pm.rewards) and twist(p0) == twist(pm) and events(p0) == events(pm),
          "the play build is the plain play build")
    print(f"  the minimal recipe ({n}):")
    for k, w in sorted(judged.items(), key=lambda x: -abs(x[1])):
      print(f"    {k:28s} {w:g}")
    print(f"    logging (<= 1e-8): {sorted(k for k in logs if k not in sources)}")
    print(f"    sources at 1e-9: {sorted(sources)}")
    print(f"    deleted ({len(gone)}): {gone}")
    print(f"    curriculum: {list(cm.curriculum)}")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
