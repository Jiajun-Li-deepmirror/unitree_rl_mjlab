#!/usr/bin/env python3
"""wall_fan_test.py -- U_WALL_FAN: the end wall's level ray fan narrowed to the flight's axis (go2-depth 2026-11-15;
robots._wall_fan, teacher_env._probe_switches). uprobe_s162 (U_KEYS_ADD=endwall,endwallshort) lifted its wheels in
front of the wall a quarter as often, held a stop on a flight 70.7 % of the time (94.1 %) and fell 188 times of 896 on
the 8 kg climbs (44) -- without the fan in the evaluation: in the training's narrow corridors the +-20 deg rays reach
the side walls inside the command's cut on every swerve.

On the built configurations (no GPU), per robot whose recipe takes the key endwall (it has longexit):
  (1) U_KEYS_ADD=endwall,endwallshort without the switch: the fan's 5 rays at -20..20 deg;
  (2) with U_WALL_FAN=5: 3 rays at -5, 0, 5 deg; the fan's height, reach, alignment and every other field, the
      command (the sensor's name, the stop distance, the slow-down), every other sensor, sub-terrain, reward term,
      constraint item and curriculum term as without it; U_WALL_FAN=10: at -10, 0, 10;
  (3) the geometry (printed, checked): in a 0.85 m corridor, the robot centred, the outermost ray's distance to the
      side wall at a yaw of 0 / 10 / 15 / 20 deg against the command's cut (from stop + slow-down on, zero at the
      stop): the 20 deg fan cuts from a yaw of 5-10 deg on and reads zero at 10 deg (the Go2W's 0.85 m); the 5 deg
      fan leaves 15 deg alone;
  (4) the play build and the students' builds on the training terrain carry the narrowed fan; the checks' terrains
      (stairs, flat) have no fan, with the switch as without it;
  (5) a half angle of 1 or 20: the build stops; the switch without the key endwall: the build stops;
  every other robot: the switch stops the build (no key endwall in its recipe; the key itself needs longexit).

Local: PYTHONPATH=$PWD python scripts/unified/wall_fan_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys

SWITCH = "U_WALL_FAN"
KEYS = "U_KEYS_ADD"
WALL_KEYS = "endwall,endwallshort"
CORRIDOR = 0.85   # m, the walled flights' narrowest (terrains.py corridor_width_range)


def side_wall(yaw_deg: float, half_deg: float, width: float = CORRIDOR) -> float:
  """The outermost ray's distance to the side wall it turns to: the robot on the corridor's axis, yawed."""
  a = math.radians(abs(yaw_deg) + half_deg)
  return math.inf if a <= 0.0 else 0.5 * width / math.sin(a)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W,As2")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  for k in (SWITCH, KEYS):
    os.environ.pop(k, None)
  sys.path.insert(0, os.getcwd())
  from src.tasks.unified import recipe as urecipe
  from src.tasks.unified import robots as urobots
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
      cfg = teacher_env.unified_teacher_env_cfg(robot, **kw)
      with urecipe.scope(robot):
        items = teacher_env.constraint_items(robot.kind)
    finally:
      for k in env or {}:
        os.environ.pop(k)
    return cfg, items

  def fails(env: dict[str, str], robot) -> bool:
    try:
      build(robot, env)
    except AssertionError:
      return True
    return False

  def fan(cfg):
    return next((s for s in cfg.scene.sensors if s.name == urobots.WALL_FAN), None)

  def angles(cfg):
    f = fan(cfg)
    return None if f is None else tuple(f.pattern.angles_deg)

  def fan_rest(cfg):
    f = fan(cfg)
    rest = {x.name: repr(getattr(f, x.name)) for x in dataclasses.fields(f) if x.name != "pattern"}
    rest.update({f"pattern.{x.name}": repr(getattr(f.pattern, x.name)) for x in dataclasses.fields(f.pattern)
                 if x.name != "angles_deg"})
    return rest

  def twist(cfg):
    t = cfg.commands["twist"]
    return {f.name: repr(getattr(t, f.name)) for f in dataclasses.fields(t)}

  def terms(group):
    return {k: (getattr(v, "weight", None), repr(sorted((v.params or {}).items(), key=repr))) for k, v in group.items()}

  def subs(cfg):
    gen = cfg.scene.terrain.terrain_generator
    return {k: repr(v) for k, v in (gen.sub_terrains if gen is not None else {}).items()}

  def sensors(cfg):
    return [repr(s) for s in cfg.scene.sensors if s.name != urobots.WALL_FAN]

  def same(c, i, c0, i0) -> bool:
    return (fan_rest(c) == fan_rest(c0) and twist(c) == twist(c0) and sensors(c) == sensors(c0)
            and subs(c) == subs(c0) and terms(c.rewards) == terms(c0.rewards) and i == i0
            and terms(c.curriculum) == terms(c0.curriculum))

  for n in names:
    robot = ROBOTS[n]
    have = urobots.unified_recipe(robot)["P"]
    print(f"{n} ({robot.kind}):")
    if "longexit" not in have:
      check(fails({SWITCH: "5"}, robot) and fails({SWITCH: "5", KEYS: "endwall"}, robot),
            f"no longexit in the recipe, so no key endwall: {SWITCH} stops the build")
      continue
    wall = {KEYS: WALL_KEYS}
    cfg0, items0 = build(robot, wall)
    check(angles(cfg0) == (-20.0, -10.0, 0.0, 10.0, 20.0), f"{KEYS}={WALL_KEYS} without {SWITCH}: the fan at {angles(cfg0)} deg")
    cfg1, items1 = build(robot, {**wall, SWITCH: "5"})
    check(angles(cfg1) == (-5.0, 0.0, 5.0), f"{SWITCH}=5: the fan at {angles(cfg1)} deg")
    t = cfg1.commands["twist"]
    check(same(cfg1, items1, cfg0, items0),
          f"the fan's other {len(fan_rest(cfg0))} fields (height {fan(cfg1).pattern.z_offset} m, reach "
          f"{fan(cfg1).max_distance} m), the command (stop {t.end_wall_stop_m} m, slow-down {t.end_wall_slow_m} m), "
          f"the other {len(sensors(cfg0))} sensors, {len(subs(cfg0))} sub-terrains, {len(cfg0.rewards)} reward terms, "
          f"{len(items0)} items and {len(cfg0.curriculum)} curriculum terms unchanged")
    cfg2, items2 = build(robot, {**wall, SWITCH: "10"})
    check(angles(cfg2) == (-10.0, 0.0, 10.0) and same(cfg2, items2, cfg0, items0), f"{SWITCH}=10: the fan at {angles(cfg2)} deg")
    stop, slow = t.end_wall_stop_m, t.end_wall_slow_m
    scale = lambda d: min(1.0, max(0.0, (d - stop) / slow))   # velocity_command._wall_slowdown
    rows = {h: [(y, side_wall(y, h)) for y in (0.0, 10.0, 15.0, 20.0)] for h in (20.0, 5.0)}
    for h, row in rows.items():
      print(f"        {CORRIDOR} m corridor, the +-{h:.0f} deg fan (stop {stop} m, full speed from {stop + slow:.2f} m): "
            + "; ".join(f"yaw {y:.0f} deg {d:.2f} m x{scale(d):.2f}" for y, d in row))
    check(scale(rows[20.0][0][1]) == 1.0 and scale(rows[20.0][1][1]) < 0.1 and scale(rows[20.0][2][1]) == 0.0,
          "the 20 deg fan: full speed straight, the command cut to nothing at a yaw of 10 and 15 deg")
    check(all(scale(d) == 1.0 for y, d in rows[5.0] if y <= 15.0) and scale(rows[5.0][3][1]) > 0.5,
          "the 5 deg fan: full speed to a yaw of 15 deg, more than half at 20 deg")
    cfgp, _ = build(robot, {**wall, SWITCH: "5"}, play=True)
    check(angles(cfgp) == (-5.0, 0.0, 5.0), "the play build carries the narrowed fan")
    for kind in ("blind", "depth"):
      cfgs, _ = build(robot, {**wall, SWITCH: "5"}, student=kind)
      check(angles(cfgs) == (-5.0, 0.0, 5.0), f"the {kind} student's build carries it")
    for terrain in ("stairs", "flat"):
      c0, _ = build(robot, wall, play=True, terrain=terrain)
      c1, _ = build(robot, {**wall, SWITCH: "5"}, play=True, terrain=terrain)
      check(fan(c0) is None and fan(c1) is None and [repr(s) for s in c1.scene.sensors] == [repr(s) for s in c0.scene.sensors]
            and twist(c1) == twist(c0), f"the {terrain} check's build: no fan, the sensors and the command as without it")
    check(fails({**wall, SWITCH: "1"}, robot) and fails({**wall, SWITCH: "20"}, robot),
          "a half angle of 1 or 20 deg: the build stops")
    check(fails({SWITCH: "5"}, robot), "the switch without the key endwall: the build stops")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
