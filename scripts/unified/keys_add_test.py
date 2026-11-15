#!/usr/bin/env python3
"""keys_add_test.py -- U_KEYS_ADD: keys of the old flow added to the recipe's final keys (go2-depth 2026-11-15;
robots.base_env_cfg, robots.ADD_KEYS). The user on the Go2W teacher's videos: "在平台的时候会出现前面是墙然后做抬腿翻越
动作" -- the unified recipe has neither the end wall (endwall, endwallshort) nor the cost on a wheel lifted with flat
ground ahead (topflat, topflat30).

On the built configurations (no GPU), per robot:
  (1) without the switch (no recipe carries it): none of the four keys in the recipe; the walled up-flights have no end
      wall, no wall sensor, the command reads no wall, no wheel_lift_flat_ahead cost;
  a wheeled robot (its recipe has longexit):
  (2) U_KEYS_ADD=endwall: a 2 m wall at the walled up-flights' end, the sensor wall_fan (5 level rays at -20..20 deg,
      0.8 m above the base, yaw-aligned, 3 m), the command on it: the stop at the robot's wall distance, the slow-down
      over 0.5 m; U_KEYS_ADD=endwall,endwallshort: over 0.2 m. Every other sub-terrain, command field, reward term,
      constraint item and curriculum term as without it;
  (3) U_KEYS_ADD=topflat: wheel_lift_flat_ahead at -8 on wheel_air_drivable's sensor and command; topflat,topflat30:
      at -30; the terrain, the command, every other reward term, the items and the curriculum (but the cost's own
      entry among the stairs quality's weights) as without it; the stairs quality brings the cost in (quality_diff:
      Q_EPS of its weight -> its weight);
  (4) both pairs together: both;
  (5) the check terrains' builds (stairs, flat; play) with the end wall's keys: as without them -- no walled flight
      there, the gates measure the same task; the students' builds on the training terrain carry the wall;
  every robot:
  (6) a key outside the four, a key twice, endwallshort without endwall, topflat30 without topflat: the build stops;
      a legged robot: topflat stops the build (a wheel cost), endwall too where the recipe has no longexit (the wall
      would stand 0.5 m past the last riser, inside the command's stop distance).

Local: PYTHONPATH=$PWD python scripts/unified/keys_add_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

SWITCH = "U_KEYS_ADD"
WALL = ("wall_sensor_name", "end_wall_stop_m", "end_wall_slow_m")
LIFT = "wheel_lift_flat_ahead"
QUALITY = ("u_quality",)   # the stairs quality's curriculum term


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W,As2")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  os.environ.pop(SWITCH, None)
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

  def build(robot, value: str | None = None, **kw):
    if value is not None:
      os.environ[SWITCH] = value
    try:
      cfg = teacher_env.unified_teacher_env_cfg(robot, **kw)
      with urecipe.scope(robot):
        items = teacher_env.constraint_items(robot.kind)
    finally:
      os.environ.pop(SWITCH, None)
    return cfg, items

  def fails(value: str, robot) -> bool:
    try:
      build(robot, value)
    except AssertionError:
      return True
    return False

  def twist(cfg, skip=()):
    t = cfg.commands["twist"]
    return {f.name: repr(getattr(t, f.name)) for f in dataclasses.fields(t) if f.name not in skip}

  def terms(group, skip=()):
    return {k: (getattr(v, "weight", None), repr(sorted((v.params or {}).items(), key=repr)))
            for k, v in group.items() if k not in skip}

  def subs(cfg, skip=()):
    gen = cfg.scene.terrain.terrain_generator
    return {k: repr(v) for k, v in (gen.sub_terrains if gen is not None else {}).items() if k not in skip}

  def sensors(cfg, skip=()):
    return [repr(s) for s in cfg.scene.sensors if s.name not in skip]

  def quality(cfg, skip=()):
    """The stairs quality's curriculum term without the weights of the costs named (it lists every cost it brings
    in: the lifted wheel's among them with the key)."""
    q = cfg.curriculum[QUALITY[0]]
    params = {**q.params, "weights": {k: v for k, v in q.params["weights"].items() if k not in skip}}
    return repr(sorted(params.items(), key=repr))

  def wall(cfg):
    gen = cfg.scene.terrain.terrain_generator
    up = (gen.sub_terrains if gen is not None else {}).get("walled_up")
    fan = [s for s in cfg.scene.sensors if s.name == "wall_fan"]
    t = cfg.commands["twist"]
    return getattr(up, "end_wall_height", None), fan, (t.wall_sensor_name, t.end_wall_stop_m, t.end_wall_slow_m)

  def walled(cfg, robot, slow: float) -> bool:
    height, fan, cmd = wall(cfg)
    return (height == 2.0 and len(fan) == 1 and tuple(fan[0].pattern.angles_deg) == (-20.0, -10.0, 0.0, 10.0, 20.0)
            and fan[0].pattern.z_offset == 0.8 and fan[0].ray_alignment == "yaw" and fan[0].max_distance == 3.0
            and cmd == ("wall_fan", robot.wall_stop_m, slow))

  def same(cfg, items, cfg0, items0, but=()) -> bool:
    """Everything as without the switch, but the parts named: "wall" (the end wall's sub-terrain, sensor and command
    fields), "lift" (the lifted wheel's cost and its entry among the stairs quality's weights)."""
    w, lift = "wall" in but, "lift" in but
    return (subs(cfg, ("walled_up",) if w else ()) == subs(cfg0, ("walled_up",) if w else ())
            and sensors(cfg, ("wall_fan",) if w else ()) == sensors(cfg0)
            and twist(cfg, WALL if w else ()) == twist(cfg0, WALL if w else ())
            and terms(cfg.rewards, (LIFT,) if lift else ()) == terms(cfg0.rewards)
            and items == items0 and terms(cfg.curriculum, QUALITY if lift else ()) == terms(cfg0.curriculum,
                                                                                         QUALITY if lift else ())
            and (not lift or quality(cfg, (LIFT,)) == quality(cfg0)))

  for n in names:
    robot = ROBOTS[n]
    have = urobots.unified_recipe(robot)["P"]
    print(f"{n} ({robot.kind}):")
    cfg0, items0 = build(robot)
    height, fan, cmd = wall(cfg0)
    check(not set(have) & set(urobots.ADD_KEYS) and not height and not fan and cmd[0] is None
          and LIFT not in cfg0.rewards,
          f"without {SWITCH}: none of {', '.join(urobots.ADD_KEYS)} among the {len(have)} keys; no end wall "
          f"({height} m), no wall sensor, the command reads none, no {LIFT}")
    check(all(fails(v, robot) for v in ("landings", "lowent", "endwall,endwall", "endwallshort", "topflat30",
                                        "endwall,nowall")),
          "a key outside the four, a key twice, endwallshort without endwall, topflat30 without topflat: the build "
          "stops")
    if robot.kind != "wheeled":
      no_exit = "longexit" not in have
      check(fails("topflat", robot) and fails("topflat,topflat30", robot), "topflat: a wheel cost -- the build stops")
      check(fails("endwall", robot) == no_exit,
            f"endwall: the build stops (no longexit in the recipe)" if no_exit else "endwall: builds (longexit there)")
      continue
    check("longexit" in have, "the recipe has longexit (the wall stands 1.4 m past the last riser)")
    cfg1, items1 = build(robot, "endwall")
    check(walled(cfg1, robot, 0.5),
          f"{SWITCH}=endwall: a 2 m wall at the walled up-flights' end, wall_fan (5 rays, 0.8 m up, yaw, 3 m), the "
          f"command stops {robot.wall_stop_m} m before it and slows over 0.5 m")
    cfg2, items2 = build(robot, "endwall,endwallshort")
    check(walled(cfg2, robot, 0.2), f"{SWITCH}=endwall,endwallshort: the same, the slow-down over 0.2 m")
    check(same(cfg1, items1, cfg0, items0, ("wall",)) and same(cfg2, items2, cfg0, items0, ("wall",)),
          f"with them the other {len(subs(cfg0)) - 1} sub-terrains, {len(cfg0.scene.sensors)} sensors, the command's "
          f"other fields, the {len(cfg0.rewards)} reward terms, {len(items0)} items and {len(cfg0.curriculum)} "
          "curriculum terms unchanged")
    cfg3, items3 = build(robot, "topflat")
    cfg4, items4 = build(robot, "topflat,topflat30")
    ref = cfg0.rewards["wheel_air_drivable"].params
    r3, r4 = cfg3.rewards.get(LIFT), cfg4.rewards.get(LIFT)
    check(r3 is not None and r4 is not None and r3.weight == -8.0 and r4.weight == -30.0
          and all(r.params["sensor_name"] == ref["sensor_name"] and r.params["command_name"] == ref["command_name"]
                  for r in (r3, r4)),
          f"{SWITCH}=topflat: {LIFT} at -8 on {ref['sensor_name']}, {ref['command_name']}; topflat,topflat30: at -30")
    check(same(cfg3, items3, cfg0, items0, ("lift",)) and same(cfg4, items4, cfg0, items0, ("lift",)),
          "with them the terrain, the sensors, the command, the other reward terms, the items and the curriculum "
          "(but the cost's entry among the stairs quality's weights) unchanged")
    os.environ[SWITCH] = "topflat,topflat30"   # (quality_diff builds the M keys' environment: the switch as in training)
    try:
      with urecipe.scope(robot):
        weights, _, _ = teacher_env.quality_diff(robot, cfg4.rewards, set(), set())
    finally:
      os.environ.pop(SWITCH)
    check(weights.get(LIFT) == (teacher_env.Q_EPS * -30.0, -30.0),
          f"the stairs quality brings it in: {weights.get(LIFT)} (Q_EPS of its weight -> its weight; the M keys' "
          "environment is without it)")
    cfg5, items5 = build(robot, "endwall,endwallshort,topflat,topflat30")
    check(walled(cfg5, robot, 0.2) and cfg5.rewards[LIFT].weight == -30.0
          and same(cfg5, items5, cfg0, items0, ("wall", "lift")), "both pairs together: both, the rest unchanged")
    for terrain in ("stairs", "flat"):
      cfgc0, itemsc0 = build(robot, play=True, terrain=terrain)
      cfgc, itemsc = build(robot, "endwall,endwallshort", play=True, terrain=terrain)
      check(same(cfgc, itemsc, cfgc0, itemsc0),
            f"the check terrain {terrain} (play) with endwall,endwallshort: as without them")
    for kind in ("blind", "depth"):
      cfgs, _ = build(robot, "endwall,endwallshort", student=kind)
      check(walled(cfgs, robot, 0.2), f"the {kind} student's build carries the wall")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
