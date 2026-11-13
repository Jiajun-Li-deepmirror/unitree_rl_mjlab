#!/usr/bin/env python3
"""descent_speed_test.py -- U_DESCENT_SPEED: going down a flight faster than told costs (go2-depth 2026-11-14;
constraints.descent_overspeed, teacher_env._add_descent_speed).

(A) The term on a small made-up environment (CPU, no simulator), one robot per case, the command 0.5 m/s ahead:
  told      nose down on uneven ground at 1.0x the command                no cost; read, not over
  edge      ... at 1.1x                                                   no cost; read, not over
  fast      ... at 1.35x                                                  0.25; read, over 1.2x
  runaway   ... at 2.5x                                                   1 (the most); read, over
  climb     nose up at 1.35x                                              no cost, not read
  even      nose down on even ground at 1.35x                             no cost, not read
  side      a sideways command, 1.35x ahead                               no cost, not read
  slope     pitched 30 deg down, 0.675 m/s along the slope                the ground covered counts: 1.17x, 0.07
(B) On the built configurations (no GPU), per robot:
  (1) the plain build: no u_descent_speed term, no descent_over item (no recipe carries the switch);
  (2) with U_DESCENT_SPEED=8: the term at the weight -8, over 1.1x, on stair_speed_deficit's command and scan; one
      item descent_over, logged only (no terms, not enforced), its metric in constraints.CUSTOM;
      U_DESCENT_SPEED=5,1.2 read as written;
  (3) every other reward term and constraint item as without the switch; the play build carries the term too.

Local: PYTHONPATH=$PWD python scripts/unified/descent_speed_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import math
import os
import sys
from types import SimpleNamespace


def term_cases(ucons) -> list[tuple[bool, str]]:
  import torch

  import src.tasks.velocity.mdp.constraints as old_cons

  names = ["told", "edge", "fast", "runaway", "climb", "even", "side", "slope"]
  n = len(names)
  i = {k: j for j, k in enumerate(names)}
  speed = 0.5
  cmd = torch.zeros(n, 3)
  cmd[:, 0] = speed
  cmd[i["side"], 1] = 0.3
  grav = torch.zeros(n, 3)
  grav[:, 0], grav[:, 2] = 0.3, -0.95   # nose down
  grav[i["climb"], 0] = -0.3
  quat = torch.zeros(n, 4)
  quat[:, 0] = 1.0                       # level, heading along x
  a = math.radians(30.0)                 # slope: pitched 30 deg nose down (a rotation about y)
  quat[i["slope"]] = torch.tensor([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0])
  vel = torch.zeros(n, 3)
  for who, x in (("told", 1.0), ("edge", 1.1), ("fast", 1.35), ("runaway", 2.5), ("climb", 1.35), ("even", 1.35),
                 ("side", 1.35)):
    vel[i[who], 0] = x * speed
  vel[i["slope"], 0], vel[i["slope"], 2] = 0.675 * math.cos(a), -0.675 * math.sin(a)   # 0.675 m/s along the slope
  uneven = torch.ones(n, dtype=torch.bool)
  uneven[i["even"]] = False
  data = SimpleNamespace(root_link_quat_w=quat, root_link_lin_vel_w=vel, root_link_lin_vel_b=vel.clone(),
                         projected_gravity_b=grav)
  env = SimpleNamespace(num_envs=n, device="cpu", scene={"robot": SimpleNamespace(data=data)},
                        command_manager=SimpleNamespace(get_command=lambda name: cmd))
  keep = old_cons.uneven_terrain_mask
  old_cons.uneven_terrain_mask = lambda env, sensor_name, *a, **k: uneven.clone()
  out: list[tuple[bool, str]] = []
  try:
    r = ucons.descent_overspeed(env, command_name="twist", sensor_name="scan")
    over, mask = ucons.CUSTOM["u_descent_over"](env)

    def ok(cond, text):
      out.append((bool(cond), text))

    for who in ("told", "edge"):
      ok(r[i[who]] == 0 and mask[i[who]] and over[i[who]] == 0, f"{who}: no cost, read, not over")
    ok(abs(float(r[i["fast"]]) - 0.25) < 1e-5 and mask[i["fast"]] and over[i["fast"]] == 1,
       f"fast: 1.35x costs {float(r[i['fast']]):.3f} (0.25), read, over 1.2x")
    ok(float(r[i["runaway"]]) == 1.0 and over[i["runaway"]] == 1, "runaway: 2.5x costs 1 (the most)")
    for who in ("climb", "even", "side"):
      ok(r[i[who]] == 0 and not mask[i[who]], f"{who}: no cost, not read")
    want = 0.675 * math.cos(a) / speed - 1.1
    ok(abs(float(r[i["slope"]]) - want) < 1e-4 and over[i["slope"]] == 0,
       f"slope: the ground covered counts ({0.675 * math.cos(a) / speed:.2f}x, cost {float(r[i['slope']]):.3f})")
    r2 = ucons.descent_overspeed(env, command_name="twist", sensor_name="scan", over=1.3)
    ok(abs(float(r2[i["fast"]]) - 0.05) < 1e-5 and r2[i["slope"]] == 0, "over 1.3: 1.35x costs 0.05")
  finally:
    old_cons.uneven_terrain_mask = keep
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W,As2")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  os.environ.pop("U_DESCENT_SPEED", None)
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

  print("the term:")
  for good, text in term_cases(ucons):
    check(good, text)

  def build(robot, **kw):
    cfg = teacher_env.unified_teacher_env_cfg(robot, **kw)
    with urecipe.scope(robot):
      items = teacher_env.constraint_items(robot.kind)
    return cfg, items

  def with_switch(value, robot, **kw):
    os.environ["U_DESCENT_SPEED"] = value
    try:
      return build(robot, **kw)
    finally:
      os.environ.pop("U_DESCENT_SPEED")

  term = teacher_env.DESCENT_SPEED_TERM
  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    check("U_DESCENT_SPEED" not in urecipe.recipe(robot), "no recipe carries the switch")
    cfg0, items0 = build(robot)
    check(term not in cfg0.rewards and not [i for i in items0 if i["name"] == "descent_over"],
          "the plain build: no item, no term")
    cfg1, items1 = with_switch("8", robot)
    new = [i for i in items1 if i["name"] == "descent_over"]
    check(len(new) == 1 and not new[0]["enforce"] and not new[0]["terms"] and new[0]["metric"] in ucons.CUSTOM
          and new[0]["sense"] == "max", "with it: one item descent_over, logged only, no terms, the metric known")
    r = cfg1.rewards.get(term)
    ref = cfg1.rewards["stair_speed_deficit"].params
    check(r is not None and r.func is ucons.descent_overspeed and r.weight == -8.0 and r.params["over"] == 1.1
          and r.params["command_name"] == ref["command_name"] and r.params["sensor_name"] == ref["sensor_name"],
          f"the term at -8, over 1.1x, on {ref['command_name']} and {ref['sensor_name']}")
    cfg2, _ = with_switch("5,1.2", robot)
    check(cfg2.rewards[term].weight == -5.0 and cfg2.rewards[term].params["over"] == 1.2,
          "U_DESCENT_SPEED=5,1.2 read as written")
    rest0 = {k: (v.weight, repr(sorted(v.params.items(), key=repr))) for k, v in cfg0.rewards.items()}
    rest1 = {k: (v.weight, repr(sorted(v.params.items(), key=repr))) for k, v in cfg1.rewards.items() if k != term}
    check(rest0 == rest1, f"the other {len(rest0)} reward terms unchanged")
    check([i for i in items1 if i["name"] != "descent_over"] == items0,
          f"the other {len(items0)} constraint items unchanged")
    cfgp, _ = with_switch("8", robot, play=True)
    check(term in cfgp.rewards, "the play build carries the term")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
