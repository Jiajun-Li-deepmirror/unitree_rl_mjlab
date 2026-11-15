#!/usr/bin/env python3
"""descent_leg_test.py -- U_DESCENT_LEG: a thigh or calf pressing on a step while the robot walks down a flight costs
(go2-depth 2026-11-15; constraints.descent_leg_contact, teacher_env._add_descent_leg). The user on the Go2 teacher's
videos: "你有记录下楼时后小腿与台阶的碰撞吗", "下楼时，主要关注后腿是否会有碰撞".

(A) The term on a small made-up environment (CPU, no simulator), one robot per case, two leg geoms, 4 substeps in the
    sensor's history, the force threshold 5 N:
  walk      nose down on uneven ground, 0.5 m/s ahead, a calf at 50 N in 2 of the 4 substeps   2; read, pressing
  both      ... both geoms pressing in the same substep                                       1 (a substep once)
  light     ... 4 N in every substep                                                          no cost; read, not pressing
  clean     ... no contact                                                                    no cost; read, not pressing
  stop      told to stop there, a calf at 50 N in every substep                               no cost, not read
  slow      told 0.1 m/s ahead (below the descent item's 0.2)                                 no cost, not read
  climb     nose up, pressing in every substep                                                no cost, not read
  even      nose down on even ground, pressing                                                no cost, not read
  side      a sideways part in the command, pressing                                          no cost, not read
  heavy     walk with the full payload, pressing in every substep                             4 at heavy 1, 2 at 0.5
  threshold 60: walk's 50 N is no contact.
(B) On the built configurations (no GPU), per robot:
  (1) with the switch off (U_DESCENT_LEG= in the environment: an empty value wins over the recipe's): no term, no
      item; the plain build: the recipe's own weight where the robot's recipe carries the switch (the legged kind's,
      6, since 2026-11-16), no term and no item where it does not;
  (2) U_DESCENT_LEG=6: the term at the weight -6 on stair_speed_deficit's command and scan and leg_ground_contact's
      sensor and force threshold, no heavy key; the item leg_down, logged only (no terms, not enforced), family
      down, its metric in constraints.CUSTOM; U_DESCENT_LEG=8,0.5 read as written; a weight of 0 or 40, a share of
      2, a third field stop the build;
  (3) every other reward term (leg_ground_contact among them) and constraint item as without the switch; the play
      build carries the term;
  (4) the stairs quality brings it in: quality_diff lists it from Q_EPS of its weight to its weight.

Local: PYTHONPATH=$PWD python scripts/unified/descent_leg_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

SWITCH = "U_DESCENT_LEG"


def term_cases(ucons) -> list[tuple[bool, str]]:
  import torch

  import src.tasks.velocity.mdp.constraints as old_cons

  names = ["walk", "both", "light", "clean", "stop", "slow", "climb", "even", "side", "heavy"]
  n = len(names)
  i = {k: j for j, k in enumerate(names)}
  cmd = torch.zeros(n, 3)
  cmd[:, 0] = 0.5
  cmd[i["stop"], 0], cmd[i["slow"], 0], cmd[i["side"], 1] = 0.0, 0.1, 0.3
  grav = torch.zeros(n, 3)
  grav[:, 0], grav[:, 2] = 0.3, -0.95   # nose down
  grav[i["climb"], 0] = -0.3
  uneven = torch.ones(n, dtype=torch.bool)
  uneven[i["even"]] = False
  force = torch.zeros(n, 2, 4, 3)       # [robots, leg geoms, substeps, xyz]
  force[i["walk"], 1, :2, 2] = 50.0
  force[i["both"], :, 3, 0] = 30.0
  force[i["light"], 0, :, 2] = 4.0
  for who in ("stop", "slow", "climb", "even", "side", "heavy"):
    force[i[who], 0, :, 2] = 50.0
  limit = 8.0
  kg = torch.zeros(n)
  kg[i["heavy"]] = limit
  vel = torch.zeros(n, 3)
  vel[:, 0] = 0.5
  data = SimpleNamespace(root_link_lin_vel_b=vel, projected_gravity_b=grav)
  env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02,
                        scene={"robot": SimpleNamespace(data=data),
                               "legs": SimpleNamespace(data=SimpleNamespace(force_history=force))},
                        command_manager=SimpleNamespace(get_command=lambda name: cmd), payload_kg=kg,
                        u_payload_full=limit)
  kw = {"command_name": "twist", "sensor_name": "scan", "leg_sensor_name": "legs"}
  keep = old_cons.uneven_terrain_mask
  old_cons.uneven_terrain_mask = lambda env, sensor_name, *a, **k: uneven.clone()
  out: list[tuple[bool, str]] = []

  def ok(cond, text):
    out.append((bool(cond), text))

  try:
    cost = ucons.descent_leg_contact(env, **kw)
    value, mask = ucons.CUSTOM["u_descent_leg"](env, leg_sensor_name="legs", command_name="twist",
                                                scan_sensor_name="scan")
    ok(float(cost[i["walk"]]) == 2.0 and mask[i["walk"]] and value[i["walk"]] == 1,
       "walk: a calf at 50 N in 2 of the 4 substeps costs 2; read, pressing")
    ok(float(cost[i["both"]]) == 1.0 and value[i["both"]] == 1, "both: two geoms in the same substep cost 1")
    ok(float(cost[i["light"]]) == 0.0 and mask[i["light"]] and value[i["light"]] == 0,
       "light: 4 N is no contact; read, not pressing")
    ok(float(cost[i["clean"]]) == 0.0 and mask[i["clean"]] and value[i["clean"]] == 0,
       "clean: no cost; read, not pressing")
    for who, why in (("stop", "told to stop on the descent"), ("slow", "told 0.1 m/s"), ("climb", "nose up"),
                     ("even", "on even ground"), ("side", "a sideways part in the command")):
      ok(float(cost[i[who]]) == 0.0 and not mask[i[who]], f"{who}: {why}, pressing -- no cost, not read")
    ok(float(cost[i["heavy"]]) == 4.0, "heavy: the full payload, 4 substeps cost 4 at heavy 1 (the default)")
    half = ucons.descent_leg_contact(env, heavy=0.5, **kw)
    ok(float(half[i["heavy"]]) == 2.0 and float(half[i["walk"]]) == 2.0,
       "heavy 0.5: the full payload costs 2, the unloaded robot as before")
    high = ucons.descent_leg_contact(env, force_threshold=60.0, **kw)
    ok(float(high.abs().sum()) == 0.0, "threshold 60 N: 50 N is no contact")
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
  os.environ.pop(SWITCH, None)
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

  def with_env(value: str, robot, **kw):
    os.environ[SWITCH] = value
    try:
      return build(robot, **kw)
    finally:
      os.environ.pop(SWITCH)

  def fails(value: str, robot) -> bool:
    try:
      with_env(value, robot)
    except AssertionError:
      return True
    return False

  def terms(cfg, skip=()):
    return {k: (v.weight, repr(sorted(v.params.items(), key=repr))) for k, v in cfg.rewards.items() if k not in skip}

  term = teacher_env.DESCENT_LEG_TERM
  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    cfg0, items0 = with_env("", robot)   # the switch off, whatever the robot's recipe carries
    check(term not in cfg0.rewards and not [i for i in items0 if i["name"] == "leg_down"],
          f"{SWITCH}= (off): no term, no item")
    cfgr, itemsr = build(robot)
    own = urecipe.recipe(robot).get(SWITCH)
    got, n_items = cfgr.rewards.get(term), len([i for i in itemsr if i["name"] == "leg_down"])
    if own:
      check(got is not None and got.weight == -float(own.split(",")[0]) and n_items == 1
            and terms(cfgr, skip=(term,)) == terms(cfg0) and [i for i in itemsr if i["name"] != "leg_down"] == items0,
            f"the plain build, the recipe's own {SWITCH}={own}: the term at {got.weight if got else None}, the item, "
            "the rest as with the switch off")
    else:
      check(got is None and n_items == 0 and terms(cfgr) == terms(cfg0) and itemsr == items0,
            f"the plain build, the recipe carries no {SWITCH}: no term, no item")
    cfg1, items1 = with_env("6", robot)
    new = [i for i in items1 if i["name"] == "leg_down"]
    check(len(new) == 1 and not new[0]["enforce"] and not new[0]["terms"] and new[0]["metric"] == "u_descent_leg"
          and "u_descent_leg" in ucons.CUSTOM and new[0]["sense"] == "max" and new[0]["stat"] == "mean"
          and new[0]["family"] == "down",
          f"{SWITCH}=6: the item leg_down (mean), logged only, no terms, family down, the metric known")
    r = cfg1.rewards.get(term)
    ref, leg = cfg1.rewards["stair_speed_deficit"].params, cfg1.rewards["leg_ground_contact"].params
    check(r is not None and r.func is ucons.descent_leg_contact and r.weight == -6.0
          and r.params == {"command_name": ref["command_name"], "sensor_name": ref["sensor_name"],
                           "leg_sensor_name": leg["sensor_name"], "force_threshold": leg["force_threshold"]},
          f"the term {term} at -6 on {ref['command_name']}, {ref['sensor_name']}, {leg['sensor_name']} above "
          f"{leg['force_threshold']} N, no heavy key")
    cfg2, _ = with_env("8,0.5", robot)
    check(cfg2.rewards[term].weight == -8.0 and cfg2.rewards[term].params["heavy"] == 0.5,
          f"{SWITCH}=8,0.5 read as written")
    check(all(fails(v, robot) for v in ("0", "40", "6,2", "6,1,1")),
          "a weight of 0 or 40, a share of 2, a third field: the build stops")
    check(terms(cfg0) == terms(cfg1, skip=(term,)),
          f"the other {len(cfg0.rewards)} reward terms unchanged (leg_ground_contact at "
          f"{cfg1.rewards['leg_ground_contact'].weight})")
    check([i for i in items1 if i["name"] != "leg_down"] == items0,
          f"the other {len(items0)} constraint items unchanged")
    cfgp, _ = with_env("6", robot, play=True)
    check(term in cfgp.rewards, "the play build carries the term")
    with urecipe.scope(robot):
      weights, _, _ = teacher_env.quality_diff(robot, cfg1.rewards, set(), set())
    check(weights.get(term) == (teacher_env.Q_EPS * -6.0, -6.0),
          f"the stairs quality brings it in: {weights.get(term)} (Q_EPS of its weight -> its weight)")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
