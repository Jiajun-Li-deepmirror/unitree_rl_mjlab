#!/usr/bin/env python3
"""stop_place_test.py -- U_STOP_PLACE: the cost of leaving the place a stop on a flight began at (go2-depth
2026-11-14; constraints.stop_place, teacher_env._add_stop_place).

(A) The term on a small made-up environment (CPU, no simulator), one robot per case:
  hold      stands where the stand began on uneven ground                 no cost; near 1; counted 2-4 s into the stand
  retreat   rolls 1.5 m back in the first second, onto even ground        nothing within 0.30 m, in full from 0.60 m on
                                                                          and for the rest of the stand; near 0
  tread     rolls 0.35 m (a tread) and stays                              a sixth of the cost; near 1
  even      the stand began on even ground, then it moves                 no cost, not counted
  drive     a moving command, the robot moves                             no cost, not counted
  pushed    as retreat, a push 0.5 s into the stand                       no cost from the push on, not counted
  heavy     as retreat, at the payload limit                              `heavy` times the cost (0: none)
  common    as retreat, at the common payload                             the cost in full
then the command moves (no cost for anyone) and falls to zero again: the new stand takes a new reference.
(B) On the built configurations (no GPU), per robot:
  (1) without the switch: no stop_place item, no u_stop_place term;
  (2) with U_STOP_PLACE=8: the term at the weight -8 with the defaults (free 0.30, full 0.60, heavy 0), the command
      and the scan of feet_stand_uneven; four items stop_place_{up,down}[_heavy], logged only (enforce False, no
      terms), their metrics in constraints.CUSTOM; U_STOP_PLACE=6,0.2,0.8,0.5 read as written;
  (3) every other reward term and constraint item as without the switch; the play build carries the term too.

Local: PYTHONPATH=$PWD python scripts/unified/stop_place_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace


def term_cases(ucons) -> list[tuple[bool, str]]:
  import torch

  import src.tasks.velocity.mdp.rewards as old_rewards

  names = ["hold", "retreat", "tread", "even", "drive", "pushed", "heavy", "common"]
  n = len(names)
  i = {k: j for j, k in enumerate(names)}
  dt = 0.02
  limit = 8.0
  pos = torch.zeros(n, 3)
  grav = torch.zeros(n, 3)
  grav[:, 0] = -0.3   # nose up
  grav[:, 2] = -0.95
  cmd = torch.zeros(n, 3)
  uneven = torch.ones(n, dtype=torch.bool)
  kg = torch.zeros(n)
  kg[i["heavy"]] = limit
  kg[i["common"]] = limit * ucons.COMMON_FRAC
  twist = SimpleNamespace(is_stair_env=torch.ones(n, dtype=torch.bool))
  env = SimpleNamespace(
    num_envs=n, device="cpu", step_dt=dt, common_step_counter=0, payload_kg=kg, u_payload_full=limit,
    _push_step=torch.zeros(n, dtype=torch.long),
    scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=pos, projected_gravity_b=grav))},
    command_manager=SimpleNamespace(get_command=lambda name: cmd, get_term=lambda name: twist))
  keep = old_rewards.uneven_terrain_mask
  old_rewards.uneven_terrain_mask = lambda env, sensor_name, *a, **k: uneven.clone()
  out: list[tuple[bool, str]] = []
  try:
    for heavy in (0.0, 0.5):
      t = ucons.stop_place(None, env)
      t.reset()
      params = {"command_name": "twist", "sensor_name": "scan", "heavy": heavy}
      pos.zero_()
      cmd.zero_()
      env._push_step.zero_()
      uneven[:] = True
      # one moving step first: the stand begins at the next one
      cmd[:, 0] = 0.5
      env.common_step_counter = 100
      t(env, **params)
      cmd.zero_()
      cmd[i["drive"], 0] = 0.5
      uneven[i["even"]] = False
      total = torch.zeros(n)
      counted = torch.zeros(n)
      near_sum = torch.zeros(n)
      late = torch.zeros(n)
      for k in range(200):   # 4 s of stand
        sec = k * dt
        back = min(sec, 1.0) * 1.5
        for who in ("retreat", "even", "drive", "pushed", "heavy", "common"):
          pos[i[who], 0] = -back
        pos[i["tread"], 0] = -min(sec, 1.0) * 0.35
        if k == 30:
          uneven[:] = False   # everyone is on even ground from here on: the term must not ask the ground again
        if k == 25:
          env._push_step[i["pushed"]] = 100 + k
        env.common_step_counter = 101 + k
        r = t(env, **params)
        total += r * dt
        counted += t.mask.float()
        near_sum += (t.near * t.mask.float())
        if sec >= 2.0:
          late += r * dt
      full = 2.0   # the last 2 s at the full cost, in cost x seconds
      def ok(cond, text):
        out.append((bool(cond), f"heavy {heavy}: {text}"))
      ok(total[i["hold"]] == 0 and counted[i["hold"]] == 100 and near_sum[i["hold"]] == 100,
         f"hold: no cost, counted {int(counted[i['hold']])} steps (2-4 s), all near")
      ok(abs(late[i["retreat"]] - full) < 1e-4 and total[i["retreat"]] > full + 1.0 and near_sum[i["retreat"]] == 0
         and counted[i["retreat"]] == 100,
         f"retreat: the full cost for the rest of the stand, on even ground too ({total[i['retreat']]:.2f} cost-s), never near")
      ok(abs(late[i["tread"]] - full * (0.05 / 0.30)) < 1e-4 and near_sum[i["tread"]] == 100,
         f"tread: 0.35 m costs {late[i['tread']] / full:.3f} of the full cost, near")
      ok(total[i["even"]] == 0 and counted[i["even"]] == 0, "even: a stand begun on even ground costs nothing, not counted")
      ok(total[i["drive"]] == 0 and counted[i["drive"]] == 0, "drive: a moving command costs nothing, not counted")
      ok(late[i["pushed"]] == 0 and counted[i["pushed"]] == 0 and total[i["pushed"]] < total[i["retreat"]] * 0.1,
         f"pushed: no cost from the push on ({total[i['pushed']]:.3f} cost-s before it), not counted")
      ok(abs(late[i["heavy"]] - full * heavy) < 1e-4, f"heavy: {heavy} x the cost at the payload limit ({late[i['heavy']]:.2f})")
      ok(abs(late[i["common"]] - full) < 1e-4, "common: the cost in full at the common payload")
      # the readings by the nose's direction when the stand began, and by the payload
      up_v, up_m = ucons.CUSTOM["u_stop_place"](env, facing="up")
      dn_v, dn_m = ucons.CUSTOM["u_stop_place"](env, facing="down")
      hv_v, hv_m = ucons.CUSTOM["u_stop_place_heavy"](env, facing="up")
      ok(bool(up_m[i["hold"]]) and bool(up_m[i["common"]]) and not bool(up_m[i["heavy"]]) and not dn_m.any()
         and bool(hv_m[i["heavy"]]) and int(hv_m.sum()) == 1,
         "readings: nose up when the stand began -> the up item; the heavy class in its own")
      # the command moves, then a new stand elsewhere: a new reference
      cmd[:, 0] = 0.5
      env.common_step_counter += 1
      r = t(env, **params)
      ok(float(r.abs().sum()) == 0 and not t.mask.any(), "the command moves: no cost, nobody counted")
      cmd.zero_()
      uneven[:] = True
      grav[:, 0] = 0.3   # nose down now
      env.common_step_counter += 1
      r = t(env, **params)
      ok(float(r.abs().sum()) == 0 and float(t.dist.abs().sum()) == 0 and bool(t.down.all()) and not t.up.any(),
         "a new stand: the reference is the place it began at (no cost), the direction taken anew")
      grav[:, 0] = -0.3
      t.reset(torch.tensor([i["retreat"]]))
      ok(not bool(t.on[i["retreat"]]) and bool(t.on[i["hold"]]), "reset: that robot's stand is forgotten, the others' kept")
  finally:
    old_rewards.uneven_terrain_mask = keep
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W,As2")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  os.environ.pop("U_STOP_PLACE", None)
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
    os.environ["U_STOP_PLACE"] = value
    try:
      return build(robot, **kw)
    finally:
      os.environ.pop("U_STOP_PLACE")

  term = teacher_env.STOP_PLACE_TERM
  want = [f"stop_place_{d}{h}" for d in ("up", "down") for h in ("", "_heavy")]
  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    cfg0, items0 = build(robot)
    check(term not in cfg0.rewards and not [i for i in items0 if i["name"].startswith("stop_place")],
          "without the switch: no item, no term")
    cfg1, items1 = with_switch("8", robot)
    new = [i for i in items1 if i["name"].startswith("stop_place")]
    check(sorted(i["name"] for i in new) == sorted(want), f"with it: the items {[i['name'] for i in new]}")
    check(all(not i["enforce"] and not i["terms"] and i["metric"] in ucons.CUSTOM and i["sense"] == "min"
              and i["target"] == 0.95 for i in new), "logged only, no terms, the metrics known, bar 0.95 (min)")
    r = cfg1.rewards.get(term)
    ref = cfg1.rewards["feet_stand_uneven"].params
    check(r is not None and r.func is ucons.stop_place and r.weight == -8.0
          and (r.params["free"], r.params["full"], r.params["heavy"]) == (0.30, 0.60, 0.0)
          and r.params["command_name"] == ref["command_name"] and r.params["sensor_name"] == ref["scan_sensor_name"],
          f"the term at -8, free 0.30 / full 0.60 / heavy 0, on {ref['command_name']} and {ref['scan_sensor_name']}")
    cfg2, _ = with_switch("6,0.2,0.8,0.5", robot)
    p = cfg2.rewards[term].params
    check(cfg2.rewards[term].weight == -6.0 and (p["free"], p["full"], p["heavy"]) == (0.2, 0.8, 0.5),
          "U_STOP_PLACE=6,0.2,0.8,0.5 read as written")
    rest0 = {k: (v.weight, repr(sorted(v.params))) for k, v in cfg0.rewards.items()}
    rest1 = {k: (v.weight, repr(sorted(v.params))) for k, v in cfg1.rewards.items() if k != term}
    check(rest0 == rest1, f"the other {len(rest0)} reward terms unchanged")
    check([i for i in items1 if not i["name"].startswith("stop_place")] == items0,
          f"the other {len(items0)} constraint items unchanged")
    cfgp, _ = with_switch("8", robot, play=True)
    check(term in cfgp.rewards, "the play build carries the term")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
