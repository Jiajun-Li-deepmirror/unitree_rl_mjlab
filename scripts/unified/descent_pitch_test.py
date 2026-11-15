#!/usr/bin/env python3
"""descent_pitch_test.py -- U_DESCENT_PITCH: the base rocking fore and aft while the robot walks down a flight costs
(go2-depth 2026-11-15; constraints.descent_pitch_rate, teacher_env._add_descent_pitch). The user on the Go2W teacher's
videos: "下楼梯的pitch角速度很大，一直会前后仰俯".

(A) The term on a small made-up environment (CPU, no simulator), one robot per case, the scale 0.8 rad/s:
  walk      nose down on uneven ground, 0.5 m/s ahead, a pitch rate of 0.8 rad/s            1; read at 0.8
  back      ... of -1.6 rad/s                                                              4; read at 1.6
  roll      ... no pitch rate, a roll rate of 2 and a yaw rate of 1                         no cost; read at 0
  peak      ... of 8 rad/s                                                                 16 (the most)
  stop      told to stop there, 1.6 rad/s                                                  no cost, not read
  slow      told 0.1 m/s ahead (below the descent item's 0.2)                              no cost, not read
  climb     nose up                                                                        no cost, not read
  even      nose down on even ground                                                       no cost, not read
  side      a sideways part in the command                                                 no cost, not read
  heavy     walk with the full payload, 0.8 rad/s                                          1 at heavy 1, 0.5 at 0.5
  scale 1.6: walk's 0.8 rad/s costs 0.25.
(B) On the built configurations (no GPU), per robot:
  (1) without the switch (no recipe carries it): no term, no item;
  (2) U_DESCENT_PITCH=1: the term at the weight -1 on stair_speed_deficit's command and scan, no scale and no heavy
      key; the item pitch_down, logged only (no terms, not enforced), family down, its metric in
      constraints.CUSTOM; U_DESCENT_PITCH=2,1.2,0.5 read as written; a weight of 0 or 30, a scale of 0.1 or 5, a
      share of 2, a fourth field stop the build;
  (3) every other reward term (body_ang_vel among them) and constraint item as without the switch; the play build
      carries the term;
  (4) the stairs quality brings it in: quality_diff lists it from Q_EPS of its weight to its weight.

Local: PYTHONPATH=$PWD python scripts/unified/descent_pitch_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

SWITCH = "U_DESCENT_PITCH"


def term_cases(ucons) -> list[tuple[bool, str]]:
  import torch

  import src.tasks.velocity.mdp.constraints as old_cons

  names = ["walk", "back", "roll", "peak", "stop", "slow", "climb", "even", "side", "heavy"]
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
  rate = torch.zeros(n, 3)              # roll, pitch, yaw rate in the base frame (rad/s)
  rate[:, 1] = 1.6
  rate[i["walk"], 1], rate[i["back"], 1], rate[i["peak"], 1], rate[i["heavy"], 1] = 0.8, -1.6, 8.0, 0.8
  rate[i["roll"]] = torch.tensor([2.0, 0.0, 1.0])
  limit = 8.0
  kg = torch.zeros(n)
  kg[i["heavy"]] = limit
  vel = torch.zeros(n, 3)
  vel[:, 0] = 0.5
  data = SimpleNamespace(root_link_lin_vel_b=vel, root_link_ang_vel_b=rate, projected_gravity_b=grav)
  env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02, scene={"robot": SimpleNamespace(data=data)},
                        command_manager=SimpleNamespace(get_command=lambda name: cmd), payload_kg=kg,
                        u_payload_full=limit)
  kw = {"command_name": "twist", "sensor_name": "scan"}
  keep = old_cons.uneven_terrain_mask
  old_cons.uneven_terrain_mask = lambda env, sensor_name, *a, **k: uneven.clone()
  out: list[tuple[bool, str]] = []

  def ok(cond, text):
    out.append((bool(cond), text))

  def near(x, y):
    return abs(float(x) - y) < 1e-5

  try:
    cost = ucons.descent_pitch_rate(env, **kw)
    value, mask = ucons.CUSTOM["u_descent_pitch"](env, command_name="twist", scan_sensor_name="scan")
    ok(near(cost[i["walk"]], 1.0) and mask[i["walk"]] and near(value[i["walk"]], 0.8),
       "walk: a pitch rate of 0.8 rad/s costs 1; read at 0.8")
    ok(near(cost[i["back"]], 4.0) and mask[i["back"]] and near(value[i["back"]], 1.6),
       "back: -1.6 rad/s costs 4; read at 1.6")
    ok(near(cost[i["roll"]], 0.0) and mask[i["roll"]] and near(value[i["roll"]], 0.0),
       "roll: a roll rate of 2 and a yaw rate of 1, no pitch rate -- no cost; read at 0")
    ok(near(cost[i["peak"]], 16.0), "peak: 8 rad/s costs 16 (the most)")
    for who, why in (("stop", "told to stop on the descent"), ("slow", "told 0.1 m/s"), ("climb", "nose up"),
                     ("even", "on even ground"), ("side", "a sideways part in the command")):
      ok(near(cost[i[who]], 0.0) and not mask[i[who]], f"{who}: {why}, 1.6 rad/s -- no cost, not read")
    ok(near(cost[i["heavy"]], 1.0), "heavy: the full payload, 0.8 rad/s costs 1 at heavy 1 (the default)")
    half = ucons.descent_pitch_rate(env, heavy=0.5, **kw)
    ok(near(half[i["heavy"]], 0.5) and near(half[i["walk"]], 1.0),
       "heavy 0.5: the full payload costs 0.5, the unloaded robot as before")
    wide = ucons.descent_pitch_rate(env, scale=1.6, **kw)
    ok(near(wide[i["walk"]], 0.25) and near(wide[i["peak"]], 16.0), "scale 1.6: 0.8 rad/s costs 0.25")
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

  term = teacher_env.DESCENT_PITCH_TERM
  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    cfg0, items0 = build(robot)
    check(term not in cfg0.rewards and not [i for i in items0 if i["name"] == "pitch_down"],
          f"without {SWITCH}: no term, no item")
    cfg1, items1 = with_env("1", robot)
    new = [i for i in items1 if i["name"] == "pitch_down"]
    check(len(new) == 1 and not new[0]["enforce"] and not new[0]["terms"] and new[0]["metric"] == "u_descent_pitch"
          and "u_descent_pitch" in ucons.CUSTOM and new[0]["sense"] == "max" and new[0]["stat"] == "mean"
          and new[0]["family"] == "down",
          f"{SWITCH}=1: the item pitch_down (mean), logged only, no terms, family down, the metric known")
    r = cfg1.rewards.get(term)
    ref = cfg1.rewards["stair_speed_deficit"].params
    check(r is not None and r.func is ucons.descent_pitch_rate and r.weight == -1.0
          and r.params == {"command_name": ref["command_name"], "sensor_name": ref["sensor_name"]},
          f"the term {term} at -1 on {ref['command_name']}, {ref['sensor_name']}, no scale and no heavy key")
    cfg2, _ = with_env("2,1.2,0.5", robot)
    check(cfg2.rewards[term].weight == -2.0 and cfg2.rewards[term].params["scale"] == 1.2
          and cfg2.rewards[term].params["heavy"] == 0.5, f"{SWITCH}=2,1.2,0.5 read as written")
    check(all(fails(v, robot) for v in ("0", "30", "1,0.1", "1,5", "1,0.8,2", "1,0.8,1,1")),
          "a weight of 0 or 30, a scale of 0.1 or 5, a share of 2, a fourth field: the build stops")
    rates = {k: cfg1.rewards[k].weight for k in ("body_ang_vel", "body_ang_vel_stairs") if k in cfg1.rewards}
    check(terms(cfg0) == terms(cfg1, skip=(term,)),
          f"the other {len(cfg0.rewards)} reward terms unchanged (the rates' costs: {rates})")
    check([i for i in items1 if i["name"] != "pitch_down"] == items0,
          f"the other {len(items0)} constraint items unchanged")
    cfgp, _ = with_env("1", robot, play=True)
    check(term in cfgp.rewards, "the play build carries the term")
    with urecipe.scope(robot):
      weights, _, _ = teacher_env.quality_diff(robot, cfg1.rewards, set(), set())
    check(weights.get(term) == (teacher_env.Q_EPS * -1.0, -1.0),
          f"the stairs quality brings it in: {weights.get(term)} (Q_EPS of its weight -> its weight)")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
