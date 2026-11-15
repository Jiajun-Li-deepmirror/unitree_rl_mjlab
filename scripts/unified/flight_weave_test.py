#!/usr/bin/env python3
"""flight_weave_test.py -- U_FLIGHT_WEAVE / U_FLIGHT_HEADING: weaving along a flight and turning off the commanded
heading on it cost (go2-depth 2026-11-15; constraints.flight_weave / flight_heading, teacher_env._add_flight_weave /
_add_flight_heading). The user on the Go2 teacher's videos: "不是旋转晃动，是左右晃动，像是喝醉了走不了直线", "需要训练
蛇行幅度，但是不用门槛", "朝向最大偏离也要训练", "Go2W 上楼的蛇行也要算在内".

(A) The terms on a small made-up environment (CPU, no simulator), one robot per case, the command 0.5 m/s ahead, the
    line along +x, 3 s of steps at 0.02 s. The weave (the smoothed speed across the line / 0.1 m/s, at most 1):
  line      nose down on uneven ground, 0.5 m/s along the line           no cost; read (down), 0 m/s across
  slip      ... and 0.08 m/s to the left all the way                     0.8; read, 0.08 m/s
  wide      ... 0.3 m/s to the left                                      1 (the most)
  turned    the heading 10 deg off the line, 0.5 m/s where the nose points   0.087 m/s across the line: 0.87
  sway      +-0.1 m/s across, alternating every 0.2 s (the trot's sway)  under 0.03 m/s smoothed (0.1 raw)
  wander    0.044 m amplitude at a period of 3 s (the Go2's 8.8 cm)      0.03-0.07 m/s smoothed on average
  climb     nose up, 0.08 m/s to the left                                0.8, read (up); nothing at up 0, 0.4 at 0.5
  even      nose down on even ground, 0.08 m/s to the left               no cost, not read
  side      a sideways part in the command                               no cost, not read, the smoothed speed 0
  yaw       a yaw part in the command                                    no cost, not read, the smoothed speed 0
  heavy     slip with the full payload                                   0.8 at heavy 1 (the default), 0.4 at 0.5
  scale 0.2 halves slip's cost; a reset zeroes the smoothed speed; nothing while the heading observation settles;
  without the heading observation: zeros.
    The heading ((the error / 0.14 rad)^2, at most 4), the same robots and:
  line 0; bar (0.14 rad off) 1; turned (10 deg) 1.55; far (0.4 rad) 4 (the most); wrap (the target 3.1, the heading
  -3.1: 0.083 rad apart) 0.35; climb (0.07 rad) 0.25, nothing at up 0; even / side / yaw (0.14 rad) no cost, not
  read; heavy (0.14 rad) 1, 0.5 at heavy 0.5; scale 0.28: the bar robot 0.25.
(B) On the built configurations (no GPU), per robot and per switch:
  (1) without the switch (no recipe carries it yet): no term, no items;
  (2) with <switch>=5: the term at the weight -5, the default scale, no up / heavy key, on stair_speed_deficit's
      command and scan; two items (weave_down / weave_up; heading_down / heading_up at the p90), logged only (no
      terms, not enforced), their metric in constraints.CUSTOM; <switch>=8,0.05,1,0.5 read as written; a weight of 0
      or 40, a scale of 1, a share of 2, a fifth field stop the build;
  (3) every other reward term and constraint item as without the switch; the play build carries the term;
  (4) the stairs quality brings it in: quality_diff lists it from Q_EPS of its weight to its weight;
  and both switches together: both terms, the four items, the rest unchanged.

Local: PYTHONPATH=$PWD python scripts/unified/flight_weave_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import math
import os
import sys
from types import SimpleNamespace

DT = 0.02
SPEED = 0.5


def term_cases(ucons) -> list[tuple[bool, str]]:
  import torch

  import src.tasks.velocity.mdp.constraints as old_cons

  names = ["line", "slip", "wide", "turned", "sway", "wander", "climb", "even", "side", "yaw", "heavy", "bar", "far",
           "wrap"]
  n = len(names)
  i = {k: j for j, k in enumerate(names)}
  cmd = torch.zeros(n, 3)
  cmd[:, 0] = SPEED
  cmd[i["side"], 1] = 0.3
  cmd[i["yaw"], 2] = 0.5
  grav = torch.zeros(n, 3)
  grav[:, 0], grav[:, 2] = 0.3, -0.95   # nose down
  grav[i["climb"], 0] = -0.3
  uneven = torch.ones(n, dtype=torch.bool)
  uneven[i["even"]] = False
  vel = torch.zeros(n, 3)
  head = torch.zeros(n)
  head[i["turned"]] = math.radians(10.0)
  for who in ("bar", "heavy", "even", "side", "yaw"):
    head[i[who]] = 0.14
  head[i["far"]], head[i["climb"]], head[i["wrap"]] = 0.4, 0.07, -3.1
  target = torch.zeros(n)
  target[i["wrap"]] = 3.1
  data = SimpleNamespace(root_link_lin_vel_w=vel, root_link_lin_vel_b=vel, projected_gravity_b=grav, heading_w=head)
  observed = SimpleNamespace(target=target, settle=torch.zeros(n, dtype=torch.long))
  limit = 8.0
  kg = torch.zeros(n)
  kg[i["heavy"]] = limit

  def obs_cfg(group, name):
    assert (group, name) == ("critic", "heading_error")
    return SimpleNamespace(func=observed)

  env = SimpleNamespace(num_envs=n, device="cpu", step_dt=DT, scene={"robot": SimpleNamespace(data=data)},
                        command_manager=SimpleNamespace(get_command=lambda name: cmd),
                        observation_manager=SimpleNamespace(get_term_cfg=obs_cfg), payload_kg=kg, u_payload_full=limit)

  def velocity(step: int) -> None:
    t = step * DT
    vel.zero_()
    vel[:, 0] = SPEED
    for who in ("slip", "climb", "even", "side", "yaw", "heavy"):
      vel[i[who], 1] = 0.08
    vel[i["wide"], 1] = 0.3
    a = math.radians(10.0)
    vel[i["turned"], 0], vel[i["turned"], 1] = SPEED * math.cos(a), SPEED * math.sin(a)
    vel[i["sway"], 1] = 0.1 if int(t / 0.2) % 2 == 0 else -0.1
    vel[i["wander"], 1] = 0.044 * (2 * math.pi / 3.0) * math.cos(2 * math.pi * t / 3.0)
    vel[i["wrap"], 0] = -SPEED   # along its own line (the target 3.1 rad)

  kw = {"command_name": "twist", "sensor_name": "scan"}
  keep = old_cons.uneven_terrain_mask
  old_cons.uneven_terrain_mask = lambda env, sensor_name, *a, **k: uneven.clone()
  out: list[tuple[bool, str]] = []

  def ok(cond, text):
    out.append((bool(cond), text))

  def near(x, want, tol=0.01) -> bool:
    return abs(float(x) - want) < tol

  try:
    # ---- the weave
    term = ucons.flight_weave(None, env)
    read = ucons.CUSTOM["u_flight_weave"]
    costs, reads = [], []
    for k in range(150):
      velocity(k)
      costs.append(term(env, **kw).clone())
      reads.append(read(env)[0].clone())
    costs, reads = torch.stack(costs), torch.stack(reads)
    last = costs[-1]
    value, down = read(env, facing="down")
    _, up = read(env, facing="up")
    ok(last[i["line"]] == 0 and down[i["line"]] and value[i["line"]] == 0,
       "weave, line: no cost, read (down), 0 m/s across")
    ok(near(last[i["slip"]], 0.8) and down[i["slip"]] and near(value[i["slip"]], 0.08, 1e-3),
       f"slip: 0.08 m/s to the left costs {float(last[i['slip']]):.3f} (0.8), read {float(value[i['slip']]):.3f} m/s")
    ok(float(last[i["wide"]]) == 1.0, "wide: 0.3 m/s costs 1 (the most)")
    want = SPEED * math.sin(math.radians(10.0))
    ok(near(value[i["turned"]], want, 1e-3) and near(last[i["turned"]], want / 0.1),
       f"turned: 10 deg off the line at 0.5 m/s reads {float(value[i['turned']]):.3f} m/s across it ({want:.3f})")
    sway = reads[50:, i["sway"]]
    ok(float(sway.max()) < 0.03 and float(sway.mean()) < 0.02,
       f"sway: +-0.1 m/s every 0.2 s reads at most {float(sway.max()):.3f} m/s smoothed, {float(sway.mean()):.3f} "
       "on average (under 0.03 / 0.02)")
    wander = reads[:, i["wander"]].mean()
    ok(0.03 <= float(wander) <= 0.07,
       f"wander: 8.8 cm peak-to-peak at a period of 3 s reads {float(wander):.3f} m/s on average (0.03-0.07)")
    ok(near(last[i["climb"]], 0.8) and up[i["climb"]] and not down[i["climb"]] and near(value[i["climb"]], 0.08, 1e-3),
       f"climb: costs {float(last[i['climb']]):.3f} (0.8: both directions), read (up)")
    ok(last[i["even"]] == 0 and not down[i["even"]] and not up[i["even"]], "even: no cost, not read")
    for who in ("side", "yaw"):
      ok(last[i[who]] == 0 and not down[i[who]] and not up[i[who]] and value[i[who]] == 0,
         f"{who}: no cost, not read, the smoothed speed 0")
    ok(near(last[i["heavy"]], 0.8), "heavy: the full payload costs as every payload at heavy 1")
    ok(int(down.sum()) == n - 4 and int(up.sum()) == 1, f"read: {n - 4} robots going down, 1 going up")
    half = term(env, heavy=0.5, **kw)
    ok(near(half[i["heavy"]], 0.4) and near(half[i["slip"]], 0.8),
       f"heavy 0.5: the full payload costs {float(half[i['heavy']]):.3f} (0.4), an empty robot as before")
    climbs = term(env, up=0.5, **kw)
    none = term(env, up=0.0, **kw)
    ok(near(climbs[i["climb"]], 0.4) and none[i["climb"]] == 0 and near(none[i["slip"]], 0.8),
       f"up 0.5: the climb costs {float(climbs[i['climb']]):.3f} (0.4); up 0: nothing, the descent as before")
    wide = term(env, scale=0.2, **kw)
    ok(near(wide[i["slip"]], 0.4), f"scale 0.2: slip costs {float(wide[i['slip']]):.3f} (0.4)")
    term.reset(torch.tensor([i["slip"]]))
    ok(term.lat[i["slip"]] == 0 and not term.down[i["slip"]] and term.lat[i["wide"]] > 0.25,
       "reset: the smoothed speed 0 again, the other robots as they were")
    observed.settle[i["wide"]] = 2
    settling = term(env, **kw)
    ok(settling[i["wide"]] == 0 and term.lat[i["wide"]] == 0 and not term.down[i["wide"]],
       "while the heading observation settles after a reset: no cost, not read")
    observed.settle[i["wide"]] = 0

    # ---- the heading
    hterm = ucons.flight_heading(None, env)
    hread = ucons.CUSTOM["u_flight_heading"]
    cost = hterm(env, **kw)
    off, down = hread(env, facing="down")
    _, up = hread(env, facing="up")
    ok(cost[i["line"]] == 0 and down[i["line"]] and off[i["line"]] == 0, "heading, line: no cost, read (down), 0 rad")
    ok(near(cost[i["bar"]], 1.0) and near(off[i["bar"]], 0.14, 1e-4), "bar: 0.14 rad off costs 1, read 0.14 rad")
    ok(near(cost[i["turned"]], (math.radians(10.0) / 0.14) ** 2),
       f"turned: 10 deg off costs {float(cost[i['turned']]):.3f} (1.554)")
    ok(float(cost[i["far"]]) == 4.0, "far: 0.4 rad off costs 4 (the most)")
    apart = 2 * math.pi - 6.2
    ok(near(off[i["wrap"]], apart, 1e-4) and near(cost[i["wrap"]], (apart / 0.14) ** 2),
       f"wrap: the target 3.1, the heading -3.1 read {float(off[i['wrap']]):.3f} rad apart ({apart:.3f})")
    ok(near(cost[i["climb"]], 0.25) and up[i["climb"]] and not down[i["climb"]],
       f"climb: 0.07 rad off costs {float(cost[i['climb']]):.3f} (0.25: both directions), read (up)")
    for who in ("even", "side", "yaw"):
      ok(cost[i[who]] == 0 and not down[i[who]] and not up[i[who]], f"{who}: no cost, not read")
    ok(near(cost[i["heavy"]], 1.0), "heavy: the full payload costs as every payload at heavy 1")
    ok(int(down.sum()) == n - 4 and int(up.sum()) == 1, f"read: {n - 4} robots going down, 1 going up")
    half = hterm(env, heavy=0.5, **kw)
    ok(near(half[i["heavy"]], 0.5) and near(half[i["bar"]], 1.0),
       f"heavy 0.5: the full payload costs {float(half[i['heavy']]):.3f} (0.5), an empty robot as before")
    none = hterm(env, up=0.0, **kw)
    ok(none[i["climb"]] == 0 and near(none[i["bar"]], 1.0), "up 0: the climb costs nothing, the descent as before")
    wide = hterm(env, scale=0.28, **kw)
    ok(near(wide[i["bar"]], 0.25), f"scale 0.28: the bar robot costs {float(wide[i['bar']]):.3f} (0.25)")
    observed.settle[i["far"]] = 2
    settling = hterm(env, **kw)
    ok(settling[i["far"]] == 0 and not hterm.down[i["far"]],
       "while the heading observation settles after a reset: no cost, not read")
    observed.settle[i["far"]] = 0
    hterm(env, **kw)
    hterm.reset(torch.tensor([i["far"]]))
    ok(hterm.off[i["far"]] == 0 and not hterm.down[i["far"]] and near(hterm.off[i["bar"]], 0.14, 1e-4),
       "reset: the reading 0 again, the other robots as they were")

    # ---- without the observation, without the terms
    def absent(group, name):
      raise ValueError(name)

    env.observation_manager = SimpleNamespace(get_term_cfg=absent)
    ok(float(term(env, **kw).abs().sum()) == 0 and float(hterm(env, **kw).abs().sum()) == 0,
       "without the heading observation: zeros (both terms)")
    del env.u_flight_weave, env.u_flight_heading
    for name in ("u_flight_weave", "u_flight_heading"):
      try:
        ucons.CUSTOM[name](env)
        ok(False, f"{name} without its term: an error")
      except RuntimeError:
        ok(True, f"{name} without its term: an error")
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
  switches = ("U_FLIGHT_WEAVE", "U_FLIGHT_HEADING")
  for sw in switches:
    os.environ.pop(sw, None)
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

  print("the terms:")
  for good, text in term_cases(ucons):
    check(good, text)

  def build(robot, **kw):
    cfg = teacher_env.unified_teacher_env_cfg(robot, **kw)
    with urecipe.scope(robot):
      items = teacher_env.constraint_items(robot.kind)
    return cfg, items

  def with_env(values: dict, robot, **kw):
    os.environ.update(values)
    try:
      return build(robot, **kw)
    finally:
      for k in values:
        os.environ.pop(k)

  def fails(values: dict, robot) -> bool:
    try:
      with_env(values, robot)
    except AssertionError:
      return True
    return False

  def terms(cfg, skip=()):
    return {k: (v.weight, repr(sorted(v.params.items(), key=repr))) for k, v in cfg.rewards.items() if k not in skip}

  # (the switch, its term, its function, the default scale, its items, their statistic, their metric)
  specs = [("U_FLIGHT_WEAVE", teacher_env.FLIGHT_WEAVE_TERM, ucons.flight_weave, 0.1, ("weave_down", "weave_up"),
            "mean", "u_flight_weave"),
           ("U_FLIGHT_HEADING", teacher_env.FLIGHT_HEADING_TERM, ucons.flight_heading, 0.14,
            ("heading_down", "heading_up"), "p90", "u_flight_heading")]
  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    cfg0, items0 = build(robot)
    for sw, term, func, scale, mine, stat, metric in specs:
      check(term not in cfg0.rewards and not [i for i in items0 if i["name"] in mine],
            f"without {sw}: no term, no items")
      cfg1, items1 = with_env({sw: "5"}, robot)
      new = [i for i in items1 if i["name"] in mine]
      check([i["name"] for i in new] == list(mine) and all(
              not i["enforce"] and not i["terms"] and i["metric"] == metric and metric in ucons.CUSTOM
              and i["sense"] == "max" and i["stat"] == stat for i in new)
            and [i["params"] for i in new] == [{"facing": "down"}, {"facing": "up"}]
            and [i["family"] for i in new] == ["down", "up"],
            f"{sw}=5: the items {mine[0]} / {mine[1]} ({stat}), logged only, no terms, the metric known")
      r = cfg1.rewards.get(term)
      ref = cfg1.rewards["stair_speed_deficit"].params
      check(r is not None and r.func is func and r.weight == -5.0
            and r.params == {"command_name": ref["command_name"], "sensor_name": ref["sensor_name"], "scale": scale},
            f"the term {term} at -5, scale {scale}, on {ref['command_name']} and {ref['sensor_name']}, no up / heavy key")
      cfg2, _ = with_env({sw: "8,0.05,1,0.5"}, robot)
      p = cfg2.rewards[term].params
      check(cfg2.rewards[term].weight == -8.0 and (p["scale"], p["up"], p["heavy"]) == (0.05, 1.0, 0.5),
            f"{sw}=8,0.05,1,0.5 read as written")
      check(all(fails({sw: v}, robot) for v in ("0", "40", "5,1", "5,0.1,2", "5,0.1,1,2", "5,0.1,1,1,1")),
            "a weight of 0 or 40, a scale of 1, a share of 2, a fifth field: the build stops")
      check(terms(cfg0) == terms(cfg1, skip=(term,)), f"the other {len(cfg0.rewards)} reward terms unchanged")
      check([i for i in items1 if i["name"] not in mine] == items0,
            f"the other {len(items0)} constraint items unchanged")
      cfgp, _ = with_env({sw: "5"}, robot, play=True)
      check(term in cfgp.rewards, "the play build carries the term")
      with urecipe.scope(robot):
        weights, _, _ = teacher_env.quality_diff(robot, cfg1.rewards, set(), set())
      check(weights.get(term) == (teacher_env.Q_EPS * -5.0, -5.0),
            f"the stairs quality brings it in: {weights.get(term)} (Q_EPS of its weight -> its weight)")
    both, items2 = with_env({"U_FLIGHT_WEAVE": "5", "U_FLIGHT_HEADING": "3"}, robot)
    mine = [x for s in specs for x in s[4]]
    check(both.rewards[specs[0][1]].weight == -5.0 and both.rewards[specs[1][1]].weight == -3.0
          and [i["name"] for i in items2 if i["name"] in mine] == mine
          and terms(cfg0) == terms(both, skip=(specs[0][1], specs[1][1]))
          and [i for i in items2 if i["name"] not in mine] == items0,
          "both switches: both terms, the four items, the rest unchanged")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
