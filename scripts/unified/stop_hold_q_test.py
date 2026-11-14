#!/usr/bin/env python3
"""stop_hold_q_test.py -- U_STOP_HOLD_Q=1: the recipe parts that hold a robot's place at a stop on a flight come in
along the stairs quality (go2-depth 2026-11-14; constraints.quality_in, flat_only's zero_by_q, descent_overspeed's
by_q; teacher_env._stop_hold_q).

(A) quality_in on a made-up environment (CPU, no simulator): 0 at q 0 and before the curriculum's first call, half
    way at Q_IN / 2, 1 from Q_IN on.
(B) descent_overspeed at 1.35x the command: with by_q nothing at q 0, half at Q_IN / 2, as without the field from
    Q_IN on (the heavy class's share kept); without the field the same at every q (as before).
(C) flat_only with zero_terms on uneven ground: with zero_by_q at q 0 the term is what the build without the switch
    gives (the pose cost at its calm weight), from Q_IN on what zero_heavy alone gives (at zero_heavy 1 and 0), half
    way in between; nothing on even ground at any q; without the field the same at every q (as before).
(D) On the built configurations (no GPU), per robot: with U_STAND_POSE_EVEN=1,0 and U_DESCENT_SPEED=10,1.1,0,
    U_STOP_HOLD_Q=1 adds zero_by_q to u_flat_only and by_q to the overspeed cost and changes nothing else; =0 and
    unset add no key; the switch without either of its two parts, or with another value, stops the build.

Local: PYTHONPATH=$PWD python scripts/unified/stop_hold_q_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

KG = [0.0, 2.0, 4.0, 4.0 + 2.0 / 3.0, 16.0 / 3.0, 8.0]   # light, light, common, half way, heavy, full
SWITCHES = ("U_DESCENT_SPEED", "U_STAND_POSE_EVEN", "U_STOP_HOLD_Q")


def term_cases(ucons) -> list[tuple[bool, str]]:
  import torch

  import src.tasks.velocity.mdp.constraints as old_cons
  import src.tasks.velocity.mdp.rewards as old_rew
  from src.tasks.unified.curriculum import Q_IN

  out: list[tuple[bool, str]] = []

  def ok(cond, text):
    out.append((bool(cond), text))

  def close(a, b):
    return bool(torch.allclose(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32),
                               atol=1e-5))

  n = len(KG)
  kg = torch.tensor(KG)
  qs = (0.0, Q_IN / 2, Q_IN, 1.0)
  ks = (0.0, 0.5, 1.0, 1.0)

  # (A)
  got = [ucons.quality_in(SimpleNamespace(u_quality=q)) for q in qs]
  ok(close(got, ks), f"quality_in at q {list(qs)}: {got}")
  ok(ucons.quality_in(SimpleNamespace()) == 0.0, "quality_in before the curriculum's first call: 0")

  # (B) every robot nose down on uneven ground at 1.35x the command
  speed = 0.5
  cmd = torch.zeros(n, 3)
  cmd[:, 0] = speed
  grav = torch.zeros(n, 3)
  grav[:, 0], grav[:, 2] = 0.3, -0.95
  quat = torch.zeros(n, 4)
  quat[:, 0] = 1.0
  vel = torch.zeros(n, 3)
  vel[:, 0] = 1.35 * speed
  data = SimpleNamespace(root_link_quat_w=quat, root_link_lin_vel_w=vel, root_link_lin_vel_b=vel.clone(),
                         projected_gravity_b=grav)
  uneven = torch.ones(n, dtype=torch.bool)
  keep_c, keep_r = old_cons.uneven_terrain_mask, old_rew.uneven_terrain_mask
  old_cons.uneven_terrain_mask = lambda env, sensor_name, *a, **k: uneven.clone()
  old_rew.uneven_terrain_mask = lambda env, sensor_name, *a, **k: uneven.clone()
  try:
    envb = SimpleNamespace(num_envs=n, device="cpu", scene={"robot": SimpleNamespace(data=data)}, payload_kg=kg,
                           command_manager=SimpleNamespace(get_command=lambda name: cmd))
    kw = {"command_name": "twist", "sensor_name": "scan"}
    for heavy in (1.0, 0.0):
      full = ucons.descent_overspeed(envb, heavy=heavy, **kw)
      same, ramp = True, True
      for q, k in zip(qs, ks):
        envb.u_quality = q
        same &= close(ucons.descent_overspeed(envb, heavy=heavy, **kw), full)
        ramp &= close(ucons.descent_overspeed(envb, heavy=heavy, by_q=True, **kw), k * full)
      del envb.u_quality
      ok(same, f"overspeed, heavy {heavy:g}, without by_q: the same at every q (as before)")
      ok(ramp, f"overspeed, heavy {heavy:g}, by_q: x {list(ks)} at q {list(qs)}")
    ok(close(ucons.descent_overspeed(envb, by_q=True, **kw), torch.zeros(n)), "overspeed, by_q, before the curriculum's first call: 0")

    # (C) flat_only: standing_pose_load at the weight -2 (a multiplier of 2 on the calm -1), another calm term
    names = ["standing_pose_load", "standing_speed"]
    w = {"standing_pose_load": -2.0, "standing_speed": -3.0}
    calm = {"standing_pose_load": -1.0, "standing_speed": -1.5}
    raw = torch.stack([torch.full((2 * n,), 1.5), torch.full((2 * n,), 0.4)], dim=1)
    step = raw * torch.tensor([w[t] for t in names])
    un2 = torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)])   # then the same on even ground
    old_rew.uneven_terrain_mask = lambda env, sensor_name, *a, **k: un2.clone()
    rm = SimpleNamespace(_term_names=names, _step_reward=step,
                         get_term_cfg=lambda t: SimpleNamespace(weight=w[t]))
    envc = SimpleNamespace(num_envs=2 * n, device="cpu", payload_kg=torch.cat([kg, kg]), reward_manager=rm,
                           u_constraints=SimpleNamespace(calm=calm))
    term = ucons.flat_only(None, envc)
    kw = {"sensor_name": "scan", "heavy_terms": []}
    zt = {"zero_terms": ("standing_pose_load",)}
    base = term(envc, **kw)   # without the switch
    for zh in (1.0, 0.0):
      end = term(envc, zero_heavy=zh, **zt, **kw)   # zero_heavy alone
      same, ramp, even = True, True, True
      for q, k in zip(qs, ks):
        envc.u_quality = q
        same &= close(term(envc, zero_heavy=zh, **zt, **kw), end)
        got = term(envc, zero_heavy=zh, zero_by_q=True, **zt, **kw)
        ramp &= close(got, base + k * (end - base))
        even &= close(got[n:], base[n:])
      del envc.u_quality
      ok(same, f"flat_only, zero_heavy {zh:g}, without zero_by_q: the same at every q (as before)")
      ok(ramp, f"flat_only, zero_heavy {zh:g}, zero_by_q: the build without the switch at q 0, zero_heavy's from q {Q_IN} on, half way at {Q_IN / 2}")
      ok(even, f"flat_only, zero_heavy {zh:g}, zero_by_q: nothing on even ground at any q")
    ok(not close(term(envc, zero_heavy=1.0, **zt, **kw), base), "(the two ends differ: the cases above say something)")
  finally:
    old_cons.uneven_terrain_mask, old_rew.uneven_terrain_mask = keep_c, keep_r
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W,As2")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  for k in SWITCHES:
    os.environ.pop(k, None)
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

  def build(robot, **env):
    os.environ.update(env)
    try:
      return teacher_env.unified_teacher_env_cfg(robot)
    finally:
      for k in env:
        os.environ.pop(k)

  def fails(robot, **env) -> bool:
    try:
      build(robot, **env)
    except AssertionError:
      return True
    return False

  def rest(cfg, skip: dict[str, str]):
    out = {}
    for k, v in cfg.rewards.items():
      p = {kk: vv for kk, vv in v.params.items() if skip.get(k) != kk}
      out[k] = (v.weight, repr(sorted(p.items(), key=repr)))
    return out

  term = teacher_env.DESCENT_SPEED_TERM
  parts = {"U_STAND_POSE_EVEN": "1,0", "U_DESCENT_SPEED": "10,1.1,0"}
  keys = {"u_flat_only": "zero_by_q", term: "by_q"}
  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    plain = build(robot, **parts)
    pose = "standing_pose_load" in plain.rewards
    check("zero_by_q" not in plain.rewards["u_flat_only"].params and "by_q" not in plain.rewards[term].params,
          "without the switch: no zero_by_q, no by_q (as before)")
    off = build(robot, U_STOP_HOLD_Q="0", **parts)
    check(rest(off, {}) == rest(plain, {}), "U_STOP_HOLD_Q=0: the same build")
    on = build(robot, U_STOP_HOLD_Q="1", **parts)
    check(on.rewards[term].params.get("by_q") is True and on.rewards[term].weight == -10.0
          and on.rewards[term].params.get("heavy") == 0.0, "U_STOP_HOLD_Q=1: the overspeed cost by_q, its weight and heavy share kept")
    if pose:
      check(on.rewards["u_flat_only"].params.get("zero_by_q") is True
            and on.rewards["u_flat_only"].params.get("zero_heavy") == 0.0, "U_STOP_HOLD_Q=1: u_flat_only zero_by_q, zero_heavy kept")
    else:
      check("zero_by_q" not in on.rewards["u_flat_only"].params, "no standing pose cost in this build: no zero_by_q")
    check(rest(on, keys) == rest(plain, keys) and list(on.rewards) == list(plain.rewards),
          f"U_STOP_HOLD_Q=1: the other parameters of the {len(plain.rewards)} terms, and their order, as without it")
    check(sorted(on.curriculum or {}) == sorted(plain.curriculum or {}) and sorted(on.events or {}) == sorted(plain.events or {}),
          "U_STOP_HOLD_Q=1: the curriculum and event terms as without it")
    one = build(robot, U_STOP_HOLD_Q="1", U_DESCENT_SPEED="10")
    check(one.rewards[term].params.get("by_q") is True, "with the overspeed cost alone: by_q")
    table = urecipe.effective(robot)
    if not any(table.get(k) for k in parts):
      check(fails(robot, U_STOP_HOLD_Q="1"), "the switch without either of its two parts stops the build")
    else:
      print(f"        (the recipe table gives this robot {[k for k in parts if table.get(k)]}: the switch alone builds)")
    check(fails(robot, U_STOP_HOLD_Q="2", **parts) and fails(robot, U_STOP_HOLD_Q="yes", **parts),
          "another value stops the build")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
