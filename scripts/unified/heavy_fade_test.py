#!/usr/bin/env python3
"""heavy_fade_test.py -- the heavy class's share of two recipe parts (go2-depth 2026-11-14; constraints.payload_fade):
U_STAND_POSE_EVEN=1,<heavy> (constraints.flat_only's zero_heavy) and U_DESCENT_SPEED=<w>,<over>,<heavy>
(constraints.descent_overspeed's heavy).

(A) payload_fade on a made-up environment (CPU, no simulator), the limit 8 kg: 0 / 2 / 4 kg in full, 4.67 kg half way,
    5.33 / 8 kg at <heavy>; 1.0 at heavy 1 and before the first payload draw.
(B) descent_overspeed at 1.35x the command (0.25 without the fade): times that share with heavy 0; as before without.
(C) flat_only with zero_terms on uneven ground: at zero_heavy 1 the term is gone for every robot (as before the
    field); at 0 the robots up to the common payload lose it, the heavy class has it at its calm weight (what the
    build without the switch gives), half way in between; nothing on even ground; another calm term as before.
(D) On the built configurations (no GPU), per robot:
    U_STAND_POSE_EVEN=1 and =1,1: u_flat_only's parameters as before this field (no zero_heavy key);
    =1,0 and =1,0.5: zero_heavy as written, every other reward term and parameter as with =1;
    U_DESCENT_SPEED=10 and =10,1.1: no heavy key; =10,1.1,0: heavy 0, the rest as with =10;
    wrong values stop the build.

Local: PYTHONPATH=$PWD python scripts/unified/heavy_fade_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

KG = [0.0, 2.0, 4.0, 4.0 + 2.0 / 3.0, 16.0 / 3.0, 8.0]   # light, light, common, half way, heavy, full
SHARE0 = [1.0, 1.0, 1.0, 0.5, 0.0, 0.0]                  # payload_fade at heavy 0


def term_cases(ucons) -> list[tuple[bool, str]]:
  import torch

  import src.tasks.velocity.mdp.constraints as old_cons
  import src.tasks.velocity.mdp.rewards as old_rew

  out: list[tuple[bool, str]] = []

  def ok(cond, text):
    out.append((bool(cond), text))

  n = len(KG)
  kg = torch.tensor(KG)
  want0 = torch.tensor(SHARE0)

  def close(a, b):
    return bool(torch.allclose(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32),
                               atol=1e-5))

  # (A)
  env = SimpleNamespace(num_envs=n, device="cpu", payload_kg=kg)
  ok(close(ucons.payload_fade(env, 0.0), want0), f"payload_fade, heavy 0: {SHARE0} at {[round(k, 2) for k in KG]} kg")
  ok(close(ucons.payload_fade(env, 0.25), 0.25 + 0.75 * want0), "payload_fade, heavy 0.25: the heavy class at a quarter")
  ok(ucons.payload_fade(env, 1.0) == 1.0, "payload_fade, heavy 1: 1 for every robot")
  ok(ucons.payload_fade(SimpleNamespace(num_envs=n, device="cpu"), 0.0) == 1.0, "payload_fade before the first draw: 1")
  env16 = SimpleNamespace(num_envs=n, device="cpu", payload_kg=2 * kg, u_payload_full=16.0)
  ok(close(ucons.payload_fade(env16, 0.0), want0), "payload_fade follows the robot's payload limit (16 kg: the same shares)")

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
    plain = ucons.descent_overspeed(envb, command_name="twist", sensor_name="scan")
    ok(close(plain, torch.full((n,), 0.25)), "overspeed without the field: 0.25 on every payload (as before)")
    faded = ucons.descent_overspeed(envb, command_name="twist", sensor_name="scan", heavy=0.0)
    ok(close(faded, 0.25 * want0), f"overspeed, heavy 0: {[round(float(x), 3) for x in faded]}")
    half = ucons.descent_overspeed(envb, command_name="twist", sensor_name="scan", heavy=0.5)
    ok(close(half, 0.25 * (0.5 + 0.5 * want0)), "overspeed, heavy 0.5: the heavy class at half")

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
    other = un2.float() * 0.4 * (calm["standing_speed"] - w["standing_speed"])   # the other calm term's part
    base = term(envc, **kw)                                                       # without the switch
    ok(close(base, un2.float() * 1.5 * (-1.0 + 2.0) + other), "flat_only without the switch: the pose cost at its calm weight on uneven ground")
    all_ = term(envc, zero_terms=("standing_pose_load",), **kw)
    ok(close(all_, un2.float() * 1.5 * 2.0 + other), "zero_terms without zero_heavy: the pose cost gone for every robot (as before)")
    one = term(envc, zero_terms=("standing_pose_load",), zero_heavy=1.0, **kw)
    ok(close(one, all_), "zero_heavy 1: the same")
    z0 = term(envc, zero_terms=("standing_pose_load",), zero_heavy=0.0, **kw)
    gone = torch.cat([want0, want0])
    total = step[:, 0] + z0 - other                                               # the pose cost with the term's part
    ok(close(total[:n], (1.0 - want0) * 1.5 * calm["standing_pose_load"]),
       f"zero_heavy 0: the pose cost per payload {[round(float(x), 2) for x in total[:n]]} (calm: -1.5; gone up to the common payload)")
    ok(close(z0[n:], other[n:]) and close(z0[4:n], base[4:n]),
       "zero_heavy 0: nothing on even ground; the heavy class as without the switch")
    zh = term(envc, zero_terms=("standing_pose_load",), zero_heavy=0.5, **kw)
    g = 0.5 + 0.5 * gone
    ok(close(step[:, 0] + zh - other, un2.float() * (1.0 - g) * 1.5 * calm["standing_pose_load"] + (~un2).float() * step[:, 0]),
       "zero_heavy 0.5: half the calm cost left in the heavy class")
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
  for k in ("U_DESCENT_SPEED", "U_STAND_POSE_EVEN"):
    os.environ.pop(k, None)
  sys.path.insert(0, os.getcwd())
  from src.tasks.unified import constraints as ucons
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

  def rest(cfg, skip: str, key: str):
    out = {}
    for k, v in cfg.rewards.items():
      p = {kk: vv for kk, vv in v.params.items() if not (k == skip and kk == key)}
      out[k] = (v.weight, repr(sorted(p.items(), key=repr)))
    return out

  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    one = build(robot, U_STAND_POSE_EVEN="1")
    p1 = one.rewards["u_flat_only"].params
    if "standing_pose_load" not in one.rewards:
      check("zero_terms" not in p1, "no standing pose cost in this build: the switch adds nothing")
    else:
      check(p1.get("zero_terms") == ("standing_pose_load",) and "zero_heavy" not in p1,
            "U_STAND_POSE_EVEN=1: the pose cost off the uneven ground, no zero_heavy key (as before)")
      check(build(robot, U_STAND_POSE_EVEN="1,1").rewards["u_flat_only"].params == p1, "=1,1: the same parameters")
      for v, want in (("1,0", 0.0), ("1,0.5", 0.5)):
        cfg = build(robot, U_STAND_POSE_EVEN=v)
        check(cfg.rewards["u_flat_only"].params.get("zero_heavy") == want
              and rest(cfg, "u_flat_only", "zero_heavy") == rest(one, "u_flat_only", "zero_heavy"),
              f"={v}: zero_heavy {want}, the other {len(one.rewards) - 1} terms and parameters as with =1")
      off = build(robot, U_STAND_POSE_EVEN="0")
      check("zero_terms" not in off.rewards["u_flat_only"].params or n in ("As2",), "=0: off")
      check(fails(robot, U_STAND_POSE_EVEN="2") and fails(robot, U_STAND_POSE_EVEN="1,1.5")
            and fails(robot, U_STAND_POSE_EVEN="1,0,0"), "wrong values stop the build")
    term = teacher_env.DESCENT_SPEED_TERM
    d1 = build(robot, U_DESCENT_SPEED="10")
    check("heavy" not in d1.rewards[term].params
          and build(robot, U_DESCENT_SPEED="10,1.1").rewards[term].params == d1.rewards[term].params,
          "U_DESCENT_SPEED=10 and =10,1.1: no heavy key (as before)")
    d0 = build(robot, U_DESCENT_SPEED="10,1.1,0")
    check(d0.rewards[term].params.get("heavy") == 0.0 and d0.rewards[term].weight == -10.0
          and rest(d0, term, "heavy") == rest(d1, term, "heavy"),
          "=10,1.1,0: heavy 0, the rest as with =10")
    check(fails(robot, U_DESCENT_SPEED="10,1.1,2") and fails(robot, U_DESCENT_SPEED="10,1.1,0,0"),
          "wrong values stop the build")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
