#!/usr/bin/env python3
"""round_cd_test.py -- go2-home's single-factor switches D (U_HARD_FIXED and its pieces), C1 (U_CMD_WIDE) and C2
(U_DR_WIDE) for the Go2W rounds after A / B (go2-depth 2026-11-16; teacher_env._round_switches,
curriculum.progress_payload's fixed).

On the built configurations (no GPU), per robot:
  (1) all off: the training build equals the plain one (the command, its curriculum, every event's parameters);
  (2) U_HARD_FIXED=1: march 0.5 with the recipe's march ramp removed, the emergency stop 0.75 with fast strafes 0.3,
      the payload limit fixed at the full one from iteration 0 (u_payload's fixed) -- nothing else changed;
  (3) each piece alone changes only its own fields; with the bundle, an empty piece turns that piece off;
  (4) U_CMD_WIDE=1.25: the final command ranges x 1.25 (the Go2W: vx -1.25..1.875, vy +-0.75, wz +-1.875), the initial
      ones unchanged;
  (5) U_DR_WIDE=1.25: foot friction (0.32, 1.25), base CoM 0.0625, kp / kd 0.8125-1.1875, lag 0-5, encoder bias
      +-0.01875, joint friction / damping tops x 1.25, armature 0.625-1.375; the push and the payload's limit and box
      unchanged;
  (6) bad values stop the build; U_FAST_STOP with U_FAST_SIDE stops it;
  (7) the play build is the same with every switch set;
  (8) progress_payload with fixed: the full limit at progress 0 (a made-up environment).

Local: PYTHONPATH=$PWD python scripts/unified/round_cd_test.py [--robots Go2W,Go2]"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from types import SimpleNamespace

SWITCHES = ("U_HARD_FIXED", "U_MARCH_FIXED", "U_FAST_STOP", "U_PAYLOAD_FULL0", "U_CMD_WIDE", "U_DR_WIDE", "U_FAST_SIDE")


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
  from src.tasks.unified import curriculum as ucur
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

  def twist(cfg):
    t = cfg.commands["twist"]
    return {f.name: repr(getattr(t, f.name)) for f in dataclasses.fields(t)}

  def events(cfg):
    return {k: repr(sorted((v.params or {}).items(), key=repr)) for k, v in cfg.events.items()}

  def cur(cfg):
    return {k: repr(sorted((v.params or {}).items(), key=repr)) for k, v in cfg.curriculum.items()}

  def march_ramps(cfg):
    return [r for r in cfg.curriculum["ramps"].params["ramps"] if r.get("key") == "march_prob"] \
      if "ramps" in cfg.curriculum else []

  def diff(c0, c1):
    t0, t1, e0, e1, u0, u1 = twist(c0), twist(c1), events(c0), events(c1), cur(c0), cur(c1)
    return ({k for k in t0 if t0[k] != t1[k]}, {k for k in e0 if e0[k] != e1[k]}, {k for k in u0 if u0[k] != u1[k]})

  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    c0 = build(robot)
    t0 = c0.commands["twist"]
    check(not c0.curriculum["u_payload"].params.get("fixed", False),
          f"plain: march {t0.march_prob} (ramp {[(r['start'], r['end']) for r in march_ramps(c0)]}), fast stop "
          f"{t0.stop_after_fast_prob}, fast side {t0.fast_side_prob}, payload ramped")
    # (1)
    c1 = build(robot, {s: "" for s in SWITCHES})
    check(diff(c0, c1) == (set(), set(), set()), "all switches empty: the same build")
    # (2)
    cd = build(robot, {"U_HARD_FIXED": "1"})
    td = cd.commands["twist"]
    dt, de, dc = diff(c0, cd)
    check(td.march_prob == 0.5 and not march_ramps(cd), "U_HARD_FIXED=1: march 0.5, its ramp removed")
    check(td.stop_after_fast_prob == 0.75 and td.fast_side_prob == 0.3, "U_HARD_FIXED=1: fast stop 0.75, fast side 0.3")
    check(cd.curriculum["u_payload"].params.get("fixed") is True, "U_HARD_FIXED=1: the payload limit fixed from it 0")
    check(dt <= {"march_prob", "stop_after_fast_prob", "fast_side_prob"} and not de
          and dc <= {"ramps", "u_payload"}, f"U_HARD_FIXED=1: nothing else changed (twist {sorted(dt)}, curriculum {sorted(dc)})")
    check(td.flat_mode_probs == t0.flat_mode_probs and td.stair_standing_prob == t0.stair_standing_prob,
          f"the zero command {t0.flat_mode_probs[0]} and the flight stop {t0.stair_standing_prob} unchanged (fixed already)")
    # (3)
    cm = build(robot, {"U_MARCH_FIXED": "0.5"})
    check(diff(c0, cm)[0] <= {"march_prob"} and not diff(c0, cm)[1] and diff(c0, cm)[2] <= {"ramps"}
          and cm.commands["twist"].march_prob == 0.5, "U_MARCH_FIXED=0.5 alone: only the march")
    cf = build(robot, {"U_FAST_STOP": "0.75,0.3"})
    check(diff(c0, cf)[0] == {"stop_after_fast_prob", "fast_side_prob"} and not diff(c0, cf)[2],
          "U_FAST_STOP=0.75,0.3 alone: only the two shares")
    cp = build(robot, {"U_PAYLOAD_FULL0": "1"})
    check(not diff(c0, cp)[0] and diff(c0, cp)[2] == {"u_payload"}, "U_PAYLOAD_FULL0=1 alone: only the payload term")
    cb = build(robot, {"U_HARD_FIXED": "1", "U_FAST_STOP": ""})
    check(cb.commands["twist"].stop_after_fast_prob == t0.stop_after_fast_prob and cb.commands["twist"].march_prob == 0.5,
          "the bundle with U_FAST_STOP= (empty): no fast stop, the rest on")
    # (4)
    cw = build(robot, {"U_CMD_WIDE": "1.25"})
    f0, f1 = c0.curriculum["u_commands"].params["final"], cw.curriculum["u_commands"].params["final"]
    want = {k: (None if r is None else (r[0] * 1.25, r[1] * 1.25)) for k, r in f0.items()}
    check(f1 == want and cw.curriculum["u_commands"].params["initial"] == c0.curriculum["u_commands"].params["initial"],
          f"U_CMD_WIDE=1.25: final {f1}, initial unchanged")
    check(diff(c0, cw)[2] == {"u_commands"} and not diff(c0, cw)[1], "U_CMD_WIDE: nothing else changed")
    # (5)
    cr = build(robot, {"U_DR_WIDE": "1.25"})
    e0, e1 = c0.events, cr.events
    fr0, fr1 = e0["foot_friction"].params["ranges"], e1["foot_friction"].params["ranges"]
    check(abs(fr1[0] - fr0[0] / 1.25) < 1e-9 and abs(fr1[1] - fr0[1] * 1.25) < 1e-9, f"friction {fr0} -> {fr1}")
    check(abs(e1["base_payload"].params["base_com_range"] - e0["base_payload"].params["base_com_range"] * 1.25) < 1e-9,
          f"base CoM {e0['base_payload'].params['base_com_range']} -> {e1['base_payload'].params['base_com_range']}")
    kp0, kp1 = e0["pd_gains"].params["kp_range"], e1["pd_gains"].params["kp_range"]
    check(abs(kp1[0] - (1 - (1 - kp0[0]) * 1.25)) < 1e-9 and abs(kp1[1] - (1 + (kp0[1] - 1) * 1.25)) < 1e-9,
          f"kp {kp0} -> {kp1} (kd the same way)")
    check(e1["actuator_delay"].params["lag_range"][1] == round(e0["actuator_delay"].params["lag_range"][1] * 1.25),
          f"lag {e0['actuator_delay'].params['lag_range']} -> {e1['actuator_delay'].params['lag_range']}")
    b0, b1 = e0["encoder_bias"].params["bias_range"], e1["encoder_bias"].params["bias_range"]
    check(abs(b1[1] - b0[1] * 1.25) < 1e-12, f"encoder bias {b0} -> {b1}")
    jf = [k for k in e0 if k.startswith(("u_joint_friction_", "u_joint_damping_"))]
    ja = [k for k in e0 if k.startswith("u_joint_armature_")]
    check(jf and all(abs(e1[k].params["ranges"][1] - e0[k].params["ranges"][1] * 1.25) < 1e-9 for k in jf),
          f"{len(jf)} joint friction / damping tops x 1.25")
    check(ja and all(abs(e1[k].params["ranges"][0] - (1 - (1 - e0[k].params["ranges"][0]) * 1.25)) < 1e-9 for k in ja),
          f"{len(ja)} armature ranges {e0[ja[0]].params['ranges']} -> {e1[ja[0]].params['ranges']}")
    p0 = {k: v for k, v in e0["base_payload"].params.items() if k != "base_com_range"}
    p1 = {k: v for k, v in e1["base_payload"].params.items() if k != "base_com_range"}
    check(repr(e0["push_robot"].params) == repr(e1["push_robot"].params) and repr(p0) == repr(p1),
          "the push and the payload's limit and box unchanged")
    check(not diff(c0, cr)[0] and not diff(c0, cr)[2], "U_DR_WIDE: the command and the curriculum unchanged")
    # (6)
    bad = [{"U_HARD_FIXED": "2"}, {"U_MARCH_FIXED": "0"}, {"U_MARCH_FIXED": "0.9"}, {"U_FAST_STOP": "0.75"},
           {"U_FAST_STOP": "1.5,0.3"}, {"U_FAST_STOP": "0.75,0.6"}, {"U_PAYLOAD_FULL0": "2"},
           {"U_FAST_STOP": "0.75,0.3", "U_FAST_SIDE": "0.1"}, {"U_CMD_WIDE": "1"}, {"U_CMD_WIDE": "3"},
           {"U_DR_WIDE": "0.9"}, {"U_DR_WIDE": "2.5"}]
    check(all(fails(robot, e) for e in bad), f"{len(bad)} bad settings stop the build")
    # (7)
    allon = {"U_HARD_FIXED": "1", "U_CMD_WIDE": "1.25", "U_DR_WIDE": "1.25"}
    q0, q1 = build(robot, play=True), build(robot, allon, play=True)
    check(diff(q0, q1) == (set(), set(), set()), "the play build is the same with every switch set")
  # (8)
  ev = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(params={}))
  env = SimpleNamespace(u_progress=0.0, u_quality=0.0, event_manager=ev)
  out_f = ucur.progress_payload(env, None, "base_payload", 8.0, fixed=True)
  out_r = ucur.progress_payload(env, None, "base_payload", 8.0)
  check(float(out_f["heavy_max"]) == 8.0 and float(out_r["heavy_max"]) == 0.0,
        f"progress_payload at progress 0: fixed {float(out_f['heavy_max'])}, ramped {float(out_r['heavy_max'])}")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
