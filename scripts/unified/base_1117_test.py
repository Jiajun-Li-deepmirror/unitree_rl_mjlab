#!/usr/bin/env python3
"""base_1117_test.py -- tomorrow's Go2W base from zero (go2-home 2026-11-16, the user's "可以" each): the minimal recipe +
the motor model gom8010 + terrain=wide + payload_box=tinynav (the robot table's params) + U_CMD_WIDE=1.25 +
U_DR_WIDE=1.25 (training only). That the pieces compose. No GPU; one process per build.

  (1) the robot table: the Go2W's params carry payload_box=tinynav; its typical mount box x +-0.15 (tinynav-sim's
      PAYLOAD_LIMITS); the full-load point (0.15, 0, 0.20) unchanged; the Go2's rows unchanged;
  (2) the training build with all of it: the payload box 1.25x tinynav's ((-0.19, 0.19), (-0.10, 0.10), (0, 0.25)) and
      12 kg (the minimal recipe's); stairs 0.05-0.23, slopes 0-0.36 (terrain=wide); effort 21.8 / 32.7, the joints'
      friction and damping x 0.375-1.625 and armature x 0.625-1.375 (the motor model's centres, U_DR_WIDE around 1);
      the final command ranges x 1.25 against the build without U_CMD_WIDE; the reward terms the minimal recipe's;
  (3) U_PAYLOAD_BOX=base: the box before it (the table's payload_point); a bad value stops the build;
  (4) the play builds (the teacher's map3 / Stairs / Flat, the students' Stairs) are the same with and without
      U_CMD_WIDE / U_DR_WIDE / U_RECIPE_MIN, and their payload box is the table's payload_point (not the training's).

Local: PYTHONPATH=$PWD python scripts/unified/base_1117_test.py"""

from __future__ import annotations

import json
import os
import subprocess
import sys

PROBE = r'''
import dataclasses, json, os, sys
sys.path.insert(0, os.getcwd())
import mjlab.tasks, src.tasks
from mjlab.actuator import DelayedActuatorCfg
from mjlab.tasks.registry import load_env_cfg
task, play = sys.argv[1], sys.argv[2] == "1"
c = load_env_cfg(task, play=play)
gen = c.scene.terrain.terrain_generator
subs = {k: {f: repr(getattr(s, f)) for f in ("step_height_range", "slope_range", "noise_range") if hasattr(s, f)} for k, s in gen.sub_terrains.items()}
ev = c.events
bp = ev.get("base_payload")
out = {"subs": subs, "rewards": sorted(c.rewards) if not play else None,
       "pos": repr(bp.params.get("pos_ranges")) if bp is not None else None,
       "heavy_max": bp.params.get("heavy_max") if bp is not None else None,
       "jdr": {k: [list(v.params["ranges"]), v.params["operation"]] for k, v in ev.items() if k.startswith("u_joint_")},
       "pd": repr(ev["pd_gains"].params.get("kp_range")) if "pd_gains" in ev else None,
       "cmd_final": repr(c.curriculum["u_commands"].params.get("final")) if "u_commands" in (c.curriculum or {}) else None,
       "effort": [(a.base_cfg if isinstance(a, DelayedActuatorCfg) else a).effort_limit for a in c.scene.entities["robot"].articulation.actuators],
       "events": sorted(ev),
       "push": repr(ev["push_robot"].params.get("velocity_range")) if "push_robot" in ev else None,
       "splay": repr(ev["reset_hip_splay"].params.get("splay_range")) if "reset_hip_splay" in ev else None,
       "friction": repr(ev["foot_friction"].params.get("ranges")) if "foot_friction" in ev else None,
       "push_interval": repr(ev["push_robot"].interval_range_s) if "push_robot" in ev else None}
print("JSON" + json.dumps(out))
'''
ok = True


def check(good: bool, text: str) -> None:
  global ok
  ok &= bool(good)
  print(f"  {'ok  ' if good else 'FAIL'}  {text}")


def probe(task, play=False, **env):
  e = {k: v for k, v in os.environ.items() if not k.startswith("U_")}
  e.update(UNIFIED_TASKS="1", U_ROBOTS="Go2W", PYTHONPATH=os.getcwd(), **env)
  r = subprocess.run([sys.executable, "-c", PROBE, task, "1" if play else "0"], env=e, capture_output=True, text=True)
  for line in r.stdout.splitlines():
    if line.startswith("JSON"):
      return json.loads(line[4:])
  return {"error": (r.stderr or r.stdout)[-500:]}


sys.path.insert(0, os.getcwd())
import src.robot_table as rt  # noqa: E402

print("(1) the robot table:")
gw = rt.ROBOTS["Go2W"]
check(gw["params"].get("payload_box") == "tinynav" and gw["params"].get("dr") == "wide", "the Go2W's params carry payload_box=tinynav, dr=wide")
check(gw["typical_payload"] == "4,0,0,0.10,0.15,0.08,0.10" and gw["accept_payload"] == "8,0.15,0,0.20",
      f"typical 4 kg in x +-0.15, y +-0.08, z 0-0.20; full {gw['accept_payload']} unchanged")
check(rt.ROBOTS["Go2"]["typical_payload"] == "4,0,0,0.10,0.20,0.08,0.10" and "payload_box" not in rt.ROBOTS["Go2"]["params"],
      "the Go2 unchanged")
BASE = {"U_RECIPE_MIN": "1", "U_CMD_WIDE": "1.25", "U_DR_WIDE": "1.25"}
T = "Unitree-Go2W-U-Teacher"
print("(2) the training build with all of it:")
a, m = probe(T, **BASE), probe(T, U_RECIPE_MIN="1")
if "error" in a or "error" in m:
  check(False, f"builds: {a.get('error') or m.get('error')}")
  sys.exit(1)
check(a["pos"] == repr(((-0.19, 0.19), (-0.10, 0.10), (0.0, 0.25))) and a["heavy_max"] == 12.0, f"payload box {a['pos']}, {a['heavy_max']} kg")
check(all(v.get("step_height_range") in (None, repr((0.05, 0.23))) for v in a["subs"].values())
      and a["subs"]["slope_up"]["slope_range"] == repr((0.0, 0.36)), "terrain=wide: stairs 0.05-0.23, slopes 0-0.36")
check(a["effort"] == [21.8, 21.8, 32.7, 21.8], f"the motor model's limits {a['effort']}")
fr = {k: v for k, v in a["jdr"].items() if not k.startswith("u_joint_armature")}
ar = {k: v for k, v in a["jdr"].items() if k.startswith("u_joint_armature")}
close = lambda x, y: all(abs(p - q) < 1e-9 for p, q in zip(x, y))
check(len(fr) == 6 and all(v[1] == "scale" and close(v[0], (0.375, 1.625)) for v in fr.values())
      and len(ar) == 3 and all(close(v[0], (0.625, 1.375)) for v in ar.values()),
      "joint friction / damping x 0.375-1.625, armature x 0.625-1.375 (gom8010's centres, U_DR_WIDE around 1)")
check(a["cmd_final"] != m["cmd_final"] and a["rewards"] == m["rewards"], "U_CMD_WIDE changes the final commands; the reward terms are the minimal recipe's")
want_push = repr({"x": (-1.25, 1.25), "y": (-1.25, 1.25), "z": (-0.4, 0.4), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)})
check(a["push"] == want_push and a["push_interval"] == repr((5.0, 6.0)), f"dr=wide: push x / y +-1.25 m/s, the rest and the 5-6 s as before ({a['push']})")
check(a["splay"] == repr((-0.175, 0.175)), f"dr=wide: the reset's hip splay +-0.175 rad ({a['splay']})")
pb = probe(T, **BASE, U_DR_MODEL="base")
check("error" not in pb and pb["push"] == repr({"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)})
      and pb["splay"] == repr((-0.15, 0.15)), f"U_DR_MODEL=base: the push and the splay before it ({pb.get('push')}, {pb.get('splay')})")
check(a["friction"] == repr((0.32, 1.25)), f"the training's foot friction with U_DR_WIDE: (0.32, 1.25) = 1.25x the checks' (0.4, 1.0) ({a['friction']})")
fl = probe("Unitree-Go2W-U-Teacher-Flat", True)
check("error" not in fl and fl["friction"] == repr((0.4, 1.0)), f"the teacher's -Flat check: foot friction (0.4, 1.0) as the students' (was the Go2's 0.3-1.6; {fl.get('friction')})")
sd = probe("Unitree-Go2W-U-Student-Depth", U_RECIPE_MIN="1")
check("error" not in sd and sd["push"] == want_push, "the depth student's training: the wide push too")
print("(3) the off switch:")
b = probe(T, **BASE, U_PAYLOAD_BOX="base")
check("error" not in b and b["pos"] == repr(tuple(tuple(x) for x in gw["payload_point"])), f"U_PAYLOAD_BOX=base: the table's box {b.get('pos')}")
x = probe(T, **BASE, U_PAYLOAD_BOX="tinynav")
check("error" in x and "U_PAYLOAD_BOX" in x["error"], "U_PAYLOAD_BOX=tinynav stops the build (base or unset)")
print("(4) the play builds:")
for task in (T, "Unitree-Go2W-U-Teacher-Stairs", "Unitree-Go2W-U-Teacher-Flat", "Unitree-Go2W-U-Student-Depth-Stairs"):
  p, p0 = probe(task, True, **BASE), probe(task, True)
  same = "error" not in p and p == p0
  check(same, f"{task} play: the same with and without the training switches")
  if task == T:
    check(p["pos"] == repr(tuple(tuple(x) for x in gw["payload_point"])), f"{task} play: the table's payload box {p['pos']}")
  check(p.get("push") is None, f"{task} play: no push event")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
