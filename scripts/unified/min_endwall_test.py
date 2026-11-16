#!/usr/bin/env python3
"""min_endwall_test.py -- the next Go2W candidate's switches together (go2-depth 2026-11-16, go2-home's round-3
candidate: the minimal recipe + the motor model + the end wall): U_RECIPE_MIN=1 U_KEYS_ADD=endwall,endwallshort
U_WALL_FAN=5 (with U_ROBOTS=Go2W: the end-wall keys need longexit, which only the Go2W's recipe has). No GPU.

Against the minimal recipe alone (U_RECIPE_MIN=1), the training build:
  (1) the end wall: the walled up-flights' exit 1.4 m (longexit, the recipe's) and a 2 m end wall; the fan wall_fan
      (3 rays at -5, 0, 5 deg, 0.8 m up, 3 m reach); the twist command reads it (stop at the table's 0.85 m, the
      slow-down 0.2 m) and is still the hard actions' command (HardShapeCommandCfg, U_HARD_FIXED);
  (2) the keys add no reward term: the reward terms and weights are the minimal recipe's (none deleted by it either);
      the observations, terminations, events and curricula the same;
  (3) the motor model on (effort 21.8 / 32.7);
  (4) without U_ROBOTS=Go2W the build refuses (the Go2 has no longexit), as the switch's check says;
  (5) the play builds of the checks' terrains (stairs, flat) are the minimal recipe's (no fan); map3's play build has
      the fan and the command's wall stop as training does -- so a map3 evaluation of such a run includes the stop.

Local: PYTHONPATH=$PWD python scripts/unified/min_endwall_test.py"""

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
out = {}
gen = c.scene.terrain.terrain_generator if c.scene.terrain is not None else None
wu = gen.sub_terrains.get("walled_up") if gen is not None else None
out["walled_up"] = None if wu is None else {"exit_length": getattr(wu, "exit_length", None), "end_wall_height": getattr(wu, "end_wall_height", None)}
fan = next((s for s in c.scene.sensors if s.name == "wall_fan"), None)
out["fan"] = None if fan is None else {"angles": list(fan.pattern.angles_deg), "z": fan.pattern.z_offset, "reach": fan.max_distance}
tw = c.commands["twist"]
out["twist"] = {"type": type(tw).__name__, "wall": getattr(tw, "wall_sensor_name", None), "stop": getattr(tw, "end_wall_stop_m", None),
                "slow": getattr(tw, "end_wall_slow_m", None), "march": tw.march_prob, "fast_stop": tw.stop_after_fast_prob}
out["rewards"] = {k: v.weight for k, v in c.rewards.items()}
out["obs"] = {g: sorted(getattr(c.observations[g], "terms", {}).keys()) for g in c.observations}
out["terms"] = sorted(c.terminations)
out["events"] = sorted(c.events)
out["curriculum"] = sorted(c.curriculum)
acts = [a.base_cfg if isinstance(a, DelayedActuatorCfg) else a for a in c.scene.entities["robot"].articulation.actuators]
out["effort"] = [a.effort_limit for a in acts]
print("JSON" + json.dumps(out))
'''

ok = True


def check(good: bool, text: str) -> None:
  global ok
  ok &= bool(good)
  print(f"  {'ok  ' if good else 'FAIL'}  {text}")


def probe(task, play=False, **env):
  e = {k: v for k, v in os.environ.items() if not k.startswith("U_")}
  e.update(UNIFIED_TASKS="1", PYTHONPATH=os.getcwd(), **env)
  r = subprocess.run([sys.executable, "-c", PROBE, task, "1" if play else "0"], env=e, capture_output=True, text=True)
  for line in r.stdout.splitlines():
    if line.startswith("JSON"):
      return json.loads(line[4:])
  return {"error": (r.stderr or r.stdout)[-500:]}


T = "Unitree-Go2W-U-Teacher"
MIN = {"U_ROBOTS": "Go2W", "U_RECIPE_MIN": "1"}
WALL = {**MIN, "U_KEYS_ADD": "endwall,endwallshort", "U_WALL_FAN": "5"}
m = probe(T, **MIN)
w = probe(T, **WALL)
if "error" in m or "error" in w:
  print(m.get("error") or w.get("error"))
  check(False, "builds")
  sys.exit(1)
print("(1) the end wall:")
check(m["walled_up"]["exit_length"] == 1.4 and w["walled_up"] == {"exit_length": 1.4, "end_wall_height": 2.0},
      f"walled up-flights: min {m['walled_up']}, with the keys {w['walled_up']}")
check(m["fan"] is None and w["fan"] == {"angles": [-5.0, 0.0, 5.0], "z": 0.8, "reach": 3.0}, f"the fan: {w['fan']}")
check(w["twist"]["type"] == "HardShapeCommandCfg" and w["twist"]["wall"] == "wall_fan" and abs(w["twist"]["stop"] - 0.85) < 1e-9
      and w["twist"]["slow"] == 0.2 and {k: v for k, v in w["twist"].items() if k not in ("wall", "stop", "slow")}
      == {k: v for k, v in m["twist"].items() if k not in ("wall", "stop", "slow")},
      f"the command: {w['twist']}")
print("(2) the rest as the minimal recipe:")
check(w["rewards"] == m["rewards"], f"the reward terms and weights ({len(w['rewards'])})")
check(w["obs"] == m["obs"] and w["terms"] == m["terms"] and w["events"] == m["events"] and w["curriculum"] == m["curriculum"],
      "observations, terminations, events, curricula")
print("(3) the motor model:")
check(w["effort"] == [21.8, 21.8, 32.7, 21.8], f"effort {w['effort']}")
print("(4) without U_ROBOTS=Go2W:")
x = probe(T, **{k: v for k, v in WALL.items() if k != "U_ROBOTS"})
check("error" in x and "longexit" in x["error"], "the build refuses (the Go2's recipe has no longexit)")
print("(5) the play builds:")
for task in ("Unitree-Go2W-U-Teacher-Stairs", "Unitree-Go2W-U-Teacher-Flat"):
  p0, p1 = probe(task, True, **MIN), probe(task, True, **WALL)
  check("error" not in p1 and p1 == p0, f"{task} play (the checks' terrains): as the minimal recipe's play build, no fan")
p1 = probe(T, True, **WALL)
check("error" not in p1 and p1["fan"] == w["fan"] and p1["twist"]["wall"] == "wall_fan" and abs(p1["twist"]["stop"] - 0.85) < 1e-9,
      f"{T} play (map3's tiles, the training terrain): the fan and the command's wall stop as in training -- every "
      f"map3 evaluation of such a run (eval_map3, map3_stairs_probe) has the command cut 0.85 m before a wall ahead")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
