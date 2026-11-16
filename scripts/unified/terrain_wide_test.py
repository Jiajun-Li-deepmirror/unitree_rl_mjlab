#!/usr/bin/env python3
"""terrain_wide_test.py -- the Go2W's training terrain terrain=wide and its off switch U_TERRAIN=base (go2-depth
2026-11-16; src/tasks/unified/robots.py TERRAINS / terrain_param / _training_terrain, src/robot_table.py). No GPU;
one process per build (the unified tasks register at import).

  (1) the robot table: the Go2W's params carry terrain=wide; no other robot's do;
  (2) the Go2W's training builds (the teacher, map3's tiles; the blind and the depth students' trainings): every stairs
      sub-terrain's step_height_range 0.05-0.23, the slopes' slope_range 0-0.36, random_rough's noise_range as today,
      the proportions, rows / columns and curriculum as today;
  (3) every play build (the teacher's map3 / Stairs / Flat, the students' Stairs) is today's (the code at TODAY);
  (4) U_TERRAIN=base: the training build's terrain is today's; a bad value stops the build;
  (5) the Go2 is unchanged.

Local: PYTHONPATH=$PWD python scripts/unified/terrain_wide_test.py   (TERRAIN_TODAY_DIR=<a tree of TODAY's code>, or a
git checkout / GIT_DIR for git archive)"""

from __future__ import annotations

import json
import os
import subprocess
import sys

TODAY = "ad72d70"   # the last commit before terrain=wide
PROBE = r'''
import dataclasses, json, os, sys
sys.path.insert(0, os.getcwd())
import mjlab.tasks, src.tasks
from mjlab.tasks.registry import load_env_cfg
task, play = sys.argv[1], sys.argv[2] == "1"
c = load_env_cfg(task, play=play)
gen = c.scene.terrain.terrain_generator
subs = {}
for k, s in gen.sub_terrains.items():
  subs[k] = {f.name: repr(getattr(s, f.name)) for f in dataclasses.fields(s) if f.name not in ("size",)}
print("JSON" + json.dumps({"rows": gen.num_rows, "cols": gen.num_cols, "curriculum": repr(getattr(gen, "curriculum", None)),
                           "difficulty": repr(gen.difficulty_range), "subs": subs}))
'''
ok = True
_today = None


def check(good: bool, text: str) -> None:
  global ok
  ok &= bool(good)
  print(f"  {'ok  ' if good else 'FAIL'}  {text}")


def today_dir() -> str:
  global _today
  if _today is None:
    given = os.environ.get("TERRAIN_TODAY_DIR")
    if given:
      _today = given
    else:
      import tempfile
      _today = tempfile.mkdtemp(prefix="terrain_today_")
      r = subprocess.run(f"git archive {TODAY} src jobs | tar -x -C {_today}", shell=True, capture_output=True, text=True)
      assert r.returncode == 0, f"git archive {TODAY}: {r.stderr[-200:]} (TERRAIN_TODAY_DIR or a git checkout)"
  return _today


def probe(robot, task, play=False, today=False, **env):
  root = today_dir() if today else os.getcwd()
  e = {k: v for k, v in os.environ.items() if not k.startswith("U_")}
  e.update(UNIFIED_TASKS="1", U_ROBOTS=robot, PYTHONPATH=root, **env)
  r = subprocess.run([sys.executable, "-c", PROBE, task, "1" if play else "0"], cwd=root, env=e, capture_output=True, text=True)
  for line in r.stdout.splitlines():
    if line.startswith("JSON"):
      return json.loads(line[4:])
  return {"error": (r.stderr or r.stdout)[-500:]}


sys.path.insert(0, os.getcwd())
import src.robot_table as rt  # noqa: E402

print("(1) the robot table:")
check(rt.ROBOTS["Go2W"]["params"].get("terrain") == "wide", "the Go2W's params carry terrain=wide")
check(all("terrain" not in v.get("params", {}) for k, v in rt.ROBOTS.items() if k != "Go2W"), "no other robot's do")

STAIRS = ("stairs_down_narrow", "stairs_down_wide", "stairs_up_narrow", "stairs_up_wide", "walled_down", "walled_up")
for task in ("Unitree-Go2W-U-Teacher", "Unitree-Go2W-U-Student-Blind", "Unitree-Go2W-U-Student-Depth"):
  print(f"(2) {task} training:")
  w, t0, b = probe("Go2W", task), probe("Go2W", task, today=True), probe("Go2W", task, U_TERRAIN="base")
  if any("error" in x for x in (w, t0, b)):
    check(False, f"builds: {w.get('error') or t0.get('error') or b.get('error')}")
    continue
  S = w["subs"]
  check(all(S[k]["step_height_range"] == repr((0.05, 0.23)) for k in STAIRS), "every stairs sub-terrain: step_height_range (0.05, 0.23)")
  check(all(S[k]["slope_range"] == repr((0.0, 0.36)) for k in ("slope_up", "slope_down")), "slopes: slope_range (0.0, 0.36)")
  diff = {k: {f for f in S[k] if S[k][f] != t0["subs"][k][f]} for k in S}
  check(set(S) == set(t0["subs"]) and all(d <= {"step_height_range", "slope_range"} for d in diff.values())
        and S["random_rough"] == t0["subs"]["random_rough"] and S["flat"] == t0["subs"]["flat"],
        f"everything else as today: {[(k, sorted(d)) for k, d in diff.items() if d]}")
  check({k: w[k] for k in ("rows", "cols", "curriculum", "difficulty")} == {k: t0[k] for k in ("rows", "cols", "curriculum", "difficulty")},
        f"rows / columns / curriculum / difficulty as today ({w['rows']} x {w['cols']}, curriculum {w['curriculum']})")
  check(b == t0, "(4) U_TERRAIN=base: today's training terrain")
print("(3) the play builds:")
for task in ("Unitree-Go2W-U-Teacher", "Unitree-Go2W-U-Teacher-Stairs", "Unitree-Go2W-U-Teacher-Flat", "Unitree-Go2W-U-Student-Depth-Stairs",
             "Unitree-Go2W-U-Student-Blind-Stairs"):
  p, p0 = probe("Go2W", task, True), probe("Go2W", task, True, today=True)
  check("error" not in p and p == p0, f"{task} play: today's terrain ({p.get('error', '')[:80]})")
print("(4) a bad value:")
x = probe("Go2W", "Unitree-Go2W-U-Teacher", U_TERRAIN="wide")
check("error" in x and "U_TERRAIN" in x["error"], "U_TERRAIN=wide stops the build (base or unset)")
print("(5) the Go2:")
g, g0 = probe("Go2", "Unitree-Go2-U-Teacher"), probe("Go2", "Unitree-Go2-U-Teacher", today=True)
check("error" not in g and g == g0, "the Go2's training terrain as today")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
