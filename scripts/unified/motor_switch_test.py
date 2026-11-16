#!/usr/bin/env python3
"""motor_switch_test.py -- the Go2W motor model gom8010 and its off switch U_MOTOR=builtin (go2-depth 2026-11-16;
src/assets/robots/uros.py MOTORS / motor / spec_fn, src/robot_table.py motor=gom8010, teacher_env._add_joint_dr).
No GPU. Each condition runs in its own process (the unified tasks resolve the robot at registration).

  (1) the robot table: the Go2W's params carry motor=gom8010; no other robot's do;
  (2) the Go2W's unified teacher builds (stairs, flat, map3; training and play, the blind student): leg effort limits
      21.8 / 21.8 / 32.7, wheel 21.8; leg friction loss 0.2, armature 0.01; the gains as before; the compiled model:
      leg joints' damping 0.1, friction loss 0.2, armature 0.01, the leg actuators' force range +-21.8 / 32.7;
  (3) the joint randomization: friction and damping x 0.5-1.5, armature x 0.7-1.3 (scale), in training and play alike;
  (4) U_MOTOR=builtin: the Go2W build is the build before the motor model (effort 23.7 / 35.55 / 23.7, armature
      0.01 / 0.01 / 0.02, no joint friction or damping, the randomization from 0) -- equal field by field to today's
      build (TODAY's code, git archive into a temporary directory);
  (5) U_MODEL=legacy: no motor model; a bad U_MOTOR stops the build;
  (6) the Go2 is unchanged by the motor model (its build with and without U_MOTOR=builtin is the same).

Local: PYTHONPATH=$PWD python scripts/unified/motor_switch_test.py
A tree without git (a box): GIT_DIR=<a clone>/.git, or MOTOR_TODAY_DIR=<a tree of 4d25c78's code>."""

from __future__ import annotations

import json
import os
import subprocess
import sys

PROBE = r'''
import dataclasses, json, os, sys
sys.path.insert(0, os.getcwd())
robot, task, play = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
import mjlab.tasks, src.tasks
from mjlab.actuator import DelayedActuatorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.entity import Entity
cfg = load_env_cfg(task, play=play)
ent = cfg.scene.entities["robot"]
acts = [a.base_cfg if isinstance(a, DelayedActuatorCfg) else a for a in ent.articulation.actuators]
out = {"acts": [{"type": type(a).__name__, "expr": a.target_names_expr[0],
                 **{f.name: getattr(a, f.name) for f in dataclasses.fields(a) if f.init and f.name not in ("target_names_expr", "transmission_type")}} for a in acts]}
out["dr"] = {k: [list(v.params["ranges"]), v.params["operation"]] for k, v in cfg.events.items() if k.startswith("u_joint_")}
m = Entity(ent).spec.compile()
import mujoco
j = {}
for i in range(m.njnt):
  n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
  if n and any(g in n for g in ("hip", "thigh", "calf", "wheel")):
    d = m.jnt_dofadr[i]
    j[n] = [float(m.dof_damping[d]), float(m.dof_frictionloss[d]), float(m.dof_armature[d])]
out["joints"] = j
f = {}
for i in range(m.nu):
  n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
  f[n] = [float(m.actuator_forcerange[i][0]), float(m.actuator_forcerange[i][1])]
out["force"] = f
print("JSON" + json.dumps(out))
'''

ok = True


def check(good: bool, text: str) -> None:
  global ok
  ok &= bool(good)
  print(f"  {'ok  ' if good else 'FAIL'}  {text}")


TODAY = "4d25c78"   # the last commit before the motor model
_today_dir = None


def today_dir() -> str:
  """Today's code: MOTOR_TODAY_DIR (a tree of TODAY's code, e.g. a deployed one on a box without git), else TODAY's
  src / jobs by git archive (a git checkout, or GIT_DIR) in a temporary directory."""
  global _today_dir
  if _today_dir is None:
    import tempfile

    given = os.environ.get("MOTOR_TODAY_DIR")
    if given:
      assert os.path.isdir(os.path.join(given, "src")), f"MOTOR_TODAY_DIR={given}: no src/ there"
      _today_dir = given
    else:
      _today_dir = tempfile.mkdtemp(prefix="motor_today_")
      r = subprocess.run(f"git archive {TODAY} src jobs | tar -x -C {_today_dir}", shell=True, capture_output=True, text=True)
      assert r.returncode == 0 and os.path.isdir(os.path.join(_today_dir, "src")), (
        f"git archive {TODAY} failed ({r.stderr.strip()[-200:]}): run in a git checkout, set GIT_DIR, or set "
        f"MOTOR_TODAY_DIR to a tree of {TODAY}'s code")
  return _today_dir


def probe(robot, task, play=False, today=False, **env):
  root = today_dir() if today else os.getcwd()
  e = {k: v for k, v in os.environ.items() if not k.startswith("U_")}
  e.update(UNIFIED_TASKS="1", U_ROBOTS=robot, PYTHONPATH=root, **env)
  r = subprocess.run([sys.executable, "-c", PROBE, robot, task, "1" if play else "0"], cwd=root,
                     env=e, capture_output=True, text=True)
  for line in r.stdout.splitlines():
    if line.startswith("JSON"):
      return json.loads(line[4:])
  return {"error": (r.stderr or r.stdout)[-600:]}


def group(expr):
  return next((g for g in ("wheel", "calf", "thigh", "hip") if g in expr), None)


sys.path.insert(0, os.getcwd())
import src.robot_table as rt  # noqa: E402

print("(1) the robot table:")
check(rt.ROBOTS["Go2W"]["params"].get("motor") == "gom8010", "the Go2W's params carry motor=gom8010")
check(all("motor" not in v.get("params", {}) for k, v in rt.ROBOTS.items() if k != "Go2W"), "no other robot's do")

LIM = {"hip": 21.8, "thigh": 21.8, "calf": 32.7, "wheel": 21.8}
OLD = {"hip": 23.7, "thigh": 23.7, "calf": 35.55, "wheel": 23.7}
OLD_ARM = {"hip": 0.01, "thigh": 0.01, "calf": 0.02}
builds = [("Unitree-Go2W-U-Teacher-Stairs", False), ("Unitree-Go2W-U-Teacher-Stairs", True),
          ("Unitree-Go2W-U-Teacher-Flat", False), ("Unitree-Go2W-U-Teacher-Flat", True),
          ("Unitree-Go2W-U-Teacher", False), ("Unitree-Go2W-U-Teacher", True)]
got = {}
for task, play in builds:
  print(f"(2)-(4) {task} {'play' if play else 'training'}:")
  g = probe("Go2W", task, play)
  b = probe("Go2W", task, play, U_MOTOR="builtin")
  n = probe("Go2W", task, play, today=True)
  if "error" in g or "error" in b or "error" in n:
    check(False, f"builds: {g.get('error') or b.get('error') or n.get('error')}")
    continue
  got[(task, play)] = g
  a = {group(x["expr"]): x for x in g["acts"]}
  check(all(abs(a[k]["effort_limit"] - LIM[k]) < 1e-9 for k in LIM), f"effort limits {[(k, a[k]['effort_limit']) for k in LIM]}")
  check(all(a[k]["frictionloss"] == 0.2 and a[k]["armature"] == 0.01 for k in ("hip", "thigh", "calf")),
        "leg actuators: friction loss 0.2, armature 0.01")
  check([(x["stiffness"], x["damping"]) for x in g["acts"]] == [(x["stiffness"], x["damping"]) for x in n["acts"]],
        f"the gains as today {[(x['stiffness'], x['damping']) for x in g['acts']]}")
  legs = {k: v for k, v in g["joints"].items() if "wheel" not in k}
  check(len(legs) == 12 and all(abs(v[0] - 0.1) < 1e-9 and abs(v[1] - 0.2) < 1e-9 and abs(v[2] - 0.01) < 1e-9 for v in legs.values()),
        f"compiled: 12 leg joints damping 0.1, friction loss 0.2, armature 0.01 (FL_calf {legs.get('FL_calf_joint')})")
  wh = {k: v for k, v in g["joints"].items() if "wheel" in k}
  check(wh == {k: v for k, v in n["joints"].items() if "wheel" in k}, f"compiled: the wheel joints as today {list(wh.values())[:1]}")
  fr = {group(k): v for k, v in g["force"].items()}
  check(all(abs(fr[k][1] - LIM[k]) < 1e-6 and abs(fr[k][0] + LIM[k]) < 1e-6 for k in ("hip", "thigh", "calf")),
        f"compiled: leg force ranges {[(k, fr[k]) for k in ('hip', 'thigh', 'calf')]}")
  dr = g["dr"]
  want = {"friction": [[0.5, 1.5], "scale"], "damping": [[0.5, 1.5], "scale"], "armature": [[0.7, 1.3], "scale"]}
  check(len(dr) == 9 and all(v == want[k.split("_")[2]] for k, v in dr.items()),
        f"(3) the joint randomization: friction / damping x 0.5-1.5, armature x 0.7-1.3 ({len(dr)} events)")
  check(b == n, f"(4) U_MOTOR=builtin == today's build ({TODAY}: actuators, compiled joints, force ranges, randomization)")
  ab = {group(x["expr"]): x for x in b["acts"]}
  check(all(abs(ab[k]["effort_limit"] - OLD[k]) < 1e-9 for k in OLD) and all(ab[k]["armature"] == OLD_ARM[k] and ab[k]["frictionloss"] == 0.0 for k in OLD_ARM)
        and all(v[0] == 0.0 and v[1] == 0.0 for k, v in b["joints"].items() if "wheel" not in k)
        and all(v[1] == "abs" and v[0][0] == 0.0 for k, v in b["dr"].items() if not k.startswith("u_joint_armature")),
        "(4) U_MOTOR=builtin: 23.7 / 35.55 / 23.7, armature 0.01 / 0.01 / 0.02, no joint friction or damping, randomization from 0")
for task in ("Unitree-Go2W-U-Teacher-Stairs", "Unitree-Go2W-U-Teacher-Flat", "Unitree-Go2W-U-Teacher"):
  if (task, False) in got and (task, True) in got:
    tr, pl = got[(task, False)], got[(task, True)]
    check(tr["acts"] == pl["acts"] and tr["joints"] == pl["joints"] and tr["force"] == pl["force"] and tr["dr"] == pl["dr"],
          f"play == training: {task}")
print("(2) the blind student:")
s = probe("Go2W", "Unitree-Go2W-U-Student-Blind", False)
check("error" not in s and {group(x["expr"]): x["effort_limit"] for x in s.get("acts", [])} == LIM, f"limits {s.get('error') or [x['effort_limit'] for x in s['acts']]}")
print("(5):")
l = probe("Go2W", "Unitree-Go2W-U-Teacher-Flat", True, U_MODEL="legacy")
check("error" not in l and all(x["effort_limit"] not in (21.8, 32.7) for x in l.get("acts", [])) and
      all(v[0] == 0.0 for k, v in l.get("joints", {}).items()), "U_MODEL=legacy: no motor model")
x = probe("Go2W", "Unitree-Go2W-U-Teacher-Flat", True, U_MOTOR="pd")
check("error" in x and "U_MOTOR" in x["error"], "U_MOTOR=pd stops the build")
print("(6) the Go2:")
g0 = probe("Go2", "Unitree-Go2-U-Teacher-Stairs", False)
g1 = probe("Go2", "Unitree-Go2-U-Teacher-Stairs", False, U_MOTOR="builtin")
g2 = probe("Go2", "Unitree-Go2-U-Teacher-Stairs", False, today=True)
check("error" not in g0 and g0 == g1 == g2 and {group(x["expr"]): x["effort_limit"] for x in g0["acts"]} == {"hip": 23.7, "thigh": 23.7, "calf": 45.43},
      "unchanged: 23.7 / 23.7 / 45.43, the same with U_MOTOR=builtin and as today")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
