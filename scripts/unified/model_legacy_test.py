#!/usr/bin/env python3
"""model_legacy_test.py -- U_MODEL=legacy: the unified checks on the robot model the old flow's policies trained on
(go2-depth 2026-11-15; robots.base_env_cfg, robots.MODEL_SWITCH). Every robot's table entry has model=uros since
2026-11-11 (src/assets/robots/uros.py); the policies deployed before that, converted for the unified checks
(baseline_ckpt.py), ran on a robot they never trained on: the delivered Go2W blind g14 at 8 kg up map3 fell 477 times
of 896, 326 of them on the first step.

On the built configurations (no GPU), per robot:
  (1) without the switch: the aligned model -- the model file <r>_uros.xml, the actuators' limits those of
      uros.ACTUATORS;
  (2) U_MODEL=legacy: the model file and the actuators as the robot's constants have them (no BLIND_PARAMS), on the
      teacher's training build, its three check builds (play: map3, stairs, flat) and the students' builds;
  (3) every reward term, command field, termination, event, curriculum term, observation term and constraint item as
      without it, but the unified build's joint friction and damping ranges: shares of each joint group's torque
      limit (teacher_env._add_joint_dr), they follow the model's limits (the old flow had neither: under the unified
      checks an old policy still meets them, as every policy does);
  (4) the robot's other parameters stay: model_params drops the model= entry alone;
  (5) another value stops the build; a runner with a log directory (a training) stops with the switch, one without
      (a play, a check, an export) does not, and neither does without the switch.

Local: PYTHONPATH=$PWD python scripts/unified/model_legacy_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

SWITCH = "U_MODEL"
JOINT_DR = ("u_joint_friction_", "u_joint_damping_")   # ranges from the model's torque limits (_add_joint_dr)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W,As2")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  os.environ.pop(SWITCH, None)
  os.environ.pop("BLIND_PARAMS", None)
  sys.path.insert(0, os.getcwd())
  from src.assets.robots import uros

  # the motor model (uros.MOTORS, the Go2W's gom8010) has its own test (scripts/unified/motor_switch_test.py): here
  # the aligned model as before it, so the checks below read the aligned model alone
  os.environ[uros.MOTOR_SWITCH] = "builtin"
  from src.tasks.unified import recipe as urecipe
  from src.tasks.unified import robots as urobots
  from src.tasks.unified import teacher_env
  from src.tasks.unified.robots import ROBOTS

  ok = True

  def check(good: bool, text: str) -> None:
    nonlocal ok
    ok &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'}  {text}")

  def build(robot, value: str | None = None, **kw):
    if value is not None:
      os.environ[SWITCH] = value
    try:
      cfg = teacher_env.unified_teacher_env_cfg(robot, **kw)
      with urecipe.scope(robot):
        items = teacher_env.constraint_items(robot.kind)
    finally:
      os.environ.pop(SWITCH, None)
    return cfg, items

  def model(cfg, key: str) -> tuple[str, dict, dict]:
    """The model file's name, each joint group's torque limit, and the actuator fields the aligned model sets for
    this robot (uros.ACTUATORS: a torque limit for the Go2 / Go2W / B2 / B2W, a joint friction loss for the As2 /
    As2W)."""
    ent = cfg.scene.entities["robot"]
    f = ent.spec_fn
    while getattr(f, "args", ()) and callable(f.args[0]):   # (uros.spec_fn wraps the loader with a motor model)
      f = f.args[0]
    path = f.args[0] if getattr(f, "args", ()) else None
    acts = {uros._group(x): x for x in (getattr(x, "base_cfg", x) for x in ent.articulation.actuators)}   # the blind
    limits = {g: x.effort_limit for g, x in acts.items()}                             # builder's delay wraps them
    over = {g: {f: getattr(acts[g], f) for f in fields} for g, fields in uros.ACTUATORS[key].items()}
    return (path.name if path is not None else "(the constants' default)"), limits, over

  def terms(group) -> dict:
    return {k: (getattr(v, "weight", None), repr(getattr(v, "func", None)),
                repr(sorted((getattr(v, "params", None) or {}).items(), key=repr))) for k, v in group.items()}

  def rest(cfg) -> dict:
    t = cfg.commands["twist"]
    events = {k: v for k, v in cfg.events.items() if not k.startswith(JOINT_DR)}
    out = {"rewards": terms(cfg.rewards), "terminations": terms(cfg.terminations), "events": terms(events),
           "curriculum": terms(cfg.curriculum),
           "twist": {f.name: repr(getattr(t, f.name)) for f in dataclasses.fields(t)},
           "sensors": sorted(s.name for s in cfg.scene.sensors)}
    for g, grp in cfg.observations.items():
      out["obs " + g] = sorted(grp.terms)
    return out

  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    params = urobots.unified_recipe(robot)["env"]
    check("model=uros" in params.split(","), f"the table's parameters: {params}")
    cfg0, items0 = build(robot)
    name0, lim0, over0 = model(cfg0, robot.key)
    check(name0.endswith("_uros.xml") and over0 == uros.ACTUATORS[robot.key],
          f"without {SWITCH}: {name0}, effort limits {lim0}, the aligned actuator fields {over0}")
    cfg1, items1 = build(robot, "legacy")
    name1, lim1, over1 = model(cfg1, robot.key)
    check(not name1.endswith("_uros.xml") and over1 != over0,
          f"{SWITCH}=legacy: {name1}, effort limits {lim1}, those fields {over1}")
    same = rest(cfg0) == rest(cfg1)
    check(same and items0 == items1,
          f"the {len(cfg0.rewards)} reward terms, {len(cfg0.terminations)} terminations, the other events, the "
          f"{len(cfg0.curriculum)} curriculum terms, the command, the sensors, the observations and the "
          f"{len(items0)} constraint items as without it")
    if not same:
      r0, r1 = rest(cfg0), rest(cfg1)
      for k in r0:
        if r0[k] != r1[k]:
          print(f"        differs: {k}")

    def joint_dr(cfg, limits) -> bool:
      good = True
      for g in ("hip", "thigh", "calf"):
        fr, da = cfg.events[f"u_joint_friction_{g}"].params["ranges"], cfg.events[f"u_joint_damping_{g}"].params["ranges"]
        good &= fr == (0.0, teacher_env.JOINT_FRICTION_SHARE * limits[g])
        good &= da == (0.0, teacher_env.JOINT_DAMPING_SHARE[g] * limits[g])
      return good

    check(joint_dr(cfg0, lim0) and joint_dr(cfg1, lim1) and set(cfg0.events) == set(cfg1.events),
          "the joint friction and damping ranges: the shares of each model's own torque limits; the same events")
    builds = [dict(play=True, terrain=t) for t in ("map3", "stairs", "flat")]
    builds += [dict(student=kind) for kind in ("blind", "depth")] + [dict(student="blind", play=True, terrain="stairs")]
    got = [model(build(robot, "legacy", **kw)[0], robot.key) for kw in builds]
    check(all(g == (name1, lim1, over1) for g in got),
          "the check builds (play: map3, stairs, flat) and the students' builds: the same legacy model")
    got0 = [model(build(robot, **kw)[0], robot.key) for kw in builds]
    check(all(g == (name0, lim0, over0) for g in got0), "without the switch each of them: the aligned model")
    os.environ[SWITCH] = "legacy"
    try:
      kept = urobots.model_params("model=uros,leg_kp_scale=1.5,leg_hold_s=2.0")
      none = urobots.model_params("model=uros")
    finally:
      os.environ.pop(SWITCH)
    check(kept == "leg_kp_scale=1.5,leg_hold_s=2.0" and none == ""
          and urobots.model_params("model=uros,leg_kp_scale=1.5") == "model=uros,leg_kp_scale=1.5",
          "the robot's other parameters stay: the model= entry alone goes, nothing without the switch")
    try:
      build(robot, "uros2")
      stopped = False
    except AssertionError:
      stopped = True
    check(stopped, "another value: the build stops")

  print("the runners:")

  def train_stops(value: str | None, log_dir) -> bool:
    if value is not None:
      os.environ[SWITCH] = value
    try:
      urobots.no_training_on_legacy(log_dir)
    except RuntimeError:
      return True
    finally:
      os.environ.pop(SWITCH, None)
    return False

  check(train_stops("legacy", "logs/rsl_rl/x/run"), "a runner with a log directory and the switch: stops")
  check(not train_stops("legacy", None), "one without a log directory (a play, a check, an export): goes on")
  check(not train_stops(None, "logs/rsl_rl/x/run") and not train_stops("", "logs/rsl_rl/x/run"),
        "a training without the switch (unset or empty): goes on")
  import inspect

  from src.tasks.unified import runner as urunner
  from src.tasks.unified import students as ustudents
  cls = (urunner.UnifiedTeacherRunner, ustudents.UStudentDistillationRunner, ustudents.UStudentPpoRunner)
  check(all("no_training_on_legacy(log_dir)" in inspect.getsource(c.__init__) for c in cls),
        "the teacher's and the two students' runners call it before they build")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
