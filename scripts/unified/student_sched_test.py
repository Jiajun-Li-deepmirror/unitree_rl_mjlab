#!/usr/bin/env python3
"""student_sched_test.py [--robots Go2,Go2W,As2] [--live ROBOT] -- the students' training builds carry the teacher's
step-clocked schedules at their END (go2-depth 2026-11-13; src/tasks/unified/student_sched.py).

Per robot and student kind, on the built configurations (no GPU):
  (1) the teacher's training build still has its schedules (stages and ramps after step 0): its end values are the
      reference, read here from the teacher's own configuration -- not written a second time;
  (2) the student's training build has no schedule waiting for a step, and every term a teacher schedule moves has
      the schedule's end value in the student's configuration;
  (3) building the students changes nothing in a teacher build made before or after them (the terms are copied, not
      edited in place);
  (4) the students' play build and their checks' builds (flat, stairs) are as the teacher builder made them: the fix
      is the training build's alone.
--live ROBOT: the blind student's training environment for real (4 robots, cuda:0): after the reset and a few steps
the values in effect in the managers (the tracking stds, the weights, the command's march share) are the end values.

Local: PYTHONPATH=$PWD python scripts/unified/student_sched_test.py [--live Go2]"""

from __future__ import annotations

import argparse
import os
import sys


def schedules(cfg) -> tuple[list[tuple], list[str]]:
  """The step-clocked schedules of a configuration: (what, name, key, end value) per moved term, and the step keys
  still after step 0."""
  from src.tasks.unified import student_sched

  ends, left = [], []
  for name, term in (cfg.curriculum or {}).items():
    p = getattr(term, "params", None)
    if not isinstance(p, dict):
      continue
    student_sched._left(p, f"curriculum/{name}", left)
    for s in p.get("param_stages", ()):
      for k, v in s["params"].items():
        ends.append(("reward_param", p["reward_name"], k, v))   # later stages win: the last entry per key counts
    for s in p.get("weight_stages", ()):
      ends.append(("reward_weight", p["reward_name"], None, s["weight"]))
    for r in p.get("ramps", ()):
      ends.append((r["kind"], r["name"], r.get("key"), r["end"]))
  return ends, left


def value(cfg, what: str, name: str, key):
  if what == "reward_param":
    return cfg.rewards[name].params[key]
  if what == "reward_weight":
    return cfg.rewards[name].weight
  if what == "command":
    return getattr(cfg.commands[name], key)
  raise ValueError(what)


def live(robot) -> bool:
  import torch
  from mjlab.envs import ManagerBasedRlEnv

  from src.tasks.unified import students as us
  from src.tasks.unified.teacher_env import unified_teacher_env_cfg as teacher

  ends, _ = schedules(teacher(robot))
  cfg = us.student_env_cfg(robot, "blind")
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  ok = True
  for stage in ("after the reset", "after 5 steps"):
    for what, name, key, end in ends:
      if what == "reward_param":
        got = env.reward_manager.get_term_cfg(name).params[key]
      elif what == "reward_weight":
        got = env.reward_manager.get_term_cfg(name).weight
      else:
        got = getattr(env.command_manager.get_term(name).cfg, key)
      good = abs(float(got) - float(end)) < 1e-9
      ok &= good
      print(f"  {'ok  ' if good else 'FAIL'}  live, {stage}: {name}{'.' + key if key else '.weight'} = {got} (the end: {end})")
    if stage == "after the reset":
      zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
      for _ in range(5):
        env.step(zero)
  print(f"  step counter {env.common_step_counter}")
  env.close()
  return ok


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W,As2")
  ap.add_argument("--live")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  if a.live and a.live not in names:
    names.append(a.live)
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)   # no registration: the builders are called here
  sys.path.insert(0, os.getcwd())
  from src.tasks.unified import recipe as urecipe
  from src.tasks.unified import students as us
  from src.tasks.unified.robots import ROBOTS
  from src.tasks.unified.teacher_env import unified_teacher_env_cfg as teacher

  ok = True

  def check(good: bool, text: str) -> None:
    nonlocal ok
    ok &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'}  {text}")

  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    ends, left = schedules(teacher(robot))
    before = repr(sorted(map(repr, ends))) + repr(sorted(left))
    check(ends and left, f"the teacher's training build has its schedules: {len(ends)} moved value(s), waiting for {len(left)} step(s)")
    final = {(w, nm, k): v for w, nm, k, v in ends}   # the last entry per term: the end
    for kind in ("blind", "depth"):
      cfg = us.student_env_cfg(robot, kind)
      s_ends, s_left = schedules(cfg)
      check(not s_left, f"{kind}: no schedule waits for a step{': ' + ' '.join(s_left) if s_left else ''}")
      check(len(s_ends) == len(ends), f"{kind}: the teacher's {len(ends)} scheduled value(s) are all there ({len(s_ends)})")
      for (w, nm, k), v in final.items():
        got = value(cfg, w, nm, k)
        check(got == v, f"{kind}: {nm}{'.' + k if k else '.weight'} = {got} (the teacher's end: {v})")
      # the play build and the checks: untouched by the fix (their schedules, where they have any, as the builder's)
      for label, kw in (("play", {"play": True}), ("flat", {"terrain": "flat"}), ("stairs", {"terrain": "stairs"})):
        other = us.student_env_cfg(robot, kind, **kw)
        ref = teacher(robot, student=kind, **kw)
        check(schedules(other) == schedules(ref), f"{kind} {label}: the schedules as the teacher builder made them")
    e2, l2 = schedules(teacher(robot))
    check(repr(sorted(map(repr, e2))) + repr(sorted(l2)) == before, "the teacher's build after the students': unchanged")
    with_recipe = urecipe.effective(robot)
    print(f"        (recipe in effect: {len(with_recipe)} switch(es))")
  if a.live:
    print(f"{a.live}: live")
    ok &= live(ROBOTS[a.live])
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
