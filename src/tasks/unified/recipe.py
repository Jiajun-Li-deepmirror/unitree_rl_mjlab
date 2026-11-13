"""The unified teacher's recipe, in code (go2-depth 2026-11-13).

Until now the recipe was a line of U_ switches on every probe's command line (teacher_env._probe_switches: each one
"off unless set"), repeated by hand for the gates, which took it from the run's timeline. The formal scripts
(jobs/unified/teacher.sh, stairs_gate.sh, flat_gate.sh, students.sh, accept.sh) carry no switches: a formal run from
them would have trained the plain task, not the recipe the probes arrived at. The user's order (2026-11-11): "配方由
代码定义，不从日志里提取 ... 一份共用基础配方，加上腿式和轮式各自的部分，再加少量每个机型自己的差异，每项差异写明原因".

So the recipe is this table -- COMMON, the robot's KIND, the few switches of one ROBOT -- and it is the default:
`scope(robot)` puts the robot's switches into the environment for the duration of a build
(teacher_env.unified_teacher_env_cfg: every task of the package, the students' too, on every terrain), the runner
reads its two through `runner_switches` (the entropy bonus, the std cap), and the gate reads the constraint items
under the same scope. The switches' code is untouched: what a probe's command line built, the clean environment
builds now (scripts/unified/recipe_test.py compares the two on every task's configuration).

A probe still works as before:
  - a switch set in the environment wins over the recipe's value (U_STD_CAP=0.34 on a command line: the recipe plus
    that one difference);
  - U_RECIPE=0 switches the table off: the environment's switches alone decide, as before this file (a run from
    before it is evaluated that way: its timeline's env_vars are its whole recipe).
A run's timeline says which it was: the start event's "recipe" (record(): the switches in effect, the ones the
environment set itself, "off" for U_RECIPE=0) beside the "env_vars" it always had.

Tested on Go2 and Go2W (the probes on t1 / t2) and As2 (go2-home's); the other robots (U_ROBOTS) take their kind's
part untested."""

from __future__ import annotations

import os
from contextlib import contextmanager

# every robot (each switch's account: teacher_env._probe_switches' docstring)
COMMON: dict[str, str] = {
  "U_Q_SCALE_MAX": "1",         # the stairs-quality items' multipliers stay at or below 1: the P weights as configured
  "U_TERMINATED_Q": "10",       # a fall costs 1x -> 10x along the stairs quality ("负载一定不能摔倒")
  "U_ENTROPY_Q": "0.002",       # the entropy bonus 0.005 -> 0.002 along the stairs quality (runner._entropy_by_q)
  "U_METER_EMA": "2400",        # the stairs meters' memory, env steps (20000 lagged a change of behaviour)
  "U_FLAT_ONLY": "1",           # a flat item's multiplier presses on flat ground only
  "U_PAYLOAD_STICKY": "0.05",   # the payload class stays with the robot, drawn anew at a reset with this chance
  "U_STAIR_SPEED_H": "1",       # the stairs speed over the ground, as the acceptance measures it
  "U_HEADING_OBS": "1",         # the line costs hold the heading the actor is shown ...
  "U_HEADING_LAG": "2",         # ... and the heading's lag behind the commands costs (weight -2)
}

KIND: dict[str, dict[str, str]] = {
  "legged": {
    "U_STOP_FEET_DIR": "3",       # four feet at a stop on a flight: one item per direction faced, both multipliers from 3
    "U_FLAT_DIAG_AXIS": "0.2",    # a diagonal command's flat speed per axis, as the acceptance judges it (the wheeled have it)
    "U_FLAT_SPEED_W": "4",        # the flat speed deficit at the weight -4 (the wheeled have it)
    "U_STAIR_PARK": "10",         # standing on a descending column under a forward command costs
    "U_BODY_CLEAR": "50",         # the bodies whose touch ends the episode keep clear of the steps
    "U_COM_SUPPORT": "15",        # at a stop on uneven ground the centre of mass, payload included, over the feet
                                  # (Go2's loaded falls on tier 2: 11 -> 5)
  },
  "wheeled": {
    "U_STAIR_APPROACH": "0",          # the stairs speed cost holds on a climbing column's level ground too (the Go2W
                                      # stood before the flight); not for the heavy class (the 0)
    "U_BODY_CLEAR": "50,0.06,0.10",   # as the legged, with the wheeled margin and reach (Go2W's loaded falls on tier
                                      # 2: 16 at the plain 50 -> 5)
    "U_STAIRS_P": "0.0,0.15",         # the stairs share ramps in over progress 0-0.15 (the Go2W from zero: progress
                                      # 0.49 at 900 iterations against 0.08); the legged keep the task's 0.3-0.8
  },
}

# one robot's own, each with its reason
ROBOT: dict[str, dict[str, str]] = {
  # the standing pose cost off the uneven ground: the heavy legged robot's loaded falls at the stops halve (179 -> 88,
  # go2-home's mAJ). Not for the others: the Go2's loaded falls tripled with it (62 against 20, s50 / s51), the
  # Go2W gained nothing (s38)
  "As2": {"U_STAND_POSE_EVEN": "1"},
}

# read by the runner when it starts (runner._entropy_by_q, _std_cap_by_q), not by the builders
RUNNER = ("U_ENTROPY_Q", "U_STD_CAP")


def off() -> bool:
  """U_RECIPE=0: the environment's switches alone (module docstring)."""
  return os.environ.get("U_RECIPE") == "0"


def recipe(robot) -> dict[str, str]:
  """The robot's switches from the table ({} with U_RECIPE=0)."""
  if off():
    return {}
  return {**COMMON, **KIND.get(robot.kind, {}), **ROBOT.get(robot.name, {})}


def explicit(robot) -> dict[str, str]:
  """The recipe's switches the environment sets itself: a probe's differences (they win)."""
  return {k: os.environ[k] for k in recipe(robot) if k in os.environ}


def effective(robot) -> dict[str, str]:
  """The recipe's switches as they apply: the table's values, the environment's where it sets one."""
  return {k: os.environ.get(k, v) for k, v in recipe(robot).items()}


@contextmanager
def scope(robot):
  """The robot's recipe in the environment for the duration: the switches the environment does not set itself.
  Restored afterwards -- the package builds every robot's tasks in one process, and one robot's switches must not
  reach the next one's build. Nested scopes of one robot are harmless (the inner one finds everything set)."""
  add = {k: v for k, v in recipe(robot).items() if k not in os.environ}
  os.environ.update(add)
  try:
    yield
  finally:
    for k in add:
      os.environ.pop(k, None)


def runner_switches(robot) -> dict[str, str]:
  """The runner's switches (RUNNER) in effect for the robot (None: no robot known -- the environment alone)."""
  table = recipe(robot) if robot is not None else {}
  out = {k: os.environ.get(k, table.get(k)) for k in RUNNER}
  return {k: v for k, v in out.items() if v}


def of_experiment(name: str | None):
  """The robot of a run by its experiment directory (<robot>_u_teacher, <robot>_u_student_<kind>[_rl]); None when
  it is none of the registered robots'."""
  from .robots import ROBOTS

  key = (name or "").split("_u_")[0]
  return next((r for r in ROBOTS.values() if r.key == key), None)


def record(robot) -> dict:
  """For a run's timeline: the recipe in effect, what the environment set itself, and whether the table was off."""
  if robot is None:
    return {"robot": None}
  return {"robot": robot.name, "off": off(), "switches": effective(robot), "explicit": explicit(robot)}
