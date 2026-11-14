#!/usr/bin/env python3
"""rest_even_test.py -- U_REST_OLD=even: a wheeled robot's stand as the delivered Go2W's on even ground, the lock form
(restlock + standtgt) on uneven ground (go2-depth 2026-11-14; src/tasks/unified/rest.py, robots.base_env_cfg).

(A) The two classes around the shared kernels, the kernels' own __call__ and the uneven mask replaced (CPU, no
    simulator): pose_lock_uneven gives the kernel's value with the lock on uneven ground and the one without it on
    even ground, passes every other parameter on unchanged, and without a hold_term calls the kernel once;
    target_hold_uneven gives the kernel's value on uneven ground and 0 on even ground and keeps the mask's
    parameters to itself.
(B) On the built configurations (no GPU; the play build, the check terrains flat / stairs and the import with
    UNIFIED_TASKS=1 -- every task built -- included): Go2W with U_REST_OLD=even has the same terms in the same order at the
    same weights as the plain build; pose and standing_target_hold are the two classes (the target hold with the
    pose term's scan sensor), stand_still_even is without its hold_term, every other term's function and parameters
    are the plain build's, the observations and commands too; the blind student's build takes the switch as well;
    U_REST_OLD=1 is the build without standing_target_hold as before; unset / 0 the plain build; the Go2 (legged)
    is unchanged by the switch; another value stops the build.

Local: PYTHONPATH=$PWD python scripts/unified/rest_even_test.py"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace


def class_cases(rest) -> list[tuple[bool, str]]:
  import torch

  from src.tasks.velocity.mdp import rewards as old_rew

  out: list[tuple[bool, str]] = []

  def ok(cond, text):
    out.append((bool(cond), text))

  uneven = torch.tensor([True, False, True, False])
  calls: list[dict] = []

  def fake_pose(self, env, **kw):
    calls.append(kw)
    return torch.full((4,), 2.0 if kw.get("hold_term") else 1.0)

  def fake_hold(self, env, **kw):
    calls.append(kw)
    return torch.full((4,), 0.3)

  masks: list[tuple] = []

  def fake_mask(env, sensor_name, radius=0.45, threshold=0.05):
    masks.append((sensor_name, radius, threshold))
    return uneven.clone()

  keep = (old_rew.variable_posture_terrain_aware.__call__, old_rew.standing_target_hold.__call__,
          old_rew.uneven_terrain_mask)
  old_rew.variable_posture_terrain_aware.__call__ = fake_pose
  old_rew.standing_target_hold.__call__ = fake_hold
  old_rew.uneven_terrain_mask = fake_mask
  try:
    env = SimpleNamespace()
    pose = object.__new__(rest.pose_lock_uneven)
    got = pose(env, sensor_name="scan", hold_term="standing_position_hold", std_standing=1, command_name="twist",
               uneven_radius=0.4, uneven_threshold=0.06, rest_hip=None)
    ok(got.tolist() == [2.0, 1.0, 2.0, 1.0], f"pose: the lock's value on uneven ground, the free one on even: {got.tolist()}")
    ok(len(calls) == 2 and calls[0]["hold_term"] is None and calls[1]["hold_term"] == "standing_position_hold",
       "pose: the kernel once without and once with the hold_term")
    same = all({k: v for k, v in c.items() if k != "hold_term"} == {
      "sensor_name": "scan", "std_standing": 1, "command_name": "twist", "uneven_radius": 0.4,
      "uneven_threshold": 0.06, "rest_hip": None} for c in calls)
    ok(same, "pose: every other parameter passed on unchanged")
    ok(masks == [("scan", 0.4, 0.06)], f"pose: the mask on the term's sensor, radius and threshold: {masks}")
    calls.clear(); masks.clear()
    got = pose(env, sensor_name="scan", std_standing=1)
    ok(got.tolist() == [1.0] * 4 and len(calls) == 1 and not masks, "pose without a hold_term: the kernel once, no mask")

    calls.clear(); masks.clear()
    hold = object.__new__(rest.target_hold_uneven)
    got = hold(env, sensor_name="scan", command_name="twist", tau_s=1.0)
    ok(torch.allclose(got, torch.tensor([0.3, 0.0, 0.3, 0.0])), f"target hold: the kernel's cost on uneven ground, 0 on even: {got.tolist()}")
    ok(calls == [{"command_name": "twist", "tau_s": 1.0}], "target hold: the kernel gets its own parameters only")
    ok(masks == [("scan", 0.45, 0.05)], "target hold: the mask at the kernel's default radius and threshold")
  finally:
    (old_rew.variable_posture_terrain_aware.__call__, old_rew.standing_target_hold.__call__,
     old_rew.uneven_terrain_mask) = keep
  return out


def main() -> int:
  os.environ["U_ROBOTS"] = "Go2W,Go2"
  os.environ.pop("UNIFIED_TASKS", None)
  os.environ.pop("U_REST_OLD", None)
  sys.path.insert(0, os.getcwd())
  from src.tasks.unified import rest, teacher_env
  from src.tasks.unified.robots import ROBOTS
  from src.tasks.velocity.mdp import rewards as old_rew

  ok = True

  def check(good: bool, text: str) -> None:
    nonlocal ok
    ok &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'}  {text}")

  print("the classes:")
  for good, text in class_cases(rest):
    check(good, text)

  def build(name, student=None, **env):
    os.environ.update(env)
    try:
      return teacher_env.unified_teacher_env_cfg(ROBOTS[name], student=student)
    finally:
      for k in env:
        os.environ.pop(k)

  def teacher_build(name, play=False, terrain="map3", **env):
    os.environ.update(env)
    try:
      return teacher_env.unified_teacher_env_cfg(ROBOTS[name], play=play, terrain=terrain)
    finally:
      for k in env:
        os.environ.pop(k)

  def sig(cfg, skip=()):
    return {k: (v.weight, getattr(v.func, "__name__", repr(v.func)), repr(sorted(v.params.items(), key=repr)))
            for k, v in cfg.rewards.items() if k not in skip}

  def obs(cfg):
    return {g: list(getattr(v, "terms", {}) or {}) for g, v in cfg.observations.items()}

  three = ("pose", "stand_still_even", "standing_target_hold")
  print("Go2W:")
  plain = build("Go2W")
  even = build("Go2W", U_REST_OLD="even")
  check(list(even.rewards) == list(plain.rewards), f"the same {len(plain.rewards)} terms in the same order")
  check(all(even.rewards[k].weight == plain.rewards[k].weight for k in plain.rewards), "at the same weights")
  check(sig(even, three) == sig(plain, three), "every other term's function and parameters as the plain build's")
  p, q = even.rewards["pose"], plain.rewards["pose"]
  check(p.func is rest.pose_lock_uneven and q.func is old_rew.variable_posture_terrain_aware
        and repr(sorted(p.params.items(), key=repr)) == repr(sorted(q.params.items(), key=repr)),
        "pose: pose_lock_uneven on the plain build's parameters (hold_term kept)")
  s, t = even.rewards["stand_still_even"], plain.rewards["stand_still_even"]
  check("hold_term" not in s.params and t.params.get("hold_term") == "standing_position_hold"
        and {k: repr(v) for k, v in s.params.items()} == {k: repr(v) for k, v in t.params.items() if k != "hold_term"},
        "stand_still_even: without its hold_term, nothing else changed")
  h, g = even.rewards["standing_target_hold"], plain.rewards["standing_target_hold"]
  check(h.func is rest.target_hold_uneven and g.func is old_rew.standing_target_hold
        and h.params.get("sensor_name") == p.params["sensor_name"]
        and {k: repr(v) for k, v in h.params.items() if k != "sensor_name"} == {k: repr(v) for k, v in g.params.items()},
        f"standing_target_hold: target_hold_uneven on the pose term's sensor ({h.params.get('sensor_name')})")
  check(obs(even) == obs(plain), "observations (actor, critic) unchanged")
  check(repr(even.commands["twist"]) == repr(plain.commands["twist"]), "command unchanged")
  check(sorted(even.curriculum or {}) == sorted(plain.curriculum or {}) and sorted(even.events or {}) == sorted(plain.events or {}),
        "curriculum and event terms unchanged")
  stu = build("Go2W", student="blind", U_REST_OLD="even")
  check(stu.rewards["pose"].func is rest.pose_lock_uneven and stu.rewards["standing_target_hold"].func is rest.target_hold_uneven,
        "the blind student's build takes the switch too")
  old = build("Go2W", U_REST_OLD="1")
  check("standing_target_hold" not in old.rewards and not old.rewards["pose"].params.get("hold_term")
        and old.rewards["pose"].func is old_rew.variable_posture_terrain_aware, "U_REST_OLD=1: as before (no target hold, no lock)")
  check(sig(build("Go2W", U_REST_OLD="0")) == sig(plain) and sig(build("Go2W", U_REST_OLD="")) == sig(plain),
        "U_REST_OLD=0 / empty: the plain build")
  try:
    build("Go2W", U_REST_OLD="2"); check(False, "another value stops the build")
  except AssertionError:
    check(True, "another value stops the build")
  # every task is built at import (src/tasks/unified/__init__.py): the play build and the check terrains' builds
  # must build with the switch too (uprobe_s120 stopped at the flat build, 2026-11-14 17:15)
  play = teacher_build("Go2W", play=True, U_REST_OLD="even")
  check(play.rewards["pose"].func is rest.pose_lock_uneven, "the play build takes the switch too")
  for terrain in ("flat", "stairs"):
    for is_play in (False, True):
      got = teacher_build("Go2W", play=is_play, terrain=terrain, U_REST_OLD="even")
      want = teacher_build("Go2W", play=is_play, terrain=terrain)
      check(sig(got) == sig(want) and "standing_target_hold" not in got.rewards,
            f"the check terrain {terrain}{' (play)' if is_play else ''}: no lock terms there, the plain build")
  print("the registration (UNIFIED_TASKS=1, every task built):")
  import subprocess

  for value in ("even", "1"):
    env = dict(os.environ, UNIFIED_TASKS="1", U_REST_OLD=value, CUDA_VISIBLE_DEVICES="")
    done = subprocess.run([sys.executable, "-c", "import src.tasks.unified"], env=env, capture_output=True, text=True)
    last = (done.stderr.strip().splitlines() or [""])[-1][:160]
    check(done.returncode == 0, f"U_REST_OLD={value}: the import builds every task" + ("" if done.returncode == 0 else f" -- {last}"))
  print("Go2:")
  check(sig(build("Go2", U_REST_OLD="even")) == sig(build("Go2")), "the legged robot: the switch changes nothing")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
