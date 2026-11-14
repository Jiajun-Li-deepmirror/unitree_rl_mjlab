#!/usr/bin/env python3
"""com_support_blind_test.py -- U_COM_SUPPORT's third field: the share under a command of the robots that walk with
their scan masked (go2-depth 2026-11-14; constraints.com_support's moving_blind, teacher_env._add_com_support).

(A) The term on a made-up environment (CPU, no simulator): at the zero command in full for both scan modes; under a
    command `moving` times it with the scan, `moving_blind` times it without; without the field as before (one share
    for both); even ground costs nothing; an environment without the scan flag falls back to `moving`.
(B) On the built configurations (no GPU): without the third field the term's parameters have no moving_blind key (the
    builds before it are unchanged); with it the key, and nothing else differs; a value outside 0-1 or a fourth
    field stops the build.

Local: PYTHONPATH=$PWD python scripts/unified/com_support_blind_test.py [--robots Go2,Go2W]"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  os.environ.pop("U_COM_SUPPORT", None)
  sys.path.insert(0, os.getcwd())
  import torch

  import src.tasks.velocity.mdp.rewards as old_rew
  from src.tasks.unified import constraints as ucons
  from src.tasks.unified import teacher_env
  from src.tasks.unified.robots import ROBOTS

  ok = True

  def check(good, text: str) -> None:
    nonlocal ok
    ok &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'}  {text}")

  def close(x, y):
    return bool(torch.allclose(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32), atol=1e-6))

  # (A) 8 robots: [stand, walk] x [seen, blind] x [uneven, even]; every one 10 cm off a 0.25 m support
  n = 8
  cmd = torch.zeros(n, 3)
  cmd[1::2, 0] = 0.5
  on = torch.tensor([1.0, 1.0, 0.0, 0.0] * 2)
  uneven = torch.tensor([True] * 4 + [False] * 4)
  dist, size = torch.full((n,), 0.10), torch.full((n,), 0.25)
  full = (0.10 / 0.25 - 0.12) / 0.80
  keep_off, keep_mask = ucons.com_offset, old_rew.uneven_terrain_mask
  ucons.com_offset = lambda env, asset_cfg: (dist.clone(), size.clone())
  old_rew.uneven_terrain_mask = lambda env, sensor_name, *a_, **k_: uneven.clone()
  try:
    env = SimpleNamespace(num_envs=n, device="cpu", u_scan_on=on,
                          command_manager=SimpleNamespace(get_command=lambda name: cmd))
    kw = {"asset_cfg": None, "sensor_name": "scan"}
    print("the term:")
    got = ucons.com_support(env, moving=0.5, moving_blind=0.0, **kw)
    check(close(got, [full, 0.5 * full, full, 0.0, 0, 0, 0, 0]),
          "0.5 with the scan, 0 without: stand full / walk 0.5 / blind stand full / blind walk 0; even ground 0")
    got = ucons.com_support(env, moving=0.5, moving_blind=0.2, **kw)
    check(close(got, [full, 0.5 * full, full, 0.2 * full, 0, 0, 0, 0]), "0.5 / 0.2: the blind walk at 0.2")
    got = ucons.com_support(env, moving=1.0, moving_blind=0.0, **kw)
    check(close(got, [full, full, full, 0.0, 0, 0, 0, 0]), "1.0 / 0: the blind walk still at 0 (the full share does not skip the field)")
    got = ucons.com_support(env, moving=0.5, **kw)
    check(close(got, [full, 0.5 * full, full, 0.5 * full, 0, 0, 0, 0]), "without the field: one share for both (as before)")
    got = ucons.com_support(env, **kw)
    check(close(got, [full, 0.0, full, 0.0, 0, 0, 0, 0]), "without both: the stops alone (as before)")
    bare = SimpleNamespace(num_envs=n, device="cpu", command_manager=env.command_manager)
    got = ucons.com_support(bare, moving=0.5, moving_blind=0.0, **kw)
    check(close(got, [full, 0.5 * full, full, 0.5 * full, 0, 0, 0, 0]), "an environment without the scan flag: `moving` for all")
    check(close(env.u_com_offset, dist), "env.u_com_offset: the distance, as before")
  finally:
    ucons.com_offset, old_rew.uneven_terrain_mask = keep_off, keep_mask

  def build(robot, value):
    os.environ["U_COM_SUPPORT"] = value
    try:
      return teacher_env.unified_teacher_env_cfg(robot)
    finally:
      os.environ.pop("U_COM_SUPPORT")

  def fails(robot, value) -> bool:
    try:
      build(robot, value)
    except (AssertionError, ValueError):
      return True
    return False

  def rest(cfg):
    return {k: (v.weight, repr(sorted(((kk, vv) for kk, vv in v.params.items() if kk != "moving_blind"), key=repr)))
            for k, v in cfg.rewards.items()}

  term = teacher_env.COM_TERM
  for name in names:
    robot = ROBOTS[name]
    print(f"{name}:")
    two = build(robot, "15,0.5")
    check("moving_blind" not in two.rewards[term].params and two.rewards[term].params["moving"] == 0.5
          and two.rewards[term].weight == -15.0, "15,0.5: no moving_blind key (the build as before)")
    one = build(robot, "15")
    check("moving_blind" not in one.rewards[term].params and one.rewards[term].params["moving"] == 0.0, "15: the stops alone, no key")
    three = build(robot, "15,0.5,0")
    check(three.rewards[term].params.get("moving_blind") == 0.0 and three.rewards[term].params["moving"] == 0.5,
          "15,0.5,0: moving 0.5, moving_blind 0")
    check(rest(three) == rest(two) and list(three.rewards) == list(two.rewards),
          f"15,0.5,0: the other parameters of the {len(two.rewards)} terms, and their order, as 15,0.5")
    check(build(robot, "15,0.5,0.2").rewards[term].params.get("moving_blind") == 0.2, "15,0.5,0.2: moving_blind 0.2")
    check(fails(robot, "15,0.5,1.5") and fails(robot, "15,0.5,0,0") and fails(robot, "15,0.5,x"),
          "a share outside 0-1, a fourth field, a non-number stop the build")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
