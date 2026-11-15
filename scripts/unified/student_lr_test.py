#!/usr/bin/env python3
"""student_lr_test.py -- the students' stage 2 trains at a steady learning rate (go2-depth 2026-11-15;
src/tasks/unified/students.py S2_LEARNING_RATE, ppo_runner_cfg). The first formal Go2 students ran the adaptive
schedule: the rate sat near 5e-6 and stage 2 moved the items stage 1 hands over short by +-0.05-0.10; the probes' only
gain was a steady 3e-5.

On the built runner configurations (no simulator, no GPU), per robot and student kind:
  (1) stage 2: schedule "fixed" at S2_LEARNING_RATE (3e-5), no ceiling ramp and no end value on the runner;
  (2) stage 2's other settings as before: behavior cloning 1.0 -> 0.3, the critic's warm-up 50 updates, desired_kl
      0.005 (unused by the fixed schedule, kept for a probe's --agent.algorithm.schedule=adaptive), entropy 0, the
      privileged-latent cost 0, 3000 iterations;
  (3) stage 1 (distillation) as before: its algorithm's fields are the class defaults, 4000 iterations;
  (4) the update itself: UStudentPPO adapts the rate under the schedule "adaptive" alone (read from its source).

Local: PYTHONPATH=$PWD python scripts/unified/student_lr_test.py [--robots Go2,Go2W,As2]"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import os
import sys


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", default="Go2,Go2W,As2")
  a = ap.parse_args()
  names = [n for n in a.robots.split(",") if n]
  os.environ["U_ROBOTS"] = ",".join(names)
  os.environ.pop("UNIFIED_TASKS", None)
  sys.path.insert(0, os.getcwd())
  from src.tasks.unified import students
  from src.tasks.unified.robots import ROBOTS

  ok = True

  def check(good: bool, text: str) -> None:
    nonlocal ok
    ok &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'}  {text}")

  check(students.S2_LEARNING_RATE == 3e-5, f"S2_LEARNING_RATE {students.S2_LEARNING_RATE:g}")
  src = inspect.getsource(students.UStudentPPO)
  check('self.schedule == "adaptive"' in src and src.count("self.learning_rate / 1.5") == 1
        and src.count("self.learning_rate * 1.5") == 1,
        "UStudentPPO: the rate moves with the KL under the schedule \"adaptive\" alone")
  for n in names:
    robot = ROBOTS[n]
    print(f"{n}:")
    for kind in ("blind", "depth"):
      cfg = students.ppo_runner_cfg(robot, kind)
      alg = cfg.algorithm
      check(alg.schedule == "fixed" and alg.learning_rate == students.S2_LEARNING_RATE
            and cfg.learning_rate_final is None and cfg.lr_ramp_iters == 0,
            f"{kind} stage 2: schedule {alg.schedule} at {alg.learning_rate:g}, no ramp, no end value")
      check(alg.bc_coef == 1.0 and alg.bc_coef_final == 0.3 and alg.critic_warmup == 50 and alg.desired_kl == 0.005
            and alg.entropy_coef == 0.0 and alg.priv_coef == 0.0 and cfg.max_iterations == students.S2_ITERATIONS == 3000,
            f"{kind} stage 2: BC {alg.bc_coef:g} -> {alg.bc_coef_final:g}, warm-up {alg.critic_warmup}, desired_kl "
            f"{alg.desired_kl:g}, entropy {alg.entropy_coef:g}, priv {alg.priv_coef:g}, {cfg.max_iterations} iterations")
      d = students.distill_runner_cfg(robot, kind)
      ref = students.UDistillationAlgorithmCfg()
      same = all(getattr(d.algorithm, f.name) == getattr(ref, f.name) for f in dataclasses.fields(ref))
      check(same and d.max_iterations == students.S1_ITERATIONS == 4000,
            f"{kind} stage 1: the algorithm's {len(dataclasses.fields(ref))} fields at their defaults, "
            f"{d.max_iterations} iterations")
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
