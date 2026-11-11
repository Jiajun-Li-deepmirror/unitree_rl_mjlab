"""The unified pipeline's robots: physical parameters only, and the environment the teacher is built on.

go2-depth 2026-11-11, the user's redesign: "一个带特权信息的老师，蒸馏出盲走和深度两个学生；验收要求直接写成训练约束；每次同时
跑多个种子" ... "腿式、轮式各一套任务，每个机型只填物理参数". New code beside src/tasks/velocity: nothing there is changed.

The environment layer (robot model, map3 terrain tiles, the terrain-aware velocity command with its stand / march /
spin / stop situations, domain randomization, payload, sensors, the gait clock, every cost term with its state) is the
existing blind builder's, with the unified recipe's final (P) keys, because that is the environment every delivered
model passed the acceptance in. What the unified teacher changes on top of it is in teacher_env.py.
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mjlab.envs import ManagerBasedRlEnvCfg

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class URobot:
  """One robot of the unified pipeline: its kind and its physical parameters (as jobs/fe_robot.sh and
  jobs/recipes/unified.py ROBOTS give them to the blind chains)."""

  name: str            # "Go2", "Go2W"
  kind: str            # "legged" or "wheeled"
  payload: str         # BLIND_PAYLOAD: limit (kg) and the payload point's range in the base frame (m)
  wall_stop_m: float   # BLIND_WALL_STOP: where the forward command reaches zero before an end wall
  foot_r_m: float | None = None   # BLIND_FOOT_R: foot sphere radius (legged)

  @property
  def key(self) -> str:
    return self.name.lower()


ROBOTS: dict[str, URobot] = {
  "Go2": URobot("Go2", "legged", "max=12,x=0.15,y=0.08,z=0.20", 0.85, foot_r_m=0.022),
  "Go2W": URobot("Go2W", "wheeled", "max=12,x=0.15,y=0.08,z=0.20", 0.85),
}


def unified_recipe(robot: URobot) -> dict:
  """jobs/recipes/unified.py's recipe for the robot (read, not changed): "T" / "M" / "P" key lists and "env", the
  robot's BLIND_PARAMS (leg gains, leg hold, stairs share cap; empty for Go2 / Go2W)."""
  spec = importlib.util.spec_from_file_location("_unified_recipe", REPO / "jobs" / "recipes" / "unified.py")
  assert spec and spec.loader
  mod = importlib.util.module_from_spec(spec)
  import sys

  sys.modules["_unified_recipe"] = mod   # its dataclasses look themselves up in sys.modules
  spec.loader.exec_module(mod)
  return mod.recipe(robot.key)


@contextmanager
def _environ(values: dict[str, str | None]):
  """Set (str) or clear (None) environment variables for the duration, then restore them: the blind builder reads
  its options from the environment when it builds, and other tasks registering in the same process must not see
  ours (nor we theirs)."""
  old = {k: os.environ.get(k) for k in values}
  try:
    for k, v in values.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v
    yield
  finally:
    for k, v in old.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v


def _factory(robot: URobot) -> Callable[..., ManagerBasedRlEnvCfg]:
  if robot.name == "Go2":
    from src.tasks.velocity.config.go2.blind_env_cfgs import unitree_go2_blind_scratch_env_cfg

    return unitree_go2_blind_scratch_env_cfg
  if robot.name == "Go2W":
    from src.tasks.velocity.config.go2w.blind_env_cfgs import unitree_go2w_blind_scratch_env_cfg

    return unitree_go2w_blind_scratch_env_cfg
  raise KeyError(robot.name)


# Keys left out of the base: the old constraint sets (the unified teacher has its own, constraints.py) and the
# lower PPO entropy (the teacher trains from zero at the normal one).
DROPPED_KEYS = ("legcons", "wheelcons", "sidetgt75", "consease", "desccons", "lowent")


def base_env_cfg(robot: URobot, play: bool = False) -> ManagerBasedRlEnvCfg:
  """The blind environment on map3 at the teacher phase's schedule (early cost ramps, legged walk gate) with the
  unified recipe's final keys, plus the depth teacher's elevated height scan in the actor (depth_env_cfg)."""
  from src.tasks.velocity.depth_env_cfg import apply_depth_teacher

  recipe = unified_recipe(robot)
  keys = [k for k in recipe["P"] if k not in DROPPED_KEYS]
  env = {
    "BLIND_PILOT": ",".join(keys),
    "BLIND_PAYLOAD": robot.payload,          # no ramp: the progress curriculum scales the payload (curriculum.py)
    "BLIND_WALL_STOP": str(robot.wall_stop_m),
    "BLIND_FOOT_R": None if robot.foot_r_m is None else str(robot.foot_r_m),
    "BLIND_PARAMS": recipe["env"],
    # nothing of a blind chain's run state
    "BLIND_PHASE_START_IT": None,
    "BLIND_CONS_INIT": None,
    "BLIND_ANCHOR_CHECKPOINT": None,
    "BLIND_ANCHOR_COEFF": None,
  }
  with _environ(env):
    cfg = _factory(robot)(terrain="map3", phase="teacher", play=play)
    cfg = apply_depth_teacher(cfg)
  return cfg
