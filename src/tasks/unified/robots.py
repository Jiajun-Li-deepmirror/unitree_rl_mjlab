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
  # the box the payload's mount point is drawn from on the training terrain, ((x lo, hi), (y lo, hi), (z lo, hi)) in
  # the base frame (m); None: the range in `payload` (the robot table's payload_point, teacher_env._payload_point)
  payload_point: tuple | None = None

  @property
  def key(self) -> str:
    return self.name.lower()


def _from_table(name: str) -> URobot:
  """The robot's physical parameters from the one robot table (src/robot_table.py, 2026-11-11)."""
  from src import robot_table

  r = robot_table.get(name)
  return URobot(name, r["kind"], r["train_payload"], r["wall_stop"],
                foot_r_m=r.get("foot_r") if r["kind"] == "legged" else None, payload_point=r.get("payload_point"))


# The robots the tasks are registered for: Go2 and Go2W, or the ones U_ROBOTS names ("As2", "As2W,B2", ...; 2026-11-12:
# the short probes of the other robots -- the robot table and the blind builders have all six with one signature, and
# registering every robot builds three times as many environments at every import).
ROBOTS: dict[str, URobot] = {
  name: _from_table(name) for name in (os.environ.get("U_ROBOTS") or "Go2,Go2W").split(",") if name
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
  """The robot's blind builder from zero: unitree_<key>_blind_scratch_env_cfg(play, terrain, phase), the same
  signature for all six robots."""
  mod = importlib.import_module(f"src.tasks.velocity.config.{robot.key}.blind_env_cfgs")
  return getattr(mod, f"unitree_{robot.key}_blind_scratch_env_cfg")


# Keys left out of the base: the old constraint sets (the unified teacher has its own, constraints.py) and the
# lower PPO entropy (the teacher trains from zero at the normal one).
DROPPED_KEYS = ("legcons", "wheelcons", "sidetgt75", "consease", "desccons", "lowent")
# U_REST_OLD=1 (a probe's switch, off unless set; teacher_env._probe_switches' docstring): a wheeled robot's stand as
# the delivered Go2W's -- without "restlock" (pose and stand_still_even take the pose at the hold's lock as their
# target) and "standtgt" (the leg-target hold).
REST_OLD_KEYS = ("restlock", "standtgt")


def base_env_cfg(robot: URobot, play: bool = False, terrain: str = "map3", keys: str = "P") -> ManagerBasedRlEnvCfg:
  """The blind environment on map3 at the teacher phase's schedule (early cost ramps, legged walk gate) with the
  unified recipe's final keys, plus the depth teacher's elevated height scan in the actor (depth_env_cfg).
  keys "M": the same with the recipe's M keys -- the environment its climbing was learned in, before the P phase's
  stairs quality (teacher_env.quality_diff reads the two's difference)."""
  from src.tasks.velocity.depth_env_cfg import apply_depth_teacher

  recipe = unified_recipe(robot)
  v = os.environ.get("U_REST_OLD")
  assert v in (None, "", "0", "1"), f"U_REST_OLD {v}: 1 or unset"
  drop = DROPPED_KEYS + (REST_OLD_KEYS if v == "1" and robot.kind == "wheeled" else ())
  keys = [k for k in recipe[keys] if k not in drop]
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
    # terrain: "map3" trains; the acceptance's checks (2026-11-12) use the blind builder's "flat" and "stairs"
    cfg = _factory(robot)(terrain=terrain, phase="teacher", play=play)
    cfg = apply_depth_teacher(cfg)
  return cfg
