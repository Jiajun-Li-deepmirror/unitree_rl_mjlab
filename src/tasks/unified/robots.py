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
from dataclasses import dataclass, replace
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
  robot's BLIND_PARAMS (model=uros for every robot since 2026-11-11 -- U_MODEL below; leg gains, leg hold and
  stairs share cap where the robot has them)."""
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
# target) and "standtgt" (the leg-target hold). U_REST_OLD=even: that form on even ground only, the two keys' form
# on uneven ground (rest.py; the keys stay in the build). U_REST_OLD=tilt[,<rad>]: as even, the ground told by the
# base's tilt (rest.TILT) instead of the scan.
REST_OLD_KEYS = ("restlock", "standtgt")
# U_KEYS_ADD=<key>[,<key>...] (a probe's switch, off unless set; teacher_env._probe_switches' docstring): keys of the
# old flow added to the recipe's P keys -- the final environment's; the M keys stay as they are, so a cost an added
# key brings comes in with the stairs quality as every cost the M keys do not have (teacher_env.quality_diff). Those
# below alone: the top landing in front of a wall (2026-11-15, the user on the Go2W teacher's videos: "在平台的时候会
# 出现前面是墙然后做抬腿翻越动作"; blind_env_cfg has each key's history).
ADD_KEYS = ("endwall", "endwallshort", "topflat", "topflat30")
ADD_NEEDS = {"endwallshort": "endwall", "topflat30": "topflat"}   # a key that only changes another's numbers
ADD_WHEELED = ("topflat", "topflat30")                            # a cost on the wheels: nothing on a legged robot
ADD_RECIPE_NEEDS = {"endwall": "longexit"}   # the wall stands at the flight's exit: 0.5 m past the last riser without
                                             # longexit (1.4 m with it), inside the command's stop distance
# U_WALL_FAN=<deg> (a probe's switch, off unless set; teacher_env._probe_switches' docstring): the key endwall's level
# ray fan -- 5 rays at -20..20 deg, any hit inside the stop distance + the slow-down's length cuts the forward
# command -- as 3 rays at -<deg>, 0, <deg>. In the walled flights' narrowest corridor (0.85 m, the hard rows; map3's
# are ~0.9 m) the +-20 deg rays reach the side walls at 1.24 m when the robot stands centred and straight, at 0.85 m
# once it yaws 10 deg: with the Go2W's stop at 0.85 m and endwallshort's 0.2 m the command falls to zero in the
# middle of a flight on every swerve (its climbs' largest heading error: p90 13.5 deg). 2026-11-15, uprobe_s162
# (endwall,endwallshort): the lifts in front of the wall 44 -> 11 %, but the stops on a flight held 70.7 % of the
# time (94.1 %) and the 8 kg climbs fell 188 times of 896 (44) -- measured WITHOUT the fan; the policy's own.
WALL_FAN = "wall_fan"                        # blind_env_cfg's sensor (the key endwall)
WALL_FAN_SWITCH = "U_WALL_FAN"
# U_MODEL=legacy (an evaluation's switch, off unless set; no recipe carries it): the robot as the old flow's packages
# were trained on it -- without the table's model=uros (src/assets/robots/uros.py: the unitree_ros-aligned model file
# and actuator limits, every robot's since 2026-11-11; its rule: a checkpoint is always evaluated and exported on the
# model it trained on). The policies deployed in tinynav-sim and every package without model= in a blind_params.txt
# are the legacy model's; converted for the unified checks (scripts/unified/baseline_ckpt.py copies weights alone)
# they ran on a robot they never trained on -- 2026-11-15, the delivered Go2W blind g14 at 8 kg up map3: 477 first
# falls of 896 robots, 326 of them on the first step (the aligned Go2W's calf torque limit is 35.55 Nm against 45,
# its calf 0.80 kg against 0.241, its wheel 0.52 kg against 0.98), against none at 12 kg in its own acceptance.
# The robot's other parameters (leg gains, leg hold, stairs share cap) stay. A training with it stops
# (no_training_on_legacy): the unified pipeline trains on the aligned model alone.
MODEL_SWITCH = "U_MODEL"


def legacy_model() -> bool:
  v = os.environ.get(MODEL_SWITCH)
  assert v in (None, "", "legacy"), f"{MODEL_SWITCH} {v}: legacy or unset"
  return v == "legacy"


def model_params(params: str) -> str:
  """The robot's BLIND_PARAMS as the build reads them: with U_MODEL=legacy without the model= entry."""
  if not legacy_model():
    return params
  return ",".join(kv for kv in params.split(",") if kv and kv.split("=", 1)[0] != "model")


def robot_motor(robot: "URobot") -> str | None:
  """The robot's motor model as its build reads it (src/assets/robots/uros.py motor; U_MOTOR=builtin: None)."""
  from src.assets.robots import uros

  return uros.motor(model_params(unified_recipe(robot)["env"]))


def no_training_on_legacy(log_dir) -> None:
  """A runner with a log directory is a training's (plays, checks, probes and exports have none): U_MODEL=legacy
  stops it."""
  if log_dir and legacy_model():
    raise RuntimeError(f"{MODEL_SWITCH}=legacy is an evaluation's switch (an old package on the model it trained on): "
                       "the unified pipeline trains on the aligned model alone")


def base_env_cfg(robot: URobot, play: bool = False, terrain: str = "map3", keys: str = "P") -> ManagerBasedRlEnvCfg:
  """The blind environment on map3 at the teacher phase's schedule (early cost ramps, legged walk gate) with the
  unified recipe's final keys, plus the depth teacher's elevated height scan in the actor (depth_env_cfg).
  keys "M": the same with the recipe's M keys -- the environment its climbing was learned in, before the P phase's
  stairs quality (teacher_env.quality_diff reads the two's difference)."""
  from src.tasks.velocity.depth_env_cfg import apply_depth_teacher

  recipe = unified_recipe(robot)
  v = os.environ.get("U_REST_OLD")
  tilt = _rest_tilt(v)
  assert tilt is not None or v in (None, "", "0", "1", "even"), f"U_REST_OLD {v}: 1, even, tilt[,<rad>] or unset"
  drop = DROPPED_KEYS + (REST_OLD_KEYS if v == "1" and robot.kind == "wheeled" else ())
  final = keys == "P"
  add = _keys_add(robot, recipe[keys]) if final else []
  keys = [k for k in recipe[keys] if k not in drop] + add
  env = {
    "BLIND_PILOT": ",".join(keys),
    "BLIND_PAYLOAD": robot.payload,          # no ramp: the progress curriculum scales the payload (curriculum.py)
    "BLIND_WALL_STOP": str(robot.wall_stop_m),
    "BLIND_FOOT_R": None if robot.foot_r_m is None else str(robot.foot_r_m),
    "BLIND_PARAMS": model_params(recipe["env"]),
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
  _wall_fan(cfg, keys, final)
  if play:
    _play_wall_stop(cfg)
  elif terrain == "map3":
    _training_terrain(cfg, recipe["env"])
  if (v == "even" or tilt is not None) and robot.kind == "wheeled" and terrain == "map3":
    # (the M and the P build alike: no difference for teacher_env.quality_diff. The check terrains' builds have none
    # of the stand's lock terms -- the blind builder gives its evaluation tasks no standing hold -- and every task is
    # built at import: uprobe_s120 stopped there at 17:15 on 2026-11-14 before this condition)
    from . import rest

    rest.apply_even(cfg, tilt)
  return cfg


def _wall_fan(cfg: ManagerBasedRlEnvCfg, keys: list[str], final: bool) -> None:
  """U_WALL_FAN=<deg> (WALL_FAN_SWITCH's comment): the end wall's fan narrowed to -<deg>, 0, <deg>; its height, reach
  and alignment and the command's stop distance and slow-down stay. Without the key endwall among the final keys
  the switch stops the build. Nothing where the build has no fan: the M keys' build, the checks' terrains (the
  blind builder adds the wall and its fan to the training terrain alone)."""
  v = os.environ.get(WALL_FAN_SWITCH)
  if not v:
    return
  d = float(v)
  assert 2.0 <= d <= 15.0, f"{WALL_FAN_SWITCH} {v}: the fan's half angle, 2-15 deg"
  assert not final or "endwall" in keys, f"{WALL_FAN_SWITCH} {v}: the fan is the key endwall's (U_KEYS_ADD)"
  fan = next((s for s in cfg.scene.sensors if s.name == WALL_FAN), None)
  if fan is not None:
    fan.pattern = replace(fan.pattern, angles_deg=(-d, 0.0, d))


# The training terrain wider than the tests (2026-11-16, the user via go2-home: "可以"; the 1.25x rule, step 3), a
# robot parameter like the motor model: terrain=wide in BLIND_PARAMS (src/robot_table.py; the Go2W). The training
# build alone (map3's tiles, play=False: the teacher's and the students' trainings); every play / evaluation terrain
# (eval_stairs' bands, map3, slope_rough_test) stays as it is. Today the stairs reach 0.20 m (the tests: 18 cm,
# map3's risers 18.6) and the slopes 0.25 rise / run (14 deg; item 23 tests 15). wide: every stairs sub-terrain's
# step_height_range 0.05-0.23 m and the slopes' slope_range 0-0.36 (19.8 deg); the terrain curriculum's levels spread
# over the wider range (difficulty interpolates it); random_rough unchanged (0.02-0.06 m against the test's +-3 cm).
# U_TERRAIN=base (an evaluation / old-checkpoint switch, off unless set): the training terrain before it.
TERRAIN_SWITCH = "U_TERRAIN"
TERRAINS = {"wide": {"step_height_range": (0.05, 0.23), "slope_range": (0.0, 0.36)}}


def terrain_param(params: str) -> str | None:
  """The robot's training terrain (terrain=... in BLIND_PARAMS); None without it or with U_TERRAIN=base."""
  v = os.environ.get(TERRAIN_SWITCH) or None
  assert v in (None, "base"), f"{TERRAIN_SWITCH}={v!r}: base or unset"
  p = dict(kv.split("=", 1) for kv in params.split(",") if "=" in kv)
  m = p.get("terrain")
  assert m in (None,) + tuple(TERRAINS), f"terrain={m!r}: one of {tuple(TERRAINS)}"
  return None if v == "base" else m


def training_terrain(robot: "URobot") -> str | None:
  """The robot's training terrain as its build reads it (terrain_param of its BLIND_PARAMS)."""
  return terrain_param(unified_recipe(robot)["env"])


def _training_terrain(cfg: ManagerBasedRlEnvCfg, params: str) -> None:
  m = terrain_param(params)
  gen = cfg.scene.terrain.terrain_generator if cfg.scene.terrain is not None else None
  if m is None or gen is None:
    return
  w = TERRAINS[m]
  subs = {k: replace(s, **{f: v for f, v in w.items() if hasattr(s, f)}) for k, s in gen.sub_terrains.items()}
  # (new objects all the way up: the play build of the same import must not see them)
  cfg.scene.terrain = replace(cfg.scene.terrain, terrain_generator=replace(gen, sub_terrains=subs))


# U_WALL_STOP_PLAY=1 (an evaluation's diagnostic switch, off unless set): a play build keeps the key endwall's fan and
# the command's wall stop. Off (2026-11-16, go2-home for the user's acceptance): every play / evaluation build drops
# them -- every model is judged on the same commands, and the deployed wall guard is outside the policy (a run with
# endwall had map3's op / noop and its tours cut 0.85 m before every wall ahead, its controls not). Training keeps them.
WALL_STOP_PLAY_SWITCH = "U_WALL_STOP_PLAY"


def _play_wall_stop(cfg: ManagerBasedRlEnvCfg) -> None:
  v = os.environ.get(WALL_STOP_PLAY_SWITCH) or "0"
  assert v in ("0", "1"), f"{WALL_STOP_PLAY_SWITCH}={v}: 1 or unset"
  if v == "1":
    return
  twist = cfg.commands.get("twist")
  if twist is not None and getattr(twist, "wall_sensor_name", None) == WALL_FAN:
    twist.wall_sensor_name = None
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != WALL_FAN)


def _keys_add(robot: URobot, have: list[str]) -> list[str]:
  """U_KEYS_ADD's keys (ADD_KEYS' comment), checked: each of ADD_KEYS, once, not in the recipe already, with the key
  it changes (ADD_NEEDS) there or among them, a wheel cost on a wheeled robot alone, and the recipe key it stands on
  (ADD_RECIPE_NEEDS) in the recipe. [] without the switch."""
  v = os.environ.get("U_KEYS_ADD")
  if not v:
    return []
  add = v.split(",")
  assert len(set(add)) == len(add) and set(add) <= set(ADD_KEYS), f"U_KEYS_ADD {v}: of {', '.join(ADD_KEYS)}, once"
  for k in add:
    assert k not in have, f"U_KEYS_ADD {v}: {k} is in {robot.name}'s recipe already"
    assert k not in ADD_NEEDS or ADD_NEEDS[k] in add or ADD_NEEDS[k] in have, f"U_KEYS_ADD {v}: {k} needs {ADD_NEEDS[k]}"
    assert k not in ADD_WHEELED or robot.kind == "wheeled", f"U_KEYS_ADD {v}: {k} is a wheeled robot's"
    assert k not in ADD_RECIPE_NEEDS or ADD_RECIPE_NEEDS[k] in have, \
      f"U_KEYS_ADD {v}: {k} needs {ADD_RECIPE_NEEDS[k]} in {robot.name}'s recipe"
  return add


def _rest_tilt(v: str | None) -> float | None:
  """U_REST_OLD=tilt[,<rad>]: the tilt from which the lock form counts (rest.TILT without the field); None for
  every other value."""
  f = (v or "").split(",")
  if f[0] != "tilt":
    return None
  from . import rest

  assert len(f) <= 2, f"U_REST_OLD {v}: tilt[,<rad>]"
  tilt = float(f[1]) if len(f) > 1 else rest.TILT
  assert 0.03 <= tilt <= 0.30, f"U_REST_OLD {v}: the tilt 0.03-0.30 rad"
  return tilt
