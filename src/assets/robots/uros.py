"""The robot models aligned with Unitree's own descriptions (go2-depth 2026-11-11, the user: "机器参数从
https://github.com/unitreerobotics/unitree_ros 里面获取 ... 先和unitree ros对齐，没有的参数再和官网对齐").

Selected by the robot parameter `model=uros` in BLIND_PARAMS (src/robot_table.py -> jobs/recipes/unified.py ->
the chains; every run keeps it in run_env.sh and every blind package in blind_params.txt, so a checkpoint is always
evaluated and exported on the model it trained on). Without it a robot is the model it always was.

With it:
  - xmls/<r>_uros.xml (tools/robot_params/align_uros.py): every body's mass, centre of mass and inertia from the
    URDF in unitree_<r>/uros/ (unitree_ros 725b59b), the links on fixed joints (motor rotors, wheel motors, head,
    lidar) folded into the body that carries them; As2 / As2W leg joints get Unitree's as2.xml passive damping;
  - the actuators below: the URDF's <limit effort> where ours differed, Unitree's as2.xml joint friction loss on the
    As2 / As2W legs (the As2W URDF gives none; its legs are the As2's).
Unchanged (the same in unitree_ros or not given there): kinematics, joint ranges, foot / wheel radii, armature, the
PD gains. Not modelled: the URDF <limit velocity> (no torque-speed curve is published to make a motor model of it).
"""

from __future__ import annotations

import dataclasses
import functools
import os
from pathlib import Path

# per robot, per actuator group (hip / thigh / calf / wheel): the fields that differ from the legacy model
ACTUATORS: dict[str, dict[str, dict[str, float]]] = {
  "go2": {"hip": {"effort_limit": 23.7}, "thigh": {"effort_limit": 23.7}, "calf": {"effort_limit": 45.43}},
  "go2w": {"hip": {"effort_limit": 23.7}, "thigh": {"effort_limit": 23.7}, "calf": {"effort_limit": 35.55},
           "wheel": {"effort_limit": 23.7}},
  "as2": {"hip": {"frictionloss": 1.0}, "thigh": {"frictionloss": 1.0}, "calf": {"frictionloss": 2.0}},
  "as2w": {"hip": {"frictionloss": 1.0}, "thigh": {"frictionloss": 1.0}, "calf": {"frictionloss": 2.0}},
  "b2": {"calf": {"effort_limit": 320.0}},
  "b2w": {"calf": {"effort_limit": 320.0}, "wheel": {"effort_limit": 20.0}},
}


def _params(params: str | None = None) -> dict[str, str]:
  s = os.environ.get("BLIND_PARAMS", "") if params is None else params
  return dict(kv.split("=", 1) for kv in s.split(",") if "=" in kv)


def active(params: str | None = None) -> bool:
  return _params(params).get("model") == "uros"


# The motor model (2026-11-16; the user, via go2-home: the motor and joint values from Unitree's own sources, then
# "可以不用那么准确，取一个中间值就行了或者平均" -- no torque-speed curve, the built-in position actuators unchanged).
# Selected by the robot parameter motor=<name> in BLIND_PARAMS (src/robot_table.py), on the aligned model only.
# gom8010, the Go2W's GO-M8010-6 joint motors (unitree_rl_lab UnitreeActuatorCfg_Go2HV: 20.2 N m up to the corner,
# 23.4 at stall): the torque limit their mean 21.8 N m (hip, thigh, wheel), the calf x 1.5 (its linkage) 32.7; the
# leg joints' friction loss 0.2 N m, passive damping 0.1 N m s/rad and armature 0.01 kg m^2 (unitree_mujoco go2.xml,
# the armature in place of 0.01 / 0.01 / 0.02); the teacher's joint randomization draws friction and damping x 0.5-1.5
# and armature x 0.7-1.3 around them (teacher_env._add_joint_dr). Before it: today's limits 23.7 / 35.55 / 23.7, no
# joint friction or damping in the model, the randomization from 0.
# U_MOTOR=builtin (an evaluation's switch, off unless set; no recipe carries it): without the motor= entry -- a
# checkpoint trained before the motor model (a run whose BLIND_PARAMS has no motor=) is evaluated on the model it
# trained on. Rule: every Go2W run before 2026-11-16 ~19:00 (go2w_min_s0 and earlier, the formal teachers and
# students) trained without it.
MOTOR_SWITCH = "U_MOTOR"
_LEG = {"frictionloss": 0.2, "armature": 0.01}
MOTORS: dict[str, dict[str, dict[str, dict[str, float]]]] = {
  "gom8010": {"go2w": {"hip": {"effort_limit": 21.8, **_LEG}, "thigh": {"effort_limit": 21.8, **_LEG},
                       "calf": {"effort_limit": 32.7, **_LEG}, "wheel": {"effort_limit": 21.8}}},
}
MOTOR_JOINT_DAMPING = {"gom8010": 0.1}   # passive, every leg joint (hip / thigh / calf)
MOTOR_JOINT_DR = {"gom8010": {"friction": (0.5, 1.5), "damping": (0.5, 1.5), "armature": (0.7, 1.3)}}
# U_MOTOR_PART=effort|joint (2026-11-17, go2-home's forward tests F1a / F1b: the motor model is the common factor of
# the runs that stop climbing): one half of the motor model, the other half built-in. effort: the torque limits alone
# (the joints' friction, armature, passive damping and their randomization as built-in -- the built-in randomization
# is a share of the effort limit, so it follows the new limits); joint: the friction loss, armature, passive damping
# and the scaled randomization alone (the built-in torque limits). Not with U_MOTOR=builtin.
MOTOR_PART_SWITCH = "U_MOTOR_PART"


def motor_part() -> str | None:
  """U_MOTOR_PART's half of the motor model (None: the whole of it)."""
  v = os.environ.get(MOTOR_PART_SWITCH) or None
  assert v in (None, "effort", "joint"), f"{MOTOR_PART_SWITCH}={v!r}: effort, joint or unset"
  assert v is None or (os.environ.get(MOTOR_SWITCH) or None) is None, f"{MOTOR_PART_SWITCH}: not with {MOTOR_SWITCH}=builtin"
  return v


def _part(over: dict[str, float]) -> dict[str, float]:
  """A joint group's motor overrides, U_MOTOR_PART's half of them."""
  p = motor_part()
  if p == "effort":
    return {k: v for k, v in over.items() if k == "effort_limit"}
  if p == "joint":
    return {k: v for k, v in over.items() if k != "effort_limit"}
  return over


def motor(params: str | None = None) -> str | None:
  """The motor model the robot's BLIND_PARAMS name (motor=...) on the aligned model; None without it, with the legacy
  model, or with U_MOTOR=builtin."""
  v = os.environ.get(MOTOR_SWITCH) or None
  assert v in (None, "builtin"), f"{MOTOR_SWITCH}={v!r}: builtin or unset"
  m = _params(params).get("motor")
  assert m in (None,) + tuple(MOTORS), f"motor={m!r}: one of {tuple(MOTORS)}"
  motor_part()   # (checks U_MOTOR_PART on every build, also beside U_MOTOR=builtin)
  if v == "builtin" or not active(params):
    return None
  return m


def _with_joint_damping(fn, damping: float):
  spec = fn()
  for j in spec.joints:
    if any(g in j.name for g in ("hip", "thigh", "calf")):
      j.damping = damping
  return spec


def spec_fn(robot: str, fn):
  """The robot's spec function with the motor model's passive joint damping on the leg joints."""
  m = motor()
  if m is None or robot not in MOTORS[m] or motor_part() == "effort":
    return fn
  return functools.partial(_with_joint_damping, fn, MOTOR_JOINT_DAMPING[m])


def xml(path: Path) -> Path:
  """The model file to load: <r>_uros.xml beside <r>.xml when the aligned model is selected."""
  if not active():
    return path
  p = path.with_name(path.stem + "_uros.xml")
  if not p.exists():
    raise FileNotFoundError(f"{p} (tools/robot_params/align_uros.py builds it)")
  return p


def _group(cfg) -> str | None:
  expr = " ".join(getattr(cfg, "target_names_expr", ()))
  return next((g for g in ("wheel", "calf", "thigh", "hip") if g in expr), None)


def articulation(robot: str, art):
  """The robot's articulation config with the aligned actuator settings when the aligned model is selected."""
  if not active():
    return art
  over = ACTUATORS.get(robot, {})
  acts = tuple(dataclasses.replace(a, **over.get(_group(a), {})) for a in art.actuators)
  m = motor()
  if m is not None and robot in MOTORS[m]:
    over = MOTORS[m][robot]
    acts = tuple(dataclasses.replace(a, **_part(over.get(_group(a), {}))) for a in acts)
  return dataclasses.replace(art, actuators=acts)
