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


def active() -> bool:
  params = dict(kv.split("=", 1) for kv in os.environ.get("BLIND_PARAMS", "").split(",") if "=" in kv)
  return params.get("model") == "uros"


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
  return dataclasses.replace(art, actuators=acts)
