"""Unitree B2-W constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from src.assets.robots.wheel_actuator import WheelVelocityActuatorCfg

##
# MJCF and assets.
##

B2W_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_b2w" / "xmls" / "b2w.xml"
)
assert B2W_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, B2W_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(B2W_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# kp 160 / kd 5 on every leg joint as go2-blind's deployed B2W v5 (200 / 10,
# calf 240 / 12 before).
B2W_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=160.0,
  damping=5.0,
  effort_limit=200.0,
  armature=0.1,
)
B2W_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=160.0,
  damping=5.0,
  effort_limit=200.0,
  armature=0.1,
)
B2W_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=160.0,
  damping=5.0,
  effort_limit=300.0,
  armature=0.1,
)
# Velocity-controlled wheels (radius 0.113 m). go2-blind's gain, armature and
# 40 Nm limit (its deployed B2W v5; unitree_mujoco's b2w motors are +-20 Nm):
# at a zero command the low-speed brake holds the robot.
B2W_ACTUATOR_WHEEL = WheelVelocityActuatorCfg(
  target_names_expr=(".*_wheel_joint",),
  velocity_limit=40.0,
  damping=2.0,
  effort_limit=40.0,
  armature=0.1,
)
# The wheels collide as spheres of this radius (*_foot_collision), not as
# cylinders: on box seams and step edges the GJK/EPA cylinder contact could
# give only sideways normals and let a wheel sink in (rare NaN observations
# on the first Go2-W runs), while the analytic sphere-box contact always
# supports the wheel from below. Spheres are wider than the real wheels.
B2W_WHEEL_RADIUS = 0.113

##
# Keyframes.
##


# go2-blind's wheeled default pose (thigh 0.9 / calf -1.8 before).
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.645),
  joint_pos={
    ".*thigh_joint": 0.8,
    ".*calf_joint": -1.5,
    ".*hip_joint": 0.0,
    ".*_wheel_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

B2W_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    B2W_ACTUATOR_HIP,
    B2W_ACTUATOR_THIGH,
    B2W_ACTUATOR_CALF,
    B2W_ACTUATOR_WHEEL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_b2w_robot_cfg() -> EntityCfg:
  """Get a fresh B2-W robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=B2W_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_b2w_robot_cfg())

  viewer.launch(robot.spec.compile())
