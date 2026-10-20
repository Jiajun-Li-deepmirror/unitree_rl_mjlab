"""Unitree As2W (As2 with wheels) constants.

Built from the unitree_ros As2W URDF (robots/as2w_description): legs have the
As2 kinematics, a wheel motor at the end of each calf drives a 0.086 m wheel.
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

As2W_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_as2w" / "xmls" / "as2w.xml"
)
assert As2W_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, As2W_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(As2W_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Leg gains and armature as the As2; effort limits from the As2W URDF (60 / 60 /
# 90 Nm, the same as the unitree_mujoco As2 motors).
As2W_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=60,
  armature=0.026,
)
As2W_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=60,
  armature=0.026,
)
As2W_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=60.0,
  damping=3.0,
  effort_limit=90,
  armature=0.038,
)


# Wheels track a velocity target: torque = damping * (target - velocity), limited
# to the URDF 15 Nm, as the real motor does with kp = 0. An explicit PD with zero
# stiffness: mjlab's BuiltinVelocityActuator inherits its control range from the
# joint range, which a continuous wheel joint does not have. The rotor inertia is
# not published; 0.005 is a guess.
As2W_ACTUATOR_WHEEL = IdealPdActuatorCfg(
  target_names_expr=(".*_wheel_joint",),
  stiffness=0.0,
  damping=1.0,
  effort_limit=15.0,
  armature=0.005,
)

WHEEL_RADIUS = 0.086

##
# Keyframes.
##


# As2 leg pose; wheels 6 mm above the ground (they touch at a base height of
# 0.364 m), so the robot settles onto them after a reset.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.37),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
    ".*_wheel_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# The wheels are the feet: {leg}_foot_collision spheres of the wheel radius (cylinders
# sink into the seams between terrain boxes, see go2w_constants.py). The sim2sim scene
# (scene_as2w.xml, plane floor) keeps the cylinders.
_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the wheels.
# Furthermore, wheel self collisions are disabled.
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
# Wheel collisions are given custom condim, friction and solimp.
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

As2W_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    As2W_ACTUATOR_HIP,
    As2W_ACTUATOR_THIGH,
    As2W_ACTUATOR_CALF,
    As2W_ACTUATOR_WHEEL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_as2w_robot_cfg() -> EntityCfg:
  """Get a fresh As2W robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=As2W_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_as2w_robot_cfg())

  viewer.launch(robot.spec.compile())
