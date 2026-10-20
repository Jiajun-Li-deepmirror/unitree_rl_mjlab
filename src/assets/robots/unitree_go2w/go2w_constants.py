"""Unitree Go2W (wheel-legged Go2) constants.

xmls/go2w.xml is converted from unitree_mujoco (unitree_robots/go2w): Go2 legs with a
velocity-controlled wheel at the end of each calf. Wheel collisions are spheres of the
wheel radius (0.086 m, measured from the wheel mesh; the tread is 24 mm wide) at the
wheel center. Cylinders sank into the seams between the flush terrain boxes and blew
up the simulation (NaN): MuJoCo collides cylinders with boxes by its general convex
method, which at a seam pushes the wheel sideways out of both boxes instead of up.
Sphere-box contacts are exact and push up. xmls/scene_go2w.xml is the unitree_mujoco
simulation scene (upstream motors, sensors and joint dynamics, mesh wheels) for sim2sim.
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

GO2W_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_go2w" / "xmls" / "go2w.xml"
)
assert GO2W_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, GO2W_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2W_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Legs: same motors and gains as Go2 (go2_constants.py).
GO2W_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2W_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2W_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45,
  armature=0.02,
)
# Wheels: torque = damping * (target velocity - velocity), like the real robot's dq
# command with kp = 0. kd 0.5 as in unitree_rl_lab's Go2W; 15 Nm is the upstream MJCF
# ctrlrange (unitree_rl_lab uses 23.5 for every joint). An explicit PD with zero
# stiffness: mjlab's BuiltinVelocityActuator inherits its control range from the joint
# range, which a continuous wheel joint does not have.
GO2W_ACTUATOR_WHEEL = IdealPdActuatorCfg(
  target_names_expr=(".*_wheel_joint",),
  stiffness=0.0,
  damping=0.5,
  effort_limit=15.0,
  armature=0.01,
)

##
# Keyframes.
##

# Stance of unitree_rl_lab's Go2W. The wheels touch the ground at a base height of
# 0.4076 m; spawn 2 cm above.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.43),
  joint_pos={
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.8,
    ".*_calf_joint": -1.5,
    ".*_wheel_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# The wheels.
_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the wheels.
# Furthermore, wheel self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Wheel collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.8,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

GO2W_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2W_ACTUATOR_HIP,
    GO2W_ACTUATOR_THIGH,
    GO2W_ACTUATOR_CALF,
    GO2W_ACTUATOR_WHEEL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2w_robot_cfg() -> EntityCfg:
  """Get a fresh Go2W robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2W_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2w_robot_cfg())

  viewer.launch(robot.spec.compile())
