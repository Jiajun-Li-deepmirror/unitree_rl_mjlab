"""Unitree B2W (wheel-legged B2) constants.

xmls/b2w.xml is converted from unitree_mujoco (unitree_robots/b2w): B2 legs with a
velocity-controlled wheel at the end of each calf. Wheel collisions are spheres of the
wheel radius (0.113 m, measured from the wheel meshes; the tread is 24 mm wide), centered
where the visual meshes and wheel inertias are (upstream's collision meshes were shifted
by -0.0458 m in y on every wheel). Spheres, not cylinders: see go2w_constants.py.
xmls/scene_b2w.xml is the unitree_mujoco simulation scene (upstream motors, sensors and
joint dynamics, mesh wheels) for sim2sim.
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

# Legs: B2 motors (unitree_rl_lab: 200 Nm hip/thigh, kp 160, kd 5; upstream MJCF limits
# the calf to 300 Nm) and the upstream MJCF armature.
B2W_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=160.0,
  damping=5.0,
  effort_limit=200.0,
  armature=0.1,
)
B2W_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=160.0,
  damping=5.0,
  effort_limit=200.0,
  armature=0.1,
)
B2W_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=160.0,
  damping=5.0,
  effort_limit=300.0,
  armature=0.1,
)
# Wheels: torque = damping * (target velocity - velocity), like the real robot's dq
# command with kp = 0. 40 Nm is the maximum wheel torque on unitree.com/b2-w (the
# upstream MJCF ctrlrange says 20 Nm). No reference value exists for the gain:
# Go2W's 0.5 scaled by the mass x wheel radius ratio (~4-6x) gives 2-3; 2.0
# saturates the torque at a 20 rad/s (2.3 m/s) velocity error. Armature as upstream.
# An explicit PD with zero stiffness: mjlab's BuiltinVelocityActuator inherits its
# control range from the joint range, which a continuous wheel joint does not have.
B2W_ACTUATOR_WHEEL = IdealPdActuatorCfg(
  target_names_expr=(".*_wheel_joint",),
  stiffness=0.0,
  damping=2.0,
  effort_limit=40.0,
  armature=0.1,
)

##
# Keyframes.
##

# Same leg angles as Go2W (unitree_rl_lab's B2 deploy also stands at 0, 0.8, -1.5).
# The wheels touch the ground at a base height of 0.6245 m; spawn 2 cm above.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.645),
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
  """Get a fresh B2W robot configuration instance.

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
