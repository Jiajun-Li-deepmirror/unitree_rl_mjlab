"""Unitree B2 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# Geometry from unitree_mujoco's B2 model; link masses and inertias (74.6 kg in
# total) from unitree_ros b2_description, with fixed-joint links (motor rotors,
# feet, head/tail and sensor links) merged into their parent bodies.
B2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_b2" / "xmls" / "b2.xml"
)
assert B2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, B2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(B2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Gains from unitree_rl_lab (UNITREE_B2_CFG); effort limits from the unitree_mujoco
# B2 motors (unitree_rl_lab lists 320 Nm for the calf, the MJCF 300 Nm); armature
# is the unitree_mujoco joint default.
B2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=160.0,
  damping=5.0,
  effort_limit=200,
  armature=0.1,
)
B2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=160.0,
  damping=5.0,
  effort_limit=200,
  armature=0.1,
)
B2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=160.0,
  damping=5.0,
  effort_limit=300,
  armature=0.1,
)

##
# Keyframes.
##

# Default pose from unitree_rl_lab; at this pose the front feet touch the ground
# with the base at 0.529 m (the rear feet are 1.5 cm higher).
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.53),
  joint_pos={
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "F[LR]_thigh_joint": 0.8,
    "R[LR]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
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

B2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    B2_ACTUATOR_HIP,
    B2_ACTUATOR_THIGH,
    B2_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_b2_robot_cfg() -> EntityCfg:
  """Get a fresh B2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=B2_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_b2_robot_cfg())

  viewer.launch(robot.spec.compile())
