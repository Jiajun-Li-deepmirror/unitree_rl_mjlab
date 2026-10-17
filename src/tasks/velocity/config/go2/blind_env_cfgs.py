"""Unitree Go2 blind (proprioception-only) velocity configurations.

The generic builder lives in config/legged/blind.py; this module holds the
Go2's LeggedTaskSpec, which the depth tasks share.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_go2_robot_cfg
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_env_cfg
from src.tasks.velocity.config.legged.blind import make_blind_env_cfg
from src.tasks.velocity.config.legged.spec import LeggedTaskSpec

GO2_SPEC = LeggedTaskSpec(
  name="go2",
  rough_env_cfg=unitree_go2_rough_env_cfg,
  robot_cfg=get_go2_robot_cfg,
  # tinynav-sim's PIE contract, plus the gait phase (go2_depth_v2).
  onnx_contract="go2_depth_v2",
)


def unitree_go2_blind_env_cfg(
  play: bool = False, stairs: bool = True
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go2 blind velocity configuration."""
  return make_blind_env_cfg(GO2_SPEC, play=play, stairs=stairs)
