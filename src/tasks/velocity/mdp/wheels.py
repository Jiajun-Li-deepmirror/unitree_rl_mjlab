"""Rewards for wheel-legged robots (Go2-W, B2-W, As2-W).

The wheels are the feet: on even ground, going forward or back (also in
arcs), the robot should roll with its legs still ("roll mode"); on stairs
or uneven ground, to move sideways or to turn in place, it steps like the
legged robots ("step mode", where the legged gait rewards apply). The mode comes from the privileged height scan
and the command, so only rewards use it; the policy has to infer it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .rewards import feet_gait, uneven_terrain_mask

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor


def step_mode_mask(
  env: ManagerBasedRlEnv,
  command_name: str,
  scan_sensor_name: str,
  lateral_threshold: float = 0.15,
  uneven_radius: float = 0.45,
  uneven_threshold: float = 0.03,
) -> torch.Tensor:
  """Per-env bool: the robot should step (uneven ground, a sideways command or
  turning in place).

  Ground counts as uneven from a 3 cm span (legged robots: 5 cm): the wheels
  roll over less, and with 5 cm the lowest stair rows (5-6 cm rises) were
  half roll mode, where the first full Go2-W teacher froze.

  Skid-steering in place reaches only ~30% of the commanded yaw rate on the
  Go2-W (the wheels scrub sideways), and go2-blind turns in place by stepping.
  Arcs (forward speed plus yaw) stay on the wheels.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  lateral = command[:, 1].abs() > lateral_threshold
  turn_in_place = (command[:, 0].abs() < lateral_threshold) & (command[:, 2].abs() > 0.2)
  uneven = uneven_terrain_mask(env, scan_sensor_name, uneven_radius, uneven_threshold)
  return lateral | turn_in_place | uneven


def _moving(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :2], dim=1) + command[:, 2].abs() > threshold


def feet_gait_stepping(
  env: ManagerBasedRlEnv,
  scan_sensor_name: str,
  lateral_threshold: float = 0.15,
  **gait_params,
) -> torch.Tensor:
  """`feet_gait` (trot phase), only in step mode."""
  step = step_mode_mask(env, gait_params["command_name"], scan_sensor_name, lateral_threshold)
  return feet_gait(env, **gait_params) * step.float()


def wheel_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  wheel_radius: float,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Squared slip speed at the contact point of each wheel on the ground.

  `asset_cfg` holds the wheel centre sites and the wheel bodies (same order).
  The contact point is taken `wheel_radius` straight below the centre (level
  ground), so rolling without slipping costs nothing, unlike `feet_slip`,
  which charges a rolling wheel for its centre moving.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  in_contact = (contact.data.found > 0).float()
  v_center = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]  # [B, N, 3]
  omega = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]  # [B, N, 3]
  down = torch.zeros_like(omega)
  down[..., 2] = -wheel_radius
  v_contact = v_center + torch.cross(omega, down, dim=-1)
  slip_sq = torch.sum(torch.square(v_contact[..., :2]), dim=-1)
  active = _moving(env, command_name, command_threshold).float()
  env.extras["log"]["Metrics/wheel_slip_mean"] = torch.sum(
    slip_sq.sqrt() * in_contact
  ) / torch.clamp(in_contact.sum(), min=1)
  return torch.sum(slip_sq * in_contact, dim=1) * active


def roll_leg_motion(
  env: ManagerBasedRlEnv,
  command_name: str,
  scan_sensor_name: str,
  command_threshold: float = 0.1,
  lateral_threshold: float = 0.15,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Squared leg joint speeds while moving in roll mode (legs should be still)."""
  asset: Entity = env.scene[asset_cfg.name]
  cost = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
  roll = ~step_mode_mask(env, command_name, scan_sensor_name, lateral_threshold)
  return cost * (roll & _moving(env, command_name, command_threshold)).float()


def roll_wheels_lifted(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  scan_sensor_name: str,
  command_threshold: float = 0.1,
  lateral_threshold: float = 0.15,
) -> torch.Tensor:
  """Fraction of wheels off the ground while moving in roll mode (a penalty).

  Roll mode only carries penalties, so rolling properly costs nothing and
  neither mode pays more than the other. With positive roll-mode rewards
  (wheels down, wheel speeds) the full Go2-W teacher learned to farm them:
  it rolled up to the first riser, where step mode took them away, backed
  off onto flat ground and rolled up again, never stepping up.
  """
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  lifted = 1.0 - torch.mean((contact.data.found > 0).float(), dim=1)
  roll = ~step_mode_mask(env, command_name, scan_sensor_name, lateral_threshold)
  moving = _moving(env, command_name, command_threshold)
  env.extras["log"]["Metrics/roll_mode_frac"] = (roll & moving).float().mean()
  return lifted * (roll & moving).float()


def roll_leg_deviation(
  env: ManagerBasedRlEnv,
  command_name: str,
  scan_sensor_name: str,
  command_threshold: float = 0.1,
  lateral_threshold: float = 0.15,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Squared leg deviation from the default pose while moving in roll mode."""
  asset: Entity = env.scene[asset_cfg.name]
  diff = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  cost = torch.sum(torch.square(diff), dim=1)
  roll = ~step_mode_mask(env, command_name, scan_sensor_name, lateral_threshold)
  return cost * (roll & _moving(env, command_name, command_threshold)).float()
