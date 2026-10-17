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
  uneven_threshold: float = 0.05,
) -> torch.Tensor:
  """Per-env bool: the robot should step (uneven ground, a sideways command or
  turning in place).

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


def roll_wheel_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  scan_sensor_name: str,
  command_threshold: float = 0.1,
  lateral_threshold: float = 0.15,
) -> torch.Tensor:
  """Fraction of wheels on the ground while moving in roll mode."""
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  ratio = torch.mean((contact.data.found > 0).float(), dim=1)
  roll = ~step_mode_mask(env, command_name, scan_sensor_name, lateral_threshold)
  moving = _moving(env, command_name, command_threshold)
  env.extras["log"]["Metrics/roll_mode_frac"] = (roll & moving).float().mean()
  return ratio * (roll & moving).float()


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


def wheel_roll_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  scan_sensor_name: str,
  wheel_radius: float,
  std: float,
  command_threshold: float = 0.1,
  lateral_threshold: float = 0.15,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """exp(-mean wheel speed error^2 / std^2) against the commanded vx and yaw
  rate (differential drive), in roll mode while moving.

  `asset_cfg` holds the wheel joints and the wheel centre sites (same order);
  a wheel `y` (base frame) to the left must turn at (vx - wz y) / r.
  """
  from mjlab.utils.lab_api.math import quat_apply_inverse

  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  rel = asset.data.site_pos_w[:, asset_cfg.site_ids] - asset.data.root_link_pos_w[:, None]
  quat = asset.data.root_link_quat_w[:, None].expand(-1, rel.shape[1], -1)
  y = quat_apply_inverse(quat, rel)[..., 1]
  target = (command[:, :1] - command[:, 2:3] * y) / wheel_radius
  speed = asset.data.joint_vel[:, asset_cfg.joint_ids]
  err = torch.mean(torch.square(speed - target), dim=1)
  roll = ~step_mode_mask(env, command_name, scan_sensor_name, lateral_threshold)
  active = roll & _moving(env, command_name, command_threshold)
  return torch.exp(-err / std**2) * active.float()
