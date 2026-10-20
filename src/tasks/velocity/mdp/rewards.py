from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, wrap_to_pi
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  z_weight: float = 2.0,
  even_std: float | None = None,
  scan_sensor_name: str | None = None,
  min_std: float | None = None,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero. Set `z_weight` to 0 on
  terrain where vertical base motion is expected (e.g. stairs). With
  `even_std` and `scan_sensor_name`, ground that is not uneven (see
  uneven_terrain_mask) uses the tighter `even_std`. With `min_std` instead
  (go2-blind's wheeled stairs), the std shrinks with the commanded planar speed
  to clamp(|v_cmd,xy|, min_std, std): standing still then keeps exp(-1) of the
  reward at any command of at least `min_std`, not most of it at slow ones.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (z_weight * z_error)
  if min_std is not None:
    assert even_std is None, "min_std and even_std are exclusive."
    std_t = torch.clamp(torch.norm(command[:, :2], dim=1), min=min_std, max=std)
    return torch.exp(-lin_vel_error / std_t**2)
  if even_std is None or scan_sensor_name is None:
    return torch.exp(-lin_vel_error / std**2)
  uneven = uneven_terrain_mask(env, scan_sensor_name)
  std_t = torch.where(uneven, std, even_std)
  return torch.exp(-lin_vel_error / std_t**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_std: float | None = None,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  if min_std is not None:
    # The yaw std shrinks with the command (go2-blind wheeled): at std 0.45 not
    # turning at all for 0.15 rad/s cost 0.1 per second, at 0.2 it costs 0.43.
    z_std = torch.clamp(command[:, 2].abs(), min=min_std, max=std)
    return torch.exp(-z_error / z_std**2 - 0.05 * xy_error / std**2)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward


##
# Terrain-aware terms for blind locomotion.
##


def uneven_terrain_mask(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  radius: float = 0.45,
  threshold: float = 0.05,
  max_height_above_base: float = 0.5,
) -> torch.Tensor:
  """Per-env bool: terrain height span within `radius` of the base exceeds `threshold`.

  Uses the (privileged) height-scan raycast, so it reflects the terrain the robot
  is currently on rather than the column it was spawned in. Hits more than
  `max_height_above_base` above the base (wall tops seen by an elevated scan)
  are ignored: a wall next to flat ground is not uneven ground.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  data = sensor.data
  hit_z = data.hit_pos_w[..., 2]
  planar_dist = torch.norm(data.hit_pos_w[..., :2] - data.pos_w[:, None, :2], dim=-1)
  below = hit_z <= data.pos_w[:, None, 2] + max_height_above_base
  near = (planar_dist <= radius) & (data.distances >= 0.0) & below
  z_max = torch.where(near, hit_z, -torch.inf).amax(dim=1)
  z_min = torch.where(near, hit_z, torch.inf).amin(dim=1)
  span = torch.where(near.any(dim=1), z_max - z_min, 0.0)
  return span > threshold


def stair_speed_deficit(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  ratio: float = 0.9,
  min_speed: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """On uneven ground (stairs), how far the speed along the commanded direction
  falls short of `ratio` times the commanded speed, as a fraction of the command
  (0 to 1). Going faster than that earns nothing, so it cannot pay to rush
  (go2-blind ea6aae8).

  go2-blind's wheeled stairs policies went down map3's flights at 0.57-0.84x
  the command; 300 iterations with this (weight -1) took them to 0.80-1.02x.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd = command[:, :2]
  speed = torch.norm(cmd, dim=1)
  direction = cmd / speed.clamp(min=1e-6).unsqueeze(1)
  along = (asset.data.root_link_lin_vel_b[:, :2] * direction).sum(dim=1)
  deficit = ((ratio * speed - along) / speed.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
  active = (speed >= min_speed) & uneven_terrain_mask(env, sensor_name)
  return torch.where(active, deficit, torch.zeros_like(deficit))


def _zero_command_mask(
  env: ManagerBasedRlEnv, command_name: str, command_threshold: float
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return total_command <= command_threshold


def _standing_scale(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  calm_std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """1 while commanded to stand and the base is calm, fading out while the base moves.

  The fade lets the robot take recovery steps after a push without being
  penalized by the standing terms.
  """
  asset: Entity = env.scene[asset_cfg.name]
  lin_vel_xy = asset.data.root_link_lin_vel_b[:, :2]
  ang_vel_z = asset.data.root_link_ang_vel_b[:, 2]
  base_motion = torch.sum(torch.square(lin_vel_xy), dim=1) + torch.square(ang_vel_z)
  calm = torch.exp(-base_motion / calm_std)
  return _zero_command_mask(env, command_name, command_threshold).float() * calm


def stand_still_even(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.1,
  calm_std: float = 0.1,
  uneven_radius: float = 0.45,
  uneven_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from the default pose while standing on even ground."""
  asset: Entity = env.scene[asset_cfg.name]
  diff = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  cost = torch.sum(torch.square(diff), dim=1)
  scale = _standing_scale(env, command_name, command_threshold, calm_std, asset_cfg)
  even = ~uneven_terrain_mask(env, sensor_name, uneven_radius, uneven_threshold)
  return cost * scale * even.float()


def standing_joint_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  calm_std: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint motion while commanded to stand."""
  asset: Entity = env.scene[asset_cfg.name]
  cost = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
  return cost * _standing_scale(env, command_name, command_threshold, calm_std, asset_cfg)


def standing_feet_contact(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.1,
  calm_std: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the fraction of feet on the ground while commanded to stand."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  contact_ratio = torch.mean((contact_sensor.data.found > 0).float(), dim=1)
  return contact_ratio * _standing_scale(
    env, command_name, command_threshold, calm_std, asset_cfg
  )


def body_orientation_terrain_aware(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  uneven_scale: float = 0.3,
  uneven_radius: float = 0.45,
  uneven_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """body_orientation_l2, scaled down on uneven terrain where pitching is expected."""
  cost = body_orientation_l2(env, asset_cfg)
  uneven = uneven_terrain_mask(env, sensor_name, uneven_radius, uneven_threshold)
  return cost * torch.where(uneven, uneven_scale, 1.0)


def lin_vel_z_even(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  uneven_radius: float = 0.45,
  uneven_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize vertical base velocity on even ground only."""
  asset: Entity = env.scene[asset_cfg.name]
  cost = torch.square(asset.data.root_link_lin_vel_b[:, 2])
  even = ~uneven_terrain_mask(env, sensor_name, uneven_radius, uneven_threshold)
  return cost * even.float()


def feet_stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  ratio: float = 2.0,
  min_force: float = 5.0,
) -> torch.Tensor:
  """Count feet pushing mostly horizontally (e.g. toes hitting a stair riser).

  Requires a world-frame force (reduce="netforce" or global_frame=True).
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  force = contact_sensor.data.force
  assert force is not None
  force_xy = torch.norm(force[..., :2], dim=-1)
  force_z = torch.abs(force[..., 2])
  stumble = force_xy > torch.clamp(ratio * force_z, min=min_force)
  return torch.sum(stumble.float(), dim=1)


class feet_swing_clearance:
  """Penalize swing clearance error, measured relative to the step's own footholds.

  At touchdown, clearance = apex height - max(liftoff height, touchdown height),
  so stepping up or down stairs is judged by how far the foot cleared the higher
  of the two footholds rather than by its world-frame height.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    num_feet = len(cfg.params["asset_cfg"].site_names)
    shape = (env.num_envs, num_feet)
    self.peak_z = torch.zeros(shape, device=env.device)
    self.liftoff_z = torch.zeros(shape, device=env.device)
    self.prev_contact = torch.ones(shape, dtype=torch.bool, device=env.device)
    # A landing only counts if its liftoff was observed in this episode.
    self.has_liftoff = torch.zeros(shape, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.prev_contact[env_ids] = True
    self.has_liftoff[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    scan_sensor_name: str | None = None,
    even_only: bool = False,
  ) -> torch.Tensor:
    """With `scan_sensor_name`, only too-low swings are penalized on uneven
    ground (lifting higher over stairs is fine); with `even_only` too, swings
    on uneven ground are not judged at all (a wheel rolling off a step edge
    lands like a step with no clearance)."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_contact = contact_sensor.data.found > 0

    liftoff = self.prev_contact & ~in_contact
    self.liftoff_z = torch.where(liftoff, foot_z, self.liftoff_z)
    self.peak_z = torch.where(liftoff, foot_z, self.peak_z)
    self.has_liftoff |= liftoff
    self.peak_z = torch.where(~in_contact, torch.maximum(self.peak_z, foot_z), self.peak_z)

    landed = ~self.prev_contact & in_contact & self.has_liftoff
    clearance = self.peak_z - torch.maximum(self.liftoff_z, foot_z)
    error = clearance / target_height - 1.0
    if scan_sensor_name is not None:
      uneven = uneven_terrain_mask(env, scan_sensor_name).unsqueeze(1)
      on_uneven = torch.zeros_like(error) if even_only else torch.clamp(error, max=0.0)
      error = torch.where(uneven, on_uneven, error)
    active = ~_zero_command_mask(env, command_name, command_threshold)
    cost = torch.sum(torch.square(error) * landed.float(), dim=1) * active.float()

    num_landings = torch.sum(landed.float())
    env.extras["log"]["Metrics/swing_clearance_mean"] = torch.sum(
      clearance * landed.float()
    ) / torch.clamp(num_landings, min=1)

    self.has_liftoff &= ~landed
    self.prev_contact = in_contact
    return cost


class variable_posture_terrain_aware(variable_posture):
  """variable_posture that uses the walking tolerance when standing on uneven ground.

  Standing on stairs needs different joint angles per leg, so the tight
  standing tolerance only applies on even ground.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    sensor_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
    uneven_radius: float = 0.45,
    uneven_threshold: float = 0.05,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    uneven = uneven_terrain_mask(env, sensor_name, uneven_radius, uneven_threshold)

    standing_mask = ((total_speed < walking_threshold) & ~uneven).float()
    walking_mask = (
      ((total_speed >= walking_threshold) & (total_speed < running_threshold))
      | ((total_speed < walking_threshold) & uneven)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))



class feet_terrain_clearance:
  """Swing feet must clear step edges just ahead of them.

  For each swing foot, scan points within `lookahead` ahead along the foot's
  horizontal motion (and `half_width` sideways, nothing behind) are checked. If their height
  span exceeds `edge_threshold` (a step edge), the sole must be `margin` above
  the highest of them; the squared shortfall is the cost. Hits more than 0.5 m
  above the base (wall tops) are ignored.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg, env

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    scan_sensor_name: str,
    asset_cfg: SceneEntityCfg,
    lookahead: float = 0.12,
    half_width: float = 0.05,
    edge_threshold: float = 0.04,
    margin: float = 0.05,
    foot_radius: float = 0.022,
    min_speed: float = 0.05,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact: ContactSensor = env.scene[sensor_name]
    scan: RayCastSensor = env.scene[scan_sensor_name]
    assert contact.data.found is not None
    swing = contact.data.found == 0  # [B, F]
    foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids]  # [B, F, 3]
    foot_vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
    # Direction of travel; slow feet use the base heading.
    heading = asset.data.heading_w
    fwd = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1).unsqueeze(1)
    speed = torch.norm(foot_vel, dim=-1, keepdim=True)
    direction = torch.where(speed > min_speed, foot_vel / speed.clamp(min=1e-6), fwd)

    hits = scan.data.hit_pos_w  # [B, R, 3]
    valid = (scan.data.distances >= 0.0) & (
      hits[..., 2] <= scan.data.pos_w[:, None, 2] + 0.5
    )
    rel = hits[:, None, :, :2] - foot_pos[:, :, None, :2]  # [B, F, R, 2]
    along = (rel * direction[:, :, None, :]).sum(-1)
    side = torch.abs(rel[..., 0] * direction[:, :, None, 1] - rel[..., 1] * direction[:, :, None, 0])
    window = valid[:, None, :] & (along >= 0.0) & (along <= lookahead) & (side <= half_width)
    z = hits[:, None, :, 2].expand_as(along)
    z_max = torch.where(window, z, -torch.inf).amax(dim=-1)
    z_min = torch.where(window, z, torch.inf).amin(dim=-1)
    edge = window.any(dim=-1) & ((z_max - z_min) > edge_threshold)
    sole = foot_pos[..., 2] - foot_radius
    short = torch.clamp(z_max + margin - sole, min=0.0)
    cost = torch.where(edge & swing, torch.square(short), 0.0)
    return cost.sum(dim=1)


class feet_foothold_height_gain:
  """Reward new height records of each foot's stable footholds this episode.

  A foothold is stable when the contact force is mostly vertical. The first
  stable foothold after reset sets the baseline without reward; afterwards a
  foothold more than `min_gain` above the foot's record is rewarded by the
  gain (as a rate, gain / step_dt, so the weight is reward per meter). Hopping
  on flat ground or stepping down and back up earns nothing. Only active with
  a non-zero command.

  With `descend=True` the records are lows instead, rewarding progress down
  stairs the same way (nothing on flat ground or when spawned at the bottom).

  Being progress rewards, discounting pays more for collecting them sooner,
  which made the policy run down stairs well above the commanded speed. With
  `overspeed_range=(lo, hi)`, a record is paid in full while the base moves at
  most `lo` m/s faster than commanded, fading to nothing at `hi`.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    num_feet = len(cfg.params["asset_cfg"].site_names)
    shape = (env.num_envs, num_feet)
    self.best_z = torch.zeros(shape, device=env.device)
    self.has_best = torch.zeros(shape, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.has_best[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    vertical_ratio: float = 2.0,
    min_force: float = 10.0,
    min_gain: float = 0.02,
    descend: bool = False,
    overspeed_range: tuple[float, float] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact: ContactSensor = env.scene[sensor_name]
    force = contact.data.force
    assert contact.data.found is not None and force is not None
    force_z = force[..., 2].abs()
    stable = (
      (contact.data.found > 0)
      & (force_z > min_force)
      & (force_z > vertical_ratio * torch.norm(force[..., :2], dim=-1))
    )
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    if descend:
      foot_z = -foot_z

    first = stable & ~self.has_best
    self.best_z = torch.where(first, foot_z, self.best_z)
    self.has_best |= first
    gain = torch.where(stable & ~first, foot_z - self.best_z, 0.0)
    record = gain > min_gain
    self.best_z = torch.where(record, foot_z, self.best_z)
    active = ~_zero_command_mask(env, command_name, command_threshold)
    reward = torch.where(record, gain, 0.0).sum(dim=1) * active.float()
    if overspeed_range is not None:
      lo, hi = overspeed_range
      command = env.command_manager.get_command(command_name)
      assert command is not None
      overspeed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1) - torch.norm(
        command[:, :2], dim=1
      )
      reward = reward * torch.clamp((hi - overspeed) / (hi - lo), 0.0, 1.0)
    return reward / env.step_dt


def wall_proximity(
  env: ManagerBasedRlEnv, sensor_name: str, safe_distance: float = 0.35
) -> torch.Tensor:
  """Squared intrusion into `safe_distance` of side walls, summed over rays.

  Expects a level, horizontal ray fan to the sides of the base (see
  HorizontalFanPatternCfg), so the policy keeps itself centered in a corridor
  instead of relying on the operator to steer it back.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  d = sensor.data.distances
  intrusion = torch.where(d >= 0.0, torch.clamp(safe_distance - d, min=0.0), 0.0)
  return torch.square(intrusion).sum(dim=1)


def stance_leg_tilt(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  scan_sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Stance legs splayed sideways on uneven ground (go2-blind).

  Per foot on the ground: (asin of the world z of the thigh's lateral axis -
  default hip angle)^2, summed. The thigh frame is the base frame turned by
  the hip (+x axis), so on a level base the angle is the hip angle, and base
  roll counts as tilt too. Swing legs and even ground are free. `asset_cfg`
  holds the thigh bodies and the hip joints in model order, which is also
  the order of the foot contact sensor's columns.
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]  # [B, 4, 4]
  lateral = torch.zeros(*quat.shape[:-1], 3, device=quat.device)
  lateral[..., 1] = 1.0
  tilt = torch.asin(quat_apply(quat, lateral)[..., 2].clamp(-1.0, 1.0))
  hip = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  in_contact = (contact.data.found > 0).float()
  cost = torch.sum(torch.square(tilt - hip) * in_contact, dim=1)
  return cost * uneven_terrain_mask(env, scan_sensor_name).float()


class straight_drift:
  """Penalize drifting off the commanded line under a command without turning.

  While the command has no yaw rate (|wz| <= `wz_threshold`) and a planar speed
  of at least `min_speed`, the heading and position at the start of that command
  (a new command or a reset restarts them) define the line the robot should walk
  along. The cost is |heading change| / `heading_scale` + |sideways offset from
  the line| / `lateral_scale`, each clamped at 1; `lateral_scale=None` leaves
  the sideways term out (it grows with the distance walked, so walking slower
  lowers it: As2 2026-10-20 capped its speed at ~0.9 m/s). The exp tracking kernels are
  flat near zero: a steady 0.02 rad/s yaw error, 11 deg in 10 s, costs under
  0.2% of the yaw tracking reward, and the blind policies (2026-10-20) drifted
  ~11 deg per 10 s walking straight and into the walls of map3 flights without
  an operator. A blind policy sees no heading, so it lowers this cost by keeping
  its yaw rate and sideways velocity at zero on average.

  `max_speed` and `forward_only` limit it to commands up to that planar speed and
  without a sideways part: penalized at 1.5 m/s and sideways, Go2 and As2
  (2026-10-20) gave up 10 points of top speed and of diagonal sideways speed,
  where walking slower drifted less.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    n = env.num_envs
    self.ref_heading = torch.zeros(n, device=env.device)
    self.ref_pos = torch.zeros(n, 2, device=env.device)
    self.ref_dir = torch.zeros(n, 2, device=env.device)
    self.prev_command = torch.zeros(n, 3, device=env.device)
    self.active = torch.zeros(n, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    # Restarted at the next call: the root pose may still be the previous episode's.
    self.active[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    wz_threshold: float = 1e-6,
    min_speed: float = 0.1,
    heading_scale: float = 0.17,
    lateral_scale: float | None = 0.2,
    max_speed: float | None = None,
    forward_only: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    heading = asset.data.heading_w
    pos = asset.data.root_link_pos_w[:, :2]
    speed = torch.norm(command[:, :2], dim=1)
    straight = (command[:, 2].abs() <= wz_threshold) & (speed >= min_speed)
    if max_speed is not None:
      straight &= speed <= max_speed
    if forward_only:
      straight &= command[:, 1].abs() <= 1e-6
    changed = (command - self.prev_command).abs().amax(dim=1) > 1e-6
    start = straight & (~self.active | changed)
    # Commanded direction in the world frame at the start.
    v = command[:, :2] / speed.clamp(min=1e-6).unsqueeze(1)
    c, s = torch.cos(heading), torch.sin(heading)
    direction = torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], dim=1)
    self.ref_heading = torch.where(start, heading, self.ref_heading)
    self.ref_pos = torch.where(start.unsqueeze(1), pos, self.ref_pos)
    self.ref_dir = torch.where(start.unsqueeze(1), direction, self.ref_dir)
    self.active = straight
    self.prev_command = command.clone()
    turn = torch.abs(wrap_to_pi(heading - self.ref_heading))
    cost = (turn / heading_scale).clamp(max=1.0)
    if lateral_scale is not None:
      rel = pos - self.ref_pos
      side = torch.abs(self.ref_dir[:, 0] * rel[:, 1] - self.ref_dir[:, 1] * rel[:, 0])
      cost = cost + (side / lateral_scale).clamp(max=1.0)
    return torch.where(straight & ~start, cost, torch.zeros_like(cost))
