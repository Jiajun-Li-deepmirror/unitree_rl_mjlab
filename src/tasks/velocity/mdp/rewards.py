from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi
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
  uneven_terrain_mask) uses the tighter `even_std`. With `min_std` instead,
  the std shrinks with the commanded planar speed to clamp(|v_cmd,xy|,
  min_std, std): standing still then keeps exp(-1) of the reward at any
  command of at least `min_std`, not most of it at slow ones (Irie et al. 2025).
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

  The commanded xy angular velocities are assumed to be zero. With `min_std`,
  the std of the yaw rate error shrinks with the commanded yaw rate to
  clamp(|wz_cmd|, min_std, std), as in track_linear_velocity; the xy term
  keeps `std`.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  if min_std is not None:
    z_std = torch.clamp(torch.abs(command[:, 2]), min=min_std, max=std)
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


def wheel_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  wheel_radius: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize stance wheels skidding: squared xy velocity of the contact point.

  The contact point is taken `wheel_radius` below the wheel center (the foot
  site), so its velocity is v_center + w x (0, 0, -r) with w the wheel body's
  angular velocity. Rolling gives zero; spinning in place (e.g. against a stair
  riser) or skidding does not. asset_cfg holds the foot sites and the wheel
  bodies, which resolve in model order like the contact sensor's feet.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, W]
  center_vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, W, 2]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]  # [B, W, 3]
  # w x (0, 0, -r) = r * (-w_y, w_x, 0).
  spin_vel = wheel_radius * torch.stack((-ang_vel[..., 1], ang_vel[..., 0]), dim=-1)
  slip = torch.norm(center_vel + spin_vel, dim=-1)  # [B, W]
  env.extras["log"]["Metrics/wheel_slip_mean"] = torch.sum(slip * in_contact) / torch.clamp(
    torch.sum(in_contact), min=1
  )
  return torch.sum(torch.square(slip) * in_contact, dim=1)


def wheel_air_spin(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize lifted wheels spinning: sum of |wheel joint velocity| off the ground.

  A lifted wheel has nothing to drive and lands with a jolt if it spins.
  asset_cfg holds the wheel joints, which resolve in model order like the
  contact sensor's feet.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_air = (contact_sensor.data.found <= 0).float()  # [B, W]
  wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # [B, W]
  return torch.sum(torch.abs(wheel_vel) * in_air, dim=1)


def wheel_air_drivable(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.1,
  max_lin_vel_y: float = 0.1,
  max_ang_vel_z: float = 0.5,
  scan_sensor_name: str | None = None,
) -> torch.Tensor:
  """Penalize lifted wheels while the command can be driven on the wheels.

  Fraction of wheels off the ground while the command is not zero and has at most
  `max_lin_vel_y` sideways and `max_ang_vel_z` yaw, i.e. driving forward or back,
  which the wheels can follow by rolling. Sideways moves and turns beyond those
  limits are free to step. With `scan_sensor_name`, only on even ground (see
  uneven_terrain_mask), since wheels must be lifted over stair edges.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_air = (contact_sensor.data.found <= 0).float()  # [B, W]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  drivable = (
    ~_zero_command_mask(env, command_name, command_threshold)
    & (torch.abs(command[:, 1]) <= max_lin_vel_y)
    & (torch.abs(command[:, 2]) <= max_ang_vel_z)
  )
  if scan_sensor_name is not None:
    drivable &= ~uneven_terrain_mask(env, scan_sensor_name)
  return torch.mean(in_air, dim=1) * drivable.float()


class wheel_riser_lift:
  """Reward lifting a wheel over a riser that blocks it.

  A wheel is blocked when its contact force is mostly horizontal, |F_xy| >
  `ratio` |F_z| (at 1.2, friction on flat ground with mu <= 1 cannot do it). The
  riser counts when its top, the highest privileged terrain-scan hit within
  `radius` of the wheel center, is above the wheel's highest stable foothold
  this episode (as in feet_foothold_height_gain), so risers already climbed or
  below the start of a descent never pay. For `window` seconds after a block,
  or until the wheel lands after leaving the ground, the term pays the fraction
  of the riser cleared while the wheel is off the ground: the rise of the sole
  since the wheel was first blocked there over the rise that clears the riser
  top by `margin`, clipped to [0, 1].

  Only the rise of the best fraction so far is paid, per second, so with dt
  scaling the episode sum is at most weight per riser and wheel; lifting and
  dropping a wheel at the same riser earns nothing more.

  Nothing else pays for partly lifting a blocked wheel, and without a gait clock
  the blind policy of the first Go2W stairs pilot (2026-10-18) stalled at the
  first riser. The contact trigger follows CTBC (arXiv 2509.02986). Requires a
  world-frame contact force.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    shape = (env.num_envs, len(cfg.params["asset_cfg"].site_names))
    self.record_z = torch.full(shape, -torch.inf, device=env.device)
    self.riser_top = torch.full(shape, torch.inf, device=env.device)
    self.start_z = torch.zeros(shape, device=env.device)
    self.needed = torch.ones(shape, device=env.device)
    self.best = torch.zeros(shape, device=env.device)
    self.time_left = torch.zeros(shape, device=env.device)
    self.airborne = torch.zeros(shape, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.record_z[env_ids] = -torch.inf
    self.riser_top[env_ids] = torch.inf
    self.best[env_ids] = 0.0
    self.time_left[env_ids] = 0.0
    self.airborne[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    scan_sensor_name: str,
    wheel_radius: float,
    radius: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    ratio: float = 1.2,
    min_force: float = 5.0,
    stance_ratio: float = 2.0,
    stance_min_force: float = 10.0,
    window: float = 1.0,
    margin: float = 0.03,
    min_rise: float = 0.03,
    tolerance: float = 0.02,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    scan: RayCastSensor = env.scene[scan_sensor_name]
    force = contact_sensor.data.force
    assert contact_sensor.data.found is not None and force is not None
    in_contact = contact_sensor.data.found > 0  # [B, W]
    force_xy = torch.norm(force[..., :2], dim=-1)
    force_z = torch.abs(force[..., 2])

    foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids]  # [B, W, 3]
    sole_z = foot_pos[..., 2] - wheel_radius
    stance = in_contact & (force_z > torch.clamp(stance_ratio * force_xy, min=stance_min_force))
    self.record_z = torch.where(stance, torch.maximum(self.record_z, sole_z), self.record_z)

    hit_pos = scan.data.hit_pos_w  # [B, N, 3]
    offset = hit_pos[:, None, :, :2] - foot_pos[:, :, None, :2]  # [B, W, N, 2]
    near = (torch.norm(offset, dim=-1) < radius) & (scan.data.distances >= 0.0)[:, None, :]
    top_z = torch.where(near, hit_pos[:, None, :, 2], -torch.inf).amax(dim=-1)  # [B, W]

    blocked = in_contact & (force_xy > torch.clamp(ratio * force_z, min=min_force))
    riser = blocked & (top_z > self.record_z + tolerance)
    new = riser & ~(torch.abs(top_z - self.riser_top) <= tolerance)
    self.riser_top = torch.where(new, top_z, self.riser_top)
    self.start_z = torch.where(new, sole_z, self.start_z)
    self.needed = torch.where(new, torch.clamp(top_z + margin - sole_z, min=min_rise), self.needed)
    self.best = torch.where(new, 0.0, self.best)
    # The window closes early when the wheel lands after leaving the ground.
    landed = in_contact & self.airborne & ~riser
    self.time_left = torch.where(
      riser, window, torch.clamp(self.time_left - env.step_dt, min=0.0)
    )
    self.time_left = torch.where(landed, 0.0, self.time_left)
    self.airborne = ~riser & (self.airborne | ~in_contact) & (self.time_left > 0.0)

    active = ~_zero_command_mask(env, command_name, command_threshold)
    lifting = (self.time_left > 0.0) & ~in_contact & active[:, None]
    fraction = torch.clamp((sole_z - self.start_z) / self.needed, min=0.0, max=1.0)
    gain = torch.where(lifting, torch.clamp(fraction - self.best, min=0.0), 0.0)
    self.best = self.best + gain
    env.extras["log"]["Metrics/wheel_riser_blocked"] = torch.mean(riser.float())
    return torch.sum(gain, dim=1) / env.step_dt


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
) -> torch.Tensor:
  """Per-env bool: terrain height span within `radius` of the base exceeds `threshold`.

  Uses the (privileged) height-scan raycast, so it reflects the terrain the robot
  is currently on rather than the column it was spawned in.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  data = sensor.data
  hit_z = data.hit_pos_w[..., 2]
  planar_dist = torch.norm(data.hit_pos_w[..., :2] - data.pos_w[:, None, :2], dim=-1)
  near = (planar_dist <= radius) & (data.distances >= 0.0)
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
  (0 to 1). Going faster than that earns nothing, so it cannot pay to rush.

  The wheeled stairs policies (2026-10-20) went down map3's flights at 0.5-0.7x
  the command: at tracking std 0.35 half of 0.4 m/s costs 28% of the tracking
  reward. An ungated descent progress reward made the go2-depth wheeled robots
  rush down at 1.35-5x instead.
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


class standing_position_hold:
  """Penalize creeping away from where the robot came to rest while commanded to stand.

  The anchor follows the base while the command is not zero or the base moves
  (faster than `moving_speed` or turning faster than `moving_yaw_rate`: driving,
  braking, being pushed), so only the slow creep after coming to rest counts:
  distance from the anchor (clamped to `max_dist`) plus `yaw_scale` times the
  heading change. Velocity-controlled wheels hold no position at zero speed, so
  wheeled robots creep a few cm per 4 s; the squared standing terms barely see it.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.anchor_pos = torch.zeros(env.num_envs, 2, device=env.device)
    self.anchor_yaw = torch.zeros(env.num_envs, device=env.device)
    self.needs_anchor = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    # Set at the next call: the root pose may still be the previous episode's.
    self.needs_anchor[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    moving_speed: float = 0.1,
    moving_yaw_rate: float = 0.2,
    max_dist: float = 0.3,
    yaw_scale: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.root_link_pos_w[:, :2]
    yaw = asset.data.heading_w
    speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
    standing = _zero_command_mask(env, command_name, command_threshold)
    follow = (
      ~standing | (speed > moving_speed) | (yaw_rate > moving_yaw_rate) | self.needs_anchor
    )
    self.anchor_pos = torch.where(follow.unsqueeze(1), pos, self.anchor_pos)
    self.anchor_yaw = torch.where(follow, yaw, self.anchor_yaw)
    self.needs_anchor[:] = False
    dist = torch.norm(pos - self.anchor_pos, dim=1).clamp(max=max_dist)
    turn = torch.abs(wrap_to_pi(yaw - self.anchor_yaw))
    return standing.float() * (dist + yaw_scale * turn)


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


def standing_speed(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  yaw_scale: float = 0.3,
  max_speed: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Base speed (not squared) while commanded to stand: planar speed plus
  `yaw_scale` times the yaw rate, each clamped to `max_speed`. Unlike the squared
  tracking error it still pulls at a slow creep, and it pays for braking promptly."""
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1).clamp(max=max_speed)
  yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2]).clamp(max=max_speed)
  standing = _zero_command_mask(env, command_name, command_threshold)
  return standing.float() * (speed + yaw_scale * yaw_rate)


def standing_wheel_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str = "wheel_vel",
  command_threshold: float = 0.1,
  calm_std: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Summed |wheel velocity target| (rad/s) while commanded to stand and calm.

  At the zero command the deployed Go2W (2026-10-19) drove its right wheels at
  +0.66 and -0.56 rad/s against each other: the grounded wheels barely turned
  (0.05 rad/s) but kept pushing, and the base crept 2-3 cm per 4 s, too little
  for the position terms to see. The policy observes its last action, so unlike
  the creep itself this is directly observable.
  """
  term = env.action_manager.get_term(action_name)
  target = term.raw_action * term.scale
  cost = torch.sum(torch.abs(target), dim=1)
  return cost * _standing_scale(env, command_name, command_threshold, calm_std, asset_cfg)


def standing_feet_air(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.1,
  calm_std: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Fraction of feet off the ground while commanded to stand and the base is calm
  (fades out while the base moves, so steps that recover from a push are free)."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_air = torch.mean((contact_sensor.data.found <= 0).float(), dim=1)
  return in_air * _standing_scale(env, command_name, command_threshold, calm_std, asset_cfg)


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


def stance_leg_tilt(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  scan_sensor_name: str,
  uneven_radius: float = 0.45,
  uneven_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize stance legs tilting sideways on uneven ground, relative to the default stance.

  A leg's sideways tilt is that of its plane (thigh and calf), taken from the
  thigh body's y axis (the thigh joint axis) in the world frame:
  asin(y_axis_z). On a level body it equals the hip joint angle (hip axes +x),
  so the default stance scores 0 and a body roll counts like hip abduction.
  Swinging legs are free, so a step may open the hip; summed squared error
  (rad^2) over the feet on the ground, on uneven ground only (see
  uneven_terrain_mask). asset_cfg holds the hip joints and the thigh bodies,
  which resolve in model order like the contact sensor's feet.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, F]
  q = asset.data.body_link_quat_w[:, asset_cfg.body_ids]  # [B, F, 4] (w, x, y, z)
  y_axis_z = 2.0 * (q[..., 2] * q[..., 3] + q[..., 0] * q[..., 1])
  tilt = torch.asin(torch.clamp(y_axis_z, -1.0, 1.0))
  default_tilt = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  cost = torch.sum(in_contact * torch.square(tilt - default_tilt), dim=1)
  uneven = uneven_terrain_mask(env, scan_sensor_name, uneven_radius, uneven_threshold)
  return cost * uneven.float()


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
    """If `scan_sensor_name` is given, overshoot is not penalized on uneven terrain,
    and with `even_only` nothing is.

    A blind robot finds a riser by bumping into it and must then try higher
    swings; a symmetric error would penalize those attempts whenever the foot
    falls back to its liftoff height. A wheel rolling off a down-stair edge lands
    after a zero-clearance swing.
    """
    assert scan_sensor_name is not None or not even_only, "even_only needs a scan."
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
      uneven = uneven_terrain_mask(env, scan_sensor_name)
      uneven_error = torch.zeros_like(error) if even_only else torch.clamp(error, max=0.0)
      error = torch.where(uneven[:, None], uneven_error, error)
    active = ~_zero_command_mask(env, command_name, command_threshold)
    cost = torch.sum(torch.square(error) * landed.float(), dim=1) * active.float()

    num_landings = torch.sum(landed.float())
    env.extras["log"]["Metrics/swing_clearance_mean"] = torch.sum(
      clearance * landed.float()
    ) / torch.clamp(num_landings, min=1)

    self.has_liftoff &= ~landed
    self.prev_contact = in_contact
    return cost


def feet_terrain_clearance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  scan_sensor_name: str,
  target_height: float = 0.05,
  radius: float = 0.12,
  edge_threshold: float = 0.04,
  foot_radius: float = 0.022,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize swing feet passing below a terrain edge they are moving towards.

  For each swing foot, take the privileged terrain-scan hits within `radius`
  (horizontally) that are not behind the foot's direction of travel. If they
  contain an edge (height span above `edge_threshold`), the sole must be
  `target_height` above the highest of them; the squared shortfall (m^2) is
  summed over feet. Flat ground has no edges, so the flat gait is unaffected,
  and a foot stepping down off an edge is free to drop once past it.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  scan: RayCastSensor = env.scene[scan_sensor_name]
  assert contact_sensor.data.found is not None

  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids]  # [B, F, 3]
  foot_vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  hit_pos = scan.data.hit_pos_w  # [B, N, 3]
  valid = (scan.data.distances >= 0.0)[:, None, :]  # [B, 1, N]

  offset = hit_pos[:, None, :, :2] - foot_pos[:, :, None, :2]  # [B, F, N, 2]
  near = (torch.norm(offset, dim=-1) < radius) & valid
  heading = foot_vel / torch.clamp(torch.norm(foot_vel, dim=-1, keepdim=True), min=1e-3)
  ahead = near & (torch.sum(offset * heading[:, :, None, :], dim=-1) > -0.02)

  hit_z = hit_pos[:, None, :, 2]
  z_max = torch.where(ahead, hit_z, -torch.inf).amax(dim=-1)  # [B, F]
  z_min = torch.where(near, hit_z, torch.inf).amin(dim=-1)
  edge = ahead.any(dim=-1) & (z_max - z_min > edge_threshold)

  swing = contact_sensor.data.found <= 0
  sole_z = foot_pos[..., 2] - foot_radius
  shortfall = torch.clamp(z_max + target_height - sole_z, min=0.0)
  return torch.sum(torch.where(edge & swing, torch.square(shortfall), 0.0), dim=1)


class feet_foothold_height_gain:
  """Reward each foot for standing higher than it has stood before this episode.

  Every foot keeps a record of its highest stable foothold; the term is the rise
  of that record per second, so with dt scaling the episode sum is weight times
  metres climbed per foot. Records never go down: bouncing on flat ground or
  stepping down and back up earns nothing, while getting a front foot onto the
  first step of a riser pays out at once, before the body makes it over.

  With `descend=True` the records are lows instead, paying for progress down
  stairs the same way.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    num_feet = len(cfg.params["asset_cfg"].site_names)
    self.record_z = torch.full((env.num_envs, num_feet), -torch.inf, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.record_z[env_ids] = -torch.inf

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    ratio: float = 2.0,
    min_force: float = 10.0,
    max_gain: float = 0.2,
    descend: bool = False,
  ) -> torch.Tensor:
    """A foothold counts when the contact force is mostly vertical.

    That excludes a foot pushed against a riser face. The first foothold after
    reset only initializes the record. Requires a world-frame force.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    force = contact_sensor.data.force
    assert contact_sensor.data.found is not None and force is not None
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    if descend:
      foot_z = -foot_z
    force_xy = torch.norm(force[..., :2], dim=-1)
    stance = (contact_sensor.data.found > 0) & (
      torch.abs(force[..., 2]) > torch.clamp(ratio * force_xy, min=min_force)
    )
    gain = torch.where(
      stance & torch.isfinite(self.record_z),
      torch.clamp(foot_z - self.record_z, min=0.0, max=max_gain),
      0.0,
    )
    self.record_z = torch.where(stance, torch.maximum(self.record_z, foot_z), self.record_z)
    active = ~_zero_command_mask(env, command_name, command_threshold)
    return torch.sum(gain, dim=1) * active.float() / env.step_dt


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
