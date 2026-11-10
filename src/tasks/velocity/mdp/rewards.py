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

from .observations import gait_clock

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
  uneven_terrain_mask) uses the tighter `even_std`. With `min_std`, the std
  shrinks with the commanded planar speed to clamp(|v_cmd,xy|, min_std, std)
  (std: the even or uneven one): standing still then keeps exp(-1) of the
  reward at any command of at least `min_std`, not most of it at slow ones
  (Irie et al. 2025).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (z_weight * z_error)
  std_t = torch.full_like(lin_vel_error, std)
  if even_std is not None and scan_sensor_name is not None:
    std_t = torch.where(uneven_terrain_mask(env, scan_sensor_name), std_t, even_std)
  if min_std is not None:
    speed = torch.norm(command[:, :2], dim=1)
    std_t = torch.minimum(speed.clamp(min=min_std), std_t)
  return torch.exp(-lin_vel_error / std_t**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_std: float | None = None,
  yaw_accel: float | None = None,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero. With `min_std`,
  the std of the yaw rate error shrinks with the commanded yaw rate to
  clamp(|wz_cmd|, min_std, std), as in track_linear_velocity; the xy term keeps
  `std`. With `yaw_accel` (rad/s^2) the yaw rate is tracked against
  _yaw_reference (the command's rate rising at most that fast; the policy still
  observes the command itself).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  target = command[:, 2] if yaw_accel is None else _yaw_reference(env, command_name, yaw_accel)
  z_error = torch.square(target - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  if min_std is not None:
    rate = torch.abs(command[:, 2])
    z_std = torch.clamp(rate, min=min_std, max=std)
    return torch.exp(-z_error / z_std**2 - 0.05 * xy_error / std**2)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def _yaw_reference(env: ManagerBasedRlEnv, command_name: str, yaw_accel: float) -> torch.Tensor:
  """The commanded yaw rate with its magnitude rising at most `yaw_accel` rad/s^2 (2026-11-01, B2 spin-up falls).

  It starts at 0 with each episode, drops at once when the command slows, stops or reverses, and is updated once
  per env step (the reward computes before resets, so a robot reset on the last step has episode length 1 here).
  B2 spun up to 80% of 1.5 rad/s in 0.12 s from a standstill, with one foot down and 16-26 deg of roll at
  0.1-0.3 s, and fell 7 of 4096 at low foot friction; the same policy under a command ramped over 0.5 s fell 0.
  """
  wz = env.command_manager.get_command(command_name)[:, 2]
  refs = env.__dict__.setdefault("_yaw_refs", {})
  step, ref = refs.get(command_name, (None, None))
  if step == env.common_step_counter:
    return ref
  if ref is None:
    ref = torch.zeros_like(wz)
  ref = torch.where((env.episode_length_buf > 1) & (wz * ref > 0), ref, 0.0)
  ref = torch.sign(wz) * torch.minimum(wz.abs(), ref.abs() + yaw_accel * env.step_dt)
  refs[command_name] = (env.common_step_counter, ref)
  return ref


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
        march_only: bool = False,
        stepping_only: bool = False,
        max_lin_vel_y: float = 0.1,
        max_ang_vel_z: float = 0.1,
        scan_sensor_name: str | None = None,
        ref_speed: float | None = None,
        foot_xy: tuple[tuple[float, float], ...] | None = None,
        march_scale: float = 1.0,
) -> torch.Tensor:
    """Reward feet whose contact state matches the gait clock (mdp.gait_clock: with
    `ref_speed` its period shortens at speed, with `foot_xy` at the fastest foot's speed,
    shared with the phase observation). With `march_only`, only
    while marching in place (the command term's march flag): wheeled robots drive with
    all wheels down and step only when told to march (seventh round). With
    `stepping_only`, while marching or under a command the wheels cannot drive
    (sideways beyond `max_lin_vel_y` or turning beyond `max_ang_vel_z`, as in
    wheel_air_drivable), and with `scan_sensor_name` only on even ground: wheeled
    robots then step to the clock wherever they step on flat ground (user
    2026-10-22), and lift freely over stair edges. While marching in place the
    reward is `march_scale` times as large: standing still matches the clock's
    stance share (56% of the wheeled trot) for free."""
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = gait_clock(env, period, command_name, ref_speed, foot_xy).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            total_command = _command_activity(env, command_name)
            scale = (total_command > command_threshold).float()
            reward *= scale
    if march_only or stepping_only:
        march = getattr(env.command_manager.get_term(command_name), "march", None)
        assert march is not None, "march_only / stepping_only: the command term has no march flag"
        gate = march
        if stepping_only:
            command = env.command_manager.get_command(command_name)
            stepping = (torch.abs(command[:, 1]) > max_lin_vel_y) | (torch.abs(command[:, 2]) > max_ang_vel_z)
            gate = gate | stepping
            if scan_sensor_name is not None:
                gate = gate & ~uneven_terrain_mask(env, scan_sensor_name)
        reward = reward * gate.float()
    if march_scale != 1.0:
        march = getattr(env.command_manager.get_term(command_name), "march", None)
        assert march is not None, "march_scale: the command term has no march flag"
        reward = torch.where(march, march_scale * reward, reward)
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
  total_command = _command_activity(env, command_name)
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
  free_splay: float | None = None,
  hip_cfg: SceneEntityCfg | None = None,
  wheel_cfg: SceneEntityCfg | None = None,
  wheel_uneven_radius: float = 0.25,
  near_scale: float = 1.0,
  wheel_ahead_only: bool = False,
) -> torch.Tensor:
  """Penalize lifted wheels while the command can be driven on the wheels.

  With `wheel_cfg` (the wheel sites, in the contact sensor's order) and a scan,
  even ground is judged per wheel instead of per robot (wheel_uneven_mask within
  `wheel_uneven_radius` of each wheel): the user (2026-11-03) saw the rear wheels
  of all three wheeled robots step on the start landing of map3's flights while
  the fronts climbed, 0.3-0.6 m short of the first riser (2-3 lifts per climb,
  2-18 cm; scratchpad check/stair_wheels.py), free because the robot's 0.45 m
  scan circle already held a riser. `near_scale` multiplies the cost of those lifts
  (this wheel on even ground, the robot's circle uneven): at 1x the Go2W teacher's
  share of such lifted wheels went 15.6 -> 15.0% in 1000 iterations (2026-11-03).

  Fraction of wheels off the ground while the command is not zero and has at most
  `max_lin_vel_y` sideways and `max_ang_vel_z` yaw, i.e. driving forward or back,
  which the wheels can follow by rolling. Sideways moves and turns beyond those
  limits are free to step. With `scan_sensor_name`, only on even ground (see
  uneven_terrain_mask), since wheels must be lifted over stair edges.

  With `free_splay`, lifting a wheel is free while its leg's hip abducts more than
  that (rad) outward of the default stance (hip_cfg: the hip joints, which resolve
  in model order like the contact sensor's wheels): rolling on, the legs splay
  out under the body weight and load the hip motors (Go2-W, arXiv 2606.21387), and
  lifting the wheel puts it back.
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
  march = getattr(env.command_manager.get_term(command_name), "march", None)
  if march is not None:
    drivable &= ~march  # marching in place steps on purpose
  if scan_sensor_name is not None and wheel_cfg is not None:
    wheel_uneven = wheel_uneven_mask(
      env, scan_sensor_name, wheel_cfg, wheel_uneven_radius, ahead_command=command_name if wheel_ahead_only else None
    )
    in_air = in_air * (~wheel_uneven).float()
    if near_scale != 1.0:
      near = uneven_terrain_mask(env, scan_sensor_name)[:, None] & ~wheel_uneven
      in_air = in_air * torch.where(near, near_scale, 1.0)
    env.extras["log"]["Metrics/wheel_lift_even"] = torch.sum(in_air * drivable[:, None].float()) / torch.clamp(
      torch.sum((~wheel_uneven & drivable[:, None]).float()), min=1
    )
  elif scan_sensor_name is not None:
    drivable &= ~uneven_terrain_mask(env, scan_sensor_name)
  if free_splay is not None:
    assert hip_cfg is not None
    in_air = in_air * (_hip_outward(env, hip_cfg) <= free_splay).float()
  return torch.mean(in_air, dim=1) * drivable.float()


def wheel_uneven_mask(
  env: ManagerBasedRlEnv,
  scan_sensor_name: str,
  wheel_cfg: SceneEntityCfg,
  radius: float,
  threshold: float = 0.05,
  ahead_command: str | None = None,
  behind_margin: float = 0.02,
) -> torch.Tensor:
  """Per wheel (envs, wheels): the privileged scan's hit heights within `radius` of
  the wheel site (planar) span more than `threshold` (uneven_terrain_mask around
  each wheel instead of the base). No hit within the radius counts as even.

  With `ahead_command` (the twist command's name), only the hits ahead of the wheel
  in the commanded travel direction count (along the base heading, forward or back by
  the command's sign; further back than `behind_margin` is ignored): a step just
  climbed behind a wheel does not make its ground uneven (2026-11-04, the user: the
  front wheels kept stepping on the top landing, 13-20 cm, every ~0.6 s from 0 to 70
  cm past the last riser, in every wheeled student old and new)."""
  asset: Entity = env.scene[wheel_cfg.name]
  scan: RayCastSensor = env.scene[scan_sensor_name]
  wheel_xy = asset.data.site_pos_w[:, wheel_cfg.site_ids, :2]  # [B, W, 2]
  hit = scan.data.hit_pos_w  # [B, N, 3]
  offset = hit[:, None, :, :2] - wheel_xy[:, :, None, :]  # [B, W, N, 2]
  near = (torch.norm(offset, dim=-1) <= radius) & (scan.data.distances >= 0.0)[:, None, :]
  if ahead_command is not None:
    command = env.command_manager.get_command(ahead_command)
    assert command is not None
    q = asset.data.root_link_quat_w  # [B, 4] (w, x, y, z)
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    sign = torch.sign(command[:, 0])
    heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1) * sign[:, None]  # [B, 2]
    along = (offset * heading[:, None, None, :]).sum(-1)
    near = near & ((along >= -behind_margin) | (sign == 0)[:, None, None])
  z = hit[:, None, :, 2]
  z_max = torch.where(near, z, -torch.inf).amax(dim=-1)
  z_min = torch.where(near, z, torch.inf).amin(dim=-1)
  return torch.where(near.any(dim=-1), z_max - z_min, 0.0) > threshold


def wheel_lift_flat_ahead(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  scan_sensor_name: str,
  wheel_cfg: SceneEntityCfg,
  radius: float = 0.35,
  command_threshold: float = 0.1,
  max_lin_vel_y: float = 0.1,
  max_ang_vel_z: float = 0.5,
) -> torch.Tensor:
  """Wheels off the ground (count per robot) while the robot's scan circle holds stairs (uneven_terrain_mask) but the
  wheel's own ground ahead is flat: no height change over 5 cm within `radius` ahead of it in the commanded direction
  (wheel_uneven_mask with ahead_command), under a command the wheels drive (wheel_air_drivable's limits, not marching).
  These are the landings at either end of a flight while the other wheels still climb or descend: the front wheels
  on the top landing (the user 2026-11-04: they kept stepping there), the rear wheels on the start landing."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_air = contact_sensor.data.found <= 0  # [B, W]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  drivable = (
    ~_zero_command_mask(env, command_name, command_threshold)
    & (torch.abs(command[:, 1]) <= max_lin_vel_y)
    & (torch.abs(command[:, 2]) <= max_ang_vel_z)
  )
  march = getattr(env.command_manager.get_term(command_name), "march", None)
  if march is not None:
    drivable &= ~march
  flat_ahead = ~wheel_uneven_mask(env, scan_sensor_name, wheel_cfg, radius, ahead_command=command_name)
  zone = flat_ahead & (drivable & uneven_terrain_mask(env, scan_sensor_name))[:, None]
  env.extras["log"]["Metrics/wheel_lift_flat_ahead"] = torch.sum(in_air & zone) / torch.clamp(torch.sum(zone), min=1)
  return torch.sum((in_air & zone).float(), dim=1)


def wheel_takeoff_flat_ahead(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  scan_sensor_name: str,
  wheel_cfg: SceneEntityCfg,
  radius: float = 0.35,
  command_threshold: float = 0.1,
  max_lin_vel_y: float = 0.1,
  max_ang_vel_z: float = 0.5,
) -> torch.Tensor:
  """Take-offs (count per robot, per second: events / step_dt, so the episode sum is weight x events) of wheels that
  leave the ground in wheel_lift_flat_ahead's zone: flat within `radius` ahead, the robot's scan circle on stairs, a
  drivable command. Judged at the take-off, where the wheel stood (2026-11-04): wheel_lift_flat_ahead counts every
  airborne step in the zone, so the last swing of a climb (past the top nosing) and of a descent (over the bottom
  landing) paid too; at -30 the As2W teacher descended map3 at 0.65 of the command (bar 0.80, 0.86 before) and the
  Go2W one landed shorter on the risers. A wheel taking off from the last tread has the riser or the drop within
  `radius` ahead, so only the steps on a landing pay."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  took_off = contact_sensor.compute_first_air(dt=env.step_dt)  # [B, W]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  drivable = (
    ~_zero_command_mask(env, command_name, command_threshold)
    & (torch.abs(command[:, 1]) <= max_lin_vel_y)
    & (torch.abs(command[:, 2]) <= max_ang_vel_z)
  )
  march = getattr(env.command_manager.get_term(command_name), "march", None)
  if march is not None:
    drivable &= ~march
  flat_ahead = ~wheel_uneven_mask(env, scan_sensor_name, wheel_cfg, radius, ahead_command=command_name)
  zone = flat_ahead & (drivable & uneven_terrain_mask(env, scan_sensor_name))[:, None]
  events = took_off & zone
  env.extras["log"]["Metrics/wheel_takeoff_flat_ahead"] = (
    torch.sum(events) / torch.clamp(torch.sum(zone), min=1) / env.step_dt
  )
  return torch.sum(events.float(), dim=1) / env.step_dt


class wheel_flat_step_air:
  """Airborne wheels (count per robot, so the episode sum is weight x wheel-seconds) that took off inside
  wheel_lift_flat_ahead's zone: from flat ground with flat ahead next to stairs, under a drivable command. The zone is
  judged at the take-off and holds until the wheel lands (2026-11-04 ~23:55). wheel_lift_flat_ahead judged every
  airborne step, so it also paid the last swing of every climb and descent (over the landing); wheel_takeoff_flat_ahead
  paid per take-off and at -15 wrecked both teachers within ~60 iterations: contact flickers made 3-5 take-offs per
  zone wheel-second, the term ran ~-6 per second against ~-0.7 for the next largest, the value loss went to ~22 and the
  map3 checks fell to 0.3%. Here a flicker costs its few milliseconds."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    shape = (env.num_envs, len(cfg.params["wheel_cfg"].site_names))
    self.prev_contact = torch.ones(shape, dtype=torch.bool, device=env.device)
    self.off_zone = torch.zeros(shape, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.prev_contact[env_ids] = True
    self.off_zone[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    scan_sensor_name: str,
    wheel_cfg: SceneEntityCfg,
    radius: float = 0.35,
    command_threshold: float = 0.1,
    max_lin_vel_y: float = 0.1,
    max_ang_vel_z: float = 0.5,
    higher: float | None = None,
    wheel_ground_radius: float = 0.1,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    in_air = contact_sensor.data.found <= 0  # [B, W]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    drivable = (
      ~_zero_command_mask(env, command_name, command_threshold)
      & (torch.abs(command[:, 1]) <= max_lin_vel_y)
      & (torch.abs(command[:, 2]) <= max_ang_vel_z)
    )
    march = getattr(env.command_manager.get_term(command_name), "march", None)
    if march is not None:
      drivable &= ~march
    flat_ahead = ~wheel_uneven_mask(env, scan_sensor_name, wheel_cfg, radius, ahead_command=command_name)
    zone = flat_ahead & (drivable & uneven_terrain_mask(env, scan_sensor_name))[:, None]
    if higher is not None:
      # Only wheels standing more than `higher` above the ground under the base (front wheels on a top landing with
      # the base still over the flight; 2026-11-05, As2W): the whole zone also held the front wheels on the bottom
      # landing of every descent, and the As2W teachers slowed their descents 0.86 -> 0.61-0.69 of the command.
      # Heights from the scan: the highest hit within wheel_ground_radius of the wheel, the median within 0.2 m of
      # the base (as top_flat_eval.py).
      scan: RayCastSensor = env.scene[scan_sensor_name]
      hit = scan.data.hit_pos_w
      valid = scan.data.distances >= 0.0
      wheel_xy = env.scene[wheel_cfg.name].data.site_pos_w[:, wheel_cfg.site_ids, :2]
      dw = torch.norm(hit[:, None, :, :2] - wheel_xy[:, :, None, :], dim=-1)
      wz = torch.where((dw <= wheel_ground_radius) & valid[:, None, :], hit[:, None, :, 2], -torch.inf).amax(dim=-1)
      db = torch.norm(hit[..., :2] - scan.data.pos_w[:, None, :2], dim=-1)
      bz = torch.where((db <= 0.2) & valid, hit[..., 2], torch.nan).nanmedian(dim=1).values
      zone = zone & (wz > bz[:, None] + higher)
    took_off = in_air & self.prev_contact
    self.off_zone = torch.where(took_off, zone, self.off_zone) & in_air
    self.prev_contact = ~in_air
    lifted = self.off_zone & drivable[:, None]
    env.extras["log"]["Metrics/wheel_flat_step_air"] = torch.sum(lifted) / torch.clamp(torch.sum(zone), min=1)
    return torch.sum(lifted.float(), dim=1)


class wheel_slide_back:
  """One count per wheel that reached up a step and came back down it: a contact that is not a stance (|F_z| <=
  stance_ratio |F_xy|: on a nosing or a riser) with the sole more than `reach` above the wheel's last stance, then
  the wheel's next stance more than `drop` below that contact (back on the tread it came from). Returned per second
  (events / step_dt), so with dt scaling the episode sum is weight times the number of such slides.

  The user (2026-11-04): after the wheelstep round the wheels still rolled back down a step in tinynav-sim (Go2W 11%,
  B2W 15% of the step-ups; real Go2W ~20%). wheel_nosing charged only a wheel hanging with its center over the tread
  top, and the riser contacts below it were 3-4x as many. This charges the outcome itself, whatever the contact was.
  A step down never counts (its contacts are not above the wheel's last stance), nor a lift with no contact.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    shape = (env.num_envs, len(cfg.params["asset_cfg"].site_names))
    self.stand_z = torch.full(shape, torch.inf, device=env.device)
    self.reach_z = torch.full(shape, -torch.inf, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.stand_z[env_ids] = torch.inf
    self.reach_z[env_ids] = -torch.inf

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    wheel_radius: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    reach: float = 0.03,
    drop: float = 0.04,
    stance_ratio: float = 2.0,
    stance_min_force: float = 10.0,
    min_force: float = 5.0,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    force = contact_sensor.data.force
    assert contact_sensor.data.found is not None and force is not None
    in_contact = contact_sensor.data.found > 0
    force_xy = torch.norm(force[..., :2], dim=-1)
    force_z = torch.abs(force[..., 2])
    stance = in_contact & (force_z > torch.clamp(stance_ratio * force_xy, min=stance_min_force))
    touch = in_contact & ~stance & (torch.norm(force, dim=-1) > min_force)
    sole_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - wheel_radius
    # a non-stance contact above the last stance: the wheel reaching up a step
    up = touch & (sole_z > self.stand_z + reach)
    self.reach_z = torch.where(up, torch.maximum(self.reach_z, sole_z), self.reach_z)
    active = ~_zero_command_mask(env, command_name, command_threshold)
    slid = stance & (sole_z < self.reach_z - drop) & active[:, None]
    self.stand_z = torch.where(stance, sole_z, self.stand_z)
    self.reach_z = torch.where(stance, -torch.inf, self.reach_z)
    env.extras["log"]["Metrics/wheel_slide_back"] = torch.sum(slid.float()) / env.num_envs
    return torch.sum(slid.float(), dim=1) / env.step_dt


class wheel_idle_step:
  """Idle wheel steps (go2-depth 2026-11-09, the user's wheeled stairs requirement "no idle steps"; tinynav-sim's
  accept_stairs.py definition): a wheel that takes off with no riser within `ahead` m ahead of it in the command's
  direction (the privileged scan, wheel_uneven_mask), stays off the ground longer than `min_air` s with its centre
  rising more than `min_apex` over the take-off height, and lands within `same_level` of that height: a lift that
  climbed nothing. Counted while the command can be driven on the wheels (moving, at most `max_lin_vel_y` sideways
  and `max_ang_vel_z` yaw, not marching). As2W blind a1c / a1d: its rear wheels made 2 such 7-8 cm lifts per climb,
  0.3+ m short of the first riser (tinynav start idle 1.0-1.5, bar 0.5), through stepclean / stepclean2.
  Returned per second (events / step_dt), so with dt scaling the episode sum is weight x idle steps."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    shape = (env.num_envs, len(cfg.params["asset_cfg"].site_names))
    self.z0 = torch.zeros(shape, device=env.device)
    self.apex = torch.zeros(shape, device=env.device)
    self.air_t = torch.zeros(shape, device=env.device)
    self.flat_off = torch.zeros(shape, dtype=torch.bool, device=env.device)
    self.prev_contact = torch.ones(shape, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.air_t[env_ids] = 0.0
    self.flat_off[env_ids] = False
    self.prev_contact[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    scan_sensor_name: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
    max_lin_vel_y: float = 0.1,
    max_ang_vel_z: float = 0.5,
    ahead: float = 0.3,
    min_air: float = 0.06,
    min_apex: float = 0.03,
    same_level: float = 0.03,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    dt = env.step_dt
    in_contact = contact_sensor.data.found > 0
    z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    drivable = (
      ~_zero_command_mask(env, command_name, command_threshold)
      & (torch.abs(command[:, 1]) <= max_lin_vel_y)
      & (torch.abs(command[:, 2]) <= max_ang_vel_z)
    )
    march = getattr(env.command_manager.get_term(command_name), "march", None)
    if march is not None:
      drivable &= ~march
    landing = in_contact & ~self.prev_contact
    # near stairs only (2026-11-09, As2W a1e 9000: counted everywhere, it charged the arcs' clock steps on flat ground
    # and the gait clock failed 11 forward+turn commands, 58-76% match, 0.8-1.1 touchdowns/s), as tinynav's near_stairs
    near_stairs = uneven_terrain_mask(env, scan_sensor_name)
    idle = (
      landing & self.flat_off & (self.air_t > min_air) & (self.apex - self.z0 > min_apex)
      & (torch.abs(z - self.z0) < same_level) & (drivable & near_stairs)[:, None]
    )
    takeoff = ~in_contact & self.prev_contact
    near = wheel_uneven_mask(env, scan_sensor_name, asset_cfg, ahead, ahead_command=command_name)
    self.flat_off = torch.where(takeoff, ~near, self.flat_off)
    self.z0 = torch.where(takeoff, z, self.z0)
    self.apex = torch.where(takeoff, z, torch.maximum(self.apex, z))
    self.air_t = torch.where(in_contact, torch.zeros_like(self.air_t), self.air_t + dt)
    self.prev_contact = in_contact
    env.extras["log"]["Metrics/wheel_idle_steps"] = idle.float().sum() / max(env.num_envs, 1)
    return idle.float().sum(dim=1) / dt


class wheel_step_touch:
  """Extra touches of the stair steps per wheel, minus clean step-ups (go2-depth 2026-11-06, the user: "每级台阶都只接触
  一次，现在可能有一级台阶接触多次的情况，主要发生在轮式上"). On ground uneven ahead of the wheel (wheel_uneven_mask with the
  command's direction) while the command moves:
  - a bump: the wheel's contact turns blocked (|F_xy| > `ratio` |F_z|: a riser face or a nosing) after `gap` s
    unblocked -- it ran into the riser instead of stepping over it;
  - a re-landing: a touchdown after >= `min_air` s off the ground that is neither blocked (a bump, counted above) nor
    a stance at least `min_rise` above or below the wheel's last stance (back on the tread it left, or on an edge);
  - a clean step-up pays `clean_share` back: a touchdown from the air as a stance >= `min_rise` above the last stance
    with no blocked contact since the wheel's last step-up.
  Returned per second (events / step_dt): with dt scaling the episode sum is weight x (extra touches - clean_share x
  clean step-ups). In tinynav-sim on the map3 flights (check/accept_map3.py) the delivered Go2W g14 13600 touched
  each step 2.27 (front) / 1.98 (rear) times climbing, its wheels hitting the riser first on 99% / 85% of the steps
  (the deployed blind_v2 52% / 96%, re-landing 13%), 46 riser touches per climb; its roll peaks (to 21 deg) came
  from wheels propped on riser faces. wheel_riser_lift pays from the block (or riserprox's proximity) on, and
  wheel_nosing charges a loaded tilted contact per second, so nothing charged the touch itself."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    shape = (env.num_envs, len(cfg.params["asset_cfg"].site_names))
    self.stand_z = torch.full(shape, torch.nan, device=env.device)
    self.air_t = torch.zeros(shape, device=env.device)
    self.blocked_t = torch.full(shape, 1e6, device=env.device)
    self.prev_contact = torch.ones(shape, dtype=torch.bool, device=env.device)
    self.bumped = torch.zeros(shape, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.stand_z[env_ids] = torch.nan
    self.air_t[env_ids] = 0.0
    self.blocked_t[env_ids] = 1e6
    self.prev_contact[env_ids] = True
    self.bumped[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    scan_sensor_name: str,
    wheel_radius: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    ratio: float = 1.2,
    min_force: float = 5.0,
    stance_ratio: float = 2.0,
    stance_min_force: float = 10.0,
    min_air: float = 0.06,
    min_rise: float = 0.04,
    gap: float = 0.1,
    uneven_margin: float = 0.15,
    clean_share: float = 1.0,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    force = contact_sensor.data.force
    assert contact_sensor.data.found is not None and force is not None
    dt = env.step_dt
    in_contact = contact_sensor.data.found > 0
    force_xy = torch.norm(force[..., :2], dim=-1)
    force_z = torch.abs(force[..., 2])
    stance = in_contact & (force_z > torch.clamp(stance_ratio * force_xy, min=stance_min_force))
    blocked = in_contact & (force_xy > torch.clamp(ratio * force_z, min=min_force))
    sole_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - wheel_radius
    uneven = wheel_uneven_mask(env, scan_sensor_name, asset_cfg, wheel_radius + uneven_margin, ahead_command=command_name)
    active = (~_zero_command_mask(env, command_name, command_threshold))[:, None] & uneven
    bump = blocked & (self.blocked_t >= gap) & active
    touchdown = in_contact & ~self.prev_contact & (self.air_t >= min_air)
    rise = sole_z - self.stand_z
    new_tread = stance & (torch.abs(rise) >= min_rise)   # NaN (no stance yet) compares False
    reland = touchdown & ~blocked & ~new_tread & active
    clean = touchdown & new_tread & (rise > 0.0) & ~self.bumped & active
    self.bumped = (self.bumped | blocked) & ~(stance & new_tread & (rise > 0.0))
    self.blocked_t = torch.where(blocked, 0.0, self.blocked_t + dt)
    self.air_t = torch.where(in_contact, 0.0, self.air_t + dt)
    self.prev_contact = in_contact
    self.stand_z = torch.where(stance, sole_z, self.stand_z)
    log = env.extras["log"]
    log["Metrics/wheel_bump"] = torch.sum(bump.float()) / env.num_envs
    log["Metrics/wheel_reland"] = torch.sum(reland.float()) / env.num_envs
    log["Metrics/wheel_clean_up"] = torch.sum(clean.float()) / env.num_envs
    events = bump.float() + reland.float() - clean_share * clean.float()
    return torch.sum(events, dim=1) / dt


def feet_support_uneven(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  scan_sensor_name: str,
  command_name: str,
  command_threshold: float = 0.1,
  min_feet: int = 2,
  uneven_radius: float = 0.45,
  uneven_threshold: float = 0.05,
) -> torch.Tensor:
  """Missing support on stairs: max(0, `min_feet` - feet in contact) while the command moves on uneven ground (go2-depth
  2026-11-06, the user: the B2 roll peak is high). In tinynav-sim the B2 blind l4 9000 rolled 8-14 deg on map3 flights
  (R3_up, R2b_up, L3_up; deployed 5.8-6.9) each time after 0.1-0.15 s on one foot or none: a rear foot left a step edge
  late while the diagonal pair already swung (check/roll_probe.py). Per second."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  n = (contact_sensor.data.found > 0).sum(dim=1)
  missing = torch.clamp(min_feet - n, min=0).float()
  active = ~_zero_command_mask(env, command_name, command_threshold)
  return missing * (active & uneven_terrain_mask(env, scan_sensor_name, uneven_radius, uneven_threshold)).float()


class feet_stand_uneven:
  """Feet off the ground while standing still on stairs (go2-depth 2026-11-07, the user: B2 "只有在下楼梯中途停住时会出现右后腿
  腾空的情况"): at the zero command (not marching) on uneven ground, after `after_s` s, the number of feet not in contact,
  per second. The B2 l8 11800 stood on three legs in 61-84% of its stops on 12-18 cm descents (eval_stairs' 4-feet
  share 0.16-0.39; l4 9000 1.00)."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.zero_time = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.zero_time[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    scan_sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
    after_s: float = 0.5,
    uneven_radius: float = 0.45,
    uneven_threshold: float = 0.05,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    zero = _zero_command_mask(env, command_name, command_threshold)
    march = getattr(env.command_manager.get_term(command_name), "march", None)
    if march is not None:
      zero = zero & ~march
    self.zero_time = torch.where(zero, self.zero_time + env.step_dt, 0.0)
    down = contact_sensor.data.found > 0
    missing = (~down).float().sum(dim=1)
    uneven = uneven_terrain_mask(env, scan_sensor_name, uneven_radius, uneven_threshold)
    return missing * ((self.zero_time >= after_s) & uneven).float()


def drive_leg_motion(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.1,
  max_lin_vel_y: float = 0.1,
  max_ang_vel_z: float = 0.1,
  uneven_radius: float = 0.45,
  uneven_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Leg joint speeds while a wheeled robot drives on even ground (go2-depth 2026-11-07, the user: Go2W "慢速前进的时候腿会
  前后晃动"): sum of squared joint velocities of `asset_cfg`'s joints (the legs) under drive_still_even's drive mask.
  At 0.2-0.3 m/s the Go2W g20b 18300 swung its thighs / calves ~2 deg (std) rolling, the deployed blind_v2 0.1-0.3."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  drive = (
    (torch.norm(command[:, :2], dim=1) + command[:, 2].abs() > command_threshold)
    & (command[:, 1].abs() <= max_lin_vel_y)
    & (command[:, 2].abs() <= max_ang_vel_z)
    & ~uneven_terrain_mask(env, sensor_name, uneven_radius, uneven_threshold)
  )
  march = getattr(env.command_manager.get_term(command_name), "march", None)
  if march is not None:
    drive = drive & ~march
  qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(qd), dim=1) * drive.float()


class drive_wheel_surge:
  """Wheel speed surging while a wheeled robot drives straight on even ground (go2-depth 2026-11-07, the user: Go2W
  "慢速前进的时候腿会前后晃动"): under drive_leg_motion's drive mask, once the command has held within `steady_tol` for
  `steady_s` s, the wheels' summed squared speed deviation from its own low-pass (time constant `tau` s). At 0.2-0.3 m/s
  in tinynav-sim the Go2W g23 18800's wheel speed swung by 1.2 rad/s (std; blind_v2 0.03) and the legs 2.8 deg with it; in
  mjlab play ~10% of the randomized robots did (p90 1.1 rad/s, p50 0.04)."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    n = asset.data.joint_vel[:, cfg.params["asset_cfg"].joint_ids].shape[1]
    self.lp = torch.zeros(env.num_envs, n, device=env.device)
    self.prev_cmd = torch.zeros(env.num_envs, 3, device=env.device)
    self.steady = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.lp[env_ids] = 0.0
    self.prev_cmd[env_ids] = 0.0
    self.steady[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
    max_lin_vel_y: float = 0.1,
    max_ang_vel_z: float = 0.1,
    uneven_radius: float = 0.45,
    uneven_threshold: float = 0.05,
    tau: float = 0.5,
    steady_s: float = 1.0,
    steady_tol: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    drive = (
      (torch.norm(command[:, :2], dim=1) + command[:, 2].abs() > command_threshold)
      & (command[:, 1].abs() <= max_lin_vel_y)
      & (command[:, 2].abs() <= max_ang_vel_z)
      & ~uneven_terrain_mask(env, sensor_name, uneven_radius, uneven_threshold)
    )
    march = getattr(env.command_manager.get_term(command_name), "march", None)
    if march is not None:
      drive = drive & ~march
    held = (command - self.prev_cmd).abs().amax(dim=1) <= steady_tol
    self.steady = torch.where(held, self.steady + env.step_dt, torch.zeros_like(self.steady))
    self.prev_cmd = torch.where(held.unsqueeze(1), self.prev_cmd, command)
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    on = drive & (self.steady >= steady_s)
    a = min(1.0, env.step_dt / tau)
    self.lp = torch.where(on.unsqueeze(1), self.lp + a * (qd - self.lp), qd)
    return torch.sum(torch.square(qd - self.lp), dim=1) * on.float()


def body_roll_uneven(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  uneven_radius: float = 0.45,
  uneven_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """The body's roll alone (projected gravity y, squared) on uneven ground (go2-depth 2026-11-06, the user: why the roll
  peak is high). body_orientation_terrain_aware scales roll and pitch together by 0.3 on stairs so the body may follow
  the slope; this charges only the sideways tilt there."""
  asset: Entity = env.scene[asset_cfg.name]
  roll = torch.square(asset.data.projected_gravity_b[:, 1])
  return roll * uneven_terrain_mask(env, sensor_name, uneven_radius, uneven_threshold).float()


def wheel_nosing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  scan_sensor_name: str,
  wheel_radius: float,
  asset_cfg: SceneEntityCfg,
  tilt_ratio: float = 0.5,
  min_force: float = 5.0,
  reach: float = 0.05,
  min_step: float | None = None,
  clearance: float = 0.02,
) -> torch.Tensor:
  """Number of wheels hanging on a stair nosing: in contact, the contact force
  tilted (|F_xy| > `tilt_ratio` |F_z|, |F| > `min_force`) and a tread top, the
  highest privileged scan hit within wheel_radius + `reach` of the wheel center,
  between the sole and the center (above the sole by more than `min_step`, default
  half the radius, and below the center + `clearance`).

  The user (2026-11-03): wheels that seemed up a step rolled back down it. In
  tinynav-sim 9-21% of the step-ups of the six wheeled policies landed on the
  nosing (wheel center 2-3 cm over the tread top) and slid back, 19-40% hung there
  and climbed over the edge; wheel_riser_lift paid the fraction lifted and
  foot_stumble is off for wheels (scratchpad check/stair_wheels.py). A riser face
  against the wheel (its top above the center) and a step lower than `min_step`
  (rolled over) do not count. Requires a world-frame contact force.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  scan: RayCastSensor = env.scene[scan_sensor_name]
  force = contact_sensor.data.force
  assert contact_sensor.data.found is not None and force is not None
  in_contact = contact_sensor.data.found > 0
  force_xy = torch.norm(force[..., :2], dim=-1)
  force_z = torch.abs(force[..., 2])
  tilted = in_contact & (force_xy > tilt_ratio * force_z) & (torch.norm(force, dim=-1) > min_force)
  center = asset.data.site_pos_w[:, asset_cfg.site_ids]  # [B, W, 3]
  sole_z = center[..., 2] - wheel_radius
  hit = scan.data.hit_pos_w
  near = (torch.norm(hit[:, None, :, :2] - center[:, :, None, :2], dim=-1) < wheel_radius + reach) & (
    scan.data.distances >= 0.0
  )[:, None, :]
  top_z = torch.where(near, hit[:, None, :, 2], -torch.inf).amax(dim=-1)
  step = 0.5 * wheel_radius if min_step is None else min_step
  perched = tilted & (top_z > sole_z + step) & (top_z < center[..., 2] + clearance)
  env.extras["log"]["Metrics/wheel_nosing"] = torch.mean(perched.float())
  return torch.sum(perched.float(), dim=1)


def _stepping_in_place(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  scan_sensor_name: str | None,
) -> torch.Tensor:
  """Marching in place, or a command with no forward part (sideways, turning in
  place): a wheeled robot moves by stepping. Not at the zero command and, with
  `scan_sensor_name`, only on even ground."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  stepping = ~_zero_command_mask(env, command_name, command_threshold) & (torch.abs(command[:, 0]) <= 1e-6)
  if scan_sensor_name is not None:
    stepping &= ~uneven_terrain_mask(env, scan_sensor_name)
  return stepping


def stepping_wheel_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str = "wheel_vel",
  command_threshold: float = 0.1,
  scan_sensor_name: str | None = None,
) -> torch.Tensor:
  """Summed |wheel velocity target| (rad/s) while stepping without a forward command
  (see _stepping_in_place), on every wheel: a grounded wheel then brakes by its
  damping, and a wheel in the air lands still.

  standing_wheel_target's cost for the same reason at the zero command: the policy
  sets the targets and observes them (its last action), the wheel speed it does
  not set alone. The v11w teachers' (charged on the wheel speed) wheels in the air turned
  at ~2.5 rad/s stepping sideways (Go2W 2000: fronts back, rears forward).
  """
  term = env.action_manager.get_term(action_name)
  target = term.raw_action * term.scale
  stepping = _stepping_in_place(env, command_name, command_threshold, scan_sensor_name)
  return torch.sum(torch.abs(target), dim=1) * stepping.float()


def _hip_outward(env: ManagerBasedRlEnv, hip_cfg: SceneEntityCfg) -> torch.Tensor:
  """Hip abduction away from the body relative to the default stance (rad), per hip
  of `hip_cfg`: +q on the left legs ({F,R}L_hip_joint), -q on the right ones."""
  asset: Entity = env.scene[hip_cfg.name]
  ids = hip_cfg.joint_ids
  names = [asset.joint_names[i] for i in ids]
  side = torch.tensor([1.0 if n[1] == "L" else -1.0 for n in names], device=env.device)
  return (asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]) * side


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

  With `max_speed_ratio`, a lift while the base moves along the command faster
  than that times the commanded speed earns nothing (its fraction is used up):
  paid per riser, the term made Go2W and As2W climb map3 at 1.46x and 1.24x the
  command (2026-10-21).

  With `land_share`, that share of the pay per riser is held back for a clean
  landing instead: the wheel's first touchdown after the trigger is a stance
  (|F_z| > stance_ratio |F_xy|) with the sole within `land_tolerance` of the riser
  top or above it; the fraction lifted pays only the rest. Paying for the fraction
  alone, the wheeled policies (2026-11-03, the user: wheels rolled back down a
  step) landed on the nosing with the center 2-3 cm over the tread top and slid
  back in 9-21% of the step-ups in tinynav-sim (see wheel_nosing).
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
    self.prev_contact = torch.ones(shape, dtype=torch.bool, device=env.device)
    self.land_paid = torch.ones(shape, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.record_z[env_ids] = -torch.inf
    self.riser_top[env_ids] = torch.inf
    self.best[env_ids] = 0.0
    self.time_left[env_ids] = 0.0
    self.airborne[env_ids] = False
    self.prev_contact[env_ids] = True
    self.land_paid[env_ids] = True

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
    max_speed_ratio: float | None = None,
    land_share: float = 0.0,
    land_tolerance: float = 0.02,
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

    active = ~_zero_command_mask(env, command_name, command_threshold)
    # A clean landing on the pending riser's top (judged before this step's trigger moves it on).
    touchdown = in_contact & ~self.prev_contact
    tried = touchdown & ~self.land_paid & (self.time_left > 0.0) & active[:, None]
    clean = (
      tried & stance
      & (sole_z >= self.riser_top - land_tolerance)
    )
    self.land_paid |= touchdown  # one try per riser: the first touchdown after the trigger
    self.prev_contact = in_contact
    land_gain = land_share * clean.float()

    blocked = in_contact & (force_xy > torch.clamp(ratio * force_z, min=min_force))
    riser = blocked & (top_z > self.record_z + tolerance)
    new = riser & ~(torch.abs(top_z - self.riser_top) <= tolerance)
    self.land_paid = torch.where(new, False, self.land_paid)
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

    lifting = (self.time_left > 0.0) & ~in_contact & active[:, None]
    fraction = torch.clamp((sole_z - self.start_z) / self.needed, min=0.0, max=1.0)
    gain = torch.where(lifting, torch.clamp(fraction - self.best, min=0.0), 0.0)
    self.best = self.best + gain
    gain = (1.0 - land_share) * gain + land_gain
    if max_speed_ratio is not None:
      command = env.command_manager.get_command(command_name)
      assert command is not None
      cmd = command[:, :2]
      speed = torch.norm(cmd, dim=1)
      along = (asset.data.root_link_lin_vel_b[:, :2] * cmd).sum(dim=1) / speed.clamp(min=1e-6)
      gain = torch.where((along > max_speed_ratio * speed)[:, None], 0.0, gain)
    env.extras["log"]["Metrics/wheel_riser_blocked"] = torch.mean(riser.float())
    if land_share > 0.0:
      env.extras["log"]["Metrics/wheel_clean_landing"] = torch.sum(clean.float()) / torch.clamp(
        torch.sum(tried.float()), min=1
      )
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
      total_command = _command_activity(env, command_name)
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
  even: bool = False,
  ratio_high: float | None = None,
  axis_min: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  max_payload: float | None = None,
) -> torch.Tensor:
  """On uneven ground (stairs), how far the speed along the commanded direction
  falls short of `ratio` times the commanded speed, as a fraction of the command
  (0 to 1). Going faster than that earns nothing, so it cannot pay to rush.
  With `ratio_high` it also charges going faster than that times the command, the
  same way (two-sided).

  With `axis_min`, commands whose forward and sideways parts are both at least
  that (diagonals) are judged per axis instead, as the acceptance does: the sum
  of the forward and the sideways deficits, each a fraction of its own part.
  Along the direction, going faster on one axis hides a shortfall on the other
  (Go2W v12w 4000 at (0.6, 0.4): x 108%, y 68%, along 0.96x, no charge).

  The wheeled stairs policies (2026-10-20) went down map3's flights at 0.5-0.7x
  the command: at tracking std 0.35 half of 0.4 m/s costs 28% of the tracking
  reward. An ungated descent progress reward made the go2-depth wheeled robots
  rush down at 1.35-5x instead.

  With `even` the same on even ground instead (2026-10-23: the phased legged
  teachers marched in place at 1.5 m/s commands: walking at 1.0 earned almost no
  tracking at std 0.35 and paid the stepping costs; the deficit costs ~1.8/s for
  not moving at 1.5 m/s at weight -2, ~0.4 at 1.0 m/s).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd = command[:, :2]
  speed = torch.norm(cmd, dim=1)
  direction = cmd / speed.clamp(min=1e-6).unsqueeze(1)
  along = (asset.data.root_link_lin_vel_b[:, :2] * direction).sum(dim=1)
  deficit = ((ratio * speed - along) / speed.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
  if ratio_high is not None:
    deficit = deficit + ((along - ratio_high * speed) / speed.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
  if axis_min is not None:
    part = cmd.abs()
    moved = asset.data.root_link_lin_vel_b[:, :2] * torch.sign(cmd)
    axes = ((ratio * part - moved) / part.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
    if ratio_high is not None:
      axes = axes + ((moved - ratio_high * part) / part.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
    diagonal = (part >= axis_min).all(dim=1)
    deficit = torch.where(diagonal, axes.sum(dim=1), deficit)
  uneven = uneven_terrain_mask(env, sensor_name)
  active = (speed >= min_speed) & (~uneven if even else uneven)
  if max_payload is not None and getattr(env, "payload_kg", None) is not None:
    # Only robots carrying at most max_payload kg (2026-11-06, Go2W blind g10 9700: on flat ground the light robots
    # ran (0.2, 0) at 127%, the 9-12 kg ones 113%, with the wheels down; a blind policy cannot tell them apart and
    # tuned its speed to the mixed fleet, while the acceptance drives unloaded robots).
    active &= env.payload_kg <= max_payload
  return torch.where(active, deficit, torch.zeros_like(deficit))


MARCH_ACTIVITY = 0.5
"""_command_activity of a robot marching in place: a walking command to the gait
and standing terms (see TerrainAwareVelocityCommandCfg.march_prob)."""


def _command_activity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """|(vx, vy)| + |wz| of the command; MARCH_ACTIVITY for a robot told to march in
  place (zero command with the gait clock running), so the gait terms apply to it
  and the standing terms do not."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  march = getattr(env.command_manager.get_term(command_name), "march", None)
  if march is not None:
    total_command = torch.where(march, MARCH_ACTIVITY, total_command)
  return total_command


def _zero_command_mask(
  env: ManagerBasedRlEnv, command_name: str, command_threshold: float
) -> torch.Tensor:
  return _command_activity(env, command_name) <= command_threshold


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


def _rest_pose(
  asset: Entity, joint_ids, default: torch.Tensor, rest_hip: float, blend: torch.Tensor | None = None
) -> torch.Tensor:
  """`default` (envs, joints of `joint_ids`) with the hip abduction joints moved to
  the rest stance: `rest_hip` rad outward (negative: inward) on the left legs,
  mirrored on the right; with `blend` (envs,) only that fraction of the way."""
  names = asset.joint_names if isinstance(joint_ids, slice) else [asset.joint_names[i] for i in joint_ids]
  rest = default.clone()
  for k, name in enumerate(names):
    if name.endswith("hip_joint"):
      target = rest_hip if name[1] == "L" else -rest_hip
      if blend is None:
        rest[:, k] = target
      else:
        rest[:, k] = default[:, k] + blend * (target - default[:, k])
  return rest


def stand_still_even(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.1,
  calm_std: float = 0.1,
  uneven_radius: float = 0.45,
  uneven_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  rest_hip: float | None = None,
  hold_term: str | None = None,
) -> torch.Tensor:
  """Penalize deviation from the default pose while standing on even ground; with
  `rest_hip`, from the rest stance (hips at that outward angle, see _rest_pose). With
  `hold_term` (a standing_position_hold term), not once that term holds its anchor."""
  asset: Entity = env.scene[asset_cfg.name]
  desired = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  if rest_hip is not None:
    desired = _rest_pose(asset, asset_cfg.joint_ids, desired, rest_hip)
  diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - desired
  cost = torch.sum(torch.square(diff), dim=1)
  scale = _standing_scale(env, command_name, command_threshold, calm_std, asset_cfg)
  even = ~uneven_terrain_mask(env, sensor_name, uneven_radius, uneven_threshold)
  if hold_term is not None:
    even &= ~_hold_locked(env, hold_term)
  return cost * scale * even.float()


def _hold_locked(env: ManagerBasedRlEnv, hold_term: str) -> torch.Tensor:
  """Robots whose standing_position_hold term `hold_term` holds its anchor (standing, settled or past its push
  window); as of that term's last call (the previous step when it comes later in the reward order)."""
  return env.reward_manager.get_term_cfg(hold_term).func.locked


def drive_still_even(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.1,
  max_lin_vel_y: float = 0.1,
  max_ang_vel_z: float = 0.1,
  uneven_radius: float = 0.45,
  uneven_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared deviation of the selected joints from the default pose while a wheeled
  robot drives on even ground (a moving command with |vy| and |wz| within the limits,
  the wheels' share): stand_still_even for driving. Unlike the posture term's exp
  kernel it does not flatten far from the default, so a crouch keeps a pull back."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  # Marching in place counts as activity (_command_activity) but is no drive.
  drive = (
    (torch.norm(command[:, :2], dim=1) + command[:, 2].abs() > command_threshold)
    & (command[:, 1].abs() <= max_lin_vel_y)
    & (command[:, 2].abs() <= max_ang_vel_z)
    & ~uneven_terrain_mask(env, sensor_name, uneven_radius, uneven_threshold)
  )
  diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(diff), dim=1) * drive.float()


class drive_still_even_lp:
  """drive_still_even on the joints' deviation low-passed over `tau_s` (exponential average, restarted at
  each reset). A held crouch pays as before; the step-to-step wobble from the action noise averages out,
  so the policy gains nothing by cancelling its own noise through the last-action input. Charged
  per step (B2W v13wg, 2026-10-28), that cancelling grew a thigh / calf action loop with gains 4-36
  from 2200 on (w13/big_gain2.py; v13wf below 2). Right after resets, deterministic actions then ran away
  x3 per step and the robots sat down (86 falls at the 3000 gate)."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.avg: torch.Tensor | None = None
    self.restart = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.restart[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    command_threshold: float = 0.1,
    max_lin_vel_y: float = 0.1,
    max_ang_vel_z: float = 0.1,
    uneven_radius: float = 0.45,
    uneven_threshold: float = 0.05,
    tau_s: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    if self.avg is None:
      self.avg = diff.clone()
    self.avg[self.restart] = diff[self.restart]
    self.restart[:] = False
    alpha = min(1.0, env.step_dt / tau_s)
    self.avg += alpha * (diff - self.avg)
    drive = (
      (torch.norm(command[:, :2], dim=1) + command[:, 2].abs() > command_threshold)
      & (command[:, 1].abs() <= max_lin_vel_y)
      & (command[:, 2].abs() <= max_ang_vel_z)
      & ~uneven_terrain_mask(env, sensor_name, uneven_radius, uneven_threshold)
    )
    return torch.sum(torch.square(self.avg), dim=1) * drive.float()


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

  With `march` it holds a robot marching in place instead (the command term's
  march flag): the fifth-round legged policies drifted 41-56 cm per 20 s marching
  (2026-10-22), which the zero-command tracking terms barely see either. Their
  steps sway the base, so `moving_speed` and `moving_yaw_rate` must sit above that
  sway.

  With `push_window` (seconds) the anchor follows pushes instead of the base's own
  motion: for `push_window` after a push (push_by_setting_velocity_marked) or after
  standing starts (braking), never because the base moves. Speed-based following
  lets the policy buy its way out: at -40 the march-hold pilots (2026-10-23) walked
  off at ~0.35 m/s, just over moving_speed 0.25, 7 m per 20 s of marching. With
  `settle` the following ends early once the base has settled (no faster than
  `moving_speed` / `moving_yaw_rate`, at least 0.2 s after the push or start):
  only a push or a start can set the anchor going, settling only stops it.
  With `smooth_s` (one gait period) the hold sees the base's mean pose over that
  window instead of the instant one, so the stepping sway cancels, and `settle`
  waits until the mean speed and yaw rate over the window are at most
  `settle_speed` / `settle_yaw_rate` (and one window has passed): marching steadily
  in place, no longer sliding out a push. The instant rule (2026-10-24, legged v3)
  locked at 0.25 m/s 0.2 s after a push, and a ~3 cm mean offset of sway and push
  residue buried the 0.2-0.7 cm/s drift the 20 s march acceptance measures.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.anchor_pos = torch.zeros(env.num_envs, 2, device=env.device)
    self.anchor_yaw = torch.zeros(env.num_envs, device=env.device)
    self.needs_anchor = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self.was_standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.excused = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.excused_since = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.locked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # standing, anchor held
    self.seen_push = torch.full((env.num_envs,), -(10**9), dtype=torch.long, device=env.device)
    self.history: torch.Tensor | None = None  # x, y, cos / sin heading over `smooth_s`
    self.slot = 0

  def _window_mean(
    self, env: ManagerBasedRlEnv, pos: torch.Tensor, yaw: torch.Tensor, window_s: float
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean base position and heading over the last `window_s`, and the mean speed and
    yaw rate across it (displacement and heading change over the window)."""
    n = max(1, round(window_s / env.step_dt))
    sample = torch.stack((pos[:, 0], pos[:, 1], torch.cos(yaw), torch.sin(yaw)), dim=1)
    if self.history is None or self.history.shape[1] != n:
      self.history = sample.unsqueeze(1).repeat(1, n, 1)
      self.slot = 0
    fresh = self.needs_anchor
    self.history[fresh] = sample[fresh].unsqueeze(1)
    oldest = self.history[:, self.slot].clone()
    self.history[:, self.slot] = sample
    self.slot = (self.slot + 1) % n
    mean = self.history.mean(dim=1)
    span = n * env.step_dt
    speed = torch.norm(pos - oldest[:, :2], dim=1) / span
    yaw_rate = torch.abs(wrap_to_pi(yaw - torch.atan2(oldest[:, 3], oldest[:, 2]))) / span
    return mean[:, :2], torch.atan2(mean[:, 3], mean[:, 2]), speed, yaw_rate

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    # Set at the next call: the root pose may still be the previous episode's.
    self.needs_anchor[env_ids] = True
    self.was_standing[env_ids] = False
    self.excused[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    moving_speed: float = 0.1,
    moving_yaw_rate: float = 0.2,
    max_dist: float = 0.3,
    yaw_scale: float = 0.3,
    march: bool = False,
    push_window: float | None = None,
    settle: bool = False,
    smooth_s: float | None = None,
    settle_speed: float = 0.03,
    settle_yaw_rate: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    standing, pos, yaw = self._follow(
      env, command_name, command_threshold, moving_speed, moving_yaw_rate, march, push_window, settle,
      smooth_s, settle_speed, settle_yaw_rate, asset_cfg,
    )
    dist = torch.norm(pos - self.anchor_pos, dim=1).clamp(max=max_dist)
    turn = torch.abs(wrap_to_pi(yaw - self.anchor_yaw))
    return standing.float() * (dist + yaw_scale * turn)

  def _follow(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float,
    moving_speed: float,
    moving_yaw_rate: float,
    march: bool,
    push_window: float | None,
    settle: bool,
    smooth_s: float | None,
    settle_speed: float,
    settle_yaw_rate: float,
    asset_cfg: SceneEntityCfg,
    spin_lin: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move the anchor with the base where it follows; the standing mask, the (smoothed) base position and
    heading. Sets self.follow. With `spin_lin` (spin_position_hold) the mask is a turn in place instead: a linear
    command of at most spin_lin m/s and |wz| above command_threshold, not marching."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.root_link_pos_w[:, :2]
    yaw = asset.data.heading_w
    speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
    calm_speed, calm_yaw_rate, calm_after = moving_speed, moving_yaw_rate, 0.2
    if smooth_s is not None:
      pos, yaw, speed, yaw_rate = self._window_mean(env, pos, yaw, smooth_s)
      calm_speed, calm_yaw_rate, calm_after = settle_speed, settle_yaw_rate, smooth_s
    if spin_lin is not None:
      command = env.command_manager.get_command(command_name)
      standing = (torch.norm(command[:, :2], dim=1) <= spin_lin) & (command[:, 2].abs() > command_threshold)
      marching = getattr(env.command_manager.get_term(command_name), "march", None)
      if marching is not None:
        standing = standing & ~marching
    elif march:
      standing = getattr(env.command_manager.get_term(command_name), "march", None)
      assert standing is not None, "march: the command term has no march flag"
    else:
      standing = _zero_command_mask(env, command_name, command_threshold)
    if push_window is None:
      follow = (
        ~standing | (speed > moving_speed) | (yaw_rate > moving_yaw_rate) | self.needs_anchor
      )
    else:
      step = env.common_step_counter
      new = standing & ~self.was_standing
      self.was_standing = standing.clone()
      stamp = getattr(env, "_push_step", None)
      if stamp is not None:
        new |= stamp > self.seen_push
        self.seen_push = stamp.clone()
      self.excused |= new
      self.excused_since = torch.where(new, step, self.excused_since)
      elapsed = (step - self.excused_since) * env.step_dt
      self.excused &= elapsed <= push_window
      if settle:
        settled = (speed <= calm_speed) & (yaw_rate <= calm_yaw_rate) & (elapsed >= calm_after)
        self.excused &= ~settled
      follow = ~standing | self.excused | self.needs_anchor
    self.anchor_pos = torch.where(follow.unsqueeze(1), pos, self.anchor_pos)
    self.anchor_yaw = torch.where(follow, yaw, self.anchor_yaw)
    self.needs_anchor[:] = False
    self.follow = follow
    self.locked = standing & ~follow
    return standing, pos, yaw


class spin_position_hold(standing_position_hold):
  """Penalize the base wandering from its spot while turning in place (go2-depth 2026-11-06, the user's design: a
  position hold for pure-yaw commands). The command is a turn in place (|v_xy| <= `spin_lin`, |wz| above
  `command_threshold`, not a march): an anchor is set where the turn starts and the base's distance from it costs
  (clamped to `max_dist`); any other command lets the anchor follow the base again (reset). go2-home's march-hold
  lessons: the anchor locks only once the base has settled (mean speed over `smooth_s`, one gait period, at most
  `settle_speed`; the yaw rate is ignored here), at the latest `push_window` s after the turn starts or a push; the
  cost is on the mean position over `smooth_s` (low-passed, so the stepping sway cancels and no per-step gain feeds
  back on the actions). The critic sees the offset (mdp.spin_hold_offsets) and an adaptive constraint scales the
  weight ("spin_hold" metric). self.dist holds the cost's distance for the metric."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    spin_lin: float = 0.05,
    max_dist: float = 0.3,
    push_window: float = 1.5,
    smooth_s: float = 0.6,
    settle_speed: float = 0.03,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    spinning, pos, _ = self._follow(
      env, command_name, command_threshold, 0.1, 1e3, False, push_window, True, smooth_s, settle_speed, 1e3,
      asset_cfg, spin_lin=spin_lin,
    )
    self.dist = torch.norm(pos - self.anchor_pos, dim=1).clamp(max=max_dist) * self.locked.float()
    return self.dist


class standing_target_hold(standing_position_hold):
  """Penalize the legs' position targets moving once the robot has come to rest while commanded to stand:
  the norm of the change of the targets (low-passed over `tau_s`, so the step-to-step action noise averages
  out, see drive_still_even_lp) since the anchor locked, clamped to `max_dev` (rad). The anchor follows as in
  standing_position_hold with the same params (one gait period's smoothing, push window, settling).

  A wheeled robot standing on braked wheels moves its base whenever its legs move. The B2W final (polish 7000,
  kp 160) kept changing its leg targets after stops (3-12 deg summed over the last 4 s) and crept 0.75-2.22 cm
  per 4 s (bar 1); frozen targets at kp 240 ("legkp15") crept 0.21-0.91 (2026-10-29, scratchpad
  w17/creep_split.py). The base hold (standing_position_hold, -40 on B2W) sees the creep only as the cm it
  makes, which the posture terms' pull toward the default pose outbid.

  With `anchor_s` the anchor is the mean of the targets over the first `anchor_s` after the lock (the low-pass
  restarts from it, nothing is charged meanwhile), not the low-passed target at the lock: that one lags the
  targets by ~`tau_s`, and after a stop, when the targets still move as the anchor locks, holding still paid
  0.8-1.1 per second (3.6 standing from the start) that only moving the targets back to the lagged average could
  cut (B2W v13wo teacher 4000, 2026-11-01, scratchpad w20/rew_attr2.py)."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.avg: torch.Tensor | None = None
    self.anchor_target: torch.Tensor | None = None
    self.restart = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self.held = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # steps since the lock

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self.restart[slice(None) if env_ids is None else env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    moving_speed: float = 0.1,
    moving_yaw_rate: float = 0.2,
    push_window: float | None = None,
    settle: bool = False,
    smooth_s: float | None = None,
    settle_speed: float = 0.03,
    settle_yaw_rate: float = 0.1,
    tau_s: float = 0.5,
    max_dev: float = 0.5,
    anchor_s: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    standing, _, _ = self._follow(
      env, command_name, command_threshold, moving_speed, moving_yaw_rate, False, push_window, settle,
      smooth_s, settle_speed, settle_yaw_rate, asset_cfg,
    )
    asset: Entity = env.scene[asset_cfg.name]
    target = asset.data.joint_pos_target[:, asset_cfg.joint_ids]
    if self.avg is None or self.anchor_target is None:
      self.avg = target.clone()
      self.anchor_target = target.clone()
    self.avg[self.restart] = target[self.restart]
    self.restart[:] = False
    if anchor_s is None:
      self.avg += min(1.0, env.step_dt / tau_s) * (target - self.avg)
      self.anchor_target = torch.where(self.follow.unsqueeze(1), self.avg, self.anchor_target)
    else:
      self.held = torch.where(self.follow, 0, self.held + 1)
      building = (~self.follow & (self.held <= max(1, round(anchor_s / env.step_dt)))).unsqueeze(1)
      n = self.held.clamp(min=1).unsqueeze(1).float()
      mean = torch.where(n > 1, self.anchor_target + (target - self.anchor_target) / n, target)
      self.anchor_target = torch.where(self.follow.unsqueeze(1), target, torch.where(building, mean, self.anchor_target))
      self.avg += min(1.0, env.step_dt / tau_s) * (target - self.avg)
      self.avg = torch.where(self.follow.unsqueeze(1) | building, self.anchor_target, self.avg)
    dev = torch.norm(self.avg - self.anchor_target, dim=1).clamp(max=max_dev)
    return standing.float() * dev


def march_joint_deviation(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Squared distance of the selected joints from their default while marching in place
  (the command term's march flag). With the hips: each step sets its foot down at the
  default stance, so a march resets a stance that drifted in or out while standing (user
  2026-10-24 ~23:20); standing still cannot move a planted foot back."""
  asset: Entity = env.scene[asset_cfg.name]
  marching = getattr(env.command_manager.get_term(command_name), "march", None)
  assert marching is not None, "march_joint_deviation: the command term has no march flag"
  ids = asset_cfg.joint_ids
  dev = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
  return marching.float() * torch.sum(torch.square(dev), dim=1)


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

  `offset_ratio_scale` adds the sideways offset as a share of the distance walked
  along the line (at least `offset_min_distance`), / `offset_ratio_scale` and
  clamped at 1: the acceptance judges the offset per distance walked (<= 5%), and
  walking slower does not lower a share (user 2026-10-23 ~02:50).
  `offset_cmd_distance` divides by the smaller of that distance and the commanded one
  (command speed x time on the line) instead: walking faster than the command does
  not lower the share either. An offset that grows with time, not distance (a
  stepping robot's back-slip per step at a fixed step rate), is a smaller share of a
  longer walk: the B2W perception teacher (2026-11-01) walked (0, 0.15, 0) at
  121-129% of the command with this term at 2.3x (acceptance bar 120%).
  `offset_cmd_only` divides by the commanded distance alone (2026-11-02): with the
  smaller of the two, a robot behind the command (accelerating into the line: walked
  / commanded p50 0.95 1-2 s in) still lowers the share by walking faster, on 20%
  of the B2W polish 9000's sideways line steps (32% from 2 s in), and the perception
  teacher with `offset_cmd_distance` and this term at 3x went (0, 0.15, 0) from 115
  to 126% of the command over 10000-12000.

  `sideways_only` limits it to commands without a forward part (walking sideways:
  the sixth-round wheeled policies turned up to 15-16 deg per 10 s there, p90).
  `max_speed` and `forward_only` limit it to commands up to that planar speed and
  without a sideways part: penalized at 1.5 m/s and sideways, Go2 and As2
  (2026-10-20) gave up 10 points of top speed and of diagonal sideways speed,
  where walking slower drifted less.

  With `uneven_sensor_name` only the heading part counts on uneven ground (stairs),
  times `uneven_scale`: the acceptance judges the line on flat ground, and on stairs
  climbing pushes the base sideways (see the "stairline" switch in apply_blind_phased).
  `uneven_share` keeps the offset share there too, next to the heading (the "stairshare"
  switch). With `uneven_heading_obs` (a critic observation term of mdp.heading_error)
  the heading part on stairs is the heading error the policy observes, clamped at
  `uneven_turn_max` instead of 1 (the "stairhead" switch). `uneven_heading_gain` scales that heading part
  alone (the "stairhead3" switch).

  `settle_pos` (the position `SETTLE_S` into the line) and `since` (steps since the
  line started) do not enter the cost: the "back_drift" constraint metric measures
  from there, past the start's carry-over (mdp.constraints._back_drift).
  """

  SETTLE_S = 1.0

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    n = env.num_envs
    self.ref_heading = torch.zeros(n, device=env.device)
    self.ref_pos = torch.zeros(n, 2, device=env.device)
    self.ref_dir = torch.zeros(n, 2, device=env.device)
    self.prev_command = torch.zeros(n, 3, device=env.device)
    self.active = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.settle_pos = torch.zeros(n, 2, device=env.device)
    self.since = torch.zeros(n, dtype=torch.long, device=env.device)

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
    sideways_only: bool = False,
    offset_ratio_scale: float | None = None,
    offset_min_distance: float = 0.5,
    offset_cmd_distance: bool = False,
    offset_cmd_only: bool = False,
    uneven_sensor_name: str | None = None,
    uneven_scale: float = 1.0,
    uneven_share: bool = False,
    uneven_heading_obs: str | None = None,
    uneven_turn_max: float = 1.0,
    uneven_heading_gain: float = 1.0,
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
    if sideways_only:
      straight &= command[:, 0].abs() <= 1e-6
    changed = (command - self.prev_command).abs().amax(dim=1) > 1e-6
    start = straight & (~self.active | changed)
    # Commanded direction in the world frame at the start.
    v = command[:, :2] / speed.clamp(min=1e-6).unsqueeze(1)
    c, s = torch.cos(heading), torch.sin(heading)
    direction = torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], dim=1)
    self.ref_heading = torch.where(start, heading, self.ref_heading)
    self.ref_pos = torch.where(start.unsqueeze(1), pos, self.ref_pos)
    self.ref_dir = torch.where(start.unsqueeze(1), direction, self.ref_dir)
    self.since = torch.where(start, torch.zeros_like(self.since), self.since + 1)
    settling = (self.since.float() * env.step_dt <= self.SETTLE_S).unsqueeze(1)
    self.settle_pos = torch.where(settling, pos, self.settle_pos)
    self.active = straight
    self.prev_command = command.clone()
    turn = torch.abs(wrap_to_pi(heading - self.ref_heading))
    cost = (turn / heading_scale).clamp(max=1.0)
    rel = pos - self.ref_pos
    side = torch.abs(self.ref_dir[:, 0] * rel[:, 1] - self.ref_dir[:, 1] * rel[:, 0])
    if lateral_scale is not None:
      cost = cost + (side / lateral_scale).clamp(max=1.0)
    share = torch.zeros_like(cost)
    if offset_ratio_scale is not None:
      along = (self.ref_dir * rel).sum(dim=1)
      # The command is constant on the line (a change restarts it).
      if offset_cmd_only:
        along = speed * self.since.float() * env.step_dt
      elif offset_cmd_distance:
        along = torch.minimum(along, speed * self.since.float() * env.step_dt)
      along = along.clamp(min=offset_min_distance)
      share = (side / along / offset_ratio_scale).clamp(max=1.0)
      cost = cost + share
    if uneven_sensor_name is not None:
      uneven = uneven_terrain_mask(env, uneven_sensor_name)
      stair_turn = turn
      if uneven_heading_obs is not None:
        # The observation's target heading (it integrates the commanded yaw rate from the
        # reset on and stays within its max_error of the base heading).
        observed = env.observation_manager.get_term_cfg("critic", uneven_heading_obs).func
        stair_turn = torch.abs(wrap_to_pi(observed.target - heading))
      stairs = (stair_turn / heading_scale).clamp(max=uneven_turn_max) * uneven_heading_gain + (share if uneven_share else 0.0)
      cost = torch.where(uneven, stairs * uneven_scale, cost)
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
    foot_uneven_radius: float | None = None,
    march_only: bool = False,
    per_second: bool = False,
  ) -> torch.Tensor:
    """With `march_only` (go2-depth 2026-11-06, "marchclear") only landings while marching in place count, and with
    `per_second` the cost is per landing (divided by step_dt): per step it is weight x dt per landing, too little to
    lift a wheeled march above ~2-3 cm.
    If `scan_sensor_name` is given, overshoot is not penalized on uneven terrain,
    and with `even_only` nothing is. With `foot_uneven_radius`, uneven is judged
    per foot within that radius (wheel_uneven_mask; asset_cfg's sites) instead of
    around the base (wheeled rear wheels stepping on a landing, 2026-11-03).

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
      if foot_uneven_radius is not None:
        uneven = wheel_uneven_mask(env, scan_sensor_name, asset_cfg, foot_uneven_radius)
      else:
        uneven = uneven_terrain_mask(env, scan_sensor_name)[:, None]
      uneven_error = torch.zeros_like(error) if even_only else torch.clamp(error, max=0.0)
      error = torch.where(uneven, uneven_error, error)
    active = ~_zero_command_mask(env, command_name, command_threshold)
    if march_only:
      march = getattr(env.command_manager.get_term(command_name), "march", None)
      active = march if march is not None else torch.zeros_like(active)
    cost = torch.sum(torch.square(error) * landed.float(), dim=1) * active.float()

    num_landings = torch.sum(landed.float())
    env.extras["log"]["Metrics/swing_clearance_mean"] = torch.sum(
      clearance * landed.float()
    ) / torch.clamp(num_landings, min=1)

    self.has_liftoff &= ~landed
    self.prev_contact = in_contact
    return cost / env.step_dt if per_second else cost


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


def feet_edge(
  env: ManagerBasedRlEnv,
  command_name: str,
  contact_sensor_name: str,
  edge_sensor_names: tuple[str, ...],
  span: float = 0.03,
) -> torch.Tensor:
  """Number of feet in contact standing at a stair edge, on robots that spawned on
  the stairs (Extreme Parkour's feet_edge; the user 2026-10-30, legged group).

  Each foot has a ray grid around it (`edge_sensor_names`, in the contact sensor's
  foot order); a foot is at an edge when the hit heights under it span more than
  `span` or a ray misses. The legged finals (2026-10-30, scratchpad
  w18/kit/go2-blind/tools/foot_edge_diag.py, +-4 cm grid, 0.5 m/s) stood at edges
  17-19% of their stance time and 21-23% of their touchdowns, about what random
  footfalls on 25-32 cm treads give, and slid there faster climbing 15-18 cm steps
  (10-13 vs 7 cm/s); none fell in simulation.
  """
  at_edge = feet_at_edge(env, contact_sensor_name, edge_sensor_names, span)
  stairs = env.command_manager.get_term(command_name).is_stair_env
  return torch.sum(at_edge, dim=1).float() * stairs.float()


def feet_at_edge(
  env: ManagerBasedRlEnv, contact_sensor_name: str, edge_sensor_names: tuple[str, ...], span: float = 0.03
) -> torch.Tensor:
  """Per foot (envs, feet): in contact and at an edge (see feet_edge)."""
  contact: ContactSensor = env.scene[contact_sensor_name]
  assert contact.data.found is not None
  edge = []
  for name in edge_sensor_names:
    data = env.scene[name].data
    miss = data.distances < 0.0
    z = data.hit_pos_w[..., 2]
    spread = z.masked_fill(miss, -1e3).amax(dim=1) - z.masked_fill(miss, 1e3).amin(dim=1)
    edge.append((spread > span) | miss.any(dim=1))
  return (contact.data.found > 0) & torch.stack(edge, dim=1)


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
    # Distance along the command and time since reset, and both at each foot's
    # last record (`step_average`).
    self.odometer = torch.zeros(env.num_envs, device=env.device)
    self.clock = torch.zeros(env.num_envs, device=env.device)
    self.record_odometer = torch.zeros(env.num_envs, num_feet, device=env.device)
    self.record_clock = torch.zeros(env.num_envs, num_feet, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.record_z[env_ids] = -torch.inf
    self.odometer[env_ids] = 0.0
    self.clock[env_ids] = 0.0
    self.record_odometer[env_ids] = 0.0
    self.record_clock[env_ids] = 0.0

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
    max_speed_ratio: float | None = None,
    step_average: bool = False,
    min_gain: float = 0.0,
  ) -> torch.Tensor:
    """A foothold counts when the contact force is mostly vertical.

    That excludes a foot pushed against a riser face. The first foothold after
    reset only initializes the record. Requires a world-frame force. A new record
    must beat the old one by at least `min_gain`.

    With `max_speed_ratio`, a new record set while the base moves along the
    command faster than that times the commanded speed earns nothing (it is used
    up): paid per metre, the descent term sent As2 and B2 down map3's flights at
    1.24x and 1.21x the command (2026-10-21), as the riser term did for the
    wheeled robots (see wheel_riser_lift). Judged by the speed at the instant a
    foot lands, which lurches forward on every step down, a cap of 1.1 then
    slowed Go2 and As2 to 0.66x (2026-10-22). With `step_average` the speed is
    instead the mean along the command since that foot's previous record, about
    one stride on stairs; with a `min_gain` of a few cm, a foot settling into its
    tread does not restart that window.
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
    known = torch.isfinite(self.record_z)
    new = stance & known & (foot_z - self.record_z >= min_gain)
    first = stance & ~known
    gain = torch.where(new, torch.clamp(foot_z - self.record_z, min=0.0, max=max_gain), 0.0)
    if max_speed_ratio is not None:
      command = env.command_manager.get_command(command_name)
      assert command is not None
      cmd = command[:, :2]
      speed = torch.norm(cmd, dim=1)
      along = (asset.data.root_link_lin_vel_b[:, :2] * cmd).sum(dim=1) / speed.clamp(min=1e-6)
      if step_average:
        self.odometer += along * env.step_dt
        self.clock += env.step_dt
        elapsed = torch.clamp(self.clock[:, None] - self.record_clock, min=env.step_dt)
        mean = (self.odometer[:, None] - self.record_odometer) / elapsed
        over = mean > max_speed_ratio * speed[:, None]
      else:
        over = (along > max_speed_ratio * speed)[:, None]
      gain = torch.where(over, 0.0, gain)
    held = new | first
    self.record_z = torch.where(held, foot_z, self.record_z)
    self.record_odometer = torch.where(held, self.odometer[:, None], self.record_odometer)
    self.record_clock = torch.where(held, self.clock[:, None], self.record_clock)
    active = ~_zero_command_mask(env, command_name, command_threshold)
    return torch.sum(gain, dim=1) * active.float() / env.step_dt


class variable_posture_terrain_aware(variable_posture):
  """variable_posture that uses the walking tolerance when standing on uneven ground.

  Standing on stairs needs different joint angles per leg, so the tight
  standing tolerance only applies on even ground. With `drive_max_lin_vel_y`
  (wheeled), moving commands the wheels drive (|vy| and |wz| within the limits)
  on even ground keep the standing tolerance too: the legs do not step there.
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
    rest_hip: float | None = None,
    calm_std: float = 0.1,
    drive_max_lin_vel_y: float | None = None,
    drive_max_ang_vel_z: float = 0.1,
    hold_term: str | None = None,
    drive_skip_march: bool = False,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    total_speed = _command_activity(env, command_name)
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
    if drive_max_lin_vel_y is not None:
      drive = (
        (total_speed >= walking_threshold)
        & (command[:, 1].abs() <= drive_max_lin_vel_y)
        & (command[:, 2].abs() <= drive_max_ang_vel_z)
        & ~uneven
      )
      if drive_skip_march:
        # A march in place is a zero command the wheels do not drive (drive_still_even's rule): walking tolerance.
        march = getattr(env.command_manager.get_term(command_name), "march", None)
        if march is not None:
          drive = drive & ~march
      std = torch.where(drive.unsqueeze(1), self.std_standing.expand_as(std), std)

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    if rest_hip is not None:
      # Standing on even ground the hips aim at the rest stance, the more the
      # calmer the base (a robot still coasting keeps its wider stance).
      lin_vel_xy = asset.data.root_link_lin_vel_b[:, :2]
      ang_vel_z = asset.data.root_link_ang_vel_b[:, 2]
      calm = torch.exp(-(torch.sum(torch.square(lin_vel_xy), dim=1) + torch.square(ang_vel_z)) / calm_std)
      desired_joint_pos = _rest_pose(asset, asset_cfg.joint_ids, desired_joint_pos, rest_hip, calm * standing_mask)
    if hold_term is not None:
      # Once the standing hold has its anchor the pose to keep is the one the robot settled in (2026-11-01):
      # on braked wheels a return to the default pose moves the base, and the B2W v13wn teacher crept 2-3 cm per
      # 4 s after stops doing so (frozen actions 0.1-0.3). The settling (up to the hold's push window) and a
      # march still bring the default pose back.
      locked = _hold_locked(env, hold_term)
      if getattr(self, "lock_pos", None) is None:
        self.lock_pos = current_joint_pos.clone()
      self.lock_pos = torch.where(locked.unsqueeze(1), self.lock_pos, current_joint_pos)
      desired_joint_pos = torch.where(locked.unsqueeze(1), self.lock_pos, desired_joint_pos)
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


class hip_torque_sq:
  """Squared hip abduction (HAA) torques relative to `torque_ref` (Nm), mean over the
  hips: motor heating goes with the current squared, so this sums the hips' heat.

  Standing or driving on wheels, the feet sit outboard of the hip axes and the body
  weight loads the hip motors without a break; Unitree's Go2-W controller drove
  them to a thermal shutdown at ~84 deg C after 34 min at 1 m/s (arXiv 2606.21387).
  `actuator_names` are the hip actuators.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_name", "robot")]
    ids, _ = asset.find_actuators(cfg.params["actuator_names"])
    self.ids = torch.tensor(ids, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    actuator_names: str | tuple[str, ...],
    torque_ref: float,
    asset_name: str = "robot",
    command_name: str | None = None,
    standing_scale: float = 1.0,
    command_threshold: float = 0.1,
    uneven_scale: float = 1.0,
    sensor_name: str | None = None,
  ) -> torch.Tensor:
    """With `command_name`, the cost is `standing_scale` times higher while the
    command is zero (the hip motors overheat mostly while the robot stands); with
    `sensor_name` (a height scan), `uneven_scale` times
    higher again while standing on uneven ground: stopped mid-flight the sixth-round
    policies loaded their hips about twice as much as standing on flat ground (B2
    49-51 Nm front)."""
    del actuator_names
    asset: Entity = env.scene[asset_name]
    tau = asset.data.actuator_force[:, self.ids]
    cost = torch.mean(torch.square(tau / torque_ref), dim=1)
    if command_name is not None:
      standing = _zero_command_mask(env, command_name, command_threshold)
      scale = torch.where(standing, standing_scale, 1.0)
      if sensor_name is not None and uneven_scale != 1.0:
        scale = torch.where(standing & uneven_terrain_mask(env, sensor_name), scale * uneven_scale, scale)
      cost = cost * scale
    return cost


def hip_splay(
  env: ManagerBasedRlEnv,
  command_name: str,
  hip_cfg: SceneEntityCfg,
  deadband: float = 0.05,
  max_lin_vel_y: float = 0.1,
  max_ang_vel_z: float = 0.5,
  inward_deadband: float | None = None,
  outward_standing_only: bool = False,
) -> torch.Tensor:
  """Squared outward hip abduction beyond `deadband` (rad) relative to the default
  stance, summed over the hips, while the command needs no sideways step
  (|vy| <= max_lin_vel_y, |wz| <= max_ang_vel_z). With `inward_deadband`, inward
  adduction beyond it costs the same. With `outward_standing_only`, the outward
  part counts under the zero command only.

  Splayed legs put the feet further outboard of the hip axes, which raises the hip
  load; on wheels the legs splay slowly while rolling on (the Go2W's rear hips
  +1.4 deg per minute at 1 m/s in simulation, 2026-10-22).

  Tucked legs lower the hip load too, down to none with the feet under the hip
  axes, but past that the body weight folds them further in: driving backward at
  1 m/s, the Go2W's fourth-round policy (penalized for hip torque and outward splay
  only) aimed its front hips 0.3 rad inside the default stance, the load folded
  them to 0.55 rad inside it, and 22-26 of 32 rolled over within 8 s (third round
  10-15, the deployed model none; 2026-10-22).
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  outward = _hip_outward(env, hip_cfg)
  over = torch.clamp(outward - deadband, min=0.0)
  if outward_standing_only:
    over = over * _zero_command_mask(env, command_name, 0.1).float().unsqueeze(1)
  if inward_deadband is not None:
    over = over + torch.clamp(-outward - inward_deadband, min=0.0)
  active = (command[:, 1].abs() <= max_lin_vel_y) & (command[:, 2].abs() <= max_ang_vel_z)
  return torch.sum(torch.square(over), dim=1) * active.float()


def body_angular_velocity_uneven(
  env: ManagerBasedRlEnv,
  scan_sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared body roll / pitch rate on uneven ground only (2026-11-05, the user: "在走台阶的时候尽量保持机身的稳定"):
  body_angular_velocity_penalty, masked by uneven_terrain_mask (the robot's scan circle holds a riser). Pilot
  stairstab adds it on top of the everywhere-term, so stairs pay about three times as much for a rocking body."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.sum(torch.square(ang_vel[:, :2]), dim=1) * uneven_terrain_mask(env, scan_sensor_name).float()


class standing_pose_load(standing_position_hold):
  """Penalize the legs standing off the default stance once the robot has come to rest while commanded to stand,
  more with a payload (go2-depth 2026-11-10, the user: "训练端：负载静止站姿项，这个添加到训练"): every leg joint's
  distance from its default beyond `deadband` (rad, clamped to `max_dev`), squared and summed, times
  1 + `load_gain` * payload / `load_ref` (env.payload_kg from mdp.base_payload_point; light draws below 0 kg
  count as 0), while the standing anchor is locked (standing_position_hold's rule with the same params: settled,
  or the push window over). After a hard stop with 12 kg the delivered Go2W blind_v3 stood 0.18-0.19 rad off its
  default (FL thigh) in tinynav-sim for 8 s, unloaded 0.07-0.09; the deploy's posture reset marches at 0.15."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    moving_speed: float = 0.1,
    moving_yaw_rate: float = 0.2,
    push_window: float | None = None,
    settle: bool = False,
    smooth_s: float | None = None,
    settle_speed: float = 0.03,
    settle_yaw_rate: float = 0.1,
    deadband: float = 0.05,
    max_dev: float = 0.4,
    load_ref: float = 0.0,
    load_gain: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    self._follow(
      env, command_name, command_threshold, moving_speed, moving_yaw_rate, False, push_window, settle,
      smooth_s, settle_speed, settle_yaw_rate, asset_cfg,
    )
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    excess = ((q - q0).abs() - deadband).clamp(min=0.0, max=max_dev)
    cost = (excess**2).sum(dim=1)
    kg = getattr(env, "payload_kg", None)
    if kg is not None and load_ref > 0.0:
      cost = cost * (1.0 + load_gain * (kg / load_ref).clamp(min=0.0))
    return self.locked.float() * cost
