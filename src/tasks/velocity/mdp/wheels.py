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
from mjlab.utils.lab_api.math import wrap_to_pi

from .rewards import (
  _DEFAULT_ASSET_CFG,
  _standing_scale,
  _zero_command_mask,
  feet_gait,
  uneven_terrain_mask,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg
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
  tracking_std: float | None = None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  **gait_params,
) -> torch.Tensor:
  """`feet_gait` (trot phase), only in step mode.

  With `tracking_std`, scaled by how well the base follows the planar command
  and yaw rate (exp(-err^2/std^2)), so a heavier weight can't be farmed by
  trotting in place.
  """
  command_name = gait_params["command_name"]
  step = step_mode_mask(env, command_name, scan_sensor_name, lateral_threshold)
  reward = feet_gait(env, **gait_params) * step.float()
  if tracking_std is not None:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    err = torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2] - command[:, :2]), dim=1)
    err = err + torch.square(asset.data.root_link_ang_vel_b[:, 2] - command[:, 2])
    reward = reward * torch.exp(-err / tracking_std**2)
  return reward


def wheel_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  wheel_radius: float,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Squared slip speed at the contact point of each wheel on the ground.

  With `command_name`, only while the command is non-zero.

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
  if command_name is None:
    active = torch.ones(env.num_envs, device=env.device)
  else:
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


def wheel_air_drivable(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.1,
  max_lin_vel_y: float = 0.1,
  max_ang_vel_z: float = 0.1,
  scan_sensor_name: str | None = None,
) -> torch.Tensor:
  """Fraction of wheels off the ground while the command can be rolled (a
  penalty; go2-blind's wheeled recipe): moving nearly straight, forward or
  back, and with `scan_sensor_name` on even ground only, so stairs can still
  be stepped. With |wz| <= 0.5 instead of 0.1 the B2-W hardly turned below
  0.3 rad/s.
  """
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  in_air = (contact.data.found <= 0).float()
  command = env.command_manager.get_command(command_name)
  assert command is not None
  drivable = (
    _moving(env, command_name, command_threshold)
    & (command[:, 1].abs() <= max_lin_vel_y)
    & (command[:, 2].abs() <= max_ang_vel_z)
  )
  if scan_sensor_name is not None:
    drivable &= ~uneven_terrain_mask(env, scan_sensor_name)
  return in_air.mean(dim=1) * drivable.float()


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


# Standing still on velocity-controlled wheels (go2-blind's wheeled v5). A
# wheel only damps toward its target, so a standing robot creeps: 2-6 cm per
# 4 s before, which the squared standing terms charged almost nothing.


class standing_position_hold:
  """Drift from where the robot settled while commanded to stand.

  The anchor (base position and heading) follows the base while the command
  is non-zero or the base moves faster than `anchor_speed` or turns faster
  than `anchor_yaw_rate`, so a push moves it and only the slow creep after
  settling costs: min(distance, `max_distance`) + `yaw_weight` * |heading
  change|.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.anchor_xy = torch.zeros(env.num_envs, 2, device=env.device)
    self.anchor_yaw = torch.zeros(env.num_envs, device=env.device)
    self.has_anchor = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.has_anchor[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    anchor_speed: float = 0.1,
    anchor_yaw_rate: float = 0.2,
    max_distance: float = 0.3,
    yaw_weight: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    xy = asset.data.root_link_pos_w[:, :2]
    yaw = asset.data.heading_w
    standing = _zero_command_mask(env, command_name, command_threshold)
    moving = (torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1) > anchor_speed) | (
      asset.data.root_link_ang_vel_b[:, 2].abs() > anchor_yaw_rate
    )
    follow = ~standing | moving | ~self.has_anchor
    self.anchor_xy = torch.where(follow.unsqueeze(1), xy, self.anchor_xy)
    self.anchor_yaw = torch.where(follow, yaw, self.anchor_yaw)
    self.has_anchor[:] = True
    dist = torch.norm(xy - self.anchor_xy, dim=1).clamp(max=max_distance)
    turn = wrap_to_pi(yaw - self.anchor_yaw).abs()
    return (dist + yaw_weight * turn) * standing.float()


def standing_speed(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  yaw_weight: float = 0.3,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Base speed (L1, each part capped at 1) while commanded to stand."""
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1).clamp(max=1.0)
  turn = asset.data.root_link_ang_vel_b[:, 2].abs().clamp(max=1.0)
  return (speed + yaw_weight * turn) * _zero_command_mask(
    env, command_name, command_threshold
  ).float()


def standing_feet_air(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.1,
  calm_std: float = 0.1,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Fraction of wheels off the ground while commanded to stand, calm."""
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  in_air = (contact.data.found <= 0).float().mean(dim=1)
  return in_air * _standing_scale(env, command_name, command_threshold, calm_std, asset_cfg)


def standing_wheel_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str = "wheel_vel",
  command_threshold: float = 0.1,
  calm_std: float = 0.1,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Sum of |wheel velocity targets| (rad/s) the policy asks for while standing.

  At a zero command the policies left opposing targets on the wheels of one
  side, which is what crept. The targets are the policy's, before the
  low-speed brake zeroes them.
  """
  term = env.action_manager.get_term(action_name)
  if hasattr(term, "policy_targets"):
    targets = term.policy_targets
  else:
    targets = term._processed_actions
  return targets.abs().sum(dim=1) * _standing_scale(
    env, command_name, command_threshold, calm_std, asset_cfg
  )


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
