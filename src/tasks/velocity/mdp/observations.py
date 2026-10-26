from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def gait_clock(
    env: ManagerBasedRlEnv,
    period: float,
    command_name: str,
    ref_speed: float | None = None,
    foot_xy: tuple[tuple[float, float], ...] | None = None,
) -> torch.Tensor:
    """Phase in [0, 1) of the gait clock, restarting with the episode.

    Without `ref_speed` it is the episode time over `period`. With it, the period
    shortens above that commanded planar speed to period * ref_speed / |v_cmd,xy| (user
    2026-10-23: quicker steps at speed; standing, marching, turning in place and slow
    commands keep `period`), and the phase advances by step_dt over the current
    period once per env step, as the deploy side's gait_phase (ref_speed in its
    params): the observation and the reward share one clock per (period, ref_speed,
    foot_xy). With `foot_xy` (the feet's nominal (x, y) in the base frame) the speed
    is the fastest foot's under the command, |(vx - wz y, vy + wz x)|: on a 1 m
    radius arc at 0.8 m/s the outer feet move 13-16% faster than the base, and the
    legged v4 / v6 teachers stepped there at 1.85-2.0 per second against the clock's
    1.67 (match 67-78%), just the cadence that keeps the outer feet's stride.
    """
    if ref_speed is None:
        return (env.episode_length_buf * env.step_dt) % period / period
    clocks = env.__dict__.setdefault("_gait_clocks", {})
    key = (period, ref_speed, None if foot_xy is None else tuple(tuple(f) for f in foot_xy))
    clock = clocks.get(key)
    length = env.episode_length_buf
    if clock is None:
        clock = clocks[key] = {
            "phase": torch.zeros(env.num_envs, device=env.device), "length": length.clone()
        }
    command = env.command_manager.get_command(command_name)
    if foot_xy is None:
        speed = torch.linalg.norm(command[:, :2], dim=1)
    else:
        feet = torch.tensor(foot_xy, device=env.device)  # [F, 2]
        vx = command[:, 0:1] - command[:, 2:3] * feet[:, 1]
        vy = command[:, 1:2] + command[:, 2:3] * feet[:, 0]
        speed = torch.sqrt(vx.square() + vy.square()).amax(dim=1)
    current = period * torch.clamp(ref_speed / speed.clamp(min=1e-6), max=1.0)
    steps = (length - clock["length"]).float()
    restarted = length < clock["length"]
    advanced = (clock["phase"] + steps.clamp(min=0.0) * env.step_dt / current) % 1.0
    clock["phase"] = torch.where(restarted, (length * env.step_dt / current) % 1.0, advanced)
    clock["length"] = length.clone()
    return clock["phase"]


def phase(
    env: ManagerBasedRlEnv,
    period: float,
    command_name: str,
    ref_speed: float | None = None,
    drivable_max_lin_vel_y: float | None = None,
    drivable_max_ang_vel_z: float | None = None,
    foot_xy: tuple[tuple[float, float], ...] | None = None,
) -> torch.Tensor:
    """The gait clock as (sin, cos), zero while standing (not marching). With the
    drivable limits (wheeled robots) also zero under a command the wheels drive
    (at most that sideways speed and yaw rate): the clock then runs only where the
    robot steps (2026-10-23: with it running while driving straight, the wheeled
    teachers trotted to it at every speed, all four wheels down 2-27% of the time)."""
    global_phase = gait_clock(env, period, command_name, ref_speed, foot_xy)
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    command = env.command_manager.get_command(command_name)
    stand_mask = torch.linalg.norm(command, dim=1) < 0.1
    if drivable_max_lin_vel_y is not None and drivable_max_ang_vel_z is not None:
        stand_mask = stand_mask | (
            (command[:, 1].abs() <= drivable_max_lin_vel_y) & (command[:, 2].abs() <= drivable_max_ang_vel_z)
        )
    # The clock also runs for a robot told to march in place (zero command; see
    # TerrainAwareVelocityCommandCfg.march_prob): the deploy side runs it to step.
    march = getattr(env.command_manager.get_term(command_name), "march", None)
    if march is not None:
        stand_mask = stand_mask & ~march
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase



def foot_height_rel_base(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Foot site heights relative to the base, so the value is meaningful on stairs."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  return foot_z - asset.data.root_link_pos_w[:, 2:3]  # (num_envs, num_sites)


class heading_error:
  """Heading the command asks for minus the base heading (rad, wrapped, + = turn left).

  The target heading starts at the base heading after a reset and integrates the
  commanded yaw rate every control step, so with wz = 0 it holds the heading the
  robot had: blind policies see their yaw rate but not how far they have turned
  in total, and drifted ~11 deg per 10 s walking straight (2026-10-20). The target
  stays within `max_error` of the base heading (anti-windup when the robot cannot
  follow, e.g. pushed or against a wall). On the robot the same is computed from
  the IMU yaw and the deadzoned stick command (deploy: observation heading_error).
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.target = torch.zeros(env.num_envs, device=env.device)
    # Set from the base heading on the next two control steps: right after a reset
    # the derived root pose may still be the previous episode's.
    self.settle = torch.full((env.num_envs,), 2, dtype=torch.long, device=env.device)
    self.last_step = -1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.settle[env_ids] = 2

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    max_error: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    heading = asset.data.heading_w
    command = env.command_manager.get_command(command_name)
    assert command is not None
    # Integrate once per control step, however often the observation is computed.
    advance = env.common_step_counter != self.last_step
    self.last_step = env.common_step_counter
    settling = self.settle > 0
    step = command[:, 2] * env.step_dt if advance else torch.zeros_like(heading)
    self.target = torch.where(settling, heading, self.target + step)
    if advance:
      self.settle = (self.settle - 1).clamp(min=0)
    error = torch.atan2(torch.sin(self.target - heading), torch.cos(self.target - heading))
    error = error.clamp(-max_error, max_error)
    self.target = heading + error
    return error.unsqueeze(1)


def _reward_term(env: ManagerBasedRlEnv, name: str):
  """The instance of class-based reward term `name`, or None (no such term, or the
  observation manager is sizing its terms before the reward manager exists)."""
  manager = getattr(env, "reward_manager", None)
  if manager is None or name not in manager.active_terms:
    return None
  return manager.get_term_cfg(name).func


def hold_offsets(
  env: ManagerBasedRlEnv, term_names: tuple[str, ...], asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Critic only: per position hold (standing_position_hold terms), the base's offset
  from the hold's anchor in the base yaw frame (m) and its heading change (rad); zero
  while the anchor follows the base. The hold costs depend on where the anchor is,
  which no other observation shows (2026-10-23: the march drift of the phased
  teachers grew 38 -> 46 cm while the hold charged it in full)."""
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.root_link_pos_w[:, :2]
  yaw = asset.data.heading_w
  c, s = torch.cos(yaw), torch.sin(yaw)
  out = []
  for name in term_names:
    term = _reward_term(env, name)
    if term is None:
      out.append(torch.zeros(env.num_envs, 3, device=env.device))
      continue
    d = pos - term.anchor_pos
    turn = torch.atan2(torch.sin(yaw - term.anchor_yaw), torch.cos(yaw - term.anchor_yaw))
    v = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], turn], dim=1)
    out.append(v * (~term.needs_anchor).float().unsqueeze(1))
  return torch.cat(out, dim=1)


def drift_offsets(
  env: ManagerBasedRlEnv, term_names: tuple[str, ...], asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Critic only: per straight-line cost (straight_drift terms) while it runs, the
  signed sideways offset from the commanded line (m), the distance along it (m) and
  the heading change since the command started (rad); zero otherwise. Like
  hold_offsets, the state those costs depend on."""
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.root_link_pos_w[:, :2]
  heading = asset.data.heading_w
  out = []
  for name in term_names:
    term = _reward_term(env, name)
    if term is None:
      out.append(torch.zeros(env.num_envs, 3, device=env.device))
      continue
    rel = pos - term.ref_pos
    side = term.ref_dir[:, 0] * rel[:, 1] - term.ref_dir[:, 1] * rel[:, 0]
    along = (term.ref_dir * rel).sum(dim=1)
    turn = torch.atan2(torch.sin(heading - term.ref_heading), torch.cos(heading - term.ref_heading))
    out.append(torch.stack([side, along, turn], dim=1) * term.active.float().unsqueeze(1))
  return torch.cat(out, dim=1)
