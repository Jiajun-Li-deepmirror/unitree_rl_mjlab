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


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
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
