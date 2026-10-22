from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from .terrain_utils import spawned_on_sub_terrain

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_root_state_terrain_aware(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  stair_terrain_names: tuple[str, ...],
  stair_yaw_noise: float = math.radians(15.0),
  velocity_range: dict[str, tuple[float, float]] | None = None,
  narrow_terrain_names: tuple[str, ...] = (),
  narrow_xy_range: float = 0.1,
  narrow_yaw_noise: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Like reset_root_state_uniform, but faces stair envs toward a staircase side.

  Pyramid stairs are axis-aligned, so stair envs get yaw = k * 90 deg plus
  noise; other envs use `pose_range["yaw"]`. Envs on `narrow_terrain_names`
  (walled corridors) get x/y noise of only +-`narrow_xy_range`, and with
  `narrow_yaw_noise` that yaw noise instead of `stair_yaw_noise` (lined up with
  the flight, as an operator would start a robot that must walk it uncorrected).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  keys = ["x", "y", "z", "roll", "pitch", "yaw"]
  ranges = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in keys], device=env.device)
  pose_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )

  on_stairs = spawned_on_sub_terrain(env, stair_terrain_names)[env_ids.long()]
  side = torch.randint(0, 4, (len(env_ids),), device=env.device).float()
  stair_yaw = side * (math.pi / 2) + sample_uniform(
    -stair_yaw_noise, stair_yaw_noise, (len(env_ids),), device=env.device
  )
  pose_samples[:, 5] = torch.where(on_stairs, stair_yaw, pose_samples[:, 5])
  if narrow_terrain_names:
    narrow = spawned_on_sub_terrain(env, narrow_terrain_names)[env_ids.long()]
    pose_samples[:, 0:2] = torch.where(
      narrow.unsqueeze(1),
      pose_samples[:, 0:2].clamp(-narrow_xy_range, narrow_xy_range),
      pose_samples[:, 0:2],
    )
    if narrow_yaw_noise is not None:
      # The same draw, scaled: no extra random numbers.
      base = side * (math.pi / 2)
      narrow_yaw = base + (stair_yaw - base) * (narrow_yaw_noise / stair_yaw_noise)
      pose_samples[:, 5] = torch.where(narrow, narrow_yaw, pose_samples[:, 5])

  positions = root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  velocity_range = velocity_range or {}
  vel_ranges = torch.tensor(
    [velocity_range.get(k, (0.0, 0.0)) for k in keys], device=env.device
  )
  vel_samples = sample_uniform(
    vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  velocities = root_states[:, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_hip_splay(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  prob: float,
  splay_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Open the hips of a `prob` share of the reset robots outward by the same angle
  from `splay_range` (rad; asset_cfg holds the hip joints). Run after the joint
  reset. Rolling on, wheeled legs splay out over minutes (2026-10-22), far longer
  than an episode: this starts robots splayed so the policy learns to return.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  names = [asset.joint_names[i] for i in ids]
  side = torch.tensor([1.0 if n[1] == "L" else -1.0 for n in names], device=env.device)
  n = len(env_ids)
  pick = (torch.rand(n, device=env.device) < prob).float().unsqueeze(1)
  splay = sample_uniform(*splay_range, (n, 1), env.device) * pick * side
  joint_ids = torch.tensor(ids, device=env.device) if isinstance(ids, list) else ids
  joint_pos = asset.data.joint_pos[env_ids][:, ids] + splay
  joint_vel = asset.data.joint_vel[env_ids][:, ids]
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids)
