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


def push_by_setting_velocity_marked(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  march_skip_prob: float = 0.0,
  command_name: str = "twist",
) -> None:
  """mjlab's push_by_setting_velocity that also stamps each pushed env with the
  step counter (env._push_step) for terms that excuse a pushed robot
  (standing_position_hold with `push_window`). With `march_skip_prob`, a robot
  marching in place (the command term's march flag) skips each push with that
  probability, so its march hold often charges a whole march undisturbed, as the
  20 s march acceptance measures."""
  from mjlab.envs.mdp.events import push_by_setting_velocity

  if march_skip_prob > 0.0:
    march = getattr(env.command_manager.get_term(command_name), "march", None)
    if march is not None:
      skip = march[env_ids] & (torch.rand(len(env_ids), device=env.device) < march_skip_prob)
      env_ids = env_ids[~skip]
      if len(env_ids) == 0:
        return
  push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
  stamp = getattr(env, "_push_step", None)
  if stamp is None:
    stamp = torch.full((env.num_envs,), -(10**9), dtype=torch.long, device=env.device)
    env._push_step = stamp
  stamp[env_ids] = env.common_step_counter


from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields  # noqa: E402


@requires_model_fields("body_mass", "body_ipos", "body_inertia", recompute=RecomputeLevel.set_const)
def base_payload_point(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  light_range: tuple[float, float],
  heavy_max: float,
  heavy_share: float,
  pos_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
  base_com_range: float = 0.05,
  ramp_steps: int = 0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: str = "uniform",
  corner_share: float = 0.0,
  corner_front: bool = False,
  corner_terrain_names: tuple[str, ...] = (),
) -> None:
  """A payload as a point mass on the base (2026-11-05, the user: payload training up to the Unitree documented
  limits, a wide payload CoM). Per robot: the payload m from `light_range` (kg, the earlier blind DR) or, for
  `heavy_share` of the robots, uniform on [0, heavy_max] (the documented maximum; ramped up over the first
  `ramp_steps` env steps), at a point p in the base frame from `pos_ranges` (x, y, z, m), plus the base's own CoM
  noise (+-`base_com_range` per axis, the old base_com event). The base body gets mass M0 + m, its CoM at the
  combined centre, and its (diagonal) inertia the parallel-axis terms of both parts about it (the base's inertial
  frame taken as aligned with the body frame). Negative m (parts removed) shifts the CoM the same way.
  `distribution` is accepted and ignored (depth_env_cfg's DEPTH_STUDENT_DR_EDGE sets it on payload events; the heavy
  share already reaches the documented limit)."""
  del distribution
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device, dtype=torch.long)
  n = len(env_ids)
  if n == 0:
    return
  b = asset.indexing.root_body_id
  M0 = env.sim.get_default_field("body_mass")[b]
  c0 = env.sim.get_default_field("body_ipos")[b]
  I0 = env.sim.get_default_field("body_inertia")[b]
  dev = env.device
  ramp = 1.0 if ramp_steps <= 0 else min(1.0, float(env.common_step_counter) / float(ramp_steps))
  heavy = torch.rand(n, device=dev) < heavy_share
  m = torch.where(
    heavy,
    torch.rand(n, device=dev) * heavy_max * ramp,
    light_range[0] + torch.rand(n, device=dev) * (light_range[1] - light_range[0]),
  )
  lo = torch.tensor([r[0] for r in pos_ranges], device=dev)
  hi = torch.tensor([r[1] for r in pos_ranges], device=dev)
  p = lo + torch.rand(n, 3, device=dev) * (hi - lo)
  if corner_share > 0.0:
    # `corner_share` of the heavy robots at the acceptance's corner (2026-11-06, B2 depth student fe3 in tinynav-sim:
    # 40 kg at x 0.30, z 0.30 flipped down top_18n twice, 1 of 2 runs each; uniform draws put < 1% of the robots
    # there): 75-100% of the maximum, x at either end of its range, z at the top, y anywhere.
    corner = heavy & (torch.rand(n, device=dev) < corner_share)
    if corner_terrain_names:
      # only robots spawned on these sub-terrains (go2-depth 2026-11-08: the front corner everywhere fixed the Go2W
      # 12 kg spawn falls on flat ground but doubled its unloaded slide-back on 14 cm pit steps)
      corner = corner & spawned_on_sub_terrain(env, corner_terrain_names)[env_ids]
    m = torch.where(corner, heavy_max * ramp * (0.75 + 0.25 * torch.rand(n, device=dev)), m)
    xs = torch.where(torch.rand(n, device=dev) < 0.5, lo[0], hi[0])
    if corner_front:
      # front end only (go2-depth 2026-11-08: every payload acceptance loads x = +max; Go2W fell 1-2 / 256 at the
      # spawn there, nose down, the wheels rolling forward)
      xs = hi[0].expand(n)
    p = torch.where(corner[:, None], torch.stack([xs, p[:, 1], hi[2].expand(n)], dim=1), p)
  cb = c0 + (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * base_com_range   # the base's own CoM, noised
  M = M0 + m
  c = (M0 * cb + m[:, None] * p) / M[:, None]
  d0, dp = cb - c, p - c
  # parallel-axis diagonal: I_xx += m (y^2 + z^2), ...
  par0 = torch.stack([d0[:, 1] ** 2 + d0[:, 2] ** 2, d0[:, 0] ** 2 + d0[:, 2] ** 2, d0[:, 0] ** 2 + d0[:, 1] ** 2], 1)
  parp = torch.stack([dp[:, 1] ** 2 + dp[:, 2] ** 2, dp[:, 0] ** 2 + dp[:, 2] ** 2, dp[:, 0] ** 2 + dp[:, 1] ** 2], 1)
  inertia = I0 + M0 * par0 + m[:, None] * parp
  env.sim.model.body_mass[env_ids, b] = M
  env.sim.model.body_ipos[env_ids, b] = c
  env.sim.model.body_inertia[env_ids, b] = torch.clamp(inertia, min=1e-4)
  # Each robot's payload (kg) for the constraints (2026-11-06, go2-depth: mdp.constraints._back_drift max_payload judges
  # the light robots as the acceptance does).
  if getattr(env, "payload_kg", None) is None:
    env.payload_kg = torch.zeros(env.num_envs, device=dev)
  env.payload_kg[env_ids] = m
