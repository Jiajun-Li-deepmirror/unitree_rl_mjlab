"""Left-right mirror of the unified teacher's actor (go2-depth 2026-11-11): mdp.symmetry's (unchanged; same rules,
built from its helpers) plus the privileged terms the teacher's actor sees:

  priv_base_lin_vel   (vx, vy, vz) in the base frame -> (vx, -vy, vz)
  priv_foot_height    per foot (FOOT_SITE_NAMES order) -> the mirror foot's
  priv_foot_air_time  per foot (the contact sensor's order) -> the mirror foot's
  priv_foot_contact   per foot (the contact sensor's order) -> the mirror foot's
  priv_dr_params      (CoM x, y, z, payload, friction) -> (x, -y, z, payload, friction)

(The feet's contact forces stay critic-only: they are world-frame vectors, whose mirror through the base's xz-plane
depends on the yaw.)
"""

from __future__ import annotations

import math

import torch

from src.tasks.velocity.mdp import symmetry as base

FOOT_TERMS_SITES = ("priv_foot_height",)
FOOT_TERMS_SENSOR = ("priv_foot_air_time", "priv_foot_contact")


def _feet_map(names: list[str]) -> tuple[list[int], list[float]]:
  legs = [n[:2] for n in names]
  return [legs.index(base._SIDE_SWAP[leg]) for leg in legs], [1.0] * len(legs)


def _term_map(env, name, term_cfg, joint_names, robot, act_perm, act_sign):
  if name in base._VECTOR_SIGNS:
    return [0, 1, 2], list(base._VECTOR_SIGNS[name])
  if name in ("joint_pos", "joint_vel"):
    asset_cfg = term_cfg.params.get("asset_cfg")
    ids = asset_cfg.joint_ids if asset_cfg is not None else slice(None)
    ids = list(range(len(joint_names))) if isinstance(ids, slice) else list(ids)
    p, s = base._joint_map([joint_names[i] for i in ids])
    if name == "joint_pos":
      base._check_invariant(robot.data.default_joint_pos[0, ids], p, s, "Default joint position")
    return p, s
  if name == "actions":
    return act_perm, act_sign
  if name == "phase":
    base._check_gait_phase(env)
    return [0, 1], [-1.0, -1.0]
  if name == "heading_error":
    return [0], [-1.0]
  if name == "stand_time":
    return [0], [1.0]
  if "scan" in name and "sensor_name" in term_cfg.params:
    return base._scan_map(env, term_cfg.params["sensor_name"])
  if name == "priv_base_lin_vel":
    return [0, 1, 2], [1.0, -1.0, 1.0]
  if name == "priv_dr_params":
    return [0, 1, 2, 3, 4], [1.0, -1.0, 1.0, 1.0, 1.0]
  if name in FOOT_TERMS_SITES:
    return _feet_map(list(term_cfg.params["asset_cfg"].site_names))
  if name in FOOT_TERMS_SENSOR:
    sensor = env.scene[term_cfg.params["sensor_name"]]
    return _feet_map([s.primary_name for s in sensor._slots if s.field_name == "found"])
  raise ValueError(f"No mirror for actor observation term {name!r}.")


def _build(env):
  robot = env.scene["robot"]
  joint_names = list(robot.joint_names)
  om = env.observation_manager
  act_perm, act_sign = base._action_map(env)
  perm: list[int] = []
  sign: list[float] = []
  offset = 0
  for name, term_cfg, dims in zip(om.active_terms["actor"], om._group_obs_term_cfgs["actor"],
                                  om.group_obs_term_dim["actor"], strict=True):
    scale = term_cfg.scale
    if scale is not None and not isinstance(scale, (int, float)):
      scale = torch.as_tensor(scale)
      if not torch.all(scale == scale.flatten()[0]):
        raise ValueError(f"Actor term {name!r} has a per-element scale.")
    p, s = _term_map(env, name, term_cfg, joint_names, robot, act_perm, act_sign)
    size = int(math.prod(dims))
    frames = max(term_cfg.history_length, 1)
    if size != frames * len(p):
      raise ValueError(f"Actor term {name!r}: size {size} != {frames} x {len(p)}.")
    for f in range(frames):
      perm += [offset + f * len(p) + i for i in p]
      sign += s
    offset += size
  return base._Mirror(perm, sign, env.device), base._Mirror(act_perm, act_sign, env.device)


def mirror_left_right(env, obs=None, actions=None):
  """rsl_rl data_augmentation_func (mirror loss only, as mdp.symmetry.mirror_left_right): the originals followed by
  their mirrors."""
  unwrapped = env.unwrapped
  maps = getattr(unwrapped, "_u_left_right_mirror", None)
  if maps is None:
    maps = _build(unwrapped)
    unwrapped._u_left_right_mirror = maps
  obs_mirror, action_mirror = maps
  obs_out = actions_out = None
  if obs is not None:
    mirrored = obs.clone()
    mirrored["actor"] = obs_mirror(obs["actor"])
    obs_out = torch.cat([obs, mirrored], dim=0)
  if actions is not None:
    actions_out = torch.cat([actions, action_mirror(actions)], dim=0)
  return obs_out, actions_out
