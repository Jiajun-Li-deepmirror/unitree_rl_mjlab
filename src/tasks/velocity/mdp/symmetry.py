"""Left-right mirror of the actor observation and the action, for rsl_rl's symmetry loss.

The mirror reflects through the base's xz-plane (y -> -y). Left and right legs
swap (FL <-> FR, RL <-> RR); hip joints (axis +x) change sign while thigh, calf
and wheel joints (axis +y) keep it. The base angular velocity, a pseudo-vector,
maps to (-wx, wy, -wz), gravity to (gx, -gy, gz) and the command (vx, vy, wz) to
(vx, -vy, -wz). Joint observations and actions are relative to the default pose,
whose own mirror must be itself (checked), as must the action scales.

Only the actor group is mirrored: the function serves the mirror loss
(symmetry_cfg with use_mirror_loss=True, use_data_augmentation=False), for which
rsl_rl feeds the mirrored observations to the actor only. Other groups are
repeated unchanged, so it must not be used for data augmentation, which would
train the critic on them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from tensordict import TensorDict

  from mjlab.envs import ManagerBasedRlEnv

_VECTOR_SIGNS = {
  "base_ang_vel": (-1.0, 1.0, -1.0),
  "projected_gravity": (1.0, -1.0, 1.0),
  "command": (1.0, -1.0, -1.0),
}
_SIDE_SWAP = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}


def _mirror_name(name: str) -> str:
  if name[:2] not in _SIDE_SWAP or name[2:3] != "_":
    raise ValueError(f"Cannot mirror joint {name!r}: expected an FL/FR/RL/RR_ prefix.")
  return _SIDE_SWAP[name[:2]] + name[2:]


def _joint_map(names: list[str]) -> tuple[list[int], list[float]]:
  """Mirror of a vector over joints: entry i takes sign[i] * entry perm[i]."""
  index = {n: i for i, n in enumerate(names)}
  perm = [index[_mirror_name(n)] for n in names]
  sign = [-1.0 if n.endswith("_hip_joint") else 1.0 for n in names]
  return perm, sign


def _check_invariant(values: torch.Tensor, perm: list[int], sign: list[float], what: str):
  mirrored = values[perm] * torch.tensor(sign, device=values.device)
  if not torch.allclose(values, mirrored, atol=1e-6):
    raise ValueError(f"{what} is not left-right symmetric: {values.tolist()}")


class _Mirror:
  """Index permutation and signs of a flat vector: y[:, i] = sign[i] * x[:, perm[i]]."""

  def __init__(self, perm: list[int], sign: list[float], device: str):
    self.perm = torch.tensor(perm, dtype=torch.long, device=device)
    self.sign = torch.tensor(sign, dtype=torch.float32, device=device)

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return x[:, self.perm] * self.sign


def _action_map(env: ManagerBasedRlEnv) -> tuple[list[int], list[float]]:
  names: list[str] = []
  for term_name in env.action_manager.active_terms:
    names += list(env.action_manager.get_term(term_name).target_names)
  perm, sign = _joint_map(names)
  start = 0
  for term_name in env.action_manager.active_terms:
    term = env.action_manager.get_term(term_name)
    n = len(term.target_names)
    # Within one term the mirror stays inside it (legs to legs, wheels to wheels).
    local_perm = [p - start for p in perm[start : start + n]]
    if min(local_perm) < 0 or max(local_perm) >= n:
      raise ValueError(f"Action term {term_name!r} does not mirror onto itself.")
    local_sign = sign[start : start + n]
    for what, value in (("scale", term.scale), ("offset", term.offset)):
      if isinstance(value, torch.Tensor):
        _check_invariant(value[0], local_perm, local_sign, f"{term_name} action {what}")
    start += n
  return perm, sign


def _build(env: ManagerBasedRlEnv) -> tuple[_Mirror, _Mirror]:
  robot = env.scene["robot"]
  joint_names = list(robot.joint_names)
  om = env.observation_manager
  act_perm, act_sign = _action_map(env)
  perm: list[int] = []
  sign: list[float] = []
  offset = 0
  terms = zip(
    om.active_terms["actor"],
    om._group_obs_term_cfgs["actor"],
    om.group_obs_term_dim["actor"],
    strict=True,
  )
  for name, term_cfg, dims in terms:
    if term_cfg.scale is not None and not isinstance(term_cfg.scale, (int, float)):
      raise ValueError(f"Actor term {name!r} has a per-element scale.")
    if name in _VECTOR_SIGNS:
      p, s = [0, 1, 2], list(_VECTOR_SIGNS[name])
    elif name in ("joint_pos", "joint_vel"):
      ids = term_cfg.params["asset_cfg"].joint_ids
      ids = list(range(len(joint_names))) if isinstance(ids, slice) else list(ids)
      p, s = _joint_map([joint_names[i] for i in ids])
      if name == "joint_pos":
        default = robot.data.default_joint_pos[0, ids]
        _check_invariant(default, p, s, "Default joint position")
    elif name == "actions":
      p, s = act_perm, act_sign
    else:
      raise ValueError(f"No mirror for actor observation term {name!r}.")
    size = int(math.prod(dims))
    frames = max(term_cfg.history_length, 1)
    if size != frames * len(p):
      raise ValueError(f"Actor term {name!r}: size {size} != {frames} x {len(p)}.")
    # Flattened history: frames one after another, each laid out like one frame.
    for f in range(frames):
      perm += [offset + f * len(p) + i for i in p]
      sign += s
    offset += size
  return _Mirror(perm, sign, env.device), _Mirror(act_perm, act_sign, env.device)


def mirror_left_right(
  env, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """rsl_rl data_augmentation_func: the originals followed by their mirrors."""
  base = env.unwrapped
  maps = getattr(base, "_left_right_mirror", None)
  if maps is None:
    maps = _build(base)
    base._left_right_mirror = maps
  obs_mirror, action_mirror = maps
  obs_out = actions_out = None
  if obs is not None:
    mirrored = obs.clone()
    mirrored["actor"] = obs_mirror(obs["actor"])
    obs_out = torch.cat([obs, mirrored], dim=0)
  if actions is not None:
    actions_out = torch.cat([actions, action_mirror(actions)], dim=0)
  return obs_out, actions_out
