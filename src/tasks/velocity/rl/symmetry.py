"""Left-right mirror of a teacher's actor observations and actions (go2-blind).

rsl_rl's symmetry extension calls `mirror_left_right(env=..., obs=..., actions=...)`
and expects the batch followed by its mirror image. Only the actor group is
mirrored (the mirror loss compares actor(mirror(obs)) with mirror(actor(obs)));
the other groups are repeated as they are.

Reflection across the base's x-z plane: angular velocity (-wx, wy, -wz),
gravity (gx, -gy, gz), command (vx, -vy, -wz); left and right legs swap, hip
angles change sign (their axes are along x), thigh, calf and wheel angles do
not (along y on both sides); the height scan's points swap y for -y. The trot
clock (sin, cos) maps to (-sin, -cos), half a period on (checked against the
foot_gait offsets), and the heading error, a yaw angle, changes sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict

_SIDE = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
_VECTOR_SIGNS = {
  "base_ang_vel": (-1.0, 1.0, -1.0),
  "projected_gravity": (1.0, -1.0, 1.0),
  "command": (1.0, -1.0, -1.0),
}


@dataclass
class _Mirror:
  obs_index: torch.Tensor
  obs_sign: torch.Tensor
  action_index: torch.Tensor
  action_sign: torch.Tensor


_CACHE: dict[int, _Mirror] = {}


def _joint_map(names: list[str]) -> tuple[list[int], list[float]]:
  index = [names.index(_SIDE[n[:2]] + n[2:]) for n in names]
  sign = [-1.0 if "_hip_" in n else 1.0 for n in names]
  return index, sign


def _scan_map(offsets: torch.Tensor) -> tuple[list[int], list[float]]:
  xy = offsets[:, :2]
  mirrored = xy * torch.tensor((1.0, -1.0), device=xy.device)
  dist = torch.cdist(mirrored, xy)
  index = dist.argmin(dim=1)
  # The grid's points carry float error from its arange (~2e-4 m).
  if dist.min(dim=1).values.max() > 1e-2:
    raise ValueError("The height scan pattern is not left-right symmetric.")
  return index.tolist(), [1.0] * len(index)


def _check_gait_phase(base) -> None:
  """Swapping left and right legs is a half-period shift of the gait clock only
  if each leg's clock offset is its mirror's plus half a period (trot)."""
  rm = base.reward_manager
  if "foot_gait" not in rm.active_terms:
    raise ValueError("Phase observation without a foot_gait reward: cannot check the clock.")
  params = rm._term_cfgs[rm.active_terms.index("foot_gait")].params
  sensor = base.scene[params["sensor_name"]]
  legs = [slot.primary_name[:2] for slot in sensor._slots if slot.field_name == "found"]
  offsets = dict(zip(legs, params["offset"], strict=True))
  for leg, value in offsets.items():
    if abs((offsets[_SIDE[leg]] - value - 0.5) % 1.0) > 1e-6:
      raise ValueError(f"Gait offsets {offsets} are not half a period apart left to right.")


def _action_map(base) -> tuple[list[int], list[float]]:
  names: list[str] = []
  for term_name in base.action_manager.active_terms:
    names.extend(base.action_manager.get_term(term_name)._target_names)
  return _joint_map(names)


def _build(env) -> _Mirror:
  base = env.unwrapped
  robot = base.scene["robot"]
  om = base.observation_manager
  index: list[int] = []
  sign: list[float] = []
  start = 0
  for name, dims, cfg in zip(
    om.active_terms["actor"], om.group_obs_term_dim["actor"], om._group_obs_term_cfgs["actor"]
  ):
    (size,) = dims
    history = max(getattr(cfg, "history_length", 0) or 0, 1)
    if name in _VECTOR_SIGNS:
      term_index, term_sign = [0, 1, 2], list(_VECTOR_SIGNS[name])
    elif name in ("joint_pos", "joint_vel"):
      # Without an asset_cfg the term reads every joint.
      asset_cfg = cfg.params.get("asset_cfg")
      ids = slice(None) if asset_cfg is None else asset_cfg.joint_ids
      joints = list(robot.joint_names[ids]) if isinstance(ids, slice) else [robot.joint_names[i] for i in ids]
      term_index, term_sign = _joint_map(joints)
    elif name == "actions":
      term_index, term_sign = _action_map(base)
    elif name == "height_scan":
      term_index, term_sign = _scan_map(base.scene[cfg.params["sensor_name"]]._local_offsets)
    elif name == "phase":
      _check_gait_phase(base)
      term_index, term_sign = [0, 1], [-1.0, -1.0]
    elif name == "heading_error":
      term_index, term_sign = [0], [-1.0]
    else:
      raise ValueError(f"No mirror rule for actor observation {name!r}.")
    frame = len(term_index)
    if frame * history != size:
      raise ValueError(f"{name}: {size} values, expected {history} x {frame}.")
    for h in range(history):
      index.extend(start + h * frame + i for i in term_index)
      sign.extend(term_sign)
    start += size
  action_index, action_sign = _action_map(base)
  device = base.device
  m = _Mirror(
    torch.tensor(index, device=device),
    torch.tensor(sign, device=device),
    torch.tensor(action_index, device=device),
    torch.tensor(action_sign, device=device),
  )
  for idx, sgn in ((m.obs_index, m.obs_sign), (m.action_index, m.action_sign)):
    if not torch.equal(idx[idx], torch.arange(len(idx), device=device)) or not torch.all(sgn * sgn[idx] == 1):
      raise ValueError("The mirror map is not an involution.")
  default = robot.data.default_joint_pos[0]
  j_index, j_sign = _joint_map(list(robot.joint_names))
  if not torch.allclose(default[j_index] * torch.tensor(j_sign, device=device), default):
    raise ValueError("The default pose is not left-right symmetric.")
  return m


def mirror_left_right(
  env, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
  m = _CACHE.get(id(env))
  if m is None:
    m = _CACHE[id(env)] = _build(env)
  obs_aug = None
  if obs is not None:
    mirrored = obs.clone()
    mirrored["actor"] = obs["actor"][:, m.obs_index] * m.obs_sign
    obs_aug = torch.cat([obs, mirrored], dim=0)
  actions_aug = None
  if actions is not None:
    actions_aug = torch.cat([actions, actions[:, m.action_index] * m.action_sign], dim=0)
  return obs_aug, actions_aug
