"""Acceptance items as constraints (2026-10-30, the user's OK to try constrained RL): a Lagrangian-style multiplier
per item scales the weight of the reward term that serves it, from training-time measurements of the item.

Several acceptance items sat on their bars for the whole day (sideways gait clock 78-81% against 80%, drift, slow
arcs), so whether a checkpoint passed was close to a draw. Here each item has a target inside its bar; while the
measured value misses the target the term's weight grows, once it is met the weight eases back, within
[1, max_scale] times the configured weight.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

from .observations import gait_clock
from .rewards import feet_at_edge, uneven_terrain_mask

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["adaptive_constraints", "measure"]


def _gait_match(
  env: ManagerBasedRlEnv,
  gait_term: str = "foot_gait",
  min_speed: float = 0.1,
  max_speed: float = 1.0,
  min_abs_vy: float = 0.0,
  scan_sensor_name: str = "terrain_scan",
) -> tuple[torch.Tensor, torch.Tensor]:
  """Share of feet whose contact matches the gait clock (as feet_gait and the acceptance's gait_clock.py), over the
  robots on even ground under a moving command with speed in [min_speed, max_speed] and |vy| >= min_abs_vy, not
  marching. Returns (per-robot value, mask)."""
  p = env.reward_manager.get_term_cfg(gait_term).params
  sensor: ContactSensor = env.scene[p["sensor_name"]]
  contact = sensor.data.current_contact_time > 0
  phase = gait_clock(env, p["period"], p["command_name"], p.get("ref_speed"), p.get("foot_xy")).unsqueeze(1)
  offset = torch.as_tensor(p["offset"], device=env.device, dtype=phase.dtype).view(1, -1)
  stance = ((phase + offset) % 1.0) < p["threshold"]
  value = (stance == contact).float().mean(dim=1)
  command = env.command_manager.get_command(p["command_name"])
  speed = torch.norm(command[:, :2], dim=1)
  mask = (speed >= min_speed) & (speed <= max_speed) & (command[:, 1].abs() >= min_abs_vy)
  mask &= ~uneven_terrain_mask(env, scan_sensor_name)
  march = getattr(env.command_manager.get_term(p["command_name"]), "march", None)
  if march is not None:
    mask &= ~march
  return value, mask


def _feet_edge_share(
  env: ManagerBasedRlEnv, edge_term: str = "feet_edge"
) -> tuple[torch.Tensor, torch.Tensor]:
  """Share of the feet in contact that stand at an edge (mdp.feet_at_edge), over the robots on the stairs with a
  foot down."""
  p = env.reward_manager.get_term_cfg(edge_term).params
  contact: ContactSensor = env.scene[p["contact_sensor_name"]]
  down = contact.data.found > 0
  at_edge = feet_at_edge(env, p["contact_sensor_name"], p["edge_sensor_names"], p.get("span", 0.03))
  value = at_edge.float().sum(dim=1) / down.float().sum(dim=1).clamp(min=1.0)
  mask = env.command_manager.get_term(p["command_name"]).is_stair_env & down.any(dim=1)
  return value, mask


def _leg_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "leg_ground_touch",
  command_name: str = "twist",
  force_threshold: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Whether a thigh or calf presses on the ground (> force_threshold N in the sensor's history, as the
  leg_ground_contact reward), over the robots on the stairs. Any touch (`found`) was ~99% of the B2 stair robots
  from light brushes (2026-10-31), while the reward and the acceptance count real contacts."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = torch.norm(sensor.data.force_history, dim=-1)  # [B, N, H]
  value = (force > force_threshold).flatten(1).any(dim=1).float()
  return value, env.command_manager.get_term(command_name).is_stair_env


def _drift_share(
  env: ManagerBasedRlEnv, term: str = "side_drift", min_along: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
  """|Offset from the commanded line| / distance along it (as the drift acceptance), from the straight_drift term
  `term`'s state, over the robots it is following at least `min_along` m into their command."""
  t = env.reward_manager.get_term_cfg(term).func
  rel = env.scene["robot"].data.root_link_pos_w[:, :2] - t.ref_pos
  along = (t.ref_dir * rel).sum(dim=1)
  side = (t.ref_dir[:, 0] * rel[:, 1] - t.ref_dir[:, 1] * rel[:, 0]).abs()
  return side / along.clamp(min=1e-3), t.active & (along >= min_along)


def _back_drift(
  env: ManagerBasedRlEnv,
  term: str = "side_drift",
  command_name: str = "twist",
  min_along: float = 1.0,
  scan_sensor_name: str = "terrain_scan",
  min_abs_vy: float = 0.0,
  settled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Signed sideways drift (2026-11-01, B2W): the offset from the commanded line toward the robot's own back (+) or
  front (-) / distance along it, from the sideways straight_drift term `term`'s state, over the robots on even ground
  it is following at least `min_along` m into their command, walking at least `min_abs_vy` sideways. _drift_share's
  |offset| p90 is set by pushes and noise (B2W 0.106 / 0.121 / 0.106 where the acceptance's back-drift went 3.1 /
  5.6 / 6.7%); this is the bias the acceptance caught.

  `settled` measures from the term's settle_pos (straight_drift.SETTLE_S into the line) on: a sideways command after
  a forward one carries the robot forward at first, which hid the bias (tools/side_calib.py, B2W |vy| >= 0.45: p50
  from the start +0.007 / -0.008 / -0.006 after a moving command, from 1 s in +0.041 / +0.026 / +0.005 over all
  starts, mixed avg 8000 / polish avg 8000 / teacher avg 5000, acceptance 6.7 / 5.4 / 3.1%)."""
  t = env.reward_manager.get_term_cfg(term).func
  ref = t.settle_pos if settled else t.ref_pos
  rel = env.scene["robot"].data.root_link_pos_w[:, :2] - ref
  along = (t.ref_dir * rel).sum(dim=1)
  # Walking left (vy > 0) the line points left, so the robot's back is the line's +90 deg side; walking right, -90.
  cross = t.ref_dir[:, 0] * rel[:, 1] - t.ref_dir[:, 1] * rel[:, 0]
  vy = env.command_manager.get_command(command_name)[:, 1]
  mask = t.active & (along >= min_along) & (vy.abs() >= max(min_abs_vy, 1e-6))
  if settled:
    mask &= t.since.float() * env.step_dt > t.SETTLE_S
  mask &= ~uneven_terrain_mask(env, scan_sensor_name)
  return cross * torch.sign(vy) / along.clamp(min=1e-3), mask


def _arc_speed(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  max_vx: float = 0.3,
  min_wz: float = 0.05,
  scan_sensor_name: str = "terrain_scan",
) -> tuple[torch.Tensor, torch.Tensor]:
  """Body-frame vx / commanded vx on slow arcs (0.1 <= |vx| <= max_vx, vy = 0, |wz| >= min_wz) on even ground."""
  command = env.command_manager.get_command(command_name)
  vx = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
  mask = (command[:, 0].abs() >= 0.1) & (command[:, 0].abs() <= max_vx) & (command[:, 1] == 0.0)
  mask &= (command[:, 2].abs() >= min_wz) & ~uneven_terrain_mask(env, scan_sensor_name)
  return vx / torch.where(mask, command[:, 0], torch.ones_like(vx)), mask


def _wheels_down(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sensor_name: str = "feet_ground_contact",
  min_speed: float = 0.5,
  scan_sensor_name: str = "terrain_scan",
) -> tuple[torch.Tensor, torch.Tensor]:
  """All four wheels down (as gait_fb.py), driving straight (|vy| <= 0.1, |wz| <= 0.03, |vx| >= min_speed) on even
  ground."""
  command = env.command_manager.get_command(command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  value = (sensor.data.found > 0).all(dim=1).float()
  mask = (command[:, 0].abs() >= min_speed) & (command[:, 1].abs() <= 0.1) & (command[:, 2].abs() <= 0.03)
  return value, mask & ~uneven_terrain_mask(env, scan_sensor_name)


def _stair_speed(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  scan_sensor_name: str = "terrain_scan",
  descend: bool = True,
  min_pitch: float = 0.05,
  min_vx: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Body-frame vx / commanded vx on uneven ground under a forward command (vx >= min_vx, |vy| <= 0.1), over the
  robots pitched nose down (`descend`) or nose up by at least `min_pitch` (the projected gravity's x part, ~sin of the
  pitch): the acceptance's descent / climb speed ratio on the map3 flights, in training terms (2026-11-01, B2W)."""
  command = env.command_manager.get_command(command_name)
  asset = env.scene["robot"]
  vx = asset.data.root_link_lin_vel_b[:, 0]
  gx = asset.data.projected_gravity_b[:, 0]
  pitched = gx >= min_pitch if descend else gx <= -min_pitch
  mask = (command[:, 0] >= min_vx) & (command[:, 1].abs() <= 0.1) & pitched
  mask &= uneven_terrain_mask(env, scan_sensor_name)
  return vx / torch.where(mask, command[:, 0], torch.ones_like(vx)), mask


METRICS = {
  "gait_match": _gait_match,
  "feet_edge_share": _feet_edge_share,
  "leg_contact": _leg_contact,
  "drift_share": _drift_share,
  "arc_speed": _arc_speed,
  "wheels_down": _wheels_down,
  "stair_speed": _stair_speed,
  "back_drift": _back_drift,
}


def measure(env: ManagerBasedRlEnv, metric: str, stat: str = "mean", **params) -> float | None:
  """One metric's statistic over the robots it applies to now ("mean" or "pNN", a percentile), or None."""
  value, mask = METRICS[metric](env, **params)
  if not mask.any():
    return None
  v = value[mask]
  return float(v.mean()) if stat == "mean" else float(torch.quantile(v, float(stat[1:]) / 100.0))


class adaptive_constraints:
  """Curriculum term: acceptance items as constraints on reward weights (see the module docstring).

  `constraints` is a list of {"name", "metric" (a key of METRICS), "params" (its keyword arguments), "target",
  "stat" ("mean" or "pNN": a percentile over the robots, default "mean"), "sense" ("min": value >= target, "max":
  value <= target), "term" (the reward term to scale), "lr", "max_scale"}.
  Each call (the manager runs it whenever robots reset, nearly every step) measures every metric over the robots it
  applies to and smooths it (exponential average with a time constant of `ema_steps` calls, a plain mean while it has
  fewer samples); once every `update_steps` env steps each multiplier moves by lr x the shortfall (target - value for
  "min", value - target for "max"), clamped to [0, max_scale - 1], and the term's weight becomes (1 + multiplier) x
  its configured weight. The values, the multipliers and the weights go to the log as Curriculum/<term>/<name>/....

  The first `skip_steps` env steps of a run are not measured and a multiplier moves only once its metric has `warmup`
  samples (2026-10-31): every robot resets at once when a run starts, and that start (legs on the ground, gait out of
  phase) seeded the averages, e.g. legs on the ground 0.99 on the B2 polish (true ~0.002) decaying ~2% per iteration,
  which held leg_ground_contact at 3x for ~250 iterations; the Go2 polish started its sideways clock at 0.70. The
  start is taken again whenever the step counter jumps: a resumed run's env resets once at counter 0 and the runner
  then restores the counter from the checkpoint, so a start taken at that reset skipped nothing (Go2W polish from 9300,
  2026-10-31: all wheels down 0.42 and slow arcs 0.54 at its first iteration, ~0.91 / 0.94 calibrated).
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.base: dict[str, float] = {}
    self.lam: dict[str, float] = {}
    self.value: dict[str, float] = {}
    self.count: dict[str, int] = {}
    self.start: int | None = None
    self.last_step = 0
    self.last_update = 0

  def reset(self, env_ids=None) -> None:
    pass

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    constraints: list[dict],
    ema_steps: float = 1000.0,
    update_steps: int = 24,
    skip_steps: int = 200,
    warmup: int = 240,
  ) -> dict[str, torch.Tensor]:
    del env_ids  # Every robot is measured.
    step = env.common_step_counter
    if self.start is None or not 0 <= step - self.last_step <= ema_steps:
      # First call, or the counter jumped (restored on resume): measure from here.
      self.start = self.last_update = step
      self.count.clear()
      self.value.clear()
    self.last_step = step
    alpha = 1.0 - math.exp(-1.0 / ema_steps)
    for c in constraints:
      name = c["name"]
      if name not in self.base:
        self.base[name] = env.reward_manager.get_term_cfg(c["term"]).weight
        self.lam[name] = 0.0
      if step - self.start < skip_steps:
        continue
      v = measure(env, c["metric"], c.get("stat", "mean"), **c.get("params", {}))
      if v is not None:
        n = self.count[name] = self.count.get(name, 0) + 1
        a = max(alpha, 1.0 / n)
        self.value[name] = v if n == 1 else self.value[name] + a * (v - self.value[name])
    if step - self.last_update >= update_steps:
      self.last_update = step
      for c in constraints:
        name = c["name"]
        if self.count.get(name, 0) < warmup:
          continue
        short = c["target"] - self.value[name] if c["sense"] == "min" else self.value[name] - c["target"]
        self.lam[name] = min(max(self.lam[name] + c["lr"] * short, 0.0), c["max_scale"] - 1.0)
        env.reward_manager.get_term_cfg(c["term"]).weight = (1.0 + self.lam[name]) * self.base[name]
    out = {}
    for c in constraints:
      name = c["name"]
      if name in self.value:
        out[f"{name}/value"] = torch.tensor(self.value[name])
        out[f"{name}/scale"] = torch.tensor(1.0 + self.lam[name])
    return out
