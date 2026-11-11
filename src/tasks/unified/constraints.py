"""Acceptance items as constraints for the unified teacher (go2-depth 2026-11-11, the user's redesign: "奖励少，约束多 ...
其余验收项写成约束，门槛直接取验收门槛 ... 约束用拉格朗日乘子 ... 推广到全部验收项").

Each constraint is one acceptance item: a measurement in training (the same quantity the acceptance judges, or for
the stairs counts, its rate per step-up), a target at the acceptance's side of the bar, and the cost terms that serve
it. A Lagrange-style multiplier scales those terms' weights: it grows while the measurement misses the target and
shrinks back when it is met, within [min_scale, max_scale] x the term's configured weight. Unlike the earlier
mdp.adaptive_constraints (still used by the blind chains, unchanged):

  - a constraint switches on at a curriculum progress (env.u_progress, curriculum.py) and, once on, tightens from a
    loose target to its bar over the next 0.3 of progress; before that its terms sit at `start_scale` of their weight
    (above 0: the reward manager skips zero-weight terms, and the stateful ones would stop updating);
  - the multiplier can also go below 1 (min_scale), so a cost that over-serves its item gives way to the task;
  - several constraints may scale one term: it takes the largest scale;
  - the multipliers and the measurements go into the checkpoint (runner.py) and come back on resume;
  - the stairs counts (slides, riser kicks, top-landing lifts, idle lifts) are rates per step-up from `meter`, a
    zero-reward term that reads the event terms every step.

The acceptance-calibrated measurements are mdp.constraints.METRICS (unchanged); `rate:<name>` are the meter's.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from src.tasks.velocity.mdp.constraints import METRICS

from .curriculum import ramp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Mean riser height of the training stairs (m): converts metres climbed into step-ups for the stairs rates.
RISER_M = 0.15


class meter:
  """Reward term with zero output (give it a tiny nonzero weight so the manager calls it, last in the reward dict):
  every step it adds the stairs event terms' counts and the metres climbed, over the robots on stairs, into sums that
  decay with a time constant of `ema_steps` steps. rate(name) = events per step-up.

  `events` maps a rate name to (reward term, unit): the term's raw value is events per second (unit "events") or
  a count of feet / wheels in a state (unit "seconds": its integral is foot- or wheel-seconds); either way raw x dt
  per step is what it adds. Climbing is feet_foothold_height_gain's raw value (metres per second, summed over the
  feet) x dt / feet / RISER_M step-ups."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.sums: dict[str, float] = {}
    self.climb = 0.0
    self.n = 0
    self.min_n = 1000
    env.u_meter = self

  def reset(self, env_ids=None) -> None:
    pass

  def state(self) -> dict:
    return {"sums": dict(self.sums), "climb": self.climb, "n": self.n}

  def load(self, state: dict) -> None:
    self.sums, self.climb, self.n = dict(state["sums"]), state["climb"], state["n"]

  def rate(self, name: str) -> float | None:
    if self.n < self.min_n or self.climb < 1e-6 or name not in self.sums:
      return None
    return self.sums[name] / self.climb

  def _raw(self, env: ManagerBasedRlEnv, term: str) -> torch.Tensor | None:
    rm = env.reward_manager
    if term not in rm._term_names:
      return None
    w = rm.get_term_cfg(term).weight
    if w == 0.0:
      return None
    return rm._step_reward[:, rm._term_names.index(term)] / w   # this step's raw value (terms before this one)

  def __call__(self, env: ManagerBasedRlEnv, events: dict, climb_term: str = "foothold_height_gain",
               feet: int = 4, ema_steps: float = 20000.0, min_n: int = 1000) -> torch.Tensor:
    self.min_n = min_n
    stairs = env.command_manager.get_term("twist").is_stair_env
    decay = math.exp(-1.0 / ema_steps)
    climb = self._raw(env, climb_term)
    if climb is not None:
      self.climb = self.climb * decay + float((climb * stairs).sum()) * env.step_dt / feet / RISER_M
      for name, term in events.items():
        raw = self._raw(env, term)
        if raw is not None:
          self.sums[name] = self.sums.get(name, 0.0) * decay + float((raw * stairs).sum()) * env.step_dt
      self.n += 1
    return torch.zeros(env.num_envs, device=env.device)


def _stairs_stop_feet(env: ManagerBasedRlEnv, sensor_name: str = "feet_ground_contact",
                      max_speed: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
  """All four feet / wheels down (the acceptance's stairs stop-feet, bar 0.9 of the robots), over the robots on a
  stairs column at the zero command that have come to rest (base speed below max_speed)."""
  from mjlab.sensor import ContactSensor

  sensor: ContactSensor = env.scene[sensor_name]
  value = (sensor.data.found > 0).all(dim=1).float()
  command = env.command_manager.get_command("twist")
  still = (command.abs().sum(dim=1) == 0.0) & (torch.norm(env.scene["robot"].data.root_link_lin_vel_b[:, :2], dim=1) < max_speed)
  return value, env.command_manager.get_term("twist").is_stair_env & still


def _stairs_roll(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """|Body roll| (rad, from the projected gravity) over the robots on a stairs column under a moving command (the
  acceptance records the roll peak on the stairs; logged, not a constraint: its bar is not calibrated)."""
  g = env.scene["robot"].data.projected_gravity_b
  roll = torch.asin(torch.clamp(g[:, 1], -1.0, 1.0)).abs()
  moving = env.command_manager.get_command("twist").abs().sum(dim=1) > 0.0
  return roll, env.command_manager.get_term("twist").is_stair_env & moving


CUSTOM = {"u_stairs_stop_feet": _stairs_stop_feet, "u_stairs_roll": _stairs_roll}


def measure(env: ManagerBasedRlEnv, metric: str, stat: str = "mean", **params) -> float | None:
  """A constraint's measurement: mdp.constraints.METRICS or CUSTOM (stat "mean" or "pNN" over the robots it applies
  to now), or "rate:<name>" from the meter."""
  if metric.startswith("rate:"):
    m = getattr(env, "u_meter", None)
    return None if m is None else m.rate(metric[5:])
  value, mask = (CUSTOM.get(metric) or METRICS[metric])(env, **params)
  if not mask.any():
    return None
  v = value[mask]
  return float(v.mean()) if stat == "mean" else float(torch.quantile(v, float(stat[1:]) / 100.0))


class constraints:
  """Curriculum term (after `progress`): the constraints (see the module docstring). `items` is a list of
  {"name", "metric", "params", "stat", "sense" ("min": value >= target, "max": value <= target), "target" (the
  bar's side), "loose" (the target when it switches on), "activate" (progress), "terms" (reward terms it scales),
  "lr", "min_scale", "max_scale", "start_scale", "enforce" (False: measured and logged only)}. Measurements are smoothed with a time constant of `ema_steps`
  calls; every `update_steps` env steps each active multiplier moves by lr x the shortfall relative to the target
  ((target - value) / |target| for "min", (value - target) / |target| for "max"), clamped to [min_scale, max_scale].
  Logged as Curriculum/u_constraints/<name>/{value,target,scale}."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.base: dict[str, float] = {}       # term -> configured weight
    self.scale: dict[str, float] = {}      # constraint -> multiplier (1 = configured weight)
    self.value: dict[str, float] = {}
    self.count: dict[str, int] = {}
    self.last_update = 0
    self.last_step = None
    env.u_constraints = self

  def reset(self, env_ids=None) -> None:
    pass

  def state(self) -> dict:
    return {"scale": dict(self.scale), "value": dict(self.value), "count": dict(self.count)}

  def load(self, state: dict) -> None:
    self.scale, self.value, self.count = dict(state["scale"]), dict(state["value"]), dict(state["count"])

  def _apply(self, env: ManagerBasedRlEnv, items: list[dict], p: float) -> None:
    """Each scaled term's weight = configured x the largest scale of the constraints on it (start_scale for one not
    yet on)."""
    per_term: dict[str, float] = {}
    for c in items:
      s = self.scale[c["name"]] if p >= c["activate"] else c.get("start_scale", 1.0)
      for t in c["terms"]:
        per_term[t] = max(per_term.get(t, 0.0), s)
    for t, s in per_term.items():
      env.reward_manager.get_term_cfg(t).weight = self.base[t] * s

  def target(self, c: dict, p: float) -> float:
    k = ramp(p, c["activate"], c["activate"] + 0.3)
    return c["loose"] + (c["target"] - c["loose"]) * k

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, items: list[dict], ema_steps: float = 1000.0,
               update_steps: int = 24, skip_steps: int = 2400, warmup: int = 240) -> dict[str, torch.Tensor]:
    del env_ids
    step = env.common_step_counter
    p = getattr(env, "u_progress", 0.0)
    for c in items:
      self.scale.setdefault(c["name"], 1.0)
      for t in c["terms"]:
        if t not in self.base:
          self.base[t] = env.reward_manager.get_term_cfg(t).weight
    if self.last_step is not None and not 0 <= step - self.last_step <= ema_steps:
      self.count.clear()   # the step counter jumped (a resume restores it): measure afresh
    self.last_step = step
    alpha = 1.0 - math.exp(-1.0 / ema_steps)
    if step >= skip_steps:
      for c in items:
        v = measure(env, c["metric"], c.get("stat", "mean"), **c.get("params", {}))
        if v is None:
          continue
        name = c["name"]
        n = self.count[name] = self.count.get(name, 0) + 1
        a = max(alpha, 1.0 / n)
        self.value[name] = v if n == 1 else self.value[name] + a * (v - self.value[name])
    if step - self.last_update >= update_steps:
      self.last_update = step
      for c in items:
        name = c["name"]
        if not c.get("enforce", True) or p < c["activate"] or self.count.get(name, 0) < warmup:
          continue
        tgt = self.target(c, p)
        norm = max(abs(tgt), 1e-6)
        short = (tgt - self.value[name]) / norm if c["sense"] == "min" else (self.value[name] - tgt) / norm
        self.scale[name] = min(max(self.scale[name] + c["lr"] * short, c.get("min_scale", 0.5)), c["max_scale"])
    self._apply(env, items, p)
    out = {}
    for c in items:
      name = c["name"]
      if name in self.value:
        out[f"{name}/value"] = torch.tensor(self.value[name])
      out[f"{name}/target"] = torch.tensor(self.target(c, p))
      out[f"{name}/scale"] = torch.tensor(self.scale[name] if p >= c["activate"] else c.get("start_scale", 1.0))
    return out
