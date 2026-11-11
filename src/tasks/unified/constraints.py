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


def _stand_hold_light(env: ManagerBasedRlEnv, term: str = "standing_position_hold",
                      max_frac: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  """mdp.constraints' stand_hold (distance from the locked hold anchor) over the robots carrying at most max_payload kg:
  the acceptance's stop-creep and march tests carry none (2026-11-11: over all robots, the heavy half of the payload
  randomization held the first Go2 teacher's creep at 3.5 cm p90 with its multiplier at the 3x cap; as back_drift's
  max_payload)."""
  value, mask = METRICS["stand_hold"](env, term=term)
  pm = payload_mask(env, hi=LIGHT_FRAC if max_frac is None else max_frac)
  return value, (mask if pm is None else mask & pm)


# The light and heavy robots as shares of the robot's payload limit (2026-11-12, doc/redesign.md: the limit is the
# walking payload -- Go2 / Go2W 8 kg -- so the 11-11 fixed 3 kg / 8 kg no longer fit): light = the acceptance's
# unloaded tests, judged by the constraints; heavy = their own reward group, no constraint (the user, 2026-11-11).
LIGHT_FRAC = 0.25
HEAVY_FRAC = 2.0 / 3.0
DEFAULT_LIMIT_KG = 12.0   # until curriculum.progress_payload has set env.u_payload_full


def payload_limit(env: ManagerBasedRlEnv) -> float:
  return float(getattr(env, "u_payload_full", DEFAULT_LIMIT_KG))


def payload_mask(env: ManagerBasedRlEnv, lo: float | None = None, hi: float | None = None) -> torch.Tensor | None:
  """The robots whose payload (mdp.base_payload_point's env.payload_kg) is in [lo, hi] x the payload limit; None before
  the first draw."""
  kg = getattr(env, "payload_kg", None)
  if kg is None:
    return None
  limit = payload_limit(env)
  m = torch.ones_like(kg, dtype=torch.bool)
  if lo is not None:
    m &= kg >= lo * limit
  if hi is not None:
    m &= kg <= hi * limit
  return m


def by_payload(metric: str, lo: float | None = None, hi: float | None = None):
  """METRICS[metric] over the robots with a payload in [lo, hi] x the limit."""
  def f(env: ManagerBasedRlEnv, **params) -> tuple[torch.Tensor, torch.Tensor]:
    value, mask = METRICS[metric](env, **params)
    pm = payload_mask(env, lo, hi)
    return value, (mask if pm is None else mask & pm)
  return f


CUSTOM = {"u_stairs_stop_feet": _stairs_stop_feet, "u_stairs_roll": _stairs_roll, "u_stand_hold_light": _stand_hold_light,
          # the flat items on the light robots (2026-11-11, the user: "现有的轻载约束 ... 只统计 ≤ 3 kg，继续对着无负载的验收门槛")
          "u_drift_light": by_payload("drift_share", hi=LIGHT_FRAC),
          "u_spin_light": by_payload("spin_hold", hi=LIGHT_FRAC),
          "u_back_drift_light": by_payload("back_drift", hi=LIGHT_FRAC)}

# The same items on the heavy robots: measured and logged (Curriculum/u_heavy/<item>/{light,heavy}), never enforced.
HEAVY_ITEMS = {
  "creep": ("stand_hold", {"term": "standing_position_hold"}, "p90", ["standing_position_hold", "standing_speed"]),
  "march": ("stand_hold", {"term": "march_position_hold"}, "p90", ["march_position_hold", "march_hip_default"]),
  "side_drift": ("drift_share", {"term": "side_drift"}, "p90", ["side_drift"]),
  "straight_drift": ("drift_share", {"term": "straight_drift"}, "p90", ["straight_drift"]),
  "spin": ("spin_hold", {}, "mean", ["spin_position_hold"]),
}


def _raw(env: ManagerBasedRlEnv, term: str) -> tuple[torch.Tensor, float] | None:
  """A reward term's raw value this step and its weight now (terms computed before the caller)."""
  rm = env.reward_manager
  if term not in rm._term_names:
    return None
  w = rm.get_term_cfg(term).weight
  if w == 0.0:
    return None
  return rm._step_reward[:, rm._term_names.index(term)] / w, w


class heavy_group:
  """Reward term (weight 1, after the terms it reads): the heavy robots' own reward group for one flat item
  (go2-depth 2026-11-11, the user: "停车蠕动、原地踏步漂移、跑偏、交替原地转，给重载机器人（比如 ≥ 8 kg）单独加一组奖励项，
  用来把重载的行为往无负载的水平拉。这组只作奖励，不作约束：不设门槛，也不用乘子往上加压").

  The item's cost terms run on every robot at the weight the constraints give them (configured x multiplier). For a
  robot with at least `min_kg` of payload this term adds raw x (heavy_scale x configured - current), so that robot
  feels each cost at heavy_scale x its configured weight, whatever the multiplier: the light robots' constraint
  never presses on the heavy ones, and the heavy ones are still pulled towards the same behaviour."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    pass

  def reset(self, env_ids=None) -> None:
    pass

  def __call__(self, env: ManagerBasedRlEnv, terms: list[str], min_frac: float = HEAVY_FRAC,
               heavy_scale: float = 1.0) -> torch.Tensor:
    out = torch.zeros(env.num_envs, device=env.device)
    heavy = payload_mask(env, lo=min_frac)
    if heavy is None:
      return out
    base = getattr(getattr(env, "u_constraints", None), "base", {})
    for t in terms:
      r = _raw(env, t)
      if r is None:
        continue
      raw, w = r
      out += heavy.float() * raw * (heavy_scale * base.get(t, w) - w)
    return out


class heavy_log:
  """Curriculum term: each HEAVY_ITEMS item measured on the light (<= LIGHT_FRAC of the limit) and on the heavy
  (>= HEAVY_FRAC) robots, smoothed like the constraints; logged as Curriculum/u_heavy/<item>/{light,heavy,gap}.
  Information only."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.value: dict[str, float] = {}

  def reset(self, env_ids=None) -> None:
    pass

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, ema_steps: float = 1000.0,
               skip_steps: int = 2400) -> dict[str, torch.Tensor]:
    del env_ids
    out = {}
    if env.common_step_counter < skip_steps:
      return out
    a = 1.0 - math.exp(-1.0 / ema_steps)
    for item, (metric, params, stat, _) in HEAVY_ITEMS.items():
      for side, (lo, hi) in (("light", (None, LIGHT_FRAC)), ("heavy", (HEAVY_FRAC, None))):
        try:
          value, mask = METRICS[metric](env, **params)
        except Exception:
          continue
        pm = payload_mask(env, lo, hi)
        if pm is None:
          continue
        mask = mask & pm
        if not mask.any():
          continue
        v = value[mask]
        x = float(v.mean()) if stat == "mean" else float(torch.quantile(v, float(stat[1:]) / 100.0))
        k = f"{item}/{side}"
        self.value[k] = x if k not in self.value else self.value[k] + a * (x - self.value[k])
        out[k] = torch.tensor(self.value[k])
      if f"{item}/light" in self.value and f"{item}/heavy" in self.value:
        out[f"{item}/gap"] = torch.tensor(self.value[f"{item}/heavy"] - self.value[f"{item}/light"])
    return out


def measure(env: ManagerBasedRlEnv, metric: str, stat: str = "mean", **params) -> float | None:
  """A constraint's measurement: mdp.constraints.METRICS or CUSTOM (stat "mean" or "pNN" over the robots it applies
  to now), "rate:<name>" from the meter, or "lib:<name>" from the shared metrics (src/metrics, lib_meter)."""
  if metric.startswith("lib:"):
    m = getattr(env, "u_lib_meter", None)
    return None if m is None else m.rate(metric[4:])
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
               update_steps: int = 24, skip_steps: int = 2400, warmup: int = 240,
               frozen: bool = False) -> dict[str, torch.Tensor]:
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
    if not frozen and step - self.last_update >= update_steps:   # frozen: the students keep the teacher's final scales
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


class lib_meter:
  """Reward term with zero output (tiny weight, so the manager calls it every step): the shared stairs metrics
  (src/metrics.mj.StairsMeter, the same counters as tinynav's acceptance) over the first `sample` envs; env.u_lib_meter.
  The "lib:<name>" constraints read its rates (go2-depth 2026-11-12, doc/redesign.md: the constraints use the
  acceptance's own measures)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    from src.metrics.mj import StairsMeter

    p = cfg.params
    n = min(int(p.get("sample", 512)), env.num_envs)
    self.meter = StairsMeter(env, torch.arange(n, device=env.device), wheeled=bool(p.get("wheeled", False)),
                             ema_steps=float(p.get("ema_steps", 20000.0)), min_steps=float(p.get("min_steps", 200.0)))
    env.u_lib_meter = self.meter
    env.u_lib_meter_term = self   # runner.py checkpoints its state

  def reset(self, env_ids=None) -> None:
    pass

  def state(self) -> dict:
    return self.meter.state()

  def load(self, state: dict) -> None:
    self.meter.load(state)

  def __call__(self, env: ManagerBasedRlEnv, wheeled: bool = False, sample: int = 512, ema_steps: float = 20000.0,
               min_steps: float = 200.0) -> torch.Tensor:
    self.meter.step(env)
    return torch.zeros(env.num_envs, device=env.device)
