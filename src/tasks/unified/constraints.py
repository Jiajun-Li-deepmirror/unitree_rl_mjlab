"""Acceptance items as constraints for the unified teacher (go2-depth 2026-11-11, the user's redesign: "奖励少，约束多 ...
其余验收项写成约束，门槛直接取验收门槛 ... 约束用拉格朗日乘子 ... 推广到全部验收项").

Each constraint is one acceptance item: a measurement in training (the same quantity the acceptance judges, or for
the stairs counts, its rate per step-up), a target at the acceptance's side of the bar, and the cost terms that serve
it. A Lagrange-style multiplier scales those terms' weights: it grows while the measurement misses the target and
shrinks back when it is met, within [min_scale, max_scale] x the term's configured weight; it moves in log space
(x exp(lr x shortfall) per update, 2026-11-12). Unlike the earlier mdp.adaptive_constraints (still used by the blind
chains, unchanged):

  - a constraint switches on at a curriculum progress (env.u_progress, curriculum.py) and, once on, tightens from a
    loose target to its own (see "a margin" below) over the next 0.3 of progress; before that its terms sit at
    `start_scale` of their weight
    (above 0: the reward manager skips zero-weight terms, and the stateful ones would stop updating);
  - the stairs-quality constraints ("by": "q"; 2026-11-12) follow the stairs quality q instead (env.u_quality,
    curriculum.py: it starts once the robots climb): off while q is 0, their terms at the recipe's M weights (a
    thousandth of the P weight for a term the M keys do not have, so it still runs and is measured); from then on the
    terms move to weight x multiplier over q 0 -> Q_IN and the target tightens from loose to its own over q 0 -> 1;
  - the multiplier can also go below 1 (min_scale), so a cost that over-serves its item gives way to the task;
  - several constraints may scale one term: it takes the largest scale;
  - the multipliers and the measurements go into the checkpoint (runner.py) and come back on resume;
  - the stairs counts (slides, riser kicks, top-landing lifts, idle lifts) are rates per step-up from `meter`, a
    zero-reward term that reads the event terms every step.

The acceptance-calibrated measurements are mdp.constraints.METRICS (unchanged); `rate:<name>` are the meter's.

Since 2026-11-12 every measurement is taken over the evaluation robots only (env.u_eval_mask, set by the teacher's
runner: the first eval_count(num_envs) robots act with the policy's mean action and take no part in the update). The
acceptance judges the policy without exploration noise, and the targets were calibrated on the blind chains' late
stages (noise ~0.3); the teacher explores at 0.5-0.7 while it learns the stairs, and under that noise its readings
missed targets it met: the Go2 teacher at 6400, mean action / sampled action: creep 0.021 / 0.042 m (target 0.008),
march 0.036 / 0.060 (0.04), straight drift 0.059 / 0.068 (0.06), clock 0.87 / 0.77 (0.85), stairs stop-feet 0.86 /
0.49 (0.6). Four multipliers went to the 3x cap on readings no cost weight could bring down, and the policy paid for
it (reward 28 -> -3, noise 0.48 -> 0.69, falls 8% -> 17%). Without the mask (any other runner) they read every robot.

Since the from-zero retrain (2026-11-12, after the peer review; doc/unified_teacher.md):

  - seen and blind apart. The teacher walks in two scan modes (teacher_env.u_scan_mask), and its students each copy
    one: the depth student the seen mode, the blind student the blind one. Every measurement is taken per mode (the
    evaluation robots are a fixed half in each, by index) and logged as <name>/value_seen and <name>/value_blind; a
    constraint has one multiplier, driven by the mode further from the target among the modes it applies to
    (item "modes"; the tread placement items, mid and edge, apply to the seen mode only -- a robot that cannot see
    the tread cannot be asked to place its feet on it), and the gate (scripts/unified/teacher_gate.py) asks every
    one of those modes to pass;
  - a margin for the students. "target" is the bar the students are accepted at; the multipliers work toward
    "tight", TEACHER_MARGIN inside it ("max" items 0.8 x the bar, "min" items a fifth of the way from the bar to 1),
    and the gate judges at the bar. Both are logged (<name>/target: the teacher's target now; <name>/bar);
  - the task first, per terrain family. Each constraint belongs to a family of ground -- "up" (the climbing stairs
    columns), "down" (the descents), "stairs" (either: relaxed when one of the two is) or "other" -- and relaxes
    while that family's episodes fail. One global fail rate let the flat ground's falls relax the stairs items and
    the other way round. The other ground reads its episodes' fail rate over the training robots
    (curriculum.progress_stairs_share's env.u_family_fail): on above RELAX_ON, off again below RELAX_OFF. The
    stairs families read the evaluation robots' on their fixed rows instead (curriculum.eval_rows: the rows the
    acceptance covers, each alike): at the curriculum's own rows a robot is promoted until it fails, ~0.6 of the
    stairs episodes whatever the policy can climb (the 11-12 Go2 teacher), which would have kept the stairs items
    relaxed for the whole run. The stairs items' readings come from those rows too: the measurements are the
    evaluation robots'.
    On the stairs the question is whether the fail rate got worse, not whether it is high (2026-11-12, the probe:
    the old Go2 teacher continued under this code read 0.36 up / 0.40 down on the fixed rows -- its top rows fail
    as a matter of course, and its loaded robots more than the light ones as the payload grows -- so a fixed
    RELAX_ON of 0.3 held every stairs constraint relaxed for the whole quality stage, 2000 iterations: no multiplier
    grew and the gate could not pass). A stairs family now reads the light evaluation robots' fail rate
    (curriculum.eval_light; <= LIGHT_FRAC of the payload limit) against a reference that follows that same reading
    slowly -- down with a time constant of RELAX_REF_DOWN_IT iterations, up with RELAX_REF_UP_IT -- and relaxes
    when the reading is more than RELAX_GAP_ON above the reference, until it is within RELAX_GAP_OFF of it again.
    Fewer than RELAX_MIN_EPISODES light episodes: not judged, not relaxed. Logged as relax_light_<family>,
    relax_ref_<family>, relax_n_<family>; the reference goes into the checkpoint.
  - gate-only items (2026-11-12): an enforced item without reward terms has no multiplier to move; it is measured,
    logged and judged by the gate like the others. The survival items (teacher_env.constraint_items: survive_up,
    survive_down) are such: "evalfail:<family>", the light evaluation robots' fail rate on the worst of the
    acceptance's step-height bands (curriculum.eval_worst_band), per scan mode.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from src.tasks.velocity.mdp.constraints import METRICS

from .curriculum import MAX_FAIL_RATE, Q_IN, STEPS_PER_IT, eval_light, eval_worst_band, ramp, slope_envs

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Mean riser height of the training stairs (m): converts metres climbed into step-ups for the stairs rates.
RISER_M = 0.15
# The evaluation robots' share of the environments (module docstring); U_EVAL_ENVS overrides the count, 0: none.
EVAL_SHARE = 0.125
# The scan modes (teacher_env.u_scan_mask) the measurements are split by.
MODES = ("seen", "blind")
# The teacher's own targets sit this far inside the students' bars (module docstring).
TEACHER_MARGIN = 0.8
# A family's constraints relax while more than RELAX_ON of its episodes fail, until fewer than RELAX_OFF do.
RELAX_ON, RELAX_OFF = MAX_FAIL_RATE, 0.25
# The stairs families (module docstring): the light evaluation robots' fail rate against its own slow reference.
# The reference's time constants (iterations) toward a lower and toward a higher reading, the gap above it that
# relaxes and the one that ends it, and the light episodes a family needs before it is judged. (The reading's own
# noise: ~0.025-0.035, each row's rate remembering its last ~40 light episodes.)
RELAX_REF_DOWN_IT, RELAX_REF_UP_IT = 500.0, 2000.0
RELAX_GAP_ON, RELAX_GAP_OFF = 0.08, 0.04
RELAX_MIN_EPISODES = 200
FAMILIES = ("up", "down", "stairs", "other")
# The multipliers' step per update, in log space: full shortfall for one iteration is x exp(LR) (1 -> 3 takes ~370
# iterations at the least; the additive 0.03 took 67 and overshot: the Go2 teacher's creep / march went 1 -> 3 in a
# dozen updates before the shortfall was clamped).
LR = 0.003


def with_margin(sense: str, bar: float) -> float:
  """The teacher's target for an item whose students' bar is `bar`: TEACHER_MARGIN inside it."""
  if sense == "max":
    return bar * TEACHER_MARGIN
  return bar + (1.0 - TEACHER_MARGIN) * (1.0 - bar)


def eval_count(num_envs: int) -> int:
  """How many robots (the first ones) evaluate the mean action instead of training."""
  v = os.environ.get("U_EVAL_ENVS")
  n = int(v) if v not in (None, "") else int(num_envs * EVAL_SHARE)
  return max(0, min(n, num_envs // 2))


def on_eval(env: ManagerBasedRlEnv, mask: torch.Tensor) -> torch.Tensor:
  """`mask` over the evaluation robots only, when the run has them."""
  ev = getattr(env, "u_eval_mask", None)
  return mask if ev is None else mask & ev


def off_slopes(env: ManagerBasedRlEnv, mask: torch.Tensor) -> torch.Tensor:
  """`mask` without the robots on a sloped tile (curriculum.SLOPE_NAMES; 2026-11-12): the acceptance has no item on
  slopes, and holding a place or a line there is not what the flat items' bars were set on. The cost terms still act
  on those robots; the stairs items never count them (a sloped tile is not a stairs column)."""
  slope = slope_envs(env)
  return mask if slope is None else mask & ~slope


def scan_modes(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor] | None:
  """The robots in each scan mode this episode ({"seen", "blind"}: masks from teacher_env.u_scan_mask's
  env.u_scan_on); None for an environment without the scan mask (or before its first reset)."""
  on = getattr(env, "u_scan_on", None)
  if on is None:
    return None
  seen = on > 0.5
  return {"seen": seen, "blind": ~seen}


class meter:
  """Reward term with zero output (give it a tiny nonzero weight so the manager calls it, last in the reward dict):
  every step it adds the stairs event terms' counts and the metres climbed, over the robots on stairs, into sums that
  decay with a time constant of `ema_steps` steps. rate(name) = events per step-up.

  `events` maps a rate name to (reward term, unit): the term's raw value is events per second (unit "events") or
  a count of feet / wheels in a state (unit "seconds": its integral is foot- or wheel-seconds); either way raw x dt
  per step is what it adds. Climbing is feet_foothold_height_gain's raw value (metres per second, summed over the
  feet) x dt / feet / RISER_M step-ups. The same sums are kept per scan mode (rate(name, scan); 2026-11-12)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.sums: dict[str, float] = {}
    self.climb = 0.0
    self.modes: dict[str, dict[str, float]] = {}   # scan mode -> {"climb": step-ups, <rate name>: events}
    self.n = 0
    self.min_n = 1000
    self.eval = False   # True: only the evaluation robots count (the teacher's runner sets it with env.u_eval_mask)
    env.u_meter = self

  def reset(self, env_ids=None) -> None:
    pass

  def state(self) -> dict:
    return {"sums": dict(self.sums), "climb": self.climb, "n": self.n, "eval": self.eval,
            "modes": {k: dict(v) for k, v in self.modes.items()}}

  def load(self, state: dict) -> None:
    """A checkpoint's sums taken over the same robots (the evaluation robots or all of them); others start afresh."""
    if bool(state.get("eval", False)) == self.eval:
      self.sums, self.climb, self.n = dict(state["sums"]), state["climb"], state["n"]
      self.modes = {k: dict(v) for k, v in (state.get("modes") or {}).items()}

  def rate(self, name: str, scan: str | None = None) -> float | None:
    """Events per step-up, over all the robots measured or over those in scan mode `scan`."""
    if scan is not None:
      a = self.modes.get(scan)
      if a is None or self.n < self.min_n or a.get("climb", 0.0) < 1e-6 or name not in a:
        return None
      return a[name] / a["climb"]
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
    stairs = on_eval(env, env.command_manager.get_term("twist").is_stair_env)
    modes = {k: stairs & m for k, m in (scan_modes(env) or {}).items()}
    decay = math.exp(-1.0 / ema_steps)
    climb = self._raw(env, climb_term)
    if climb is not None:
      self.climb = self.climb * decay + float((climb * stairs).sum()) * env.step_dt / feet / RISER_M
      for k, m in modes.items():
        a = self.modes.setdefault(k, {})
        a["climb"] = a.get("climb", 0.0) * decay + float((climb * m).sum()) * env.step_dt / feet / RISER_M
      for name, term in events.items():
        raw = self._raw(env, term)
        if raw is not None:
          self.sums[name] = self.sums.get(name, 0.0) * decay + float((raw * stairs).sum()) * env.step_dt
          for k, m in modes.items():
            a = self.modes[k]
            a[name] = a.get(name, 0.0) * decay + float((raw * m).sum()) * env.step_dt
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
DEFAULT_LIMIT_KG = 8.0   # Go2 / Go2W's walking payload; used only until curriculum.progress_payload sets env.u_payload_full


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
        mask = off_slopes(env, on_eval(env, mask & pm))
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


def measure(env: ManagerBasedRlEnv, metric: str, stat: str = "mean", scan: str | None = None, **params) -> float | None:
  """A constraint's measurement: mdp.constraints.METRICS or CUSTOM (stat "mean" or "pNN" over the robots it applies
  to now), "rate:<name>" from the meter, "lib:<name>" from the shared metrics (src/metrics, lib_meter), or
  "evalfail:<family>": the light evaluation robots' fail rate on the worst step-height band of their fixed stairs
  rows (curriculum.eval_worst_band; None in a run without evaluation robots). `scan` ("seen" / "blind"): over the
  robots in that scan mode only (None where the environment has no scan modes)."""
  if metric.startswith("evalfail:"):
    return eval_worst_band(env, metric[9:], scan or "all")
  if metric.startswith("lib:"):
    m = getattr(env, "u_lib_meter", None)
    return None if m is None else m.rate(metric[4:], scan)
  if metric.startswith("rate:"):
    m = getattr(env, "u_meter", None)
    return None if m is None else m.rate(metric[5:], scan)
  value, mask = (CUSTOM.get(metric) or METRICS[metric])(env, **params)
  mask = off_slopes(env, on_eval(env, mask))
  if scan is not None:
    modes = scan_modes(env)
    if modes is None:
      return None
    mask = mask & modes[scan]
  if not mask.any():
    return None
  v = value[mask]
  return float(v.mean()) if stat == "mean" else float(torch.quantile(v, float(stat[1:]) / 100.0))


class constraints:
  """Curriculum term (after `progress`): the constraints (see the module docstring). `items` is a list of
  {"name", "metric", "params", "stat", "sense" ("min": value >= target, "max": value <= target), "target" (the
  students' bar: what the gate judges), "tight" (the teacher's own target, inside the bar; absent: the bar), "loose"
  (the target when it switches on), "activate" (progress), "terms" (reward terms it scales), "lr", "min_scale",
  "max_scale", "start_scale", "enforce" (False: measured and logged only; "target" may then be None: no bar yet;
  True with no terms: a gate-only item, judged but with no multiplier),
  "family" (the ground whose fail rate relaxes it: FAMILIES), "modes" (the scan modes it applies to: MODES by
  default)}; a stairs-quality item also has "by": "q", "start" and "base" ({term: weight}: the M and the P weight;
  teacher_env.py) and follows env.u_quality instead of the progress (module docstring).

  Each item is measured per scan mode (every mode, also one it does not apply to: logged only) and smoothed with a
  time constant of `ema_steps` calls. Every `update_steps` env steps each active multiplier is multiplied by
  exp(lr x shortfall), the shortfall relative to the target ((target - value) / |target| for "min", (value - target)
  / |target| for "max", within [-1, 1]) and the larger of its modes', and clamped to [min_scale, max_scale].
  self.value[name] is the reading that drives the multiplier (the mode further from the target).

  Logged as Curriculum/u_constraints/<name>/{value_seen,value_blind,value,target,bar,scale}, relaxing (any family),
  relaxing_<family>, and for the stairs families relax_light_<family> (the light evaluation robots' fail rate),
  relax_ref_<family> (its reference) and relax_n_<family> (the episodes behind it)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.base: dict[str, float] = {}       # term -> configured weight
    self.scale: dict[str, float] = {}      # constraint -> multiplier (1 = configured weight)
    self.value: dict[str, float] = {}      # constraint -> the driving reading
    self.values: dict[str, float] = {}     # "<constraint>@<mode>" (no scan modes: "<constraint>") -> smoothed reading
    self.count: dict[str, int] = {}        # the same keys -> measurements so far
    self.relax: dict[str, bool] = {f: False for f in FAMILIES}
    self.ref: dict[str, float] = {}        # stairs family -> the reference its light fail rate is compared with
    self.light: dict[str, dict] = {}       # stairs family -> {"fail", "n"}: that fail rate now, its episodes
    self.ref_step = None
    self.last_update = 0
    self.last_step = None
    env.u_constraints = self

  def reset(self, env_ids=None) -> None:
    pass

  def state(self) -> dict:
    return {"scale": dict(self.scale), "value": dict(self.value), "values": dict(self.values),
            "count": dict(self.count), "relax": dict(self.relax), "ref": dict(self.ref)}

  def load(self, state: dict) -> None:
    """A checkpoint's multipliers and readings. One from before the scan modes' split (no "values") gives the
    multipliers; its readings are taken afresh."""
    self.scale = dict(state["scale"])
    self.value = dict(state.get("value", {}))
    self.values = dict(state.get("values", {}))
    self.count = dict(state["count"]) if "values" in state else {}
    self.relax.update({f: bool(v) for f, v in (state.get("relax") or {}).items() if f in self.relax})
    # (a checkpoint from before the stairs families' reference has none: it starts at the first judged reading)
    self.ref = {f: float(v) for f, v in (state.get("ref") or {}).items()}

  @staticmethod
  def on(c: dict, p: float, q: float) -> bool:
    """Whether constraint c is switched on: a stairs-quality one ("by": "q") once the quality stage has begun, the
    others from their own progress."""
    return q > 0.0 if c.get("by") == "q" else p >= c["activate"]

  def _apply(self, env: ManagerBasedRlEnv, items: list[dict], p: float, q: float) -> None:
    """Each scaled term's weight: configured x the constraint's scale (start_scale for one not yet on). A
    stairs-quality constraint ("by": "q", curriculum.py) holds its terms at their "start" weights (the recipe's M
    values) while q is 0 and moves them to configured x scale over q 0 -> Q_IN. Several constraints on one term: the
    strongest weight."""
    per_term: dict[str, float] = {}
    k = ramp(q, 0.0, Q_IN)
    for c in items:
      for t in c["terms"]:
        if c.get("by") == "q":
          w0 = c["start"][t]
          w = w0 + (self.base[t] * self.scale[c["name"]] - w0) * k
        else:
          w = self.base[t] * (self.scale[c["name"]] if p >= c["activate"] else c.get("start_scale", 1.0))
        if t not in per_term or abs(w) > abs(per_term[t]):
          per_term[t] = w
    for t, w in per_term.items():
      env.reward_manager.get_term_cfg(t).weight = w

  def target(self, c: dict, p: float, q: float = 1.0) -> float | None:
    """The teacher's target now: from "loose" to "tight" (the margin inside the students' bar "target"). None for
    an item without a bar (logged only)."""
    if c["target"] is None:
      return None
    k = ramp(q, 0.0, 1.0) if c.get("by") == "q" else ramp(p, c["activate"], c["activate"] + 0.3)
    return c["loose"] + (c.get("tight", c["target"]) - c["loose"]) * k

  @staticmethod
  def shortfall(c: dict, value: float, target: float) -> float:
    """How far `value` is on the wrong side of `target`, relative to it (negative: met)."""
    norm = max(abs(target), 1e-6)
    return (target - value) / norm if c["sense"] == "min" else (value - target) / norm

  @staticmethod
  def keys(c: dict, split: bool, every: bool = False) -> list[str]:
    """The readings' keys of constraint c: per scan mode when the environment has them (`every`: also the modes the
    constraint does not apply to)."""
    if not split:
      return [c["name"]]
    return [f"{c['name']}@{m}" for m in (MODES if every else c.get("modes", MODES))]

  def _families(self, env: ManagerBasedRlEnv, step: int | None = None) -> None:
    """Which families of ground are failing (module docstring). The other ground: the training robots' fail rate
    (the global one, curriculum.progress, before it has had an episode), on above RELAX_ON, off below RELAX_OFF. The
    stairs families: the light evaluation robots' fail rate on their fixed rows against its slow reference, on more
    than RELAX_GAP_ON above it, off within RELAX_GAP_OFF; not judged (not relaxed) with fewer than
    RELAX_MIN_EPISODES light episodes or in a run without evaluation robots. `step`: the env step counter (the
    reference's clock)."""
    glob = getattr(getattr(env, "u_progress_term", None), "fail", None) or 0.0
    other = (getattr(env, "u_family_fail", None) or {}).get("other", glob)
    if other > RELAX_ON:
      self.relax["other"] = True
    elif other < RELAX_OFF:
      self.relax["other"] = False
    step = env.common_step_counter if step is None else step
    # (a resume restores the step counter: no time passed; a gap is counted as ten iterations at the most)
    dt = 0.0 if self.ref_step is None else float(min(max(step - self.ref_step, 0), 10 * STEPS_PER_IT))
    self.ref_step = step
    for f in ("up", "down"):
      v, n = eval_light(env, f)
      self.light[f] = {"fail": v, "n": n}
      if v is None or n < RELAX_MIN_EPISODES:
        self.relax[f] = False
        continue
      ref = self.ref.get(f)
      if ref is None:
        ref = v
      else:
        tau = (RELAX_REF_DOWN_IT if v < ref else RELAX_REF_UP_IT) * STEPS_PER_IT
        ref += (1.0 - math.exp(-dt / tau)) * (v - ref)
      self.ref[f] = ref
      if v > ref + RELAX_GAP_ON:
        self.relax[f] = True
      elif v < ref + RELAX_GAP_OFF:
        self.relax[f] = False
    self.relax["stairs"] = self.relax["up"] or self.relax["down"]

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, items: list[dict], ema_steps: float = 1000.0,
               update_steps: int = 24, skip_steps: int = 2400, warmup: int = 240,
               frozen: bool = False) -> dict[str, torch.Tensor]:
    del env_ids
    step = env.common_step_counter
    p = getattr(env, "u_progress", 0.0)
    q = getattr(env, "u_quality", 0.0)
    for c in items:
      self.scale.setdefault(c["name"], 1.0)
      if c.get("by") == "q" and q <= 0.0:
        # a stairs-quality constraint starts its stage at 1 (a checkpoint from before the stage carries the
        # multipliers of constraints that were on from progress 0.5)
        self.scale[c["name"]] = 1.0
      for t in c["terms"]:
        if t not in self.base:
          # the configured (P) weight: the item's own record when it has one, else the live term's at the first call
          self.base[t] = c.get("base", {}).get(t, env.reward_manager.get_term_cfg(t).weight)
    if self.last_step is not None and not 0 <= step - self.last_step <= ema_steps:
      self.count.clear()   # the step counter jumped (a resume restores it): measure afresh
    self.last_step = step
    alpha = 1.0 - math.exp(-1.0 / ema_steps)
    split = scan_modes(env) is not None
    if step >= skip_steps:
      for c in items:
        for mode in (MODES if split else (None,)):
          v = measure(env, c["metric"], c.get("stat", "mean"), scan=mode, **c.get("params", {}))
          if v is None:
            continue
          key = c["name"] if mode is None else f"{c['name']}@{mode}"
          n = self.count[key] = self.count.get(key, 0) + 1
          a = max(alpha, 1.0 / n)
          self.values[key] = v if n == 1 else self.values[key] + a * (v - self.values[key])
    # the reading that drives each multiplier: of the modes the constraint applies to, the one further from the target
    for c in items:
      tgt = self.target(c, p, q)
      read = [self.values[k] for k in self.keys(c, split) if self.count.get(k, 0) > 0]
      if read:   # (an item without a bar: the larger reading)
        self.value[c["name"]] = max(read, key=(lambda v: v) if tgt is None else (lambda v: self.shortfall(c, v, tgt)))
    # The task first (2026-11-12): while more than RELAX_ON of a family's episodes end in a fall or an illegal
    # contact, none of that family's multipliers grows and those above 1 relax toward it (x exp(-lr) per update). The
    # Go2W teacher v2 had six at the 3x cap (drift, creep, march, wheels down) while it could not meet them; its action
    # std rose 0.75 -> 1.08, its falls 26% -> 65% and its terrain level 2.5 -> 0.6. Per family since the from-zero
    # retrain (module docstring, _families): the other ground on the training robots' fail rate, the stairs families
    # on the light evaluation robots' on their fixed rows, against its own reference.
    self._families(env, step)
    if not frozen and step - self.last_update >= update_steps:   # frozen: the students keep the teacher's final scales
      self.last_update = step
      for c in items:
        name = c["name"]
        if not c.get("enforce", True) or not c["terms"] or not self.on(c, p, q):   # (no terms: a gate-only item)
          continue
        lr = c.get("lr", LR)
        if self.relax[c.get("family", "other")]:
          if self.scale[name] > 1.0:
            self.scale[name] = max(1.0, self.scale[name] * math.exp(-lr))
          continue
        # a mode not measured (a student's task, U_SCAN: one mode only) does not hold the others back
        read = [self.values[k] for k in self.keys(c, split) if self.count.get(k, 0) >= warmup]
        if not read:
          continue
        tgt = self.target(c, p, q)
        # at most lr per update (2026-11-12): the shortfall relative to a small target (creep 0.009 m) was 5x, and
        # the Go2 teacher's creep / march multipliers went 1 -> 3 in a dozen updates, drift 0.5 -> 2.3 as the flat
        # targets tightened by progress 0.45; its action std 0.66 -> 1.05, falls 10% -> 46%
        short = max(-1.0, min(1.0, max(self.shortfall(c, v, tgt) for v in read)))
        self.scale[name] = min(max(self.scale[name] * math.exp(lr * short), c.get("min_scale", 0.5)), c["max_scale"])
    self._apply(env, items, p, q)
    ev = getattr(env, "u_eval_mask", None)
    out = {"relaxing": torch.tensor(float(any(self.relax.values()))),
           "eval_envs": torch.tensor(0.0 if ev is None else float(ev.sum()))}
    for f in FAMILIES:
      out[f"relaxing_{f}"] = torch.tensor(float(self.relax[f]))
    for f, d in self.light.items():
      out[f"relax_n_{f}"] = torch.tensor(float(d["n"]))
      if d["fail"] is not None:
        out[f"relax_light_{f}"] = torch.tensor(d["fail"])
      if f in self.ref:
        out[f"relax_ref_{f}"] = torch.tensor(self.ref[f])
    for c in items:
      name = c["name"]
      for mode in MODES:
        if f"{name}@{mode}" in self.values:
          out[f"{name}/value_{mode}"] = torch.tensor(self.values[f"{name}@{mode}"])
      if name in self.value:
        out[f"{name}/value"] = torch.tensor(self.value[name])
      if c["target"] is not None:
        out[f"{name}/target"] = torch.tensor(self.target(c, p, q))
        out[f"{name}/bar"] = torch.tensor(float(c["target"]))
      # the scale as applied: a stairs-quality one before its stage reads 0 (its terms at their start weights)
      off = 0.0 if c.get("by") == "q" else c.get("start_scale", 1.0)
      out[f"{name}/scale"] = torch.tensor(self.scale[name] if self.on(c, p, q) else off)
    return out


class quality_terms:
  """Curriculum term (after `progress` and the recipe's own ramps): the reward weights and parameters that differ
  between the recipe's M and P keys (teacher_env.quality_diff), at the M values while the stairs quality q
  (env.u_quality, curriculum.py) is 0 and at the P values from q = Q_IN on.
    weights  {term: (m, p)}                 linear in k = ramp(q, 0, Q_IN); terms the constraints scale are not here
    params   {term: {key: (m, mode)}}       mode "lerp": numbers, linear in k from m to the term's own (P) value;
                                            "switch": anything else, the P value once q > 0
  The P values are the live term's own (read at the first call: the manager has resolved its scene entities)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.p_value: dict[tuple[str, str], object] = {}
    env.u_quality_terms = self

  def reset(self, env_ids=None) -> None:
    pass

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, weights: dict, params: dict) -> dict[str, torch.Tensor]:
    del env_ids
    q = getattr(env, "u_quality", 0.0)
    k = ramp(q, 0.0, Q_IN)
    rm = env.reward_manager
    for t, (a, b) in weights.items():
      rm.get_term_cfg(t).weight = a + (b - a) * k
    for t, keys in params.items():
      tp = rm.get_term_cfg(t).params
      for key, (a, mode) in keys.items():
        if (t, key) not in self.p_value:
          self.p_value[t, key] = tp[key]
        b = self.p_value[t, key]
        tp[key] = a + (b - a) * k if mode == "lerp" else (b if q > 0.0 else a)
    return {"k": torch.tensor(k)}


class lib_meter:
  """Reward term with zero output (tiny weight, so the manager calls it every step): the shared stairs metrics
  (src/metrics.mj.StairsMeter, the same counters as tinynav's acceptance) over the first `sample` envs; env.u_lib_meter.
  The "lib:<name>" constraints read its rates (go2-depth 2026-11-12, doc/redesign.md: the constraints use the
  acceptance's own measures), per scan mode too (the meter's groups)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    from src.metrics.mj import StairsMeter

    p = cfg.params
    n = min(int(p.get("sample", 512)), env.num_envs)
    self.meter = StairsMeter(env, torch.arange(n, device=env.device), wheeled=bool(p.get("wheeled", False)),
                             ema_steps=float(p.get("ema_steps", 20000.0)), min_steps=float(p.get("min_steps", 200.0)),
                             straight=bool(p.get("straight", True)))
    self.meter.groups = self._groups
    env.u_lib_meter = self.meter
    env.u_lib_meter_term = self   # runner.py checkpoints its state

  @staticmethod
  def _groups(env: ManagerBasedRlEnv, ids: torch.Tensor) -> dict:
    """The meter's robots by scan mode ({} for an environment without the modes)."""
    on = getattr(env, "u_scan_on", None)
    if on is None:
      return {}
    seen = (on[ids] > 0.5).cpu().numpy()
    return {"seen": seen, "blind": ~seen}

  def reset(self, env_ids=None) -> None:
    pass

  def state(self) -> dict:
    return self.meter.state()

  def load(self, state: dict) -> None:
    self.meter.load(state)

  def __call__(self, env: ManagerBasedRlEnv, wheeled: bool = False, sample: int = 512, ema_steps: float = 20000.0,
               min_steps: float = 200.0, straight: bool = True) -> torch.Tensor:
    self.meter.step(env)
    return torch.zeros(env.num_envs, device=env.device)
