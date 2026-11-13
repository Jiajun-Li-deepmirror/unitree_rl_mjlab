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
  - the stop creep on the acceptance's time base (2026-11-12, `stop_creep`): the net displacement from 2 s after the
    robot was last disturbed, over a window of up to 4 s, instead of the distance from the hold anchor over the
    whole stand. An item whose measure changes starts afresh on a checkpoint from before the change (multiplier 1,
    no readings: `constraints.load`, the items' signatures).
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


def off_scripts(env: ManagerBasedRlEnv, mask: torch.Tensor) -> torch.Tensor:
  """`mask` without the robots under a flat script (flat.flat_measure; 2026-11-12): they follow the acceptance's
  fixed commands and give the "flat:" readings; the readings defined on the drawn commands stay on those."""
  flat = getattr(env, "u_flat", None)
  return mask if flat is None else mask & ~flat.on


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


class stop_creep:
  """Reward term with zero output (a tiny nonzero weight; after `term` in the reward dict: it reads the clock that
  term updated this step): the stop creep the way the acceptance measures it (go2-depth 2026-11-12).

  The acceptance (scripts/blind/eval/eval_standing.py) releases the command, waits 2 s and takes the base's net
  displacement over the following 4 s; each group's mean must stay under 1 cm. The earlier measure
  (mdp.constraints' stand_hold on `term`) is the distance from the hold anchor: the anchor locks as soon as the base
  has settled (~0.6 s into the stand) and stays for the whole stand (4-8 s), so the reading also holds the rest of
  the slide before the acceptance's 2 s mark and grows with the stand's length. The Go2 teacher at 2300 under the
  training randomization (unitree-rl-mjlab-62's comparison, 1060 stops, no pushes): the anchor measure p50 1.0 /
  p90 2.1 cm, the acceptance's p50 0.34 / p90 0.96 cm; the anchor's smoothing was 3-5% of the gap. The creep
  multiplier sat at its 3x cap against a 0.64 cm target on a p90 the lineage does not meet on that measure.

  Here: `settle_s` after the robot was last disturbed -- the stand began or it was pushed (the hold term's own
  clock, excused_since; a new episode restarts it) -- the base position is kept as the reference, and from then on
  value = the distance from it. It counts (mask) while the window since the reference is `min_window_s` ..
  `window_s` long: a shorter one has seen too little of the acceptance's 4 s, a longer one is past it. A moving
  command, a push or a new episode ends the window. The value is not scaled up to 4 s (a slide that decays reads
  nearly the same over 2 s as over 4; a steady drift reads up to a quarter low on average).

  env.u_stop_creep; the constraints read it as "u_stop_creep" (every robot) / "u_stop_creep_light".

  The march's drift, the same way (`key` "u_march_drift" on march_position_hold, `repeat`; 2026-11-12). The
  acceptance (scripts/blind/eval/march_eval.py) stands 3 s, marches 5 s and takes the net displacement over those
  5 s; the mean over the robots must stay under 5 cm. The earlier measure was the p90 of the distance from the
  march's anchor over the whole march (10-20 s) against 4 cm: at the 3x cap on every Go2W seed with 6-7 cm, while
  seed 1's model_2600 passed the acceptance's march with 3.2 cm (Go2 model_3300: 2.5 cm over 5 s, 9.6 cm over
  20 s -- a steady drift, about 0.5 cm/s, no start transient). With `repeat` the windows are chained: a new
  reference every `window_s`, each window counted over its last `window_s - min_window_s`. The first reference is
  taken `settle_s` after the march began or the last push, as the hold term's own excuse (its push_window): a
  march here may begin out of a walk, the acceptance's begins out of a stand.

  `key`: the attribute on env this instance is kept under (params["key"]; "u_stop_creep" without one)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    n, dev = env.num_envs, env.device
    self.ref = torch.zeros(n, 2, device=dev)
    self.has_ref = torch.zeros(n, dtype=torch.bool, device=dev)
    self.k = torch.zeros(n, dtype=torch.long, device=dev)       # which window the reference belongs to (repeat)
    self.value = torch.zeros(n, device=dev)                     # distance from the reference (m)
    self.mask = torch.zeros(n, dtype=torch.bool, device=dev)    # the robots whose value counts now
    setattr(env, (getattr(cfg, "params", None) or {}).get("key") or "u_stop_creep", self)

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.has_ref[env_ids] = False
    self.mask[env_ids] = False

  def __call__(self, env: ManagerBasedRlEnv, term: str = "standing_position_hold", settle_s: float = 2.0,
               window_s: float = 4.0, min_window_s: float = 2.0, repeat: bool = False,
               key: str | None = None) -> torch.Tensor:
    del key   # (read in __init__)
    cfg = env.reward_manager.get_term_cfg(term)
    assert cfg.params.get("push_window") is not None, f"stop_creep: {term} keeps no clock without push_window"
    t = cfg.func
    standing = t.was_standing   # at the zero command / marching now (the hold term ran earlier in this step)
    # in env steps (whole numbers: the windows' edges fall on the same step whatever the rounding)
    settle_n, window_n, min_n = (round(x / env.step_dt) for x in (settle_s, window_s, min_window_s))
    w = env.common_step_counter - t.excused_since - settle_n   # since the reference was due (the stand began / a push)
    pos = env.scene["robot"].data.root_link_pos_w[:, :2]
    settled = standing & (w >= 0)
    k = torch.div(w.clamp(min=0), window_n, rounding_mode="floor") if repeat else torch.zeros_like(self.k)
    take = settled & (~self.has_ref | (k != self.k))
    self.ref = torch.where(take.unsqueeze(1), pos, self.ref)
    self.k = torch.where(take, k, self.k)
    self.has_ref = settled
    phase = w - k * window_n
    self.value = torch.norm(pos - self.ref, dim=1)
    self.mask = settled & (phase >= min_n) & (phase <= window_n)
    return torch.zeros(env.num_envs, device=env.device)


class stop_place:
  """Reward term (U_STOP_PLACE=<w>[,<free>[,<full>[,<heavy>]]], go2-depth 2026-11-14): a robot told to stop on a
  flight stays where it was told.

  Why. The Go2W's formal teachers from zero (84e7965) answer the zero command on a climbing 12-18 cm flight by
  rolling back down to the start platform (seed 0, 72 robots a band: model_2600 29 hold / 34 leave on 12-15 cm and
  23 / 43 on 15-18 cm; model_4400 14 / 55 and 0 / 72, 1.2-1.7 m back inside 2 s), and nothing charges it:
  standing_position_hold's anchor follows the base for up to 2 s after the stand begins (its push_window, the
  braking's excuse), standing_pose_load waits for that anchor to lock, wheel_slide_back does not count at a zero
  command, and on the platform every stop reading is perfect -- four wheels down (stop_feet), no speed, no creep.
  Tier 2's stop run read the displacement between 2 s and 4 s after the command only, so the faster the retreat,
  the better it read (84% -> 92% "stop ok"). The probe lineage's s51 model_7991 holds on the same flights (69 of 72
  on 12-15 cm): it can be done. Keeping the flat items' hold terms strong on the flights (U_FLAT_ONLY_KEEP, s59)
  did not stop the retreat and tripled the falls; the standing pose cost off the flights (U_STAND_POSE_EVEN, s61)
  changed nothing.

  Here: when the velocity command falls to zero (a stand, or a march in place) over uneven ground
  (mdp.uneven_terrain_mask on `sensor_name`: a step within 0.45 m), the base position at that step is the reference
  for the whole stand -- no following, no settling time. Per second: nothing within `free` of it (0.30 m, a tread:
  the braking, a wheel rolling onto the tread below), 1 from `full` on (0.60 m), linear in between, until the
  command moves again -- wherever the robot is by then (the ground under it now is not asked: the platform it
  retreated to counts). A push during the stand ends the cost for that stand (the robot was moved; it is not asked
  back). In full up to the common payload (COMMON_FRAC of the limit), `heavy` times it from the heavy class
  (HEAVY_FRAC) up, linear in between, as stairs_park: the acceptance judges the stops unloaded and at the common
  payload and the full payload on its falls alone -- a heavy robot may leave the flight (`heavy` 0).

  The reading (env.u_stop_place; the constraints' "u_stop_place" / "u_stop_place_heavy", items
  stop_place_{up,down}[_heavy]): `near` = within `stay` of the reference (0.50 m, tier 2's bar), counted (`mask`)
  while the stand is `read_from` .. `read_to` old (2-4 s; a stand on a stairs column lasts 1.5-4 s) -- `up` / `down`
  say which way the nose pointed when the stand began (the pitch by `min_pitch` or more), so a robot back on level
  ground still counts for the flight it left. Unlike U_STOP_STILL's reading (the robots still pitched 3.5-4 s into
  the stand: 1.0 on a checkpoint whose robots had all left)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    n, dev = env.num_envs, env.device
    self.ref = torch.zeros(n, 2, device=dev)                    # the base position when the stand began
    self.was = torch.zeros(n, dtype=torch.bool, device=dev)     # at the zero command at the last step
    self.on = torch.zeros(n, dtype=torch.bool, device=dev)      # this stand began over uneven ground, no push since
    self.up = torch.zeros(n, dtype=torch.bool, device=dev)      # nose up when the stand began
    self.down = torch.zeros(n, dtype=torch.bool, device=dev)    # nose down when the stand began
    self.since = torch.zeros(n, dtype=torch.long, device=dev)   # the step the stand began
    self.seen_push = None                                       # events' push stamp at the last step
    self.dist = torch.zeros(n, device=dev)                      # distance from the reference (m)
    self.near = torch.zeros(n, device=dev)                      # 1 within `stay` of the reference
    self.mask = torch.zeros(n, dtype=torch.bool, device=dev)    # the robots whose reading counts now
    env.u_stop_place = self

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.was[env_ids] = False
    self.on[env_ids] = False
    self.mask[env_ids] = False

  def __call__(self, env: ManagerBasedRlEnv, command_name: str, sensor_name: str, command_threshold: float = 0.1,
               free: float = 0.30, full: float = 0.60, heavy: float = 0.0, stay: float = 0.50,
               min_pitch: float = 0.05, read_from: float = 2.0, read_to: float = 4.0) -> torch.Tensor:
    from src.tasks.velocity.mdp.rewards import uneven_terrain_mask

    command = env.command_manager.get_command(command_name)
    zero = (torch.norm(command[:, :2], dim=1) + command[:, 2].abs()) <= command_threshold
    data = env.scene["robot"].data
    pos = data.root_link_pos_w[:, :2]
    step = int(env.common_step_counter)
    new = zero & ~self.was
    self.was = zero.clone()
    gx = data.projected_gravity_b[:, 0]
    self.ref = torch.where(new.unsqueeze(1), pos, self.ref)
    self.since = torch.where(new, torch.full_like(self.since, step), self.since)
    self.on = torch.where(new, uneven_terrain_mask(env, sensor_name), self.on) & zero
    self.up = torch.where(new, gx <= -min_pitch, self.up)
    self.down = torch.where(new, gx >= min_pitch, self.down)
    stamp = getattr(env, "_push_step", None)
    if stamp is not None:
      if self.seen_push is not None:
        self.on &= ~(stamp > self.seen_push)
      self.seen_push = stamp.clone()
    self.dist = torch.norm(pos - self.ref, dim=1)
    self.near = (self.dist <= stay).float()
    age = (step - self.since) * env.step_dt
    self.mask = self.on & (age >= read_from) & (age <= read_to)
    out = self.on.float() * ((self.dist - free) / (full - free)).clamp(min=0.0, max=1.0)
    kg = getattr(env, "payload_kg", None)
    if kg is not None:
      frac = (kg.float() / payload_limit(env) - COMMON_FRAC) / (HEAVY_FRAC - COMMON_FRAC)
      out = out * (1.0 - (1.0 - heavy) * frac.clamp(0.0, 1.0))
    return out


class march_reset:
  """Reward term with zero output (a tiny nonzero weight): whether a march sets a splayed stance back, the way the
  acceptance measures it (go2-depth 2026-11-12).

  The acceptance (scripts/blind/eval/march_eval.py) turns every hip 8 deg in or out, stands 3 s, marches 5 s and
  reads the worst hip's distance from its default at the end: the mean over the robots must stay under 2 deg
  ("没复位" above it). Until now no measure in the training read it: the hips' cost (march_hip_default) hung on the
  march item's drift reading. On the acceptance's test the Go2 teacher at 3300 read 2.4 deg (5.4 before the march),
  the Go2W seed 1 at 2600 1.2 deg (7.4 before).

  Here: when a march begins, the worst hip's distance from its default is kept (`start`); the robots that began with
  at least `min_start` (the stance had something to set back: a reset with splayed hips, events' reset_hip_splay,
  then a stand or a march at once) count `after_s` .. `until_s` into the march, value = the worst hip's distance
  then (rad). A march out of a walk begins with the hips near their default and is not counted: it would read the
  march's own hip motion, not the reset. The hips are `term`'s (the cost this item scales).

  env.u_march_reset; the constraints read it as "u_march_hip" (every robot) / "u_march_hip_light"."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    n, dev = env.num_envs, env.device
    self.was = torch.zeros(n, dtype=torch.bool, device=dev)     # marching at the last step
    self.since = torch.zeros(n, dtype=torch.long, device=dev)   # the env step the march began at
    self.start = torch.zeros(n, device=dev)                     # the worst hip's distance from default then (rad)
    self.value = torch.zeros(n, device=dev)                     # the worst hip's distance from default now (rad)
    self.mask = torch.zeros(n, dtype=torch.bool, device=dev)    # the robots whose value counts now
    env.u_march_reset = self

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.was[env_ids] = False
    self.mask[env_ids] = False

  def __call__(self, env: ManagerBasedRlEnv, term: str = "march_hip_default", command_name: str = "twist",
               after_s: float = 4.0, until_s: float = 5.0, min_start: float = 0.05) -> torch.Tensor:
    acfg = env.reward_manager.get_term_cfg(term).params["asset_cfg"]
    data = env.scene[acfg.name].data
    off = (data.joint_pos[:, acfg.joint_ids] - data.default_joint_pos[:, acfg.joint_ids]).abs().amax(dim=1)
    marching = env.command_manager.get_term(command_name).march
    new = marching & ~self.was
    step = env.common_step_counter
    self.since = torch.where(new, torch.full_like(self.since, step), self.since)
    self.start = torch.where(new, off, self.start)
    self.was = marching.clone()
    n = step - self.since
    self.value = off
    self.mask = (marching & (self.start >= min_start) & (n >= round(after_s / env.step_dt))
                 & (n <= round(until_s / env.step_dt)))
    return torch.zeros(env.num_envs, device=env.device)


def _tracker(env: ManagerBasedRlEnv, key: str, term: str) -> tuple[torch.Tensor, torch.Tensor]:
  t = getattr(env, key, None)
  if t is None:
    raise RuntimeError(f"{key}: the {term} reward term is not in this environment (teacher_env._rewards)")
  return t.value, t.mask


def _light(value: torch.Tensor, mask: torch.Tensor, env: ManagerBasedRlEnv,
           max_frac: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  """`mask` over the light robots (<= LIGHT_FRAC of the payload limit): the acceptance's flat tests carry none."""
  pm = payload_mask(env, hi=LIGHT_FRAC if max_frac is None else max_frac)
  return value, (mask if pm is None else mask & pm)


def _stop_creep(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """`stop_creep`'s reading: the distance from where the robot stood 2 s after it was last disturbed, over the
  robots 2-4 s past that point."""
  return _tracker(env, "u_stop_creep", "stop_creep")


def _stop_creep_light(env: ManagerBasedRlEnv, max_frac: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  """The same over the light robots (<= LIGHT_FRAC of the payload limit): the acceptance's stop test carries none."""
  return _light(*_stop_creep(env), env, max_frac)


def _march_drift(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """The march's drift (`stop_creep` on march_position_hold, chained 5 s windows): the distance from where the
  robot was at the window's start, over the robots in the last second of a window."""
  return _tracker(env, "u_march_drift", "stop_creep (key u_march_drift)")


def _march_drift_light(env: ManagerBasedRlEnv, max_frac: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  return _light(*_march_drift(env), env, max_frac)


def _march_hip(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """`march_reset`'s reading: the worst hip's distance from its default (rad) 4-5 s into a march that began with
  the stance splayed."""
  return _tracker(env, "u_march_reset", "march_reset")


def _march_hip_light(env: ManagerBasedRlEnv, max_frac: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  return _light(*_march_hip(env), env, max_frac)


def _stairs_stop_feet(env: ManagerBasedRlEnv, sensor_name: str = "feet_ground_contact", max_speed: float = 0.1,
                      facing: str | None = None, min_pitch: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
  """All four feet / wheels down (the acceptance's stairs stop-feet, bar 0.9 of the robots), over the robots on a
  stairs column at the zero command that have come to rest (base speed below max_speed).

  `facing` "down" / "up": only the robots stopped on the flight nose down / nose up (the body pitched by min_pitch
  or more, as mdp.constraints' stair_speed tells a descent from a climb). Without it the reading pools both
  directions and the robots stopped on the level ground of a stairs column, which always stand on four feet: the Go2
  candidate teacher (2026-11-13, s29 model_5598) read 0.93 in training -- above the bar, its multiplier relaxed to
  0.5 -- while tier 2 found four feet down in 0.97 / 0.75 / 0.66 / 0.57 of the stops going down the four step-height
  bands (0.99-1.00 going up)."""
  from mjlab.sensor import ContactSensor

  assert facing in (None, "up", "down"), facing
  sensor: ContactSensor = env.scene[sensor_name]
  value = (sensor.data.found > 0).all(dim=1).float()
  command = env.command_manager.get_command("twist")
  data = env.scene["robot"].data
  still = (command.abs().sum(dim=1) == 0.0) & (torch.norm(data.root_link_lin_vel_b[:, :2], dim=1) < max_speed)
  mask = env.command_manager.get_term("twist").is_stair_env & still
  if facing is not None:
    gx = data.projected_gravity_b[:, 0]
    mask = mask & (gx >= min_pitch if facing == "down" else gx <= -min_pitch)
  return value, mask


def _stairs_stop_still(env: ManagerBasedRlEnv, facing: str = "up", min_pitch: float = 0.05,
                       max_drift: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
  """Whether a robot stopped on a flight holds still, as tier 2's stop run judges it (go2-depth 2026-11-13).

  The stop run (scripts/go2_blind/eval_stairs.py) gives the zero command in the middle of a flight and takes the
  displacement between 2 s and 4 s after it; "stop ok" needs less than 5 cm, the gate 95% of the robots. Nothing in
  the training read that: stop_feet counts only the robots that have come to rest (base speed below 0.1 m/s), the
  creep item reads on even ground under U_FLAT_ONLY, the park items read a standstill under a command to move. The
  Go2W's formal teachers from zero (84e7965) read 82-85% on tier 2 at 2600-3300 iterations -- only the climbing
  8-18 cm bands (still moving 8-39 cm), and with the full payload falls 2-4 s after a stop going down -- with every
  training reading as the probe lineage's, which stops at 99%.

  Here: `stop_creep` a second time (env.u_stairs_stop: the reference 2 s after the stand began, counted 3.5-4 s after
  it began), value 1 where the distance from the reference is below `max_drift`, over the robots on a stairs column
  whose body is pitched by `min_pitch` or more -- `facing` "up" nose up, "down" nose down (as _stairs_stop_feet).

  Not in it: a robot that fell inside the window (its episode ended; tier 2 counts it "not ok") and one that rolled
  back onto level ground (no longer pitched). The falls at a stop are in the episode records: run_report.py's
  per-ground table, the failed episodes' command speed "<0.1m/s" with the payload class and the pitch at the end."""
  t = getattr(env, "u_stairs_stop", None)
  if t is None:
    raise RuntimeError("u_stairs_stop: U_STOP_STILL's stop_creep term is not in this environment (teacher_env._rewards)")
  assert facing in ("up", "down"), facing
  gx = env.scene["robot"].data.projected_gravity_b[:, 0]
  pitched = gx >= min_pitch if facing == "down" else gx <= -min_pitch
  return (t.value < max_drift).float(), t.mask & env.command_manager.get_term("twist").is_stair_env & pitched


def _stop_place(env: ManagerBasedRlEnv, facing: str = "up") -> tuple[torch.Tensor, torch.Tensor]:
  """`stop_place`'s reading: 1 for a robot within 0.50 m of where its stand began, over the stands that began on a
  flight of a stairs column nose up (`facing` "up") / nose down ("down"), 2-4 s into the stand -- wherever the robot
  is by then (tier 2's stop run from the next freeze on: the displacement from the zero command to 4 s after it)."""
  t = getattr(env, "u_stop_place", None)
  if t is None:
    raise RuntimeError("u_stop_place: U_STOP_PLACE's term is not in this environment (teacher_env._add_stop_place)")
  assert facing in ("up", "down"), facing
  return t.near, t.mask & env.command_manager.get_term("twist").is_stair_env & (t.up if facing == "up" else t.down)


def _yaw_frame_velocity(env: ManagerBasedRlEnv, asset_name: str = "robot") -> torch.Tensor:
  """The base's velocity in the frame that turns with its heading only (x ahead, y to the left, z up): the ground it
  covers. The body frame's x of a base pitched along a flight is the distance along the slope -- at 30 deg 15% more
  than the ground covered, which is what the acceptance's flight speed (0.8-1.2x the command) is measured in."""
  from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

  data = env.scene[asset_name].data
  return quat_apply_inverse(yaw_quat(data.root_link_quat_w), data.root_link_lin_vel_w)


def _stair_speed_h(env: ManagerBasedRlEnv, **params) -> tuple[torch.Tensor, torch.Tensor]:
  """mdp.constraints' stair_speed (forward speed / command over the robots pitched on uneven ground under a forward
  command) with the speed over the ground instead of the body frame's (_yaw_frame_velocity); the same robots.
  2026-11-13: the candidate teachers read 0.9 and more in training and 0.73-0.81x (down) / 0.82-0.91x (up) on tier
  2's flights."""
  _, mask = METRICS["stair_speed"](env, **params)
  command = env.command_manager.get_command(params.get("command_name", "twist"))
  vx = _yaw_frame_velocity(env)[:, 0]
  return vx / torch.where(mask, command[:, 0], torch.ones_like(vx)), mask


def up_column(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per robot: it is on a stairs column that climbs (curriculum.UP_NAMES); all False without such columns."""
  cols = getattr(env, "_u_up_columns", None)
  if cols is None:
    from src.tasks.unified.curriculum import UP_NAMES
    from src.tasks.velocity.mdp.terrain_utils import sub_terrain_column_mask

    cols = sub_terrain_column_mask(env, UP_NAMES)
    if cols is None:
      return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._u_up_columns = cols
  return cols[env.scene.terrain.terrain_types]


def _stairs_park(env: ManagerBasedRlEnv, command_name: str = "twist", sensor_name: str = "terrain_scan",
                 min_vx: float = 0.2, slow: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
  """1 for a robot that stands where it is told to walk on (its speed over the ground along x below `slow` x the
  command), over the robots on the level ground of a climbing column -- before the flight, or on its landing --
  under a forward command (vx >= min_vx, |vy| <= 0.1). A robot that has just been told to walk reads 1 until it is
  up to speed, so a healthy policy does not read 0.

  2026-11-13, the Go2W package teachers (uprobe_s32 -> s34 -> s37, iterations 4997 -> 5995): tier 2's seen mode went
  from no robot stalled to 19% (the 12-15 cm climbs 54%, the 15-18 cm ones 92%; reach 0.96 / 0.76 m: before the
  first riser), with no fall -- and no reading of the training said so. The climb item reads the robots pitched on
  the steps (0.82-0.91, at its target), the survival item counts the fails (0.25 -> 0.06: a robot that does not climb
  does not fall), and a robot that walks less than half its commanded distance goes down a row, so the rows' own
  readings stayed good while up_level sank (6.56 -> 6.19). scripts/unified/up_stall_probe.py on s37: 74% of the
  steps under a forward command before the first riser at |v| < 0.1 m/s, 90% of them within 0.5 m of the spawn."""
  from src.tasks.velocity.mdp.rewards import uneven_terrain_mask

  command = env.command_manager.get_command(command_name)
  vx = _yaw_frame_velocity(env)[:, 0]
  mask = (command[:, 0] >= min_vx) & (command[:, 1].abs() <= 0.1) & up_column(env)
  mask &= ~uneven_terrain_mask(env, sensor_name)
  return (vx < slow * command[:, 0]).float(), mask


def down_column(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per robot: it is on a stairs column that descends (curriculum.DOWN_NAMES); all False without such columns."""
  cols = getattr(env, "_u_down_columns", None)
  if cols is None:
    from src.tasks.unified.curriculum import DOWN_NAMES
    from src.tasks.velocity.mdp.terrain_utils import sub_terrain_column_mask

    cols = sub_terrain_column_mask(env, DOWN_NAMES)
    if cols is None:
      return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._u_down_columns = cols
  return cols[env.scene.terrain.terrain_types]


def _stairs_park_down(env: ManagerBasedRlEnv, command_name: str = "twist", min_vx: float = 0.2,
                      slow: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
  """_stairs_park for the descents: 1 for a robot that stands where it is told to walk on (its speed over the ground
  along x below `slow` x the command), over the robots on a descending column under a forward command (vx >= min_vx,
  |vy| <= 0.1) -- on the top platform, on the flight or past it. A robot that has just been told to walk reads 1
  until it is up to speed, so a healthy policy does not read 0 (the Go2W candidates, on the climbs: 0.03).

  2026-11-13, the Go2 candidates (uprobe_s47 -> s51 -> s52, iterations 6596 -> 7594): tier 2's seen mode went from
  no robot stalled to 68% / 100% / 100% of the 8-12 / 12-15 / 15-18 cm descents (reach 0.76-0.90 m: the edge of the
  top platform), unloaded and at the common payload, with one fall -- the blind mode, which cannot see the edge,
  walked every descent. In training the descent item read it only as a slower speed over the robots pitched on the
  steps (seen 0.86 -> 0.65 -> 0.29, its multiplier at the cap 1), and the survival item improved (a robot that does
  not descend does not fall)."""
  command = env.command_manager.get_command(command_name)
  vx = _yaw_frame_velocity(env)[:, 0]
  mask = (command[:, 0] >= min_vx) & (command[:, 1].abs() <= 0.1) & down_column(env)
  return (vx < slow * command[:, 0]).float(), mask


def stairs_park(env: ManagerBasedRlEnv, command_name: str, min_vx: float = 0.2, free: float = 0.5,
                heavy: float = 0.0) -> torch.Tensor:
  """Reward term (U_STAIR_PARK=<w>[,<heavy>], 2026-11-13): standing on a descending column under a command to walk
  on. Per robot on such a column (down_column: the top platform, the flight, the ground past it) under a forward
  command (vx >= min_vx): 1 at a standstill or moving against the command, falling linearly to 0 at `free` x the
  commanded speed over the ground along the command -- nothing for a robot that walks, however carefully. In full
  up to the common payload (COMMON_FRAC of the limit), `heavy` times it from the heavy class (HEAVY_FRAC) up, linear
  in between: the acceptance wants every descent unloaded and at the common payload (reach, speed, at most 5%
  stalled) and judges the full payload on its falls alone, and the user (2026-11-12): with payload the robot must
  never fall, what is physically out of reach with it is not required -- so a heavy robot may wait on the platform
  (`heavy` 0), where it stands on its four feet.

  Why (S/probe/down_stall_probe.py on uprobe_s52 model_7594, seen mode, rows 5-8, light robots, per second): standing
  before the first step down +0.26 (the stairs speed cost -1.75: it is charged there already, the edge is within its
  0.45 m; the turn / pose / gait rewards a standing robot still earns), walking towards it +0.53, descending -3.34
  (riser touches -1.25, line drift -1.31, leg tilt -0.98, edge placements -0.60, uneven support -0.45, leg contacts
  -0.31, slide-backs -0.23, ...; the speed cost -0.49 even at 0.87x), and on rows 7-9 a fifth of the descents end
  the episode (-40). Standing pays 3.6 a second, the descent item's multiplier -- the one thing that pushes back --
  is at U_Q_SCALE_MAX's cap of 1, and lifting that cap (U_ITEM_CAP=descent:<x>) raises the speed cost on every
  flight, the climbs with the full payload too (constraints._apply: the strongest weight). This cost charges the
  standstill alone; the item park_down scales it (teacher_env.constraint_items).

  The same account with the cost at -5 (the same checkpoint, which still parks: 70% of those steps at a standstill):
  standing -4.08 a second (the cost -4.34 of it), creeping towards the edge at under half the commanded speed -2.33
  (-2.80), against the descent's -3.3 and about -1 of the cost on its slow steps -- even, not decided. So the item's
  multiplier has a cap of its own (teacher_env.PARK_CAP), and the probe from the parked checkpoint runs at 10."""
  command = env.command_manager.get_command(command_name)
  cmd = command[:, :2]
  speed = torch.norm(cmd, dim=1)
  safe = speed.clamp(min=1e-6)
  along = (_yaw_frame_velocity(env)[:, :2] * (cmd / safe.unsqueeze(1))).sum(dim=1)
  out = (1.0 - along / (free * safe)).clamp(min=0.0, max=1.0)
  out = torch.where((command[:, 0] >= min_vx) & down_column(env), out, torch.zeros_like(out))
  kg = getattr(env, "payload_kg", None)
  if kg is not None:
    frac = (kg.float() / payload_limit(env) - COMMON_FRAC) / (HEAVY_FRAC - COMMON_FRAC)
    out = out * (1.0 - (1.0 - heavy) * frac.clamp(0.0, 1.0))
  return out


def com_offset(env: ManagerBasedRlEnv, asset_cfg) -> tuple[torch.Tensor, torch.Tensor]:
  """Per robot: how far the whole robot's centre of mass -- every body at its mass now, so the payload and the base's
  own CoM noise with it (mdp.base_payload_point writes the base's mass and CoM) -- lies from the centre of its four
  feet / wheels (`asset_cfg`'s sites) over the ground, in metres, and the feet's mean distance from their own centre
  (the size of the support: Go2 0.245 m, Go2W 0.25-0.29 m).

  From the bodies' masses and CoM positions themselves, so whatever the model's derived fields hold (S/probe/
  com_check.py: equal to MuJoCo's subtree_com of the base to the millimetre, with and without a payload)."""
  asset = env.scene[asset_cfg.name]
  mass = env.sim.model.body_mass[:, asset.indexing.body_ids]
  com = (asset.data.body_com_pos_w * mass.unsqueeze(-1)).sum(dim=1) / mass.sum(dim=1, keepdim=True)
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids][..., :2]
  centre = feet.mean(dim=1)
  size = torch.norm(feet - centre.unsqueeze(1), dim=-1).mean(dim=1)
  return torch.norm(com[:, :2] - centre, dim=1), size


def com_support(env: ManagerBasedRlEnv, asset_cfg, sensor_name: str, command_name: str = "twist",
                free: float = 0.12, span: float = 0.80, moving: float = 0.0) -> torch.Tensor:
  """Reward term (U_COM_SUPPORT=<w>[,<moving>], 2026-11-13): over uneven ground, the whole robot's centre of mass
  off the centre of its four feet / wheels (com_offset; over the ground, so along gravity). In units of the
  support's size: nothing within `free` of it (Go2: 3 cm), 1 from `free` + `span` on (23 cm), linear in between --
  a slope all the way from a robot at rest to one about to go over, so no flat part where the loaded ones stand. In
  full at the zero command, `moving` times it under a command (0: the stops alone). Even ground costs nothing: the
  flat items (creep, the stance, the march) stay as they are.

  What it answers (2026-11-13, tier 2 with the acceptance's payload at x +0.15 m, z +0.20 m): the three robots' falls
  with payload are one fall. Go2W uprobe_s44 6993: 19 loaded falls, 13 of them 2-4 s after a stop on a descent,
  forwards. Go2 uprobe_s58 8093: 29, 15 within 2 s of the stop, forwards. As2 mAH 2996 (15 kg): 93 / 57 / 32% of the
  stops on the 15-18 / 12-15 / 8-12 cm descents and 79 / 24 / 11% of the descents themselves, forwards (and a quarter
  of the 15-18 cm climbs backwards, sliding back first). Nothing in the recipe says where the weight should be: the
  costs of a stop are the speed, the position hold, the legs' tilt and the pose (standing_pose_load: each leg joint's
  distance from the default, times 1 + the payload's share -- at the Go2W's loaded stops nose down the largest cost,
  7.8 a second, S/probe/stand_stairs_terms.py), and the payload's mount point is drawn, so a pose that holds one
  payload tips another. Without the pose cost on the steps (U_STAND_POSE_EVEN) the Go2 fell more and earlier (72
  loaded falls against 20, 58 of them walking down or within 2 s of the stop): the pose cost stays, this one is
  added to it. Where the robots stand now (S/probe/com_check.py, unloaded, zero command): the Go2 (uprobe_s52 7594)
  nose down on a flight 6.5 cm ahead of the feet's centre -- the default pose on a slope: the trunk parallel to it,
  the legs square to the trunk -- and 3.6 cm behind it nose up; the Go2W (uprobe_s44 6993) 2.1 cm / 1.3 cm; both
  1.6 cm on even ground. The acceptance's payload (8 kg, 0.15 m ahead, 0.20 m up) moves the Go2's centre of mass
  another 5 cm ahead and 7 cm up, about 13 cm ahead of the feet's centre nose down in the same pose, with the front
  feet 19 cm ahead of it. The cost pays for bringing the base back over the feet, wherever the payload sits, and
  the unloaded robot for the same posture at a smaller price (0.18 of the weight at 6.5 cm).

  Stores each robot's distance (m) as env.u_com_offset for the logged readings (_com_stop)."""
  from src.tasks.velocity.mdp.rewards import uneven_terrain_mask

  dist, size = com_offset(env, asset_cfg)
  env.u_com_offset = dist
  out = ((dist / size.clamp(min=1e-3) - free) / span).clamp(0.0, 1.0)
  out = torch.where(uneven_terrain_mask(env, sensor_name), out, torch.zeros_like(out))
  if moving < 1.0:
    stand = env.command_manager.get_command(command_name).abs().sum(dim=1) == 0.0
    out = torch.where(stand, out, moving * out)
  return out


def _com_stop(env: ManagerBasedRlEnv, min_pitch: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
  """The centre of mass off the centre of the feet (m; com_support's env.u_com_offset, the step before), over the
  robots at the zero command on a flight of a stairs column (the body pitched by min_pitch or more, either way).
  No robot without the cost (the reading is logged with U_COM_SUPPORT alone)."""
  dist = getattr(env, "u_com_offset", None)
  command = env.command_manager.get_command("twist")
  mask = env.command_manager.get_term("twist").is_stair_env & (command.abs().sum(dim=1) == 0.0)
  mask = mask & (env.scene["robot"].data.projected_gravity_b[:, 0].abs() >= min_pitch)
  if dist is None:
    return torch.zeros(env.num_envs, device=env.device), torch.zeros_like(mask)
  return dist, mask


def stair_speed_deficit_h(env: ManagerBasedRlEnv, command_name: str, sensor_name: str, ratio: float = 0.9,
                          min_speed: float = 0.1, ratio_high: float | None = None, axis_min: float | None = None,
                          max_payload: float | None = None, approach_heavy: float | None = None) -> torch.Tensor:
  """Reward term: mdp.stair_speed_deficit on uneven ground (its docstring: how far the speed along the commanded
  direction falls short of `ratio` x the command, or exceeds `ratio_high` x, as a fraction of the command; per axis
  for diagonals with `axis_min`; the robots carrying at most `max_payload`), with the speed over the ground
  (_yaw_frame_velocity) instead of the body frame's. With the body frame's, the free band 0.9-1.1x is 0.78-0.95x
  of ground speed on a base pitched 30 deg: where tier 2 read the candidate teachers' descents (2026-11-13,
  0.73-0.81x; the acceptance wants 0.8-1.2x).

  `approach_heavy` (U_STAIR_APPROACH=<x>, 2026-11-13): the cost also holds on the level ground of a climbing column
  (up_column: before the flight and on its landing) under a forward command -- in full up to the common payload
  (COMMON_FRAC of the limit), `approach_heavy` times it from the heavy class (HEAVY_FRAC) up, linear in between.
  Without it the cost starts within 0.45 m of a step (uneven_terrain_mask), and a robot that stands before the
  flight pays the flat speed cost alone -- a third of this one. The Go2W package teacher learnt to stand there
  (_stairs_park's docstring); its rewards on rows 7-9 (up_stall_probe.py on uprobe_s37, per second): standing before
  the riser +1.46 (the flat speed cost -3.5, the turn / pose / gait rewards a standing robot still earns), climbing
  -1.91 (slide-backs -1.4, nosing -1.1, line drift -1.2, leg tilt -1.0, this cost -1.6 even at 0.92x, ...), standing
  on the flight -17.5. The acceptance wants every climb unloaded and at the common payload (reach, speed, at most 5%
  stalled) and judges the full payload on its falls alone, and the user (2026-11-12): with payload the robot must
  never fall, what is physically out of reach with it is not required -- so a heavy robot may wait on the level
  (`approach_heavy` 0), where it stands on its four feet; on the steps the cost holds for every payload."""
  from src.tasks.velocity.mdp.rewards import uneven_terrain_mask

  command = env.command_manager.get_command(command_name)
  cmd = command[:, :2]
  vel = _yaw_frame_velocity(env)[:, :2]
  speed = torch.norm(cmd, dim=1)
  safe = speed.clamp(min=1e-6)
  along = (vel * (cmd / safe.unsqueeze(1))).sum(dim=1)
  deficit = ((ratio * speed - along) / safe).clamp(min=0.0, max=1.0)
  if ratio_high is not None:
    deficit = deficit + ((along - ratio_high * speed) / safe).clamp(min=0.0, max=1.0)
  if axis_min is not None:
    part = cmd.abs()
    moved = vel * torch.sign(cmd)
    axes = ((ratio * part - moved) / part.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
    if ratio_high is not None:
      axes = axes + ((moved - ratio_high * part) / part.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
    deficit = torch.where((part >= axis_min).all(dim=1), axes.sum(dim=1), deficit)
  uneven = uneven_terrain_mask(env, sensor_name)
  moving = speed >= min_speed
  kg = getattr(env, "payload_kg", None)
  if max_payload is not None and kg is not None:
    moving = moving & (kg <= max_payload)
  out = torch.where(moving & uneven, deficit, torch.zeros_like(deficit))
  if approach_heavy is not None:
    level = moving & ~uneven & (cmd[:, 0] >= min_speed) & up_column(env)
    share = torch.ones_like(deficit)
    if kg is not None:
      frac = (kg.float() / payload_limit(env) - COMMON_FRAC) / (HEAVY_FRAC - COMMON_FRAC)
      share = 1.0 - (1.0 - approach_heavy) * frac.clamp(0.0, 1.0)
    out = torch.where(level, deficit * share, out)
  return out


def _stairs_roll(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """|Body roll| (rad, from the projected gravity) over the robots on a stairs column under a moving command (the
  acceptance records the roll peak on the stairs; logged, not a constraint: its bar is not calibrated)."""
  g = env.scene["robot"].data.projected_gravity_b
  roll = torch.asin(torch.clamp(g[:, 1], -1.0, 1.0)).abs()
  moving = env.command_manager.get_command("twist").abs().sum(dim=1) > 0.0
  return roll, env.command_manager.get_term("twist").is_stair_env & moving


def _stairs_heading(env: ManagerBasedRlEnv, obs_term: str = "heading_error",
                    drift_term: str = "straight_drift") -> tuple[torch.Tensor, torch.Tensor]:
  """How far the base has turned off the heading its command holds (rad): |the heading observation's target - the
  base heading|, over the robots on a stairs column that walk straight on (forward or backward, no sideways or yaw
  part) with steps under them (the drift cost's uneven-ground mask, where it has one). The acceptance's "heading off
  the flight" (tinynav, the wheeled robots, mean <= 3 deg = 0.052 rad; accept_stairs.py) is against the flight's axis
  from a spawn within 3 deg of it; here the reference is the commanded heading -- the observation's target: the
  heading at the reset plus the commanded turns, kept within its max_error of the base heading, so a reading
  saturates there (0.5 rad). Logged only (2026-11-13): the training records had no heading reading at all, the
  delivered Go2W reads 3.1 deg in tinynav, and how this reading relates to tinynav's is not calibrated. No reading in
  an environment without the observation."""
  from mjlab.utils.lab_api.math import wrap_to_pi

  from src.tasks.velocity.mdp.rewards import uneven_terrain_mask

  heading = env.scene["robot"].data.heading_w
  try:
    observed = env.observation_manager.get_term_cfg("critic", obs_term).func
  except ValueError:
    return torch.zeros_like(heading), torch.zeros_like(heading, dtype=torch.bool)
  off = torch.abs(wrap_to_pi(observed.target - heading))
  command = env.command_manager.get_command("twist")
  straight = (command[:, 0].abs() > 0.0) & (command[:, 1] == 0.0) & (command[:, 2] == 0.0)
  mask = env.command_manager.get_term("twist").is_stair_env & straight
  rm = env.reward_manager
  sensor = rm.get_term_cfg(drift_term).params.get("uneven_sensor_name") if drift_term in rm._term_names else None
  if sensor is not None:
    mask = mask & uneven_terrain_mask(env, sensor)
  return off, mask


def _flat_heading(env: ManagerBasedRlEnv, obs_term: str = "heading_error", drift_term: str = "straight_drift",
                  min_speed: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
  """_stairs_heading on even ground: |the heading observation's target - the base heading| (rad) over the robots
  under a command without a yaw part and with a planar speed of at least `min_speed`, off the drift cost's
  uneven-ground mask (off the stairs columns where it has none). Logged only (2026-11-13): tier 2 read the Go2
  teacher's heading 10 s into a line to the right at -12.6 / -17.3 deg (median, seen / blind; to the left -0.6 /
  -1.2) while the training's flat bins pool left with right; this is the error the policy itself is shown, on every
  straight command, not the lines' own (flat.py measures those from each line's start)."""
  from mjlab.utils.lab_api.math import wrap_to_pi

  from src.tasks.velocity.mdp.rewards import uneven_terrain_mask

  heading = env.scene["robot"].data.heading_w
  try:
    observed = env.observation_manager.get_term_cfg("critic", obs_term).func
  except ValueError:
    return torch.zeros_like(heading), torch.zeros_like(heading, dtype=torch.bool)
  off = torch.abs(wrap_to_pi(observed.target - heading))
  command = env.command_manager.get_command("twist")
  mask = (command[:, 2] == 0.0) & (torch.norm(command[:, :2], dim=1) >= min_speed) & (observed.settle == 0)
  rm = env.reward_manager
  sensor = rm.get_term_cfg(drift_term).params.get("uneven_sensor_name") if drift_term in rm._term_names else None
  if sensor is not None:
    mask = mask & ~uneven_terrain_mask(env, sensor)
  else:
    mask = mask & ~env.command_manager.get_term("twist").is_stair_env
  return off, mask


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
COMMON_FRAC = 0.5   # the acceptance's common payload: half the limit (doc/redesign.md 2.1; Go2 / Go2W 4 kg of 8)
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


def upto_common(metric):
  """`metric` (a name in METRICS, or a function like them) over the robots carrying at most the common payload
  (COMMON_FRAC of the limit): the stairs speed items under U_STAIR_SPEED_LOADED, where the heavier robots' speed cost
  is discounted (stair_speed_loaded) -- the acceptance judges the stairs speed unloaded and at the common payload,
  and at the full payload only the falls."""
  def f(env: ManagerBasedRlEnv, **params) -> tuple[torch.Tensor, torch.Tensor]:
    value, mask = (METRICS[metric] if isinstance(metric, str) else metric)(env, **params)
    pm = payload_mask(env, hi=COMMON_FRAC)
    return value, (mask if pm is None else mask & pm)
  return f


def over_heavy(metric):
  """`metric` (a function like METRICS') over the robots of the heavy class (HEAVY_FRAC of the limit and more)."""
  def f(env: ManagerBasedRlEnv, **params) -> tuple[torch.Tensor, torch.Tensor]:
    value, mask = metric(env, **params)
    pm = payload_mask(env, lo=HEAVY_FRAC)
    return value, (mask if pm is None else mask & pm)
  return f


CUSTOM = {"u_stairs_stop_feet": _stairs_stop_feet, "u_stairs_roll": _stairs_roll, "u_stairs_heading": _stairs_heading,
          "u_flat_heading": _flat_heading,
          "u_stair_speed_h": _stair_speed_h,
          "u_stairs_park": upto_common(_stairs_park), "u_stairs_park_heavy": over_heavy(_stairs_park),
          "u_stairs_park_down": upto_common(_stairs_park_down),
          "u_stairs_park_down_heavy": over_heavy(_stairs_park_down),
          "u_com_stop": upto_common(_com_stop), "u_com_stop_heavy": over_heavy(_com_stop),
          "u_stairs_stop_still": upto_common(_stairs_stop_still),
          "u_stairs_stop_still_heavy": over_heavy(_stairs_stop_still),
          "u_stop_place": upto_common(_stop_place), "u_stop_place_heavy": over_heavy(_stop_place),
          "u_stair_speed_c": upto_common("stair_speed"), "u_stair_speed_hc": upto_common(_stair_speed_h),
          "u_stand_hold_light": _stand_hold_light,
          "u_stop_creep": _stop_creep, "u_stop_creep_light": _stop_creep_light,
          "u_march_drift": _march_drift, "u_march_drift_light": _march_drift_light,
          "u_march_hip": _march_hip, "u_march_hip_light": _march_hip_light,
          # the flat items on the light robots (2026-11-11, the user: "现有的轻载约束 ... 只统计 ≤ 3 kg，继续对着无负载的验收门槛")
          "u_drift_light": by_payload("drift_share", hi=LIGHT_FRAC),
          "u_spin_light": by_payload("spin_hold", hi=LIGHT_FRAC),
          "u_back_drift_light": by_payload("back_drift", hi=LIGHT_FRAC)}

# The same items on the heavy robots: measured and logged (Curriculum/u_heavy/<item>/{light,heavy}), never enforced.
HEAVY_ITEMS = {
  # (creep: the acceptance's time base and statistic since 2026-11-12, as the constraint; before: stand_hold, p90)
  "creep": ("u_stop_creep", {}, "mean", ["standing_position_hold", "standing_speed"]),
  # (march: the same since 2026-11-12 -- the drift over 5 s and, as its own item, the hips' reset; before: one item,
  # stand_hold on the march's anchor, p90, with both terms)
  "march": ("u_march_drift", {}, "mean", ["march_position_hold"]),
  "march_hip": ("u_march_hip", {}, "mean", ["march_hip_default"]),
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


class flat_only:
  """Reward term (weight 1, after the terms it reads and the heavy groups): a flat item's multiplier presses on flat
  ground only (go2-depth 2026-11-13; the probe switch U_FLAT_ONLY=1, teacher_env._rewards).

  A multiplier scales its terms' weights for every robot, wherever it walks, and the flat items' terms run on the
  stairs too. The Go2 teacher's flat line item sat at its 3x cap (5.6% read against 5%), and on the climbing flights
  straight_drift then cost 1.51 a second -- as much as the slide, edge, leg-tilt and kick costs together (1.56) --
  with the climb netting -0.09 a second against +1.24 walking up to the stairs (up_stall_probe.py on uprobe_s29's
  model_5598). The flat items are measured on flat ground: that is where their pressure belongs.

  For a robot over uneven ground (mdp.uneven_terrain_mask on `sensor_name`: the height span under the base, the mask
  of straight_drift's own stairs part) this term adds raw x (calm - current) for each term a flat item presses on.
  `calm` (constraints.calm) is the term's weight with every flat item's multiplier held at 1 or below: a multiplier
  that relaxed below 1 stays relaxed everywhere, and a stairs-quality item's weight on the same term stands. The
  heavy robots (>= `min_frac` of the payload limit) are left out on `heavy_terms`: their own group (heavy_group)
  holds those terms at the configured weight already.

  `zero_terms` (the probe switch U_STAND_POSE_EVEN=1, 2026-11-13): these terms leave the uneven ground altogether,
  for every robot -- raw x (0 - current). standing_pose_load (the leg joints' distance from the default pose of a
  robot at rest, times 1 + payload / 8 kg) is the flat stance item's cost and has no terrain mask: on a flight, where
  a level stance has the front and rear legs far from the default, it was the largest cost of a stop -- 4.2 a second
  nose up and 5.4 nose down with the full payload, 1.7 / 2.1 with none (stand_stairs_terms on uprobe_s41's
  model_6097; a stop on even ground earns about +4 a second, on the stairs -0.3 to -9.3) -- and more than half of
  the full payload's falls in tier 2 were stops on a descent, 0-4 s after the stop.

  `keep_terms` (the probe switch U_FLAT_ONLY_KEEP=<term>,<term>, 2026-11-13): these terms keep their items'
  multipliers over uneven ground -- nothing is added back for them. Why: the Go2W's formal teachers from zero
  (84e7965, ufull_s0 / s1 at 2600-3300 iterations) fail tier 2's stairs test on the stop in the middle of a flight --
  12-18 cm steps going up: still moving 2-4 s after the zero command (8-39 cm in those 2 s, "stop ok" 82-85% against
  95%), and with the full payload falls 2-4 s after a stop going down (31 -> 87 and 11 -> 26 of 576 over 600
  iterations) -- where the probe lineage's teachers read 97-100% from uprobe_s22's model_4498 on. That lineage
  learned its stop before this term existed (it came in at 4498): until then the creep item's multiplier (3x) scaled
  standing_position_hold and standing_speed on the flights too; a teacher from zero under this term has them at 1x
  there, and no training reading counts the creep of a stop on a flight."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    pass

  def reset(self, env_ids=None) -> None:
    pass

  def __call__(self, env: ManagerBasedRlEnv, sensor_name: str, heavy_terms: list[str],
               min_frac: float = HEAVY_FRAC, zero_terms: tuple[str, ...] = (),
               keep_terms: tuple[str, ...] = ()) -> torch.Tensor:
    from src.tasks.velocity.mdp.rewards import uneven_terrain_mask
    out = torch.zeros(env.num_envs, device=env.device)
    calm = getattr(getattr(env, "u_constraints", None), "calm", {})
    if not calm and not zero_terms:
      return out
    uneven = uneven_terrain_mask(env, sensor_name)
    heavy = payload_mask(env, lo=min_frac)
    for t in zero_terms:   # off the uneven ground, for every robot
      r = _raw(env, t)
      if r is not None:
        out -= uneven.float() * r[0] * r[1]
    for t, w_calm in calm.items():
      if t in zero_terms or t in keep_terms:
        continue
      r = _raw(env, t)
      if r is None:
        continue
      raw, w = r
      on = uneven & ~heavy if heavy is not None and t in heavy_terms else uneven
      out += on.float() * raw * (w_calm - w)
    return out


def gait_free(env: ManagerBasedRlEnv, sensor_name: str, term: str = "foot_gait", command_name: str = "twist",
              command_threshold: float = 0.1) -> torch.Tensor:
  """Reward term (weight 1, after `term`): over uneven ground the gait clock is not judged -- a robot under a command
  to move earns the clock's reward in full there, whatever its feet do: raw (1 - current) x the term's weight now
  (go2-depth 2026-11-13; the probe switch U_GAIT_EVEN=1, teacher_env._rewards; the legged only -- the wheeled group's
  clock has stepped on even ground only since the old flow, mdp.feet_gait's scan_sensor_name).

  The legged clock is a 0.6 s trot at every speed below 0.8 m/s, so a foot's stride is speed x 0.6 s: 0.27-0.28 m at
  the 0.45-0.47 m/s the Go2 teacher climbs with, against treads of 0.25 and 0.32 m. The footholds then drift along
  the treads by 2-5 cm a step and cross an edge every 6-12 steps, whatever the kick and edge costs weigh: tier 2 read
  0.41-0.49 kicks a step and 75 % mid-tread footholds from 5598 to 7095 iterations, the same seen and blind, and the
  kick costs at three times the weight did not lower it (uprobe_s31 / s33). The acceptance judges the gait clock on
  even ground only (doc/redesign.md 2: the stairs tests do not read it). A robot told to march in place keeps its
  clock (the march flag is not a command to move)."""
  from src.tasks.velocity.mdp.rewards import uneven_terrain_mask
  out = torch.zeros(env.num_envs, device=env.device)
  r = _raw(env, term)
  if r is None:
    return out
  raw, w = r
  command = env.command_manager.get_command(command_name)
  moving = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold
  return (uneven_terrain_mask(env, sensor_name) & moving).float() * (1.0 - raw) * w


def gait_march_calm(env: ManagerBasedRlEnv, sensor_name: str, even_only: bool = True, term: str = "foot_gait",
                    command_name: str = "twist") -> torch.Tensor:
  """Reward term (weight 1, after `term`): the clock item's multiplier presses on the commands to move only -- a
  robot told to march in place earns the gait clock at the calm weight (constraints.calm: the term's weight with
  every flat item's multiplier held at 1 or below): raw x (calm - current) while its march flag is up (go2-depth
  2026-11-13; the probe switch U_CLOCK_FLOOR=<x>,move, teacher_env._rewards). With `even_only` on even ground only:
  over uneven ground flat_only (U_FLAT_ONLY=1) takes the multiplier off every robot already.

  The clock item is read on the commands to move (flat:gait: the worst command's share in step; the acceptance's
  gait_clock.py judges the strafes, diagonals and arcs), but its multiplier scales foot_gait for the march too, where
  the clock competes with march_position_hold. Go2 uprobe_s63 9091 (U_CLOCK_FLOOR=3, 500 iterations from s59's
  8592), tier 2: the strafes (0, +-0.6, 0) 92-94 % in step and the diagonals (0.6, +-0.4, 0) 89-93 % (s59: 62-79 %,
  bar 80 %) -- and the blind march's drift 26.9 cm against s59's 12.6."""
  out = torch.zeros(env.num_envs, device=env.device)
  calm = getattr(getattr(env, "u_constraints", None), "calm", {})
  march = getattr(env.command_manager.get_term(command_name), "march", None)
  r = _raw(env, term)
  if term not in calm or march is None or r is None:
    return out
  raw, w = r
  on = march
  if even_only:
    from src.tasks.velocity.mdp.rewards import uneven_terrain_mask
    on = on & ~uneven_terrain_mask(env, sensor_name)
  return on.float() * raw * (calm[term] - w)


def loaded_fall(env: ManagerBasedRlEnv, scale: float = 3.0, term: str = "is_terminated") -> torch.Tensor:
  """Reward term (weight 1, after `term`): an episode ended by a fall or an illegal contact costs a loaded robot more,
  from 1x at LIGHT_FRAC of the payload limit to `scale` times at the limit, on top of whatever `term` weighs now (the
  stairs quality's schedule included). 2026-11-12, the user: with payload the robot must never fall, its abilities
  may lag the unloaded ones. The Go2 teacher's heavy evaluation robots failed 0.38-0.42 of their fixed-row stairs
  episodes against the light ones' 0.12, the gap growing with the payload limit; with the acceptance's 8 kg the
  teachers fell on 14-44% of the stairs runs. The teacher sees the payload, so the cost can depend on it: the same
  step is worth trying unloaded and not worth it with 8 kg."""
  out = torch.zeros(env.num_envs, device=env.device)
  kg = getattr(env, "payload_kg", None)
  r = _raw(env, term)
  if kg is None or r is None:
    return out
  raw, w = r
  frac = (kg.float() / payload_limit(env) - LIGHT_FRAC) / (1.0 - LIGHT_FRAC)
  return raw * w * (scale - 1.0) * frac.clamp(0.0, 1.0)


def climbing_mask(env: ManagerBasedRlEnv, sensor_name: str, command_name: str = "twist", near: float = 0.15,
                  far: float = 0.45, half_width: float = 0.15, rise: float = 0.03) -> torch.Tensor:
  """Per robot: the ground ahead along the commanded direction (the height scan's hits `near`..`far` from the base,
  within `half_width` of the line) is more than `rise` higher than the ground as far behind -- it is commanded up a
  flight (or a kerb). False without a command."""
  data = env.scene[sensor_name].data
  cmd = env.command_manager.get_command(command_name)[:, :2]
  speed = torch.norm(cmd, dim=1)
  q = env.scene["robot"].data.root_link_quat_w
  yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  c, s = torch.cos(yaw), torch.sin(yaw)
  d = torch.stack([c * cmd[:, 0] - s * cmd[:, 1], s * cmd[:, 0] + c * cmd[:, 1]], dim=1) / speed.clamp(min=1e-6)[:, None]
  rel = data.hit_pos_w[..., :2] - data.pos_w[:, None, :2]
  along = (rel * d[:, None, :]).sum(dim=-1)
  across = rel[..., 0] * d[:, None, 1] - rel[..., 1] * d[:, None, 0]
  ok = (data.distances >= 0.0) & (across.abs() <= half_width)
  z = data.hit_pos_w[..., 2]

  def mean(m: torch.Tensor) -> torch.Tensor:
    return torch.where(m, z, torch.zeros_like(z)).sum(dim=1) / m.sum(dim=1).clamp(min=1)

  ahead, behind = ok & (along >= near) & (along <= far), ok & (along <= -near) & (along >= -far)
  return (speed > 1e-3) & ahead.any(dim=1) & behind.any(dim=1) & (mean(ahead) - mean(behind) > rise)


def stair_speed_loaded(env: ManagerBasedRlEnv, floor: float = 0.0, term: str = "stair_speed_deficit",
                       up_only: bool = False, sensor_name: str | None = None) -> torch.Tensor:
  """Reward term (weight 1, after `term`): going slowly on the stairs costs a heavily loaded robot less -- the full
  cost up to the common payload (COMMON_FRAC of the limit), `floor` times it from the heavy class (HEAVY_FRAC) up,
  linear in between, on top of whatever `term` weighs now (the probe switch U_STAIR_SPEED_LOADED=<floor>,
  teacher_env._rewards).

  2026-11-13, tier 2 at the walking payload on the candidate teachers: Go2W fell on 97 of the stairs runs (s27, 10%
  / 22% of the 12-15 / 15-18 cm climbs), Go2 on 50 (s36), against a bar of 6 -- and every descent was reached. The
  costs say why: standing on a flight under a forward command costs the speed deficit for as long as it lasts (Go2W
  -12 x 0.9 a second, some -216 over a 20 s episode; Go2 -36), a fall costs once (-40 with U_TERMINATED_Q=10), so a
  robot that cannot make the step is paid to try it anyway. The user (2026-11-12): with payload the robot must never
  fall; what is physically out of reach with it is not required. The acceptance judges the stairs speed unloaded and
  at the common payload (80% of the unloaded reading) and at the full payload only the falls: up to the common
  payload nothing changes, above it the robot may take the stairs as slowly as it needs, or not at all. The teacher
  sees the payload, so its behaviour can depend on it.

  `up_only` (U_STAIR_SPEED_LOADED_UP=1, 2026-11-13): the discount only where the robot is commanded up
  (climbing_mask on `sensor_name`). Tier 2 at 8 kg on uprobe_s40 (floor 0, every direction) against its control
  s38: the falls on the 15-18 cm climbs 49% -> 6%, on the 15-18 cm descents 6% -> 21% (all runs 77 -> 59): a robot
  that may wait before a climb it cannot make stands on its four feet on the level; one that may wait on a descent
  stands nose down on the treads with the load, and the descents were no problem with the cost on."""
  out = torch.zeros(env.num_envs, device=env.device)
  kg = getattr(env, "payload_kg", None)
  r = _raw(env, term)
  if kg is None or r is None:
    return out
  raw, w = r
  frac = (kg.float() / payload_limit(env) - COMMON_FRAC) / (HEAVY_FRAC - COMMON_FRAC)
  relief = raw * w * (floor - 1.0) * frac.clamp(0.0, 1.0)
  if up_only:
    relief = torch.where(climbing_mask(env, sensor_name), relief, out)
  return relief


class heading_ref:
  """Reward term (weight ~0: it returns zeros; after the terms it reads): the straight-line costs `terms`
  (mdp.straight_drift) hold the heading the policy is shown (go2-depth 2026-11-13; the probe switch U_HEADING_OBS=1,
  teacher_env._rewards).

  The actor's one heading reading is mdp.heading_error: a target set at the reset that integrates the commanded yaw
  rate, kept across the commands. The line costs take the heading at the start of each straight command instead: a
  new command forgives what was drifted before it, and after a turn the robot did not follow in full the observation
  says "turn on" where the cost says "hold". The actor cannot tell which heading its command started at, so on even
  ground it could only keep its yaw rate near zero, and a small steady bias adds up: tier 2 on the Go2 teacher
  (s34 model_6097) read -12.6 / -17.3 deg (median, seen / blind) 10 s into a line to the right, -0.6 / -1.2 deg to
  the left, with the heading error in its observation all the way; line drift 14.8 / 18.7% against 5%. On the stairs
  the Go2W recipe's cost already follows the observation (`uneven_heading_obs`) and its heading reads 1.1-1.3 deg.

  Each step this term sets the costs' reference heading to the observation's target and, on the step a line starts,
  the line's direction to the command's direction at that target -- for every robot, so the costs' own stairs part
  follows the observation too. A line that starts at a reset is as before (the target is the heading there); the
  weights, multipliers, groups and readings on those costs are untouched. Nothing while the observation settles
  after a reset (its target is the base heading then, as the line's own start)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    pass

  def reset(self, env_ids=None) -> None:
    pass

  def __call__(self, env: ManagerBasedRlEnv, terms: list[str], obs_term: str = "heading_error",
               command_name: str = "twist") -> torch.Tensor:
    out = torch.zeros(env.num_envs, device=env.device)
    try:
      observed = env.observation_manager.get_term_cfg("critic", obs_term).func
    except ValueError:
      return out
    ready = observed.settle == 0
    command = env.command_manager.get_command(command_name)
    v = command[:, :2] / torch.norm(command[:, :2], dim=1).clamp(min=1e-6).unsqueeze(1)
    c, s = torch.cos(observed.target), torch.sin(observed.target)
    direction = torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], dim=1)
    rm = env.reward_manager
    for t in terms:
      if t not in rm._term_names:
        continue
      term = rm.get_term_cfg(t).func
      term.ref_heading = torch.where(ready, observed.target, term.ref_heading)
      started = ready & term.active & (term.since == 0)   # (the cost ran before this term: since = 0 on a start)
      term.ref_dir = torch.where(started.unsqueeze(1), direction, term.ref_dir)
    return out


def heading_lag(env: ManagerBasedRlEnv, obs_term: str = "heading_error", command_name: str = "twist",
                sensor_name: str | None = "terrain_scan", wz_threshold: float = 1e-6,
                min_speed: float = 0.1) -> torch.Tensor:
  """Reward term (a cost; the probe switch U_HEADING_LAG=<w>, teacher_env._rewards): under a command to move, on
  even ground, how far the base heading is from the heading the commands have turned to -- the heading
  observation's target plus this step's commanded turn (the observation integrates it after the rewards) -- over
  the observation's max_error, at most 1: the error the actor is shown, over its whole range.

  Under a turn: the yaw tracking reward reads the rate, and its kernel is flat near zero; the lag is the rate error
  summed. Tier 2 on the Go2 teacher (s34 model_6097): the yaw of the compound commands at 46-86% of the command,
  24 / 23 of the 44 flat speed commands failing (bar 80%). Under a straight command: the line costs' heading part
  is at its ceiling from heading_scale on (0.17 rad = 9.7 deg) and the observation reaches 0.5 rad, so after a turn
  that was not followed in full they say nothing until the robot is within 9.7 deg; this cost falls all the way.
  With U_HEADING_OBS (without it the line costs hold another heading). Zero on the stairs (the line costs' stairs
  part is there, and a robot is not asked to turn on a flight), under a zero command (the holds keep the robot
  where it stopped) and while the observation settles after a reset."""
  from mjlab.utils.lab_api.math import wrap_to_pi

  from src.tasks.velocity.mdp.rewards import uneven_terrain_mask

  out = torch.zeros(env.num_envs, device=env.device)
  try:
    cfg = env.observation_manager.get_term_cfg("critic", obs_term)
  except ValueError:
    return out
  observed = cfg.func
  command = env.command_manager.get_command(command_name)
  heading = env.scene["robot"].data.heading_w
  off = torch.abs(wrap_to_pi(observed.target + command[:, 2] * env.step_dt - heading))
  cost = (off / float(cfg.params.get("max_error", 0.5))).clamp(max=1.0)
  moving = (command[:, 2].abs() > wz_threshold) | (torch.norm(command[:, :2], dim=1) >= min_speed)
  on = moving & (observed.settle == 0)
  if sensor_name is not None:
    on = on & ~uneven_terrain_mask(env, sensor_name)
  return torch.where(on, cost, out)


class riser_keepout:
  """Reward term (legged; the probe switch U_KICK_KEEPOUT, teacher_env._add_riser_keepout): a swinging foot in the
  keep-out zone of a step face -- within `r_near` / `r_far` of it over the ground (the foot's centre) and lower than
  the step's top + `margin`.

  2026-11-13, Go2 candidate teachers on tier 2: 0.34-0.43 kicks per step against 0.31 (their gate) and 0.245 (the
  students' bar), after the stumble costs' multiplier had sat at its cap. Where the kicks are (kick_anatomy on
  uprobe_s29's model_5598): 70% a swinging foot against the face -- going up with the riser ahead, the foot rising
  (20%) or coming down onto the nosing (21%), going down with the riser behind, the foot coming down (15%). The
  stumble costs act at the touch, a few steps of an episode; this one acts on the swing's path, at every step the
  foot is where a touch can happen: lift before closing in on a riser going up, clear the nosing before dropping
  going down.

  The heights are the shared metrics' raster of the terrain boxes (metrics.terrain, 2 cm cells; the stairs meter's
  grid when the environment has one). Per foot: the cell under its centre and 8 points on each of the two rings
  around it. A ring point is a step face when it is higher than the centre's cell by more than the metrics' JUMP (3
  cm) and by at most `max_rise` (the flights' side walls are no steps). The depth in the zone is (top + margin -
  sole) / `cap`, clipped to 0..1 (sole: the centre less `foot_radius`); the foot's value is the larger of its depth
  at r_near and `far_weight` x its depth at r_far. The term's value is the sum over the feet out of contact
  (`sensor_name`'s found <= 0): a foot standing next to a riser costs nothing.

  `approach` (U_KICK_KEEPOUT_DIR=1, 2026-11-13): only the faces the foot is moving towards -- a ring point counts
  when the foot's horizontal velocity (its site's displacement over the step; zero on the step after a reset) has
  more than `approach_speed` along the point's direction and the point lies within the cone `cone` (a cosine: 0.85,
  32 deg, so one of the 8 directions at least) around that velocity: a foot moving along a face is not closing in
  on it. Tier 2 on uprobe_s39 (weight 4, every face): the kicks
  went from 0.53 / 0.51 to 0.49 / 0.41 a step, and the seen mode stopped at the top of 24% of its descents (stop ok
  74%, the descent's speed x0.65; no stop in the blind mode): going down, a foot past the nosing has the face it
  just left behind it, and it was charged until it had moved 8 cm out without dropping -- no descent's swing is
  like that. With `approach` the face behind a foot that moves away from it costs nothing (a kick of the heel
  against it is still the stumble costs'), and going up the foot pays while it closes in on a riser below its top,
  not while it rises next to it."""

  DIRS = 8

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.H: torch.Tensor | None = None
    self.prev: torch.Tensor | None = None    # the feet's sites at the last call (approach)
    self.fresh: torch.Tensor | None = None   # reset since: no velocity on the next call

  def reset(self, env_ids=None) -> None:
    if self.fresh is not None:
      self.fresh[slice(None) if env_ids is None else env_ids] = True

  def set_grid(self, H, x0: float, y0: float, res: float, device) -> None:
    from src.metrics.terrain import JUMP

    self.H = torch.as_tensor(H, dtype=torch.float32, device=device)
    self.x0, self.y0, self.res, self.jump = float(x0), float(y0), float(res), float(JUMP)
    ang = torch.arange(self.DIRS, device=device) * (2.0 * math.pi / self.DIRS)
    self.ring = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)

  def height(self, xy: torch.Tensor) -> torch.Tensor:
    """The raster's height at (..., 2) world points (the cell as metrics.terrain.Grid.ij takes it; the border's cell
    outside the raster)."""
    i = ((xy[..., 0] - self.x0) / self.res).long().clamp(0, self.H.shape[0] - 1)
    j = ((xy[..., 1] - self.y0) / self.res).long().clamp(0, self.H.shape[1] - 1)
    return self.H[i, j]

  def zone(self, pos: torch.Tensor, foot_radius: float = 0.022, r_near: float = 0.04, r_far: float = 0.08,
           far_weight: float = 0.5, margin: float = 0.02, cap: float = 0.06, max_rise: float = 0.30,
           vel: torch.Tensor | None = None, approach_speed: float = 0.05, cone: float = 0.85) -> torch.Tensor:
    """Each foot's value, 0..1, from the feet's centres (..., 3); with `vel` (..., 2), the feet's horizontal
    velocities, only the faces a foot moves towards (the class's `approach`)."""
    sole = pos[..., 2] - foot_radius
    here = self.height(pos[..., :2])
    cost = torch.zeros_like(sole)
    face = None
    if vel is not None:
      along = (vel[..., None, :] * self.ring).sum(dim=-1)
      face = (along > approach_speed) & (along >= cone * torch.norm(vel, dim=-1, keepdim=True))
    for r, wgt in ((r_near, 1.0), (r_far, far_weight)):
      top = self.height(pos[..., None, :2] + r * self.ring)
      rise = top - here[..., None]
      depth = ((top + margin - sole[..., None]) / cap).clamp(0.0, 1.0)
      on = (rise > self.jump) & (rise <= max_rise)
      if face is not None:
        on = on & face
      depth = torch.where(on, depth, torch.zeros_like(depth))
      cost = torch.maximum(cost, wgt * depth.amax(dim=-1))
    return cost

  def __call__(self, env: ManagerBasedRlEnv, sensor_name: str, asset_cfg, foot_radius: float = 0.022,
               r_near: float = 0.04, r_far: float = 0.08, far_weight: float = 0.5, margin: float = 0.02,
               cap: float = 0.06, max_rise: float = 0.30, approach: bool = False,
               approach_speed: float = 0.05) -> torch.Tensor:
    if self.H is None:
      grid = getattr(getattr(env, "u_lib_meter", None), "grid", None)
      if grid is None:
        from src.metrics.mj import terrain_grid
        grid = terrain_grid(env)
      self.set_grid(grid.H, grid.x0, grid.y0, grid.res, env.device)
    pos = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids]
    swing = env.scene[sensor_name].data.found <= 0
    vel = None
    if approach:
      if self.prev is None:
        self.prev, self.fresh = pos.clone(), torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
      vel = (pos[..., :2] - self.prev[..., :2]) / env.step_dt
      vel = torch.where(self.fresh[:, None, None], torch.zeros_like(vel), vel)
      self.prev, self.fresh = pos.clone(), torch.zeros_like(self.fresh)
    cost = self.zone(pos, foot_radius, r_near, r_far, far_weight, margin, cap, max_rise, vel, approach_speed)
    return (cost * swing.float()).sum(dim=1)


class body_clear(riser_keepout):
  """Reward term (the probe switch U_BODY_CLEAR=<w>, teacher_env._add_body_clear): the bodies whose touch ends the
  episode -- the geoms of the illegal-contact termination's sensor `sensor_name`: trunk, hips, the As2's thigh roots
  -- close to a step. Each geom is a set of surface points (a box's corners, 8 points on each rim of a cylinder, a
  sphere's or a capsule end's centre with its radius). A point is in the zone when it is less than `margin` over
  the terrain under it, or within `reach` over the ground of a step face (the keep-out's: higher than the cell
  under the point by more than the metrics' JUMP and by at most `max_rise`, so the flights' side walls are none)
  and lower than its top + `margin`. Its depth is (top + margin - the point's height) / `cap`, clipped to 0..1; the
  term's value is the deepest point's.

  2026-11-13, where the Go2 candidate (uprobe_s29 model_5598) fails its climbs, robots spread over the rows
  (up_stall_probe 5.): 69-89% of the fails on the rows 7-9 are illegal contacts whatever the payload (light 50% of
  the episodes there, mid 60%, heavy 81%), 55-79% of them the chin (base3_collision) on a step, half before the
  rear feet are on the flight; the As2 probes fail on FL / FR_thigh1 on the low rows and on base1 / base2 at the
  first riser of the high ones (go2-home's stall check). Tier 2 with a payload: the Go2 falls on 36% / 18% of the
  15-18 cm climbs at the common 4 kg and on 49% at 8 kg, wherever the load sits (front, centre, rear: 49-62%),
  against a few of the unloaded runs. The touch itself costs once and ends the episode (-40): nothing tells the
  policy how close the trunk was on the runs that got through. This cost is there on every step a trunk point is
  within the margin of a step, loaded or not -- a loaded trunk rides lower on the same gains, and pays it sooner."""

  RIM = 8

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.ids: torch.Tensor | None = None   # the geoms (the model's ids)
    self.local: torch.Tensor | None = None    # (K, 3) the surface points, each in its geom's frame
    self.owner: torch.Tensor | None = None    # (K,) the index into `ids` of the point's geom
    self.radius: torch.Tensor | None = None   # (K,) a ball around the point (spheres, capsule ends), else 0

  def points(self, env: ManagerBasedRlEnv, sensor_name: str, asset_name: str) -> None:
    import mujoco

    asset = env.scene[asset_name]
    names = list(dict.fromkeys(sl.primary_name for sl in env.scene[sensor_name]._slots))
    local_names = [n.split("/")[-1] for n in names]
    own, found = asset.find_geoms(local_names, preserve_order=True)
    assert len(own) == len(names), f"body_clear: {names} against the robot's geoms {found}"
    self.ids = asset.indexing.geom_ids[own].long()
    model = env.sim.mj_model
    local, owner, radius = [], [], []
    ang = [2.0 * math.pi * k / self.RIM for k in range(self.RIM)]
    for n, gid in enumerate(self.ids.tolist()):
      kind, size = int(model.geom_type[gid]), [float(x) for x in model.geom_size[gid]]
      if kind == mujoco.mjtGeom.mjGEOM_BOX:
        pts = [((sx * size[0], sy * size[1], sz * size[2]), 0.0) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
      elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        pts = [((size[0] * math.cos(t), size[0] * math.sin(t), e * size[1]), 0.0) for e in (-1, 1) for t in ang]
      elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        pts = [((0.0, 0.0, e * size[1]), size[0]) for e in (-1, 1)]
      elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        pts = [((0.0, 0.0, 0.0), size[0])]
      else:   # (a mesh or an ellipsoid: its centre and its bounding radius -- deeper than the surface is)
        pts = [((0.0, 0.0, 0.0), float(model.geom_rbound[gid]))]
      for p, rad in pts:
        local.append(p); owner.append(n); radius.append(rad)
    self.local = torch.tensor(local, dtype=torch.float32, device=env.device)
    self.owner = torch.tensor(owner, dtype=torch.long, device=env.device)
    self.radius = torch.tensor(radius, dtype=torch.float32, device=env.device)

  def depth(self, pos: torch.Tensor, radius: torch.Tensor, reach: float = 0.05, margin: float = 0.03,
            cap: float = 0.05, max_rise: float = 0.30) -> torch.Tensor:
    """Each point's depth in the zone, 0..1, from the points (..., K, 3) and their balls' radii (K,)."""
    low = pos[..., 2] - radius
    here = self.height(pos[..., :2])
    out = ((here + margin - low) / cap).clamp(0.0, 1.0)
    top = self.height(pos[..., None, :2] + (radius[:, None, None] + reach) * self.ring)
    rise = top - here[..., None]
    side = ((top + margin - low[..., None]) / cap).clamp(0.0, 1.0)
    side = torch.where((rise > self.jump) & (rise <= max_rise), side, torch.zeros_like(side))
    return torch.maximum(out, side.amax(dim=-1))

  def __call__(self, env: ManagerBasedRlEnv, sensor_name: str, asset_name: str = "robot", reach: float = 0.05,
               margin: float = 0.03, cap: float = 0.05, max_rise: float = 0.30) -> torch.Tensor:
    if self.H is None:
      grid = getattr(getattr(env, "u_lib_meter", None), "grid", None)
      if grid is None:
        from src.metrics.mj import terrain_grid
        grid = terrain_grid(env)
      self.set_grid(grid.H, grid.x0, grid.y0, grid.res, env.device)
    if self.ids is None:
      self.points(env, sensor_name, asset_name)
    sim = env.scene[asset_name].data.data   # (the simulation's arrays: only these geoms' poses are read)
    centre = sim.geom_xpos[:, self.ids][:, self.owner]   # (N, K, 3)
    frame = sim.geom_xmat[:, self.ids][:, self.owner]    # (N, K, 3, 3)
    pos = centre + torch.einsum("nkij,kj->nki", frame, self.local)
    return self.depth(pos, self.radius, reach, margin, cap, max_rise).amax(dim=1)


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
          value, mask = (CUSTOM.get(metric) or METRICS[metric])(env, **params)
        except Exception:
          continue
        pm = payload_mask(env, lo, hi)
        if pm is None:
          continue
        mask = off_scripts(env, off_slopes(env, on_eval(env, mask & pm)))
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
  rows (curriculum.eval_worst_band; None in a run without evaluation robots), or "flat:<reading>": the worst command
  bin's statistic of the acceptance's flat protocols (flat.flat_measure.read; `stat` also "tail", "dev1", "absmed";
  None in a run without evaluation robots or with U_FLAT_SCRIPTS=0). `scan` ("seen" / "blind"): over the
  robots in that scan mode only (None where the environment has no scan modes)."""
  if metric.startswith("evalfail:"):
    return eval_worst_band(env, metric[9:], scan or "all")
  if metric.startswith("lib:"):
    m = getattr(env, "u_lib_meter", None)
    return None if m is None else m.rate(metric[4:], scan)
  if metric.startswith("rate:"):
    m = getattr(env, "u_meter", None)
    return None if m is None else m.rate(metric[5:], scan)
  if metric.startswith("flat:"):   # the acceptance's flat protocols on evaluation robots (flat.py)
    m = getattr(env, "u_flat", None)
    return None if m is None else m.read(metric[5:], stat, scan, **params)
  value, mask = (CUSTOM.get(metric) or METRICS[metric])(env, **params)
  mask = off_scripts(env, off_slopes(env, on_eval(env, mask)))
  if scan is not None:
    modes = scan_modes(env)
    if modes is None:
      return None
    mask = mask & modes[scan]
  if not mask.any():
    return None
  v = value[mask]
  return float(v.mean()) if stat == "mean" else float(torch.quantile(v, float(stat[1:]) / 100.0))


def item_sig(c: dict) -> str:
  """How constraint c is measured: its metric and statistic (constraints.load)."""
  return f"{c['metric']}|{c.get('stat', 'mean')}"


# The signatures of the items whose measure has changed since the checkpoints that carry none (2026-11-12).
LEGACY_SIG = {"creep": "u_stand_hold_light|p90", "march": "u_stand_hold_light|p90", "clock": "gait_match|mean",
              "side_drift": "u_drift_light|p90", "straight_drift": "u_drift_light|p90"}

# An item whose measure changed to a stricter reading of the same quantity keeps its multiplier across the change
# ({item: the signatures it is carried over from}); its readings start afresh. 2026-11-12, the flat items on the
# acceptance's own protocols (flat.py): the costs they scale are the same, and a resume that put them back to 1 would
# take e.g. the hips' cost from 3x to 1x on a policy that needs more of it, not less.
CARRY_SCALE = {"clock": {"gait_match|mean"}, "side_drift": {"u_drift_light|p90"},
               "straight_drift": {"u_drift_light|p90"}, "creep": {"u_stop_creep_light|mean"},
               "march_hip": {"u_march_hip_light|mean"},
               # (2026-11-13, U_STAIR_SPEED_H: the stairs speed over the ground instead of the body frame's;
               # U_STAIR_SPEED_LOADED: either one over the robots up to the common payload -- u_stair_speed_c / _hc)
               "descent": {"stair_speed|mean", "u_stair_speed_h|mean", "u_stair_speed_c|mean", "u_stair_speed_hc|mean"},
               "climb": {"stair_speed|mean", "u_stair_speed_h|mean", "u_stair_speed_c|mean", "u_stair_speed_hc|mean"}}

# An item split off another one: {new item: (the item its terms hung on, that item's signature then)}. A checkpoint
# from before the split gives the new item the multiplier its terms had (2026-11-12: march_hip_default was scaled
# by the march item; the hips' cost keeps its weight across the resume, the new reading moves it from there).
SPLIT_FROM = {"march_hip": ("march", "u_stand_hold_light|p90")}


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
    self.calm: dict[str, float] = {}       # term -> its weight with no flat item pressing (_apply), where it differs
    self.value: dict[str, float] = {}      # constraint -> the driving reading
    self.values: dict[str, float] = {}     # "<constraint>@<mode>" (no scan modes: "<constraint>") -> smoothed reading
    self.count: dict[str, int] = {}        # the same keys -> measurements so far
    self.relax: dict[str, bool] = {f: False for f in FAMILIES}
    self.ref: dict[str, float] = {}        # stairs family -> the reference its light fail rate is compared with
    self.light: dict[str, dict] = {}       # stairs family -> {"fail", "n"}: that fail rate now, its episodes
    self.ref_step = None
    self.last_update = 0
    self.last_step = None
    # how each item is measured (into the checkpoint: `load` starts an item afresh when it has changed)
    self.sig: dict[str, str] = {c["name"]: item_sig(c) for c in (getattr(cfg, "params", None) or {}).get("items", [])}
    self.redefined: list[str] = []         # the items `load` started afresh
    self.remeasured: list[str] = []        # the items `load` kept the multiplier of and measures anew (CARRY_SCALE)
    self._env = env
    env.u_constraints = self

  def reset(self, env_ids=None) -> None:
    pass

  def state(self) -> dict:
    out = {"scale": dict(self.scale), "value": dict(self.value), "values": dict(self.values),
           "count": dict(self.count), "relax": dict(self.relax), "ref": dict(self.ref), "sig": dict(self.sig)}
    flat = getattr(self._env, "u_flat", None)
    if flat is not None:   # the flat protocols' samples (flat.flat_measure): the "flat:" items' readings
      out["flat"] = flat.state()
    return out

  def forget(self, prefix: str) -> None:
    """Drop the readings of the items measured by metric `prefix` (their source stopped: flat.flat_measure's
    fault): their multipliers hold where they are instead of following a reading that no longer moves."""
    for name, sig in self.sig.items():
      if sig.startswith(prefix):
        self.value.pop(name, None)
        for k in [k for k in {*self.values, *self.count} if k == name or k.startswith(name + "@")]:
          self.values.pop(k, None)
          self.count.pop(k, None)

  def load(self, state: dict) -> None:
    """A checkpoint's multipliers and readings. One from before the scan modes' split (no "values") gives the
    multipliers; its readings are taken afresh. An item measured another way than in the checkpoint (its
    signature; LEGACY_SIG for a checkpoint from before they were saved) starts afresh: multiplier 1, no readings --
    a multiplier that another quantity drove says nothing about this one (2026-11-12: the creep item's sat at its 3x
    cap on the anchor measure); an item in CARRY_SCALE whose old measure is listed there keeps its multiplier and
    only its readings start afresh. The flat protocols' samples ("flat") go to flat.flat_measure when both exist."""
    self.scale = dict(state["scale"])
    self.value = dict(state.get("value", {}))
    self.values = dict(state.get("values", {}))
    self.count = dict(state["count"]) if "values" in state else {}
    self.relax.update({f: bool(v) for f, v in (state.get("relax") or {}).items() if f in self.relax})
    # (a checkpoint from before the stairs families' reference has none: it starts at the first judged reading)
    self.ref = {f: float(v) for f, v in (state.get("ref") or {}).items()}
    old = state.get("sig")
    self.redefined, self.remeasured = [], []
    for name, sig in self.sig.items():
      was = LEGACY_SIG.get(name, sig) if old is None else old.get(name, sig)
      if was == sig:
        continue
      keep = was in CARRY_SCALE.get(name, ()) and name in self.scale
      if not keep:
        self.scale.pop(name, None)
      self.value.pop(name, None)
      for k in [k for k in {*self.values, *self.count} if k == name or k.startswith(name + "@")]:
        self.values.pop(k, None)
        self.count.pop(k, None)
      (self.remeasured if keep else self.redefined).append(name)
      print(f"[unified] constraint {name}: measured as {sig} now, {was} in the checkpoint -- "
            + (f"its multiplier stays ({self.scale[name]:.2f}), its readings start afresh" if keep else
               "its multiplier and readings start afresh"))
    flat = getattr(self._env, "u_flat", None)
    if flat is not None and state.get("flat"):
      flat.load(state["flat"])
    for name, (src, src_sig) in SPLIT_FROM.items():
      if name not in self.sig or name in state["scale"] or src not in state["scale"]:
        continue
      if (LEGACY_SIG.get(src) if old is None else old.get(src)) != src_sig:
        continue
      self.scale[name] = float(state["scale"][src])
      print(f"[unified] constraint {name}: split off {src} -- it starts at {src}'s multiplier in the checkpoint "
            f"({self.scale[name]:.2f})")

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
    calm: dict[str, float] = {}   # the same with every flat item's multiplier at 1 or below (flat_only)
    k = ramp(q, 0.0, Q_IN)
    for c in items:
      for t in c["terms"]:
        if c.get("by") == "q":
          w0 = c["start"][t]
          w = w_calm = w0 + (self.base[t] * self.scale[c["name"]] - w0) * k
        else:
          s = self.scale[c["name"]] if p >= c["activate"] else c.get("start_scale", 1.0)
          w, w_calm = self.base[t] * s, self.base[t] * min(s, 1.0)
        if t not in per_term or abs(w) > abs(per_term[t]):
          per_term[t] = w
        if t not in calm or abs(w_calm) > abs(calm[t]):
          calm[t] = w_calm
    for t, w in per_term.items():
      env.reward_manager.get_term_cfg(t).weight = w
    self.calm = {t: w for t, w in calm.items() if w != per_term[t]}

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
      # ("init_scale": where an item new to a checkpoint that is already in the quality stage starts -- a probe's
      # item, teacher_env U_STOP_FEET_DIR; from zero a stairs-quality item starts its stage at 1 like the others)
      self.scale.setdefault(c["name"], c.get("init_scale", 1.0))
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
