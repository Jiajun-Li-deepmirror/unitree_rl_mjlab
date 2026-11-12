"""One training run through the whole curriculum, driven by how the robots do (the unified teacher, go2-depth
2026-11-11; the user: "一个训练跑完全部课程，不再分 T/M/P ... 地形难度、指令范围、负载、随机化强度，都按表现自动升级，楼梯份额也由
课程控制").

`progress` measures one number p in [0, 1], never going back, the larger of
  - performance: the terrain curriculum's mean level (mjlab's terrain levels, promoted by distance walked, demoted by
    falls and short walks), its rise from where the run started toward GOAL_LEVEL_FRAC of the levels, smoothed;
  - a gated clock (2026-11-11, the user: "直接修改然后重新训练"): from TIME_START_IT it covers 0 -> 1 in TIME_SPAN_IT
    iterations, at a quarter of the pace while more than MAX_FAIL_RATE of the training robots' episodes on the
    other ground (not the stairs) end in a fall or an illegal contact (the blind recipe's staged clock,
    mdp.staged_clock; all the episodes until 2026-11-12: the stairs episodes fail ~0.6 at the curriculum's rows by
    design -- further down -- so at the final stairs share of 0.65 the clock would have crawled from p ~0.55 on,
    whatever the robots do). The terrain levels alone stalled the first Go2 teacher
    at p 0.19 (mean level ~1 at 3000-4000 it: with 8% stairs and many stand / march / spin commands few robots walk the
    3 m a promotion needs; the blind T sits there too), and the stairs share only rises from p 0.3.
Since 2026-11-12 each robot keeps two difficulty rows, one for the stairs columns and one for the other ground
(env.u_rows, progress_stairs_share): every reset draws the robot's column anew, and with one shared row the flat
ground's promotions set the stairs' difficulty -- the Go2W teacher sat at row ~3.2 (~10 cm rises) and failed 100% of
its stairs-up episodes (the chin, base3_collision, into the riser), the Go2 50%, while their stairs failures could not
bring the stairs down to what they could climb. The performance part of p reads the mean of both rows.
The stairs row is two rows since the quality stage (below): one for the columns that climb (UP_NAMES) and one for the
descents, so the climbing row says what the robots climb (shared, the descents' promotions held it up).

`progress` also keeps the stairs quality q in [0, 1], never going back (env.u_quality; 2026-11-12). The recipe's
stairs quality -- the keys its P phase added to a policy that already climbed (per-wheel lift costs near the stairs
at 4x, nosing / slide-back / idle-step / top-landing costs, the stairs speed at 6x, half the riser-lift pay kept for a
clean landing; for legs the kick, slide and mid-tread costs) -- used to be in the teacher's environment from the first
iteration, and the stairs constraints switched on at p 0.5. p is mostly the clock: the Go2W teacher reached p 0.70
without ever lifting a wheel at a riser (3600-3900 it: it drove into the first riser as on flat ground and toppled
over its front wheels, 95-100% of its stairs-up episodes at every row, the 5 cm one included, while the legged Go2
learned to climb in the same setup). Now the environment starts at the recipe's M values (teacher_env.py: the reward
weights and parameters that differ between its M and P keys, and the stairs constraints off), and q moves them to
the P values and the constraints to their bars. q is a gated clock of its own: it covers 0 -> 1 in Q_SPAN_IT
iterations and runs only while the robots climb -- the smoothed mean of the training robots' climbing row is at
least Q_ROW. U_QUALITY=<number> fixes q instead (0: the M values throughout; 1: the P values from the start, as
before).

No fail rate gates q, and none read at the curriculum's rows gates anything on the stairs (2026-11-12, before the
from-zero retrain; the peer review and the 11-12 Go2 teacher's curves). The terrain curriculum moves every robot up
until it fails: a row is promoted after 3 m walked and demoted on a fall, so the fail rate at the robots' own rows
says where the curriculum holds them, not how well they climb. The Go2 teacher at 8000-10800 it: all episodes' fail
rate 0.20-0.32 at a stairs share of 0.27-0.41 with the other ground at ~0.05, that is ~0.6 of its stairs episodes,
while its stairs rows rose 3.4 -> 4.4. A limit of MAX_FAIL_RATE on that number (q's clock, and the stairs
constraints' relaxing in constraints.py) would have held q at 0 and the stairs constraints relaxed for the whole
run. Instead the evaluation robots (env.u_eval_mask: the mean action, outside the update, constraints.py) keep fixed
stairs rows: spread evenly over the rows the acceptance covers (steps up to EVAL_STEP_MAX; eval_rows), the same row
on the climbing columns and on the descents, both scan modes on every row. Their fail rate there
(env.u_eval_fail, per family and per row) is a number that means the same at every iteration -- what share of the
acceptance's stairs the policy fails; the stairs constraints' readings, which are taken over the evaluation robots,
come from those rows too (not from rows above the acceptance's, and not from wherever the curriculum stands). The
training robots' rows, the terrain levels behind p and q's climbing row are the training robots' alone. The same
episodes are kept split by payload class and scan mode (env.u_eval_split, PAY_CLASSES): the light robots' rate is
what relaxes the stairs constraints -- when it rises above its own slowly tracked reference, not above a fixed
level (constraints._families) -- and, on the worst step-height band, what the teacher's gate asks for
(teacher_env.constraint_items: survive_up / survive_down).

Everything else reads env.u_progress:

  stairs share     TEACHER_SHARE until p 0.3, then up to the robot's final share by p 0.8 (mdp.stairs_share's
                   column draw, with p in place of the training step)
  commands         the initial ranges until p 0.25, the final ones after (as mdp.commands_vel's two stages); a
                   wheeled robot's sideways range opens from 0.1 m/s over p 0.10 -> 0.25 (progress_commands'
                   lateral_ramp, teacher_env.LAT_RAMP)
  payload          the heavy robots' payload limit rises with p to half the robot's limit at p 0.25 and stays there
                   until the stairs quality begins; the other half follows q (progress_payload: the smaller of
                   p / 0.5 and 0.5 + 0.5 q; mdp.base_payload_point's heavy_max)
  slopes           two sloped tiles among the other ground (SLOPE_NAMES, teacher_env._add_slopes): a fixed share of
                   the other ground's episodes (progress_stairs_share's other_shares), their gradient with the other
                   ground's row; no acceptance item is measured there (slope_envs)
  constraints      the flat ones switch on at their own p and tighten from a loose target to the teacher's own (a
                   margin inside the students' bars); the stairs ones follow q; each relaxes while its own family of
                   ground fails (constraints.py: the other ground above a fixed fail rate, the stairs when the light
                   evaluation robots' fail rate rises above its reference)

Each term logs what it set (Curriculum/u_*), and p goes into the checkpoint (runner.py), so a resumed run carries on
where it was.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

GOAL_LEVEL_FRAC = 0.5   # p = 1 when the mean level is here (robots past the top level are sent back to random rows)
# The clock (2026-11-12, the user: "尽可能的快速迭代版本 ... 不要耗 6 小时才知道要修，大概两三个小时就知道版本哪里有问题"; 1500 and
# 10000 before, a teacher's gate at 12000 iterations or 13 hours at the earliest). The 11-12 Go2 teacher on the old
# clock walked the flat ground by iteration 800-1000 (fail rate 0.25 -> 0.21) and then waited: 4000 iterations at a
# stairs share of 3-7% and the first command ranges, the rows creeping from 0.3 to 1.7; once the share rose (p 0.3 at
# 5000) the stairs rows reached 3 within 2000 iterations. The clock still halts itself at a quarter of its pace while
# more than MAX_FAIL_RATE of the other ground's episodes fail, and the constraints' multipliers (constraints.LR:
# 1 -> 3 in ~370 iterations) fit the 1200 iterations a target takes from loose to tight. What a run has to have
# reached by when: diagnostics.expectations().
TIME_START_IT, TIME_SPAN_IT, MAX_FAIL_RATE = 500, 4000, 0.3
STEPS_PER_IT = 24   # the runners' num_steps_per_env
# the stairs quality q (module docstring): its clock's span, the climbing row it needs, and the part of q over which
# the reward terms go from the M to the P values (the constraints' targets tighten over all of it)
Q_SPAN_IT, Q_ROW, Q_IN = 1000, 3.0, 0.5   # (the span 2000 before the faster clock above)
# The stairs quality by health (progress's q_back, 2026-11-12): how far the light evaluation robots' fixed-row fail
# rate may sit above its slow reference (constraints._families: the same two numbers decide the relaxing) before q
# waits (the first) and before it goes back (the second, the gap at which the constraints relax); the light episodes
# a stairs family needs before it is judged.
Q_GAP = (0.04, 0.08)
Q_HEALTH_MIN = 200
UP_NAMES = ("stairs_up_narrow", "stairs_up_wide", "walled_up")   # the stairs columns that climb
ROWS = ("up", "down", "other")
SLOPE_NAMES = ("slope_down", "slope_up")   # the sloped tiles (teacher_env._add_slopes): other ground, not measured
# The evaluation robots' stairs rows (module docstring): the rows that can hold a step below this height (m), the
# acceptance's highest (doc/acceptance.md: stairs to 18 cm; the training tiles go to 20). EVAL_FAIL_ALPHA: each
# row's fail rate moves by this much per finished episode (the mean of its first 1 / alpha episodes before that).
EVAL_STEP_MAX, EVAL_FAIL_ALPHA = 0.18, 0.05
# The same episodes split (2026-11-12, env.u_eval_split): by the finished episode's payload class (PAY_CLASSES:
# constraints.LIGHT_FRAC / HEAVY_FRAC of the robot's payload limit) and its scan mode ("all": both), per fixed row.
# The light robots' rates (the acceptance's unloaded tests) are the ones acted on: they decide whether the stairs
# constraints relax (constraints._families) and they are the teacher gate's survival items (eval_worst_band: the
# worst of the acceptance's step-height bands; EVAL_BANDS: the bands' inner edges in m, a row by its lowest step).
# The mid and heavy classes are recorded only. The all-payload rate above cannot do either job: it rises with the
# payload curriculum whatever the policy does (the 11-12 probe, the old Go2 teacher continued: loaded 46-79% against
# light 32-68% on every stairs ground; its fixed-row rate 0.36-0.40 kept all stairs constraints relaxed for the
# whole quality stage, 2000 iterations, and the gate could not pass).
PAY_CLASSES = ("light", "mid", "heavy")
SPLIT_MODES = ("all", "seen", "blind")
EVAL_BANDS = (0.08, 0.12, 0.15)   # 5-8 / 8-12 / 12-15 / 15-18 cm


def eval_rows(env: ManagerBasedRlEnv, stair_names: tuple[str, ...], step_max: float = EVAL_STEP_MAX) -> torch.Tensor | None:
  """(N,) each evaluation robot's fixed stairs row, -1 for the training robots; None for a run without evaluation
  robots (the students, the probes). The rows 0 .. k-1 whose lowest step is below step_max (every row on a terrain
  without a generator), one pair of robots after the other: an even and an odd index share a row, so both scan modes
  (teacher_env.u_scan_mask: even seen, odd blind) stand on every row. Kept on the env (u_eval_rows; the number of
  rows as u_eval_row_count)."""
  fixed = getattr(env, "u_eval_rows", None)
  if fixed is not None:
    return fixed
  ev = getattr(env, "u_eval_mask", None)   # (set by the teacher's runner after the environment's first reset)
  if ev is None or not bool(ev.any()):
    return None
  terrain = env.scene.terrain
  n_rows = int(terrain.max_terrain_level)
  k = n_rows
  steps = None   # each row's lowest step (m); None on a terrain without a generator
  gen = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
  heights = [s.step_height_range for name, s in (gen.sub_terrains.items() if gen is not None else ())
             if name in stair_names and getattr(s, "step_height_range", None) is not None]
  if heights:
    lo, hi = gen.difficulty_range
    # a tile of row r draws its difficulty in [r, r + 1) / n_rows of the range, and its step height is linear in it
    lowest = [min(a + (b - a) * (lo + (hi - lo) * r / n_rows) for a, b in heights) for r in range(n_rows)]
    k = max(1, sum(h < step_max - 1e-9 for h in lowest))
    steps = [float(h) for h in lowest[:k]]
  idx = ev.nonzero().flatten()
  fixed = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
  fixed[idx] = (torch.arange(len(idx), device=env.device) // 2) % k
  env.u_eval_rows, env.u_eval_row_count, env.u_eval_row_step = fixed, k, steps
  print(f"[unified] evaluation robots: {len(idx)} on fixed stairs rows 0..{k - 1} of {n_rows} "
        f"(steps below {step_max} m), {len(idx) / k:.1f} per row; step-height bands of the rows "
        f"{row_bands(env)}")
  return fixed


def row_bands(env: ManagerBasedRlEnv) -> list[int]:
  """Each fixed row's step-height band (0 .. len(EVAL_BANDS)), by the row's lowest step; one band for all on a
  terrain without a generator."""
  steps = getattr(env, "u_eval_row_step", None)
  if steps is None:
    return [0] * int(getattr(env, "u_eval_row_count", 0) or 0)
  return [sum(h >= e - 1e-9 for e in EVAL_BANDS) for h in steps]


def _split_new(k: int) -> dict:
  return {c: {m: {"rate": [0.0] * k, "n": [0] * k} for m in SPLIT_MODES} for c in PAY_CLASSES}


def _split_fits(fam: dict | None, k: int) -> bool:
  return bool(fam) and all(len(fam.get(c, {}).get(m, {}).get(x, ())) == k
                           for c in PAY_CLASSES for m in SPLIT_MODES for x in ("rate", "n"))


def pay_class(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> list[int] | None:
  """The payload class (index into PAY_CLASSES) of each of env_ids' robots, from the payload they carry now (at the
  curriculum's call: the finished episode's, the reset events run after it); None before the first draw."""
  from .constraints import HEAVY_FRAC, LIGHT_FRAC, payload_limit   # (constraints.py imports this module)

  kg = getattr(env, "payload_kg", None)
  if kg is None:
    return None
  limit = payload_limit(env)
  k = kg.reshape(env.num_envs, -1)[env_ids, 0]
  return ((k > LIGHT_FRAC * limit).long() + (k >= HEAVY_FRAC * limit).long()).tolist()


def eval_light(env: ManagerBasedRlEnv, family: str, mode: str = "all", cls: str = "light") -> tuple[float | None, int]:
  """The light evaluation robots' fail rate on their fixed rows of `family` (the mean over the rows that have had an
  episode; None without one) and the number of episodes behind it. `cls`: another payload class."""
  s = ((getattr(env, "u_eval_split", None) or {}).get(family) or {}).get(cls, {}).get(mode)
  if not s:
    return None, 0
  seen = [v for v, n in zip(s["rate"], s["n"]) if n > 0]
  return (sum(seen) / len(seen) if seen else None), int(sum(s["n"]))


def eval_bands(env: ManagerBasedRlEnv, family: str, mode: str = "all", cls: str = "light") -> list[float | None]:
  """The same per step-height band (row_bands): the mean of the band's rows that have had an episode, None for a
  band without one."""
  s = ((getattr(env, "u_eval_split", None) or {}).get(family) or {}).get(cls, {}).get(mode)
  bands = row_bands(env)
  out: list[float | None] = []
  for b in range(max(bands) + 1 if bands else 0):
    v = [s["rate"][r] for r, rb in enumerate(bands) if rb == b and s and r < len(s["n"]) and s["n"][r] > 0]
    out.append(sum(v) / len(v) if v else None)
  return out


def eval_worst_band(env: ManagerBasedRlEnv, family: str, mode: str = "all") -> float | None:
  """The light evaluation robots' fail rate on the worst step-height band of `family` (the teacher's survival
  items, teacher_env.constraint_items); None before any band has an episode."""
  v = [x for x in eval_bands(env, family, mode) if x is not None]
  return max(v) if v else None


def _eval_fail(env: ManagerBasedRlEnv, family: str, rows: list[int], failed: list[bool], alpha: float,
               pay: list[int] | None = None, seen: list[bool] | None = None) -> None:
  """Add the evaluation robots' finished stairs episodes (their fixed rows, whether each failed) to the family's
  fail rates: per row in env.u_eval_stat[family] = {"rate": [...], "n": [...]} and, over the rows that have had an
  episode, their mean in env.u_eval_fail[family]. With `pay` (each episode's payload class) also split by class and
  scan mode (`seen`; None: an environment without scan modes, "all" only):
  env.u_eval_split[family][class][mode] = {"rate": [...], "n": [...]}."""
  k = env.u_eval_row_count
  if pay is not None:
    split = getattr(env, "u_eval_split", None)
    if split is None:
      split = env.u_eval_split = {}
    if not _split_fits(split.get(family), k):   # none yet, or a checkpoint's of other rows
      split[family] = _split_new(k)
    for i, (r, bad) in enumerate(zip(rows, failed)):
      for m in ("all",) + (() if seen is None else ("seen" if seen[i] else "blind",)):
        c = split[family][PAY_CLASSES[pay[i]]][m]
        c["n"][r] += 1
        c["rate"][r] += max(alpha, 1.0 / c["n"][r]) * (float(bad) - c["rate"][r])
  stat = getattr(env, "u_eval_stat", None)
  if stat is None:
    stat = env.u_eval_stat = {}
  s = stat.get(family)
  if s is None or len(s.get("rate", ())) != k or len(s.get("n", ())) != k:   # none yet, or a checkpoint's of other rows
    s = stat[family] = {"rate": [0.0] * k, "n": [0] * k}
  for r, bad in zip(rows, failed):
    s["n"][r] += 1
    s["rate"][r] += max(alpha, 1.0 / s["n"][r]) * (float(bad) - s["rate"][r])
  seen = [v for v, n in zip(s["rate"], s["n"]) if n > 0]
  fails = getattr(env, "u_eval_fail", None)
  if fails is None:
    fails = env.u_eval_fail = {}
  fails[family] = sum(seen) / len(seen)


def slope_envs(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  """(N,) True for the robots on a sloped tile this episode; None on a terrain without them."""
  cols = getattr(env, "u_slope_cols", False)
  if cols is False:
    from src.tasks.velocity.mdp.terrain_utils import sub_terrain_column_mask

    cols = sub_terrain_column_mask(env, SLOPE_NAMES)
    if cols is not None and not bool(cols.any()):
      cols = None
    env.u_slope_cols = cols
  if cols is None:
    return None
  return cols[env.scene.terrain.terrain_types]


def other_column_weights(env: ManagerBasedRlEnv, shares: dict[str, float], stair_mask: torch.Tensor) -> torch.Tensor:
  """(columns,) the chance of each column for an episode on the other ground: each tile name's share (`shares`),
  split evenly over its columns; 0 on the stairs columns. Kept on the env."""
  w = getattr(env, "u_other_col_w", None)
  if w is None:
    from src.tasks.velocity.mdp.terrain_utils import sub_terrain_column_mask

    w = torch.zeros(len(stair_mask), device=env.device)
    for name, share in shares.items():
      cols = sub_terrain_column_mask(env, (name,))
      assert cols is not None and cols.any() and not (cols & stair_mask).any(), f"other ground {name!r}: no columns"
      w[cols] = float(share) / float(cols.sum())
    assert bool((w[~stair_mask] > 0).all()), "an other-ground column without a share"
    w = env.u_other_col_w = w / w.sum()
  return w


def rows_from(state_rows: dict | None, n: int) -> dict | None:
  """A checkpoint's rows as {"up", "down", "other"} lists (one from before the climbing / descending split: both
  from its stairs row); None when they do not fit n robots."""
  if not state_rows:
    return None
  r = dict(state_rows)
  if "stairs" in r:
    st = r.pop("stairs")
    r.setdefault("up", st)
    r.setdefault("down", st)
  if set(r) != set(ROWS) or any(len(v) != n for v in r.values()):
    return None
  return r


def ramp(p: float, a: float, b: float) -> float:
  """0 below a, 1 above b, linear between."""
  if b <= a:
    return float(p >= a)
  return min(1.0, max(0.0, (p - a) / (b - a)))


class progress:
  """Curriculum term: env.u_progress from the terrain levels (see the module docstring). Runs before the terms that
  read it (first in the curriculum)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.p = 0.0
    self.level = None        # smoothed mean level
    self.level0 = None       # where the run started
    self.last_step = None
    self.clock = None        # the gated clock (env steps)
    self.fail = None         # smoothed share of resets that ended in a fall / illegal contact
    self.clock_fail = None   # the fail rate the clock's gate read last (the other ground's)
    self.fail_mode: dict[str, float] = {}   # the same over the evaluation robots per scan mode; logged only
    self.q = 0.0             # the stairs quality (module docstring)
    self.up = None           # smoothed mean of the climbing row
    self.q_last_step = None
    env.u_progress = 0.0
    env.u_quality = 0.0
    env.u_progress_term = self   # the runner saves and restores it
    self._env = env

  def reset(self, env_ids=None) -> None:
    pass

  def state(self) -> dict:
    out = {"p": self.p, "level": self.level, "level0": self.level0, "clock": self.clock, "fail": self.fail,
           "q": self.q, "up": self.up, "fail_mode": dict(self.fail_mode),
           # each ground family's fail rate at the training robots' rows, and the evaluation robots' on their fixed
           # stairs rows (progress_stairs_share): the constraints' relaxing reads them
           "family_fail": dict(getattr(self._env, "u_family_fail", None) or {}),
           "eval_fail": dict(getattr(self._env, "u_eval_fail", None) or {}),
           "eval_stat": {k: {"rate": list(v["rate"]), "n": list(v["n"])}
                         for k, v in (getattr(self._env, "u_eval_stat", None) or {}).items()},
           # the same split by payload class and scan mode (the light robots': the stairs relaxing, the survival items)
           "eval_split": {k: {c: {m: {"rate": list(x["rate"]), "n": list(x["n"])} for m, x in cm.items()}
                              for c, cm in v.items()}
                          for k, v in (getattr(self._env, "u_eval_split", None) or {}).items()}}
    terrain = getattr(self, "_terrain", None)
    if terrain is not None and getattr(terrain, "terrain_levels", None) is not None:
      out["terrain_levels"] = terrain.terrain_levels.cpu().tolist()
      out["terrain_types"] = terrain.terrain_types.cpu().tolist()
    rows = getattr(self._env, "u_rows", None)
    if rows is not None:
      out["rows"] = {k: v.cpu().tolist() for k, v in rows.items()}
    cls = getattr(self._env, "u_pay_class", None)   # (teacher_env.sticky_payload: the rows are the class's)
    if cls is not None:
      out["pay_class"] = cls.cpu().tolist()
    return out

  def load(self, state: dict, env: ManagerBasedRlEnv) -> None:
    self.p, self.level, self.level0 = state["p"], state["level"], state["level0"]
    # a checkpoint from before the clock: it starts at the restored step count
    self.clock = state.get("clock") if state.get("clock") is not None else float(env.common_step_counter)
    self.fail = state.get("fail")
    self.fail_mode = dict(state.get("fail_mode") or {})
    if state.get("family_fail"):
      env.u_family_fail = dict(state["family_fail"])
    if state.get("eval_stat"):   # (a checkpoint from before the fixed rows has none: counted afresh)
      env.u_eval_fail = dict(state.get("eval_fail") or {})
      env.u_eval_stat = {k: {"rate": [float(x) for x in v["rate"]], "n": [int(x) for x in v["n"]]}
                         for k, v in state["eval_stat"].items()}
    if state.get("eval_split"):   # (a checkpoint from before the split has none: counted afresh)
      env.u_eval_split = {k: {c: {m: {"rate": [float(x) for x in s["rate"]], "n": [int(x) for x in s["n"]]}
                                  for m, s in cm.items()} for c, cm in v.items()}
                          for k, v in state["eval_split"].items()}
    self.last_step = self.q_last_step = env.common_step_counter
    # a checkpoint from before the quality stage: q 0, the environment back at the M values until the robots climb
    self.q = float(state.get("q") or 0.0)
    self.up = state.get("up")
    env.u_progress = self.p
    env.u_quality = self.q
    self._restore_terrain(state, env)
    n = env.scene.terrain.terrain_levels.shape[0] if getattr(env.scene, "terrain", None) is not None else 0
    rows = rows_from(state.get("rows"), n)
    if rows:
      dev, dt = env.scene.terrain.terrain_levels.device, env.scene.terrain.terrain_levels.dtype
      env.u_rows = {k: torch.tensor(v, device=dev, dtype=dt) for k, v in rows.items()}
      print("[unified] difficulty rows restored: "
            + ", ".join(f"{k} {float(env.u_rows[k].float().mean()):.2f}" for k in ROWS) + f"; quality {self.q:.3f}")
      cls = state.get("pay_class")
      if cls is not None and len(cls) == n:   # (read only by teacher_env.sticky_payload)
        env.u_pay_class = torch.tensor(cls, device=dev, dtype=torch.long)
    elif n:   # a checkpoint from before the rows: all start at the robot's restored row
      lv = env.scene.terrain.terrain_levels
      env.u_rows = {k: lv.clone() for k in ROWS}
      print(f"[unified] difficulty rows: none in the checkpoint, all start at each robot's restored row "
            f"(mean {float(lv.float().mean()):.2f})")

  @staticmethod
  def stairs_gap(env: ManagerBasedRlEnv) -> float | None:
    """How far the light evaluation robots' fixed-row fail rate is above its slow reference, the worse of the two
    stairs families (constraints._families keeps both numbers and relaxes on the same gap); None until a family has
    Q_HEALTH_MIN light episodes and a reference."""
    con = getattr(env, "u_constraints", None)
    gaps = []
    for f in ("up", "down"):
      d = (getattr(con, "light", None) or {}).get(f) or {}
      ref = (getattr(con, "ref", None) or {}).get(f)
      if d.get("fail") is not None and ref is not None and d.get("n", 0) >= Q_HEALTH_MIN:
        gaps.append(d["fail"] - ref)
    return max(gaps) if gaps else None

  def _restore_terrain(self, state: dict, env: ManagerBasedRlEnv) -> None:
    """The robots' terrain levels (and columns) as the checkpoint left them (2026-11-12). A fresh environment draws
    them anew, and the progress -- which only rises -- read that draw as an advance: the Go2 teacher's level jumped
    0.77 -> 1.91 and its progress 0.10 -> 0.35 on a resume (the 11-11 v1 "level 1 -> 2.07" was the same). A checkpoint
    from before this saves no levels: the robots get levels averaging the checkpoint's level reading instead."""
    terrain = env.scene.terrain
    self._terrain = terrain
    if getattr(terrain, "terrain_levels", None) is None or getattr(terrain, "terrain_origins", None) is None:
      return
    n = terrain.terrain_levels.shape[0]
    dev = terrain.terrain_levels.device
    if len(state.get("terrain_levels") or []) == n:
      terrain.terrain_levels[:] = torch.tensor(state["terrain_levels"], device=dev, dtype=terrain.terrain_levels.dtype)
      terrain.terrain_types[:] = torch.tensor(state["terrain_types"], device=dev, dtype=terrain.terrain_types.dtype)
      how = "restored"
    elif self.level is None:   # neither levels nor a level reading in the checkpoint: as the fresh environment drew them
      return
    else:
      lo = math.floor(self.level)
      up = torch.rand(n, device=dev) < (self.level - lo)
      lv = torch.clamp(lo + up.long(), 0, int(terrain.max_terrain_level) - 1)
      terrain.terrain_levels[:] = lv.to(terrain.terrain_levels.dtype)
      how = f"drawn around the checkpoint's level {self.level:.2f}"
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    before = float(terrain.terrain_levels.float().mean())
    # every robot onto its restored origin now: left where the fresh environment put it, its first reset measured its
    # "distance walked" from the new origin and promoted it (the Go2 teacher's level 0.75 -> 1.50 in 30 it, progress
    # 0.10 -> 0.19, 2026-11-12). The reset itself runs the curriculum first -- the same promotion (0.75 -> 1.57) --
    # so the curriculum is held off for this one reset.
    cm = env.curriculum_manager
    compute = cm.compute
    cm.compute = lambda *a, **k: None
    try:
      env.reset()
    finally:
      cm.compute = compute
    after = float(terrain.terrain_levels.float().mean())
    print(f"[unified] terrain levels {how}: mean {before:.2f}, after the robots' reset onto them {after:.2f}")

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, ema_steps: float = 2000.0,
               skip_steps: int = 2400, force_rate: float = 0.0, time_start_it: int = TIME_START_IT,
               time_span_it: int = TIME_SPAN_IT, max_fail_rate: float = MAX_FAIL_RATE,
               fixed_p: float | None = None, q_span_it: int = Q_SPAN_IT, q_row: float = Q_ROW,
               force_q: bool = False, q_back: float | None = None,
               q_gap: tuple[float, float] = Q_GAP) -> dict[str, torch.Tensor]:
    terrain = env.scene.terrain
    self._terrain = terrain   # state() saves its levels
    assert terrain is not None
    rows = getattr(env, "u_rows", None)
    # the training robots: the evaluation robots' stairs rows are fixed (module docstring) and say nothing of the
    # curriculum, and what they fail on rows the policy has not reached is not the training's fail rate
    ev = getattr(env, "u_eval_mask", None)
    if ev is not getattr(self, "_ev", None):   # (the teacher's runner sets it once, after the first reset)
      self._ev, self._train = ev, (None if ev is None or not bool(ev.any()) else ~ev)
    train = getattr(self, "_train", None)
    if rows is not None:   # the stairs' (climbing and descending) and the other ground's rows alike (module docstring)
      m = lambda v: float((v if train is None else v[train]).float().mean())   # noqa: E731
      up = m(rows["up"])
      mean = 0.5 * (0.5 * (up + m(rows["down"])) + m(rows["other"]))
    else:
      up = mean = float(terrain.terrain_levels.float().mean())
    step = env.common_step_counter
    # the gated clock: the resets now, how many ended in a fall or illegal contact (not a time-out)
    if env_ids is not None and len(env_ids) > 0:
      tm = env.termination_manager
      bad = tm.terminated[env_ids] & ~tm.time_outs[env_ids]
      mine = bad if train is None else bad[train[env_ids]]
      if mine.numel() > 0:
        failed = float(mine.float().mean())
        self.fail = failed if self.fail is None else self.fail + 0.01 * (failed - self.fail)
      # the same over the evaluation robots (constraints.py: the mean action, no exploration noise; on the stairs
      # their fixed rows); logged only
      if ev is not None and bool(ev[env_ids].any()):
        f = float(bad[ev[env_ids]].float().mean())
        self.fail_eval = f if getattr(self, "fail_eval", None) is None else self.fail_eval + 0.05 * (f - self.fail_eval)
        # and per scan mode (teacher_env.u_scan_mask: the flag still is the finished episode's, the reset events
        # run after the curriculum)
        on = getattr(env, "u_scan_on", None)
        if on is not None:
          seen = on[env_ids] > 0.5
          for k, m in (("seen", ev[env_ids] & seen), ("blind", ev[env_ids] & ~seen)):
            if bool(m.any()):
              f = float(bad[m].float().mean())
              self.fail_mode[k] = f if k not in self.fail_mode else self.fail_mode[k] + 0.05 * (f - self.fail_mode[k])
    if self.clock is None or self.last_step is None:
      self.clock = float(step)
    else:
      d = max(0, step - self.last_step)
      # its gate: do the robots walk -- the other ground's fail rate (progress_stairs_share's, the training robots';
      # every episode's until there is one). Not the stairs': module docstring.
      walk = (getattr(env, "u_family_fail", None) or {}).get("other")
      self.clock_fail = (self.fail or 0.0) if walk is None else walk
      self.clock_slow = self.clock_fail > max_fail_rate   # (both read by the training records, diagnostics.py)
      self.clock += d * (0.25 if self.clock_slow else 1.0)
    self.last_step = step
    p_time = ramp(self.clock / STEPS_PER_IT, time_start_it, time_start_it + time_span_it)
    if step >= skip_steps:   # every robot resets at the start; the first levels say nothing yet
      if self.level is None:
        self.level = self.level0 = mean
      else:
        a = 1.0 - math.exp(-1.0 / ema_steps)
        self.level += a * (mean - self.level)
      goal = GOAL_LEVEL_FRAC * float(terrain.max_terrain_level)
      if goal > self.level0:
        self.p = max(self.p, min(1.0, (self.level - self.level0) / (goal - self.level0)))
      self.p = max(self.p, p_time)
    if force_rate > 0.0:   # smoke runs only: progress per env step regardless of the levels
      self.p = min(1.0, max(self.p, step * force_rate))
    # the stairs quality: its clock runs while the robots climb (the training robots' smoothed climbing row at q_row
    # or above). No fail rate gates it (module docstring: at the curriculum's rows ~0.6 of the stairs episodes fail by
    # design); a policy that stops climbing is demoted, the row falls below q_row and the clock waits.
    if step >= skip_steps:
      a = 1.0 - math.exp(-1.0 / ema_steps)
      self.up = up if self.up is None else self.up + a * (up - self.up)
    climbing = self.up is not None and self.up >= q_row
    # By health (q_back, off when None: by the clock alone). 2026-11-12: by the clock q took the Go2W teachers from
    # 0.1 to 0.5 -- the P weights fully in -- within 400 iterations of climbing row 4; the light fixed-row climbing
    # fail rate went 0.18 -> 0.25 -> 0.54 over the last 150 of them and the stage ended with every stairs episode
    # failing, whatever the multipliers' cap or the cost of a fall. So q follows what the policy copes with: while the
    # light evaluation robots' fixed-row fail rate (the worse stairs family) is more than q_gap[0] above its slow
    # reference q waits, more than q_gap[1] above it (where the constraints relax) q goes back at q_back times its
    # pace -- from 1 too.
    gap = self.stairs_gap(env) if q_back is not None else None
    hold = gap is not None and gap > q_gap[0]
    back = gap is not None and gap > q_gap[1]
    self.q_gap = gap
    # why the stage's clock runs or waits, for the training records (diagnostics.py)
    self.q_gate = ("back" if back and self.q > 0.0 else "done" if self.q >= 1.0 else "waiting_health" if hold
                   else "running" if climbing else "waiting_row")
    if self.q_last_step is not None:
      dq = max(0, step - self.q_last_step) / (STEPS_PER_IT * q_span_it)
      if back:
        self.q = max(0.0, self.q - q_back * dq)
      elif climbing and not hold:
        self.q = min(1.0, self.q + dq)
    self.q_last_step = step
    if force_q:   # smoke runs only: q follows the forced progress, so every switch-on runs
      self.q = max(self.q, ramp(self.p, 0.3, 0.8))
    fixed_q = os.environ.get("U_QUALITY", "")
    if fixed_q not in ("", "auto"):
      self.q = min(1.0, max(0.0, float(fixed_q)))
    if fixed_p is not None:   # the students (2026-11-12): the teacher's final curriculum, all the way
      self.p = float(fixed_p)
      self.q = 1.0
    env.u_progress = self.p
    env.u_quality = self.q
    out = {"p": torch.tensor(self.p), "level": torch.tensor(mean), "p_time": torch.tensor(p_time),
           "fail_rate": torch.tensor(self.fail or 0.0), "q": torch.tensor(self.q),
           "q_running": torch.tensor(float(climbing and not hold and not back))}
    if gap is not None:
      out["q_gap"] = torch.tensor(gap)
    if self.up is not None:
      out["up_level"] = torch.tensor(self.up)
    if getattr(self, "fail_eval", None) is not None:
      out["fail_rate_eval"] = torch.tensor(self.fail_eval)
    for k, v in self.fail_mode.items():
      out[f"fail_rate_eval_{k}"] = torch.tensor(v)
    return out


def progress_stairs_share(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stair_terrain_names: tuple[str, ...],
  start: float,
  end: float,
  p_start: float = 0.3,
  p_end: float = 0.8,
  up_terrain_names: tuple[str, ...] = UP_NAMES,
  fail_alpha: float = 0.02,
  other_shares: dict[str, float] | None = None,
  eval_step_max: float = EVAL_STEP_MAX,
  eval_fail_alpha: float = EVAL_FAIL_ALPHA,
) -> dict[str, torch.Tensor]:
  """mdp.stairs_share's draw (each resetting robot to a stairs column with probability s), with s = start -> end as
  env.u_progress goes p_start -> p_end. Each robot has three difficulty rows (env.u_rows, module docstring): for the
  stairs columns that climb, for the descents and for the other ground. The row the terrain curriculum just set for
  the finished episode goes back to the family it was walked on, and the robot starts its next episode on the row of
  the family drawn. An evaluation robot's two stairs rows are fixed instead (eval_rows; its other-ground row follows
  the curriculum as the training robots'). Also keeps and logs the share of episodes that ended in a fall or an
  illegal contact: per family over the training robots at their own rows (env.u_family_fail, <family>_fail), and
  per stairs family over the evaluation robots on their fixed rows (env.u_eval_fail, eval_<family>_fail; split by
  the payload class and the scan mode in env.u_eval_split, eval_<family>_fail_<class>).
  other_shares ({tile name: share}): an episode on the other ground goes to each tile with its share (the sloped
  tiles' fixed share, teacher_env._add_slopes); None: every other column alike."""
  from src.tasks.velocity.mdp.terrain_utils import sub_terrain_column_mask

  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  mask = sub_terrain_column_mask(env, stair_terrain_names)
  assert mask is not None and mask.any() and (~mask).any(), "need stair and other columns"
  up_mask = sub_terrain_column_mask(env, tuple(n for n in stair_terrain_names if n in up_terrain_names))
  assert up_mask is not None and up_mask.any() and (mask & ~up_mask).any(), "need climbing and descending stairs columns"
  s = start + (end - start) * ramp(getattr(env, "u_progress", 0.0), p_start, p_end)
  rows = getattr(env, "u_rows", None)
  if rows is None:   # a fresh run, or a checkpoint from before the rows: all from the robots' rows now
    rows = env.u_rows = {k: terrain.terrain_levels.clone() for k in ROWS}
  fails = getattr(env, "u_family_fail", None)
  if fails is None:
    fails = env.u_family_fail = {}
  fixed = eval_rows(env, stair_terrain_names, eval_step_max)
  if len(env_ids) > 0:
    col = terrain.terrain_types[env_ids]
    was = {"up": up_mask[col], "down": mask[col] & ~up_mask[col], "other": ~mask[col]}
    lv = terrain.terrain_levels[env_ids]
    tm = env.termination_manager
    bad = tm.terminated[env_ids] & ~tm.time_outs[env_ids]
    counted = env.common_step_counter > 0
    ev_row = None if fixed is None else fixed[env_ids]
    ev_now = torch.zeros_like(bad) if ev_row is None else ev_row >= 0
    # the finished episodes' payload class and scan mode (both redrawn by reset events, after the curriculum)
    pay = pay_class(env, env_ids)
    pay_t = None if pay is None else torch.tensor(pay, device=bad.device)
    cls_fails = getattr(env, "u_class_fail", None)
    if cls_fails is None:
      cls_fails = env.u_class_fail = {}
    scan = getattr(env, "u_scan_on", None)
    seen = None if scan is None else scan[env_ids] > 0.5
    for k in ROWS:
      own = was[k] if k == "other" else was[k] & ~ev_now   # (an evaluation robot's stairs rows are not the curriculum's)
      rows[k][env_ids] = torch.where(own, lv, rows[k][env_ids])
      tr = was[k] & ~ev_now
      if counted and bool(tr.any()):
        f = float(bad[tr].float().mean())
        a = fail_alpha * min(1.0, float(tr.sum()) / 8.0)   # a handful of episodes moves it less
        fails[k] = f if k not in fails else fails[k] + a * (f - fails[k])
        if k != "other" and pay_t is not None:
          # the same by the finished episode's payload class (2026-11-13: with the payload drawn anew at every reset
          # the heavy episodes ran on rows earned by light ones; teacher_env.sticky_payload)
          for ci, cname in enumerate(PAY_CLASSES):
            trc = tr & (pay_t == ci)
            if bool(trc.any()):
              fc = float(bad[trc].float().mean())
              ac = fail_alpha * min(1.0, float(trc.sum()) / 8.0)
              key = f"{cname}_{k}_fail"
              cls_fails[key] = fc if key not in cls_fails else cls_fails[key] + ac * (fc - cls_fails[key])
      if counted and k != "other" and ev_row is not None:
        e = was[k] & ev_now
        if bool(e.any()):
          j = e.nonzero().flatten().tolist()
          _eval_fail(env, k, ev_row[e].tolist(), bad[e].tolist(), eval_fail_alpha,
                     None if pay is None else [pay[i] for i in j], None if seen is None else seen[e].tolist())
    if ev_row is not None and bool(ev_now.any()):
      # theirs whatever a checkpoint from before the fixed rows restored, or the first reset drew
      for k in ("up", "down"):
        rows[k][env_ids] = torch.where(ev_now, ev_row.to(rows[k].dtype), rows[k][env_ids])
    stair_cols = mask.nonzero().flatten()
    other_cols = (~mask).nonzero().flatten()
    n = len(env_ids)
    on_stairs = torch.rand(n, device=env.device) < s
    pick_s = stair_cols[torch.randint(len(stair_cols), (n,), device=env.device)]
    if other_shares:
      pick_o = torch.multinomial(other_column_weights(env, other_shares, mask), n, replacement=True)
    else:
      pick_o = other_cols[torch.randint(len(other_cols), (n,), device=env.device)]
    new_col = torch.where(on_stairs, pick_s, pick_o)
    new_row = torch.where(up_mask[new_col], rows["up"][env_ids],
                          torch.where(mask[new_col], rows["down"][env_ids], rows["other"][env_ids]))
    terrain.terrain_types[env_ids] = new_col
    terrain.terrain_levels[env_ids] = new_row
    terrain.env_origins[env_ids] = terrain.terrain_origins[new_row, new_col]
  share = mask[terrain.terrain_types].float().mean()
  m = lambda v: (v if fixed is None else v[fixed < 0]).float().mean()   # noqa: E731  (the training robots' rows)
  up, down = m(rows["up"]), m(rows["down"])
  out = {"target": torch.tensor(s), "share": share, "stairs_level": 0.5 * (up + down), "up_level": up,
         "down_level": down, "other_level": m(rows["other"])}
  for k, v in fails.items():
    out[f"{k}_fail"] = torch.tensor(v)
  for k, v in (getattr(env, "u_class_fail", None) or {}).items():   # <class>_<family>_fail, the training robots
    out[k] = torch.tensor(v)
  cls = getattr(env, "u_pay_class", None)
  if cls is not None:   # each payload class's rows (teacher_env.sticky_payload: the class stays with the robot)
    tr_cls = cls if fixed is None else torch.where(fixed < 0, cls, torch.full_like(cls, -1))
    for ci, cname in enumerate(PAY_CLASSES):
      sel = tr_cls == ci
      if bool(sel.any()):
        out[f"{cname}_up_level"] = rows["up"][sel].float().mean()
        out[f"{cname}_down_level"] = rows["down"][sel].float().mean()
  for k, v in (getattr(env, "u_eval_fail", None) or {}).items():
    out[f"eval_{k}_fail"] = torch.tensor(v)
  for k in (getattr(env, "u_eval_split", None) or {}):
    for c in PAY_CLASSES:
      v, _ = eval_light(env, k, cls=c)
      if v is not None:
        out[f"eval_{k}_fail_{c}"] = torch.tensor(v)
  return out


def progress_commands(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  initial: dict,
  final: dict,
  p_switch: float = 0.25,
  lateral_ramp: tuple[float, float, float] | None = None,
) -> dict[str, torch.Tensor]:
  """The command ranges: `initial` until env.u_progress reaches p_switch, `final` from then on (both
  {"lin_vel_x": (lo, hi), "lin_vel_y": ..., "ang_vel_z": ...}, as mdp.commands_vel's stages).

  lateral_ramp (start, p_from, p_to), off by default: before p_switch the lateral range is the initial one scaled so
  that its half-width is `start` m/s until p_from and the initial one from p_to on (2026-11-12, the from-zero Go2W
  teacher: at 300-1000 it 47-56% of its episodes on the other ground failed, 86% of those under a sideways command
  -- a wheeled robot has to step for |vy| above 0.1 m/s, and three quarters of the +-0.4 m/s range ask for that from
  the first iteration, before it can drive)."""
  del env_ids
  ranges = env.command_manager.get_term(command_name).cfg.ranges
  p = getattr(env, "u_progress", 0.0)
  stage = final if p >= p_switch else initial
  for key, value in stage.items():
    setattr(ranges, key, tuple(value))
  if lateral_ramp is not None and stage is initial and initial.get("lin_vel_y") is not None:
    start, p_from, p_to = lateral_ramp
    lo, hi = initial["lin_vel_y"]
    full = max(abs(lo), abs(hi))
    if full > start:
      k = (start + (full - start) * ramp(p, p_from, p_to)) / full
      ranges.lin_vel_y = (lo * k, hi * k)
  return {"final": torch.tensor(float(stage is final)), "vx_max": torch.tensor(float(ranges.lin_vel_x[1])),
          "vy_max": torch.tensor(float(ranges.lin_vel_y[1]))}


def progress_payload(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  full_max: float,
  p_full: float = 0.5,
) -> dict[str, torch.Tensor]:
  """The payload event's heavy_max (mdp.base_payload_point: the heavy robots' payload is uniform on [0, heavy_max])
  = full_max x the smaller of env.u_progress / p_full and 0.5 + 0.5 x the stairs quality q (env.u_quality), at most
  full_max: half the limit by the progress alone, the rest as the robot learns to place its feet on the stairs
  (2026-11-12, the from-zero retrain: the full payload at progress 0.5 came before the robots climbed, and the stairs
  quality stage then had to be learned under it). Read at each robot's reset."""
  del env_ids
  env.u_payload_full = full_max   # the robot's payload limit: the light / heavy shares (constraints.payload_mask)
  q = getattr(env, "u_quality", 0.0)
  if os.environ.get("U_PAYLOAD_Q") == "0":   # a probe's switch (teacher_env._probe_switches): half the limit, no more
    q = 0.0
  limit = full_max * min(ramp(getattr(env, "u_progress", 0.0), 0.0, p_full), 0.5 + 0.5 * q)
  env.event_manager.get_term_cfg(event_name).params["heavy_max"] = limit
  return {"heavy_max": torch.tensor(limit)}
