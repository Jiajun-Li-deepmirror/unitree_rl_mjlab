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
acceptance's stairs the policy fails -- and it is what relaxes the stairs constraints; the stairs constraints'
readings, which are taken over the evaluation robots, come from those rows too (not from rows above the
acceptance's, and not from wherever the curriculum stands). The training robots' rows, the terrain levels behind p
and q's climbing row are the training robots' alone.

Everything else reads env.u_progress:

  stairs share     TEACHER_SHARE until p 0.3, then up to the robot's final share by p 0.8 (mdp.stairs_share's
                   column draw, with p in place of the training step)
  commands         the initial ranges until p 0.25, the final ones after (as mdp.commands_vel's two stages)
  payload          the heavy robots' payload limit rises with p to half the robot's limit at p 0.25 and stays there
                   until the stairs quality begins; the other half follows q (progress_payload: the smaller of
                   p / 0.5 and 0.5 + 0.5 q; mdp.base_payload_point's heavy_max)
  slopes           two sloped tiles among the other ground (SLOPE_NAMES, teacher_env._add_slopes): a fixed share of
                   the other ground's episodes (progress_stairs_share's other_shares), their gradient with the other
                   ground's row; no acceptance item is measured there (slope_envs)
  constraints      the flat ones switch on at their own p and tighten from a loose target to the teacher's own (a
                   margin inside the students' bars); the stairs ones follow q; each relaxes while its own family of
                   ground fails (constraints.py)

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
UP_NAMES = ("stairs_up_narrow", "stairs_up_wide", "walled_up")   # the stairs columns that climb
ROWS = ("up", "down", "other")
SLOPE_NAMES = ("slope_down", "slope_up")   # the sloped tiles (teacher_env._add_slopes): other ground, not measured
# The evaluation robots' stairs rows (module docstring): the rows that can hold a step below this height (m), the
# acceptance's highest (doc/acceptance.md: stairs to 18 cm; the training tiles go to 20). EVAL_FAIL_ALPHA: each
# row's fail rate moves by this much per finished episode (the mean of its first 1 / alpha episodes before that).
EVAL_STEP_MAX, EVAL_FAIL_ALPHA = 0.18, 0.05


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
  gen = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
  heights = [s.step_height_range for name, s in (gen.sub_terrains.items() if gen is not None else ())
             if name in stair_names and getattr(s, "step_height_range", None) is not None]
  if heights:
    lo, hi = gen.difficulty_range
    # a tile of row r draws its difficulty in [r, r + 1) / n_rows of the range, and its step height is linear in it
    lowest = [min(a + (b - a) * (lo + (hi - lo) * r / n_rows) for a, b in heights) for r in range(n_rows)]
    k = max(1, sum(h < step_max - 1e-9 for h in lowest))
  idx = ev.nonzero().flatten()
  fixed = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
  fixed[idx] = (torch.arange(len(idx), device=env.device) // 2) % k
  env.u_eval_rows, env.u_eval_row_count = fixed, k
  print(f"[unified] evaluation robots: {len(idx)} on fixed stairs rows 0..{k - 1} of {n_rows} "
        f"(steps below {step_max} m), {len(idx) / k:.1f} per row")
  return fixed


def _eval_fail(env: ManagerBasedRlEnv, family: str, rows: list[int], failed: list[bool], alpha: float) -> None:
  """Add the evaluation robots' finished stairs episodes (their fixed rows, whether each failed) to the family's
  fail rates: per row in env.u_eval_stat[family] = {"rate": [...], "n": [...]} and, over the rows that have had an
  episode, their mean in env.u_eval_fail[family]."""
  k = env.u_eval_row_count
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
                         for k, v in (getattr(self._env, "u_eval_stat", None) or {}).items()}}
    terrain = getattr(self, "_terrain", None)
    if terrain is not None and getattr(terrain, "terrain_levels", None) is not None:
      out["terrain_levels"] = terrain.terrain_levels.cpu().tolist()
      out["terrain_types"] = terrain.terrain_types.cpu().tolist()
    rows = getattr(self._env, "u_rows", None)
    if rows is not None:
      out["rows"] = {k: v.cpu().tolist() for k, v in rows.items()}
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
    elif n:   # a checkpoint from before the rows: all start at the robot's restored row
      lv = env.scene.terrain.terrain_levels
      env.u_rows = {k: lv.clone() for k in ROWS}
      print(f"[unified] difficulty rows: none in the checkpoint, all start at each robot's restored row "
            f"(mean {float(lv.float().mean()):.2f})")

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
               force_q: bool = False) -> dict[str, torch.Tensor]:
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
    # why the stage's clock runs or waits, for the training records (diagnostics.py)
    self.q_gate = "done" if self.q >= 1.0 else "running" if climbing else "waiting_row"
    if self.q_last_step is not None and climbing:
      self.q = min(1.0, self.q + max(0, step - self.q_last_step) / (STEPS_PER_IT * q_span_it))
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
           "q_running": torch.tensor(float(climbing))}
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
  per stairs family over the evaluation robots on their fixed rows (env.u_eval_fail, eval_<family>_fail).
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
    for k in ROWS:
      own = was[k] if k == "other" else was[k] & ~ev_now   # (an evaluation robot's stairs rows are not the curriculum's)
      rows[k][env_ids] = torch.where(own, lv, rows[k][env_ids])
      tr = was[k] & ~ev_now
      if counted and bool(tr.any()):
        f = float(bad[tr].float().mean())
        a = fail_alpha * min(1.0, float(tr.sum()) / 8.0)   # a handful of episodes moves it less
        fails[k] = f if k not in fails else fails[k] + a * (f - fails[k])
      if counted and k != "other" and ev_row is not None:
        e = was[k] & ev_now
        if bool(e.any()):
          _eval_fail(env, k, ev_row[e].tolist(), bad[e].tolist(), eval_fail_alpha)
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
  for k, v in (getattr(env, "u_eval_fail", None) or {}).items():
    out[f"eval_{k}_fail"] = torch.tensor(v)
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
  limit = full_max * min(ramp(getattr(env, "u_progress", 0.0), 0.0, p_full), 0.5 + 0.5 * q)
  env.event_manager.get_term_cfg(event_name).params["heavy_max"] = limit
  return {"heavy_max": torch.tensor(limit)}
