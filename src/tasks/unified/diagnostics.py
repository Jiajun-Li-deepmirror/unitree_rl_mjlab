"""Training records (go2-depth 2026-11-12; the user: "添加足够多的训练日志和记录，之后只需要看训练记录就能发现训练哪里出现了
问题"). Everything a later look needs to say where a training went wrong is written into the run directory while it
trains; scripts/unified/run_report.py reads only these files:

  <run>/records/timeline.jsonl       one line per event: start (code version, settings, the constraint items, the
                                     terrain's columns, the resume), constraint_on, relax / cap / floor / clock_slow
                                     on and off, q_gate, milestones (progress, quality, payload, rows), checkpoint,
                                     alert / clear, fault, exception (with its traceback), end
  <run>/records/record.jsonl         a snapshot every RECORD_EVERY iterations (and at the first, the last and on an
                                     exception): speed and memory, losses, policy noise, the rollout (explained
                                     variance, action clipping), gradient norms, the curriculum, every constraint's
                                     readings per scan mode / target / multiplier, the live reward weights, the gate
                                     as it would read now, the physics (torque saturation, joint limits), the alerts,
                                     and "ep": every episode that ended in the window, counted per ground
  <run>/records/status.json          the current state, rewritten every STATUS_EVERY iterations: the heartbeat (a run
                                     with no end event and an old status was killed or crashed hard)
  <run>/records/crash_traceback.txt  Python's stacks on a hard crash (segfault, abort), and on `kill -USR1 <pid>`: where
                                     a hung training is stuck, without stopping it

Two parts:

  `episodes`   a curriculum term (teacher_env: right after `progress`, before the terms that redraw the robot's
               column and row), so at each reset it still sees the finished episode: its ground (sub-terrain), row,
               how it ended (time-out, or which termination term), which bodies touched the ground on an illegal
               contact, scan mode, payload class, where the payload sits (centre / front / back / side), command
               class, training or evaluation robot, and for a failed one where it ended (distance from the spawn point, seconds, nose-down pitch, height over the spawn ground,
               commanded speed). Kept on the GPU and counted once per iteration.
  `Recorder`   attach(runner): wraps the runner's logger.log (each iteration), learn (start / end / exception), save,
               load and the optimizer's step (gradient norms after clipping). It reads the curriculum's state from
               the environment (u_progress_term, u_constraints, u_rows, ...) -- what the trainings act on, not the
               console's averages.

Alerts (ALERTS below; thresholds are judgment calls, records only -- nothing here gates or changes a training) are
raised after a condition has held for a number of evaluations, repeated every REPEAT_IT iterations while it lasts, and
cleared when it is gone. Extra TensorBoard tags: Diag/*.

The reference schedule (expectations(); the user, 2026-11-12: a version that does not work has to show within two or
three hours, not after six): what a teacher has to have reached by which iteration. Each milestone's reaching is a
timeline event (milestone "expect:<name>"); one not reached by its iteration is the alert late:<name>. The record's
and the status' "expect" say which are reached.

A fault in the records never stops a training: every hook is guarded, the first fault of each is written as a `fault`
event with its traceback, and that part stays off. U_RECORDS=0 turns the records off."""

from __future__ import annotations

import atexit
import faulthandler
import inspect
import json
import math
import os
import re
import signal
import socket
import sys
import time
import traceback
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from .constraints import HEAVY_FRAC, LIGHT_FRAC, MODES
from .curriculum import (
  MAX_FAIL_RATE, PAY_CLASSES, Q_ROW, Q_SPAN_IT, SPLIT_MODES, STEPS_PER_IT, TIME_SPAN_IT, TIME_START_IT, UP_NAMES,
  eval_bands, eval_light, row_bands,
)

STATUS_EVERY, RECORD_EVERY, TAGS_EVERY = 10, 20, 100   # iterations: status + alerts, snapshots, all console tags
REPEAT_IT, CLEAR_HOLD, FLIP_IT = 1000, 3, 50
FAIL_FROM_IT = 800
FAIL_FROM_Q = 0.5   # a stairs ground's fail_high is judged from this stairs quality on (the evaluation robots' episodes)   # the per-ground fail alerts start here: before it the robots are learning to stand and walk
SCAN, PAY, CMD, ROLE = ("blind", "seen"), ("light", "mid", "heavy"), ("stand", "straight", "turning", "sideways"), \
  ("train", "eval")
# where a mid / heavy robot's payload puts the base's centre of mass (the user, 2026-11-12: the payload's mount on the
# final machine is not known, and with a payload the robot must not fall -- so the records say where the falling
# payloads sit): within LOAD_SHIFT_M of the unloaded one, else the larger of the forward / backward and sideways shift
LOAD, LOAD_SHIFT_M = ("centre", "front", "back", "side"), 0.03
CLASSES = (("scan", SCAN), ("pay", PAY), ("cmd", CMD), ("role", ROLE), ("load", LOAD))
# an episode's packed columns (floats), and the histograms' edges of where the failed ones end
F_GROUND, F_ROW, F_BAD, F_TIMEOUT, F_SCAN, F_PAY, F_CMD, F_ROLE, F_SEC, F_DIST, F_PITCH, F_Z, F_SPEED, F_LOAD = \
  range(14)
F_CLASS = {"scan": F_SCAN, "pay": F_PAY, "cmd": F_CMD, "role": F_ROLE, "load": F_LOAD}
WHERE = {
  "dist": (F_DIST, [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),        # m from the spawn point
  "sec": (F_SEC, [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]),              # s into the episode
  "pitch": (F_PITCH, [-40.0, -25.0, -15.0, -5.0, 5.0, 15.0, 25.0, 40.0]),   # deg, nose down positive
  "z": (F_Z, [-1.0, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0]),         # m, the base over the spawn ground
  "speed": (F_SPEED, [0.1, 0.3, 0.5, 0.8, 1.2]),                      # m/s commanded
}
# The alerts: name -> what it means (run_report.py and doc/unified_teacher.md list them from here).
ALERTS = {
  "nonfinite": "a loss, the action noise or the robots' state is not finite",
  "nan_resets": "episodes ended by the nan termination",
  "std_rising": "the action noise rose by a quarter in 300 iterations (above 0.5)",
  "std_high": "the action noise is above 1.0 after iteration 1000",
  "action_clip": "more than 5% of the action components are at the action clip",
  "fail_high": "more than half of a ground's episodes fail (from iteration FAIL_FROM_IT; a stairs ground: of the "
               "evaluation robots' episodes on their fixed rows, from stairs quality FAIL_FROM_Q)",
  "payload_fail": "on a ground the robots carrying a payload (mid and heavy) fail 0.1 more often than the light ones",
  "late": "a milestone of the reference schedule (expectations) was not reached by its iteration",
  "no_episodes": "a ground had no finished episode for 500 iterations",
  "len_drop": "the mean episode length fell by 30% in 300 iterations",
  "level_falling": "the mean terrain level is a row below its recent maximum",
  "eval_gap": "off the stairs the training robots fail 0.2 more than the mean-action robots: the exploration noise "
              "is what falls",
  "value_loss_spike": "the value loss is 10x its recent median",
  "ev_low": "the critic explains less than 10% of the returns' variance",
  "lr_floor": "the adaptive learning rate sits at its floor",
  "fps_drop": "the training runs at less than half its usual speed",
  "q_waiting": "progress is past 0.5 and the stairs quality stage has not begun",
  "q_stalled": "the stairs quality stage has not moved for 300 iterations",
  "clock_slow": "the progress clock runs at a quarter pace (too many of the other ground's episodes fail)",
  "relax": "a ground family's constraints are on and relaxed (the other ground: it fails too often; the stairs: the "
           "light evaluation robots' fail rate on their fixed rows rose above its reference)",
  "cap": "a constraint's multiplier sits at its upper limit",
  "unmeasured": "an enforced constraint that is on has no new readings",
  "torque_sat": "more than 20% of a joint group's torques are at the limit",
}


def _light_brief(cur: dict) -> dict | None:
  """A record's cur["eval_light"] for an event: per stairs family the light evaluation robots' fail rate, its
  reference and episodes."""
  light = cur.get("eval_light")
  return {k: {x: d.get(x) for x in ("fail", "ref", "n")} for k, d in light.items()} if light else None


def low_fail(cur: dict, family: str = "up", below: float = Q_ROW) -> float | None:
  """The evaluation robots' fail rate on their fixed stairs rows below `below` (a record's cur["eval_fail_rows"];
  curriculum.eval_rows): the mean over the rows that have had an episode, None without one."""
  v = [x for x in ((cur.get("eval_fail_rows") or {}).get(family) or [])[:int(math.ceil(below))] if x is not None]
  return sum(v) / len(v) if v else None


def expectations() -> list[dict]:
  """The reference schedule (2026-11-12, the user: "如果版本不行，不要耗 6 小时才知道要修，大概两三个小时就知道版本哪里有问题"):
  what a teacher's training has to have reached by which iteration, from the curriculum's clock (curriculum.py: p
  leaves 0 at TIME_START_IT and covers 0 -> 1 in TIME_SPAN_IT while few enough episodes fail; the clock runs at a
  quarter of its pace over the first ~700 iterations, before the robots walk -- the allowance in the iterations
  below). A milestone not reached by its iteration is the alert late:<name>, and the report's first finding.

  The iterations are judgment calls, set against the 11-12 Go2 teacher on the slower clock (seed 0: all episodes'
  fail rate 0.25 at iteration 800 and 0.21 at 1000; the stairs share began to rise at 5000 (p 0.3), the stairs rows
  were at 2.5 after 1500 more iterations and at 3 after 2000, with the share rising at less than half of this
  clock's pace); to be set again from the first run on this schedule. `test`: reached, from the record's "cur".

  The stairs are read where the numbers mean something (curriculum.py's docstring): the training robots' climbing
  row (the curriculum promotes a robot until it fails, so its fail rate there is ~0.6 whatever it climbs), and the
  evaluation robots' fail rate on their fixed rows."""
  t0, span = TIME_START_IT, TIME_SPAN_IT
  fam = lambda c, k: (c.get("family_fail") or {}).get(k)   # noqa: E731
  row = lambda c, k: (c.get("rows") or {}).get(k)          # noqa: E731
  below = lambda v, bar: v is not None and v <= bar        # noqa: E731
  ef = lambda c: (c.get("eval_fail") or {}).get("up")      # noqa: E731
  return [
    {"name": "walk", "by": t0 + 500, "what": f"the flat ground's fail rate at {MAX_FAIL_RATE} or below",
     "test": lambda c: below(fam(c, "other") if fam(c, "other") is not None else c.get("fail"), MAX_FAIL_RATE),
     "show": lambda c: {"other_fail": fam(c, "other"), "fail": c.get("fail")}},
    {"name": "commands", "by": t0 + span // 4 + 1000, "what": "progress 0.25 (the full command ranges)",
     "test": lambda c: (c.get("p") or 0.0) >= 0.25, "show": lambda c: {"p": c.get("p"), "fail": c.get("fail")}},
    {"name": "climb_first", "by": t0 + span // 2,
     "what": f"on the low climbing rows (below {Q_ROW:g}) half of the mean-action robots' episodes or more succeed",
     "test": lambda c: below(low_fail(c), 0.5),
     "show": lambda c: {"low_fail": low_fail(c), "eval_up_fail": ef(c), "up_row": row(c, "up")}},
    {"name": "climb_low", "by": t0 + span // 2 + 1000, "what": "the climbing row at 1.5",
     "test": lambda c: (row(c, "up") or 0.0) >= 1.5,
     "show": lambda c: {"up_row": row(c, "up"), "eval_up_fail": ef(c)}},
    {"name": "climb", "by": t0 + 3 * span // 4 + 1000,
     "what": f"the climbing row at {Q_ROW}: the stairs quality stage runs",
     "test": lambda c: (c.get("q") or 0.0) > 0.0 or c.get("q_gate") in ("running", "done"),
     "show": lambda c: {"up": c.get("up"), "eval_up_fail": ef(c), "q_gate": c.get("q_gate")}},
    {"name": "p_done", "by": t0 + span + 1000, "what": "progress 1",
     "test": lambda c: (c.get("p") or 0.0) >= 1.0, "show": lambda c: {"p": c.get("p"), "fail": c.get("fail")}},
    {"name": "q_done", "by": t0 + span + Q_SPAN_IT + 1000, "what": "stairs quality 1",
     "test": lambda c: (c.get("q") or 0.0) >= 1.0,
     "show": lambda c: {"q": c.get("q"), "up": c.get("up"), "eval_up_fail": ef(c), "q_gate": c.get("q_gate")}},
  ]


def _clean(x):
  """JSON-safe: numbers rounded to 5 significant digits (never coarser than one decimal: a clock time keeps its
  seconds), non-finite ones as strings, tensors and arrays as lists."""
  if isinstance(x, dict):
    return {str(k): _clean(v) for k, v in x.items()}
  if isinstance(x, (list, tuple, deque)):
    return [_clean(v) for v in x]
  if isinstance(x, torch.Tensor):
    return _clean(x.detach().cpu().tolist())
  if isinstance(x, np.ndarray):
    return _clean(x.tolist())
  if isinstance(x, (bool, np.bool_)):
    return bool(x)
  if isinstance(x, (int, np.integer)):
    return int(x)
  if isinstance(x, (float, np.floating)):
    x = float(x)
    if not math.isfinite(x):
      return str(x)
    # (2026-11-12, the from-zero runs' first minutes: 5 digits turned every "t" -- time.time() -- into 1794500000.0,
    # so the records had no times: no DEAD alarm, no iterations per hour in the report)
    return round(x, 1) if abs(x) >= 1e4 else float(f"{x:.5g}")
  if x is None or isinstance(x, str):
    return x
  return str(x)


def column_grounds(terrain) -> list[str] | None:
  """Each curriculum column's sub-terrain name (terrain_utils.sub_terrain_column_mask's allocation); None for a
  terrain without columns."""
  gen = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
  if gen is None or getattr(terrain, "terrain_types", None) is None:
    return None
  names = list(gen.sub_terrains.keys())
  prop = np.array([c.proportion for c in gen.sub_terrains.values()], dtype=float)
  cum = np.cumsum(prop / prop.sum())
  return [names[int(np.min(np.where(i / gen.num_cols + 0.001 < cum)[0]))] for i in range(gen.num_cols)]


class episodes:
  """Curriculum term: every finished episode, counted per ground (module docstring). Returns nothing (no console
  lines); the recorder collects the counts with take()."""

  def __init__(self, cfg, env):
    self.rows_f: list[torch.Tensor] = []
    self.rows_i: list[torch.Tensor] = []
    self.last_flush = 0
    self.meta: dict | None = None   # built at the first finished episode: the scene and the managers are complete
    self.win: dict | None = None
    self.fault: str | None = None   # the first fault's traceback (the recorder writes it out)
    env.u_episodes = self

  def reset(self, env_ids=None) -> None:
    pass

  def __call__(self, env, env_ids, stair_terrain_names: tuple[str, ...] = (), flush_steps: int = STEPS_PER_IT) -> None:
    if self.fault is not None:
      return None
    try:
      self._record(env, env_ids, stair_terrain_names)
      if env.common_step_counter - self.last_flush >= flush_steps:
        self.flush(env.common_step_counter)
    except Exception:
      self.fault = traceback.format_exc()
      self.rows_f, self.rows_i = [], []
    return None

  def _build(self, env, stair_names) -> None:
    terrain = env.scene.terrain
    cols = column_grounds(terrain)
    grounds = list(dict.fromkeys(cols)) if cols else ["all"]
    tm = env.termination_manager
    causes = [n for n in tm.active_terms if not tm.get_term_cfg(n).time_out]
    sensor, thr, bodies = None, 10.0, []
    if "illegal_contact" in causes:
      try:
        params = tm.get_term_cfg("illegal_contact").params
        sensor = env.scene[params["sensor_name"]]
        thr = float(params.get("force_threshold", 10.0))
        bodies = [sl.primary_name for sl in sensor._slots][:62]
      except Exception:
        sensor, bodies = None, []
    self.meta = {
      "grounds": grounds,
      "columns": cols,
      "family": {g: ("other" if g not in stair_names else "up" if g in UP_NAMES else "down") for g in grounds},
      "rows": max(1, int(getattr(terrain, "max_terrain_level", 1) or 1)),
      "causes": causes,
      "bodies": bodies,
    }
    self._col2g = None if cols is None else torch.tensor([grounds.index(c) for c in cols], device=env.device)
    self._sensor, self._thr = sensor, thr
    self._shifts = torch.arange(len(bodies), device=env.device)
    self._root = self._ipos0 = None
    try:   # the base's own centre of mass: the payload event moves it (mdp.base_payload_point)
      self._root = env.scene["robot"].indexing.root_body_id
      self._ipos0 = env.sim.get_default_field("body_ipos")[self._root]
    except Exception:
      self._ipos0 = None
    self.win = self._empty()

  def _empty(self) -> dict:
    m = self.meta
    g = len(m["grounds"])
    z = lambda *shape: np.zeros(shape, dtype=np.float64)   # noqa: E731
    return {"n": z(g), "fail": z(g), "timeout": z(g), "sec": z(g), "dist": z(g), "cause": z(g, len(m["causes"])),
            "row": z(g, m["rows"], 2), **{k: z(g, len(names), 2) for k, names in CLASSES},
            "where": {k: z(g, len(e) + 1) for k, (_, e) in WHERE.items()}, "bodies": Counter()}

  def _record(self, env, env_ids, stair_names) -> None:
    if not isinstance(env_ids, torch.Tensor) or env.common_step_counter == 0 or env_ids.numel() == 0:
      return
    tm = env.termination_manager
    ids = env_ids[tm.dones[env_ids]]
    n = int(ids.numel())
    if n == 0:
      return
    if self.meta is None:
      self._build(env, stair_names)
    dev = ids.device
    terrain = env.scene.terrain
    zero = torch.zeros(n, device=dev)
    if self._col2g is not None:
      ground, row = self._col2g[terrain.terrain_types[ids]].float(), terrain.terrain_levels[ids].float()
    else:
      ground, row = zero, zero
    time_out = tm.time_outs[ids]
    bad = tm.terminated[ids] & ~time_out
    cause = torch.zeros(n, dtype=torch.long, device=dev)
    for j, name in enumerate(self.meta["causes"]):
      cause |= tm.get_term(name)[ids].long() << j
    on = getattr(env, "u_scan_on", None)
    scan = zero - 1.0 if on is None else (on[ids] > 0.5).float()
    kg, limit = getattr(env, "payload_kg", None), float(getattr(env, "u_payload_full", 0.0) or 0.0)
    if kg is None or limit <= 0.0:
      pay = zero - 1.0
    else:
      k = kg.reshape(env.num_envs, -1)[ids, 0]
      pay = torch.where(k <= LIGHT_FRAC * limit, 0.0, torch.where(k >= HEAVY_FRAC * limit, 2.0, 1.0))
    load = zero - 1.0
    if self._ipos0 is not None:
      try:
        d = env.sim.model.body_ipos[ids, self._root] - self._ipos0
        dx, dy = d[:, 0], d[:, 1]
        place = torch.where(torch.hypot(dx, dy) < LOAD_SHIFT_M, 0.0,
                            torch.where(dx.abs() >= dy.abs(), torch.where(dx > 0, 1.0, 2.0), 3.0))
        load = torch.where(pay >= 1.0, place, load)   # the light robots are not counted
      except Exception:
        self._ipos0 = None
    try:
      cmd = env.command_manager.get_command("twist")[ids]
      vx, vy, wz = cmd[:, 0].abs(), cmd[:, 1].abs(), cmd[:, 2].abs()
      # fail_probe.py's classes: stand, sideways, turning, else straight
      cmd_class = torch.where(torch.maximum(torch.maximum(vx, vy), wz) < 0.1, 0.0,
                              torch.where(vy > 0.15, 3.0, torch.where(wz > 0.3, 2.0, 1.0)))
      speed = torch.hypot(cmd[:, 0], cmd[:, 1])
    except Exception:
      cmd_class, speed = zero - 1.0, zero
    ev = getattr(env, "u_eval_mask", None)
    role = zero if ev is None else ev[ids].float()
    robot = env.scene["robot"]
    origin = terrain.env_origins[ids] if getattr(terrain, "env_origins", None) is not None else 0.0
    rel = robot.data.root_link_pos_w[ids] - origin
    q = robot.data.root_link_quat_w[ids]   # w, x, y, z
    pitch = torch.rad2deg(torch.asin(torch.clamp(2.0 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1.0, 1.0)))
    sec = env.episode_length_buf[ids].float() * float(env.step_dt)
    body = torch.zeros(n, dtype=torch.long, device=dev)
    if self._sensor is not None and len(self.meta["bodies"]) > 0:
      d = self._sensor.data
      if getattr(d, "force_history", None) is not None:
        hit = (torch.norm(d.force_history[ids], dim=-1) > self._thr).any(dim=-1)
      else:
        hit = d.found[ids] > 0
      hit = hit.reshape(n, -1)[:, :len(self.meta["bodies"])]
      body = (hit.long() << self._shifts[:hit.shape[1]]).sum(dim=1)
    self.rows_f.append(torch.stack([ground, row, bad.float(), time_out.float(), scan, pay, cmd_class, role, sec,
                                    torch.hypot(rel[:, 0], rel[:, 1]), pitch, rel[:, 2], speed, load], dim=1).float())
    self.rows_i.append(torch.stack([cause, body], dim=1))

  def flush(self, step: int | None = None) -> None:
    """The episodes gathered on the GPU into the window's counts (one transfer)."""
    if step is not None:
      self.last_flush = step
    if not self.rows_f:
      return
    f = torch.cat(self.rows_f).cpu().numpy().astype(np.float64)
    i = torch.cat(self.rows_i).cpu().numpy()
    self.rows_f, self.rows_i = [], []
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    w, m = self.win, self.meta
    g = np.clip(f[:, F_GROUND].astype(int), 0, len(m["grounds"]) - 1)
    bad = f[:, F_BAD] > 0.5
    np.add.at(w["n"], g, 1.0)
    np.add.at(w["fail"], g, bad)
    np.add.at(w["timeout"], g, f[:, F_TIMEOUT] > 0.5)
    np.add.at(w["sec"], g, f[:, F_SEC])
    np.add.at(w["dist"], g, f[:, F_DIST])
    for j in range(len(m["causes"])):
      np.add.at(w["cause"], (g, j), (i[:, 0] >> j) & 1)
    row = np.clip(f[:, F_ROW].astype(int), 0, m["rows"] - 1)
    np.add.at(w["row"], (g, row, 0), 1.0)
    np.add.at(w["row"], (g, row, 1), bad)
    for key, col in F_CLASS.items():
      c = np.floor(f[:, col]).astype(int)
      ok = (c >= 0) & (c < w[key].shape[1])
      np.add.at(w[key], (g[ok], c[ok], 0), 1.0)
      np.add.at(w[key], (g[ok], c[ok], 1), bad[ok])
    for key, (col, edges) in WHERE.items():
      np.add.at(w["where"][key], (g[bad], np.searchsorted(edges, f[bad, col], side="right")), 1.0)
    touched = bad & (i[:, 1] != 0)
    w["bodies"].update(zip(g[touched].tolist(), i[touched, 1].tolist()))

  def take(self) -> dict | None:
    """The counts since the last take (None before the first finished episode)."""
    if self.meta is None:
      return None
    self.flush()
    w, self.win = self.win, self._empty()
    return w


def merge(a: dict | None, b: dict | None) -> dict | None:
  """Two windows' episode counts added (take()'s dicts)."""
  if a is None or b is None:
    return a if b is None else b
  out = {}
  for k, v in a.items():
    if k == "where":
      out[k] = {x: v[x] + b[k][x] for x in v}
    elif k == "bodies":
      out[k] = v + b[k]
    else:
      out[k] = v + b[k]
  return out


def episodes_json(w: dict | None, meta: dict | None) -> dict:
  """A window's counts per ground as the record's "ep" block: {ground: {"n", "fail", "timeout", "sec" (mean),
  "dist" (mean), "family" (up / down: stairs; other), "cause": {term: n}, "row": {row: [n, fail]}, "scan" / "pay" / "cmd" / "role" / "load": {class:
  [n, fail]} ("load": where a mid / heavy robot's payload puts the base's centre of mass, LOAD),
  "bodies": {"a+b": n} (the bodies touching on a failed episode's illegal contact, the commonest sets), "where":
  {quantity: histogram of the failed episodes}}}. Grounds without episodes are left out."""
  if w is None or meta is None:
    return {}
  out = {}
  for gi, name in enumerate(meta["grounds"]):
    n = int(w["n"][gi])
    if n == 0:
      continue
    d = {"n": n, "fail": int(w["fail"][gi]), "timeout": int(w["timeout"][gi]), "sec": w["sec"][gi] / n,
         "dist": w["dist"][gi] / n, "family": (meta.get("family") or {}).get(name, "other"),
         "cause": {c: int(v) for c, v in zip(meta["causes"], w["cause"][gi]) if v > 0},
         "row": {str(r): [int(v[0]), int(v[1])] for r, v in enumerate(w["row"][gi]) if v[0] > 0}}
    for key, names in CLASSES:
      d[key] = {c: [int(v[0]), int(v[1])] for c, v in zip(names, w[key][gi]) if v[0] > 0}
    sets = Counter({mask: k for (g, mask), k in w["bodies"].items() if g == gi})
    if sets:
      d["bodies"] = {"+".join(b for j, b in enumerate(meta["bodies"]) if mask >> j & 1): int(k)
                     for mask, k in sets.most_common(8)}
    if d["fail"] > 0:
      d["where"] = {k: [int(x) for x in w["where"][k][gi]] for k in WHERE}
    out[name] = d
  return out


def _effort(actuator) -> float | None:
  a = actuator
  while not hasattr(a, "effort_limit") and hasattr(a, "base_cfg"):
    a = a.base_cfg
  return getattr(a, "effort_limit", None)


class physics:
  """The robots' state right now, over every robot: per joint group (hip / thigh / calf / wheel) the share of
  torques at 98% of the limit or more, the share of leg joints within 0.01 rad of a soft limit, the leg joints'
  speeds, the base tilt, how many states are not finite, and (with stick_slip.py) the share of the feet in contact
  that slide."""

  def __init__(self, env):
    robot = env.scene["robot"]
    self.env = env
    self.robot = robot
    self.groups: dict[str, tuple[torch.Tensor, float]] = {}
    names = list(robot.actuator_names)
    for a in env.cfg.scene.entities["robot"].articulation.actuators:
      expr = tuple(getattr(a, "target_names_expr", ()) or getattr(getattr(a, "base_cfg", None), "target_names_expr", ()))
      effort = _effort(a)
      if not expr or effort is None:
        continue
      ids = [i for i, n in enumerate(names) if any(re.fullmatch(e, n) for e in expr)]
      if not ids:
        continue
      group = next((g for g in ("hip", "thigh", "calf", "wheel") if any(g in e for e in expr)), expr[0])
      self.groups[group] = (torch.tensor(ids, device=env.device), float(effort))
    legs = [i for i, n in enumerate(robot.joint_names) if "wheel" not in n]
    self.legs = torch.tensor(legs, device=env.device)

  def __call__(self) -> dict:
    d = self.robot.data
    vals, keys = [], []
    force = d.actuator_force
    for g, (ids, effort) in self.groups.items():
      keys.append(f"sat_{g}")
      vals.append((force[:, ids].abs() >= 0.98 * effort).float().mean())
    pos, vel = d.joint_pos[:, self.legs], d.joint_vel[:, self.legs]
    lim = d.soft_joint_pos_limits[:, self.legs]
    real = torch.isfinite(lim).all(dim=-1) & (lim[..., 1] > lim[..., 0])
    at = ((pos <= lim[..., 0] + 0.01) | (pos >= lim[..., 1] - 0.01)) & real
    speed = vel.abs().flatten()
    tilt = torch.rad2deg(torch.acos(torch.clamp(-d.projected_gravity_b[:, 2], -1.0, 1.0)))
    keys += ["joint_limit", "joint_vel_p99", "joint_vel_max", "tilt_p90", "tilt_max", "nonfinite"]
    vals += [at.float().sum() / real.float().sum().clamp(min=1.0),
             torch.quantile(speed[:: max(1, speed.numel() // 100000)].float(), 0.99), speed.max(),
             torch.quantile(tilt.float(), 0.9), tilt.max(),
             (~torch.isfinite(d.joint_pos)).float().sum() + (~torch.isfinite(d.root_link_pos_w)).float().sum()]
    sliding = getattr(self.env, "u_foot_sliding", None)   # stick_slip.py: the share of the feet in contact that slide
    if sliding is not None:
      keys.append("foot_sliding")
      vals.append(sliding)
    return dict(zip(keys, torch.stack([v.float() for v in vals]).cpu().tolist()))


class Recorder:
  """attach()'s object (module docstring)."""

  def __init__(self, runner, kind: str):
    self.runner, self.kind = runner, kind
    self.env = runner.env.unwrapped
    self.dir = Path(runner.logger.log_dir) / "records"
    self.started = self.ended = False
    self.faults: set[str] = set()
    self.resume: dict | None = None
    self.hist: dict[str, deque] = {}
    self.flips: dict[tuple, dict] = {}
    self.miles: dict[str, int] = {}
    self.alerts: dict[str, dict] = {}
    self.grads: list[torch.Tensor] = []
    self.acc: dict[str, list[float]] = {}     # per-iteration numbers of the snapshot's window
    self.ep_win: dict | None = None           # the episodes since the last snapshot
    self.ep_recent: deque = deque(maxlen=30)  # per evaluation: (n, fail, ...) per ground, for the alerts (300 it)
    self.expect = expectations() if kind == "teacher" else []
    self.reached: dict[str, int] = {}         # the reference schedule's milestones: name -> the iteration reached
    self.phys = None
    self.items: list[dict] = []
    self.last: dict = {}                      # the latest evaluation (status, the exception's snapshot)
    self.start_it = self.total_it = None
    self.max_grad = float(getattr(runner.alg, "max_grad_norm", 0.0) or 0.0)
    self._fh = None
    self._wrap()

  # ---- files ----
  def _append(self, name: str, obj: dict) -> None:
    self.dir.mkdir(parents=True, exist_ok=True)
    with open(self.dir / name, "a") as f:
      f.write(json.dumps(_clean(obj), ensure_ascii=False, separators=(",", ":")) + "\n")

  def event(self, ev: str, it: int | None = None, **data) -> None:
    self._append("timeline.jsonl", {"ev": ev, "it": it, "t": round(time.time(), 1), **data})

  def _status(self, **extra) -> None:
    self.dir.mkdir(parents=True, exist_ok=True)
    tmp = self.dir / "status.json.tmp"
    tmp.write_text(json.dumps(_clean({**self.last.get("status", {}), **extra}), ensure_ascii=False))
    os.replace(tmp, self.dir / "status.json")

  def _fault(self, where: str, trace: str) -> None:
    """A fault of the records themselves: written once per `where`; that part stays off, the training goes on."""
    if where in self.faults:
      return
    self.faults.add(where)
    print(f"[records] fault in {where} (the training goes on; this record stays off):\n{trace}", flush=True)
    try:
      self.event("fault", getattr(self.runner, "current_learning_iteration", None), where=where, trace=trace[-3000:])
    except Exception:
      pass

  def _safe(self, where: str, fn, *args):
    """fn(*args), None on a fault (and from then on)."""
    if where in self.faults:
      return None
    try:
      return fn(*args)
    except Exception:
      self._fault(where, traceback.format_exc())
      return None

  # ---- hooks ----
  def _wrap(self) -> None:
    r = self.runner
    log = r.logger.log
    sig = inspect.signature(log)

    def logged(*a, **k):
      try:
        args = dict(sig.bind(*a, **k).arguments)
      except Exception:
        args = dict(k)
      tags = self._safe("tags", self._tags, args)   # before the logger clears the iteration's console values
      out = log(*a, **k)
      self._safe("iteration", self.on_iteration, args, tags)
      return out

    r.logger.log = logged
    alg = r.alg
    opt = getattr(alg, "optimizer", None)
    nets = [[p for p in m.parameters() if p.requires_grad]
            for m in (getattr(alg, "actor", None), getattr(alg, "critic", None)) if isinstance(m, torch.nn.Module)]
    if opt is not None and nets:
      step = opt.step

      def norm(ps):
        g = [p.grad.detach().pow(2).sum() for p in ps if p.grad is not None]
        return torch.stack(g).sum().sqrt() if g else torch.zeros((), device=ps[0].device)

      def stepped(*a, **k):
        if "grad" not in self.faults:
          try:
            self.grads.append(torch.stack([norm(ps) for ps in nets]))
          except Exception:
            self._fault("grad", traceback.format_exc())
        return step(*a, **k)

      opt.step = stepped
    learn, save, load = r.learn, r.save, r.load

    def learning(*a, **k):
      self._safe("start", self.on_start, a, k)
      try:
        out = learn(*a, **k)
      except BaseException as e:
        self._safe("end", self.on_end, "exception", e)
        raise
      self._safe("end", self.on_end, "done", None)
      return out

    def saving(path, *a, **k):
      out = save(path, *a, **k)
      self._safe("checkpoint", lambda: self.event("checkpoint", r.current_learning_iteration,
                                                  file=os.path.basename(str(path))))
      return out

    def loading(path, *a, **k):
      try:
        out = load(path, *a, **k)
      except BaseException as e:
        # a checkpoint that does not load, into a fresh run directory's records only (not into an older run's)
        if not list(Path(r.logger.log_dir).glob("model_*.pt")):
          self._safe("load", lambda: self.event("exception", None, where="load", checkpoint=str(path),
                                                error=repr(e)[:500], trace=traceback.format_exc()[-6000:]))
        raise
      self.resume = {"checkpoint": str(path), "it": r.current_learning_iteration}
      return out

    r.learn, r.save, r.load = learning, saving, loading
    atexit.register(self._atexit)

  def _atexit(self) -> None:
    if self.started and not self.ended:
      self._safe("end", self.on_end, "exit", None)

  def on_start(self, args, kwargs) -> None:
    r, env = self.runner, self.env
    self.started = True
    self.dir.mkdir(parents=True, exist_ok=True)
    try:   # hard crashes and `kill -USR1`: Python's stacks into the records
      self._fh = open(self.dir / "crash_traceback.txt", "a")
      faulthandler.enable(file=self._fh, all_threads=True)
      if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=self._fh, all_threads=True, chain=False)
    except Exception:
      self._fh = None
    try:   # `kill` (SIGTERM; a background job ignores SIGINT, so jobs/unified/teacher.sh stops its seeds with it): the
      # records get their last record and their end ("stopped"), then the signal does what it would have done
      def _term(signum, frame):
        self._safe("end", self.on_end, "stopped", None)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
      signal.signal(signal.SIGTERM, _term)
    except Exception:   # not the main thread
      pass
    try:
      self.items = list(env.curriculum_manager.get_term_cfg("u_constraints").params["items"])
    except Exception:
      self.items = []
    ep = getattr(env, "u_episodes", None)
    terrain = getattr(env.scene, "terrain", None)
    tm = env.termination_manager
    cfg = r.cfg if isinstance(getattr(r, "cfg", None), dict) else {}
    alg_cfg = cfg.get("algorithm", {}) if isinstance(cfg.get("algorithm"), dict) else {}
    self._safe("carry", self._carry)
    self.event(
      "start", r.current_learning_iteration, kind=self.kind, code=getattr(r, "_commit", None),
      run_dir=str(r.logger.log_dir), host=socket.gethostname(), pid=os.getpid(), argv=sys.argv,
      env_vars={k: v for k, v in sorted(os.environ.items()) if k.startswith(("U_", "UNIFIED_", "RESUME", "TRAIN_LOG"))},
      num_envs=env.num_envs, eval_envs=int(env.u_eval_mask.sum()) if getattr(env, "u_eval_mask", None) is not None else 0,
      iterations=kwargs.get("num_learning_iterations", args[0] if args else None),
      resumed=self.resume, segments=getattr(r, "_segments", []) + [getattr(r, "_segment", {})],
      rl={k: cfg.get(k) for k in ("num_steps_per_env", "save_interval", "seed", "clip_actions", "experiment_name")},
      algorithm={k: v for k, v in alg_cfg.items() if isinstance(v, (int, float, str, bool))},
      clip_actions=getattr(r.env, "clip_actions", None),
      items=[{k: c.get(k) for k in ("name", "metric", "stat", "sense", "target", "tight", "loose", "activate", "by",
                                    "terms", "min_scale", "max_scale", "enforce", "family", "modes", "lr")}
             for c in self.items],
      columns=column_grounds(terrain),
      rows=int(getattr(terrain, "max_terrain_level", 0) or 0),
      terminations={n: bool(tm.get_term_cfg(n).time_out) for n in tm.active_terms},
      rewards={n: env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager.active_terms},
      actions=dict(zip(env.action_manager.active_terms, env.action_manager.action_term_dim)),
      where_edges={k: e for k, (_, e) in WHERE.items()}, alerts=ALERTS,
      expect=[{k: e[k] for k in ("name", "by", "what")} for e in self.expect], reached=dict(self.reached),
      clock={"start": TIME_START_IT, "span": TIME_SPAN_IT, "q_span": Q_SPAN_IT, "q_row": Q_ROW,
             "max_fail": MAX_FAIL_RATE, "gate_hold": 500},
      every={"status": STATUS_EVERY, "record": RECORD_EVERY, "tags": TAGS_EVERY},
      torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      episodes="on" if ep is not None else "off (no u_episodes term in this environment)",
    )

  def _carry(self) -> None:
    """On a resume: the reference schedule's milestones that the resumed run's records hold as reached up to the
    checkpoint's iteration (its events, and its own start event's carried ones) -- a milestone is reached once per
    training, not once per run. Without those records a milestone already met counts from this run's first
    evaluation."""
    res = self.resume
    if not res or not self.expect:
      return
    path = Path(res["checkpoint"]).parent / "records" / "timeline.jsonl"
    if not path.is_file():
      return
    names = {e["name"] for e in self.expect}
    with open(path, errors="replace") as f:
      for line in f:
        try:
          e = json.loads(line)
        except ValueError:
          continue   # a line cut off by a kill
        if e.get("ev") == "start":
          got = e.get("reached") or {}
        elif e.get("ev") == "milestone" and str(e.get("name", "")).startswith("expect:"):
          got = {e["name"][7:]: e.get("it")}
        else:
          continue
        for k, v in got.items():
          if k in names and isinstance(v, (int, float)) and v <= res["it"]:
            self.reached[k] = int(min(v, self.reached.get(k, v)))

  def on_end(self, reason: str, error: BaseException | None) -> None:
    if self.ended:
      return
    self.ended = True
    it = getattr(self.runner, "current_learning_iteration", None)
    data = {"reason": reason}
    if error is not None:
      name = type(error).__name__
      data["reason"] = {"KeyboardInterrupt": "interrupt", "SystemExit": "exit"}.get(name, "exception")
      data.update(error=repr(error)[:1000],
                  trace="".join(traceback.format_exception(type(error), error, error.__traceback__))[-8000:])
      self.event("exception", it, **{k: data[k] for k in ("error", "trace")})
    self._safe("final", self._final, it, data["reason"])
    self.event("end", it, **{k: v for k, v in data.items() if k != "trace"},
               faults=sorted(self.faults), active_alerts=sorted(n for n, a in self.alerts.items() if a["active"]))
    self._status(ended=data["reason"], t_end=round(time.time(), 1))

  def _final(self, it, reason: str) -> None:
    """The run's last record: the latest state the records had, with the episodes finished since the last record
    (the iterations just before a crash are the ones to look at)."""
    snap = self.last.get("snapshot")
    ep = getattr(self.env, "u_episodes", None)
    if ep is not None and ep.fault is None:
      self.ep_win = merge(self.ep_win, ep.take())
    if snap is None or (self.last.get("recorded") == snap["it"] and self.ep_win is None):
      return
    self._append("record.jsonl", {**snap, "final": reason, "it_end": it,
                                  "ep": episodes_json(self.ep_win, getattr(ep, "meta", None))})
    self.ep_win = None

  # ---- one iteration ----
  def _tags(self, args: dict) -> dict | None:
    """The console's values of this iteration (everything the environment logs: per-term episode rewards,
    terminations, metrics, curriculum), on the snapshots that carry them."""
    it = args.get("it")
    if it is None or not (it % TAGS_EVERY == 0 or it == args.get("start_it") or it == (args.get("total_it") or 0) - 1):
      return None
    eps = getattr(self.runner.logger, "ep_extras", None) or []
    out, gpu = {}, {}
    for key in (eps[0] if eps else {}):
      on_gpu, on_cpu = [], []
      for e in eps:
        x = e.get(key)
        if isinstance(x, torch.Tensor):
          (on_gpu if x.is_cuda else on_cpu).append(x.detach().float().reshape(-1))
        elif x is not None:
          on_cpu.append(torch.tensor([float(x)]))
      if on_gpu:
        gpu[key] = torch.cat(on_gpu).mean()
      elif on_cpu:
        out[key] = float(torch.cat(on_cpu).mean())
    if gpu:
      out.update(zip(gpu, torch.stack(list(gpu.values())).cpu().tolist()))
    return out

  def _add(self, key: str, value) -> None:
    if value is not None:
      self.acc.setdefault(key, []).append(float(value))

  def _mean(self, key: str):
    v = self.acc.get(key)
    return sum(v) / len(v) if v else None

  def _hist(self, key: str, it: int, value) -> None:
    if value is not None:
      self.hist.setdefault(key, deque(maxlen=150)).append((it, float(value)))

  def _ago(self, key: str, it: int, back: int):
    """The value at least `back` iterations ago (the newest such), None without one."""
    old = None
    for i, v in self.hist.get(key, ()):
      if i <= it - back:
        old = v
      else:
        break
    return old

  def _median(self, key: str, min_n: int = 30):
    v = sorted(x for _, x in self.hist.get(key, ()))
    return v[len(v) // 2] if len(v) >= min_n else None

  def on_iteration(self, args: dict, tags: dict | None) -> None:
    it = args.get("it")
    if it is None:
      return
    self.start_it, self.total_it = args.get("start_it"), args.get("total_it")
    logger = self.runner.logger
    collect, learn = float(args.get("collect_time") or 0.0), float(args.get("learn_time") or 0.0)
    size = int(logger.cfg["num_steps_per_env"]) * int(getattr(logger, "num_envs", self.env.num_envs))
    fps = size / max(collect + learn, 1e-9)
    self._add("fps", fps)
    self._add("collect_s", collect)
    self._add("learn_s", learn)
    for k, v in (args.get("loss_dict") or {}).items():
      self._add(f"loss_{k}", v)
    self._add("lr", args.get("learning_rate"))
    if self.grads:
      g = torch.stack(self.grads).cpu()
      self.grads = []
      for j, name in enumerate(("actor", "critic")[:g.shape[1]]):
        self._add(f"grad_{name}", g[:, j].mean())
        self._add(f"grad_{name}_max", g[:, j].max())
        if self.max_grad > 0.0:
          self._add(f"grad_{name}_clipped", (g[:, j] >= 0.999 * self.max_grad).float().mean())
    first, last = it == self.start_it, self.total_it is not None and it == self.total_it - 1
    if not (it % STATUS_EVERY == 0 or first or last):
      return
    s = self._gather(it, args, fps)
    self.last["status"] = {
      "it": it, "t": round(time.time(), 1), "start_it": self.start_it, "total_it": self.total_it, "fps": fps,
      "run_dir": str(logger.log_dir), "pid": os.getpid(), "code": getattr(self.runner, "_commit", None),
      **{k: s["cur"].get(k) for k in ("p", "q", "level", "up", "fail", "fail_eval", "q_gate")},
      "std": s["policy"].get("std"), "reward": s["episode"].get("reward"), "len": s["episode"].get("len"),
      "alerts": sorted(n for n, a in self.alerts.items() if a["active"]), "faults": sorted(self.faults),
    }
    self._safe("transitions", self._transitions, it, s)
    self._safe("alerts", self._alerts, it, s, args)
    self.last["status"]["alerts"] = sorted(n for n, a in self.alerts.items() if a["active"])
    self.last["status"]["expect"] = {e["name"]: [e["by"], self.reached.get(e["name"])] for e in self.expect}
    self._status()
    self._safe("tensorboard", self._tensorboard, it, s)
    snap = {"it": it, "t": round(time.time(), 1),
            "perf": {"fps": self._mean("fps"), "collect_s": self._mean("collect_s"), "learn_s": self._mean("learn_s"),
                     **s["mem"]},
            "loss": {k[5:]: self._mean(k) for k in self.acc if k.startswith("loss_")} | {"lr": self._mean("lr")},
            "grad": {k[5:]: (max(v) if k.endswith("_max") else sum(v) / len(v)) for k, v in self.acc.items()
                     if k.startswith("grad_")},
            **{k: s[k] for k in ("policy", "rollout", "episode", "cur", "cons", "relax", "weights", "gate", "phys")},
            "alerts": self.last["status"]["alerts"],
            "expect": {e["name"]: self.reached.get(e["name"]) for e in self.expect}}
    self.last["snapshot"] = snap
    if it % RECORD_EVERY == 0 or first or last:
      snap["ep"] = episodes_json(self.ep_win, getattr(getattr(self.env, "u_episodes", None), "meta", None))
      if tags:
        snap["tags"] = tags
      self._append("record.jsonl", snap)
      self.last["recorded"] = it
      self.ep_win = None
      self.acc = {}

  def _block(self, name: str, fn, *args) -> dict:
    out = self._safe(name, fn, *args)
    return out if isinstance(out, dict) else {}

  def _gather(self, it: int, args: dict, fps: float) -> dict:
    env = self.env
    ep = getattr(env, "u_episodes", None)
    if ep is not None:
      if ep.fault is not None:
        self._fault("episodes", ep.fault)
      w = self._safe("episodes_take", ep.take)
      if w is not None:
        self.ep_recent.append((w["n"].copy(), w["fail"].copy(), w["cause"].copy(), Counter(w["bodies"]),
                               w["pay"].copy(), w["role"].copy()))
        self.ep_win = merge(self.ep_win, w)
    cur = self._block("cur", self._cur)
    cons = self._block("cons", self._cons, cur)
    s = {"cur": cur, "cons": cons, "policy": self._block("policy", self._policy, args),
         "rollout": self._block("rollout", self._rollout), "episode": self._block("episode", self._episode),
         "relax": dict(getattr(getattr(env, "u_constraints", None), "relax", {}) or {}),
         "weights": self._block("weights", self._weights), "gate": self._block("gate", self._gate, cur, cons),
         "phys": self._block("phys", self._phys), "mem": self._block("mem", self._mem), "fps": fps}
    return s

  def _cur(self) -> dict:
    env = self.env
    t = getattr(env, "u_progress_term", None)
    out = {}
    if t is not None:
      out.update(p=t.p, q=t.q, level=t.level, up=t.up, fail=t.fail, fail_eval=getattr(t, "fail_eval", None),
                 fail_seen=t.fail_mode.get("seen"), fail_blind=t.fail_mode.get("blind"),
                 clock_it=None if t.clock is None else t.clock / STEPS_PER_IT,
                 q_gate=getattr(t, "q_gate", None), q_gap=getattr(t, "q_gap", None),
                 clock_slow=getattr(t, "clock_slow", None),
                 clock_fail=getattr(t, "clock_fail", None))
    fam = getattr(env, "u_family_fail", None)   # the training robots', at the curriculum's rows
    if fam:
      out["family_fail"] = dict(fam)
    stat = getattr(env, "u_eval_stat", None)    # the evaluation robots', on their fixed stairs rows
    if stat:
      out["eval_fail"] = dict(getattr(env, "u_eval_fail", None) or {})
      out["eval_fail_rows"] = {k: [r if n > 0 else None for r, n in zip(v["rate"], v["n"])] for k, v in stat.items()}
      out["eval_fail_n"] = {k: list(v["n"]) for k, v in stat.items()}
    split = getattr(env, "u_eval_split", None)   # the same by payload class and scan mode (curriculum._eval_fail)
    if split:
      # eval_light[family]: the light robots' fail rate (what relaxes the stairs constraints, against "ref"), its
      # episodes, and per scan mode the rate on each step-height band (the gate's survival items read the worst
      # band of "seen" and of "blind"); eval_class[family][class]: every class's rate and episodes
      ref = getattr(getattr(env, "u_constraints", None), "ref", None) or {}
      light, cls = {}, {}
      for k in split:
        v, n = eval_light(env, k)
        light[k] = {"fail": v, "n": n, "ref": ref.get(k),
                    "bands": {m: eval_bands(env, k, m) for m in SPLIT_MODES},
                    "n_mode": {m: eval_light(env, k, m)[1] for m in SPLIT_MODES}}
        cls[k] = {c: dict(zip(("fail", "n"), eval_light(env, k, "all", c))) for c in PAY_CLASSES}
      out["eval_light"], out["eval_class"], out["eval_row_bands"] = light, cls, row_bands(env)
    rows = getattr(env, "u_rows", None)
    if rows:   # the training robots' (the evaluation robots' stairs rows are fixed)
      ev = getattr(env, "u_eval_mask", None)
      tr = None if ev is None or not bool(ev.any()) or bool(ev.all()) else ~ev
      out["rows"] = dict(zip(rows, torch.stack([(v if tr is None else v[tr]).float().mean()
                                                 for v in rows.values()]).cpu().tolist()))
    full = getattr(env, "u_payload_full", None)
    if full is not None:
      out["payload_full"] = full
      out["payload_limit"] = env.event_manager.get_term_cfg("base_payload").params.get("heavy_max")
    try:
      r = env.command_manager.get_term("twist").cfg.ranges
      out["cmd_max"] = [r.lin_vel_x[1], r.lin_vel_y[1], r.ang_vel_z[1]]
    except Exception:
      pass
    return out

  def _cons(self, cur: dict) -> dict:
    cons = getattr(self.env, "u_constraints", None)
    if cons is None:
      return {}
    p, q = cur.get("p") or 0.0, cur.get("q") or 0.0
    out = {}
    for c in self.items:
      name = c["name"]
      d = {"on": bool(cons.on(c, p, q)), "scale": cons.scale.get(name), "value": cons.value.get(name),
           "target": cons.target(c, p, q), "bar": c.get("target")}   # the teacher's target now; the students' bar
      for m in MODES:
        if f"{name}@{m}" in cons.values:
          d[m], d[f"n_{m}"] = cons.values[f"{name}@{m}"], cons.count.get(f"{name}@{m}", 0)
      if name in cons.values:   # an environment without scan modes
        d["all"], d["n_all"] = cons.values[name], cons.count.get(name, 0)
      out[name] = d
    return out

  def _gate(self, cur: dict, cons: dict) -> dict:
    """teacher_gate.py's items as they read now (the gate itself averages the last 500 iterations at a checkpoint):
    which enforced item x scan mode is on the wrong side of the students' bar."""
    if self.kind != "teacher" or not self.items:
      return {}
    miss = []
    for c in self.items:
      if not c.get("enforce", True) or c.get("target") is None:
        continue
      d = cons.get(c["name"], {})
      modes = ["all"] if "all" in d else list(c.get("modes", MODES))
      for m in modes:
        v = d.get(m)
        if v is None:
          miss.append(f"{c['name']}_{m} unmeasured")
        elif not (v >= c["target"] if c["sense"] == "min" else v <= c["target"]):
          miss.append(f"{c['name']}_{m}")
    return {"p_done": (cur.get("p") or 0.0) >= 1.0, "q_done": (cur.get("q") or 0.0) >= 1.0, "miss": miss}

  def _weights(self) -> dict:
    cons = getattr(self.env, "u_constraints", None)
    if cons is None:
      return {}
    rm = self.env.reward_manager
    return {t: rm.get_term_cfg(t).weight for t in cons.base}

  def _policy(self, args: dict) -> dict:
    std = args.get("action_std")
    if not isinstance(std, torch.Tensor):
      return {}
    std = std.detach().float()
    std = std.reshape(-1, std.shape[-1]).mean(dim=0)   # per action component (the last update's mini-batch)
    am = self.env.action_manager
    parts, i = [], 0
    for name, dim in zip(am.active_terms, am.action_term_dim):
      parts.append((name, std[i:i + dim].mean()))
      i += dim
    v = torch.stack([std.mean(), std.min(), std.max()] + [x for _, x in parts]).cpu().tolist()
    return {"std": v[0], "std_min": v[1], "std_max": v[2], "std_by_term": dict(zip((n for n, _ in parts), v[3:]))}

  def _rollout(self) -> dict:
    st = getattr(self.runner.alg, "storage", None)
    if st is None:
      return {}
    ev = getattr(self.env, "u_eval_mask", None)
    k = int(ev.sum()) if ev is not None else 0   # the evaluation robots: the first k (runner.py), not trained on
    keys, vals = [], []

    def put(key, value):
      keys.append(key)
      vals.append(value.float())

    if getattr(st, "rewards", None) is not None:
      r = st.rewards[:, k:]
      put("reward_step", r.mean())
      put("reward_step_min", r.min())
    if getattr(st, "values", None) is not None and getattr(st, "returns", None) is not None:
      v, ret = st.values[:, k:].float(), st.returns[:, k:].float()
      err = ret - v
      put("value", v.mean())
      put("return", ret.mean())
      put("ev", 1.0 - err.var() / (ret.var() + 1e-8))   # the critic's explained variance
      put("adv_std", err.std())
    if getattr(st, "dones", None) is not None:
      put("done_share", st.dones.float().mean())
    a = getattr(st, "actions", None)
    if a is not None:
      put("action_abs", a[:, k:].abs().mean())
      put("action_max", a.abs().max())
      clip = getattr(self.runner.env, "clip_actions", None)
      if clip:
        put("action_clipped", (a[:, k:].abs() >= 0.999 * float(clip)).float().mean())
    return dict(zip(keys, torch.stack(vals).cpu().tolist())) if vals else {}

  def _episode(self) -> dict:
    lg = self.runner.logger
    rew, ln = list(getattr(lg, "rewbuffer", ())), list(getattr(lg, "lenbuffer", ()))
    return {"reward": sum(rew) / len(rew) if rew else None, "len": sum(ln) / len(ln) if ln else None}

  def _phys(self) -> dict:
    if self.phys is None:
      self.phys = physics(self.env)
    return self.phys()

  def _mem(self) -> dict:
    out = {}
    if torch.cuda.is_available():
      out = {"gpu_mb": torch.cuda.memory_allocated() / 2**20, "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20}
    try:
      import resource

      out["rss_peak_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
      pass
    return out

  # ---- the timeline: changes of state ----
  def _flip(self, it: int, ev: str, key: str, state, **data) -> None:
    """A state's change as an event, at most one per FLIP_IT iterations per key (`flips`: the changes since the
    last one written); the first state is written only when it is set."""
    f = self.flips.setdefault((ev, key), {"state": None, "it": -10**9, "n": 0, "cur": None, "first": True})
    if f["cur"] is not None and f["cur"] != state:
      f["n"] += 1
    f["cur"] = state
    if f["first"]:
      f["first"] = False
      if not state:
        f["state"] = state
        return
    if state != f["state"] and it - f["it"] >= FLIP_IT:
      self.event(ev, it, name=key, state=state, flips=f["n"], **data)
      f.update(state=state, it=it, n=0)

  def _mile(self, it: int, key: str, value, marks: tuple, **data) -> None:
    """`value` passing each of `marks` for the first time (those already passed when first seen are not written)."""
    if value is None:
      return
    n = sum(1 for m in marks if value >= m)
    if key not in self.miles:
      self.miles[key] = n
      return
    for m in marks[self.miles[key]:n]:
      self.event("milestone", it, name=key, mark=m, value=value, **data)
    self.miles[key] = max(self.miles[key], n)

  def _transitions(self, it: int, s: dict) -> None:
    cur, cons = s["cur"], s["cons"]
    p, q = cur.get("p"), cur.get("q")
    self._mile(it, "p", p, (0.15, 0.25, 0.3, 0.5, 0.8, 1.0))
    self._mile(it, "q", q, (1e-9, 0.5, 1.0), up=cur.get("up"))
    for k, v in (cur.get("rows") or {}).items():
      self._mile(it, f"row_{k}", v, tuple(float(i) for i in range(1, 10)))
    if cur.get("payload_full"):
      self._mile(it, "payload", (cur.get("payload_limit") or 0.0) / cur["payload_full"], (0.5, 0.999))
    if cur.get("q_gate") is not None:
      self._flip(it, "q_gate", "q", cur["q_gate"], up=cur.get("up"), eval_fail=cur.get("eval_fail"),
                 q=cur.get("q"), gap=cur.get("q_gap"))
    if cur.get("clock_slow") is not None:
      self._flip(it, "clock_slow", "p", bool(cur["clock_slow"]),
                 fail=cur.get("clock_fail") if cur.get("clock_fail") is not None else cur.get("fail"))
    for fam, on in (s.get("relax") or {}).items():
      self._flip(it, "relax", fam, bool(on), family_fail=cur.get("family_fail"), eval_fail=cur.get("eval_fail"),
                 fail=cur.get("fail"), eval_light=_light_brief(cur))
    for c in self.items:
      d = cons.get(c["name"])
      if not d:
        continue
      self._flip(it, "constraint_on", c["name"], d["on"], p=p, q=q)
      if c.get("enforce", True) and d["on"] and d.get("scale") is not None:
        self._flip(it, "cap", c["name"], d["scale"] >= c["max_scale"] - 1e-6, scale=d["scale"], value=d.get("value"),
                   target=d.get("target"))
        if c.get("min_scale", 0.5) < 1.0:
          self._flip(it, "floor", c["name"], d["scale"] <= c.get("min_scale", 0.5) + 1e-6, scale=d["scale"],
                     value=d.get("value"), target=d.get("target"))

  # ---- alerts ----
  def _alert(self, it: int, name: str, cond: bool, msg="", hold: int = 3, **data) -> None:
    """One rule's evaluation: `cond` for `hold` evaluations in a row raises the alert (an event; again every
    REPEAT_IT while it lasts), CLEAR_HOLD evaluations without it clear it. `msg`: text, or a function giving it."""
    a = self.alerts.setdefault(name, {"n": 0, "off": 0, "active": False, "last": -10**9, "since": None})
    if cond:
      a["n"] += 1
      a["off"] = 0
      if a["n"] >= hold and (not a["active"] or it - a["last"] >= REPEAT_IT):
        if not a["active"]:
          a["since"] = it
        text = msg() if callable(msg) else msg   # (long floats shortened: "0.40000006556510925" -> "0.4")
        text = re.sub(r"-?\d+\.\d{4,}(?:e-?\d+)?", lambda m: f"{float(m.group()):.3g}", str(text))
        self.event("alert", it, name=name, msg=text, since=a["since"], again=a["active"], **data)
        a["active"], a["last"] = True, it
    else:
      a["n"] = 0
      if a["active"]:
        a["off"] += 1
        if a["off"] >= CLEAR_HOLD:
          self.event("clear", it, name=name, since=a["since"])
          a["active"] = False

  def _alerts(self, it: int, s: dict, args: dict) -> None:
    """Every group of rules on its own: a fault in one leaves the others running."""
    for part in (self._al_numbers, self._al_task, self._al_optim, self._al_curriculum, self._al_constraints,
                 self._al_physics, self._al_schedule):
      self._safe("alerts_" + part.__name__[4:], part, it, s, args)

  def _al_numbers(self, it: int, s: dict, args: dict) -> None:
    pol, roll, phys, A = s["policy"], s["rollout"], s["phys"], self._alert
    # numbers that are not finite
    nums = list((args.get("loss_dict") or {}).values()) + [pol.get("std"), roll.get("reward_step"), roll.get("value")]
    bad = [x for x in nums if x is not None and not math.isfinite(float(x))]
    A(it, "nonfinite", bool(bad) or (phys.get("nonfinite") or 0) > 0, "non-finite losses / noise / states", hold=1,
      losses=args.get("loss_dict"), states=phys.get("nonfinite"))
    # the policy's noise
    std = pol.get("std")
    self._hist("std", it, std)
    old = self._ago("std", it, 300)
    A(it, "std_rising", std is not None and old is not None and std > 1.25 * old and std > 0.5,
      f"action std {old} -> {std} in 300 it", hold=5, std=std, before=old)
    A(it, "std_high", std is not None and it >= 1000 and std > 1.0, f"action std {std}", hold=5, std=std)
    clipped = roll.get("action_clipped")
    A(it, "action_clip", clipped is not None and clipped > 0.05, f"{clipped} of the action components at the clip",
      hold=5, share=clipped, action_max=roll.get("action_max"))

  def _al_task(self, it: int, s: dict, args: dict) -> None:
    cur, A = s["cur"], self._alert
    # the task: per ground, over the last 300 iterations' episodes
    ep = getattr(self.env, "u_episodes", None)
    meta = getattr(ep, "meta", None)
    if meta is not None and self.ep_recent:
      n = sum(x[0] for x in self.ep_recent)
      fail = sum(x[1] for x in self.ep_recent)
      cause = sum(x[2] for x in self.ep_recent)
      bodies = sum((x[3] for x in self.ep_recent), Counter())
      pay = sum(x[4] for x in self.ep_recent)   # [ground, payload class, (episodes, failed)]
      role = sum(x[5] for x in self.ep_recent)  # [ground, (training, evaluation), (episodes, failed)]
      i_tr, i_ev = ROLE.index("train"), ROLE.index("eval")
      q = cur.get("q") or 0.0
      for gi, g in enumerate(meta["grounds"]):
        rate = fail[gi] / n[gi] if n[gi] > 0 else 0.0
        top = meta["causes"][int(np.argmax(cause[gi]))] if len(meta["causes"]) and cause[gi].sum() > 0 else None
        sets = Counter({m: k for (x, m), k in bodies.items() if x == gi}).most_common(1)
        body = "+".join(b for j, b in enumerate(meta["bodies"]) if sets[0][0] >> j & 1) if sets else None
        if meta["family"].get(g, "other") == "other":
          A(it, f"fail_high:{g}", it >= FAIL_FROM_IT and n[gi] >= 100 and rate > 0.5,
            f"{g}: {rate:.0%} of {int(n[gi])} episodes fail, mostly {top}" + (f" ({body})" if body else ""), hold=10,
            ground=g, rate=rate, n=int(n[gi]), cause=top, bodies=body)
        else:
          # A stairs ground: at the curriculum's rows ~0.6 of the episodes fail whatever the policy climbs
          # (curriculum.py), so the evaluation robots' episodes on their fixed rows are judged, once the quality
          # stage is half through (before that the reference schedule judges the climbing). The cause and the
          # bodies are all robots'.
          ne, fe = role[gi, i_ev]
          re_ = fe / ne if ne > 0 else 0.0
          A(it, f"fail_high:{g}", q >= FAIL_FROM_Q and ne >= 50 and re_ > 0.5,
            f"{g}: {re_:.0%} of the evaluation robots' {int(ne)} episodes on their fixed rows fail, mostly {top}"
            + (f" ({body})" if body else ""), hold=10, ground=g, rate=re_, n=int(ne), cause=top, bodies=body,
            role="eval", all_rate=rate, all_n=int(n[gi]))
        (nl, fl), (nh, fh) = pay[gi, 0], pay[gi, 1] + pay[gi, 2]
        rl, rh = (fl / nl if nl > 0 else 0.0), (fh / nh if nh > 0 else 0.0)
        A(it, f"payload_fail:{g}", it >= FAIL_FROM_IT and nl >= 100 and nh >= 100 and rh - rl > 0.1,
          f"{g}: with a payload {rh:.0%} of {int(nh)} episodes fail, the light robots {rl:.0%} of {int(nl)}", hold=10,
          ground=g, loaded=rh, light=rl, n_loaded=int(nh), n_light=int(nl))
        A(it, f"no_episodes:{g}", it >= 500 and len(self.ep_recent) == self.ep_recent.maxlen and n[gi] == 0,
          f"{g}: no finished episode in 300 it", hold=20, ground=g)
      if "nan" in meta["causes"]:
        k = self.ep_recent[-1][2][:, meta["causes"].index("nan")].sum()
        A(it, "nan_resets", k > 0, f"{int(k)} episodes ended by the nan termination", hold=1, n=int(k))
    ln = s["episode"].get("len")
    self._hist("len", it, ln)
    old = self._ago("len", it, 300)
    A(it, "len_drop", ln is not None and old is not None and it >= 500 and ln < 0.7 * old,
      f"mean episode length {old} -> {ln} steps in 300 it", hold=5, len=ln, before=old)
    level = cur.get("level")
    self._hist("level", it, level)
    top = max((v for i, v in self.hist.get("level", ()) if i >= it - 500), default=None)
    A(it, "level_falling", level is not None and top is not None and level < top - 1.0,
      f"mean terrain level {level}, {top} within the last 500 it", hold=5, level=level, recent_max=top)
    # the exploration noise: off the stairs, where both kinds of robots stand on the curriculum's rows
    if meta is not None and self.ep_recent:
      other = [gi for gi, g in enumerate(meta["grounds"]) if meta["family"].get(g, "other") == "other"]
      (nt, ft), (ne, fe) = role[other, i_tr].sum(0), role[other, i_ev].sum(0)
      f_tr, f_ev = (ft / nt if nt > 0 else 0.0), (fe / ne if ne > 0 else 0.0)
      A(it, "eval_gap", nt >= 200 and ne >= 100 and f_tr - f_ev > 0.2,
        f"off the stairs the training robots fail {f_tr:.0%} of {int(nt)} episodes, the mean-action ones {f_ev:.0%} "
        f"of {int(ne)}", hold=30, fail=f_tr, fail_eval=f_ev, n=int(nt), n_eval=int(ne))

  def _al_optim(self, it: int, s: dict, args: dict) -> None:
    roll, A = s["rollout"], self._alert
    vl = (args.get("loss_dict") or {}).get("value")
    med = self._median("value_loss")
    A(it, "value_loss_spike", vl is not None and med is not None and med > 0 and vl > 10.0 * med,
      f"value loss {vl}, recent median {med}", hold=3, value_loss=vl, median=med)
    self._hist("value_loss", it, vl)
    ev = roll.get("ev")
    A(it, "ev_low", ev is not None and it >= 300 and ev < 0.1, f"explained variance {ev}", hold=10, ev=ev)
    lr = args.get("learning_rate")
    A(it, "lr_floor", lr is not None and lr <= 1.5e-5, f"learning rate {lr}", hold=10, lr=lr)
    med = self._median("fps")
    A(it, "fps_drop", med is not None and s["fps"] < 0.5 * med, lambda: f"{s['fps']:.0f} steps/s, usually {med:.0f}",
      hold=6, fps=s["fps"], median=med)
    self._hist("fps", it, s["fps"])

  def _al_curriculum(self, it: int, s: dict, args: dict) -> None:
    cur, A = s["cur"], self._alert
    p, q, gate = cur.get("p"), cur.get("q"), cur.get("q_gate")
    if self.kind == "teacher" and p is not None and q is not None:
      up, ef = cur.get("up"), (cur.get("eval_fail") or {}).get("up")
      why = f"the climbing row {up} is below {Q_ROW}" if gate == "waiting_row" else str(gate)
      A(it, "q_waiting", p >= 0.5 and q <= 0.0, f"progress {p}, quality 0: {why}", hold=30, p=p, up=up, gate=gate,
        eval_up_fail=ef)
      self._hist("q", it, q)
      old = self._ago("q", it, 300)
      A(it, "q_stalled", 0.0 < q < 1.0 and old is not None and q <= old, f"quality {q} for 300 it: {why}", hold=1, q=q,
        up=up, gate=gate, eval_up_fail=ef)
      slow = cur.get("clock_fail") if cur.get("clock_fail") is not None else cur.get("fail")
      A(it, "clock_slow", bool(cur.get("clock_slow")) and p < 1.0,
        f"the other ground's fail rate {slow} over {MAX_FAIL_RATE}: the progress clock at a quarter pace", hold=50,
        fail=slow, family_fail=cur.get("family_fail"))
    # a family's relaxing matters once one of its constraints is on (before, nothing is held back: the evaluation
    # robots fail the stairs' higher rows until the policy climbs them)
    live = {c.get("family", "other") for c in self.items if (s["cons"].get(c["name"]) or {}).get("on")}
    for fam, on in (s.get("relax") or {}).items():
      A(it, f"relax:{fam}", bool(on) and fam in live, f"the {fam} constraints relaxed for 500 it", hold=50,
        family_fail=cur.get("family_fail"), eval_fail=cur.get("eval_fail"), fail=cur.get("fail"),
        eval_light=_light_brief(cur))

  def _al_constraints(self, it: int, s: dict, args: dict) -> None:
    cons, A = s["cons"], self._alert
    for c in self.items:
      d = cons.get(c["name"])
      if not d or not c.get("enforce", True) or not d["on"]:
        continue
      name = c["name"]
      A(it, f"cap:{name}", d.get("scale") is not None and d["scale"] >= c["max_scale"] - 1e-6,
        f"{name}: multiplier at its limit {c['max_scale']} for 300 it, reading {d.get('value')} vs target "
        f"{d.get('target')}", hold=30, scale=d.get("scale"), value=d.get("value"), target=d.get("target"))
      if self.kind == "teacher":
        for m in (["all"] if "all" in d else c.get("modes", MODES)):
          n = d.get(f"n_{m}", 0)
          self._hist(f"n:{name}@{m}", it, n)
          old = self._ago(f"n:{name}@{m}", it, 300)
          A(it, f"unmeasured:{name}@{m}", old is not None and n <= old, f"{name} ({m}): no new reading in 300 it "
            f"({n} so far)", hold=1, n=n)

  def _al_schedule(self, it: int, s: dict, args: dict) -> None:
    """The reference schedule (expectations): each milestone's first reaching is an event, one not reached by its
    iteration an alert until it is."""
    cur = s["cur"]
    for e in self.expect:
      name = e["name"]
      if name not in self.reached and it >= 300 and e["test"](cur):   # (the first rows and fail rates say nothing)
        self.reached[name] = it
        self.event("milestone", it, name="expect:" + name, mark=e["by"], value=it, what=e["what"])
      self._alert(it, f"late:{name}", name not in self.reached and it >= e["by"],
                  lambda: f"{e['what']}: not reached by iteration {e['by']} ({e['show'](cur)})", hold=1, by=e["by"],
                  what=e["what"], now=e["show"](cur))

  def _al_physics(self, it: int, s: dict, args: dict) -> None:
    for k, v in s["phys"].items():
      if k.startswith("sat_") and v is not None:
        self._alert(it, f"torque_sat:{k[4:]}", v > 0.2, lambda: f"{v:.0%} of the {k[4:]} torques at the limit", hold=10,
                    share=v)

  def _tensorboard(self, it: int, s: dict) -> None:
    w = getattr(self.runner.logger, "writer", None)
    if w is None:
      return
    out = {}
    for k, v in s["rollout"].items():
      if k in ("ev", "adv_std", "action_abs", "action_max", "action_clipped"):
        out[f"Diag/{k}"] = v
    for k in ("grad_actor", "grad_critic", "grad_actor_clipped", "grad_critic_clipped"):
      if self.acc.get(k):
        out[f"Diag/{k}"] = self.acc[k][-1]
    for k, v in s["phys"].items():
      out[f"Diag/phys/{k}"] = v
    for k, v in (s["policy"].get("std_by_term") or {}).items():
      out[f"Diag/std/{k}"] = v
    if s["policy"].get("std_max") is not None:
      out["Diag/std/max"] = s["policy"]["std_max"]
    meta = getattr(getattr(self.env, "u_episodes", None), "meta", None)
    if meta is not None and self.ep_recent:
      n, fail = self.ep_recent[-1][0], self.ep_recent[-1][1]
      for gi, g in enumerate(meta["grounds"]):
        if n[gi] > 0:
          out[f"Diag/fail/{g}"] = fail[gi] / n[gi]
    for k, v in (s["cur"].get("eval_fail_rows") or {}).items():
      for r, x in enumerate(v):
        if x is not None:
          out[f"Diag/eval_fail/{k}_row{r}"] = x
    for k, d in (s["cur"].get("eval_light") or {}).items():
      for m, v in d["bands"].items():
        for b, x in enumerate(v):
          if x is not None:
            out[f"Diag/eval_band/{k}_{m}_b{b}"] = x
    for k, d in (s["cur"].get("eval_class") or {}).items():
      for c, x in d.items():
        if x["fail"] is not None:
          out[f"Diag/eval_class/{k}_{c}"] = x["fail"]
    out["Diag/alerts"] = float(sum(1 for a in self.alerts.values() if a["active"]))
    for k, v in out.items():
      if v is not None and math.isfinite(float(v)):
        w.add_scalar(k, float(v), it)


def attach(runner, kind: str = "teacher") -> Recorder | None:
  """The records for this runner's training (module docstring); None when it has no log directory (plays, probes,
  exports), on U_RECORDS=0, or on a fault -- the training runs without them."""
  try:
    log_dir = getattr(getattr(runner, "logger", None), "log_dir", None)
    if not log_dir or os.environ.get("U_RECORDS", "1") == "0" or getattr(runner, "gpu_global_rank", 0) != 0:
      return None
    return Recorder(runner, kind)
  except Exception:
    print("[records] not attached (the training runs without them):\n" + traceback.format_exc(), flush=True)
    return None
