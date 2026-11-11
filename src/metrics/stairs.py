"""Legged stairs metrics: per-foot contact state machines over a terrain.Grid (go2-depth 2026-11-12; the definitions
of tools/go2_depth/acceptance/accept_stairs.py, the tinynav stairs acceptance since 2026-11-05, unchanged).

Per robot and control step the caller gives, for each foot: whether it touches the static terrain, whether any of
its contacts has a near-horizontal normal (|n_z| < 0.5), whether all of them are near-vertical (|n_z| > 0.9), and
its first contact's point (x, y); plus the robot's heading (unit x, y) and whether it is climbing up.
  kicks      onsets of a near-horizontal contact over stairs (the contact point within 0.5 m of a step edge)
  slips      a stance (>= 3 control steps of contact) whose level drops > 6 cm while it lasts, or, climbing, one that
             starts > 6 cm below the same foot's previous stance
  placement  once per stance on a tread 0.15-0.6 m deep along the heading, flat contacts only, over stairs: its
             position along the tread (0 / 1 = the ends); "mid" 0.2-0.8, "edge" within 3 cm of either end
  ups        (2026-11-12) a stance over stairs that starts more than JUMP above the same foot's previous stance: one
             foot stepping up; ups / feet = the robot's step-ups, the denominator that makes kicks and slips comparable
             between tinynav's routes and mjlab's terrain (per step-up instead of per run)
  downs      the same going down: a stance over stairs starting more than JUMP below the previous one while not climbing
             ((ups + downs) / feet = the steps a robot walked over)
Counts per robot: kicks, slips, placed, mid, edge, ups. The tinynav bars (doc/redesign.md 2.3): kicks <= 2 per run,
mid >= 80% and edge <= 10% of placed, slips <= 0.2 per run.
"""

from __future__ import annotations

import numpy as np

from .terrain import JUMP, Grid

STANCE_STEPS = 3      # control steps of contact that make a stance
SLIP_DROP = 0.06      # m
TREAD_MIN, TREAD_MAX = 0.15, 0.6   # m: a tread's depth along the heading for a placement
MID = (0.2, 0.8)
EDGE = 0.03           # m
ALONG_LIMIT = 0.7     # m


class LeggedStairs:
  def __init__(self, grid: Grid, n: int, feet: int = 4):
    self.grid = grid
    self.on = np.zeros((n, feet), np.int64)
    self.kick = np.zeros((n, feet), bool)
    self.level = np.full((n, feet), np.nan)
    self.placed_flag = np.zeros((n, feet), bool)
    self.kicks = np.zeros(n, np.int64)
    self.slips = np.zeros(n, np.int64)
    self.placed = np.zeros(n, np.int64)
    self.mid = np.zeros(n, np.int64)
    self.edge = np.zeros(n, np.int64)
    self.ups = np.zeros(n, np.int64)
    self.downs = np.zeros(n, np.int64)

  def reset(self, idx) -> None:
    """Forget the per-foot state of robots idx (a robot reset in training); the counts stay."""
    self.on[idx] = 0
    self.kick[idx] = False
    self.level[idx] = np.nan
    self.placed_flag[idx] = False

  def step(self, contact: np.ndarray, horiz: np.ndarray, flat: np.ndarray, cp: np.ndarray, heading: np.ndarray,
           up: np.ndarray, active: np.ndarray | None = None) -> None:
    """contact / horiz / flat (n, feet) bool; cp (n, feet, 2); heading (n, 2) unit; up (n,) bool; active (n,): the
    robots measured this step (default all; a fallen or finished robot can be left out)."""
    g = self.grid
    if active is not None:
      contact = contact & active[:, None]
    over = contact & g.near05_v(cp)
    lv = np.where(contact, g.h_v(cp), np.nan)
    # kicks: the onset of a near-horizontal contact over stairs
    self.kicks += (contact & horiz & ~self.kick & over).sum(axis=1)
    self.kick = contact & horiz
    self.on = np.where(contact, self.on + 1, 0)
    valid = contact & ~np.isnan(lv)
    has_level = ~np.isnan(self.level)
    drop = has_level & (lv < self.level - SLIP_DROP)
    start = valid & (self.on == STANCE_STEPS)
    during = valid & (self.on > STANCE_STEPS) & drop & over
    slip_start = start & drop & up[:, None] & over
    self.ups += (start & has_level & (lv > self.level + JUMP) & over).sum(axis=1)
    self.downs += (start & has_level & (lv < self.level - JUMP) & over & ~up[:, None]).sum(axis=1)
    self.slips += (slip_start | during).sum(axis=1)
    self.level = np.where(start | during, lv, self.level)
    self.placed_flag = np.where(start, False, self.placed_flag)
    # placement: once per stance, flat contacts only, over stairs
    cand = valid & (self.on >= STANCE_STEPS) & ~self.placed_flag & over & flat
    if cand.any():
      idx = np.nonzero(cand)
      p, h = cp[idx], heading[idx[0]]
      d_f = g.along_v(p, h, lv[idx], ALONG_LIMIT)
      d_b = g.along_v(p, -h, lv[idx], ALONG_LIMIT)
      depth = d_f + d_b
      ok = (depth >= TREAD_MIN) & (depth <= TREAD_MAX)
      with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(ok, d_b / np.where(ok, depth, 1.0), 0.0)
      np.add.at(self.placed, idx[0], ok)
      np.add.at(self.mid, idx[0], ok & (frac >= MID[0]) & (frac <= MID[1]))
      np.add.at(self.edge, idx[0], ok & (np.minimum(d_f, d_b) < EDGE))
    self.placed_flag |= cand

  def counts(self) -> dict[str, np.ndarray]:
    return {"kicks": self.kicks, "slips": self.slips, "placed": self.placed, "mid": self.mid, "edge": self.edge,
            "ups": self.ups, "downs": self.downs}


SLIDE_REPEAT_S = 0.5   # s: a wheel bumping the same riser again within this is one slide-back
IDLE_AIR_S = 0.06      # s: a wheel in the air longer than this ...
IDLE_APEX = 0.03       # m: ... with an apex this far over its take-off is a visible (idle) step
IDLE_AHEAD = 0.3       # m: taken off with no height change this far ahead along the heading (flat ground)


class WheeledStairs:
  """Wheeled stairs metrics (accept_stairs.py's since 2026-11-05, unchanged), per robot and wheel:
    ups / slips  climbing: a wheel's flat contact one level above its stand is a step-up ("clean", or "edged" when it
                 touched that riser's nose first); a flat contact back on its stand after touching the riser above is a
                 slide-back (counted as a step-up too; repeats within 0.5 s are one); slide-back % = slips / ups
    idle steps   a wheel airborne > 0.06 s with an apex > 3 cm over its take-off, taken off from flat ground (no
                 height change within 0.3 m ahead) while the base is near stairs; after the run split into "start"
                 (taken off the start level, not the end level: idle_rear) and "end" (taken off the end level: idle_front)
  Scalar per wheel (a few robots in tinynav-sim, a few hundred in mjlab checks)."""

  def __init__(self, grid: Grid, n: int, wheels: int = 4, start_levels=None):
    self.grid, self.n, self.w = grid, n, wheels
    self.st = [[dict(on=0, stand=None, nosed=False, last_slide=None, air=0.0, flat_off=False, off_L=-1, t_off=0.0,
                     z0=0.0, apex=0.0, z_last=0.0) for _ in range(wheels)] for _ in range(n)]
    self.ups = np.zeros(n, np.int64)
    self.slips = np.zeros(n, np.int64)
    self.clean = np.zeros(n, np.int64)
    self.edged = np.zeros(n, np.int64)
    self.idle = [[] for _ in range(n)]       # (take-off level, wheel, time, apex cm, landing level)
    self.reached = [[] for _ in range(n)]    # wheel stand levels
    self.start_L = list(start_levels) if start_levels is not None else [-1] * n

  def step(self, t: float, dt: float, contacts, wheel_pos: np.ndarray, heading: np.ndarray, near_stairs: np.ndarray,
           up: np.ndarray, active: np.ndarray | None = None) -> None:
    """contacts[r][w]: a list of (contact point (x, y, z), normal z) with the static terrain, empty when none;
    wheel_pos (n, wheels, 3); heading (n, 2) unit; near_stairs (n,) the base within 1 m of a step edge; up (n,)."""
    g = self.grid
    for r in range(self.n):
      if active is not None and not active[r]:
        continue
      hx, hy = float(heading[r, 0]), float(heading[r, 1])
      for w in range(self.w):
        s = self.st[r][w]
        c = contacts[r][w]
        x, y, z = (float(v) for v in wheel_pos[r, w])
        if c:
          s["on"] += 1
          s["z_last"] = z
          if up[r]:
            best = None
            for cpos, nz in c:
              L = g.level(float(cpos[2]))
              if L < 0:
                continue
              key = (L, abs(nz) > 0.9)
              if best is None or key > best:
                best = key
            if best is not None:
              L, flat = best
              if s["stand"] is not None and L == s["stand"] + 1 and not flat and not s["nosed"]:
                s["nosed"] = True
              elif flat:
                if s["stand"] is not None and L == s["stand"] + 1:
                  self.ups[r] += 1
                  if s["nosed"]:
                    self.edged[r] += 1
                  else:
                    self.clean[r] += 1
                elif s["stand"] is not None and L == s["stand"] and s["nosed"] and not (
                    s["last_slide"] is not None and s["last_slide"][0] == L and t - s["last_slide"][1] < SLIDE_REPEAT_S):
                  self.ups[r] += 1
                  self.slips[r] += 1
                  s["last_slide"] = (L, t)
                if L != s["stand"] or s["nosed"]:
                  s["stand"], s["nosed"] = L, False
                self.reached[r].append(L)
          if s["air"] > 0.0:   # landing after a flight: an idle step?
            if s["flat_off"] and s["air"] > IDLE_AIR_S + 1e-6 and s["apex"] - s["z0"] > IDLE_APEX:
              self.idle[r].append((s["off_L"], w, round(s["t_off"], 2), round(100 * (s["apex"] - s["z0"]), 1),
                                   g.level(float(c[0][0][2]))))
            s["air"] = 0.0
        else:
          if s["on"] > 0:   # take-off
            lv = g.h(x, y)
            ahead = g.along(x, y, hx, hy, lv, IDLE_AHEAD) if lv is not None else 0.0
            s["flat_off"] = bool(near_stairs[r]) and np.isinf(ahead) and s["on"] >= 3
            s["off_L"] = g.level(lv) if lv is not None else -1
            s["t_off"] = t
            s["z0"] = s["z_last"]
            s["apex"] = z
          s["air"] += dt
          s["apex"] = max(s["apex"], z)
          s["on"] = 0

  def counts(self, up: np.ndarray) -> dict[str, np.ndarray]:
    """The counts, the idle steps split by the flight's start and end levels (up: per robot)."""
    front = np.zeros(self.n, np.int64)
    rear = np.zeros(self.n, np.int64)
    for r in range(self.n):
      reached = self.reached[r]
      end_L = (max(reached) if up[r] else min(reached)) if reached else -1
      for L, _w, _t, _a, _ll in self.idle[r]:
        if L == self.start_L[r] and L != end_L:
          rear[r] += 1
        elif L == end_L:
          front[r] += 1
    return {"ups": self.ups, "slips": self.slips, "clean": self.clean, "edged": self.edged, "idle_front": front,
            "idle_rear": rear}
