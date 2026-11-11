"""A height raster of a MuJoCo model's static, axis-up boxes (go2-depth 2026-11-12; from
tools/go2_depth/acceptance/accept_stairs.py, where the tinynav stairs acceptance has used it since 2026-11-05).

Grid(model, window): the window is (x0, x1, y0, y1, zlo, zhi, res). Each cell holds the highest top within [zlo, zhi]
of the boxes over it (zlo where none). Cells next to a height change > JUMP are step edges; near05 / near1 are the
cells within 0.5 / 1.0 m of one ("on / near stairs"). Scalar queries (h, ij, along, level) are the original ones;
the *_v forms take arrays of points (..., 2) and return arrays, with the same arithmetic (truncating cell indices,
the same step sum in along) so both give identical results.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

JUMP = 0.03   # m: a height change larger than this is a step edge


def static_box_geoms(model: mujoco.MjModel, world_only: bool = False) -> list[int]:
  """Boxes on the world body (world_only: tinynav-sim scenes, as the acceptance always took them) or also on bodies
  without degrees of freedom fixed to it (an mjlab terrain body)."""
  out = []
  for g in range(model.ngeom):
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
      continue
    b = model.geom_bodyid[g]
    if world_only and b != 0:
      continue
    static = True
    while b != 0:
      if model.body_dofnum[b] > 0 or model.body_mocapid[b] >= 0:
        static = False
        break
      b = model.body_parentid[b]
    if static:
      out.append(g)
  return out


def raster(model: mujoco.MjModel, win, data: mujoco.MjData | None = None, world_only: bool = False):
  """H (cells x, y), the step-edge mask and (x0, y0, res). Box poses from data (geom_xpos / xmat) when given, else the
  model's (world-body boxes)."""
  x0, x1, y0, y1, zlo, zhi, res = win
  xs, ys = np.arange(x0, x1, res), np.arange(y0, y1, res)
  H = np.full((len(xs), len(ys)), -np.inf)
  for g in static_box_geoms(model, world_only):
    if data is not None:
      pos, R = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
    else:
      pos = model.geom_pos[g]
      R = np.zeros(9)
      mujoco.mju_quat2Mat(R, model.geom_quat[g])
      R = R.reshape(3, 3)
    half = model.geom_size[g]
    if abs(R[2, 2]) < 0.99:
      continue
    top = pos[2] + half[2] * abs(R[2, 2])
    if not zlo <= top <= zhi:
      continue
    # only the cells the box can cover (its bounding circle), the same test as over the whole grid
    rad = math.hypot(half[0], half[1])
    i0 = max(int(np.searchsorted(xs, pos[0] - rad - res)), 0)
    i1 = min(int(np.searchsorted(xs, pos[0] + rad + res)) + 1, len(xs))
    j0 = max(int(np.searchsorted(ys, pos[1] - rad - res)), 0)
    j1 = min(int(np.searchsorted(ys, pos[1] + rad + res)) + 1, len(ys))
    if i0 >= i1 or j0 >= j1:
      continue
    gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1], indexing="ij")
    dx, dy = gx - pos[0], gy - pos[1]
    lx = R[0, 0] * dx + R[1, 0] * dy
    ly = R[0, 1] * dx + R[1, 1] * dy
    inside = (np.abs(lx) <= half[0]) & (np.abs(ly) <= half[1])
    sub = H[i0:i1, j0:j1]
    H[i0:i1, j0:j1] = np.where(inside & (top > sub), top, sub)
  H[~np.isfinite(H)] = zlo
  jump = np.zeros(H.shape, bool)
  for ax in (0, 1):
    d = np.abs(np.diff(H, axis=ax)) > JUMP
    if ax == 0:
      jump[:-1] |= d
      jump[1:] |= d
    else:
      jump[:, :-1] |= d
      jump[:, 1:] |= d
  return H, jump, (x0, y0, res)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
  out = mask.copy()
  for i in range(-r, r + 1):
    for j in range(-r, r + 1):
      if i * i + j * j <= r * r:
        out |= np.roll(np.roll(mask, i, 0), j, 1)
  return out


class Grid:
  def __init__(self, model: mujoco.MjModel, win, data: mujoco.MjData | None = None, world_only: bool = False):
    self.H, jump, (self.x0, self.y0, self.res) = raster(model, win, data, world_only)
    self.near1 = dilate(jump, int(round(1.0 / self.res)))
    self.near05 = dilate(jump, int(round(0.5 / self.res)))
    lv = np.unique(np.round(self.H[np.isfinite(self.H)], 3))
    self.levels = lv[np.concatenate([[True], np.diff(lv) > JUMP])]   # distinct walkable heights (tread tops)
    # along()'s distances: res, 2 res, ... summed as the original loop sums them
    self._steps = {}

  # ---- scalar (the original queries) ----
  def level(self, z: float) -> int:
    """Index of the level within 3 cm of z (a contact height), or -1."""
    i = int(np.argmin(np.abs(self.levels - z)))
    return i if abs(self.levels[i] - z) < 0.03 else -1

  def ij(self, x: float, y: float):
    i, j = int((x - self.x0) / self.res), int((y - self.y0) / self.res)
    if 0 <= i < self.H.shape[0] and 0 <= j < self.H.shape[1]:
      return i, j
    return None

  def h(self, x: float, y: float):
    c = self.ij(x, y)
    return None if c is None else float(self.H[c])

  def along(self, x: float, y: float, hx: float, hy: float, h0: float, limit: float) -> float:
    """Distance along (hx, hy) from (x, y) to the first cell whose height differs from h0 by > JUMP (inf within
    limit)."""
    s = self.res
    while s <= limit:
      v = self.h(x + hx * s, y + hy * s)
      if v is None or abs(v - h0) > JUMP:
        return s
      s += self.res
    return math.inf

  # ---- vectorised ----
  def ij_v(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell indices (truncated as int() truncates) and whether they are inside the grid."""
    i = np.trunc((xy[..., 0] - self.x0) / self.res).astype(np.int64)
    j = np.trunc((xy[..., 1] - self.y0) / self.res).astype(np.int64)
    ok = (i >= 0) & (i < self.H.shape[0]) & (j >= 0) & (j < self.H.shape[1])
    return np.where(ok, i, 0), np.where(ok, j, 0), ok

  def h_v(self, xy: np.ndarray) -> np.ndarray:
    i, j, ok = self.ij_v(xy)
    return np.where(ok, self.H[i, j], np.nan)

  def near05_v(self, xy: np.ndarray) -> np.ndarray:
    i, j, ok = self.ij_v(xy)
    return ok & self.near05[i, j]

  def near1_v(self, xy: np.ndarray) -> np.ndarray:
    i, j, ok = self.ij_v(xy)
    return ok & self.near1[i, j]

  def _step_list(self, limit: float) -> list[float]:
    if limit not in self._steps:
      out, s = [], self.res
      while s <= limit:
        out.append(s)
        s += self.res
      self._steps[limit] = out
    return self._steps[limit]

  def along_v(self, xy: np.ndarray, d: np.ndarray, h0: np.ndarray, limit: float) -> np.ndarray:
    """along() for arrays: points (..., 2), directions (..., 2), start heights (...)."""
    out = np.full(h0.shape, np.inf)
    open_ = np.ones(h0.shape, bool)
    for s in self._step_list(limit):
      v = self.h_v(xy + d * s)
      hit = open_ & (np.isnan(v) | (np.abs(v - h0) > JUMP))
      out = np.where(hit, s, out)
      open_ &= ~hit
      if not open_.any():
        break
    return out
