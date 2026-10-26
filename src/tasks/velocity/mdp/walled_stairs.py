"""Walled straight flights modelled on the tinynav-sim map3 stairwell.

map3 flights are ~0.9 m wide between walls or rails, with rises of 12-19 cm and
treads of 19-44 cm that vary from step to step. Adapted from the go2-depth
branch's WalledCrossStairsTerrainCfg for a blind policy: a blind robot cannot
see walls, so the flights run out to the tile edge with no end wall to walk
into, and the tile edge stays at ground level like the other sub-terrains.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)

_FLOOR_RGBA = (0.55, 0.55, 0.55, 1.0)
_STEP_RGBA = (0.78, 0.72, 0.60, 1.0)
_WALL_RGBA = (0.70, 0.35, 0.30, 1.0)

_FLOOR_THICKNESS = 0.5


class _BoxBuilder:
  """Adds axis-aligned boxes given by their min/max corners."""

  def __init__(self, spec: mujoco.MjSpec):
    self._body = spec.body("terrain")
    self.geometries: list[TerrainGeometry] = []

  def add(
    self,
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
  ) -> None:
    half = 0.5 * (np.asarray(hi) - np.asarray(lo))
    if np.any(half <= 1e-6):
      return
    geom = self._body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=tuple(half),
      pos=tuple(0.5 * (np.asarray(hi) + np.asarray(lo))),
    )
    self.geometries.append(TerrainGeometry(geom=geom, color=rgba))


@dataclass(kw_only=True)
class WalledCrossStairsTerrainCfg(SubTerrainCfg):
  """Four walled flights running from a central junction to the tile edges.

  The flights run along +-x and +-y through the tile center, so the k * 90 deg
  stair spawn yaw faces one of them. Down (`inverted=False`): the junction is a
  platform and the flights descend to ground level. Up (`inverted=True`): the
  junction is a pit below ground and the flights rise to ground level. Each
  flight ends in a short ground-level run open at the tile edge.
  """

  inverted: bool = False
  step_height_range: tuple[float, float] = (0.05, 0.18)
  """Mean rise, interpolated by difficulty."""
  step_height_noise: float = 0.015
  """Per-step uniform rise perturbation (map3 rises vary by a few cm)."""
  step_width_range: tuple[float, float] = (0.20, 0.30)
  """Mean tread, sampled per sub-terrain (map3: 0.19-0.44 m)."""
  step_width_noise: float = 0.02
  corridor_width_range: tuple[float, float] = (0.85, 1.5)
  """(narrowest, widest). Easy rows get the widest corridors, hard rows the
  narrowest (map3: 0.91-0.96 m between rails)."""
  corridor_width_noise: float = 0.05
  junction_width_range: tuple[float, float] = (1.2, 2.0)
  """(hard, easy) junction side, interpolated by difficulty like the corridors."""
  exit_length: float = 0.5
  """Ground-level run between the last riser and the tile edge."""
  max_steps: int = 12
  wall_height: float = 0.8
  """Wall height above the highest walkable surface."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    b = _BoxBuilder(spec)
    sx, sy = self.size
    cx, cy = 0.5 * sx, 0.5 * sy

    narrow, wide = self.corridor_width_range
    w = wide - difficulty * (wide - narrow)
    w = float(np.clip(w + rng.uniform(-1.0, 1.0) * self.corridor_width_noise, narrow, wide))
    j_hard, j_easy = self.junction_width_range
    j = max(j_easy - difficulty * (j_easy - j_hard), w)
    # Flight length from the junction edge to the tile edge.
    length = min(cx, cy) - 0.5 * j

    # One mean tread for all flights keeps their step counts equal; they share
    # the junction, so they also share the total height.
    tread = rng.uniform(*self.step_width_range)
    rise = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )
    room = length - self.exit_length
    num_steps = int(min(self.max_steps, max(1, room // (tread + self.step_width_noise))))
    flights = []
    for _ in range(4):
      rises = rise + rng.uniform(-self.step_height_noise, self.step_height_noise, num_steps)
      treads = tread + rng.uniform(-self.step_width_noise, self.step_width_noise, num_steps)
      flights.append((np.clip(rises, 0.03, None), treads))
    total = max(float(np.sum(rises)) for rises, _ in flights)

    # Down: junction at +total, ground at 0. Up: junction (pit floor) at -total.
    z_floor = -total if self.inverted else 0.0
    top = (0.0 if self.inverted else total) + self.wall_height
    b.add((0.0, 0.0, z_floor - _FLOOR_THICKNESS), (sx, sy, z_floor), _FLOOR_RGBA)
    if not self.inverted:
      b.add((cx - 0.5 * j, cy - 0.5 * j, 0.0), (cx + 0.5 * j, cy + 0.5 * j, total), _STEP_RGBA)

    # (axis, sign): flight direction in the tile frame.
    for (axis, sign), (rises, treads) in zip(
      ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)), flights, strict=True
    ):
      rises = rises * (total / float(np.sum(rises)))

      def add_local(u0, u1, z0, z1, axis=axis, sign=sign):
        """Box across the flight: u outward from the junction edge."""
        a0, a1 = sorted((sign * (0.5 * j + u0), sign * (0.5 * j + u1)))
        if axis == 0:
          b.add((cx + a0, cy - 0.5 * w, z0), (cx + a1, cy + 0.5 * w, z1), _STEP_RGBA)
        else:
          b.add((cx - 0.5 * w, cy + a0, z0), (cx + 0.5 * w, cy + a1, z1), _STEP_RGBA)

      # Step k spans [u, u + tread_k]; its top is the junction height minus
      # (down) or plus (up) the rises so far. The last step is at ground level.
      u = 0.0
      height = 0.0
      for rise_k, tread_k in zip(rises, treads, strict=True):
        height += rise_k
        z = total - height if not self.inverted else -total + height
        add_local(u, u + tread_k, z_floor, z)
        u += tread_k
      if self.inverted:
        add_local(u, length, z_floor, 0.0)

    # Solid wall blocks filling each quadrant outside the corridors and junction.
    for qx in (-1.0, 1.0):
      for qy in (-1.0, 1.0):
        xs = sorted((cx + qx * 0.5 * j, cx + qx * cx))
        ys = sorted((cy + qy * 0.5 * w, cy + qy * cy))
        b.add((xs[0], ys[0], z_floor), (xs[1], ys[1], top), _WALL_RGBA)
        xs = sorted((cx + qx * 0.5 * w, cx + qx * 0.5 * j))
        ys = sorted((cy + qy * 0.5 * j, cy + qy * cy))
        b.add((xs[0], ys[0], z_floor), (xs[1], ys[1], top), _WALL_RGBA)

    origin = np.array([cx, cy, -total if self.inverted else total])
    return TerrainOutput(origin=origin, geometries=b.geometries)
