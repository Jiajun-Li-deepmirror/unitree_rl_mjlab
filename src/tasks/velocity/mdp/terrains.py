"""Walled staircase sub-terrains modelled on the tinynav-sim map3 stairwell.

map3 flights are ~0.9 m wide with full-height walls on both sides, rises of
14-19 cm that vary from step to step, treads of 22-30 cm, and 180-degree
turns on the landings. A depth camera mostly sees those walls, so the policy
has to train among them.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.sensor import GridPatternCfg
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)

_FLOOR_RGBA = (0.55, 0.55, 0.55, 1.0)
_STEP_RGBA = (0.78, 0.72, 0.60, 1.0)
_WALL_RGBA = (0.70, 0.35, 0.30, 1.0)

_FLOOR_THICKNESS = 0.5
_WALL_THICKNESS = 0.1


@dataclass
class ElevatedGridPatternCfg(GridPatternCfg):
  """Downward grid whose rays start `z_offset` above the sensor frame.

  A height scan cast from base height starts inside any wall next to the
  robot and passes through it, so walls read as floor. Starting above the
  walls makes wall tops visible; `height_scan` still measures from the
  frame, so floor and step values are unchanged.
  """

  z_offset: float = 8.0

  def generate_rays(self, mj_model, device):
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 2] += self.z_offset
    return offsets, directions


@dataclass
class HorizontalFanPatternCfg:
  """Horizontal rays from the frame origin at the given yaw angles (degrees).

  With ray_alignment="yaw" they stay level; used to measure the distance to
  walls beside the robot.
  """

  angles_deg: tuple[float, ...] = (70.0, 80.0, 90.0, 100.0, 110.0, -70.0, -80.0, -90.0, -100.0, -110.0)

  def generate_rays(self, mj_model, device):
    del mj_model
    angles = torch.deg2rad(torch.tensor(self.angles_deg, device=device, dtype=torch.float32))
    directions = torch.stack([torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1)
    offsets = torch.zeros_like(directions)
    return offsets, directions


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
class _WalledStairsBaseCfg(SubTerrainCfg):
  step_height_range: tuple[float, float] = (0.08, 0.20)
  """Mean rise, interpolated by difficulty."""
  step_height_noise: float = 0.025
  """Per-step uniform rise perturbation (map3 rises vary by a few cm)."""
  step_width_range: tuple[float, float] = (0.17, 0.32)
  """Mean tread, sampled per sub-terrain. map3 has a few short treads (L0 step 7
  is 0.188 m between 0.25 m ones), so ~15% of treads here are below 0.19 m."""
  step_width_noise: float = 0.04
  min_tread: float = 0.16
  corridor_width_range: tuple[float, float] = (0.85, 1.5)
  """(narrowest, widest). Easy rows get the widest corridors and hard rows the
  narrowest (map3 is ~0.9 m), so the policy first learns to climb with the
  walls far away."""
  corridor_width_noise: float = 0.05

  def _corridor_width(self, difficulty: float, rng: np.random.Generator) -> float:
    narrow, wide = self.corridor_width_range
    w = wide - difficulty * (wide - narrow) + rng.uniform(-1.0, 1.0) * self.corridor_width_noise
    return float(np.clip(w, narrow, wide))
  wall_height: float = 1.5
  """Wall height above the highest walkable surface."""

  def _sample_flight(
    self,
    difficulty: float,
    rng: np.random.Generator,
    length: float,
    max_steps: int,
    tread: float,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Rises and treads of one flight that fits in `length` meters."""
    rise = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )
    num_steps = int(min(max_steps, max(1, length // (tread + self.step_width_noise))))
    rises = rise + rng.uniform(-self.step_height_noise, self.step_height_noise, num_steps)
    treads = tread + rng.uniform(-self.step_width_noise, self.step_width_noise, num_steps)
    return np.clip(rises, 0.03, None), np.clip(treads, self.min_tread, None)


@dataclass(kw_only=True)
class WalledCrossStairsTerrainCfg(_WalledStairsBaseCfg):
  """Four walled flights radiating from a central junction.

  The flights run along +-x and +-y, so the existing k * 90 deg stair spawn
  yaw faces one of them. Each flight ends in a landing closed by an end wall,
  like a map3 flight ending at a landing. With `inverted=False` the junction
  is the top (spawn and walk down); with `inverted=True` it is a pit (walk up).
  """

  inverted: bool = False
  junction_width_range: tuple[float, float] = (1.2, 2.0)
  """(hard, easy) junction side, interpolated by difficulty like the corridors."""
  landing_length: float = 0.6
  max_steps: int = 12

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    b = _BoxBuilder(spec)
    sx, sy = self.size
    cx, cy = 0.5 * sx, 0.5 * sy
    w = self._corridor_width(difficulty, rng)
    j_hard, j_easy = self.junction_width_range
    j = max(j_easy - difficulty * (j_easy - j_hard), w)
    # Corridor length measured outward from the junction edge to the end wall.
    corridor = min(cx, cy) - 0.5 * j - _WALL_THICKNESS

    # One mean tread for all flights keeps their step counts equal; they share
    # the junction, so they also share the total height.
    tread = rng.uniform(*self.step_width_range)
    flights = [
      self._sample_flight(
        difficulty, rng, corridor - self.landing_length, self.max_steps, tread
      )
      for _ in range(4)
    ]
    total = max(float(np.sum(rises)) for rises, _ in flights)
    top = total + self.wall_height

    b.add((0.0, 0.0, -_FLOOR_THICKNESS), (sx, sy, 0.0), _FLOOR_RGBA)
    if not self.inverted:
      b.add((cx - 0.5 * j, cy - 0.5 * j, 0.0), (cx + 0.5 * j, cy + 0.5 * j, total), _STEP_RGBA)

    # (axis, sign): flight direction in the tile frame.
    for (axis, sign), (rises, treads) in zip(
      ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)), flights, strict=True
    ):
      rises = rises * (total / float(np.sum(rises)))

      def add_local(u0, u1, v0, v1, z0, z1, rgba, axis=axis, sign=sign):
        """Box in corridor coordinates: u outward from the junction edge, v lateral."""
        a0, a1 = sorted((sign * (0.5 * j + u0), sign * (0.5 * j + u1)))
        if axis == 0:
          b.add((cx + a0, cy + v0, z0), (cx + a1, cy + v1, z1), rgba)
        else:
          b.add((cx + v0, cy + a0, z0), (cx + v1, cy + a1, z1), rgba)

      u = 0.0
      height = 0.0
      for rise, step in zip(rises, treads, strict=True):
        height += rise
        z = total - height if not self.inverted else height
        add_local(u, u + step, -0.5 * w, 0.5 * w, 0.0, z, _STEP_RGBA)
        u += step
      if self.inverted:
        add_local(u, corridor, -0.5 * w, 0.5 * w, 0.0, total, _STEP_RGBA)
      add_local(corridor, corridor + _WALL_THICKNESS, -0.5 * w, 0.5 * w, 0.0, top, _WALL_RGBA)

    # Solid wall blocks filling each quadrant outside the corridors and junction.
    for qx in (-1.0, 1.0):
      for qy in (-1.0, 1.0):
        xs = sorted((cx + qx * 0.5 * j, cx + qx * cx))
        ys = sorted((cy + qy * 0.5 * w, cy + qy * cy))
        b.add((xs[0], ys[0], 0.0), (xs[1], ys[1], top), _WALL_RGBA)
        xs = sorted((cx + qx * 0.5 * w, cx + qx * 0.5 * j))
        ys = sorted((cy + qy * 0.5 * j, cy + qy * cy))
        b.add((xs[0], ys[0], 0.0), (xs[1], ys[1], top), _WALL_RGBA)

    origin = np.array([cx, cy, 0.0 if self.inverted else total])
    return TerrainOutput(origin=origin, geometries=b.geometries)


@dataclass(kw_only=True)
class StairwellUTurnTerrainCfg(_WalledStairsBaseCfg):
  """Two walled flights side by side joined by a landing (a map3-style U-turn).

  Lane A (y < center) rises along +x to a full-width landing; lane B rises
  back along -x from the landing to a top landing. The spawn faces +x: at the
  bottom of lane A (`descend=False`) or on the top landing of lane B, walking
  down to the turn landing (`descend=True`). Spawn yaw must be 0 (see
  reset_root_state_terrain_aware's `fixed_yaw_terrain_names`).
  """

  descend: bool = False
  well_width_range: tuple[float, float] = (0.3, 0.6)
  """Open stairwell well between the lanes (map3: 0.61 m), down to the floor."""
  rail_thickness: float = 0.05
  """Thin guard rails on both sides of the well (map3: 0.05 m)."""
  turn_landing_length: float = 1.2
  end_landing_length: float = 1.4
  margin: float = 0.3
  max_steps: int = 10

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    b = _BoxBuilder(spec)
    sx, sy = self.size
    cy = 0.5 * sy
    w = self._corridor_width(difficulty, rng)
    well = rng.uniform(*self.well_width_range)
    r = self.rail_thickness
    a_hi = cy - 0.5 * well - r
    a_lo = a_hi - w
    b_lo = cy + 0.5 * well + r
    b_hi = b_lo + w

    x_wall = sx - self.margin  # Inner face of the far end wall.
    x_turn = x_wall - self.turn_landing_length
    flight_room = x_turn - self.margin - _WALL_THICKNESS - self.end_landing_length
    tread = rng.uniform(*self.step_width_range)
    rises_a, treads_a = self._sample_flight(
      difficulty, rng, flight_room, self.max_steps, tread
    )
    rises_b = self._sample_flight(difficulty, rng, flight_room, self.max_steps, tread)[0]
    rises_b = np.resize(rises_b, len(rises_a))
    h_turn = float(np.sum(rises_a))
    h_top = h_turn + float(np.sum(rises_b))
    x_start = x_turn - float(np.sum(treads_a))
    x_back = x_start - self.end_landing_length
    top = h_top + self.wall_height

    b.add((0.0, 0.0, -_FLOOR_THICKNESS), (sx, sy, 0.0), _FLOOR_RGBA)

    # Lane A rises along +x; lane B mirrors the treads and rises along -x.
    x = x_start
    za = 0.0
    zb = h_turn
    for rise, step in zip(rises_a, treads_a, strict=True):
      za += rise
      b.add((x, a_lo, 0.0), (x + step, a_hi, za), _STEP_RGBA)
      x += step
    x = x_turn
    for rise, step in zip(rises_b, treads_a[::-1], strict=True):
      zb += rise
      b.add((x - step, b_lo, 0.0), (x, b_hi, zb), _STEP_RGBA)
      x -= step
    # Turn landing (full width) and lane B's top landing.
    b.add((x_turn, a_lo, 0.0), (x_wall, b_hi, h_turn), _STEP_RGBA)
    b.add((x_back, b_lo, 0.0), (x_start, b_hi, h_top), _STEP_RGBA)

    # Thin rails along both lanes with the open well between them (the turn
    # landing bridges it), solid outer walls, far end, and the lane back ends.
    b.add((self.margin, a_hi, 0.0), (x_turn, a_hi + r, top), _WALL_RGBA)
    b.add((x_back, b_lo - r, 0.0), (x_turn, b_lo, top), _WALL_RGBA)
    b.add((0.0, 0.0, 0.0), (sx, a_lo, top), _WALL_RGBA)
    b.add((0.0, b_hi, 0.0), (sx, sy, top), _WALL_RGBA)
    b.add((x_wall, a_lo, 0.0), (sx, b_hi, top), _WALL_RGBA)
    b.add((0.0, a_lo, 0.0), (self.margin, a_hi, top), _WALL_RGBA)
    b.add((0.0, b_lo, 0.0), (x_back, b_hi, top), _WALL_RGBA)

    if self.descend:
      origin = np.array([0.5 * (x_back + x_start), 0.5 * (b_lo + b_hi), h_top])
    else:
      origin = np.array([x_start - 0.7, 0.5 * (a_lo + a_hi), 0.0])
    return TerrainOutput(origin=origin, geometries=b.geometries)
