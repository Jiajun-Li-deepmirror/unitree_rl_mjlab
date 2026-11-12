"""The mjlab side of src/metrics (go2-depth 2026-11-12, doc/redesign.md step 1): the same stairs counters as tinynav's
acceptance, fed from an mjlab environment's batched state.

  add_sensor(cfg)        a feet / wheels contact sensor with each contact's point and normal (up to SLOTS per foot,
                         nearest first), beside the env's feet_ground_contact (same feet, same terrain)
  terrain_grid(env)      terrain.Grid over the whole terrain (its static boxes, posed)
  StairsMeter(env, ids)  per control step: the legged (LeggedStairs) or wheeled (WheeledStairs) counters over env ids
                         (a fixed sample), only while a robot is on a stairs column; counts decayed with a time constant
                         of ema_steps; rate(name) gives the acceptance quantities, rate(name, group) the same over a
                         group of the robots (meter.groups: the unified teacher's seen / blind scan modes)

Differences from tinynav-sim, by necessity: "climbing" (up) is read from the terrain (the raster 0.5 m ahead along the
heading is higher than under the base) instead of the scenario; a foot's "first contact" is the nearest one (the
sensor's mindist order) instead of MuJoCo's contact-list order.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch

from .stairs import LeggedStairs, WheeledStairs
from .terrain import JUMP, Grid, static_box_geoms

SENSOR = "u_feet_metric"
SLOTS = 3
FEET_SENSOR = "feet_ground_contact"


def add_sensor(cfg) -> None:
  """Add the metrics' contact sensor to an env cfg that has feet_ground_contact (its feet and terrain)."""
  from mjlab.sensor import ContactSensorCfg

  sensors = tuple(cfg.scene.sensors or ())
  if any(s.name == SENSOR for s in sensors):
    return
  feet = next(s for s in sensors if s.name == FEET_SENSOR)
  cfg.scene.sensors = sensors + (ContactSensorCfg(name=SENSOR, primary=feet.primary, secondary=feet.secondary,
                                                  fields=("found", "pos", "normal"), reduce="mindist",
                                                  num_slots=SLOTS),)


def terrain_grid(env, res: float = 0.02) -> Grid:
  """A Grid over all of the env's static terrain boxes, at their posed positions."""
  model = env.sim.mj_model
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  geoms = static_box_geoms(model)
  if not geoms:
    raise RuntimeError("terrain_grid: no static boxes in the model")
  lo = np.full(3, np.inf)
  hi = np.full(3, -np.inf)
  for g in geoms:
    r = float(np.linalg.norm(model.geom_size[g]))
    lo = np.minimum(lo, data.geom_xpos[g] - r)
    hi = np.maximum(hi, data.geom_xpos[g] + r)
  win = (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]) - 1.0, float(hi[2]) + 1.0, res)
  return Grid(model, win, data=data)


def _wheel_geom_ids(env) -> np.ndarray:
  """The global ids of the feet sensor's primary geoms, in the sensor's order: the sensor resolves its patterns with
  Entity.find_geoms, which gives the entity's geom order (FL FR RL RR). Until 2026-11-12 this went by the patterns'
  own order (FR FL RR RL), so each wheel's contacts were read with the opposite side's wheel centre; only the idle
  steps (take-off place and apex height: logged, no bar) use the centre."""
  sensor = env.scene[SENSOR]
  prim = sensor.cfg.primary
  ent = env.scene[prim.entity]
  pats = (prim.pattern,) if isinstance(prim.pattern, str) else tuple(prim.pattern)
  local, _ = ent.find_geoms(pats)
  gids = ent.indexing.geom_ids
  return gids[torch.as_tensor(local, device=gids.device, dtype=torch.long)].cpu().numpy()


class StairsMeter:
  """The stairs counters over a fixed sample of envs, every control step, while each robot is on a stairs column.

  rate(name) for the acceptance quantities (None before min_steps walked):
    legged   kicks / slips per step walked ((ups + downs) / feet), mid / edge share of placements
    wheeled  slide_back (slips / step-ups), idle_start / idle_end per step-up
  Counts decay with a time constant of ema_steps control steps (the curriculum moves on; old behaviour fades)."""

  def __init__(self, env, ids: torch.Tensor, wheeled: bool, ema_steps: float = 20000.0, min_steps: float = 200.0,
               straight: bool = False):
    self.ids = ids.to(env.device)
    self.n = len(ids)
    self.wheeled = wheeled
    # straight: count only robots walking the flight as the acceptance crosses it -- a forward / backward command
    # (|vx| >= 0.25 m/s), little sideways or turning (|vy| <= 0.15, |wz| <= 0.3) -- so the training's reading means
    # what the acceptance's does (2026-11-12: over every stairs robot, stops and turns on the flights included, the
    # Go2 teacher read 3.4 kicks / step against the acceptance's 0.245 bar)
    self.straight = straight
    self.decay = math.exp(-1.0 / ema_steps)
    self.min_steps = min_steps
    self.grid = terrain_grid(env)
    s = env.scene[SENSOR]
    self.feet = s.data.found.shape[1] // SLOTS
    if wheeled:
      self.counter = WheeledStairs(self.grid, self.n, self.feet)
      self.wheel_geoms = torch.as_tensor(_wheel_geom_ids(env), device=env.device)
    else:
      self.counter = LeggedStairs(self.grid, self.n, self.feet)
    self.last_len = None
    self.prev: dict[str, np.ndarray] = {}   # each count per robot at the last step
    self.sums: dict[str, float] = {}
    # groups(env, ids) -> {group: (n,) bool array}: the counts are kept per group too, each robot's increments going
    # to the groups it is in at that step (2026-11-12: the unified teacher's scan modes, tasks/unified/constraints.py)
    self.groups = None
    self.group_sums: dict[str, dict[str, float]] = {}
    # True when only the env's evaluation robots count (env.u_eval_mask; the teacher's runner sets both): a
    # checkpoint's counts taken over every robot, exploration noise included, are not carried on then
    self.eval = False

  def _eval_only(self, env):
    """(n,) the env's evaluation robots among this meter's, or None when every one of them counts. Only a meter the
    teacher's runner marked (self.eval: the env's own training meters, tasks/unified/runner.py) is held to them. A
    meter an evaluation script builds for itself counts every robot it was given (until 2026-11-12 any meter on an
    env with u_eval_mask was held to it: eval_stairs.py loads the teacher through its runner, which sets the mask, so
    the second tier's teacher readings came from the first 72 of 576 robots -- the first 4 tiles)."""
    ev = getattr(env, "u_eval_mask", None) if self.eval else None
    return None if ev is None else ev[self.ids]

  def _state(self, env):
    ids = self.ids
    d = env.scene[SENSOR].data
    F, K = self.feet, SLOTS
    found = d.found[ids].view(self.n, F, K)[:, :, 0].cpu().numpy()
    pos = d.pos[ids].view(self.n, F, K, 3).cpu().numpy()
    nz = d.normal[ids].view(self.n, F, K, 3)[..., 2].abs().cpu().numpy()
    valid = np.arange(K)[None, None, :] < found[:, :, None]
    robot = env.scene["robot"]
    base = robot.data.root_link_pos_w[ids, :2].cpu().numpy()
    q = robot.data.root_link_quat_w[ids].cpu().numpy()   # w, x, y, z
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    h_here = self.grid.h_v(base)
    h_ahead = self.grid.h_v(base + 0.5 * heading)
    up = np.nan_to_num(h_ahead, nan=-np.inf) > np.nan_to_num(h_here, nan=np.inf) + JUMP
    on_stairs = env.command_manager.get_term("twist").is_stair_env[ids]
    ev = self._eval_only(env)
    if ev is not None:
      on_stairs = on_stairs & ev
    on_stairs = on_stairs.cpu().numpy()
    if self.straight:
      c = env.command_manager.get_command("twist")[ids]
      on_stairs = on_stairs & ((c[:, 0].abs() >= 0.25) & (c[:, 1].abs() <= 0.15) & (c[:, 2].abs() <= 0.3)).cpu().numpy()
    return found, pos, nz, valid, base, heading, up, on_stairs

  def step(self, env) -> None:
    ep = env.episode_length_buf[self.ids].cpu().numpy()
    if self.last_len is not None:
      fresh = np.nonzero(ep < self.last_len)[0]
      if len(fresh):
        self.counter.reset(fresh)
    self.last_len = ep
    found, pos, nz, valid, base, heading, up, active = self._state(env)
    contact = found > 0
    if self.wheeled:
      wp = env.sim.data.geom_xpos[self.ids][:, self.wheel_geoms].cpu().numpy()
      contacts = [[[(pos[r, w, k], float(nz[r, w, k])) for k in range(SLOTS) if valid[r, w, k]] for w in range(self.feet)]
                  for r in range(self.n)]
      t = float(env.common_step_counter) * env.step_dt
      self.counter.step(t, env.step_dt, contacts, wp, heading, self.grid.near1_v(base), up, active)
    else:
      horiz = (valid & (nz < 0.5)).any(axis=2)
      flat = contact & (~valid | (nz > 0.9)).all(axis=2)
      self.counter.step(contact, horiz, flat, pos[:, :, 0, :2], heading, up, active)
    self._accumulate(env, up)

  def _counts(self, up) -> dict[str, np.ndarray]:
    """Each count per robot, (n,)."""
    if self.wheeled:
      c = self.counter.counts(up)
      return {"ups": c["ups"], "slips": c["slips"], "idle_start": c["idle_rear"], "idle_end": c["idle_front"]}
    return self.counter.counts()

  def _accumulate(self, env, up) -> None:
    """This step's new events into the decayed sums. Per robot since 2026-11-12 (the groups need it): a count that
    drops -- the wheeled idle steps restart with a robot's episode and move between "start" and "end" as it climbs --
    adds nothing for that robot; taken on the total before, such a drop also hid the other robots' new events of that
    step. The legged counts and the wheeled step-ups and slide-backs only rise: they read as before."""
    groups = self.groups(env, self.ids) if self.groups is not None else {}
    for g in groups:
      self.group_sums.setdefault(g, {})
    for k, v in self._counts(up).items():
      v = np.asarray(v, dtype=np.float64).copy()   # the counters go on counting in their own arrays
      inc = np.maximum(v - self.prev[k], 0.0) if k in self.prev else v
      self.prev[k] = v
      self.sums[k] = self.sums.get(k, 0.0) * self.decay + float(inc.sum())
      for g, m in groups.items():
        gs = self.group_sums[g]
        gs[k] = gs.get(k, 0.0) * self.decay + float(inc[m].sum())

  def rate(self, name: str, group: str | None = None) -> float | None:
    """The acceptance quantity `name` over all the robots measured, or over those of `group` (None before min_steps
    walked there; None for a group never seen)."""
    if group is None:
      s = self.sums
    elif group in self.group_sums:
      s = self.group_sums[group]
    else:
      return None
    if self.wheeled:
      ups = s.get("ups", 0.0)
      if ups < self.min_steps:
        return None
      return {"slide_back": s.get("slips", 0.0) / ups, "idle_start": s.get("idle_start", 0.0) / ups,
              "idle_end": s.get("idle_end", 0.0) / ups}.get(name)
    steps = (s.get("ups", 0.0) + s.get("downs", 0.0)) / self.feet
    placed = s.get("placed", 0.0)
    if steps < self.min_steps:
      return None
    return {"kicks": s.get("kicks", 0.0) / steps, "slips": s.get("slips", 0.0) / steps,
            "mid": s.get("mid", 0.0) / placed if placed else None,
            "edge": s.get("edge", 0.0) / placed if placed else None}.get(name)

  def state(self) -> dict:
    return {"sums": dict(self.sums), "groups": {g: dict(v) for g, v in self.group_sums.items()},
            "straight": self.straight, "eval": self.eval}

  def load(self, state: dict) -> None:
    """The counts of a checkpoint taken with the same counting (straight or not, the evaluation robots or every
    robot); others start afresh."""
    if bool(state.get("straight", False)) == self.straight and bool(state.get("eval", False)) == self.eval:
      self.sums = dict(state.get("sums", {}))
      self.group_sums = {g: dict(v) for g, v in (state.get("groups") or {}).items()}
