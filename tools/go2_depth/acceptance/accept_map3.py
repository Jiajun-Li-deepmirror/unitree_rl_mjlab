"""accept_map3.py ROBOT ONNX [RUNS] (go2-depth 2026-11-06): the user's stairs requirements on map3 only (the user,
11-06: "楼梯的验收直接用map3地图，不需要金字塔楼梯地形测试了"), every map3 flight up and down (eval_map3.py --set
flights: 18 scenarios, spawned on the landings, 0.5 m/s up / 0.4 m/s down, 1 s stand + walk + 1 s stand). Built from
accept_stairs.py (same metrics and bars) plus the stairs posture (the user, 11-06: legs as vertical as possible, each
step touched once, mainly wheeled):
  touches  per foot / wheel and inner tread of the flight (not its start / end landing): separate touches of that step
           (its riser, nosing or tread; a new one after 0.08 s off it). 1 = the foot lands on each step once.
  riser    touches of a riser face (|n_z| < 0.5) per climb (wheeled too; legged = accept_stairs' kicks without the window)
  leg      thigh / calf / hip / base touches of the stairs per climb
  angle    hip (thigh joint) -> foot / wheel centre from the vertical in the heading plane while in stance on the
           stairs, front / rear legs (+ = foot ahead of the hip); |lat| the sideways angle; pitch = base nose-up on stairs
Posture items are reported, not judged (no calibrated bars yet). ONLY=<names> picks scenarios; LEGGED_BARS / WHEELED_BARS
override bars ("k=v,..."). Run from a tinynav-sim worktree's mujoco/ with ROBOT_ASSETS and MUJOCO_GL=egl.
"""
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, ".")
from sim.keyboard import ConstantCmd  # noqa: E402
from sim.plant import DOWN_SPAWN, DOWN_YAW_DEG, SCENES, build_model, reset_to_spawn  # noqa: E402
from sim.policy import PieOnnxPolicy  # noqa: E402
from sim.robots import get_robot  # noqa: E402
from sim.runtime import PieRuntime  # noqa: E402

DT, JUMP = 0.02, 0.03
robot, onnx = sys.argv[1], Path(sys.argv[2])
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
WHEELED = get_robot(robot).wheeled
ONLY = os.environ.get("ONLY")  # comma list of scenario names

# eval_map3.py --set flights --list (mjlab map3 = tinynav-sim map3 coordinates; z = the landing top, the spawn hint is
# 0.4 m above it): name, (x, y, z), yaw deg, walk s, vx
FLIGHTS = [
  ("F1_entry_X_up", (-8.48, 5.67, 0.000), -180.0, 2.8, 0.5), ("F1_entry_X_down", (-10.17, 5.67, 0.431), 0.0, 3.4, 0.4),
  ("L0_up", (-10.34, 5.75, 0.431), -90.0, 8.8, 0.5), ("L0_down", (-10.34, 1.08, 2.659), 90.0, 11.0, 0.4),
  ("L1_up", (-10.34, 5.72, 4.692), -90.0, 8.7, 0.5), ("L1_down", (-10.34, 1.08, 6.761), 90.0, 10.9, 0.4),
  ("L2_up", (-10.34, 5.73, 8.643), -90.0, 7.8, 0.5), ("L2_down", (-10.34, 1.50, 10.423), 90.0, 9.8, 0.4),
  ("L3_up", (-10.34, 5.70, 11.802), -90.0, 8.7, 0.5), ("L3_down", (-10.34, 1.07, 13.829), 90.0, 10.9, 0.4),
  ("R1_up", (-8.79, 1.07, 2.659), 90.0, 8.8, 0.5), ("R1_down", (-8.79, 5.78, 4.692), -90.0, 11.1, 0.4),
  ("R2_up", (-8.79, 1.60, 6.761), 90.0, 7.6, 0.5), ("R2_down", (-8.79, 5.73, 8.643), -90.0, 9.5, 0.4),
  ("R2b_up", (-8.79, 2.15, 10.423), 90.0, 6.6, 0.5), ("R2b_down", (-8.79, 5.80, 11.802), -90.0, 8.3, 0.4),
  ("R3_up", (-8.79, 1.11, 13.829), 90.0, 8.8, 0.5), ("R3_down", (-8.79, 5.78, 16.169), -90.0, 11.0, 0.4),
]


def _scen(name, p, yaw, walk, vx):
  up = name.endswith("_up")
  h = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)))
  l = (-h[1], h[0])
  pts = [(p[0] + h[0] * d + l[0] * o, p[1] + h[1] * d + l[1] * o) for d in (-1.0, walk * vx + 1.5) for o in (-0.8, 0.8)]
  xs, ys = [q[0] for q in pts], [q[1] for q in pts]
  zw = (p[2] - 0.6, p[2] + 2.6) if up else (p[2] - 2.6, p[2] + 0.4)
  win = (min(xs), max(xs), min(ys), max(ys), zw[0], zw[1], 0.01)
  return (name, "map3", (p[0], p[1], p[2] + 0.4), yaw % 360.0, [(1.0, 0.0), (walk, vx), (1.0, 0.0)], win, up)


SCEN = [_scen(*f) for f in FLIGHTS]
if ONLY:
  SCEN = [s for s in SCEN if s[0] in ONLY.split(",")]
_models = {}


PAYLOAD = os.environ.get("PAYLOAD")   # "kg,x,y,z": a point mass on the base (training's mdp.base_payload_point)


def add_payload(model, spec):
  m, px, py, pz = (float(v) for v in spec.split(","))
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  M0, c0, I0 = float(model.body_mass[bid]), model.body_ipos[bid].copy(), model.body_inertia[bid].copy()
  p = np.array([px, py, pz])
  M = M0 + m
  c = (M0 * c0 + m * p) / M
  d0, dp = c0 - c, p - c
  par = lambda d: np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
  model.body_mass[bid] = M
  model.body_ipos[bid] = c
  model.body_inertia[bid] = np.maximum(I0 + M0 * par(d0) + m * par(dp), 1e-4)
  mujoco.mj_setConst(model, mujoco.MjData(model))


def model_for(scene):
  if scene not in _models:
    _models[scene] = build_model(scene, robot)
    if PAYLOAD:
      add_payload(_models[scene][0], PAYLOAD)
  return _models[scene]


def raster(model, win):
  x0, x1, y0, y1, zlo, zhi, res = win
  xs, ys = np.arange(x0, x1, res), np.arange(y0, y1, res)
  gx, gy = np.meshgrid(xs, ys, indexing="ij")
  H = np.full(gx.shape, -np.inf)
  for g in range(model.ngeom):
    if model.geom_bodyid[g] != 0 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
      continue
    pos, half = model.geom_pos[g], model.geom_size[g]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.geom_quat[g])
    R = R.reshape(3, 3)
    if abs(R[2, 2]) < 0.99:
      continue
    top = pos[2] + half[2] * abs(R[2, 2])
    if not zlo <= top <= zhi:
      continue
    dx, dy = gx - pos[0], gy - pos[1]
    lx = R[0, 0] * dx + R[1, 0] * dy
    ly = R[0, 1] * dx + R[1, 1] * dy
    inside = (np.abs(lx) <= half[0]) & (np.abs(ly) <= half[1])
    H = np.where(inside & (top > H), top, H)
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


def dilate(mask, r):
  out = mask.copy()
  for i in range(-r, r + 1):
    for j in range(-r, r + 1):
      if i * i + j * j <= r * r:
        out |= np.roll(np.roll(mask, i, 0), j, 1)
  return out


class Grid:
  def __init__(self, model, win):
    self.H, jump, (self.x0, self.y0, self.res) = raster(model, win)
    self.near1 = dilate(jump, int(round(1.0 / self.res)))
    self.near05 = dilate(jump, int(round(0.5 / self.res)))
    lv = np.unique(np.round(self.H[np.isfinite(self.H)], 3))
    self.levels = lv[np.concatenate([[True], np.diff(lv) > JUMP])]   # distinct walkable heights (tread tops)

  def level(self, z):
    """Index of the level within 3 cm of z (a contact height), or -1."""
    i = int(np.argmin(np.abs(self.levels - z)))
    return i if abs(self.levels[i] - z) < 0.03 else -1

  def ij(self, x, y):
    i, j = int((x - self.x0) / self.res), int((y - self.y0) / self.res)
    if 0 <= i < self.H.shape[0] and 0 <= j < self.H.shape[1]:
      return i, j
    return None

  def h(self, x, y):
    c = self.ij(x, y)
    return None if c is None else float(self.H[c])

  def along(self, x, y, hx, hy, h0, limit):
    """Distance along (hx, hy) from (x, y) to the first cell whose height differs from h0 by > JUMP (inf within limit)."""
    s = self.res
    while s <= limit:
      v = self.h(x + hx * s, y + hy * s)
      if v is None or abs(v - h0) > JUMP:
        return s
      s += self.res
    return math.inf


GAP = 4   # control steps (0.08 s) off a step before a new touch of it counts


def path_levels(grid, x, y, hx, hy, up):
  """Tread heights along the heading from the spawn, up to where the flight turns."""
  hs = []
  for d in np.arange(0.0, 9.0, grid.res):
    v = grid.h(x + hx * d, y + hy * d)
    if v is None:
      break
    if hs and ((up and v < max(hs) - JUMP) or (not up and v > min(hs) + JUMP)):
      break
    hs.append(round(v, 3))
  lv = np.unique(hs)
  return lv[np.concatenate([[True], np.diff(lv) > JUMP])] if lv.size else np.array([0.0])


def lvl_of(lv, z, nz):
  """The flight level a contact belongs to: a tread top for an up-facing normal, else the step whose riser / nosing it is."""
  if abs(nz) > 0.9:   # MuJoCo's contact normal points geom1 -> geom2: its sign depends on the geom order
    i = int(np.argmin(np.abs(lv - z)))
    return i if abs(lv[i] - z) < 0.03 else -1
  above = np.flatnonzero(lv >= z - 0.01)
  return int(above[0]) if above.size else -1


def run(sc, seed, grid):
  name, scene, pos, yaw, segs, win, up = sc
  secs = sum(d for d, _ in segs)
  model, b = model_for(scene)
  rng = np.random.default_rng(seed)
  data = mujoco.MjData(model)
  p = (pos[0] + rng.uniform(-0.05, 0.05), pos[1] + rng.uniform(-0.05, 0.05), pos[2])
  reset_to_spawn(model, data, b, settle=True, pos=p, yaw_deg=yaw + rng.uniform(-3, 3))
  rt = PieRuntime(model, b, data, PieOnnxPolicy(onnx, provider="cpu"), None,
                  ConstantCmd(vx=0.0, segments=[(d, v, 0) for d, v in segs]))
  root = b.root_qpos_adr
  base_body = model.jnt_bodyid[np.flatnonzero(model.jnt_qposadr == root)[0]]
  dof0 = int(model.jnt_dofadr[np.flatnonzero(model.jnt_qposadr == root)[0]])
  robot_geoms = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == base_body}
  feet = {g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in robot_geoms
          if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")}
  front = {g for g, n in feet.items() if n[:1].upper() == "F"}
  thigh = {}
  for g in feet:
    bid = model.geom_bodyid[g]
    while bid > 0 and not model.body(bid).name.endswith("_thigh"):
      bid = model.body_parentid[bid]
    thigh[g] = bid
  q0 = data.qpos[root + 3:root + 7]
  y0 = math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))
  FL_ = path_levels(grid, float(data.qpos[root]), float(data.qpos[root + 1]), math.cos(y0), math.sin(y0), up)
  pz0 = grid.h(float(data.qpos[root]), float(data.qpos[root + 1]))
  p_start = lvl_of(FL_, pz0, 1.0) if pz0 is not None else -1
  last_t = {g: {} for g in feet}
  ons = {g: {} for g in feet}
  rlast = {g: -99 for g in feet}
  P = dict(riser=0, legc=0, legc_last=-99, ang=[], pitch=[])
  st = {g: dict(on=0, level=None, prev_level=None, kick=False, air=0.0, z0=0.0, apex=0.0, flat_off=False,
                placed=False, stand=None, nosed=False, off_L=-1, t_off=0.0, z_last=0.0, last_slide=None) for g in feet}
  m = dict(kicks=0, mid=0, edge=0, placed=0, slips=0, ups=0, idle_front=0, idle_rear=0, fell=False,
           roll=0.0, w2=0.0, wn=0, clean=0, edged=0)
  # the flight's start level (under the base at the spawn) and end level (the highest / lowest level ahead)
  hz0 = grid.h(float(data.qpos[root]), float(data.qpos[root + 1]))
  start_L = grid.level(hz0) if hz0 is not None else -1
  idle = []        # (take-off level, wheel, time, apex cm): split into start / end after the run
  reached = []     # wheel stance levels (wheeled: the step-up classes' stand)
  t_now = 0.0
  for step in range(int(round(secs / DT))):
    t_now = step * DT
    rt.tick()
    m["fell"] |= rt.is_fallen()
    q = data.qpos[root + 3:root + 7]
    yawn = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    hx, hy = math.cos(yawn), math.sin(yawn)
    bx, by = float(data.qpos[root]), float(data.qpos[root + 1])
    cb = grid.ij(bx, by)
    near_stairs = cb is not None and grid.near1[cb]
    on_stairs = cb is not None and grid.near05[cb]
    if on_stairs:
      roll = math.degrees(math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
      m["roll"] = max(m["roll"], abs(roll))
      P["pitch"].append(math.degrees(math.asin(max(-1.0, min(1.0, float(data.xmat[base_body][6]))))))
      w = data.qvel[dof0 + 3:dof0 + 5]   # free joint angular velocity (body frame) x, y
      m["w2"] += float(w[0] ** 2 + w[1] ** 2)
      m["wn"] += 1
    cons = {}
    legtouch = False
    for con in data.contact[:data.ncon]:
      for g, o in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
        if g in feet and o not in robot_geoms and model.geom_bodyid[o] == 0:
          cons.setdefault(g, []).append((np.array(con.pos), float(con.frame[2])))
        elif g in robot_geoms and g not in feet and o not in robot_geoms and model.geom_bodyid[o] == 0:
          legtouch = True
    if legtouch:
      if step - P["legc_last"] > GAP:
        P["legc"] += 1
      P["legc_last"] = step
    for g, c in cons.items():   # posture: touches per step, riser touches, leg angle
      lv_ = set()
      for cpos, nz in c:
        L = lvl_of(FL_, float(cpos[2]), nz)
        if L >= 0:
          lv_.add(L)
        cc = grid.ij(cpos[0], cpos[1])
        if abs(nz) < 0.5 and cc is not None and grid.near05[cc]:
          if step - rlast[g] > GAP:
            P["riser"] += 1
          rlast[g] = step
      for L in lv_:
        if step - last_t[g].get(L, -99) > GAP:
          ks = [nz for cpos, nz in c if lvl_of(FL_, float(cpos[2]), nz) == L]
          kind = "t" if any(abs(nz) > 0.9 for nz in ks) else ("r" if any(abs(nz) < 0.5 for nz in ks) else "n")
          ons[g].setdefault(L, []).append(kind)
        last_t[g][L] = step
      if on_stairs:
        v = data.geom_xpos[g] - data.xpos[thigh[g]]
        P["ang"].append((feet[g][:1].upper(), math.degrees(math.atan2(v[0] * hx + v[1] * hy, -v[2])),
                         abs(math.degrees(math.atan2(-v[0] * hy + v[1] * hx, -v[2])))))
    for g in feet:
      s = st[g]
      c = cons.get(g)
      x, y, z = data.geom_xpos[g]
      if c:
        horiz = any(abs(nz) < 0.5 for _, nz in c)
        cp = c[0][0]
        cc = grid.ij(cp[0], cp[1])
        over = cc is not None and grid.near05[cc]
        if not WHEELED and horiz and not s["kick"] and over:
          m["kicks"] += 1
        s["kick"] = horiz
        s["on"] += 1
        s["z_last"] = float(z)
        lv = grid.h(cp[0], cp[1])
        if WHEELED and up:   # stair_wheels.py's step-up classes on the contact heights snapped to the levels
          best = None
          for cpos, nz in c:
            L = grid.level(float(cpos[2]))
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
                m["ups"] += 1
                m["edged" if s["nosed"] else "clean"] += 1
              elif s["stand"] is not None and L == s["stand"] and s["nosed"] and not (
                  s["last_slide"] is not None and s["last_slide"][0] == L and t_now - s["last_slide"][1] < 0.5):
                # a wheel bumping the same riser again within 0.5 s is one slide-back, not several (B2W in the
                # pyramid pit: 4 bumps in 0.2 s)
                m["ups"] += 1
                m["slips"] += 1
                s["last_slide"] = (L, t_now)
                if os.environ.get("DEBUG_SLIDE"):
                  print(f"    slide {feet[g]} t{t_now:.2f} back on level {L} ({grid.levels[L]:.3f}) at "
                        f"({float(cp[0]):.2f}, {float(cp[1]):.2f}) heading ({hx:.2f}, {hy:.2f})", flush=True)
              if L != s["stand"] or s["nosed"]:
                s["stand"], s["nosed"] = L, False
              reached.append(L)
        if WHEELED and s["air"] > 0.0:   # landing after a flight: an idle step?
          if s["flat_off"] and s["air"] > 0.06 + 1e-6 and s["apex"] - s["z0"] > 0.03:
            idle.append((s["off_L"], feet[g], round(s["t_off"], 2), round(100 * (s["apex"] - s["z0"]), 1)))
          s["air"] = 0.0
        if not WHEELED and s["on"] == 3 and lv is not None:     # a stance starts
          if s["level"] is not None and up and lv < s["level"] - 0.06 and over:
            m["slips"] += 1
          s["prev_level"], s["level"] = s["level"], lv
          s["placed"] = False
        elif not WHEELED and s["on"] > 3 and lv is not None and s["level"] is not None and lv < s["level"] - 0.06 and over:
          m["slips"] += 1                       # slid off during the stance
          s["level"] = lv
        if (not WHEELED and s["on"] >= 3 and not s["placed"] and lv is not None and over
                and all(abs(nz) > 0.9 for _, nz in c)):
          d_f = grid.along(cp[0], cp[1], hx, hy, lv, 0.7)
          d_b = grid.along(cp[0], cp[1], -hx, -hy, lv, 0.7)
          depth = d_f + d_b
          if 0.15 <= depth <= 0.6:
            frac = d_b / depth   # 0 / 1 = the tread's ends along the heading (symmetric bars)
            m["placed"] += 1
            m["mid"] += 0.2 <= frac <= 0.8
            m["edge"] += min(d_f, d_b) < 0.03
          s["placed"] = True
      else:
        if WHEELED and s["on"] > 0:          # take-off
          lv = grid.h(x, y)
          ahead = grid.along(x, y, hx, hy, lv, 0.3) if lv is not None else 0.0
          s["flat_off"] = near_stairs and math.isinf(ahead) and s["on"] >= 3
          s["off_L"] = grid.level(lv) if lv is not None else -1
          s["t_off"] = t_now
          s["z0"] = s["z_last"]                 # the height at the last contact step (stair_wheels.py's)
          s["apex"] = float(z)
        if WHEELED:
          s["air"] += DT
          s["apex"] = max(s["apex"], float(z))
        s["on"] = 0
        s["kick"] = False
  rt.close()
  allL = sorted({L for g in feet for L in ons[g]})
  p_end = (max(allL) if up else min(allL)) if allL else -1
  m["touch"] = [(feet[g][:1].upper(), "".join(k)) for g in feet for L, k in ons[g].items()
                if (p_start < L < p_end if up else p_end < L < p_start)]
  m["riser_t"], m["legc"], m["ang"], m["pitch"] = P["riser"], P["legc"], P["ang"], P["pitch"]
  if WHEELED:
    end_L = (max(reached) if up else min(reached)) if reached else -1
    for L, w, t, a in idle:
      if L == start_L and L != end_L:
        m["idle_rear"] += 1        # "start": taken off from the start level (any wheel)
      elif L == end_L:
        m["idle_front"] += 1       # "end": from the level the flight ends on
      if os.environ.get("DEBUG_IDLE"):
        print(f"    idle {w} t{t} level {L} (start {start_L}, end {end_L}) apex {a} cm", flush=True)
  return m


def posture(ms):
  tk = [x for m in ms for x in m["touch"]]
  t = [len(k) for _, k in tk]
  ex = {}
  for side in "FR":
    ks = [k for sd, k in tk if sd == side]
    ex[side] = (np.mean([len(k) for k in ks]) if ks else float("nan"),                  # touches per step
                100 * np.mean([k[0] != "t" for k in ks]) if ks else float("nan"),       # first touch not on the tread
                100 * np.mean([k.count("t") >= 2 for k in ks]) if ks else float("nan"))  # landed on its tread again
  a = [x for m in ms for x in m["ang"]]
  F = np.array([(sg, lt) for sd, sg, lt in a if sd == "F"]).reshape(-1, 2)
  Rr = np.array([(sg, lt) for sd, sg, lt in a if sd == "R"]).reshape(-1, 2)
  p = np.array([x for m in ms for x in m["pitch"]])
  nan = float("nan")
  return dict(touch=np.mean(t) if t else nan, multi=100 * np.mean([n >= 2 for n in t]) if t else nan,
              riser=np.mean([m["riser_t"] for m in ms]), legc=np.mean([m["legc"] for m in ms]),
              aF=F[:, 0].mean() if len(F) else nan, aR=Rr[:, 0].mean() if len(Rr) else nan,
              absF=np.abs(F[:, 0]).mean() if len(F) else nan, absR=np.abs(Rr[:, 0]).mean() if len(Rr) else nan,
              latF=F[:, 1].mean() if len(F) else nan, latR=Rr[:, 1].mean() if len(Rr) else nan,
              pitch=p.mean() if len(p) else nan, pitch_sd=p.std() if len(p) else nan, ex=ex)


def pline(q):
  e = q["ex"]
  return (f"touches {q['touch']:.2f} (>=2: {q['multi']:4.1f}%; front {e['F'][0]:.2f} bump-first {e['F'][1]:3.0f}% re-land {e['F'][2]:3.0f}%, "
          f"rear {e['R'][0]:.2f} bump-first {e['R'][1]:3.0f}% re-land {e['R'][2]:3.0f}%) riser {q['riser']:4.1f} leg {q['legc']:3.1f} | angle F {q['aF']:+5.1f} "
          f"R {q['aR']:+5.1f} (|F| {q['absF']:4.1f} |R| {q['absR']:4.1f}, lat {q['latF']:4.1f}/{q['latR']:4.1f}) pitch {q['pitch']:+5.1f} sd {q['pitch_sd']:3.1f}")


KEYS = ("kicks", "mid", "edge", "placed", "slips", "ups", "idle_front", "idle_rear", "w2", "wn", "clean", "edged")
print(f"{robot} {onnx.stem} ({'wheeled' if WHEELED else 'legged'}) map3 flights, {runs} runs per scenario"
      + (f", payload {PAYLOAD} (kg, x, y, z m)" if PAYLOAD else ""), flush=True)
grids, groups = {}, {"up": [], "down": []}
for sc in SCEN:
  name, scene, pos, yaw, _, win, up = sc
  model, _b = model_for(scene)
  grid = grids.setdefault((scene, win), Grid(model, win))
  ms = [run(sc, s, grid) for s in range(runs)]
  groups["up" if up else "down"].append((name, ms))
  agg = {k: sum(mm[k] for mm in ms) for k in KEYS}
  falls = sum(mm["fell"] for mm in ms)
  roll = max(mm["roll"] for mm in ms)
  wr = (agg["w2"] / max(agg["wn"], 1)) ** 0.5
  if WHEELED:
    line = (f"slide-back {100 * agg['slips'] / max(agg['ups'], 1):4.1f}% of {agg['ups']:3d} | idle start "
            f"{agg['idle_rear'] / runs:.1f} end {agg['idle_front'] / runs:.1f}")
  else:
    line = (f"kicks {agg['kicks'] / runs:4.1f} | mid {100 * agg['mid'] / max(agg['placed'], 1):4.1f}% edge "
            f"{100 * agg['edge'] / max(agg['placed'], 1):4.1f}% of {agg['placed']:3d} | slips {agg['slips'] / runs:.1f}")
  print(f"  {name:16s} {line} | roll {roll:4.1f} | falls {falls}/{runs} | {pline(posture(ms))}", flush=True)


def total(label, entries):
  ms = [m for _, mm in entries for m in mm]
  if not ms:
    return
  n = len(ms)
  agg = {k: sum(m[k] for m in ms) for k in KEYS}
  falls = sum(m["fell"] for m in ms)
  roll = max(m["roll"] for m in ms)
  wr = (agg["w2"] / max(agg["wn"], 1)) ** 0.5
  if WHEELED:
    bars = {"slide": 5.0, "start": 0.5, "end": 1.0, "roll": 4.0}
    bars.update({k: float(v) for k, v in (kv.split("=") for kv in filter(None, os.environ.get("WHEELED_BARS", "").split(",")))})
    sb = 100 * agg["slips"] / max(agg["ups"], 1)
    checks = [("slide-back", sb <= bars["slide"], f"{sb:.1f}%"), ("start idle", agg["idle_rear"] / n <= bars["start"], f"{agg['idle_rear'] / n:.2f}"),
              ("end idle", agg["idle_front"] / n <= bars["end"], f"{agg['idle_front'] / n:.2f}"),
              ("roll", roll <= bars["roll"], f"{roll:.1f} deg"), ("falls", falls == 0, str(falls))]
  else:
    mid = 100 * agg["mid"] / max(agg["placed"], 1)
    edge = 100 * agg["edge"] / max(agg["placed"], 1)
    bars = {"kicks": 2.0, "mid": 80.0, "edge": 10.0, "slips": 0.2, "roll": 6.0}
    bars.update({k: float(v) for k, v in (kv.split("=") for kv in filter(None, os.environ.get("LEGGED_BARS", "").split(",")))})
    checks = [("kicks", agg["kicks"] / n <= bars["kicks"], f"{agg['kicks'] / n:.2f}/run"),
              ("mid-tread", mid >= bars["mid"], f"{mid:.1f}%"), ("edge", edge <= bars["edge"], f"{edge:.1f}%"),
              ("slips", agg["slips"] / n <= bars["slips"], f"{agg['slips'] / n:.2f}/run"),
              ("roll", roll <= bars["roll"], f"{roll:.1f} deg"), ("falls", falls == 0, str(falls))]
  fails = [k for k, ok, _ in checks if not ok]
  print(f"{label} {robot} {onnx.stem}{' payload ' + PAYLOAD if PAYLOAD else ''}: "
        + " | ".join(f"{k} {v}{'' if ok else ' FAIL'}" for k, ok, v in checks)
        + f" | rate RMS {wr:.2f} rad/s -> {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}", flush=True)
  print(f"{label.replace('TOTAL', 'POSTURE')} {robot} {onnx.stem}: {pline(posture(ms))}", flush=True)


total("TOTAL-up", groups["up"])
total("TOTAL-down", groups["down"])
total("TOTAL", groups["up"] + groups["down"])
