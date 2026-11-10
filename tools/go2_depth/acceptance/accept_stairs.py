"""accept_stairs.py ROBOT ONNX [RUNS] (go2-depth 2026-11-05): the user's stairs requirements, legged and wheeled, in
tinynav-sim (map3 L0 up / R2b down, and the "pyramid" scene's pits and pyramids). The user (11-05): legged -- few riser
kicks, feet mid-tread, no foot slipping back from the step above, a stable body; wheeled -- no riser / placement
requirement, slide-back limited, no idle steps at the start and end of a flight, a stable body.

Step levels come from a height raster of the scene's static boxes (1-2 cm cells, a z window per scenario). Per foot /
wheel and control step: its contacts with static boxes (position, normal). A "stance" = >= 3 control steps (0.06 s) of
contact; its level = the raster under the contact point. Metrics (sum over the runs of a scenario):
  kicks        onsets of a contact with a near-horizontal normal (|n_z| < 0.5) over stairs            (legged)
  placement    each stance on a tread (0.15-0.6 m deep along the heading): its position along the tread (0 / 1 = its ends); "mid" = 0.2-0.8, "edge" = the contact within 3 cm of either end   (legged)
  slips        a stance whose level drops > 6 cm while it lasts (slid off an edge), or a stance > 6 cm below the
               same foot / wheel's previous stance while climbing (went back down a step); wheeled: per step-up %
  idle steps   a wheel airborne > 0.06 s with an apex > 3 cm over its take-off (a visible step; 2 cm caught start-up
               jitter stair_wheels.py did not), taken off from flat ground with no
               height change > 3 cm within 0.3 m ahead along the heading, while the robot is near stairs (a jump
               within 1.0 m of the base); split front / rear wheels                                      (wheeled)
  stability    over stairs: peak |roll| (deg) and the RMS of the body roll / pitch rates (rad/s)
plus falls. Output: one line per scenario and a TOTAL line with the verdict against the initial bars
(stairs-acceptance-requirements memory): legged kicks <= 2 / climb, mid >= 80%, edge <= 10%, slips <= 0.2 / climb,
roll <= 6 deg; wheeled slide-back <= 5% of step-ups, idle steps start (rear) <= 0.5 and end (front) <= 1.0 per climb,
roll <= 4 deg; falls 0. Run from a tinynav-sim worktree's mujoco/ with ROBOT_ASSETS and MUJOCO_GL=egl.
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

# (name, scene, spawn (x, y, z hint), yaw, (seconds, vx), raster window x0 x1 y0 y1 zlo zhi res, climbing up?)
SCEN = [
  ("map3_up", "map3", (-10.34, 5.75, 0.83), 270.0, (11.0 if not WHEELED else 10.5, 0.5), (-12.5, -7.5, -1.0, 8.0, -1.0, 3.0, 0.01), True),
  ("map3_down", "map3", tuple(DOWN_SPAWN), DOWN_YAW_DEG, (8.0, 0.4), None, False),
  ("pit_14_up", "pyramid", SCENES["pyramid"].spawns["pit_14"][0], 0.0, (7.5, 0.5), (6.0, 18.0, -2.0, 10.0, -3.0, 3.0, 0.02), True),
  ("pit_18n_up", "pyramid", SCENES["pyramid"].spawns["pit_18_narrow"][0], 0.0, (7.5, 0.5), (22.0, 34.0, -2.0, 10.0, -3.0, 3.0, 0.02), True),
  ("top_14_down", "pyramid", SCENES["pyramid"].spawns["top_14"][0], 0.0, (9.0, 0.4), (6.0, 18.0, 6.0, 18.0, -3.0, 3.0, 0.02), False),
  ("top_18n_down", "pyramid", SCENES["pyramid"].spawns["top_18_narrow"][0], 0.0, (9.0, 0.4), (22.0, 34.0, 6.0, 18.0, -3.0, 3.0, 0.02), False),
]
if ONLY:
  SCEN = [s for s in SCEN if s[0] in ONLY.split(",")]
_models = {}


PAYLOAD = os.environ.get("PAYLOAD")   # "kg,x,y,z": a point mass on the base (training's mdp.base_payload_point)
NOSLIP = int(os.environ.get("NOSLIP", "0"))   # MuJoCo noslip iterations (2026-11-10: static friction, no contact creep)


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
    if NOSLIP:
      _models[scene][0].opt.noslip_iterations = NOSLIP
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


def run(sc, seed, grid):
  name, scene, pos, yaw, (secs, vx), win, up = sc
  model, b = model_for(scene)
  rng = np.random.default_rng(seed)
  data = mujoco.MjData(model)
  p = (pos[0] + rng.uniform(-0.05, 0.05), pos[1] + rng.uniform(-0.05, 0.05), pos[2])
  reset_to_spawn(model, data, b, settle=True, pos=p, yaw_deg=yaw + rng.uniform(-3, 3))
  rt = PieRuntime(model, b, data, PieOnnxPolicy(onnx, provider="cpu"), None,
                  ConstantCmd(vx=0.0, segments=[(secs, vx, 0)]))
  root = b.root_qpos_adr
  base_body = model.jnt_bodyid[np.flatnonzero(model.jnt_qposadr == root)[0]]
  dof0 = int(model.jnt_dofadr[np.flatnonzero(model.jnt_qposadr == root)[0]])
  robot_geoms = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == base_body}
  feet = {g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in robot_geoms
          if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")}
  front = {g for g, n in feet.items() if n[:1].upper() == "F"}
  st = {g: dict(on=0, level=None, prev_level=None, kick=False, air=0.0, z0=0.0, apex=0.0, flat_off=False,
                placed=False, stand=None, nosed=False, off_L=-1, t_off=0.0, z_last=0.0, last_slide=None) for g in feet}
  m = dict(kicks=0, mid=0, edge=0, placed=0, slips=0, ups=0, idle_front=0, idle_rear=0, fell=False,
           roll=0.0, w2=0.0, wn=0, clean=0, edged=0, hd_max=0.0, hd_sum=0.0, r_sum=0.0, hn=0)
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
      # heading off the flight's axis (11-07, the user: "它不是头朝前上楼梯"): the spawn yaw without its +-3 deg noise
      hd = abs(math.degrees(math.atan2(math.sin(yawn - math.radians(yaw)), math.cos(yawn - math.radians(yaw)))))
      m["hd_max"] = max(m["hd_max"], hd)
      m["hd_sum"] += hd
      m["r_sum"] += abs(roll)
      m["hn"] += 1
      w = data.qvel[dof0 + 3:dof0 + 5]   # free joint angular velocity (body frame) x, y
      m["w2"] += float(w[0] ** 2 + w[1] ** 2)
      m["wn"] += 1
    cons = {}
    for con in data.contact[:data.ncon]:
      for g, o in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
        if g in feet and o not in robot_geoms and model.geom_bodyid[o] == 0:
          cons.setdefault(g, []).append((np.array(con.pos), float(con.frame[2])))
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
            idle.append((s["off_L"], feet[g], round(s["t_off"], 2), round(100 * (s["apex"] - s["z0"]), 1),
                         grid.level(float(cp[2]))))
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
  if WHEELED:
    end_L = (max(reached) if up else min(reached)) if reached else -1
    for L, w, t, a, Lland in idle:
      if L == start_L and L != end_L:
        m["idle_rear"] += 1        # "start": taken off from the start level (any wheel)
      elif L == end_L:
        m["idle_front"] += 1       # "end": from the level the flight ends on
      if os.environ.get("DEBUG_IDLE"):
        print(f"    idle {w} t{t} level {L} -> landed on {Lland} (start {start_L}, end {end_L}) apex {a} cm", flush=True)
  return m


tot = dict(kicks=0, mid=0, edge=0, placed=0, slips=0, ups=0, idle_front=0, idle_rear=0, falls=0, roll=0.0, w2=0.0,
           wn=0, climbs=0, clean=0, edged=0, hd_sum=0.0, hn=0)
print(f"{robot} {onnx.stem} ({'wheeled' if WHEELED else 'legged'}), {runs} runs per scenario"
      + (f", payload {PAYLOAD} (kg, x, y, z m)" if PAYLOAD else "") + (f", noslip {NOSLIP}" if NOSLIP else ""), flush=True)
grids = {}
for sc in SCEN:
  name, scene, pos, yaw, _, win, up = sc
  model, _b = model_for(scene)
  if win is None:   # map3 R2b: around the down spawn
    win = (pos[0] - 3.0, pos[0] + 3.0, pos[1] - 6.0, pos[1] + 6.0, pos[2] - 3.5, pos[2] + 0.3, 0.01)
  grid = grids.setdefault((scene, win), Grid(model, win))
  ms = [run(sc, s, grid) for s in range(runs)]
  agg = {k: sum(mm[k] for mm in ms) for k in ("kicks", "mid", "edge", "placed", "slips", "ups", "idle_front", "idle_rear", "w2", "wn", "clean", "edged")}
  falls = sum(mm["fell"] for mm in ms)
  roll = max(mm["roll"] for mm in ms)
  for k in agg:
    tot[k] += agg[k]
  tot["falls"] += falls
  tot["roll"] = max(tot["roll"], roll)
  tot["climbs"] += runs
  wr = (agg["w2"] / max(agg["wn"], 1)) ** 0.5
  if WHEELED:
    line = (f"slide-back {100 * agg['slips'] / max(agg['ups'], 1):4.1f}% of {agg['ups']} step-ups (clean {agg['clean']}, edge {agg['edged']}) | idle steps start "
            f"{agg['idle_rear'] / runs:.1f} end {agg['idle_front'] / runs:.1f} / run")
  else:
    line = (f"kicks {agg['kicks'] / runs:.1f} / run | mid-tread {100 * agg['mid'] / max(agg['placed'], 1):4.1f}% edge "
            f"{100 * agg['edge'] / max(agg['placed'], 1):4.1f}% of {agg['placed']} | slips {agg['slips'] / runs:.1f} / run")
  print(f"  {name:13s} {line} | roll peak {roll:4.1f} deg, rate RMS {wr:.2f} rad/s | falls {falls}/{runs}", flush=True)
  hn = max(sum(mm["hn"] for mm in ms), 1)
  tot["hd_sum"] += sum(mm["hd_sum"] for mm in ms)
  tot["hn"] += sum(mm["hn"] for mm in ms)
  hdm = " ".join(f"{mm['hd_max']:.0f}" for mm in ms)
  print(f"    {name} on stairs: heading off the flight mean {sum(mm['hd_sum'] for mm in ms) / hn:4.1f} deg, max per run "
        f"{hdm} | |roll| mean {sum(mm['r_sum'] for mm in ms) / hn:4.1f} deg", flush=True)

n = max(tot["climbs"], 1)
wr = (tot["w2"] / max(tot["wn"], 1)) ** 0.5
if WHEELED:
  sb = 100 * tot["slips"] / max(tot["ups"], 1)
  checks = [("slide-back", sb <= 5.0, f"{sb:.1f}%"), ("start idle", tot["idle_rear"] / n <= 0.5, f"{tot['idle_rear'] / n:.1f}"),
            ("end idle", tot["idle_front"] / n <= 1.0, f"{tot['idle_front'] / n:.1f}"),
            ("roll", tot["roll"] <= 4.0, f"{tot['roll']:.1f} deg"), ("falls", tot["falls"] == 0, str(tot["falls"]))]
  # heading off the flights' axis on the stairs, mean (11-07, the user: "它不是头朝前上楼梯"; Go2W blind g20b 18300 5.6 deg
  # on the up flights, the deployed go2w blind_v2 / depth_v5_ws 1.5-1.6). WHEELED_HEADING_BAR (deg) overrides the bar.
  hdm = tot["hd_sum"] / max(tot["hn"], 1)
  checks.insert(4, ("heading", hdm <= float(os.environ.get("WHEELED_HEADING_BAR", "3.0")), f"{hdm:.1f} deg"))
else:
  mid = 100 * tot["mid"] / max(tot["placed"], 1)
  edge = 100 * tot["edge"] / max(tot["placed"], 1)
  # LEGGED_BARS="kicks=,mid=,edge=,slips=,roll=" (2026-11-06): the initial bars calibrated on a deployed baseline (no B2
  # meets kicks <= 2 / mid >= 80: deployed depth b2_depth_v3_fe 3.1 / 76.6%, blind b2_fe_mixed_5000 6.1 / 78.3%).
  bars = {"kicks": 2.0, "mid": 80.0, "edge": 10.0, "slips": 0.2, "roll": 6.0}
  bars.update({k: float(v) for k, v in (kv.split("=") for kv in filter(None, os.environ.get("LEGGED_BARS", "").split(",")))})
  checks = [("kicks", tot["kicks"] / n <= bars["kicks"], f"{tot['kicks'] / n:.1f}/run"),
            ("mid-tread", mid >= bars["mid"], f"{mid:.1f}%"), ("edge", edge <= bars["edge"], f"{edge:.1f}%"),
            ("slips", tot["slips"] / n <= bars["slips"], f"{tot['slips'] / n:.1f}/run"),
            ("roll", tot["roll"] <= bars["roll"], f"{tot['roll']:.1f} deg"), ("falls", tot["falls"] == 0, str(tot["falls"]))]
fails = [k for k, ok, _ in checks if not ok]
print(f"TOTAL {robot} {onnx.stem}{' payload ' + PAYLOAD if PAYLOAD else ''}: " + " | ".join(f"{k} {v}{'' if ok else ' FAIL'}" for k, ok, v in checks)
      + f" | rate RMS {wr:.2f} rad/s -> {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}", flush=True)
