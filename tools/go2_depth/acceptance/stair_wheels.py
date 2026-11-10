"""Wheel behaviour on map3's L0 climb in tinynav-sim (2026-11-03, the user's two wheeled-stairs complaints):
1. needless steps: a wheel still on the start landing leaves the ground (airborne > AIR_MIN s, apex > LIFT above
   its take-off) although the first riser is farther than NEED m ahead of it (it could roll there);
2. roll-back: a wheel that touched a tread is later supported by the tread below for > LOW_MIN s (it went back down
   a step): how long it had touched the upper one, how far it moved back along the heading, whether it left the
   ground in between, and the wheel's rim speed (policy target / actual, m/s along the heading) just before.

stair_wheels.py ROBOT ONNX [RUNS]   (from a tinynav-sim worktree's mujoco/, with ROBOT_ASSETS and MUJOCO_GL=egl)
The climb: edge_tn.py's (spawn (-10.34, 5.75) yaw 270 +- seeds, 0.5 m/s for 10 s). Wheel support level = the
raster height under the wheel centre at contact, snapped to the flight's tread heights.
"""
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, ".")
from sim.keyboard import ConstantCmd  # noqa: E402
from sim.plant import build_map3_model, reset_to_spawn  # noqa: E402
from sim.policy import PieOnnxPolicy  # noqa: E402
from sim.runtime import PieRuntime  # noqa: E402

JUMP, RES, DT = 0.03, 0.01, 0.02
AIR_MIN, LIFT, NEED, LOW_MIN = 0.06, 0.02, 0.3, 0.1
robot, onnx = sys.argv[1], Path(sys.argv[2])
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
model, b = build_map3_model(robot)

X0, X1, Y0, Y1 = -12.5, -7.5, -1.0, 8.0
xs, ys = np.arange(X0, X1, RES), np.arange(Y0, Y1, RES)
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
  dx, dy = gx - pos[0], gy - pos[1]
  lx = R[0, 0] * dx + R[1, 0] * dy
  ly = R[0, 1] * dx + R[1, 1] * dy
  inside = (np.abs(lx) <= half[0]) & (np.abs(ly) <= half[1])
  top = pos[2] + half[2] * abs(R[2, 2])
  H = np.where(inside & (top > H) & (top < 3.0), top, H)  # L0 only: the flights above share its x/y
H[~np.isfinite(H)] = 0.0


def h_at(x, y):
  i, j = int((x - X0) / RES), int((y - Y0) / RES)
  return float(H[min(max(i, 0), H.shape[0] - 1), min(max(j, 0), H.shape[1] - 1)])


def riser_ahead(x, y, hd, h0):
  """Distance along the heading to the first cell higher than h0 + JUMP (inf within 1.5 m)."""
  for k in range(1, 151):
    if h_at(x + hd[0] * k * RES, y + hd[1] * k * RES) > h0 + JUMP:
      return k * RES
  return math.inf


# the flight's tread heights along the climb line
line = sorted({round(h_at(-10.34, y), 3) for y in np.arange(5.75, -1.0, -RES)})
treads = np.array([h for i, h in enumerate(line) if i == 0 or h - line[i - 1] > JUMP])


NOSE_Y = {1: math.inf}
for g in range(model.ngeom):
  nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
  if nm.startswith("step_L0_"):
    NOSE_Y[int(nm[-2:]) + 1] = float(model.geom_pos[g][1] + model.geom_size[g][1])
TOP_L0 = max(NOSE_Y)
_d = __import__("os").environ.get("DBG")   # DBG=FL,SEED,T0,T1
DBG = (_d.split(",")[0], int(_d.split(",")[1]), float(_d.split(",")[2]), float(_d.split(",")[3])) if _d else None
FLOOR = 1  # treads[0] = 0 is outside the boxes, [1] the landing the climb starts on


def level(z):
  return int(np.argmin(np.abs(treads - z)))


def run(seed):
  rng = np.random.default_rng(seed)
  data = mujoco.MjData(model)
  pos = (-10.34 + rng.uniform(-0.05, 0.05), 5.75 + rng.uniform(-0.05, 0.05), 0.83)
  reset_to_spawn(model, data, b, settle=True, pos=pos, yaw_deg=270.0 + rng.uniform(-3, 3))
  rt = PieRuntime(model, b, data, PieOnnxPolicy(onnx, provider="cpu"), None, ConstantCmd(vx=0.0, segments=[(10.0, 0.5, 0)]))
  root = b.root_qpos_adr
  base_body = model.jnt_bodyid[np.flatnonzero(model.jnt_qposadr == root)[0]]
  robot_geoms = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == base_body}
  wheels = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)[:2]: g for g in robot_geoms
            if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")}
  r = float(model.geom_size[next(iter(wheels.values()))][0])
  # each wheel's joint (the one on the wheel's body) and its velocity actuator
  wj = {k: int(np.flatnonzero(model.jnt_bodyid == model.geom_bodyid[g])[0]) for k, g in wheels.items()}
  wa = {k: int(np.flatnonzero((model.actuator_trnid[:, 0] == j))[0]) for k, j in wj.items()}
  log = {k: [] for k in wheels}   # t, contact, level, on the nosing, along, z, rim target, rim actual, riser ahead
  fell = False
  for step in range(int(round(10.0 / DT))):
    rt.tick()
    fell |= rt.is_fallen()
    t = step * DT
    q = data.qpos[root + 3:root + 7]
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    hd = (math.cos(yaw), math.sin(yaw))
    on, top = set(), {}
    for con in data.contact[:data.ncon]:
      for g, o in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
        if o not in robot_geoms:
          on.add(g)
          zc = float(con.pos[2])
          i = int(np.argmin(np.abs(treads - zc)))
          if abs(treads[i] - zc) < 0.03:   # on a tread's top face or its nosing (not low on a riser)
            nose = abs(con.frame[2]) <= 0.9
            if g not in top or i > top[g][0] or (i == top[g][0] and not nose):
              top[g] = (i, nose)
    if DBG and seed == DBG[1] and DBG[2] <= t <= DBG[3]:
      g = wheels[DBG[0]]
      cs = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2 if c.geom1 == g else c.geom1),
             np.round(c.pos, 3).tolist(), round(float(c.frame[2]), 2)) for c in data.contact[:data.ncon] if g in (c.geom1, c.geom2)]
      print(f"DBG t {t:.2f} {DBG[0]} centre {np.round(data.geom_xpos[g], 3).tolist()} contacts {cs}")
    for k, g in wheels.items():
      x, y, z = data.geom_xpos[g]
      ax = data.xaxis[wj[k]]
      roll = r * (ax[1] * hd[0] - ax[0] * hd[1])   # rim speed along the heading per rad/s of the joint
      log[k].append((t, g in on, top[g][0] if g in top else -1, top.get(g, (0, False))[1], y, z,
                     roll * float(data.ctrl[wa[k]]), roll * float(data.qvel[model.jnt_dofadr[wj[k]]]),
                     riser_ahead(x, y, hd, h_at(x, y))))
  rt.close()
  return log, fell


TOP_LEVEL = len(treads) - 1          # the landing above L0 (z 2.659)
WALL_Y = 0.40                          # the stairwell wall beyond the top landing (rail_outer_south)


def analyse(log):
  lifts, backs, tops = [], [], []
  for k, rows in log.items():
    a = np.array(rows, dtype=float)
    t, on, lv, nose, y, z, tgt, act, ahead = a.T
    on = on > 0
    # lifts on the start landing
    i, n = 0, len(t)
    while i < n:
      if not on[i] and i > 0 and lv[i - 1] == FLOOR and t[i] < 4.0:
        j = i
        while j < n and not on[j]:
          j += 1
        if (j - i) * DT > AIR_MIN and z[i:j].max() - z[i - 1] > LIFT and ahead[i - 1] > NEED:
          lifts.append((k, t[i], ahead[i - 1], z[i:j].max() - z[i - 1]))
        i = j
      i += 1
    # steps on the top landing: the wheel stood on the landing above L0, then left the ground for > AIR_MIN s with
    # an apex > LIFT above its take-off while still more than 0.5 m short of the wall beyond (flat ahead)
    on_top = np.flatnonzero((lv == TOP_LEVEL) & (nose == 0))
    if len(on_top):
      i = int(on_top[0]) + 1
      while i < n:
        if not on[i] and on[i - 1] and y[i - 1] > WALL_Y + 0.5:
          j = i
          while j < n and not on[j]:
            j += 1
          if (j - i) * DT > AIR_MIN and z[i:j].max() - z[i - 1] > LIFT:
            tops.append((k, t[i], y[i - 1] - WALL_Y, z[i:j].max() - z[i - 1]))
          i = j
        i += 1
    # each step up: the wheel's next tread contact after standing on tread L-1 is on tread L's top face (clean),
    # or on its nosing first and then on its top (climbed the edge), or back on L-1's top (slid back off the edge)
    stand, nosed, n0 = None, False, None
    for idx in np.flatnonzero(lv >= 0):
      L, nz = int(lv[idx]), bool(nose[idx])
      if L > TOP_L0:
        break
      if stand is not None and L == stand + 1 and nz and not nosed:
        nosed, n0 = True, idx
      elif not nz:
        if stand is not None and L == stand + 1:
          backs.append((k, t[idx], L, "climbed the edge" if nosed else "clean", (idx - n0) * DT if nosed else 0.0,
                        z[n0] - treads[L] if nosed else 0.0, tgt[n0:idx + 1].mean() if nosed else 0.0,
                        act[n0:idx + 1].mean() if nosed else 0.0))
        elif stand is not None and L == stand and nosed:
          backs.append((k, t[idx], stand + 1, "slid back", (idx - n0) * DT, z[n0] - treads[stand + 1],
                        tgt[n0:idx + 1].mean(), act[n0:idx + 1].mean()))
        if L != stand or nosed:
          stand, nosed = L, False
  return lifts, backs, tops


print(f"{robot} {onnx.stem}: treads {np.round(treads, 3).tolist()}", flush=True)
tot_l = tot_b = tot_t = 0
top_front = top_rear = 0
n_how = {"clean": 0, "climbed the edge": 0, "slid back": 0}
SEED0 = int(os.environ.get("SEED0", "0"))  # first seed (slide_gate.sh reruns 12..23)
for seed in range(SEED0, SEED0 + runs):
  log, fell = run(seed)
  lifts, backs, tops = analyse(log)
  tot_l, tot_b, tot_t = tot_l + len(lifts), tot_b + len(backs), tot_t + len(tops)
  top_front += sum(1 for x in tops if x[0][0] == "F"); top_rear += sum(1 for x in tops if x[0][0] == "R")
  print(f" seed {seed}: fell {fell}; landing steps with the riser > {NEED} m ahead {len(lifts)}; steps up: "
        + ", ".join(f"{h} {sum(1 for x in backs if x[3] == h)}" for h in n_how)
        + f"; top-landing steps {len(tops)} (" + ", ".join(f"{x[0]} {100 * x[3]:.0f}cm @{100 * (1.325 - x[2]):.0f}cm past the edge t{x[1]:.1f}" for x in tops) + ")", flush=True)
  for k, t, d, h in lifts:
    print(f"   step {k} t {t:5.2f} s riser {d:4.2f} m ahead, lift {100 * h:4.1f} cm")
  if "VERBOSE" in __import__("os").environ:
    for k, t, L, how, dur, dz, tg, ac in backs:
      if how != "clean":
        print(f"   {k} t {t:5.2f} s tread {L}: {how} after {dur:4.2f} s on the nosing (wheel centre {100 * dz:+5.1f} cm "
              f"over the tread top); rim target {tg:+.2f} actual {ac:+.2f} m/s")
  for how in ("clean", "climbed the edge", "slid back"):
    n_how[how] += sum(1 for x in backs if x[3] == how)
ups = max(sum(n_how.values()), 1)
print(f"{robot} {onnx.stem} TOTAL {runs} climbs: landing steps {tot_l} ({tot_l / runs:.1f} per climb); steps up {ups}: "
      + ", ".join(f"{h} {n} ({100 * n / ups:.0f}%)" for h, n in n_how.items())
      + f"; top-landing steps {tot_t} ({tot_t / runs:.1f} per climb: front {top_front}, rear {top_rear})")
