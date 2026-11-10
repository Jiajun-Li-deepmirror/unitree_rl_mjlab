"""roll_probe.py ROBOT ONNX [RUNS] (go2-depth 2026-11-06, the user: why is the roll peak high): accept_map3.py's map3
flights; per run the |roll| peak on stairs with its context (time, base, each foot's contact / height over the tread
under it, the roll rate, command), and the roll / roll-rate shares above 4 / 6 deg. ONLY=<names>, TOP=<n> events."""
import math
import os
import sys

import mujoco
import numpy as np

C = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(C, "accept_map3.py")).read()
exec(src[:src.index("def run(")])   # robot, onnx, runs, SCEN, Grid, model_for, ...

TOP = int(os.environ.get("TOP", "8"))
events, shares = [], []
for sc in SCEN:
  name, scene, pos, yaw, segs, win, up = sc
  model, b = model_for(scene)
  grid = Grid(model, win)
  for seed in range(runs):
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
    order = sorted(feet, key=lambda g: feet[g])
    hist, rolls = [], []
    for step in range(int(round(sum(d for d, _ in segs) / DT))):
      rt.tick()
      q = data.qpos[root + 3:root + 7]
      roll = math.degrees(math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
      bx, by = float(data.qpos[root]), float(data.qpos[root + 1])
      cb = grid.ij(bx, by)
      on = cb is not None and grid.near05[cb]
      touch = {g: False for g in feet}
      for con in data.contact[:data.ncon]:
        for g, o in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
          if g in feet and o not in robot_geoms and model.geom_bodyid[o] == 0:
            touch[g] = True
      # each foot's height over the tread under it (cm; +: in the air) and its sideways offset from the base (cm)
      yawn = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
      fz = []
      for g in order:
        x, y, z = data.geom_xpos[g]
        hgt = grid.h(x, y)
        fz.append((feet[g][:2], touch[g], 100 * (z - (hgt if hgt is not None else z)) - 100 * float(model.geom_size[g][0])))
      wx = float(data.qvel[dof0 + 3])
      hist.append((step * DT, roll, wx, on, fz, float(rt.cmd_source.vector()[0]) if False else None))
      if on:
        rolls.append(abs(roll))
    rt.close()
    if rolls:
      shares.append((name, seed, np.mean(np.array(rolls) > 4), np.mean(np.array(rolls) > 6), max(rolls)))
    onh = [h for h in hist if h[3]]
    if onh:
      k = max(range(len(hist)), key=lambda i: abs(hist[i][1]) if hist[i][3] else -1)
      events.append((abs(hist[k][1]), name, seed, k, hist))
events.sort(key=lambda e: -e[0])
print(f"{robot} {onnx.stem}: |roll| on stairs, share of steps > 4 / > 6 deg, peak (per flight, mean over runs)")
by = {}
for n, s, a4, a6, pk in shares:
  by.setdefault(n, []).append((a4, a6, pk))
for n, v in by.items():
  v = np.array(v)
  print(f"  {n:16s} >4: {100 * v[:, 0].mean():5.1f}%  >6: {100 * v[:, 1].mean():5.1f}%  peak {v[:, 2].mean():5.1f} (max {v[:, 2].max():5.1f})")
allv = np.array([(a4, a6) for _, _, a4, a6, _ in shares])
print(f"  ALL              >4: {100 * allv[:, 0].mean():5.1f}%  >6: {100 * allv[:, 1].mean():5.1f}%")
for pk, name, seed, k, hist in events[:TOP]:
  print(f"PEAK {pk:5.1f} deg  {name} run {seed} t {hist[k][0]:.2f}")
  for i in range(max(0, k - 15), min(len(hist), k + 6), 3):
    t, roll, wx, on, fz, _ = hist[i]
    feet_s = "  ".join(f"{n}{'*' if c else ' '}{h:+5.1f}" for n, c, h in fz)
    print(f"    t {t:5.2f} roll {roll:+6.1f} rate {wx:+5.2f} | {feet_s}  (* = in contact; cm over the tread below)")
