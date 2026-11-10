"""Stance feet on stair edges in tinynav-sim (2026-10-31, the feetedge round): a policy climbs map3's L0 at 0.5 m/s
and descends R2b at 0.4 m/s (obstacles.py's up_clean / down_clean starts), a few seeds of the spawn (small x/y/yaw
offsets). Every control step each foot in contact counts as on an edge when its contact disc (radius DISC_R around
the foot geom centre) reaches a riser edge, a height jump > JUMP between 1 cm cells of map3's rasterized boxes (as
jetson_b2/tests/edge_stance.py). Only feet over stairs count (the raster has a jump within 0.5 m).

edge_tn.py ROBOT ONNX [RUNS]   (from the worktree's mujoco/, with ROBOT_ASSETS and MUJOCO_GL=egl)
DEPTH_LATENCY=N (control steps): the depth the runtime takes is the frame rendered N steps earlier (training's real
camera: 7-9.5 steps; navcore measured 158 ms = 8 steps); unset or 0 = tinynav-sim's own (rendered when taken).
"""
import math
import os
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, ".")
from sim.keyboard import ConstantCmd  # noqa: E402
from sim.plant import DOWN_SPAWN, DOWN_YAW_DEG, build_map3_model, reset_to_spawn  # noqa: E402
from sim.policy import PieOnnxPolicy  # noqa: E402
from sim.runtime import PieRuntime  # noqa: E402

DISC_R, JUMP, RES, DT = 0.035, 0.03, 0.01, 0.02
robot, onnx = sys.argv[1], Path(sys.argv[2])
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
model, b = build_map3_model(robot)

# Height raster of the static boxes over the two flights' area.
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
  H = np.where(inside & (top > H) & (top < 30.0), top, H)
H[~np.isfinite(H)] = 0.0
jump = np.zeros(H.shape, bool)
for ax in (0, 1):
  d = np.abs(np.diff(H, axis=ax)) > JUMP
  if ax == 0:
    jump[:-1] |= d
    jump[1:] |= d
  else:
    jump[:, :-1] |= d
    jump[:, 1:] |= d


def dilate(mask, r_cells):
  out = mask.copy()
  for i in range(-r_cells, r_cells + 1):
    for j in range(-r_cells, r_cells + 1):
      if i * i + j * j <= r_cells * r_cells:
        out |= np.roll(np.roll(mask, i, 0), j, 1)
  return out


edge = dilate(jump, int(round(DISC_R / RES)))
stairs = dilate(jump, int(round(0.5 / RES)))


def cell(x, y):
  i, j = int((x - X0) / RES), int((y - Y0) / RES)
  if 0 <= i < H.shape[0] and 0 <= j < H.shape[1]:
    return i, j
  return None


def run(kind, seed):
  rng = np.random.default_rng(seed)
  data = mujoco.MjData(model)
  if kind == "down":
    pos, yaw, segs = DOWN_SPAWN, DOWN_YAW_DEG, [(8.0, 0.4, 0)]
  else:
    pos, yaw, segs = (-10.34, 5.75, 0.83), 270.0, [(10.0, 0.5, 0)]
  pos = (pos[0] + rng.uniform(-0.05, 0.05), pos[1] + rng.uniform(-0.05, 0.05), pos[2])
  reset_to_spawn(model, data, b, settle=True, pos=pos, yaw_deg=yaw + rng.uniform(-3, 3))
  rt = PieRuntime(model, b, data, PieOnnxPolicy(onnx, provider="cpu"), None, ConstantCmd(vx=0.0, segments=segs))
  lat = int(os.environ.get("DEPTH_LATENCY", "0"))
  if lat > 0 and rt.uses_depth:
    render = rt.camera.capture
    ring = deque(maxlen=lat + 1)
    rt.camera.capture = lambda _data: ring[0]
  root = b.root_qpos_adr
  base_body = model.jnt_bodyid[np.flatnonzero(model.jnt_qposadr == root)[0]]
  robot_geoms = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == base_body}
  feet = [g for g in robot_geoms if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
  n_st = n_edge = 0
  fell = False
  z0 = float(data.qpos[root + 2])
  for _ in range(int(round(segs[0][0] / DT))):
    if lat > 0 and rt.uses_depth:
      ring.append(render(data))
    rt.tick()
    fell |= rt.is_fallen()
    on = set()
    for con in data.contact[:data.ncon]:
      for g, o in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
        if g in feet and o not in robot_geoms:
          on.add(g)
    for g in on:
      c = cell(*data.geom_xpos[g][:2])
      if c is None or not stairs[c]:
        continue
      n_st += 1
      n_edge += bool(edge[c])
  dz = float(data.qpos[root + 2]) - z0
  rt.close()
  return n_st, n_edge, fell, dz


print(f"{robot} {onnx.stem} depth latency {os.environ.get('DEPTH_LATENCY', '0')} steps", flush=True)
for kind in ("up", "down"):
  tot_st = tot_edge = 0
  falls, dzs = 0, []
  for s in range(runs):
    n_st, n_edge, fell, dz = run(kind, s)
    tot_st, tot_edge, falls = tot_st + n_st, tot_edge + n_edge, falls + fell
    dzs.append(dz)
  share = 100 * tot_edge / max(tot_st, 1)
  print(f"{onnx.stem} {kind:4s}: feet on edges {share:5.1f}% of stair contacts ({tot_st} foot-steps), "
        f"falls {falls}/{runs}, dz {np.mean(dzs):+.2f} m", flush=True)
