"""Depth students against things in front of the camera, in tinynav-sim (robustness step 1).

obstacles.py ROBOT ONNX [SCENARIO...]  (run from the worktree's mujoco/ with ROBOT_ASSETS and
MUJOCO_GL=egl; no SCENARIO = all). The policy runs bare: no wall guard, no fallback.

Physical objects (mocap bodies on flat ground, placed along the robot's heading from its
depth camera when the event starts; they collide with the robot):
  box   0.5 deep x 0.8 wide x 0.6 tall      low   0.4 x 1.0 x 0.10 (a kerb)
  pole  cylinder r 0.15, 1.7 tall (a person), sweeping across 3 m at 1 m/s
Camera faults (the rendered frame is replaced; nothing physical):
  patch   the lower-middle 40% x 40% of the frame at 0.25 m (a hand, a branch)
  blind   every pixel a miss (lens covered, glare)        near  every pixel at 0.20 m
  frozen  the frame from the fault's start, held

Per scenario: fell; max |roll| / |pitch| in the event window (deg); foot touchdowns/s before
(1-2 s) and in the window (flailing); base travel in the window (stand) or forward speed
(walk / down); base rise (climbing onto the object); nearest camera-to-object distance.
"""
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, ".")
from sim.keyboard import ConstantCmd  # noqa: E402
from sim.plant import (  # noqa: E402
  DOWN_SPAWN, DOWN_YAW_DEG, _add_depth_camera_and_shell, _apply_plant_options, _compile,
  _load_robot_spec, build_map3_model, reset_to_spawn)
from sim.policy import PieOnnxPolicy  # noqa: E402
from sim.robots import get_robot  # noqa: E402
from sim.guard import WallGuard  # noqa: E402
from sim.runtime import PieRuntime  # noqa: E402

# env FALLBACK=<blind onnx>: the runtime's depth monitor + blind fallback; GUARD=1: the wall guard
FALLBACK = os.environ.get("FALLBACK")
GUARD = os.environ.get("GUARD") == "1"

DT = 0.02
OBSTACLES = {  # name: (geom type, size, centre height)
  "box": (mujoco.mjtGeom.mjGEOM_BOX, (0.25, 0.4, 0.3), 0.3),
  "low": (mujoco.mjtGeom.mjGEOM_BOX, (0.2, 0.5, 0.05), 0.05),
  "pole": (mujoco.mjtGeom.mjGEOM_CYLINDER, (0.15, 0.85, 0.0), 0.85),
}
HIDDEN = (50.0, 50.0)

# name: (terrain, command segments, event, event start, event end, window end)
#   event: ("obj", name, distance camera->front face, how) | ("cam", fault)
SCENARIOS = {
  "stand_clean": ("flat", [(9.0, 0, 0)], None, 2.0, 2.0, 9.0),
  "walk_clean": ("flat", [(9.0, 0.5, 0)], None, 2.0, 2.0, 9.0),
  "stand_box": ("flat", [(9.0, 0, 0)], ("obj", "box", 0.6, "fixed"), 2.0, 9.0, 9.0),
  "stand_pole": ("flat", [(9.0, 0, 0)], ("obj", "pole", 0.8, "sweep"), 2.0, 5.0, 7.0),
  "walk_box_go": ("flat", [(9.0, 0.5, 0)], ("obj", "box", 1.5, "fixed"), 2.0, 9.0, 9.0),
  **{f"walk_box_go_{d}_{v}": ("flat", [(9.0, v, 0)], ("obj", "box", d, "fixed"), 2.0, 9.0, 9.0)
     for d, v in ((1.2, 0.5), (1.8, 0.5), (2.1, 0.5), (1.5, 0.3), (1.5, 0.8))},
  "walk_box_stop": ("flat", [(2.5, 0.5, 0), (6.5, 0, 0)], ("obj", "box", 1.0, "fixed"), 2.0, 9.0, 9.0),
  "walk_low": ("flat", [(9.0, 0.5, 0)], ("obj", "low", 1.2, "fixed"), 2.0, 9.0, 9.0),
  "walk_pole": ("flat", [(9.0, 0.5, 0)], ("obj", "pole", 1.2, "sweep"), 2.0, 5.0, 7.0),
  "stand_patch": ("flat", [(9.0, 0, 0)], ("cam", "patch"), 2.0, 3.5, 6.0),
  "walk_patch": ("flat", [(9.0, 0.5, 0)], ("cam", "patch"), 2.0, 3.5, 6.0),
  "walk_blind": ("flat", [(9.0, 0.5, 0)], ("cam", "blind"), 2.0, 4.0, 6.0),
  "stand_near": ("flat", [(9.0, 0, 0)], ("cam", "near"), 2.0, 4.0, 6.0),
  "walk_near": ("flat", [(9.0, 0.5, 0)], ("cam", "near"), 2.0, 4.0, 6.0),
  "up_clean": ("up", [(10.0, 0.5, 0)], None, 1.5, 1.5, 10.0),
  "down_clean": ("down", [(8.0, 0.4, 0)], None, 2.5, 2.5, 7.0),
  "down_patch": ("down", [(8.0, 0.4, 0)], ("cam", "patch"), 2.5, 4.0, 7.0),
  "down_blind": ("down", [(8.0, 0.4, 0)], ("cam", "blind"), 2.5, 4.5, 7.0),
  "down_frozen": ("down", [(8.0, 0.4, 0)], ("cam", "frozen"), 2.5, 4.5, 7.0),
}


def build_flat_with_obstacles(robot: str):
  rb = get_robot(robot)
  scene = mujoco.MjSpec.from_string(
    "<mujoco model='obstacles'><worldbody>"
    "<geom name='ground' type='plane' pos='0 0 0' size='40 40 0.1' rgba='0.55 0.55 0.52 1'"
    " friction='0.6 0.005 0.0001'/></worldbody></mujoco>")
  scene.worldbody.add_light(name="sun", pos=(0.0, 0.0, 5.0), dir=(0.0, 0.0, -1.0),
                            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False)
  for name, (gtype, size, z) in OBSTACLES.items():
    body = scene.worldbody.add_body(name=f"obst_{name}", mocap=True, pos=(*HIDDEN, z))
    body.add_geom(name=f"obst_{name}_geom", type=gtype, size=size, rgba=(0.8, 0.4, 0.2, 1),
                  contype=1, conaffinity=1, friction=(0.6, 0.005, 0.0001))
  robot_spec = _load_robot_spec(rb)
  frame = scene.worldbody.add_frame(pos=(0.0, 0.0, 0.0))
  base = next(b for b in robot_spec.worldbody.bodies if b.name == "base_link")
  frame.attach_body(base, "")
  base_in_scene = next(b for b in scene.worldbody.bodies if b.name == "base_link")
  _add_depth_camera_and_shell(base_in_scene, rb)
  _apply_plant_options(scene)
  return _compile(scene, with_shell=True, robot=rb, scene_name="flat")


def rpy(q):
  w, x, y, z = q
  roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
  yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return roll, pitch, yaw


def fault_capture(capture, state):
  def wrapped(data):
    depth = capture(data)
    fault = state["fault"]
    if fault is None:
      state["last"] = depth
      return depth
    out = depth.copy()
    h, w = out.shape
    if fault == "patch":
      out[int(0.5 * h):int(0.9 * h), int(0.3 * w):int(0.7 * w)] = 0.25
    elif fault == "blind":
      out[:] = 0.0
    elif fault == "near":
      out[:] = 0.20
    elif fault == "frozen":
      out = state["frozen"] if state["frozen"] is not None else out
    return out
  return wrapped


def run(robot, onnx, name, models):
  terrain, segs, event, t0, t1, tw = SCENARIOS[name]
  model, b = models[terrain]
  data = mujoco.MjData(model)
  if terrain == "down":
    reset_to_spawn(model, data, b, settle=True, pos=DOWN_SPAWN, yaw_deg=DOWN_YAW_DEG)
  elif terrain == "up":   # the foot of map3's L0, as eval_map3's L0_stop_midway
    reset_to_spawn(model, data, b, settle=True, pos=(-10.34, 5.75, 0.83), yaw_deg=270.0)
  else:
    reset_to_spawn(model, data, b, settle=True)
  policy = PieOnnxPolicy(onnx, provider="cpu")
  holder = [None]
  src = ConstantCmd(vx=0.0, segments=list(segs))
  cmd = WallGuard(src, lambda: holder[0]) if GUARD else src
  fallback = PieOnnxPolicy(Path(FALLBACK), provider="cpu") if FALLBACK else None
  rt = PieRuntime(model, b, data, policy, None, cmd, fallback_policy=fallback)
  holder[0] = rt
  fb_ticks = 0
  state = {"fault": None, "last": None, "frozen": None}
  if rt.uses_depth:
    rt.camera.capture = fault_capture(rt.camera.capture, state)
  root = b.root_qpos_adr
  base_body = model.jnt_bodyid[np.flatnonzero(model.jnt_qposadr == root)[0]]
  robot_geoms = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == base_body}
  feet = [g for g in robot_geoms if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
  mocap = {n: model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obst_{n}")]
           for n in OBSTACLES} if terrain == "flat" else {}
  obj = None
  fell, tilt, touch_pre, touch_win, prev_on = False, [0.0, 0.0], 0, 0, None
  p_win0 = z0 = None
  z_max = -math.inf
  min_dist = math.inf
  vel_x = []
  steps = int(round(max(tw, sum(s[0] for s in segs)) / DT))
  for k in range(steps):
    t = (k + 1) * DT
    # events
    if event is not None and abs(t - t0) < 1e-6:
      if event[0] == "cam":
        state["fault"] = event[1]
        state["frozen"] = state["last"]
      else:
        _, oname, dist, how = event
        cam = data.cam_xpos[b.depth_camera_id].copy()
        _, _, yaw = rpy(data.qpos[root + 3:root + 7])
        hd = np.array([math.cos(yaw), math.sin(yaw)])
        lat = np.array([-hd[1], hd[0]])
        depth_half = OBSTACLES[oname][1][0]
        obj = {"name": oname, "how": how, "centre": cam[:2] + hd * (dist + depth_half),
               "hd": hd, "lat": lat, "dist": dist, "start": t}
    if event is not None and event[0] == "cam" and abs(t - t1) < 1e-6:
      state["fault"] = None
    if obj is not None:
      c = obj["centre"]
      if obj["how"] == "sweep":
        # across the robot's path, keeping its distance ahead of the camera
        s = -1.5 + 1.0 * (t - obj["start"])
        cam = data.cam_xpos[b.depth_camera_id]
        c = cam[:2] + obj["hd"] * (obj["dist"] + OBSTACLES["pole"][1][0]) + obj["lat"] * s
        c = c if s <= 1.5 else np.array(HIDDEN)
      data.mocap_pos[mocap[obj["name"]]][:2] = c
      data.mocap_pos[mocap[obj["name"]]][2] = OBSTACLES[obj["name"]][2]
    rt.tick()
    q = data.qpos[root + 3:root + 7]
    r, p, _ = rpy(q)
    fell |= rt.is_fallen()
    on = np.zeros(len(feet), bool)
    for con in data.contact[:data.ncon]:
      for g, o in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
        if g in feet and o not in robot_geoms:
          on[feet.index(g)] = True
    if prev_on is not None:
      new = int((on & ~prev_on).sum())
      if 1.0 < t <= 2.0:
        touch_pre += new
      if t0 < t <= tw:
        touch_win += new
    prev_on = on
    if abs(t - t0) < 1e-6:
      p_win0, z0 = rt.base_pos().copy(), float(data.qpos[root + 2])
    if t0 < t <= tw:
      tilt = [max(tilt[0], abs(math.degrees(r))), max(tilt[1], abs(math.degrees(p)))]
      z_max = max(z_max, float(data.qpos[root + 2]))
      fb_ticks += bool(getattr(rt, "fallback_active", False))
      _, _, yaw = rpy(q)
      v = data.qvel[model.jnt_dofadr[np.flatnonzero(model.jnt_qposadr == root)[0]]:][:2]
      vel_x.append(math.cos(yaw) * v[0] + math.sin(yaw) * v[1])
      if obj is not None and obj["how"] == "fixed":
        cam = data.cam_xpos[b.depth_camera_id]
        face = obj["centre"] - obj["hd"] * OBSTACLES[obj["name"]][1][0]
        min_dist = min(min_dist, float(np.dot(face - cam[:2], obj["hd"])))
  win = max(tw - t0, 1e-6)
  travel = float(np.linalg.norm(rt.base_pos()[:2] - p_win0[:2])) * 100 if p_win0 is not None else float("nan")
  rise = (z_max - z0) * 100 if z0 is not None else float("nan")
  moving = any(s[1] for s in segs)
  rt.close()
  speed = f"speed {np.mean(vel_x):5.2f}" if moving else f"travel {travel:5.1f} cm"
  dist = f" | nearest {min_dist:5.2f} m" if math.isfinite(min_dist) else ""
  return (f"{name:13s} | fell {str(fell):5s} | roll {tilt[0]:5.1f} pitch {tilt[1]:5.1f} | "
          f"steps/s {touch_pre / 1.0 / max(len(feet), 1):4.2f} -> {touch_win / win / max(len(feet), 1):4.2f} | "
          f"{speed} | rise {rise:+5.1f} cm{dist}" + (f" | fallback {fb_ticks * DT:3.1f} s" if FALLBACK else ""))


def main():
  robot, onnx = sys.argv[1], Path(sys.argv[2])
  names = sys.argv[3:] or list(SCENARIOS)
  models = {}
  if any(SCENARIOS[n][0] == "flat" for n in names):
    models["flat"] = build_flat_with_obstacles(robot)
  if any(SCENARIOS[n][0] in ("down", "up") for n in names):
    models["down"] = models["up"] = build_map3_model(robot)
  print(f"{robot} {onnx.stem}", flush=True)
  for n in names:
    print(run(robot, onnx, n, models), flush=True)


if __name__ == "__main__":
  main()
