"""Record a scripted joystick sequence of a blind policy on flat ground to MP4.

Works for the legged and the wheeled blind tasks. One robot on one 60 m flat tile
(play config: no pushes, no observation noise) follows a fixed list of commands, in
real time (no slow motion). Two views side by side: a tracking 3/4 view and a low
close-up of the feet / wheels that turns with the robot. The bar below shows the
segment, the command and the measured base velocity with its steady-state mean
(after the first second of the segment), per foot / wheel whether it touches the
ground, its current clearance and the peak height of its last lift, and at the zero
command the time to stop and the drift since 2 s after the release (wheeled robots:
also the parking brake state).

  python scripts/blind/record_flat.py --task Unitree-As2-Blind-Flat --foot-radius 0.028 \
    logs/rsl_rl/as2_blind/<run>/model_N.pt --out flat.mp4 --title "As2 model_N"

--foot-radius (alias --wheel-radius) is the radius of the foot sphere / wheel, so
clearance is measured from the bottom of the foot. Defaults to the size of the
robot's foot collision geom.
"""

import argparse
import math
import re
from dataclasses import asdict
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

# (label, (vx, vy, wz), seconds)
SEGMENTS = [
  ("站立（零指令）", (0.0, 0.0, 0.0), 4.0),
  ("前进 1.0 m/s", (1.0, 0.0, 0.0), 3.0),
  ("前进 1.0 + 小幅左转 0.15", (1.0, 0.0, 0.15), 4.0),
  ("前进 1.0 + 小幅右转 -0.15", (1.0, 0.0, -0.15), 4.0),
  ("高速 1.5 + 左转 0.2", (1.5, 0.0, 0.2), 3.0),
  ("高速 1.5 + 右转 -0.2", (1.5, 0.0, -0.2), 3.0),
  ("松杆：从 1.5 m/s 停下（零指令）", (0.0, 0.0, 0.0), 6.0),
  ("慢速前进 0.2 m/s", (0.2, 0.0, 0.0), 4.0),
  ("左移 0.6 m/s", (0.0, 0.6, 0.0), 3.0),
  ("右移 -0.6 m/s", (0.0, -0.6, 0.0), 3.0),
  ("斜向 vx 0.6 + vy 0.4", (0.6, 0.4, 0.0), 3.0),
  ("原地左转 1.5 rad/s", (0.0, 0.0, 1.5), 3.0),
  ("原地右转 -1.5 rad/s", (0.0, 0.0, -1.5), 3.0),
  ("原地慢转 0.3 rad/s", (0.0, 0.0, 0.3), 4.0),
  ("后退 1.0 m/s", (-1.0, 0.0, 0.0), 3.0),
  ("松杆：从后退 1.0 m/s 停下（零指令）", (0.0, 0.0, 0.0), 6.0),
]
FPS = 25
VIEW_W, VIEW_H, BAR_H = 640, 480, 216
STEADY_AFTER = 1.0  # s into a segment before the steady-state mean starts
CREEP_AFTER = 2.0  # s after the zero command starts before drift is counted (as eval_standing.py)
TRAIL_DT, TRAIL_N = 0.2, 25  # base trail: a point every TRAIL_DT s of sim time, last TRAIL_N points
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--task", required=True, help="Blind flat task whose play config matches the policy.")
p.add_argument("checkpoint")
p.add_argument("--foot-radius", "--wheel-radius", dest="foot_radius", type=float, default=None)
p.add_argument("--out", default="flat.mp4")
p.add_argument("--title", default="")
p.add_argument("--frames-only", default=None, help="write a few PNG frames here instead")
p.add_argument("--distance", type=float, default=1.8)
p.add_argument("--lookat-z", type=float, default=0.22, help="wide view target height (m)")
args = p.parse_args()

cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = 1
cfg.episode_length_s = 1000.0
# One 60 m flat tile with the robot in its middle: the sequence drives ~20 m, and
# on the play grid (5 x 10 tiles of 8 m) a robot spawned near the edge drove off it.
gen = cfg.scene.terrain.terrain_generator
gen.size = (60.0, 60.0)
gen.num_rows = gen.num_cols = 1
agent_cfg = load_rl_cfg(args.task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = load_runner_cls(args.task)(venv, asdict(agent_cfg), device="cuda:0")
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
dt = env.step_dt
robot = env.scene["robot"]
feet = env.scene["feet_ground_contact"]
cmd = env.command_manager.get_term("twist")
model = env.sim.mj_model

# Foot order of the contact sensor (its geoms in model order), so that each contact
# flag is paired with the site of the same foot.
contact_geoms = [s.primary_name for s in feet._slots if s.field_name == "found"]
legs = [re.search(r"(FL|FR|RL|RR)", name) for name in contact_geoms]
if not all(legs):
  raise RuntimeError(f"Cannot tell the legs of the contact geoms {contact_geoms}")
site_ids, site_names = robot.find_sites(tuple(m.group(1) for m in legs), preserve_order=True)
if args.foot_radius is None:
  # Compiled geom names carry the entity prefix ("robot/") for some robots.
  names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(model.ngeom)}
  gid = names.get(contact_geoms[0], names.get(f"robot/{contact_geoms[0]}"))
  if gid is None:
    raise RuntimeError(f"No geom {contact_geoms[0]}; pass --foot-radius")
  args.foot_radius = float(model.geom_size[gid, 0])
r = args.foot_radius
print(f"feet {site_names} (contact geoms {contact_geoms}), radius {r:.3f} m")

wheeled = "wheel_vel" in env.action_manager.active_terms
wheel_ids, brake_speed = None, 0.0
if wheeled:
  wheel_term = env.action_manager.get_term("wheel_vel")
  wheel_ids = wheel_term._target_ids
  brake_speed = float(getattr(wheel_term.cfg, "brake_speed", 0.0))
  print(f"wheel brake speed {brake_speed:.3f} rad/s ({brake_speed * r:.3f} m/s at the rim)")
DRIFT = "溜车" if wheeled else "漂移"

model.stat.extent = 3.0  # local to this process: sharper shadows, sane clipping
model.vis.global_.offwidth = VIEW_W
model.vis.global_.offheight = VIEW_H
free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
robot_root = model.body_rootid[model.jnt_bodyid[free[0]]]
floor = np.where(model.body_rootid[model.geom_bodyid] != robot_root)[0]
model.geom_matid[floor] = -1
model.geom_rgba[floor] = (0.80, 0.80, 0.78, 1.0)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=VIEW_H, width=VIEW_W)


def floor_top(x, y):
  """Height of the highest floor surface at (x, y): planes and axis-aligned boxes.

  The marks go on this surface; a ground height from the feet sits a few mm low
  (soft contact), which hid the marks inside the floor in most frames.
  """
  top = -np.inf
  for g in floor:
    px, py, pz = data.geom_xpos[g]
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
      top = max(top, pz)
    elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
      sx, sy, sz = model.geom_size[g]
      if abs(x - px) <= sx and abs(y - py) <= sy:
        top = max(top, pz + sz)
  return float(top)


trail = []
font = ImageFont.truetype(FONT, 22, index=2)
font_big = ImageFont.truetype(FONT, 26, index=2)


def camera(lookat, distance, azimuth, elevation):
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = lookat
  cam.distance = distance
  cam.azimuth = azimuth
  cam.elevation = elevation
  return cam


def draw_bar(state):
  img = Image.new("RGB", (2 * VIEW_W, BAR_H), (24, 24, 28))
  d = ImageDraw.Draw(img)
  d.text((16, 6), state["title"], font=font_big, fill=(255, 220, 120))
  c, v = state["cmd"], state["vel"]
  d.text(
    (16, 44),
    f"指令  vx {c[0]:+.2f}  vy {c[1]:+.2f}  wz {c[2]:+.2f}      "
    f"实际  vx {v[0]:+.2f}  vy {v[1]:+.2f}  wz {v[2]:+.2f}",
    font=font,
    fill=(230, 230, 230),
  )
  rows = zip(site_names, state["contact"], state["clear"], state["last"])
  for i, (name, on, clr, last) in enumerate(rows):
    col = (120, 220, 120) if on else (255, 120, 90)
    tag = "着地" if on else "离地"
    d.text(
      (16 + (i % 2) * VIEW_W, 82 + (i // 2) * 32),
      f"{name}  {tag}   当前离地 {clr * 100:4.1f} cm   上次抬起 {last * 100:4.1f} cm",
      font=font,
      fill=col,
    )
  d.text((16, 150), state["line1"], font=font, fill=state["line1_color"])
  d.text((16, 182), state["line2"], font=font, fill=(200, 200, 200))
  d.text((2 * VIEW_W - 16 - 8 * 22, 6), f"t = {state['t']:5.1f} s", font=font, fill=(160, 160, 160))
  d.text((2 * VIEW_W - 16 - 15 * 22, 44), "地面方格 0.5 m，橙点间隔 0.2 s", font=font, fill=(160, 160, 160))
  return img


def add_geom(scene, kind, size, pos, rgba):
  if scene.ngeom >= scene.maxgeom:
    return
  mujoco.mjv_initGeom(
    scene.geoms[scene.ngeom], kind, np.array(size, dtype=np.float64),
    np.array(pos, dtype=np.float64), np.eye(3).flatten(), np.array(rgba, dtype=np.float32),
  )
  scene.ngeom += 1


def add_floor_marks(scene, center, z, cell=0.5, half=5.0):
  """Dark checker tiles fixed in the world (0.5 m) and the base trail."""
  i0 = math.floor((center[0] - half) / cell)
  j0 = math.floor((center[1] - half) / cell)
  n = int(2 * half / cell) + 1
  for i in range(i0, i0 + n):
    for j in range(j0, j0 + n):
      if (i + j) % 2 == 0:
        add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, [cell / 2, cell / 2, 0.0005],
                 [(i + 0.5) * cell, (j + 0.5) * cell, z + 0.0006], (0.16, 0.17, 0.21, 1.0))
  for k, (x, y) in enumerate(trail):
    a = 0.35 + 0.65 * (k + 1) / len(trail)
    add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.018, 0, 0], [x, y, z + 0.018], (1.0, 0.55, 0.1, a))


def frame(state):
  base = state["base"]
  yaw = state["yaw"]
  wide = camera((base[0], base[1], args.lookat_z), args.distance, 135.0, -15.0)
  close = camera(
    (base[0], base[1], args.lookat_z * 0.36), 0.75 * args.distance, math.degrees(yaw) - 125.0, -6.0
  )
  views = []
  for cam in (wide, close):
    renderer.update_scene(data, camera=cam)
    add_floor_marks(renderer.scene, base, floor_top(base[0], base[1]))
    views.append(renderer.render())
  top = np.concatenate(views, axis=1)
  bar = np.asarray(draw_bar(state))
  return np.concatenate([top, bar], axis=0)


env.reset()
obs = venv.get_observations()
ground = robot.data.site_pos_w[0, site_ids, 2].clone() - r
peak = ground.clone() + r
last_lift = torch.zeros(len(site_ids), device=env.device)
prev = feet.data.found[0] > 0
writer = None
if args.frames_only is None:
  writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
else:
  Path(args.frames_only).mkdir(parents=True, exist_ok=True)
t = 0.0
step = 0
saved = set()
summary = []
for si, (label, c, seconds) in enumerate(SEGMENTS):
  C = torch.tensor([c], device=env.device, dtype=torch.float32)
  n = int(round(seconds / dt))
  title = f"{args.title}  {si + 1}/{len(SEGMENTS)}  {label}"
  zero = all(x == 0.0 for x in c)
  steady = []  # base velocity samples after STEADY_AFTER s
  stop_k = None  # zero command: base speed < 0.05 m/s from this step on
  creep_pos = None  # zero command: base position CREEP_AFTER s after the zero command started
  pos_release = robot.data.root_link_pos_w[0, :2].clone()
  brake_steps = 0
  for k in range(n):
    cmd.vel_command_b[:] = C
    cmd.time_left[:] = 1e9
    with torch.no_grad():
      obs, _, dones, _ = venv.step(policy(obs))
    if bool(dones[0]):
      print(f"reset at t={t:.2f} s ({label})")
    t += dt
    step += 1
    vel_now = robot.data.root_link_lin_vel_b[0, :2].tolist() + [robot.data.root_link_ang_vel_b[0, 2].item()]
    if (k + 1) * dt > STEADY_AFTER:
      steady.append(vel_now)
    pos = robot.data.root_link_pos_w[0, :2]
    braking = False
    if zero:
      if wheeled:
        braking = brake_speed > 0 and bool((robot.data.joint_vel[0, wheel_ids].abs() < brake_speed).all())
        brake_steps += braking
      if math.hypot(*robot.data.root_link_lin_vel_w[0, :2].tolist()) < 0.05:
        if stop_k is None:
          stop_k = k
      else:
        stop_k = None
      if creep_pos is None and (k + 1) * dt >= CREEP_AFTER:
        creep_pos = pos.clone()
    if zero:
      head = "零指令：还在减速" if stop_k is None else f"零指令：{(stop_k + 1) * dt:4.2f} s 停住（速度 < 0.05 m/s）"
      if creep_pos is None:
        tail = f"离松杆处 {torch.norm(pos - pos_release).item() * 100:5.1f} cm（{CREEP_AFTER:.0f} s 后开始计{DRIFT}）"
      else:
        tail = f"{CREEP_AFTER:.0f} s 后的{DRIFT} {torch.norm(pos - creep_pos).item() * 100:5.2f} cm"
      line1 = f"{head}   {tail}"
      line1_color = (120, 220, 120) if stop_k is not None else (255, 200, 90)
      if not wheeled:
        line2 = ""
      elif brake_speed > 0:
        line2 = f"轮子刹车：{'锁住' if braking else '松开'}（四个轮速都 < {brake_speed * r:.2f} m/s 且零指令时锁住）"
      else:
        line2 = "轮子刹车：本版本没有"
    else:
      m = np.mean(steady, axis=0) if steady else None
      parts = []
      for name, ci, j in (("vx", c[0], 0), ("vy", c[1], 1), ("wz", c[2], 2)):
        if ci != 0.0:
          parts.append(f"{name} {m[j]:+.2f}（{m[j] / ci * 100:3.0f}%）" if m is not None else f"{name} …")
      line1 = f"本段稳态均值（{STEADY_AFTER:.0f} s 后）：" + "   ".join(parts)
      line1_color = (150, 200, 255)
      line2 = ""
    if step % int(round(TRAIL_DT / dt)) == 0:
      trail.append(robot.data.root_link_pos_w[0, :2].tolist())
      del trail[:-TRAIL_N]
    on = feet.data.found[0] > 0
    zc = robot.data.site_pos_w[0, site_ids, 2]
    lift_off = prev & ~on
    peak = torch.where(lift_off, zc, torch.where(~on, torch.maximum(peak, zc), peak))
    touch = ~prev & on
    last_lift = torch.where(touch, peak - r - ground, last_lift)
    ground = torch.where(on, zc - r, ground)
    prev = on
    # 25 fps of sim time: every 2nd step (dt 0.02 s).
    if step % 2:
      continue
    qw, qx, qy, qz = robot.data.root_link_quat_w[0].tolist()
    state = {
      "title": title,
      "cmd": c,
      "vel": robot.data.root_link_lin_vel_b[0, :2].tolist() + [robot.data.root_link_ang_vel_b[0, 2].item()],
      "contact": on.tolist(),
      "clear": (zc - r - ground).clamp(min=0).tolist(),
      "last": last_lift.clamp(min=0).tolist(),
      "yaw": math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)),
      "base": robot.data.root_link_pos_w[0].tolist(),
      "ground_z": float(ground.mean()),
      "t": t,
      "line1": line1,
      "line1_color": line1_color,
      "line2": line2,
    }
    sim = env.sim.data
    data.qpos[:] = sim.qpos[0].cpu().numpy()
    data.qvel[:] = sim.qvel[0].cpu().numpy()
    mujoco.mj_forward(model, data)
    if step % 250 == 0:
      print(f"t={t:5.1f} foot ground {state['ground_z']:+.4f} floor top {floor_top(*state['base'][:2]):+.4f}")
    img = frame(state)
    if writer is not None:
      writer.append_data(img)
    elif k >= n // 2 and si not in saved:
      Image.fromarray(img).save(f"{args.frames_only}/seg{si + 1:02d}.png")
      saved.add(si)
  if zero:
    creep = torch.norm(robot.data.root_link_pos_w[0, :2] - creep_pos).item() * 100 if creep_pos is not None else float("nan")
    stop_s = (stop_k + 1) * dt if stop_k is not None else float("nan")
    brake = f", brake held {brake_steps * dt:.2f} s" if wheeled else ""
    summary.append(f"{label}: stop {stop_s:.2f} s, drift from {CREEP_AFTER:.0f} s to {seconds:.0f} s {creep:.2f} cm{brake}")
  else:
    m = np.mean(steady, axis=0)
    ratios = [f"{name} {m[j] / ci * 100:.0f}%" for name, ci, j in (("vx", c[0], 0), ("vy", c[1], 1), ("wz", c[2], 2)) if ci != 0.0]
    summary.append(f"{label}: steady mean {m[0]:+.2f} {m[1]:+.2f} {m[2]:+.2f} ({', '.join(ratios)})")
print("\n".join(summary))
if writer is not None:
  writer.close()
  print(f"wrote {args.out}: {t:.1f} s of sim time")
env.close()
