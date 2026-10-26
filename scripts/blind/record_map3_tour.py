"""Record a scripted tour of the map3 stairwell with a blind policy to MP4.

An operator autopilot drives the command along a fixed route: stand, L0 up, turn on
the landing and cross to R1, R1 up to y = 5.0 m, stand, turn around in place, R1
down, cross the landing back to L0, L0 down, stand. While walking it holds the
leg's heading and steers back to the leg's centerline like a person with a joystick
(the gains of scripts/go2_blind/eval_map3.py's operator); turns are in place. map3
and the play config come from eval_map3.py (no pushes, no observation noise, one
robot). The chase camera follows behind and above the robot; for the video only,
rails are semi-transparent and everything overhead (upper flights, landings,
floors) is see-through. A termination ends the tour: the last frame is held with
its reason.

Modelled on the go2-depth branch's scripts/go2_depth/record_map3.py.

  python scripts/blind/record_map3_tour.py --task Unitree-As2-Blind \
    logs/rsl_rl/as2_blind/<run>/model_N.pt --out map3_tour.mp4 --title "As2 model_N"
"""

import argparse
import math
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "go2_blind"))
import eval_map3  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

# map3 spot on the landing below L0, facing up the flight.
SPAWN, SPAWN_YAW = (-10.34, 5.75, 0.431), -90.0
# R1's last step ends at y = 5.12, where its top landing starts; the well-side wall
# beside R1 ends at y = 4.98 and the other wall runs on past the landing. The flight
# is 0.91 m wide, so a turn in place short of the landing hits a wall.
R1_TOP_Y = 5.0


def make_route(r1_top_y: float, r1_turn_x: float | None = None):
  """(label, kind, args) in map3 coordinates, headings in degrees:
    pause  seconds
    turn   heading                  (turn in place)
    follow ((x, y), heading, vx)    (walk along the line through (x, y) at heading)
    strafe ((x, y), heading)        (step sideways to the line through (x, y), holding heading)
  With `r1_turn_x` the robot steps sideways on R1's top landing to that x before
  turning around and back after: R1's centerline is 0.46 m from the landing's east
  wall (x = -8.34), less than half the length of B2W, whose turn there hit it.
  """
  turn_around = [("原地掉头", "turn", -90.0)]
  if r1_turn_x is not None:
    turn_around = [
      ("平台上横移", "strafe", ((r1_turn_x, r1_top_y), 90.0)),
      ("原地掉头", "turn", -90.0),
      ("横移回 R1", "strafe", ((-8.80, r1_top_y), -90.0)),
    ]
  return [
    ("站立", "pause", 1.5),
    ("L0 上楼", "follow", ((-10.34, 1.15), -90.0, 0.5)),
    ("平台上转身", "turn", 0.0),
    ("横穿平台到 R1", "follow", ((-8.80, 1.15), 0.0, 0.4)),
    ("转向 R1", "turn", 90.0),
    (f"R1 上楼到 y = {r1_top_y:g} m", "follow", ((-8.80, r1_top_y), 90.0, 0.5)),
    ("站立", "pause", 1.5),
    *turn_around,
    ("R1 下楼", "follow", ((-8.80, 1.15), -90.0, 0.4)),
    ("平台上转身", "turn", 180.0),
    ("横穿平台到 L0", "follow", ((-10.34, 1.15), 180.0, 0.4)),
    ("转向 L0", "turn", 90.0),
    ("L0 下楼", "follow", ((-10.34, 5.60), 90.0, 0.4)),
    ("站立", "pause", 2.0),
  ]


FOLLOW_TIMEOUT, TURN_TIMEOUT = 30.0, 15.0  # s
FPS = 25
RAIL_RGB = (0.85, 0.3, 0.3)  # eval_map3.Map3TerrainCfg's rail color
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def wrap(a: float) -> float:
  return math.atan2(math.sin(a), math.cos(a))


class Autopilot:
  """Joystick commands for one route step at a time, from the robot's map3 pose."""

  def __init__(self, env, turn_rate: float):
    self.env = env
    self.robot = env.scene["robot"]
    self.origin = env.scene.env_origins[0, :2].cpu().numpy()
    self.turn_rate = turn_rate
    self.reset()

  def reset(self):
    self.steps = 0
    self.settled = 0

  def pose(self):
    """Base (x, y) in map3 coordinates and heading (rad)."""
    pos = self.robot.data.root_link_pos_w[0, :2].cpu().numpy() - self.origin + np.asarray(SPAWN[:2])
    return pos, float(self.robot.data.heading_w[0])

  def command(self, kind, args):
    """((vx, vy, wz), finished, timed_out) for the next control step."""
    self.steps += 1
    t = self.steps * self.env.step_dt
    pos, yaw = self.pose()
    if kind == "pause":
      return (0.0, 0.0, 0.0), t >= args, False
    if kind == "turn":
      err = wrap(math.radians(args) - yaw)
      self.settled = self.settled + 1 if abs(err) < 0.06 else 0
      wz = float(np.clip(2.0 * err, -self.turn_rate, self.turn_rate))
      timed_out = t > TURN_TIMEOUT
      return (0.0, 0.0, wz), self.settled > 10 or timed_out, timed_out
    if kind == "strafe":
      target, heading = args
      h = math.radians(heading)
      rel = np.asarray(target) - pos
      left = float(-math.sin(h) * rel[0] + math.cos(h) * rel[1])  # target offset to the robot's left
      # At least 0.12 m/s: commands under 0.1 m/s are zeroed (the deadzone).
      vy = float(np.clip(math.copysign(max(0.12, 1.5 * abs(left)), left), -0.3, 0.3))
      wz = float(np.clip(wrap(h - yaw), -0.5, 0.5))
      timed_out = t > FOLLOW_TIMEOUT
      done = abs(left) < 0.06
      return (0.0, 0.0 if done else vy, wz), done or timed_out, timed_out
    target, heading, vx = args
    h = math.radians(heading)
    axis = np.array([math.cos(h), math.sin(h)])
    rel = np.asarray(target) - pos
    remaining = float(rel @ axis)
    left = float(-axis[1] * rel[0] + axis[0] * rel[1])  # offset of the line to the robot's left
    vy = float(np.clip(left, -0.2, 0.2))
    wz = float(np.clip(wrap(h - yaw), -0.5, 0.5))
    timed_out = t > FOLLOW_TIMEOUT
    return (vx if remaining > 0.1 else 0.0, vy, wz), remaining <= 0.1 or timed_out, timed_out


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("checkpoint")
  p.add_argument("--task", default=eval_map3.TASK, help="Blind stairs task whose play config matches the policy.")
  p.add_argument("--out", default="map3_tour.mp4")
  p.add_argument("--title", default="")
  p.add_argument("--width", type=int, default=1280)
  p.add_argument("--height", type=int, default=720)
  # Behind and above the robot: high enough to look over the flight behind it when
  # descending (the steps rise ~0.6 m per m).
  p.add_argument("--cam-distance", type=float, default=2.6)
  p.add_argument("--cam-elevation", type=float, default=-40.0)
  p.add_argument("--cam-side", type=float, default=0.0, help="azimuth offset from straight behind (deg)")
  p.add_argument("--turn-rate", type=float, default=1.0, help="max |wz| of the in-place turns (rad/s)")
  p.add_argument("--r1-top-y", type=float, default=R1_TOP_Y, help="map3 y where the robot stops on R1 and turns around")
  p.add_argument("--r1-turn-x", type=float, default=None, help="step sideways to this map3 x on R1's top landing to turn around (B2W: -9.3)")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  route = make_route(args.r1_top_y, args.r1_turn_x)

  scenario = eval_map3.Scenario("tour", SPAWN, SPAWN_YAW, [(600.0, 0.0, 0.0, 0.0)], "track")
  cfg = eval_map3.make_env_cfg(scenario, eval_map3.load_map3_boxes(), 1, args.seed, args.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv, policy = eval_map3.load_policy(env, args.checkpoint, args.task)
  env.reset()
  obs = venv.get_observations()
  robot = env.scene["robot"]
  term = env.command_manager.get_term("twist")
  dt = env.step_dt

  # Video-only model edits (the physics runs on the warp model).
  model = env.sim.mj_model
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
  model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
  model.stat.extent = 6.0  # sharper shadows than the extent of the whole terrain tile
  # The terrain's straight-down lights plus the headlight wash upward faces and the
  # robot seen from above out to white.
  model.light_diffuse[:] = 0.3
  model.light_specular[:] = 0.1
  model.vis.headlight.diffuse[:] = 0.45
  model.vis.headlight.ambient[:] = 0.25
  terrain_geoms = np.nonzero(model.geom_bodyid == model.body("terrain").id)[0]
  model.geom_matid[terrain_geoms] = -1
  base_rgba = model.geom_rgba[terrain_geoms].copy()
  rail = np.all(np.abs(base_rgba[:, :3] - RAIL_RGB) < 1e-3, axis=1)
  base_rgba[~rail, :3] *= 0.8
  base_rgba[rail, 3] = 0.25
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  geom_bottom = data.geom_xpos[terrain_geoms, 2] - model.geom_size[terrain_geoms, 2]  # axis-aligned boxes
  renderer = mujoco.Renderer(model, height=args.height, width=args.width)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.distance, cam.elevation = args.cam_distance, args.cam_elevation
  cam_azimuth = None

  fonts = {k: ImageFont.truetype(FONT_B if k == "title" else FONT_R, s, index=2)
           for k, s in (("title", 30), ("text", 24), ("big", 30))}

  def overlay(img, lines, alert=None):
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size)
    d = ImageDraw.Draw(ov)
    d.rectangle((0, 0, 820, 52 + 34 * (len(lines) - 1) + 14), fill=(0, 0, 0, 150))
    d.text((16, 8), lines[0], font=fonts["title"], fill=(255, 220, 120))
    for i, (text, color) in enumerate(lines[1:]):
      d.text((16, 56 + 34 * i), text, font=fonts["text"], fill=color)
    if alert:
      w = args.width
      d.rectangle((w // 2 - 330, args.height // 2 - 40, w // 2 + 330, args.height // 2 + 30), fill=(0, 0, 0, 180))
      d.text((w // 2 - 310, args.height // 2 - 32), alert, font=fonts["big"], fill=(255, 110, 110))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))

  def render():
    nonlocal cam_azimuth
    sim = env.sim.data
    data.qpos[:] = sim.qpos[0].cpu().numpy()
    data.qvel[:] = sim.qvel[0].cpu().numpy()
    mujoco.mj_forward(model, data)
    base = robot.data.root_link_pos_w[0].cpu().numpy()
    rgba = base_rgba.copy()
    rgba[geom_bottom > base[2] + 0.45, 3] = 0.08  # overhead geometry
    model.geom_rgba[terrain_geoms] = rgba
    target = math.degrees(float(robot.data.heading_w[0])) + args.cam_side  # looking along the heading
    if cam_azimuth is None:
      cam_azimuth = target
    cam_azimuth += 0.08 * (((target - cam_azimuth + 180.0) % 360.0) - 180.0)
    cam.azimuth = cam_azimuth
    cam.lookat[:] = base + np.array([0.0, 0.0, 0.1])
    renderer.update_scene(data, camera=cam)
    return renderer.render()

  pilot = Autopilot(env, args.turn_rate)
  z0 = float(robot.data.root_link_pos_w[0, 2])
  writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
  t, step, falls, last = 0.0, 0, 0, None
  log = []
  for k, (label, kind, seg) in enumerate(route):
    pilot.reset()
    finished, timed_out = False, False
    while not finished:
      c, finished, timed_out = pilot.command(kind, seg)
      term.vel_command_b[0] = torch.tensor(c, device=env.device)
      term.time_left[:] = 1e9
      with torch.no_grad():
        obs, _, dones, extras = venv.step(policy(obs))
      t += dt
      step += 1
      if bool(dones[0]):
        tm = env.termination_manager
        why = [n for n in tm.active_terms if bool(tm.get_term(n)[0])]
        falls += 1
        log.append(f"{k + 1}/{len(route)} {label}: FALL at t={t:.1f} s ({', '.join(why)})")
        print(log[-1], flush=True)
        if last is not None:
          img = overlay(*last, alert=f"摔倒，巡游结束（{', '.join(why)}）")
          for _ in range(2 * FPS):
            writer.append_data(img)
        break
      if step % 2:
        continue
      v = robot.data.root_link_lin_vel_b[0]
      act = (float(v[0]), float(v[1]), float(robot.data.root_link_ang_vel_b[0, 2]))
      dz = float(robot.data.root_link_pos_w[0, 2]) - z0
      lines = [
        f"{args.title}  map3 巡游  {k + 1}/{len(route)} {label}",
        (f"指令  vx {c[0]:+.2f}  vy {c[1]:+.2f}  wz {c[2]:+.2f}", (150, 200, 255)),
        (f"实际  vx {act[0]:+.2f}  vy {act[1]:+.2f}  wz {act[2]:+.2f}", (140, 255, 220)),
        (f"相对起点高度 {dz:+.2f} m    t = {t:5.1f} s    摔倒 {falls}", (255, 230, 120)),
      ]
      img = render()
      last = (img, lines)
      writer.append_data(overlay(img, lines))
    if falls:
      break
    pos, yaw = pilot.pose()
    dz = float(robot.data.root_link_pos_w[0, 2]) - z0
    log.append(
      f"{k + 1}/{len(route)} {label}: {'TIMEOUT' if timed_out else 'done'} at t={t:.1f} s, "
      f"map3 ({pos[0]:.2f}, {pos[1]:.2f}), heading {math.degrees(yaw):.0f} deg, height {dz:+.2f} m"
    )
    print(log[-1], flush=True)
  writer.close()
  renderer.close()
  print(f"wrote {args.out}: {t:.1f} s of sim time, falls {falls}")
  env.close()


if __name__ == "__main__":
  main()
