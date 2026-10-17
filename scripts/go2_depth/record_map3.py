"""Record a scripted tour of the map3 stairwell with a teacher or student.

An "operator" autopilot drives the twist command along a route (L0 up, across
the landing, R1 up, turn around, R1 down, L0 down, then an omnidirectional
demo), holding each flight's heading and centerline like a person with a
joystick would. Rails are rendered semi-transparent so the chase camera can
see into the 0.9 m flights (rendering only; physics and the teacher scan are
those of play_map3).

  PYTHONPATH=. python scripts/go2_depth/record_map3.py \
    --checkpoint logs/rsl_rl/go2_depth_teacher/<run>/model_N.pt --out map3_tour.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_map3  # noqa: E402
import play_map3  # noqa: E402

SPAWN = "l0_bottom"
# (label, kind, args) in map3 coordinates; headings in degrees.
ROUTE = [
  ("stand", "pause", 1.5),
  ("L0 up the stairs", "follow", ((-10.34, 1.15), -90.0, 0.5)),
  ("turn on the landing", "turn", 0.0),
  ("cross to R1", "follow", ((-8.80, 1.15), 0.0, 0.4)),
  ("turn to R1", "turn", 90.0),
  ("R1 up the stairs", "follow", ((-8.80, 5.55), 90.0, 0.5)),
  ("stand on the landing", "pause", 1.5),
  ("turn around in place", "turn", -90.0),
  ("R1 down the stairs", "follow", ((-8.80, 1.15), -90.0, 0.4)),
  ("turn on the landing", "turn", 180.0),
  ("cross to L0", "follow", ((-10.34, 1.15), 180.0, 0.4)),
  ("turn to L0", "turn", 90.0),
  ("L0 down the stairs", "follow", ((-10.34, 5.60), 90.0, 0.4)),
  ("strafe left", "fixed", ((0.0, 0.3, 0.0), 1.5)),
  ("strafe right", "fixed", ((0.0, -0.3, 0.0), 1.5)),
  ("turn in place", "fixed", ((0.0, 0.0, 1.0), 3.0)),
  ("stand", "pause", 2.0),
]


def wrap(a: float) -> float:
  return math.atan2(math.sin(a), math.cos(a))


class Autopilot:
  def __init__(self, env, spawn_map3):
    self.env = env.unwrapped
    self.robot = self.env.scene["robot"]
    self.origin = self.env.scene.env_origins[0, :2].cpu().numpy()
    self.spawn = np.asarray(spawn_map3[:2])
    self.step = 0
    self.done_count = 0

  def world(self, p):
    return self.origin + (np.asarray(p) - self.spawn)

  def pose(self):
    pos = self.robot.data.root_link_pos_w[0, :2].cpu().numpy()
    return pos, float(self.robot.data.heading_w[0])

  def command(self, kind, args):
    """Return ((vx, vy, wz), finished)."""
    self.step += 1
    pos, yaw = self.pose()
    if kind == "pause":
      return (0.0, 0.0, 0.0), self.step * self.env.step_dt >= args
    if kind == "fixed":
      cmd, seconds = args
      return cmd, self.step * self.env.step_dt >= seconds
    if kind == "turn":
      err = wrap(math.radians(args) - yaw)
      self.done_count = self.done_count + 1 if abs(err) < 0.06 else 0
      return (0.0, 0.0, float(np.clip(2.0 * err, -1.0, 1.0))), self.done_count > 10
    target, heading, vx = args
    h = math.radians(heading)
    axis = np.array([math.cos(h), math.sin(h)])
    rel = self.world(target) - pos
    remaining = float(rel @ axis)
    lateral = float(-axis[1] * rel[0] + axis[0] * rel[1])  # Left of the robot is +.
    wz = float(np.clip(1.5 * wrap(h - yaw), -0.8, 0.8))
    vy = float(np.clip(1.0 * lateral, -0.3, 0.3))
    timeout = self.step * self.env.step_dt > 20.0
    return (vx if remaining > 0.1 else 0.0, vy, wz), remaining <= 0.1 or timeout

  def reset_segment(self):
    self.step = 0
    self.done_count = 0


def overlay(frame, lines):
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img)
  try:
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
  except OSError:
    font = ImageFont.load_default()
  y = 8
  for i, text in enumerate(lines):
    draw.text((10, y), text, fill=(255, 255, 255) if i else (255, 220, 80), font=font,
              stroke_width=2, stroke_fill=(0, 0, 0))
    y += 26
  return np.asarray(img)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--out", default="map3_tour.mp4")
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=540)
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.viewer.viewer_config import ViewerConfig

  ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  teacher = "student_state_dict" not in ck
  task = "Unitree-Go2-Depth-Teacher" if teacher else "Unitree-Go2-Depth-Student"
  spawn, yaw = play_map3.SPAWNS[SPAWN]
  cfg = play_map3.make_env_cfg(task, spawn, yaw, teacher)
  cfg.scene.terrain.terrain_generator.sub_terrains["map3"].rail_alpha = 0.18
  for sensor in cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      sensor.debug_vis = False
  cfg.commands["twist"].debug_vis = False
  v = cfg.viewer
  v.origin_type = ViewerConfig.OriginType.ASSET_BODY
  v.entity_name, v.body_name = "robot", "base_link"
  v.width, v.height = args.width, args.height
  v.distance, v.elevation = 2.2, -22.0
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  )
  obs = env.get_observations()
  kind = "teacher" if teacher else "student"
  model = eval_map3.load_policy(kind, args.checkpoint, TensorDict(obs, batch_size=[1]), env.unwrapped.device)
  term = env.unwrapped.command_manager.get_term("twist")
  pilot = Autopilot(env, spawn)
  renderer_cam = env.unwrapped._offline_renderer._cam
  cam_azimuth = None
  falls = 0
  writer = imageio.get_writer(args.out, fps=25, quality=7, macro_block_size=8)
  sim_t = 0.0
  for label, kind_, seg_args in ROUTE:
    pilot.reset_segment()
    print(f"[record] {label}", flush=True)
    finished = False
    while not finished:
      cmd, finished = pilot.command(kind_, seg_args)
      term.vel_command_b[0] = torch.tensor(cmd, device=env.unwrapped.device)
      with torch.inference_mode():
        obs, _, dones, _ = env.step(model(obs))
      if bool(dones[0]):
        falls += 1
        print(f"[record] FALL during '{label}'", flush=True)
      sim_t += env.unwrapped.step_dt
      if pilot.step % 2 == 0:
        # Chase camera behind and a bit to the side, smoothed.
        target = math.degrees(pilot.pose()[1]) + 180.0 + 25.0
        if cam_azimuth is None:
          cam_azimuth = target
        cam_azimuth += 0.08 * (((target - cam_azimuth + 180.0) % 360.0) - 180.0)
        renderer_cam.azimuth = cam_azimuth
        frame = env.unwrapped.render()
        z = float(env.unwrapped.scene["robot"].data.root_link_pos_w[0, 2]) - float(env.unwrapped.scene.env_origins[0, 2]) + spawn[2]
        writer.append_data(overlay(frame, [
          f"{kind} on map3: {label}",
          f"cmd vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
          f"t {sim_t:5.1f} s   height {z:4.2f} m   falls {falls}",
        ]))
  writer.close()
  print(f"[record] wrote {args.out} ({sim_t:.1f} s, falls {falls})", flush=True)


if __name__ == "__main__":
  main()
