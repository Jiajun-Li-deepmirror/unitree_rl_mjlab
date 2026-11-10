"""Record a scripted tour of the map3 stairwell with a teacher or student (any --robot).

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


def soften_lights(mj_model) -> None:
  """Tone down the video lighting (render-only: the CPU model drives the
  offline renderer; physics and the depth camera run on the warp model).

  The terrain's two straight-down directional lights (diffuse 0.7 each) plus
  the headlight (0.6 diffuse, 0.3 ambient) blow upward faces, and the robot
  seen from above, out to white.
  """
  mj_model.light_diffuse[:] = 0.3
  mj_model.light_specular[:] = 0.1
  mj_model.vis.headlight.diffuse[:] = 0.45
  mj_model.vis.headlight.ambient[:] = 0.25


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
  parser.add_argument("--robot", choices=tuple(eval_map3.ROBOT_TASK_NAMES), default="go2")
  parser.add_argument("--out", default="map3_tour.mp4")
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=540)
  parser.add_argument(
    "--solid-gaps", action="store_true",
    help="Fill the open well and slot beside the flights, landings kept open.",
  )
  # Chase camera behind and above the robot: high enough to look over the
  # flight behind it when descending (steps rise ~0.6 m per m).
  parser.add_argument("--cam-distance", type=float, default=2.6)
  parser.add_argument("--cam-elevation", type=float, default=-40.0)
  parser.add_argument("--cam-side", type=float, default=0.0, help="Azimuth offset from straight behind (deg).")
  parser.add_argument("--seed", type=int, default=None, help="Env seed (reset randomization: delays, ...).")
  parser.add_argument("--l0-bottom-y", type=float, default=5.60,
                      help="map3 y where L0's descent ends before the strafes (B2 2026-11-09: at 5.60 its rear legs were "
                      "still on the steps and the strafe left brushed the rail; 5.85)")
  parser.add_argument("--r1-top-y", type=float, default=None, help="map3 y where R1's climb ends (B2: 5.6)")
  parser.add_argument("--payload", default=None,
                      help="kg,x,y,z: a fixed point mass on the base (the payload acceptance's, e.g. Go2W 12,0.15,0,0.20)")
  parser.add_argument("--contact-goes-on", action="store_true",
                      help="count base / thigh touches (illegal_contact) instead of ending the tour (as the blind tour)")
  parser.add_argument("--r1-turn-x", type=float, default=None,
                      help="step sideways to this map3 x on R1's top landing before turning around (B2: -9.3)")
  args = parser.parse_args()
  for i, (label, kind_, seg) in enumerate(ROUTE):
    if label == "L0 down the stairs":
      (x, _y), hd, sp = seg
      ROUTE[i] = (label, kind_, ((x, args.l0_bottom_y), hd, sp))
    if label == "R1 up the stairs" and args.r1_top_y is not None:
      (x, _y), hd, sp = seg
      ROUTE[i] = (label, kind_, ((x, args.r1_top_y), hd, sp))
  if args.r1_turn_x is not None:
    # B2 (2026-11-09): turning in place at R1's top its body brushed the well-side rail (as the blind tour's
    # --r1-turn-x): step sideways onto the landing first, turn, step back onto R1's line, then descend.
    dx = args.r1_turn_x - (-8.80)
    secs = abs(dx) / 0.3
    j = next(k for k, r in enumerate(ROUTE) if r[0] == "turn around in place")
    ROUTE[j:j + 1] = [
      ("step onto the landing", "fixed", ((0.0, -0.3 if dx > 0 else 0.3, 0.0), secs)),   # facing +y, left is -x
      ROUTE[j],
      ("step back onto R1", "fixed", ((0.0, 0.3 if dx < 0 else -0.3, 0.0), secs)),       # facing -y, left is +x
    ]

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.viewer.viewer_config import ViewerConfig

  ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  teacher = "student_state_dict" not in ck
  robot_name = eval_map3.ROBOT_TASK_NAMES[args.robot]
  task = f"Unitree-{robot_name}-Depth-{'Teacher' if teacher else 'Student'}"
  spawn, yaw = play_map3.SPAWNS[SPAWN]
  cfg = play_map3.make_env_cfg(task, spawn, yaw, teacher)
  if args.seed is not None:
    cfg.seed = args.seed
  if args.payload or "base_payload" in cfg.events:
    # 2026-11-09 (the user: depth acceptance videos with and without the payload): as stop_then_move.py, a fixed
    # point mass on the base at every reset (0 kg without --payload: no payload randomization in the plain tour),
    # the base CoM randomization off
    import src.tasks.velocity.mdp as mdp
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    pm, px, py, pz = (float(v) for v in (args.payload or "0,0,0,0").split(","))
    cfg.events["base_payload"] = EventTermCfg(
      mode="reset", func=mdp.base_payload_point,
      params={"asset_cfg": SceneEntityCfg("robot"), "light_range": (pm, pm), "heavy_max": 0.0, "heavy_share": 0.0,
              "pos_ranges": ((px, px), (py, py), (pz, pz)), "base_com_range": 0.0, "ramp_steps": 0},
    )
    cfg.events.pop("base_com", None)
  map3 = cfg.scene.terrain.terrain_generator.sub_terrains["map3"]
  map3.rail_alpha = 0.18
  if args.solid_gaps:
    map3.boxes = map3.boxes + eval_map3.solid_gap_boxes(keep_landings=True)
  for sensor in cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      sensor.debug_vis = False
  cfg.commands["twist"].debug_vis = False
  v = cfg.viewer
  v.origin_type = ViewerConfig.OriginType.ASSET_BODY
  v.entity_name, v.body_name = "robot", "base_link"
  v.width, v.height = args.width, args.height
  v.distance, v.elevation = args.cam_distance, args.cam_elevation
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  )
  soften_lights(env.unwrapped.sim.mj_model)
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
  last = None
  terminations = env.unwrapped.termination_manager
  touches, touching, ic = 0, False, None
  if args.contact_goes_on and "illegal_contact" in terminations.active_terms:
    # 2026-11-09 (B2 depth student): the term stops ending the episode; its contacts are counted instead
    i = terminations._term_names.index("illegal_contact")
    ic = terminations._term_cfgs[i]
    del terminations._term_names[i], terminations._term_cfgs[i]
  for label, kind_, seg_args in ROUTE:
    if falls:
      break
    pilot.reset_segment()
    print(f"[record] {label}", flush=True)
    finished = False
    while not finished:
      cmd, finished = pilot.command(kind_, seg_args)
      term.vel_command_b[0] = torch.tensor(cmd, device=env.unwrapped.device)
      with torch.inference_mode():
        obs, _, dones, _ = env.step(model(obs))
      if bool(dones[0]):
        # The env has already reset the robot to the spawn, so the rest of the
        # route would be meaningless: hold the last frame, marked, and stop.
        falls += 1
        why = [t for t in terminations.active_terms if bool(terminations.get_term(t)[0])]
        print(f"[record] FALL during '{label}' ({', '.join(why)})", flush=True)
        if last is not None:
          frame, lines = last
          marked = overlay(frame, lines[:2] + [f"t {sim_t:5.1f} s   FALL ({', '.join(why)})"])
          for _ in range(40):
            writer.append_data(marked)
        break
      sim_t += env.unwrapped.step_dt
      if ic is not None:
        now = bool(ic.func(env.unwrapped, **ic.params)[0])
        if now and not touching:
          touches += 1
          print(f"[record] touch {touches} during '{label}' at t {sim_t:.1f} s", flush=True)
        touching = now
      if pilot.step % 2 == 0:
        # Chase camera behind (and --cam-side to the side), smoothed.
        target = math.degrees(pilot.pose()[1]) + 180.0 + args.cam_side
        if cam_azimuth is None:
          cam_azimuth = target
        cam_azimuth += 0.08 * (((target - cam_azimuth + 180.0) % 360.0) - 180.0)
        renderer_cam.azimuth = cam_azimuth
        frame = env.unwrapped.render()
        z = float(env.unwrapped.scene["robot"].data.root_link_pos_w[0, 2]) - float(env.unwrapped.scene.env_origins[0, 2]) + spawn[2]
        lines = [
          f"{robot_name} {kind} on map3{' (gaps filled)' if args.solid_gaps else ''}"
          f"{(', payload ' + args.payload.split(',')[0] + ' kg') if args.payload else ', no payload'}: {label}",
          f"cmd vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
          f"t {sim_t:5.1f} s   height {z:4.2f} m   falls {falls}" + (f"   body / thigh touches {touches}" if ic is not None else ""),
        ]
        last = (frame, lines)
        writer.append_data(overlay(frame, lines))
  writer.close()
  print(f"[record] wrote {args.out} ({sim_t:.1f} s, falls {falls}" + (f", body / thigh touches {touches}" if ic is not None else "") + ")", flush=True)


if __name__ == "__main__":
  main()
