"""Interactive web play of a Go2 depth teacher or student in the map3 stairwell.

Opens the mjlab viser viewer (default :8080) and a gamepad page (default
:8090) that embeds the viewer and drives the twist command from a browser
gamepad (Gamepad API, "standard" mapping):

  left stick      vx (forward up to 1.0 m/s, back up to 0.5) and vy (0.5 m/s)
  right stick X   wz (1.2 rad/s)
  A / Cross       toggle gamepad control on/off
  B / Circle      stop (zero command)

The gamepad writes the viewer's own Twist sliders (and ticks "Enable"), so the
GUI always shows the command being sent. If the page stops sending for 0.5 s
the command is zeroed.

Teacher (height-scan) checkpoints get a scan that works in a multi-story
building: rays start 1.8 m above the robot (below the next floor) instead of
8 m, the real 19 m rails are hidden from the scan, and invisible, non-colliding
1.5 m rail strips follow each step/landing/floor along the rails, so the walls
read like the training walls. Student checkpoints see the unmodified map.

  PYTHONPATH=. python scripts/go2_depth/play_map3.py \
    --checkpoint logs/rsl_rl/go2_depth_teacher/<run>/model_N.pt --spawn l0_bottom
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_map3  # noqa: E402

from mjlab.terrains.terrain_generator import (  # noqa: E402
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

# map3 world position on the walking surface and heading (deg).
SPAWNS = {
  "l0_bottom": ((-10.34, 5.75, 0.431), -90.0),  # Bottom of the L0 flight, facing up it.
  "f1": ((-5.30, 3.45, 0.0), 180.0),  # F1 floor, facing the stairwell.
  "f2_landing": ((-10.34, 1.05, 2.659), 90.0),  # Top of L0, facing down it.
  "r1_bottom": ((-8.80, 1.05, 2.659), 90.0),  # Bottom of R1, facing up it.
  "r2b_top": ((-8.75, 5.70, 11.802), -90.0),  # tinynav DOWN_SPAWN.
}
VX_RANGE = (-0.5, 1.0)
VY_MAX = 0.5
WZ_MAX = 1.2
SCAN_Z_OFFSET_PLAY = 1.8
PROXY_HEIGHT = 1.5
_WALKABLE = ("step", "landing", "floor", "connector")


def rail_proxies(boxes) -> list[tuple[str, np.ndarray, np.ndarray]]:
  """Short rail strips on top of every walkable box running along a rail."""
  rails = [b for b in boxes if b[0].startswith("rail")]
  surfaces = [b for b in boxes if b[0].startswith(_WALKABLE)]
  out = []
  for rname, rsize, rpos in rails:
    rlo, rhi = rpos - rsize, rpos + rsize
    long_axis = 0 if rsize[0] >= rsize[1] else 1
    lat = 1 - long_axis
    for sname, ssize, spos in surfaces:
      slo, shi = spos - ssize, spos + ssize
      top = shi[2]
      if not (rlo[2] - 0.05 <= top <= rhi[2]):
        continue
      a0, a1 = max(rlo[long_axis], slo[long_axis]), min(rhi[long_axis], shi[long_axis])
      if a1 - a0 < 0.02:
        continue
      gap = max(slo[lat] - rhi[lat], rlo[lat] - shi[lat], 0.0)
      if gap > 0.12:
        continue
      lo, hi = rlo.copy(), rhi.copy()
      lo[long_axis], hi[long_axis] = a0, a1
      lo[2], hi[2] = top, min(top + PROXY_HEIGHT, rhi[2])
      if hi[2] - lo[2] < 0.05:
        continue
      out.append((f"proxy_{rname}_{sname}", (hi - lo) / 2, (hi + lo) / 2))
  return out


@dataclass(kw_only=True)
class Map3PlayTerrainCfg(SubTerrainCfg):
  spawn: tuple[float, float, float] = (0.0, 0.0, 0.0)
  size: tuple[float, float] = eval_map3.MAP3_TILE_SIZE
  boxes: list = field(default_factory=list)
  scan_proxies: bool = False
  rail_alpha: float = 1.0
  """Rail transparency, for rendering only (cameras can see into the flights)."""

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    geometries = []
    for name, size, pos in self.boxes:
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(size), pos=tuple(pos + eval_map3.MAP3_OFFSET)
      )
      rail = name.startswith("rail")
      if rail and self.scan_proxies:
        geom.group = 1  # Collides and renders, but the scan (group 0) ignores it.
      rgba = (0.85, 0.3, 0.3, self.rail_alpha) if rail else (0.78, 0.72, 0.6, 1.0)
      geometries.append(TerrainGeometry(geom=geom, color=rgba))
    if self.scan_proxies:
      for name, size, pos in rail_proxies(self.boxes):
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(size), pos=tuple(pos + eval_map3.MAP3_OFFSET)
        )
        geom.contype = 0
        geom.conaffinity = 0
        # Nearly transparent: ray casting skips geoms with alpha 0.
        geometries.append(TerrainGeometry(geom=geom, color=(1.0, 0.3, 0.3, 0.02)))
    origin = np.asarray(self.spawn) + eval_map3.MAP3_OFFSET
    return TerrainOutput(origin=origin, geometries=geometries)


def make_env_cfg(task: str, spawn, yaw_deg: float, teacher: bool):
  from mjlab.tasks.registry import load_env_cfg

  import src.tasks.velocity.mdp as mdp

  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=eval_map3.MAP3_TILE_SIZE,
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    curriculum=False,
    sub_terrains={
      "map3": Map3PlayTerrainCfg(
        spawn=spawn, boxes=eval_map3._load_map3_boxes(), scan_proxies=teacher
      )
    },
    add_lights=True,
  )
  cfg.scene.terrain.max_init_terrain_level = None
  if teacher:
    for sensor in cfg.scene.sensors or ():
      if sensor.name == "terrain_scan":
        p = sensor.pattern
        sensor.pattern = mdp.ElevatedGridPatternCfg(
          size=p.size, resolution=p.resolution, z_offset=SCAN_Z_OFFSET_PLAY
        )
  for name in ("randomize_terrain", "push_robot"):
    cfg.events.pop(name, None)
  yaw = math.radians(yaw_deg)
  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (yaw, yaw)}
  reset_base.params["stair_terrain_names"] = ()
  cfg.curriculum = {}
  cfg.episode_length_s = 1.0e5
  twist = cfg.commands["twist"]
  twist.stair_terrain_names = ()
  twist.corridor_terrain_names = ()
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.ranges.lin_vel_x = VX_RANGE
  twist.ranges.lin_vel_y = (-VY_MAX, VY_MAX)
  twist.ranges.ang_vel_z = (-WZ_MAX, WZ_MAX)
  # Start standing still; the gamepad or the sliders take over from there.
  twist.flat_mode_probs = (1.0, 0.0, 0.0, 0.0, 0.0)
  return cfg


class GamepadState:
  def __init__(self):
    self.lock = threading.Lock()
    self.cmd = (0.0, 0.0, 0.0)
    self.enabled = False
    self.last_seen = 0.0
    self.name = ""


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Go2 map3 gamepad</title>
<style>body{margin:0;font-family:sans-serif;background:#111;color:#ddd}
#bar{padding:6px 10px;display:flex;gap:18px;align-items:center;font-size:14px}
#v{width:100vw;height:calc(100vh - 34px);border:0}.on{color:#5f5}.off{color:#f77}</style></head>
<body><div id="bar"><b>Gamepad</b><span id="pad" class="off">not connected (press any button)</span>
<span id="st"></span><span id="cmd"></span><span>A: on/off &nbsp; B: stop &nbsp; left stick: vx/vy &nbsp; right stick: wz</span></div>
<iframe id="v"></iframe>
<script>
const VIEWER_PORT=__VIEWER_PORT__, VX=[__VXMIN__,__VXMAX__], VY=__VY__, WZ=__WZ__, DZ=0.12;
document.getElementById('v').src = location.protocol+'//'+location.hostname+':'+VIEWER_PORT+'/';
let enabled=false, prevA=false, prevB=false;
function dz(x){return Math.abs(x)<DZ?0:(x-Math.sign(x)*DZ)/(1-DZ);}
function tick(){
  const pads=navigator.getGamepads?navigator.getGamepads():[]; let p=null;
  for(const g of pads){if(g&&g.connected){p=g;break;}}
  const padEl=document.getElementById('pad');
  if(!p){padEl.textContent='not connected (press any button)';padEl.className='off';return;}
  padEl.textContent=p.id.slice(0,40);padEl.className='on';
  const a=p.buttons[0]&&p.buttons[0].pressed, b=p.buttons[1]&&p.buttons[1].pressed;
  if(a&&!prevA) enabled=!enabled; prevA=a;
  let stop=false; if(b&&!prevB) stop=true; prevB=b;
  const ly=dz(p.axes[1]||0), lx=dz(p.axes[0]||0), rx=dz(p.axes[2]||0);
  let vx=-ly*(ly<0?VX[1]:-VX[0]), vy=-lx*VY, wz=-rx*WZ;
  if(stop||b){vx=vy=wz=0;}
  document.getElementById('st').innerHTML=enabled?'<span class="on">CONTROL ON</span>':'<span class="off">control off</span>';
  document.getElementById('cmd').textContent=`vx ${vx.toFixed(2)}  vy ${vy.toFixed(2)}  wz ${wz.toFixed(2)}`;
  fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vx,vy,wz,enabled,name:p.id})}).catch(()=>{});
}
setInterval(tick, 33);
</script></body></html>"""


def serve_gamepad(state: GamepadState, port: int, viewer_port: int) -> None:
  page = (
    PAGE.replace("__VIEWER_PORT__", str(viewer_port))
    .replace("__VXMIN__", str(VX_RANGE[0]))
    .replace("__VXMAX__", str(VX_RANGE[1]))
    .replace("__VY__", str(VY_MAX))
    .replace("__WZ__", str(WZ_MAX))
  ).encode()

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
      pass

    def do_GET(self):
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.end_headers()
      self.wfile.write(page)

    def do_POST(self):
      data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
      with state.lock:
        state.cmd = (float(data.get("vx", 0)), float(data.get("vy", 0)), float(data.get("wz", 0)))
        state.enabled = bool(data.get("enabled", False))
        state.name = str(data.get("name", ""))
        state.last_seen = time.monotonic()
      self.send_response(204)
      self.end_headers()

  ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


def drive_sliders(state: GamepadState, term, robot) -> None:
  """Copy the gamepad command into the viewer's Twist sliders (30 Hz)."""
  was_enabled = False
  last_status = 0.0
  while True:
    time.sleep(1 / 30)
    sliders, enable = term._joystick_sliders, term._joystick_enabled
    if enable is None or len(sliders) != 3:
      continue
    with state.lock:
      fresh = time.monotonic() - state.last_seen < 0.5
      on, cmd = state.enabled and fresh, state.cmd
    if on:
      enable.value = True
      for s, v in zip(sliders, cmd, strict=True):
        s.value = float(np.clip(v, s.min, s.max))
    elif was_enabled:
      for s in sliders:
        s.value = 0.0
    if on != was_enabled:
      print(f"[gamepad] control {'ON' if on else 'off'} ({state.name or 'no pad'})", flush=True)
    if on and time.monotonic() - last_status > 3.0:
      last_status = time.monotonic()
      z = float(robot.data.root_link_pos_w[0, 2]) - float(eval_map3.MAP3_OFFSET[2])
      print(f"[gamepad] cmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f} | base height {z:.2f} m", flush=True)
    was_enabled = on


def load_policy(checkpoint: str, env):
  ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
  kind = "student" if "student_state_dict" in ck else "teacher"
  obs = env.get_observations()
  model = eval_map3.load_policy(kind, checkpoint, TensorDict(obs, batch_size=[1]), env.unwrapped.device)

  @torch.inference_mode()
  def policy(o):
    return model(o)

  return kind, policy


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--spawn", choices=sorted(SPAWNS), default="l0_bottom")
  parser.add_argument("--port", type=int, default=8080)
  parser.add_argument("--gamepad-port", type=int, default=8090)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import viser
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.viewer import ViserPlayViewer

  teacher = "student_state_dict" not in torch.load(
    args.checkpoint, map_location="cpu", weights_only=False
  )
  task = "Unitree-Go2-Depth-Teacher" if teacher else "Unitree-Go2-Depth-Student"
  spawn, yaw = SPAWNS[args.spawn]
  env = ManagerBasedRlEnv(cfg=make_env_cfg(task, spawn, yaw, teacher), device=args.device)
  env = RslRlVecEnvWrapper(env)
  kind, policy = load_policy(args.checkpoint, env)

  state = GamepadState()
  term = env.unwrapped.command_manager.get_term("twist")
  threading.Thread(target=serve_gamepad, args=(state, args.gamepad_port, args.port), daemon=True).start()
  robot = env.unwrapped.scene["robot"]
  threading.Thread(target=drive_sliders, args=(state, term, robot), daemon=True).start()
  print(f"[play_map3] {kind} {args.checkpoint} at spawn {args.spawn}")
  print(f"[play_map3] viewer:  http://<host>:{args.port}")
  print(f"[play_map3] gamepad: http://<host>:{args.gamepad_port}  (viewer embedded)")
  server = viser.ViserServer(port=args.port, label="go2 map3")
  ViserPlayViewer(env, policy, viser_server=server).run()


if __name__ == "__main__":
  main()
