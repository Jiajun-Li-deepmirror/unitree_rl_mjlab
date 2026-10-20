"""Evaluate a blind Go2 policy on the tinynav-sim map3 stairwell.

map3 (src/assets/terrains/map3: 123 collision boxes plus guard rails) is loaded as
a single terrain tile; it is never part of the training terrain. Each scenario
spawns all robots at one map3 spot with a fixed heading and runs a fixed command
script on the play config of --task (default Unitree-Go2-Blind). Stair scenarios count a robot as
`reached` when it gains (climb) or loses (descend) the target height before any
termination; `reset_before_goal` is the share terminated before that. With
--operator (default) the command also holds the flight heading and steers back
to the flight centerline, like an operator with a joystick.

--time-scale stretches the moving segments of the stair scenarios: at 1.0 a
policy must track the commanded speed to finish in time, at 1.6 slower robots
can still finish.

L0_down walks 10 s at 0.4 m/s and then stops for 2 s on the landing, like the
go2-depth eval: the 3.65 m flight alone takes ~9.1 s at 0.4 m/s, so the earlier
9 s script needed ~12% overspeed. The same run also reports the result at the end
of that earlier script (`reached_legacy`, `reset_before_goal_legacy`; the new
script starts with the same commands), so results stay comparable with older ones.

Stair scenarios with one moving segment also report the speed along the flight
over the commanded speed (`speed_p10`/`speed_p50`/`speed_p90`): the horizontal
speed along the flight heading from 0.15 m of height change to 0.15 m short of the
target, over the robots that got that far before any termination (`speed_share`,
so read it at --time-scale 1.6, where nearly all robots get there). Pick stairs
checkpoints by it as well: a share at a time limit does not tell a slow policy from
an on-time one.

  python scripts/go2_blind/eval_map3.py logs/rsl_rl/go2_blind/<run>/model_N.pt --num-envs 512
"""

from __future__ import annotations

import argparse
import gc
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)
from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

TASK = "Unitree-Go2-Blind"
MAP3_DIR = Path(__file__).resolve().parents[2] / "src" / "assets" / "terrains" / "map3"
# Shift map3 world coordinates into a positive tile.
MAP3_OFFSET = np.array([13.0, 1.0, 0.0])
MAP3_TILE_SIZE = (14.0, 10.0)
_BOX_RE = re.compile(r'<geom name="([^"]+)"\s+type="box"\s+[^>]*?size="([^"]+)"\s+pos="([^"]+)"')


@dataclass
class Scenario:
  name: str
  spawn: tuple[float, float, float]
  """map3 world position on the walking surface (x, y, top z)."""
  yaw_deg: float
  segments: list[tuple[float, float, float, float]]
  """(duration_s, vx, vy, wz) command script."""
  success: str
  """'climb:<dz>' (gain at least dz), 'descend:<dz>' (lose at least dz) or
  'track' (no fall, follow commands)."""
  legacy_segments: list[tuple[float, float, float, float]] | None = None
  """Earlier, shorter command script that `segments` starts with; the result at its
  end is reported too."""


SCENARIOS = [
  Scenario("L0_up", (-10.34, 5.75, 0.431), -90.0, [(1.0, 0, 0, 0), (9.0, 0.5, 0, 0)], "climb:2.0"),
  Scenario(
    "L0_down",
    (-10.34, 1.05, 2.659),
    90.0,
    [(1.0, 0, 0, 0), (10.0, 0.4, 0, 0), (2.0, 0, 0, 0)],
    "descend:2.0",
    legacy_segments=[(1.0, 0, 0, 0), (9.0, 0.4, 0, 0)],
  ),
  Scenario("R1_up", (-8.80, 1.05, 2.659), 90.0, [(1.0, 0, 0, 0), (9.0, 0.5, 0, 0)], "climb:1.7"),
  Scenario("R2b_down", (-8.75, 5.70, 11.802), -90.0, [(1.0, 0, 0, 0), (7.0, 0.4, 0, 0)], "descend:1.0"),
  Scenario(
    "L0_stop_midway",
    (-10.34, 5.75, 0.431),
    -90.0,
    [(1.0, 0, 0, 0), (4.0, 0.5, 0, 0), (4.0, 0, 0, 0), (5.0, 0.5, 0, 0)],
    "climb:2.0",
  ),
  Scenario(
    "F1_omni",
    (-5.30, 3.45, 0.0),
    180.0,
    [
      (1.0, 0, 0, 0),
      (2.0, 0.6, 0, 0),
      (2.0, -0.4, 0, 0),
      (2.0, 0, 0.4, 0),
      (2.0, 0, -0.4, 0),
      (2.0, 0, 0, 1.0),
      (2.0, 0, 0, -1.0),
      (2.0, 0.5, 0.2, 0.5),
      (2.0, 0, 0, 0),
    ],
    "track",
  ),
]


# --set flights: up and down every map3 flight (step_<flight>_NN boxes).
APPROACH = 0.65
"""Spawn distance before the first riser (up) or the top edge (down), m."""
GOAL_SHARE = 0.85
"""Share of the flight's rise that counts as done."""
TIME_SLACK = 1.25
"""Walking time = this x (approach + goal share of the flight) / commanded speed."""


def _surface_z(boxes, x: float, y: float, below: float | None = None, above: float | None = None) -> float:
  """Top of the highest box under (x, y) up to `below`, or of the lowest one from `above`."""
  tops = [
    p[2] + s[2]
    for name, s, p in boxes
    if not name.startswith(("rail", "roof")) and abs(x - p[0]) <= s[0] and abs(y - p[1]) <= s[1]
  ]
  if below is not None:
    return max(t for t in tops if t <= below)
  return min(t for t in tops if t >= above)


def flight_scenarios(boxes, up_vx: float = 0.5, down_vx: float = 0.4) -> list[Scenario]:
  """Up and down every map3 flight, spawned on its landings."""
  flights: dict[str, list] = {}
  for name, size, pos in boxes:
    m = re.match(r"step_(.+)_\d+$", name)
    if m:
      flights.setdefault(m.group(1), []).append((size, pos))
  scenarios = []
  for flight, steps in sorted(flights.items()):
    steps.sort(key=lambda b: b[1][2] + b[0][2])  # by tread height
    (s_lo, p_lo), (s_hi, p_hi) = steps[0], steps[-1]
    d = p_hi[:2] - p_lo[:2]
    axis = int(abs(d[1]) > abs(d[0]))  # ascent along x (0) or y (1)
    sign = 1.0 if d[axis] > 0 else -1.0
    other = 1 - axis
    lateral = float(np.mean([p[other] for _, p in steps]))
    length = max(p[axis] + s[axis] for s, p in steps) - min(p[axis] - s[axis] for s, p in steps)
    first_riser = p_lo[axis] - sign * s_lo[axis]
    top_edge = p_hi[axis] + sign * s_hi[axis]

    def point(u: float, axis=axis, other=other, lateral=lateral) -> tuple[float, float]:
      xy = [0.0, 0.0]
      xy[axis], xy[other] = u, lateral
      return xy[0], xy[1]

    bottom = point(first_riser - sign * APPROACH)
    top = point(top_edge + sign * APPROACH)
    z_bottom = _surface_z(boxes, *bottom, below=p_lo[2] + s_lo[2] - 0.02)
    z_top = _surface_z(boxes, *top, above=p_hi[2] + s_hi[2] + 0.02)
    rise = z_top - z_bottom
    yaw_up = math.degrees(math.atan2(sign * axis, sign * (1 - axis)))
    for direction, (x, y), z, yaw, vx, kind in (
      ("up", bottom, z_bottom, yaw_up, up_vx, "climb"),
      ("down", top, z_top, yaw_up + 180.0, down_vx, "descend"),
    ):
      walk = round(float(TIME_SLACK * (APPROACH + GOAL_SHARE * length) / vx), 1)
      scenarios.append(
        Scenario(
          f"{flight}_{direction}",
          (x, y, z),
          (yaw + 180.0) % 360.0 - 180.0,
          [(1.0, 0, 0, 0), (walk, vx, 0, 0), (2.0, 0, 0, 0)],
          f"{kind}:{GOAL_SHARE * rise:.2f}",
        )
      )
  return scenarios


def load_map3_boxes(map3_dir: Path = MAP3_DIR) -> list[tuple[str, np.ndarray, np.ndarray]]:
  text = (map3_dir / "map3_collision.xml").read_text() + (map3_dir / "scene.xml").read_text()
  boxes = []
  for name, size, pos in _BOX_RE.findall(text):
    if name.startswith(("step", "landing", "connector", "floor", "roof", "rail")):
      boxes.append((name, np.fromstring(size, sep=" "), np.fromstring(pos, sep=" ")))
  if len(boxes) < 123:
    raise RuntimeError(f"Expected >= 123 map3 boxes, parsed {len(boxes)}.")
  return boxes


@dataclass(kw_only=True)
class Map3TerrainCfg(SubTerrainCfg):
  spawn: tuple[float, float, float] = (0.0, 0.0, 0.0)
  size: tuple[float, float] = MAP3_TILE_SIZE
  boxes: list = field(default_factory=list)

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    geometries = []
    for name, size, pos in self.boxes:
      geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(size), pos=tuple(pos + MAP3_OFFSET))
      rgba = (0.85, 0.3, 0.3, 1.0) if name.startswith("rail") else (0.78, 0.72, 0.6, 1.0)
      geometries.append(TerrainGeometry(geom=geom, color=rgba))
    return TerrainOutput(origin=np.asarray(self.spawn) + MAP3_OFFSET, geometries=geometries)


def make_env_cfg(scenario: Scenario, boxes, num_envs: int, seed: int, task: str = TASK):
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=MAP3_TILE_SIZE,
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    curriculum=False,
    sub_terrains={"map3": Map3TerrainCfg(spawn=scenario.spawn, boxes=boxes)},
    add_lights=True,
  )
  cfg.scene.terrain.max_init_terrain_level = None
  cfg.events.pop("randomize_terrain", None)
  cfg.events.pop("push_robot", None)
  yaw = math.radians(scenario.yaw_deg)
  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (yaw - 0.05, yaw + 0.05)}
  reset_base.params["stair_terrain_names"] = ()
  cfg.curriculum = {}
  cfg.episode_length_s = sum(s[0] for s in scenario.segments) + 1.0
  twist = cfg.commands["twist"]
  twist.stair_terrain_names = ()
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.zero_threshold = 0.0
  return cfg


def set_command(env, vx: float, vy: float, wz: float) -> None:
  """Make the command term resample exactly (vx, vy, wz) on the next step.

  Going through the term (instead of writing the command tensor) keeps every
  observation group, history included, consistent; like a joystick, the new
  command reaches the policy one control step later.
  """
  term = env.command_manager.get_term("twist")
  fixed = UniformVelocityCommandCfg.Ranges(lin_vel_x=(vx, vx), lin_vel_y=(vy, vy), ang_vel_z=(wz, wz))
  term.cfg.ranges = fixed
  term.cfg.stair_ranges = fixed
  term.cfg.flat_mode_probs = (0.0, 0.0, 0.0, 0.0, 1.0)
  term.cfg.stair_standing_prob = 0.0
  term.time_left[:] = 0.0


def load_policy(env, checkpoint: str, task: str = TASK):
  agent_cfg = load_rl_cfg(task)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(task)(venv, asdict(agent_cfg), device="cuda:0")
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return venv, runner.get_inference_policy(device="cuda:0")


def run_scenario(args, scenario: Scenario, boxes) -> dict:
  env = ManagerBasedRlEnv(
    cfg=make_env_cfg(scenario, boxes, args.num_envs, args.seed, args.task), device="cuda:0"
  )
  venv, policy = load_policy(env, args.checkpoint, args.task)
  env.reset()
  obs = venv.get_observations()

  robot = env.scene["robot"]
  z0 = robot.data.root_link_pos_w[:, 2].clone()
  zmin, zmax = z0.clone(), z0.clone()
  fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  err_xy, err_yaw, n = 0.0, 0.0, 0
  yaw_target = math.radians(scenario.yaw_deg)
  axis = torch.tensor((math.cos(yaw_target), math.sin(yaw_target)), device=env.device)
  term = env.command_manager.get_term("twist")
  kind, _, thr = scenario.success.partition(":")
  reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def goal_met():
    if kind == "climb":
      return zmax - z0 >= float(thr)
    if kind == "descend":
      return z0 - zmin >= float(thr)
    return torch.zeros_like(reached)

  legacy_steps, legacy = None, None
  if scenario.legacy_segments:
    legacy_steps = sum(int(round(d / env.step_dt)) for d, *_ in scenario.legacy_segments)
  # Speed along the flight: time and position along the heading when a robot has
  # changed height by 0.15 m and when it is 0.15 m short of the target.
  moving_vx = [vx for _, vx, _, _ in scenario.segments if vx != 0.0]
  measure_speed = kind != "track" and len(moving_vx) == 1
  sign = 1.0 if kind == "climb" else -1.0
  t_in = torch.full((env.num_envs,), float("nan"), device=env.device)
  t_out, s_in, s_out = t_in.clone(), t_in.clone(), t_in.clone()
  # Heading error (deg, + = turned left) and offset from the spawn line along the
  # flight heading (m, + = left) when a robot reaches the goal: drift with
  # --no-operator, which the operator otherwise corrects.
  yaw_at, lat_at = t_in.clone(), t_in.clone()
  steps_done = 0

  for duration, vx, vy, wz in scenario.segments:
    set_command(env, vx, vy, wz)
    for step in range(int(round(duration / env.step_dt))):
      if args.operator and kind != "track" and vx != 0.0 and step > 0:
        heading_err = torch.atan2(
          torch.sin(yaw_target - robot.data.heading_w), torch.cos(yaw_target - robot.data.heading_w)
        )
        rel = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
        lateral = -axis[1] * rel[:, 0] + axis[0] * rel[:, 1]
        term.vel_command_b[:, 1] = (-lateral).clamp(-0.2, 0.2)
        term.vel_command_b[:, 2] = heading_err.clamp(-0.5, 0.5)
      with torch.no_grad():
        obs, _, dones, extras = venv.step(policy(obs))
      terminated = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
      # Goal reached before any termination (wall hits after the end of a flight,
      # where the command keeps walking forward, do not undo it).
      reached |= goal_met() & ~fell
      new = reached & torch.isnan(yaw_at) & ~terminated
      if new.any():
        err = robot.data.heading_w - yaw_target
        err = torch.rad2deg(torch.atan2(torch.sin(err), torch.cos(err)))
        rel = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
        yaw_at = torch.where(new, err, yaw_at)
        lat_at = torch.where(new, -axis[1] * rel[:, 0] + axis[0] * rel[:, 1], lat_at)
      fell |= terminated
      z = robot.data.root_link_pos_w[:, 2]
      zmin = torch.where(fell, zmin, torch.minimum(zmin, z))
      zmax = torch.where(fell, zmax, torch.maximum(zmax, z))
      steps_done += 1
      if measure_speed:
        t = torch.full_like(t_in, steps_done * env.step_dt)
        dz = sign * (z - z0)
        s = robot.data.root_link_pos_w[:, :2] @ axis
        enter = torch.isnan(t_in) & (dz >= 0.15) & ~fell
        t_in, s_in = torch.where(enter, t, t_in), torch.where(enter, s, s_in)
        leave = torch.isnan(t_out) & ~torch.isnan(t_in) & (dz >= float(thr) - 0.15) & ~fell
        t_out, s_out = torch.where(leave, t, t_out), torch.where(leave, s, s_out)
      if steps_done == legacy_steps:
        legacy = (reached | (goal_met() & ~fell), fell.clone())
      alive = ~fell
      if kind == "track" and alive.any():
        v = robot.data.root_link_lin_vel_b[alive]
        w = robot.data.root_link_ang_vel_b[alive, 2]
        err_xy += float(torch.norm(v[:, :2] - torch.tensor((vx, vy), device=env.device), dim=1).mean())
        err_yaw += float((w - wz).abs().mean())
        n += 1
  reached |= goal_met() & ~fell
  result = {"scenario": scenario.name}
  if kind == "track":
    result |= {"success": float((~fell).float().mean()), "err_vxy": err_xy / n, "err_wz": err_yaw / n}
  else:
    result |= {"reached": float(reached.float().mean()), "reset_before_goal": float((fell & ~reached).float().mean())}
  if legacy is not None:
    legacy_reached, legacy_fell = legacy
    result |= {
      "reached_legacy": float(legacy_reached.float().mean()),
      "reset_before_goal_legacy": float((legacy_fell & ~legacy_reached).float().mean()),
    }
  if kind != "track":
    ok = ~torch.isnan(yaw_at)
    qs = torch.tensor((0.1, 0.5, 0.9), device=env.device)
    yq = yaw_at[ok].quantile(qs).tolist() if ok.any() else [math.nan] * 3
    lq = lat_at[ok].quantile(qs).tolist() if ok.any() else [math.nan] * 3
    result |= {
      "yaw_deg_p10": yq[0], "yaw_deg_p50": yq[1], "yaw_deg_p90": yq[2],
      "lat_m_p10": lq[0], "lat_m_p50": lq[1], "lat_m_p90": lq[2],
    }
  if measure_speed:
    ok = ~torch.isnan(t_out) & (t_out > t_in)
    ratio = ((s_out - s_in) / (t_out - t_in) / moving_vx[0])[ok]
    q = ratio.quantile(torch.tensor((0.1, 0.5, 0.9), device=env.device)).tolist() if ok.any() else [math.nan] * 3
    result |= {"speed_p10": q[0], "speed_p50": q[1], "speed_p90": q[2], "speed_share": float(ok.float().mean())}
  env.close()
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("checkpoint")
  parser.add_argument("--task", default=TASK, help="Blind stairs task whose play config matches the policy.")
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--time-scale", type=float, default=1.0)
  parser.add_argument(
    "--set", choices=("default", "flights"), default="default",
    help="default: the scenarios above; flights: up and down every map3 flight.",
  )
  parser.add_argument("--up-vx", type=float, default=0.5, help="--set flights: climbing command (m/s).")
  parser.add_argument("--down-vx", type=float, default=0.4, help="--set flights: descending command (m/s).")
  parser.add_argument("--scenarios", nargs="*", default=None, help="Names to run (default: the whole set).")
  parser.add_argument("--list", action="store_true", help="Print the scenarios and exit.")
  parser.add_argument("--operator", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  boxes = load_map3_boxes()
  scenarios = SCENARIOS if args.set == "default" else flight_scenarios(boxes, args.up_vx, args.down_vx)
  if args.list:
    for s in scenarios:
      print(f"{s.name:16s} spawn ({s.spawn[0]:7.2f}, {s.spawn[1]:5.2f}, {s.spawn[2]:6.3f}) "
            f"yaw {s.yaw_deg:6.1f}  {s.success:14s} segments {s.segments}")
    return
  results = []
  for s in scenarios:
    if args.scenarios is not None and s.name not in args.scenarios:
      continue
    if s.success != "track":
      s.segments = [(d * args.time_scale if vx else d, vx, vy, wz) for d, vx, vy, wz in s.segments]
      if s.legacy_segments:
        s.legacy_segments = [(d * args.time_scale if vx else d, vx, vy, wz) for d, vx, vy, wz in s.legacy_segments]
    results.append(run_scenario(args, s, boxes))
    # A closed env's GPU arrays go only when Python collects its reference cycles;
    # without this the trainers failed to capture a later scenario's CUDA graph
    # (~11th of the 18 flights; found by the go2-depth session, 2026-10-20).
    gc.collect()
    torch.cuda.empty_cache()
  print(f"\n[map3] {args.checkpoint}: {args.num_envs} robots per scenario, time scale {args.time_scale}")
  for r in results:
    print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
  main()
