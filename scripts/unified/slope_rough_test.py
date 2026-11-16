"""slope_rough_test.py --task <-Stairs task> <checkpoint> (go2-depth 2026-11-16; acceptance item 23, 斜坡和不平地面, the
user via go2-home: a 15 deg ramp up and down and rough ground +-3 cm, walked across at 0.5 and 1.0 m/s; bar: reach
>= 95 % unloaded and typical, >= 90 % full, 0 falls).

The task's play build (a teacher's -Stairs task, or a student's -Stairs task through jobs/unified/u_eval.py) with its
terrain generator replaced by three tile types, 10 x 10 m, the robots spawned at the tile's start facing +x:
  slope15 up    a flat start (2 m), a ramp of 6 m horizontal run at 15 deg (rise 1.61 m), a flat top (2 m) -- box
                geoms, the ramp one box tilted about y (mjlab's HfPyramidSlopedTerrain is a bilinear tent, not a plane);
  slope15 down  the same, mirrored: spawned on the top, down the ramp;
  rough3        the training's random-rough generator (HfRandomUniformTerrainCfg, there 2-6 cm in 1 cm steps) at
                -3..+3 cm in 1 cm steps over the whole tile.
Every robot walks the tile's axis: the command vx (slewed at 2 m/s^2 from a 1 s stand), the heading held on +x and the
lateral offset held at its spawn line by the operator's correction (eval_map3's: wz = heading error, vy = -offset, both
clipped). Reached: the base 7.5 m along the axis from its spawn (past the ramp's end by 0.5 m) within 7.5 / vx x 1.5
+ 2 s; the command goes to zero once there (a robot that runs on off the tile is not judged after it). A fall: a
termination before the reach (the first only; the play build has no time-out). Speed: 7.5 m over the time from the
command's start to the reach, against vx (the reached robots' mean).

Cells: (slope15 up, slope15 down, rough3) x (0.5, 1.0 m/s), --num-envs robots each. Payload from U_PAYLOAD (the
unified build's base_payload), scan mode from U_SCAN, as the other unified tests. Output: one row per cell
  " slope15 up   0.5 | reached 98.4 % | falls 0/64 | speed 0.93"
and the line "ALL slopes (item 23): reach min X % over 6 cells, falls F of N robots".

  python jobs/unified/u_eval.py scripts/unified/slope_rough_test.py --task Unitree-Go2W-U-Teacher-Stairs <ckpt>
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "go2_blind"))
sys.path.insert(1, os.path.join(os.getcwd(), "scripts", "go2_blind"))   # (run from the repo root, a copy elsewhere)

import mjlab.tasks  # noqa: F401,E402
import src.tasks  # noqa: F401,E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.terrains import HfRandomUniformTerrainCfg  # noqa: E402
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput  # noqa: E402

TILE = 10.0          # m, square tiles
START = 2.0          # m of flat before the ramp (the spawn at its middle)
RUN = 6.0            # m, the ramp's horizontal run
SLOPE_DEG = 15.0
ROUGH = (-0.03, 0.03)
ROUGH_STEP = 0.01
SPAWN_X = START / 2  # m from the tile's start edge
JUDGED = START / 2 + RUN + 0.5   # m along the axis from the spawn: 0.5 m past the ramp's end
SPEEDS = (0.5, 1.0)
LIN_ACCEL = 2.0      # m/s^2, the command's rise
SETTLE_S = 1.0


@dataclass(kw_only=True)
class RampTerrainCfg(SubTerrainCfg):
  """A flat start, a plane ramp along +x (one box tilted about y), a flat end; spawn on the start."""
  slope_deg: float = SLOPE_DEG
  start_length: float = START
  ramp_run: float = RUN
  down: bool = False
  thickness: float = 0.4

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    L, W = self.size
    a = math.radians(self.slope_deg)
    rise = self.ramp_run * math.tan(a)
    z0, z1 = (rise, 0.0) if self.down else (0.0, rise)
    floor = -self.thickness
    geoms = []

    def slab(x0, x1, ztop, rgba):
      g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=((x1 - x0) / 2, W / 2, (ztop - floor) / 2),
                        pos=((x0 + x1) / 2, W / 2, (ztop + floor) / 2))
      geoms.append(TerrainGeometry(geom=g, color=rgba))

    s, e = self.start_length, self.start_length + self.ramp_run
    slab(0.0, s, z0, (0.55, 0.55, 0.55, 1.0))
    slab(e, L, z1, (0.55, 0.55, 0.55, 1.0))
    # the ramp: a box whose top face runs from (s, z0) to (e, z1); rotated about y by -a (up) / +a (down)
    th = a if self.down else -a
    xa = np.array([math.cos(th), 0.0, -math.sin(th)])   # the box's x axis in the tile frame
    n = np.array([math.sin(th), 0.0, math.cos(th)])     # its z axis: the ramp's normal
    top = np.array([(s + e) / 2, W / 2, (z0 + z1) / 2])
    c = top - n * (self.thickness / 2)
    assert abs(np.dot(xa, [self.ramp_run, 0.0, z1 - z0]) - math.hypot(self.ramp_run, z1 - z0)) < 1e-9
    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(math.hypot(self.ramp_run, rise) / 2, W / 2, self.thickness / 2),
                      pos=tuple(c), quat=(math.cos(th / 2), 0.0, math.sin(th / 2), 0.0))
    geoms.append(TerrainGeometry(geom=g, color=(0.75, 0.45, 0.30, 1.0)))
    return TerrainOutput(origin=np.array([SPAWN_X, W / 2, z0]), geometries=geoms)


@dataclass(kw_only=True)
class RoughStartCfg(HfRandomUniformTerrainCfg):
  """The random-rough tile with the spawn at the tile's start instead of its centre."""

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    out = super().function(difficulty, spec, rng)
    out.origin = np.array([SPAWN_X, self.size[1] / 2, float(out.origin[2])])
    return out


TERRAINS = ("slope15 up", "slope15 down", "rough3")
# the turns on the rough tile (2026-11-16, the user via go2-home: the Go2W videos showed very low leg lifts in big turns,
# and the minimal recipe dropped foot_swing_clearance): a spin in place both ways and a big arc, TURN_S each after the
# settle; judged 0 falls and no wheel stuck (in contact, |wheel speed| < STUCK_W rad/s for more than STUCK_S); reported
# the stepping feet's swing: each lift-off airborne > AIR_S, its apex above its take-off, p50 / p90 per leg
TURNS = (("spin +1.0", (0.0, 0.0, 1.0)), ("spin -1.0", (0.0, 0.0, -1.0)), ("arc 1.0+1.0", (1.0, 0.0, 1.0)))
TURN_S = 8.0
YAW_ACCEL = 3.0      # rad/s^2, the turn's rise (the acceptance's controllers)
STUCK_W, STUCK_S = 0.5, 1.0
AIR_S = 0.06
FORCE_N = 5.0
LEGS = ("FL", "FR", "RL", "RR")


def depth_report(env, t_idx) -> None:
  """--depth-check: per tile type, the depth camera's lower image half -- the mean depth of the pixels with a hit and
  the share without (0, inf or nan)."""
  try:
    sensor = env.scene["front_depth"]
  except KeyError:
    print("depth check: no front_depth sensor (not a depth student's task)")
    return
  d = sensor.data.depth.float()
  if d.dim() == 4 and d.shape[-1] == 1:   # (N, H, W, 1)
    d = d[..., 0]
  if d.dim() == 3:                         # (N, H, W): the lower half, the ground ahead
    d = d[:, d.shape[1] // 2:, :]
  d = d.reshape(d.shape[0], -1)
  miss = ~torch.isfinite(d) | (d <= 0.0)
  for ti, name in enumerate(TERRAINS):
    m = t_idx == ti
    dm = torch.where(miss[m], torch.zeros_like(d[m]), d[m])
    hit = (~miss[m]).float()
    print(f"depth check: {name:12s} lower image half: mean depth {float(dm.sum() / hit.sum().clamp(min=1)):.2f} m, "
          f"no hit {100 * float(1 - hit.mean()):.1f} % of the pixels (image {tuple(sensor.data.depth.shape[1:])})")


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task", required=True, help="a -Stairs play task (teacher, or student through u_eval.py)")
  p.add_argument("checkpoint")
  p.add_argument("--num-envs", type=int, default=64, help="robots per (terrain, speed) cell")
  p.add_argument("--seed", type=int, default=None, help="the environment's seed; unset: none, as the flat tests")
  p.add_argument("--no-turns", action="store_true", help="the walks only (no turning cells)")
  p.add_argument("--depth-check", action="store_true",
                 help="a depth student: print what its camera sees of each tile after the settle (the lower half of the "
                      "image: mean depth and the share of pixels with no hit) -- the rough tile is a heightfield")
  a = p.parse_args()

  cfg = load_env_cfg(a.task, play=True)
  if a.seed is not None:
    cfg.seed = a.seed
  gen = cfg.scene.terrain.terrain_generator
  gen.size = (TILE, TILE)
  gen.curriculum = True
  gen.sub_terrains = {
    "slope15_up": RampTerrainCfg(proportion=1.0, down=False),
    "slope15_down": RampTerrainCfg(proportion=1.0, down=True),
    "rough3": RoughStartCfg(proportion=1.0, noise_range=ROUGH, noise_step=ROUGH_STEP, border_width=0.25),
  }
  gen.num_rows = 2
  gen.num_cols = 3
  cfg.events.pop("randomize_terrain", None)
  cfg.events.pop("push_robot", None)
  rb = cfg.events["reset_base"]
  rb.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-1.0, 1.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
  rb.params["velocity_range"] = {}
  n = a.num_envs
  cells = [(t, v) for t in range(len(TERRAINS)) for v in SPEEDS]
  N = n * len(cells)
  cfg.scene.num_envs = N

  agent = load_rl_cfg(a.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(a.task)(venv, asdict(agent), device="cuda:0")
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  dev, dt = env.device, env.step_dt
  robot = env.scene["robot"]
  terrain = env.scene.terrain
  term = env.command_manager.get_term("twist")

  # columns -> tile types (the generator's curriculum rule, as eval_stairs.py)
  names = list(gen.sub_terrains.keys())
  cum = np.cumsum(np.full(len(names), 1.0 / len(names)))
  col_of = {names[int(np.min(np.where(c / gen.num_cols + 0.001 < cum)[0]))]: c for c in range(gen.num_cols)}
  cell = torch.arange(N, device=dev) // n
  t_idx = torch.tensor([cells[c][0] for c in range(len(cells))], device=dev)[cell]
  vx = torch.tensor([cells[c][1] for c in range(len(cells))], device=dev, dtype=torch.float32)[cell]
  types = torch.tensor([col_of[["slope15_up", "slope15_down", "rough3"][t]] for t in range(len(TERRAINS))], device=dev)[t_idx]
  levels = torch.arange(N, device=dev) % gen.num_rows
  terrain.terrain_levels[:] = levels
  terrain.terrain_types[:] = types
  terrain.env_origins[:] = terrain.terrain_origins[levels, types]

  import eval_map3 as em

  em.set_command(env, 0.0, 0.0, 0.0)   # the command term's training replacements off, no resample
  env.reset()
  obs = venv.get_observations()
  origin = robot.data.root_link_pos_w[:, :2].clone()
  z_start = robot.data.root_link_pos_w[:, 2].clone()
  z_reach = torch.full((N,), float("nan"), device=dev)
  deadline = JUDGED / vx * 1.5 + 2.0
  t_end = SETTLE_S + float(deadline.max())
  fell = torch.zeros(N, dtype=torch.bool, device=dev)
  reached = torch.zeros(N, dtype=torch.bool, device=dev)
  t_reach = torch.full((N,), float("nan"), device=dev)
  cur = torch.zeros(N, device=dev)
  offtile = torch.zeros(N, dtype=torch.bool, device=dev)
  for k in range(int(math.ceil(t_end / dt))):
    t = k * dt
    if k == int(round(SETTLE_S / dt)):
      origin = robot.data.root_link_pos_w[:, :2].clone()   # the start line: where each robot stands after the settle
      z_start = robot.data.root_link_pos_w[:, 2].clone()
      if a.depth_check:
        depth_report(env, t_idx)
    pos = robot.data.root_link_pos_w[:, :2]
    along = pos[:, 0] - origin[:, 0]
    lateral = pos[:, 1] - origin[:, 1]
    going = (t >= SETTLE_S) & ~reached & ~fell
    want = torch.where(going, vx, torch.zeros_like(vx))
    cur = torch.where(want > cur, torch.minimum(cur + LIN_ACCEL * dt, want), want)
    yaw = robot.data.heading_w
    herr = torch.atan2(torch.sin(-yaw), torch.cos(-yaw))
    term.vel_command_b[:, 0] = cur
    term.vel_command_b[:, 1] = torch.where(going, (-lateral).clamp(-0.2, 0.2), torch.zeros_like(cur))
    term.vel_command_b[:, 2] = torch.where(going, herr.clamp(-0.5, 0.5), torch.zeros_like(cur))
    term.time_left[:] = 1e9
    with torch.no_grad():
      obs, _, dones, extras = venv.step(policy(obs))
    terminated = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
    pos = robot.data.root_link_pos_w[:, :2]
    along = pos[:, 0] - origin[:, 0]
    tt = (k + 1) * dt - SETTLE_S
    new_fall = terminated & ~fell & ~reached & (tt > 0)
    fell |= new_fall
    offtile |= ~fell & ~reached & ((pos[:, 1] - origin[:, 1]).abs() > TILE / 2 - 1.0)
    hit = ~fell & ~reached & (along >= JUDGED) & (tt <= deadline)
    t_reach = torch.where(hit, torch.full_like(t_reach, tt), t_reach)
    z_reach = torch.where(hit, robot.data.root_link_pos_w[:, 2], z_reach)
    reached |= hit

  print(f"{a.checkpoint}\n{a.task}: item 23, {n} robots per cell; tiles {TILE:g} x {TILE:g} m: slope {SLOPE_DEG:g} deg "
        f"(box ramp, run {RUN:g} m, rise {RUN * math.tan(math.radians(SLOPE_DEG)):.2f} m, flat {START:g} m before / after), "
        f"rough {100 * ROUGH[0]:+.0f}..{100 * ROUGH[1]:+.0f} cm in {100 * ROUGH_STEP:g} cm steps; judged {JUDGED:g} m along the "
        f"axis within {JUDGED:g} / vx x 1.5 + 2 s; payload {os.environ.get('U_PAYLOAD', '(task default)')}, scan "
        f"{os.environ.get('U_SCAN', '(task default)')}, seed {a.seed}")
  print(f"{'terrain':>13s} {'vx':>4s} | reached | falls | speed along / command (reached robots' mean)")
  R, F, worst = reached.cpu().numpy(), fell.cpu().numpy(), 100.0
  tr, v, off = t_reach.cpu().numpy(), vx.cpu().numpy(), offtile.cpu().numpy()
  rise = (z_reach - z_start).cpu().numpy()
  for c, (ti, sp) in enumerate(cells):
    g = slice(c * n, (c + 1) * n)
    pct = 100.0 * R[g].mean()
    worst = min(worst, pct)
    ok = R[g] & np.isfinite(tr[g])
    spd = float(np.mean(JUDGED / tr[g][ok] / v[g][ok])) if ok.any() else float("nan")
    extra = f" | off the tile's line {int(off[g].sum())}" if off[g].any() else ""
    dz = float(np.nanmean(rise[g])) if np.isfinite(rise[g]).any() else float("nan")
    print(f" {TERRAINS[ti]:12s} {sp:3.1f} | reached {pct:5.1f} % | falls {int(F[g].sum())}/{n} | speed {spd:.2f} | base rise "
          f"{dz:+.2f} m{extra}")
  print(f"ALL slopes (item 23): reach min {worst:.1f} % over {len(cells)} cells, falls {int(F.sum())} of {N} robots")
  env.close()
  if not a.no_turns:
    turns(a)
  return 0


def turns(a) -> None:
  """The turning cells on the rough tile (TURNS' comment): falls, stuck wheels, the swing of the stepping feet."""
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  cfg = load_env_cfg(a.task, play=True)
  if a.seed is not None:
    cfg.seed = a.seed
  gen = cfg.scene.terrain.terrain_generator
  gen.size = (TILE, TILE)
  gen.curriculum = True
  gen.sub_terrains = {"rough3": HfRandomUniformTerrainCfg(proportion=1.0, noise_range=ROUGH, noise_step=ROUGH_STEP,
                                                           border_width=0.25)}
  gen.num_rows, gen.num_cols = 2, 3
  cfg.events.pop("randomize_terrain", None)
  cfg.events.pop("push_robot", None)
  rb = cfg.events["reset_base"]
  rb.params["pose_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 0.0), "yaw": (-math.pi, math.pi)}
  rb.params["velocity_range"] = {}
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
    ContactSensorCfg(name=f"turn_{leg}_foot", primary=ContactMatch(mode="geom", entity="robot", pattern=(rf"^{leg}_foot_collision$",)),
                     secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="none",
                     num_slots=1, history_length=4) for leg in LEGS)
  n = a.num_envs
  N = n * len(TURNS)
  cfg.scene.num_envs = N
  # every robot on the rough heightfield and turning: more contacts than the walks' mix (t1 11-16: "nconmax overflow
  # (nconmax must be >= 43)"), as blind_env_cfg raises it for its contact-heavy keys
  if (cfg.sim.nconmax or 0) < 128:
    cfg.sim.nconmax = 128
  agent = load_rl_cfg(a.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(a.task)(venv, asdict(agent), device="cuda:0")
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  dev, dt = env.device, env.step_dt
  robot = env.scene["robot"]
  term = env.command_manager.get_term("twist")
  feet = [env.scene[f"turn_{leg}_foot"] for leg in LEGS]
  wheel_ids = [robot.joint_names.index(f"{leg}_wheel_joint") for leg in LEGS if f"{leg}_wheel_joint" in robot.joint_names]
  # the feet's height: the wheel's centre, else the calf's end (the foot sphere at its tip; map3_stairs_probe's way)
  foot_ids, _ = robot.find_bodies([rf"^{leg}_wheel_link$" if wheel_ids else rf"^{leg}_calf$" for leg in LEGS], preserve_order=True)

  def feet_z():
    if wheel_ids:
      return robot.data.body_link_pos_w[:, foot_ids, 2]
    qq = robot.data.body_link_quat_w[:, foot_ids]
    zz = 1 - 2 * (qq[..., 1] ** 2 + qq[..., 2] ** 2)
    return robot.data.body_link_pos_w[:, foot_ids, 2] - 0.213 * zz
  cell = torch.arange(N, device=dev) // n
  target = torch.tensor([c for _, c in TURNS], device=dev, dtype=torch.float32)[cell]
  import eval_map3 as em

  em.set_command(env, 0.0, 0.0, 0.0)
  env.reset()
  obs = venv.get_observations()
  cur = torch.zeros(N, 3, device=dev)
  fell = torch.zeros(N, dtype=torch.bool, device=dev)
  stuck_t = torch.zeros(N, max(len(wheel_ids), 1), device=dev)
  stuck = torch.zeros(N, dtype=torch.bool, device=dev)
  prev = torch.ones(N, 4, dtype=torch.bool, device=dev)
  air = torch.zeros(N, 4, dtype=torch.bool, device=dev)
  t_air = torch.zeros(N, 4, device=dev)
  z_take = torch.zeros(N, 4, device=dev)
  z_apex = torch.zeros(N, 4, device=dev)
  swings = {leg: [] for leg in LEGS}
  steps = torch.zeros(N, 4, device=dev)
  for k in range(int(math.ceil((SETTLE_S + TURN_S) / dt))):
    t = k * dt
    on = t >= SETTLE_S
    want = target if on else torch.zeros_like(target)
    cur[:, 0] = torch.minimum(cur[:, 0] + LIN_ACCEL * dt, want[:, 0])
    cur[:, 2] = cur[:, 2] + (want[:, 2] - cur[:, 2]).clamp(-YAW_ACCEL * dt, YAW_ACCEL * dt)
    term.vel_command_b[:] = cur
    term.time_left[:] = 1e9
    with torch.no_grad():
      obs, _, dones, extras = venv.step(policy(obs))
    terminated = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
    fell |= terminated & on
    live = ~fell
    c = torch.stack([torch.norm(f.data.force_history.reshape(N, -1, 3), dim=-1).max(dim=1).values > FORCE_N for f in feet], dim=1)
    z = feet_z()
    judged = on & (t >= SETTLE_S + 1.0)   # (the first second of the turn: the rise)
    if wheel_ids:
      w = robot.data.joint_vel[:, wheel_ids].abs()
      still = c[:, :len(wheel_ids)] & (w < STUCK_W) & live[:, None]
      stuck_t = torch.where(still, stuck_t + dt, torch.zeros_like(stuck_t))
      if judged:
        stuck |= (stuck_t > STUCK_S).any(dim=1) & live
    start = prev & ~c & live[:, None] & on
    air = air | start
    z_take = torch.where(start, z, z_take)
    z_apex = torch.where(start, z, torch.where(air, torch.maximum(z_apex, z), z_apex))
    t_air = torch.where(start, torch.zeros_like(t_air), torch.where(air, t_air + dt, t_air))
    land = air & c
    done = land & (t_air > AIR_S)
    if bool(done.any()):
      r_, l_ = done.nonzero(as_tuple=True)
      lift = (z_apex - z_take)[r_, l_]
      for li, leg in enumerate(LEGS):
        swings[leg].append(lift[l_ == li].cpu())
      steps += done.float()
    air = air & ~c
    prev = c
  F, S = fell.cpu().numpy(), stuck.cpu().numpy()
  q = lambda x: (100 * float(np.percentile(x, 50)), 100 * float(np.percentile(x, 90))) if len(x) else (float("nan"),) * 2
  print(f"{a.checkpoint}\n{a.task}: item 23's turns on the rough tile (+-3 cm), {n} robots per cell, {TURN_S:g} s after "
        f"a {SETTLE_S:g} s settle (wz slewed at {YAW_ACCEL:g} rad/s^2); stuck = a wheel in contact below {STUCK_W:g} rad/s "
        f"for more than {STUCK_S:g} s; swing = a lift-off airborne > {AIR_S:g} s, its apex above its take-off")
  st = steps.cpu().numpy()
  for ci, (name, _c) in enumerate(TURNS):
    g = slice(ci * n, (ci + 1) * n)
    print(f" {name:12s} | falls {int(F[g].sum())}/{n} | stuck {int(S[g].sum())}/{n} | steps per robot "
          + " ".join(f"{leg} {st[g, li].mean():.1f}" for li, leg in enumerate(LEGS)))
  sw = {leg: (torch.cat(v).numpy() if v else np.array([])) for leg, v in swings.items()}
  print("ALL turns (item 23): falls %d of %d robots, stuck %d (a wheel still > %g s); swing apex p50 / p90 cm %s" % (
    int(F.sum()), N, int(S.sum()), STUCK_S, ", ".join(f"{leg} {q(sw[leg])[0]:.1f} / {q(sw[leg])[1]:.1f}" for leg in LEGS)))
  env.close()


if __name__ == "__main__":
  sys.exit(main())
