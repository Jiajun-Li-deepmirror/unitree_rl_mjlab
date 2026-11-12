#!/usr/bin/env python3
"""camera_fail.py R DEPTH_CKPT BLIND_CKPT --part flat|stairs [--scale S] [--seed N] (unitree-rl-mjlab-62 2026-11-12;
doc/redesign.md 2.2.1, the user: "加「相机失效、切到盲走」测试"): the delivered pair -- the depth student and the
blind student, each accepted on its own -- driving one robot together as the robot's depth controller does
(robot_ctrl DepthPolicy.h; tinynav-sim sim/runtime.py, sim/depth_monitor.py), with the depth camera failing.

Both students run every control step (20 ms). The blind student reads its own input (the teacher's actor terms
without the scan: its training's "actor" group, built here as "blind_actor"), whose proprio history the environment
keeps updating all the time, so a takeover does not start from an empty history. The depth student keeps running
through a takeover on what it is shown (the last frame of a stalled stream, or frames with no reading), its GRU state
kept. One state machine per robot picks whose action is applied:

  reset        fallback, healthy_s = 0
  each step    fault = no depth frame yet, or the newest frame older than STALE_S (0.5 s), or BLIND_FRAMES (3)
               frames in a row with no reading (frames come at 10 Hz, every DEPTH_UPDATE_PERIOD_STEPS)
               fault -> fallback, healthy_s = 0; else, in fallback, healthy_s += dt and at HEALTHY_S (1.0 s) back to
               the depth student
  action       fallback ? blind : depth

The failure, per robot: half "stall" (no new frame: the depth student sees the frame of the failure moment; the
takeover comes 0.5 s on), half "blind" (frames with no reading anywhere -- the pipeline's no-reading value, max range,
1.0 normalized; the takeover on the third such frame); half of each for the rest of the run, half for RECOVER_S
(3 s), then the camera works again and the depth student takes back over after 1.0 s healthy. The failure time is
drawn per robot: flat, in [2, 5] s; stairs, in [1, 7] s (the start of the flight to its end at the 0.5 m/s command).

--part flat    the 44 flat commands of accept_matrix.py x 16 robots (ramped 2 m/s^2 / 3 rad/s^2), plus 96 robots at
               the zero command, 11 s
--part stairs  eval_stairs.py's tiles (5-18 cm in 4 bands x up / down x narrow / wide treads), 36 robots per band and
               direction, the 0.5 m/s straight command for 14.4 s; a robot reaches when its base gets past the last
               step (Chebyshev distance 3.6 m from the tile centre) without terminating; where it was at the takeover:
               "steps" from 1.0 m (the 2 m central platform's edge) to 3.6 m, else "platform"
The payload is the environment's (U_PAYLOAD, as every check: 0 kg unloaded, the walking payload loaded).

Judged (doc/redesign.md 2.2.1):
  1. no fall (both conditions), after the failure and before it too -- the pair drives from the start (the blind
     student for the first second), and a robot down before its failure was never tested; the two are reported apart.
     Only the robots standing when their camera failed count as tested (the takeovers and the readings below)
  2. unloaded, stairs: the robots that took over on the steps reach >= REACH_BAR (0.90)
  3. unloaded, flat: from 2 s after the takeover, 1 s mean velocity at 80-120% of the command (each command's robots
     pooled, every axis of at least 0.05, as accept_matrix.py)
Reported only: the lowest planar speed within 1 s of the takeover (share of the command), the time back to 80%,
the roll / pitch peaks after the takeover, the creep of the standing robots over the 4 s after it.
Last line: "camera_fail verdict: PASS" or "camera_fail verdict: FAIL: <reasons>". Run with UNIFIED_TASKS=1 from the
repo root; jobs/unified/pair_accept.sh runs both parts in both conditions.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from copy import deepcopy
from dataclasses import asdict

os.environ.setdefault("UNIFIED_TASKS", "1")
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mjlab.tasks  # noqa: E402,F401
import src.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.observation_manager import ObservationGroupCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from src.tasks.unified.students import UBlindStudentModel  # noqa: E402
from src.tasks.velocity.blind_env_cfg import STAIR_TERRAIN_NAMES  # noqa: E402
from src.tasks.velocity.depth_env_cfg import TEACHER_SCAN_NAME  # noqa: E402
from src.tasks.velocity.mdp.depth import DEPTH_UPDATE_PERIOD_STEPS  # noqa: E402

STALE_S, BLIND_FRAMES, HEALTHY_S, RECOVER_S = 0.5, 3, 1.0, 3.0
NO_READING = 1.0       # a depth pixel without a reading: max range, normalized (mdp.depth.preprocess_ray_depth)
REACH_BAR, SPEED_TOL = 0.90, 0.2
LIN_ACCEL, YAW_ACCEL = 2.0, 3.0
# accept_matrix.py's 44 commands (vx, vy, wz), in its order
CMDS = [
  (0.3, 0, 0), (0.5, 0, 0), (1.0, 0, 0), (1.5, 0, 0), (-0.5, 0, 0), (-1.0, 0, 0),
  (0, 0.3, 0), (0, -0.3, 0), (0, 0.6, 0), (0, -0.6, 0), (0.6, 0.4, 0), (0.6, -0.4, 0),
  (0, 0, 0.2), (0, 0, -0.2), (0, 0, 0.5), (0, 0, -0.5), (0, 0, 1.0), (0, 0, -1.0), (0, 0, 1.5), (0, 0, -1.5),
  (0.5, 0, 0.15), (0.5, 0, -0.15), (1.0, 0, 0.15), (1.0, 0, -0.15), (1.0, 0, 0.2), (1.0, 0, -0.2),
  (1.5, 0, 0.15), (1.5, 0, -0.15), (1.5, 0, 0.2), (1.5, 0, -0.2), (1.5, 0, 0.3), (1.5, 0, -0.3),
  (-0.5, 0, 0.2), (-0.5, 0, -0.2), (-1.0, 0, 0.2), (-1.0, 0, -0.2), (0.8, 0, 0.8), (0.8, 0, -0.8),
  (0.3, 0.3, 0.2), (0, 0.4, -0.3), (0.15, 0, 0.3), (0.2, 0, 0), (0, 0.15, 0), (1.0, 0, 0.1),
]
assert len(CMDS) == 44
STAIR_ROWS, STEP_RANGE = 4, (0.05, 0.18)   # eval_stairs.py's bands
PLATFORM_HALF, REACH_M = 1.0, 3.6


def add_blind_group(cfg) -> None:
  """The blind student's input, as its own training builds it (students.student_env_cfg, kind blind): the teacher's
  actor terms without the scan and the scan flag."""
  teacher = cfg.observations["actor"]
  g = deepcopy(teacher)
  g.terms = {k: deepcopy(v) for k, v in teacher.terms.items() if k not in (TEACHER_SCAN_NAME, "scan_flag")}
  cfg.observations["blind_actor"] = g


class Fallback:
  """The per-robot depth / blind switch (module docstring)."""

  def __init__(self, n: int, dt: float, fail_t: torch.Tensor, stall: torch.Tensor, recover: torch.Tensor):
    dev = fail_t.device
    self.dt, self.fail_t, self.stall, self.recover = dt, fail_t, stall, recover
    self.received = torch.zeros(n, dtype=torch.bool, device=dev)
    self.last_frame = torch.full((n,), -1e9, device=dev)
    self.invalid_run = torch.zeros(n, dtype=torch.long, device=dev)
    self.healthy = torch.zeros(n, device=dev)
    self.fallback = torch.ones(n, dtype=torch.bool, device=dev)
    self.takeover = torch.full((n,), -1.0, device=dev)   # the first takeover at or after the failure (s)
    self.handback = torch.full((n,), -1.0, device=dev)   # the depth student back after it (s)
    self.frozen = None

  def failing(self, t: float) -> torch.Tensor:
    return (t >= self.fail_t) & ~(self.recover & (t >= self.fail_t + RECOVER_S))

  def depth_seen(self, t: float, depth: torch.Tensor) -> torch.Tensor:
    """What the depth student is shown this step."""
    f = self.failing(t)
    if self.frozen is None:
      self.frozen = depth.clone()
    start = f & self.stall & (self.frozen_at_start(t))
    self.frozen = torch.where(start.view(-1, *([1] * (depth.dim() - 1))), depth, self.frozen)
    shown = torch.where((f & self.stall).view(-1, *([1] * (depth.dim() - 1))), self.frozen, depth)
    return torch.where((f & ~self.stall).view(-1, *([1] * (depth.dim() - 1))), torch.full_like(depth, NO_READING), shown)

  def frozen_at_start(self, t: float) -> torch.Tensor:
    # the step the stall begins: the frame then is the one kept
    return (t >= self.fail_t) & (t - self.dt < self.fail_t)

  def step(self, k: int, t: float) -> torch.Tensor:
    f = self.failing(t)
    tick = k % DEPTH_UPDATE_PERIOD_STEPS == 0
    got = torch.full_like(f, tick) & ~(f & self.stall)
    self.received |= got
    self.last_frame = torch.where(got, torch.full_like(self.last_frame, t), self.last_frame)
    blank = f & ~self.stall
    self.invalid_run = torch.where(got, torch.where(blank, self.invalid_run + 1, torch.zeros_like(self.invalid_run)),
                                   self.invalid_run)
    fault = ~self.received | (t - self.last_frame > STALE_S + 1e-6) | (self.invalid_run >= BLIND_FRAMES)
    was = self.fallback.clone()
    self.healthy = torch.where(fault, torch.zeros_like(self.healthy),
                               torch.where(self.fallback, self.healthy + self.dt, self.healthy))
    self.fallback = fault | (self.fallback & (self.healthy < HEALTHY_S - 1e-6))
    # the takeover: the first fault at or after the failure
    new_take = (t >= self.fail_t) & fault & (self.takeover < 0)
    self.takeover = torch.where(new_take, torch.full_like(self.takeover, t), self.takeover)
    back = (self.takeover >= 0) & was & ~self.fallback & (self.handback < 0)
    self.handback = torch.where(back, torch.full_like(self.handback, t), self.handback)
    return self.fallback


def build(task: str, n: int, stairs: bool, seed: int):
  cfg = load_env_cfg(task, play=True)
  cfg.seed = seed
  cfg.episode_length_s = 1000.0
  if stairs:
    gen = cfg.scene.terrain.terrain_generator
    gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k in STAIR_TERRAIN_NAMES}
    for v in gen.sub_terrains.values():
      v.proportion = 1.0
    gen.num_rows = STAIR_ROWS
    gen.num_cols = len(gen.sub_terrains) * 2
    cfg.events.pop("randomize_terrain", None)
  cfg.scene.num_envs = n
  add_blind_group(cfg)
  return cfg


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("robot")
  ap.add_argument("depth_checkpoint")
  ap.add_argument("blind_checkpoint")
  ap.add_argument("--part", choices=("flat", "stairs"), required=True)
  ap.add_argument("--scale", type=float, default=1.0, help="robot counts x this (smoke: 0.1)")
  ap.add_argument("--seed", type=int, default=0)
  a = ap.parse_args()
  stairs = a.part == "stairs"
  task = f"Unitree-{a.robot}-U-Student-Depth-{'Stairs' if stairs else 'Flat'}"
  pay = os.environ.get("U_PAYLOAD", "0,0,0,0")
  loaded = float(pay.split(",")[0]) > 0.0
  torch.manual_seed(a.seed)

  if stairs:
    names = [k for k in load_env_cfg(task, play=True).scene.terrain.terrain_generator.sub_terrains if k in STAIR_TERRAIN_NAMES]
    cols = len(names) * 2
    per_tile = max(1, round(36 * a.scale / (cols // 2)))   # 36 per band and direction over its narrow + wide tiles
    n = STAIR_ROWS * cols * per_tile
  else:
    per_cmd, n_stand = max(1, round(16 * a.scale)), max(2, round(96 * a.scale))
    n = len(CMDS) * per_cmd + n_stand
  cfg = build(task, n, stairs, a.seed)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  agent = load_rl_cfg(task)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
  runner.load(a.depth_checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  depth_policy = runner.get_inference_policy(device="cuda:0")
  obs = venv.get_observations()
  blind = UBlindStudentModel(obs, {"blind": ["blind_actor"]}, "blind", venv.num_actions, hidden_dims=(512, 256, 128),
                             obs_normalization=True,
                             distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"})
  sd = torch.load(a.blind_checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
  blind.load_state_dict(sd, strict=True)
  blind = blind.to("cuda:0").eval()

  dev, dt = env.device, env.step_dt
  robot = env.scene["robot"]
  terrain = env.scene.terrain
  cmd_term = env.command_manager.get_term("twist")
  tm = env.termination_manager

  # the scenario: commands, failure times and modes
  idx = torch.arange(n, device=dev)
  stall = idx % 2 == 0
  recover = (idx // 2) % 2 == 0
  if stairs:
    gen = cfg.scene.terrain.terrain_generator
    cum = np.cumsum(np.full(len(names), 1.0 / len(names)))
    col_name = [names[np.min(np.where(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
    tile = idx // per_tile
    levels, types = tile % STAIR_ROWS, tile // STAIR_ROWS
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = types
    terrain.env_origins[:] = terrain.terrain_origins[levels, types]
    env.reset()
    obs = venv.get_observations()
    C = torch.tensor([0.5, 0.0, 0.0], device=dev).expand(n, 3).clone()
    fail_t = 1.0 + 6.0 * torch.rand(n, device=dev)
    steps = int(round(REACH_M / 0.5 * 2.0 / dt))
    env_dir = np.array(["down" if "down" in col_name[t] else "up" for t in types.cpu().numpy()])
    env_level = levels.cpu().numpy()
  else:
    env.reset()
    obs = venv.get_observations()
    C = torch.zeros(n, 3, device=dev)
    C[: len(CMDS) * per_cmd] = torch.tensor(CMDS, device=dev, dtype=torch.float32).repeat_interleave(per_cmd, 0)
    fail_t = 2.0 + 3.0 * torch.rand(n, device=dev)
    steps = int(round(11.0 / dt))
  standing = (C.abs().sum(dim=1) == 0)
  planar = C[:, :2].norm(dim=1).clamp(min=1e-6)
  # the "speed" a robot is told: planar where it is told to move, else its yaw rate (turning in place)
  turning = ~standing & (C[:, :2].norm(dim=1) < 0.05)
  depth_policy.reset()
  fb = Fallback(n, dt, fail_t, stall, recover)

  alive = torch.ones(n, dtype=torch.bool, device=dev)
  fell_before = torch.zeros_like(alive)
  fell_after = torch.zeros_like(alive)
  reached = torch.zeros_like(alive)
  reached_before = torch.zeros_like(alive)
  where_take = torch.zeros(n, dtype=torch.long, device=dev)   # 0 none, 1 platform, 2 steps
  vel_win = torch.zeros(n, 3, device=dev)
  vel_n = torch.zeros(n, device=dev)
  min_ratio = torch.full((n,), float("inf"), device=dev)
  back80 = torch.full((n,), float("nan"), device=dev)
  rp_peak = torch.zeros(n, device=dev)
  pos_take = torch.zeros(n, 2, device=dev)
  creep = torch.zeros(n, device=dev)
  origin_xy = terrain.env_origins[:, :2].clone() if stairs else None

  alive_at_fail = torch.zeros(n, dtype=torch.bool, device=dev)   # standing when its camera failed: the robots tested
  for k in range(steps):
    t = k * dt
    alive_at_fail |= alive & (t >= fail_t) & (t - dt < fail_t)
    # ramped command, as the acceptance gives it
    lin = (LIN_ACCEL * dt * (k + 1) / planar).clamp(max=1.0)
    cmd_term.vel_command_b[:, :2] = C[:, :2] * lin.unsqueeze(1)
    cmd_term.vel_command_b[:, 2] = C[:, 2].clamp(-YAW_ACCEL * dt * (k + 1), YAW_ACCEL * dt * (k + 1))
    cmd_term.time_left[:] = 1e9
    fallback = fb.step(k, t)
    with torch.no_grad():
      shown = obs.clone(False)
      shown["depth"] = fb.depth_seen(t, obs["depth"])
      a_depth = depth_policy(shown)
      a_blind = blind(obs)
      act = torch.where(fallback.unsqueeze(1), a_blind, a_depth)
      obs, _, _, _ = venv.step(act)
    # a fall counts while the robot is still on its run (stairs: until it is past the last step, as eval_stairs.py)
    bad = tm.terminated & ~tm.time_outs & alive & ~reached
    after = t >= fail_t
    fell_before |= bad & ~after
    fell_after |= bad & after
    alive &= ~bad
    took = (fb.takeover >= 0) & alive_at_fail
    if stairs:
      cheb = (robot.data.root_link_pos_w[:, :2] - origin_xy).abs().max(dim=1).values
      newly = alive & ~reached & (cheb >= REACH_M)
      reached_before |= newly & ~took
      reached |= newly
      just = took & (where_take == 0) & (torch.abs(fb.takeover - t) < 0.5 * dt)
      where_take = torch.where(just, torch.where((cheb >= PLATFORM_HALF) & (cheb < REACH_M), 2, 1), where_take)
    # readings after the takeover
    since = t - fb.takeover
    v = torch.cat([robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]], dim=1)
    win = took & alive & (since >= 2.0) & (since < 3.0)
    vel_win += torch.where(win.unsqueeze(1), v, torch.zeros_like(v))
    vel_n += win.float()
    ratio = torch.where(standing, torch.ones_like(planar),
                        torch.where(turning, v[:, 2] / torch.where(C[:, 2] == 0, torch.ones_like(planar), C[:, 2]),
                                    v[:, :2].norm(dim=1) / planar))
    first = took & alive & (since >= 0) & (since < 1.0)
    min_ratio = torch.where(first, torch.minimum(min_ratio, ratio), min_ratio)
    rec = took & alive & (since >= 0) & torch.isnan(back80) & (ratio >= 0.8) & (since > 0)
    back80 = torch.where(rec, since, back80)
    g = robot.data.projected_gravity_b
    rp = torch.atan2(torch.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2), -g[:, 2])
    rp_peak = torch.where(took & alive & (since >= 0), torch.maximum(rp_peak, rp), rp_peak)
    at_take = took & (torch.abs(since) < 0.5 * dt)
    pos_take = torch.where(at_take.unsqueeze(1), robot.data.root_link_pos_w[:, :2], pos_take)
    hold = took & alive & standing & (since > 0) & (since <= 4.0)
    creep = torch.where(hold, (robot.data.root_link_pos_w[:, :2] - pos_take).norm(dim=1), creep)

  # ---- report and verdict ----
  took = ((fb.takeover >= 0) & alive_at_fail).cpu().numpy()
  fa, fbf = fell_after.cpu().numpy(), fell_before.cpu().numpy()
  stall_np, rec_np = stall.cpu().numpy(), recover.cpu().numpy()
  cond = "loaded" if loaded else "unloaded"
  print(f"\ncamera_fail {a.robot} {a.part} {cond} (U_PAYLOAD {pay}): depth {a.depth_checkpoint} + blind {a.blind_checkpoint}")
  print(f"robots {n}; standing at their failure and taken over {took.sum()}; handed back "
        f"{((fb.handback >= 0) & alive_at_fail).sum().item()} (of {int((rec_np & took).sum())} of those that recover)")
  delay = (fb.takeover - fail_t).cpu().numpy()
  for m, nm in ((stall_np, "stall"), (~stall_np, "blind")):
    sel = m & took
    if sel.any():
      print(f"  {nm}: takeover {np.median(delay[sel]):.2f} s after the failure (min {delay[sel].min():.2f}, max {delay[sel].max():.2f})")
  print(f"falls after the failure {int(fa.sum())} / {n} (stall {int((fa & stall_np).sum())}, blind {int((fa & ~stall_np).sum())}; "
        f"recovering {int((fa & rec_np).sum())}, failed to the end {int((fa & ~rec_np).sum())}); before it {int(fbf.sum())}")
  bad = []
  # every fall fails it: before the failure the pair drives too (the blind student for the first second, then the
  # depth one) -- and a robot down before its failure was never tested, so a pass must not rest on it
  if fa.sum() + fbf.sum() > 0:
    bad.append(f"摔倒 {int(fa.sum() + fbf.sum())}/{n}（失效前 {int(fbf.sum())}，失效后 {int(fa.sum())}）")
  mr = min_ratio.cpu().numpy()
  b80 = back80.cpu().numpy()
  rpd = np.degrees(rp_peak.cpu().numpy())
  if stairs:
    wt, rch, rb = where_take.cpu().numpy(), reached.cpu().numpy(), reached_before.cpu().numpy()
    on_steps = took & (wt == 2)
    print(f"{'band':>9} {'dir':>4} | {'on steps':>8} {'reach':>6} | {'platform':>8} {'reach':>6} | {'fell':>4} | done before")
    for lvl in range(STAIR_ROWS):
      lo, hi = STEP_RANGE
      band = f"{100 * (lo + (hi - lo) * lvl / STAIR_ROWS):.0f}-{100 * (lo + (hi - lo) * (lvl + 1) / STAIR_ROWS):.0f}cm"
      for d in ("up", "down"):
        m = (env_level == lvl) & (env_dir == d)
        s, p = m & on_steps, m & took & (wt == 1)
        rs = rch[s].mean() if s.any() else float("nan")
        rp_ = rch[p].mean() if p.any() else float("nan")
        print(f"{band:>9} {d:>4} | {int(s.sum()):8d} {rs:6.0%} | {int(p.sum()):8d} {rp_:6.0%} | {int(fa[m].sum()):4d} | {int(rb[m].sum())}")
    rate = rch[on_steps].mean() if on_steps.any() else float("nan")
    print(f"took over on the steps: {int(on_steps.sum())}, reached {rate:.1%} (bar {REACH_BAR:.0%}, judged unloaded)")
    if not loaded and (not on_steps.any() or not rate >= REACH_BAR):
      bad.append(f"台阶上切换后到达率 {rate:.1%}")
  else:
    vm = (vel_win / vel_n.clamp(min=1).unsqueeze(1)).cpu().numpy()
    vn = vel_n.cpu().numpy()
    fails = []
    print(f"{'command (vx, vy, wz)':>22} | {'2-3 s after the takeover':>24} | falls | lowest in 1 s | back to 80% s")
    for ci, c in enumerate(CMDS):
      sl = slice(ci * per_cmd, (ci + 1) * per_cmd)
      ok = (vn[sl] > 0) & ~fa[sl]
      mean = vm[sl][ok].mean(0) if ok.any() else np.full(3, np.nan)
      parts, cb = [], False
      for axis, nm in enumerate("xyw"):
        if abs(c[axis]) >= 0.05:
          r = mean[axis] / c[axis]
          parts.append(f"{nm}{r * 100:.0f}%")
          cb |= not (abs(r - 1.0) <= SPEED_TOL)
      if cb:
        fails.append(str(c))
      print(f"{str(c):>22} | {' '.join(parts):>24} | {int(fa[sl].sum()):2d}/{per_cmd} | "
            f"{np.nanmin(np.where(np.isfinite(mr[sl]), mr[sl], np.nan)) if np.isfinite(mr[sl]).any() else float('nan'):5.0%} | "
            f"{np.nanmedian(b80[sl]) if np.isfinite(b80[sl]).any() else float('nan'):.2f}")
    st = standing.cpu().numpy() & took
    cr = creep.cpu().numpy()[st]
    print(f"standing: {int(st.sum())} robots, creep over the 4 s after the takeover median {100 * np.median(cr) if cr.size else float('nan'):.1f} cm, "
          f"p90 {100 * np.quantile(cr, 0.9) if cr.size else float('nan'):.1f} cm (reported)")
    print(f"speed 2 s after the takeover: {len(CMDS) - len(fails)} of {len(CMDS)} commands at 80-120% (judged unloaded)")
    if not loaded and fails:
      bad.append(f"切换后速度不在 80–120%：{len(fails)} 条指令（{'; '.join(fails[:6])}{' …' if len(fails) > 6 else ''}）")
  sel = took & np.isfinite(rpd)
  print(f"roll / pitch peak after the takeover: median {np.median(rpd[sel]) if sel.any() else float('nan'):.1f} deg, "
        f"max {rpd[sel].max() if sel.any() else float('nan'):.1f} deg (reported)")
  print("camera_fail verdict: " + ("PASS" if not bad else "FAIL: " + "；".join(bad)))
  env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
