#!/usr/bin/env python3
"""reward_attribution.py --scene SCENE --task <play task> <checkpoint> (go2-depth 2026-11-17, go2-home's round-1
diagnostic): which reward terms pay for a failing behaviour. Some of the 68 terms may push the policy towards what an
acceptance item fails on; this probe runs one test scene, reads every reward term's per-step value per robot (the
reward manager's _step_reward: raw value x weight, the rate before dt) and the termination terms, and compares the
terms over the "bad" windows (where the failure happens) with the "good" ones (the same phase in the robots that pass).

Scenes (each robot in one cell; payload from U_PAYLOAD, scan mode from U_SCAN, as the other unified tests):
  stairs_stop   the stair tiles of eval_stairs / descent_probe (4 bands x up / down x narrow / wide, spawned square to
                the steps), forward VX from the tile centre; at STOP_D of Chebyshev distance (mid-flight) the command
                goes to zero for STOP_S. Bad: a robot that moves more than SLIDE_M along its heading after the first
                STOP_SETTLE_S of the stop (the braking; slide-back on the way up, creep on the way down) or falls in
                the stop; good: the robots that hold. The window: the stop after STOP_SETTLE_S. -Stairs task.
  slope         slope_rough_test's 15 deg ramps (up and down), walked at VX. Bad: the FALL_S before a fall; good: the
                robots that never fall, on the ramp (the 6 m run). -Stairs task.
  climb         the climbing stair tiles (4 bands x narrow / wide, spawned square to the steps), forward VX from the
                tile centre to past the last step (CLIMB_END m of Chebyshev distance), heading held (go2-home
                2026-11-17: B1 loses map3 reach 95 -> 74 % from 3000 to 5900, slower climbs, idling at the flights'
                ends). Bad: the robot-steps while a robot stalls -- no CLIMB_PROGRESS_M of progress for CLIMB_STALL_S
                before it is through -- or has turned back (CLIMB_BACK_M below its furthest point), and the falls;
                good: the climbing robot-steps of the robots that go through without a stall, turn or fall. --rows
                picks the bands judged (default 3: 15-18 cm); every band's counts and stall places are printed. -Stairs.
  flat_reverse  flat ground, the command (-VX, 0, 0) (default 1.0: w27d's full-load fall case) after a 1 s stand,
                slewed at 2 m/s^2, for SECONDS. Bad: the FALL_S before a fall; good: the robots that never fall, from
                2 s on. -Flat task.
  (The wall probe's landing: map3_stairs_probe.py --attribution OUT, the same table -- its guard drive-up lives there.)
Falls in every scene also count as bad windows (the FALL_S before the termination), with their termination terms.

Per term: the mean over the good and over the bad robot-steps, bad - good, each window's share of the sum of the
terms' absolute means, and the push: bad - good > 0 means the term pays more (or costs less) while the robot fails --
it rewards the failure; < 0 -- it punishes it. Sorted by bad - good, highest first. --out FILE also writes the table
as JSON. Judges nothing.

  python jobs/unified/u_eval.py scripts/unified/reward_attribution.py --scene stairs_stop \\
      --task Unitree-Go2W-U-Teacher-Stairs logs/avg/<run>_avg4/model_3000.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "go2_blind"))
sys.path.insert(0, str(ROOT / "scripts" / "unified"))
os.environ.setdefault("UNIFIED_TASKS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

NUM_ROWS = 4
STOP_D = 2.2       # m of Chebyshev distance from the tile centre: mid-flight (the flight runs 1.0-3.6 m)
STOP_S = 3.0
STOP_SETTLE_S = 1.0   # the braking from VX: the stop window and the movement are measured after it
SLIDE_M = 0.05
FALL_S = 1.0
LIN_ACCEL = 2.0
SETTLE_S = 1.0
CLIMB_END = 3.6        # m of Chebyshev distance: past the last step (eval_stairs' traverse)
CLIMB_PROGRESS_M = 0.05
CLIMB_STALL_S = 2.0
CLIMB_BACK_M = 0.3
STEP_RANGE = (0.05, 0.18)   # (eval_stairs' bands: row r of NUM_ROWS spans lo + (hi - lo) * r / NUM_ROWS ...)


def attribution_rows(names, weights, good_sum, good_n, bad_sum, bad_n) -> list[dict]:
  """The table: per term (weight != 0, acting in either window) the good / bad means over their robot-steps, bad - good,
  each window's share of the sum of the terms' absolute means and the push; sorted by bad - good, highest first. Also
  map3_stairs_probe.py --attribution's (the landing scene)."""
  g = np.asarray(good_sum, dtype=float) / max(float(good_n), 1.0)
  b = np.asarray(bad_sum, dtype=float) / max(float(bad_n), 1.0)
  gs, bs = np.abs(g).sum() or 1.0, np.abs(b).sum() or 1.0
  rows = []
  for i, nm in enumerate(names):
    w = float(weights[i])
    if w == 0.0 or (g[i] == 0.0 and b[i] == 0.0):   # (a term that never acted in either window)
      continue
    d = float(b[i] - g[i])
    rows.append({"term": nm, "weight": w, "good": float(g[i]), "bad": float(b[i]), "bad_minus_good": d,
                 "share_good": float(abs(g[i]) / gs), "share_bad": float(abs(b[i]) / bs),
                 "push": "rewards the failure" if d > 0 else "punishes it" if d < 0 else "-"})
  rows.sort(key=lambda x: -x["bad_minus_good"])
  return rows


def print_rows(rows: list[dict], top: int) -> None:
  print(f"{'term':34s} {'weight':>9s} {'good':>9s} {'bad':>9s} {'bad-good':>9s} {'sh.good':>7s} {'sh.bad':>7s}  push")
  show = rows if len(rows) <= top + 5 else rows[:top]
  for r in show:
    print(f"{r['term'][:34]:34s} {r['weight']:9.3g} {r['good']:9.4f} {r['bad']:9.4f} {r['bad_minus_good']:+9.4f} "
          f"{100 * r['share_good']:6.1f}% {100 * r['share_bad']:6.1f}%  {r['push']}")
  if len(rows) > len(show):
    print(f"... {len(rows) - top - 5} more; the most punishing:")
    for r in rows[-5:]:
      print(f"{r['term'][:34]:34s} {r['weight']:9.3g} {r['good']:9.4f} {r['bad']:9.4f} {r['bad_minus_good']:+9.4f} "
            f"{100 * r['share_good']:6.1f}% {100 * r['share_bad']:6.1f}%  {r['push']}")


class Attribution:
  """Per-term sums over bad and good robot-steps; a ring buffer of each robot's last FALL_S of step rewards for the
  fall windows; per-robot window sums for the scenes whose verdict comes at the end."""

  def __init__(self, env, dt: float):
    self.rm = env.reward_manager
    self.tm = env.termination_manager
    self.names = list(self.rm.active_terms)
    n, T, dev = env.num_envs, len(self.names), env.device
    self.W = max(1, int(round(FALL_S / dt)))
    self.ring = torch.zeros(n, self.W, T, device=dev)
    self.filled = torch.zeros(n, dtype=torch.long, device=dev)
    self.k = 0
    self.bad = torch.zeros(T, device=dev)
    self.bad_n = 0.0
    self.good = torch.zeros(T, device=dev)
    self.good_n = 0.0
    self.win = torch.zeros(n, T, device=dev)     # per-robot window sums (the scene decides good / bad at the end)
    self.win_n = torch.zeros(n, device=dev)
    self.causes: dict[str, int] = {}
    self.falls = 0

  def step(self, terminated: torch.Tensor, first: torch.Tensor, window: torch.Tensor) -> None:
    """After env.step: the step's rewards into the ring; a robot's first termination (`first`) adds its last FALL_S
    to the bad sums; `window` robots add this step to their window sums."""
    r = self.rm._step_reward.detach()
    self.ring[:, self.k % self.W] = r
    self.k += 1
    self.filled = torch.clamp(self.filled + 1, max=self.W)
    ids = first.nonzero(as_tuple=False).flatten()
    if len(ids):
      self.falls += len(ids)
      for i in ids.tolist():
        m = int(self.filled[i])
        self.bad += self.ring[i].sum(dim=0)   # (rows not yet written are 0)
        self.bad_n += m
      for nm in self.tm.active_terms:
        if self.tm.get_term_cfg(nm).time_out:
          continue
        c = int(self.tm.get_term(nm)[ids].sum())
        if c:
          self.causes[nm] = self.causes.get(nm, 0) + c
    w = window.float()
    self.win += r * w[:, None]
    self.win_n += w

  def add_windows(self, bad: torch.Tensor, good: torch.Tensor) -> None:
    """The per-robot window sums into the bad / good sums (bad, good: robot masks)."""
    self.bad += self.win[bad].sum(dim=0)
    self.bad_n += float(self.win_n[bad].sum())
    self.good += self.win[good].sum(dim=0)
    self.good_n += float(self.win_n[good].sum())

  def table(self) -> list[dict]:
    return attribution_rows(self.names, [self.rm.get_term_cfg(nm).weight for nm in self.names],
                            self.good.cpu().numpy(), self.good_n, self.bad.cpu().numpy(), self.bad_n)


def build(a):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = load_env_cfg(a.task, play=True)
  cfg.seed = a.seed
  cfg.events.pop("push_robot", None)
  extra = {}
  if a.scene in ("stairs_stop", "climb"):
    from src.tasks.velocity.blind_env_cfg import STAIR_TERRAIN_NAMES

    gen = cfg.scene.terrain.terrain_generator
    keep = [n for n in STAIR_TERRAIN_NAMES if a.scene == "stairs_stop" or "up" in n]
    gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k in keep}
    for v in gen.sub_terrains.values():
      v.proportion = 1.0
    gen.num_rows = NUM_ROWS
    gen.num_cols = len(gen.sub_terrains) * 2
    cfg.events.pop("randomize_terrain", None)
    cfg.events["reset_base"].params["stair_yaw_noise"] = 1e-6
    cfg.events["reset_base"].params["narrow_yaw_noise"] = 1e-6
    n = gen.num_rows * gen.num_cols * a.envs_per_cell
  elif a.scene == "slope":
    import slope_rough_test as srt

    gen = cfg.scene.terrain.terrain_generator
    gen.size = (srt.TILE, srt.TILE)
    gen.curriculum = True
    gen.sub_terrains = {"slope15_up": srt.RampTerrainCfg(proportion=1.0, down=False),
                        "slope15_down": srt.RampTerrainCfg(proportion=1.0, down=True)}
    gen.num_rows, gen.num_cols = 2, 2
    cfg.events.pop("randomize_terrain", None)
    rb = cfg.events["reset_base"]
    rb.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-1.0, 1.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
    rb.params["velocity_range"] = {}
    n = 2 * a.envs_per_cell
    extra["srt"] = srt
  else:   # flat_reverse
    gen = cfg.scene.terrain.terrain_generator
    if gen is not None:
      gen.size = (120.0, 120.0)
      gen.num_rows = gen.num_cols = 1
    n = a.envs_per_cell
  cfg.scene.num_envs = n
  cfg.episode_length_s = 1e6
  agent = load_rl_cfg(a.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(a.task)(venv, asdict(agent), device="cuda:0")
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return cfg, env, venv, runner.get_inference_policy(device="cuda:0"), extra


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--scene", required=True, choices=("stairs_stop", "climb", "slope", "flat_reverse"))
  ap.add_argument("--task", required=True)
  ap.add_argument("checkpoint")
  ap.add_argument("--envs-per-cell", type=int, default=None,
                  help="robots per tile (stairs_stop: 16 per band x direction x tread; climb 32; slope: 128 per ramp; flat: 256)")
  ap.add_argument("--rows", default="3", help="climb: the bands judged (rows 0-3 = 5-8 / 8-12 / 12-15 / 15-18 cm)")
  ap.add_argument("--vx", type=float, default=None, help="m/s (stairs_stop 0.5, slope 1.0, flat_reverse 1.0)")
  ap.add_argument("--seconds", type=float, default=10.0, help="flat_reverse: the reverse's length")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--top", type=int, default=20, help="table rows printed (every term in --out)")
  ap.add_argument("--out", default=None, help="the table as JSON")
  a = ap.parse_args()
  a.envs_per_cell = a.envs_per_cell or {"stairs_stop": 16, "climb": 32, "slope": 128, "flat_reverse": 256}[a.scene]
  a.vx = a.vx or {"stairs_stop": 0.5, "climb": 0.5, "slope": 1.0, "flat_reverse": 1.0}[a.scene]
  cfg, env, venv, policy, extra = build(a)
  dev, dt, n = env.device, env.step_dt, env.num_envs
  robot = env.scene["robot"]
  terrain = env.scene.terrain
  cmd = env.command_manager.get_term("twist")
  import eval_map3 as em

  em.set_command(env, 0.0, 0.0, 0.0)
  att = Attribution(env, dt)
  info = ""
  if a.scene in ("stairs_stop", "climb"):
    gen = cfg.scene.terrain.terrain_generator
    names = list(gen.sub_terrains.keys())
    cum = np.cumsum(np.full(len(names), 1.0 / len(names)))
    col = [names[np.min(np.where(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
    tile = torch.arange(n, device=dev) // a.envs_per_cell
    levels, types = tile % gen.num_rows, tile // gen.num_rows
    up = torch.tensor(["down" not in col[t] for t in types.cpu().numpy()], device=dev)
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = types
    terrain.env_origins[:] = terrain.terrain_origins[levels, types]
  elif a.scene == "slope":
    gen = cfg.scene.terrain.terrain_generator
    names = list(gen.sub_terrains.keys())
    cum = np.cumsum(np.full(len(names), 1.0 / len(names)))
    col_of = {names[int(np.min(np.where(c / gen.num_cols + 0.001 < cum)[0]))]: c for c in range(gen.num_cols)}
    down = torch.arange(n, device=dev) >= a.envs_per_cell
    types = torch.where(down, torch.full((n,), col_of["slope15_down"], device=dev), torch.full((n,), col_of["slope15_up"], device=dev))
    levels = torch.arange(n, device=dev) % gen.num_rows
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = types
    terrain.env_origins[:] = terrain.terrain_origins[levels, types]
  env.reset()
  obs = venv.get_observations()
  yaw0 = robot.data.heading_w.clone()
  head = torch.stack((torch.cos(yaw0), torch.sin(yaw0)), dim=1)
  p0 = robot.data.root_link_pos_w[:, :2].clone()
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  cur = torch.zeros(n, device=dev)
  if a.scene == "climb":
    judged = torch.zeros(n, dtype=torch.bool, device=dev)
    for r_ in (int(x) for x in a.rows.split(",")):
      judged |= levels == r_
    best = torch.zeros(n, device=dev)        # the furthest Chebyshev distance so far
    ref = torch.zeros(n, device=dev)         # the distance at the last progress
    t_ref = torch.zeros(n, device=dev)
    through = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_stall = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_back = torch.zeros(n, dtype=torch.bool, device=dev)
    stall_d = torch.full((n,), float("nan"), device=dev)   # where a robot's first stall began (its best distance)
    stall_steps = torch.zeros(n, device=dev)
    T = int(math.ceil((SETTLE_S + CLIMB_END / a.vx * 2.5 + 4.0) / dt))
  if a.scene == "stairs_stop":
    stop_k = torch.full((n,), -1, dtype=torch.long, device=dev)   # the step the stop began
    p_stop = torch.zeros(n, 2, device=dev)
    moved = torch.zeros(n, device=dev)
    fell_stop = torch.zeros(n, dtype=torch.bool, device=dev)
    T = int(math.ceil((SETTLE_S + 3.6 / a.vx * 2.0 + STOP_S) / dt))
  elif a.scene == "slope":
    srt = extra["srt"]
    T = int(math.ceil((SETTLE_S + srt.JUDGED / a.vx * 1.5 + 2.0) / dt))
  else:
    T = int(math.ceil((SETTLE_S + a.seconds) / dt))
  stop_steps = int(round(STOP_S / dt))
  import src.tasks.velocity.mdp as _mdp
  from mjlab.managers.scene_entity_config import SceneEntityCfg as _SEC
  from src.tasks.velocity.blind_env_cfg import FOOT_SITE_NAMES as _FSN
  feet_cfg = _SEC("robot", site_names=_FSN)
  feet_cfg.resolve(env.scene)
  um = [0.0, 0.0, 0.0]   # pose's uneven mask in the scene's window: steps, the 0.45 m circle, the feet (U_POSE_UNEVEN)
  # the slope's whole ramp (go2-home 2026-11-17: do the upright-leg costs fight the ramp before the tip?): every robot's
  # term sums per ramp third (entry / mid / exit), each step added FALL_S late so a fall's last FALL_S never counts;
  # at the end the robots that fell against those that never did
  T_ = len(att.names)
  W_ = att.W
  ramp_sum = torch.zeros(n, 3, T_, device=dev)
  ramp_n = torch.zeros(n, 3, device=dev)
  dl_r = torch.zeros(n, W_, T_, device=dev)
  dl_s = torch.full((n, W_), -1, dtype=torch.long, device=dev)   # the ramp third of each delayed step (-1: none)
  settle_steps = int(round(STOP_SETTLE_S / dt))
  for k in range(T):
    t = k * dt
    rel = robot.data.root_link_pos_w[:, :2] - p0
    along = (rel * head).sum(dim=1)
    if a.scene == "stairs_stop":
      d = (robot.data.root_link_pos_w[:, :2] - terrain.env_origins[:, :2]).abs().max(dim=1).values
      start = (stop_k < 0) & (d >= STOP_D) & ~fell
      stop_k = torch.where(start, torch.full_like(stop_k, k), stop_k)
      p_stop = torch.where(start[:, None], robot.data.root_link_pos_w[:, :2], p_stop)
      stopping = (stop_k >= 0) & (k - stop_k < stop_steps)
      settled = stopping & (k - stop_k >= settle_steps)
      at_ref = (stop_k >= 0) & (k - stop_k == settle_steps)
      p_stop = torch.where(at_ref[:, None], robot.data.root_link_pos_w[:, :2], p_stop)   # the reference after the braking
      going = (t >= SETTLE_S) & (stop_k < 0) & ~fell
      want = torch.where(going, torch.full((n,), a.vx, device=dev), torch.zeros(n, device=dev))
      vy = torch.zeros(n, device=dev)
      window = settled & ~fell
    elif a.scene == "climb":
      d = (robot.data.root_link_pos_w[:, :2] - terrain.env_origins[:, :2]).abs().max(dim=1).values
      through |= d >= CLIMB_END
      going = (t >= SETTLE_S) & ~through & ~fell
      best = torch.maximum(best, d)
      moved_on = d > ref + CLIMB_PROGRESS_M
      ref = torch.where(moved_on | (t < SETTLE_S), d, ref)
      t_ref = torch.where(moved_on | (t < SETTLE_S), torch.full_like(t_ref, t), t_ref)
      stalled = going & (t - t_ref >= CLIMB_STALL_S)
      back = going & (d < best - CLIMB_BACK_M)
      new_stall = stalled & ~ever_stall
      stall_d = torch.where(new_stall, best, stall_d)
      ever_stall |= stalled
      ever_back |= back
      bad_now = (stalled | back) & judged
      stall_steps += (stalled | back).float()
      want = torch.where(going, torch.full((n,), a.vx, device=dev), torch.zeros(n, device=dev))
      vy = torch.zeros(n, device=dev)
      window = going & ~stalled & ~back & judged   # (good: kept for the robots that go through clean)
    elif a.scene == "slope":
      going = (t >= SETTLE_S) & (along < srt.JUDGED) & ~fell
      want = torch.where(going, torch.full((n,), a.vx, device=dev), torch.zeros(n, device=dev))
      lateral = -head[:, 1] * rel[:, 0] + head[:, 0] * rel[:, 1]
      vy = torch.where(going, (-lateral).clamp(-0.2, 0.2), torch.zeros(n, device=dev))
      on_ramp = (along >= srt.START / 2) & (along <= srt.START / 2 + srt.RUN)
      window = going & on_ramp
    else:
      going = (t >= SETTLE_S) & ~fell
      want = torch.where(going, torch.full((n,), -a.vx, device=dev), torch.zeros(n, device=dev))
      vy = torch.zeros(n, device=dev)
      window = (t >= SETTLE_S + 1.0) & ~fell
    step_v = LIN_ACCEL * dt
    cur = torch.where(want == 0, want, cur + (want - cur).clamp(-step_v, step_v))
    yaw = robot.data.heading_w
    herr = torch.atan2(torch.sin(yaw0 - yaw), torch.cos(yaw0 - yaw))
    cmd.vel_command_b[:, 0] = cur
    cmd.vel_command_b[:, 1] = vy
    cmd.vel_command_b[:, 2] = torch.where(going, herr.clamp(-0.5, 0.5), torch.zeros(n, device=dev))
    cmd.time_left[:] = 1e9
    with torch.no_grad():
      obs, _, dones, extras = venv.step(policy(obs))
    terminated = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
    first = terminated & ~fell
    if a.scene == "stairs_stop":
      fell_stop |= first & stopping
      sp = robot.data.root_link_pos_w[:, :2] - p_stop
      moved = torch.where(settled & ~at_ref & ~fell & ~first, (sp * head).sum(dim=1).abs().maximum(moved), moved)
    if a.scene == "climb":
      att.step(terminated, first & judged, window & ~first)
      bm = bad_now & ~first
      if bool(bm.any()):
        att.bad += att.rm._step_reward.detach()[bm].sum(dim=0)
        att.bad_n += float(bm.sum())
    else:
      att.step(terminated, first, window & ~first)
    if a.scene == "slope":
      j = k % W_
      old_s, old_r = dl_s[:, j].clone(), dl_r[:, j].clone()
      ok_old = (old_s >= 0) & ~fell & ~first   # (a robot down by now: its delayed steps are its last FALL_S)
      ramp_sum += torch.nn.functional.one_hot(old_s.clamp(min=0), 3).float()[:, :, None] * old_r[:, None, :] * ok_old.float()[:, None, None]
      ramp_n += torch.nn.functional.one_hot(old_s.clamp(min=0), 3).float() * ok_old.float()[:, None]
      third = ((along - srt.START / 2) / (srt.RUN / 3)).floor().long().clamp(0, 2)
      on = on_ramp & ~fell & ~first
      dl_s[:, j] = torch.where(on, third, torch.full_like(third, -1))
      dl_r[:, j] = att.rm._step_reward.detach()
    if a.scene != "flat_reverse":
      wm = window & ~first
      um[0] += float(wm.sum())
      um[1] += float((wm & _mdp.uneven_terrain_mask(env, "terrain_scan", 0.45, 0.05)).sum())
      um[2] += float((wm & _mdp.uneven_feet_mask(env, "terrain_scan", feet_cfg, 0.05)).sum())
    fell |= terminated
  if a.scene == "climb":
    clean = through & ~ever_stall & ~ever_back & ~fell & judged
    att.add_windows(torch.zeros(n, dtype=torch.bool, device=dev), clean)
    lo_, hi_ = STEP_RANGE
    lines = []
    for r_ in range(gen.num_rows):
      m_ = levels == r_
      band = f"{100 * (lo_ + (hi_ - lo_) * r_ / gen.num_rows):.0f}-{100 * (lo_ + (hi_ - lo_) * (r_ + 1) / gen.num_rows):.0f}cm"
      sd = stall_d[m_ & ever_stall]
      q = (lambda x: " / ".join(f"{float(v):.2f}" for v in torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], device=dev))) if len(x) else "-")
      lines.append(f"  {band:>9} {'*' if r_ in [int(x) for x in a.rows.split(',')] else ' '} robots {int(m_.sum()):4d}: through "
                   f"{int((m_ & through).sum()):4d}, clean {int((m_ & through & ~ever_stall & ~ever_back & ~fell).sum()):4d}, stalled "
                   f"{int((m_ & ever_stall).sum()):4d}, turned back {int((m_ & ever_back).sum()):4d}, fell {int((m_ & fell).sum()):4d}; "
                   f"first stall at the furthest distance p10 / p50 / p90 {q(sd)} m (flight {1.0:.1f}-{CLIMB_END:.1f})")
    info = (f"climb at {a.vx} m/s, bands judged {a.rows} (*); stall = no {CLIMB_PROGRESS_M} m of progress for {CLIMB_STALL_S:g} s, "
            f"turned back = {CLIMB_BACK_M} m below the furthest point\n" + "\n".join(lines) +
            f"\n  judged: bad = stalled / turned-back robot-steps + falls, good = the clean robots' climbing steps ({int(clean.sum())} robots)")
  if a.scene == "stairs_stop":
    stopped = stop_k >= 0
    bad = stopped & ((moved > SLIDE_M) | fell_stop)
    good = stopped & ~bad & ~fell
    att.add_windows(bad, good)
    info = (f"stops {int(stopped.sum())} of {n} (up {int((stopped & up).sum())}, down {int((stopped & ~up).sum())}); bad "
            f"{int(bad.sum())} (moved > {SLIDE_M} m after {STOP_SETTLE_S:g} s: {int((stopped & (moved > SLIDE_M)).sum())}, fell in the stop: "
            f"{int(fell_stop.sum())}; up {int((bad & up).sum())}, down {int((bad & ~up).sum())}); good {int(good.sum())}")
  elif a.scene != "climb":
    att.add_windows(torch.zeros(n, dtype=torch.bool, device=dev), ~fell)
    info = f"robots {n}; never fell {int((~fell).sum())}"
  if a.scene == "slope":
    for j in range(W_):   # the never-fallen robots' last delayed steps count
      s_, r_ = dl_s[:, j], dl_r[:, j]
      ok_ = (s_ >= 0) & ~fell
      ramp_sum += torch.nn.functional.one_hot(s_.clamp(min=0), 3).float()[:, :, None] * r_[:, None, :] * ok_.float()[:, None, None]
      ramp_n += torch.nn.functional.one_hot(s_.clamp(min=0), 3).float() * ok_.float()[:, None]
    ramp_lines = []
    for nm in ("stance_leg_tilt", "pose", "body_orientation", "com_support", "track_linear_velocity", "standing_feet_contact"):
      if nm not in att.names:
        ramp_lines.append(f"  {nm:22s} (not in this build)")
        continue
      i = att.names.index(nm)
      cells = []
      for s_i, s_nm in enumerate(("entry", "mid", "exit")):
        nf = ramp_n[~fell, s_i].sum().clamp(min=1)
        ff = ramp_n[fell, s_i].sum()
        mn = float(ramp_sum[~fell, s_i, i].sum() / nf)
        mf = float(ramp_sum[fell, s_i, i].sum() / ff.clamp(min=1)) if ff > 0 else float("nan")
        cells.append(f"{s_nm} {mn:+.3f} / {mf:+.3f}")
      ramp_lines.append(f"  {nm:22s} " + " | ".join(cells))
    counts = " | ".join(f"{s_nm} {int(ramp_n[~fell, s_i].sum())} / {int(ramp_n[fell, s_i].sum())}"
                        for s_i, s_nm in enumerate(("entry", "mid", "exit")))
  rows = att.table()
  print(f"checkpoint {a.checkpoint}\ntask {a.task}, scene {a.scene}, vx {a.vx}, payload {os.environ.get('U_PAYLOAD', '(task default)')}, "
        f"scan {os.environ.get('U_SCAN', '(task default)')}")
  print(f"{info}; falls {att.falls} (their last {FALL_S:g} s are bad windows): causes {att.causes or '-'}")
  print(f"robot-steps: bad {att.bad_n:.0f}, good {att.good_n:.0f}")
  if a.scene == "slope":
    print(f"the whole ramp before any tip (each fall's last {FALL_S:g} s left out), never fell / later fell, the terms' "
          f"means per ramp third (robot-steps {counts}):")
    print("\n".join(ramp_lines))
  if um[0]:
    print(f"pose's uneven mask in the window ({um[0]:.0f} robot-steps): the 0.45 m circle {100 * um[1] / um[0]:.1f} %, "
          f"the ground under the feet {100 * um[2] / um[0]:.1f} %")
  if att.bad_n == 0 or att.good_n == 0:
    print(f"no {'bad' if att.bad_n == 0 else 'good'} windows in this run: nothing to compare (more robots, or another scene)")
    if a.out:
      Path(a.out).write_text(json.dumps({"scene": a.scene, "checkpoint": a.checkpoint, "info": info, "falls": att.falls,
                                         "causes": att.causes, "bad_n": att.bad_n, "good_n": att.good_n, "rows": []}, indent=1))
    env.close()
    return 0
  print_rows(rows, a.top)
  if a.out:
    Path(a.out).write_text(json.dumps({"scene": a.scene, "checkpoint": a.checkpoint, "info": info, "falls": att.falls,
                                       "causes": att.causes, "bad_n": att.bad_n, "good_n": att.good_n, "rows": rows}, indent=1))
  env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
