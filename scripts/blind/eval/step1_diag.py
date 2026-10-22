"""Step-1 diagnostics of the final blind models (evaluation only, nothing trained).

  map3fail   : why robots fail given map3 flights: which termination term fires,
               when, and where on the flight (height change, distance, attitude).
  stairbrake : the eval_stairs stop test with wheel bookkeeping: does the base creep
               by the wheels rolling (the zero-command brake only damps, kd * speed)
               or by sliding; --hold-kp K adds a simulated position-hold brake.
  hipload    : hip abduction (HAA) angle and torque over a long straight drive.

Run from the repo root with PYTHONPATH=<repo>.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]  # the repo root
sys.path.insert(0, str(ROOT / "scripts" / "go2_blind"))

import mjlab.tasks  # noqa: F401,E402
import src.tasks  # noqa: F401,E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

Q = (0.1, 0.5, 0.9)


def qs(x) -> str:
  x = np.asarray(x, dtype=float)
  x = x[np.isfinite(x)]
  if x.size == 0:
    return "   -     -     -"
  return " ".join(f"{v:6.2f}" for v in np.quantile(x, Q))


def make_policy(env, checkpoint: str, task: str):
  from dataclasses import asdict

  agent_cfg = load_rl_cfg(task)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(task)(venv, asdict(agent_cfg), device="cuda:0")
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return venv, runner.get_inference_policy(device="cuda:0")


def roll_pitch_deg(robot) -> tuple[torch.Tensor, torch.Tensor]:
  g = robot.data.projected_gravity_b  # (B, 3), (0, 0, -1) when level
  pitch = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
  roll = torch.rad2deg(torch.atan2(-g[:, 1], -g[:, 2]))
  return roll, pitch


# ----------------------------------------------------------------------------- map3fail
def map3fail(args) -> None:
  import eval_map3 as em

  boxes = em.load_map3_boxes()
  scen = {s.name: s for s in em.flight_scenarios(boxes, args.up_vx, args.down_vx)}
  scen |= {s.name: s for s in em.SCENARIOS}
  print(f"[map3fail] {args.checkpoint} task {args.task} operator {args.operator} "
        f"time scale {args.time_scale}, {args.num_envs} robots per scenario")
  for name in args.scenarios:
    s = scen[name]
    if args.shift_along:
      yaw = math.radians(s.yaw_deg)
      s.spawn = (s.spawn[0] + args.shift_along * math.cos(yaw), s.spawn[1] + args.shift_along * math.sin(yaw), s.spawn[2])
      print(f"   {name}: spawn moved {args.shift_along:+.2f} m along the heading to {s.spawn}")
    if s.success != "track":
      s.segments = [(d * args.time_scale if vx else d, vx, vy, wz) for d, vx, vy, wz in s.segments]
    env = ManagerBasedRlEnv(cfg=em.make_env_cfg(s, boxes, args.num_envs, args.seed, args.task), device="cuda:0")
    venv, policy = make_policy(env, args.checkpoint, args.task)
    env.reset()
    obs = venv.get_observations()
    robot = env.scene["robot"]
    tm = env.termination_manager
    term_names = [n for n in tm.active_terms if not tm.get_term_cfg(n).time_out]
    cmd = env.command_manager.get_term("twist")
    n = env.num_envs
    dev = env.device
    yaw_target = math.radians(s.yaw_deg)
    axis = torch.tensor((math.cos(yaw_target), math.sin(yaw_target)), device=dev)
    z0 = robot.data.root_link_pos_w[:, 2].clone()
    xy0 = robot.data.root_link_pos_w[:, :2].clone()
    kind, _, thr = s.success.partition(":")
    zmin, zmax = z0.clone(), z0.clone()
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    reached = torch.zeros_like(fell)
    t_fail = torch.full((n,), float("nan"), device=dev)
    rec = {k: torch.full((n,), float("nan"), device=dev) for k in ("dz", "along", "lat", "roll", "pitch", "vx")}
    which = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in term_names}
    steps = 0
    for duration, vx, vy, wz in s.segments:
      em.set_command(env, vx, vy, wz)
      for step in range(int(round(duration / env.step_dt))):
        if args.operator and kind != "track" and vx != 0.0 and step > 0:
          heading_err = torch.atan2(torch.sin(yaw_target - robot.data.heading_w), torch.cos(yaw_target - robot.data.heading_w))
          rel = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
          lateral = -axis[1] * rel[:, 0] + axis[0] * rel[:, 1]
          cmd.vel_command_b[:, 1] = (-lateral).clamp(-0.2, 0.2)
          cmd.vel_command_b[:, 2] = heading_err.clamp(-0.5, 0.5)
        # State before the step: a terminated env is reset inside the step.
        pos = robot.data.root_link_pos_w.clone()
        roll, pitch = roll_pitch_deg(robot)
        vxb = robot.data.root_link_lin_vel_b[:, 0].clone()
        with torch.no_grad():
          obs, _, dones, extras = venv.step(policy(obs))
        steps += 1
        terminated = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
        new = terminated & ~fell & ~reached
        if new.any():
          rel = pos[:, :2] - xy0
          t_fail = torch.where(new, torch.full_like(t_fail, steps * env.step_dt), t_fail)
          vals = {
            "dz": pos[:, 2] - z0, "along": rel @ axis, "lat": -axis[1] * rel[:, 0] + axis[0] * rel[:, 1],
            "roll": roll, "pitch": pitch, "vx": vxb,
          }
          for k, v in vals.items():
            rec[k] = torch.where(new, v, rec[k])
          for k in term_names:
            which[k] |= new & tm.get_term(k).bool()
        if kind == "climb":
          met = zmax - z0 >= float(thr)
        elif kind == "descend":
          met = z0 - zmin >= float(thr)
        else:
          met = torch.zeros_like(reached)
        reached |= met & ~fell
        fell |= terminated & ~reached
        z = robot.data.root_link_pos_w[:, 2]
        zmin = torch.where(fell, zmin, torch.minimum(zmin, z))
        zmax = torch.where(fell, zmax, torch.maximum(zmax, z))
    f = fell.cpu().numpy()
    print(f"\n== {name} ({s.success}): reached {reached.float().mean():.1%}, failed (terminated first) {f.mean():.1%}")
    if f.any():
      counts = {k: int(which[k][fell].sum()) for k in term_names}
      print("   termination terms among failures: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v) +
            f"   (of {int(f.sum())})")
      tf = t_fail.cpu().numpy()[f]
      print(f"   time of failure s p10/p50/p90: {qs(tf)}   (first 2 s: {np.mean(tf < 2.0):.0%})")
      for k, unit in (("dz", "m"), ("along", "m"), ("lat", "m"), ("roll", "deg"), ("pitch", "deg"), ("vx", "m/s")):
        print(f"   {k:>5} {unit:>3} at failure p10/p50/p90: {qs(rec[k].cpu().numpy()[f])}")
    env.close()
    del env, venv, policy
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- stairbrake
def stairbrake(args) -> None:
  from src.tasks.velocity.blind_env_cfg import STAIR_TERRAIN_NAMES

  num_rows, step_range = 4, (0.05, 0.18)
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.seed = args.seed
  gen = env_cfg.scene.terrain.terrain_generator
  gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k in STAIR_TERRAIN_NAMES}
  for v in gen.sub_terrains.values():
    v.proportion = 1.0
  gen.num_rows = num_rows
  gen.num_cols = len(gen.sub_terrains) * 2
  env_cfg.events.pop("randomize_terrain")
  num_envs = gen.num_rows * gen.num_cols * args.envs_per_tile
  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  venv, policy = make_policy(env, args.checkpoint, args.task)
  dev, dt = env.device, env.step_dt
  terrain = env.scene.terrain
  robot = env.scene["robot"]
  cmd_term = env.command_manager.get_term("twist")
  wheel_ids, wheel_names = robot.find_joints(".*_wheel_joint")
  wheel_ids = torch.tensor(wheel_ids, device=dev)
  brake_speed = env.action_manager.get_term("wheel_vel").cfg.brake_speed
  wheel_act = next(a for a in robot._actuators if all("wheel" in t for t in a.target_names))
  while not hasattr(wheel_act, "stiffness"):  # unwrap DelayedActuator
    wheel_act = wheel_act._base_actuator
  names = list(gen.sub_terrains.keys())
  cum = np.cumsum(np.full(len(names), 1.0 / len(names)))
  col_name = [names[np.min(np.where(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
  tile = torch.arange(num_envs, device=dev) // args.envs_per_tile
  levels, types = tile % gen.num_rows, tile // gen.num_rows
  env_level = levels.cpu().numpy()
  env_dir = np.array(["down" if "down" in col_name[t] else "up" for t in types.cpu().numpy()])
  print(f"[stairbrake] {args.checkpoint} task {args.task}: {num_envs} robots, vx {args.vx}, "
        f"wheels {wheel_names} r {args.wheel_radius} m, brake below {brake_speed:.3f} rad/s, "
        f"hold kp {args.hold_kp} (0 = brake as trained: velocity target 0, damping only)")
  terrain.terrain_levels[:] = levels
  terrain.terrain_types[:] = types
  terrain.env_origins[:] = terrain.terrain_origins[levels, types]
  env.reset()
  obs = venv.get_observations()

  def cheb():
    return (robot.data.root_link_pos_w[:, :2] - terrain.env_origins[:, :2]).abs().max(dim=1).values

  def set_cmd(c, ids=None):
    t = torch.tensor(c, device=dev, dtype=torch.float32)
    if ids is None:
      cmd_term.vel_command_b[:] = t
    else:
      cmd_term.vel_command_b[ids] = t
    cmd_term.time_left[:] = 1e9

  set_cmd((args.vx, 0.0, 0.0))
  stopped_at = torch.full((num_envs,), -1, dtype=torch.long, device=dev)
  alive = torch.ones(num_envs, dtype=torch.bool, device=dev)
  hold = int(4.0 / dt)
  theta_hold = torch.zeros(num_envs, len(wheel_ids), device=dev)
  was_braked = torch.zeros(num_envs, dtype=torch.bool, device=dev)
  pos_hist, wpos_hist, brake_hist, wvel_hist = [], [], [], []
  for t in range(int(10.0 / dt) + hold):
    newly = (stopped_at < 0) & (cheb() >= 2.2) & alive
    if newly.any():
      set_cmd((0.0, 0.0, 0.0), newly)
      stopped_at[newly] = t
    wvel = robot.data.joint_vel[:, wheel_ids]
    braked = torch.all(cmd_term.vel_command_b == 0.0, dim=1) & torch.all(wvel.abs() < brake_speed, dim=1)
    if args.hold_kp > 0.0:
      wpos = robot.data.joint_pos[:, wheel_ids]
      theta_hold = torch.where((braked & ~was_braked).unsqueeze(1), wpos, theta_hold)
      robot.data.joint_pos_target[:, wheel_ids] = torch.where(braked.unsqueeze(1), theta_hold, wpos)
      wheel_act.stiffness[:] = torch.where(braked.unsqueeze(1), args.hold_kp, 0.0)
    was_braked = braked
    with torch.no_grad():
      obs, _, _, _ = venv.step(policy(obs))
    term = env.termination_manager.terminated
    in_hold = (stopped_at >= 0) & (t - stopped_at < hold)
    alive &= ~(term & ((stopped_at < 0) | in_hold))
    pos_hist.append(robot.data.root_link_pos_w[:, :2].clone())
    wpos_hist.append(robot.data.joint_pos[:, wheel_ids].clone())
    wvel_hist.append(robot.data.joint_vel[:, wheel_ids].abs().max(dim=1).values.clone())
    brake_hist.append(braked.clone())
    if ((stopped_at >= 0) & (t - stopped_at >= hold) | ~alive).all():
      break
  P = torch.stack(pos_hist).cpu().numpy()
  W = torch.stack(wpos_hist).cpu().numpy()
  V = torch.stack(wvel_hist).cpu().numpy()
  B = torch.stack(brake_hist).cpu().numpy()
  st, al = stopped_at.cpu().numpy(), alive.cpu().numpy()
  drift = np.full(num_envs, np.nan)
  roll = np.full(num_envs, np.nan)
  braked_frac = np.full(num_envs, np.nan)
  wmax = np.full(num_envs, np.nan)
  for b in range(num_envs):
    s = st[b]
    if s >= 0 and al[b] and s + hold <= len(P):
      a, e = s + hold // 2, s + hold - 1
      drift[b] = np.linalg.norm(P[e, b] - P[a, b])
      roll[b] = np.mean(np.abs(W[e, b] - W[a, b])) * args.wheel_radius
      braked_frac[b] = B[a:e + 1, b].mean()
      wmax[b] = V[a:e + 1, b].max()
  ok = (st >= 0) & al & (np.nan_to_num(drift, nan=1.0) < 0.05)
  print("     band  dir | stop ok  fell | base drift cm | wheel roll cm | roll/drift | braked | max wheel speed")
  for lvl in range(num_rows):
    lo, hi = step_range
    band = f"{100 * (lo + (hi - lo) * lvl / num_rows):.0f}-{100 * (lo + (hi - lo) * (lvl + 1) / num_rows):.0f}cm"
    for d in ("up", "down"):
      m = (env_level == lvl) & (env_dir == d)
      mm = m & np.isfinite(drift)
      ratio = np.nanmean(roll[mm]) / max(np.nanmean(drift[mm]), 1e-6) if mm.any() else np.nan
      print(f"  {band:>7} {d:>4} | {ok[m].mean():7.0%} {(~al[m]).mean():5.0%} | {100 * np.nanmean(drift[mm]):13.1f} | "
            f"{100 * np.nanmean(roll[mm]):13.1f} | {ratio:10.2f} | {np.nanmean(braked_frac[mm]):6.0%} | {np.nanmean(wmax[mm]):6.2f} rad/s")
  fin = np.isfinite(drift)
  print(f"overall: stop ok {ok.mean():.0%}, fell {(~al).mean():.0%}; base drift {100 * np.nanmean(drift):.1f} cm, "
        f"wheel roll {100 * np.nanmean(roll):.1f} cm (mean over held robots, last 2 s); braked {np.nanmean(braked_frac[fin]):.0%} of those steps")


# ------------------------------------------------------------------------------ hipload
def hipload(args) -> None:
  cmds = [tuple(float(x) for x in c.split(",")) for c in args.cmds.split(";")]
  print(f"[hipload] {args.checkpoint} task {args.task}: {args.num_envs} robots, {args.seconds:.0f} s per command")
  print("  outward = hip abduction away from the body, relative to the default pose (deg); "
        "torque = HAA actuator torque (Nm), mean |tau| and RMS over the run after the first 2 s")
  for c in cmds:
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.episode_length_s = args.seconds + 5.0
    for k in ("push_robot", "randomize_terrain"):
      cfg.events.pop(k, None)
    cfg.curriculum = {}
    # One 200 m flat tile: a 60 s drive at 1.5 m/s must not leave the terrain.
    gen = cfg.scene.terrain.terrain_generator
    gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k == "flat"}
    gen.sub_terrains["flat"].proportion = 1.0
    gen.size, gen.num_rows, gen.num_cols, gen.border_width = (200.0, 200.0), 1, 1, 1.0
    gen.curriculum = False
    cfg.scene.terrain.max_init_terrain_level = None
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    venv, policy = make_policy(env, args.checkpoint, args.task)
    env.reset()
    obs = venv.get_observations()
    robot = env.scene["robot"]
    dev, dt = env.device, env.step_dt
    cmd_term = env.command_manager.get_term("twist")
    hip_ids, hip_names = robot.find_joints(".*_hip_joint")
    act_names = list(robot.actuator_names)
    hip_act = [act_names.index(nm) for nm in hip_names]
    side = torch.tensor([1.0 if nm[1] == "L" else -1.0 for nm in hip_names], device=dev)
    front = np.array([nm[0] == "F" for nm in hip_names])
    q0 = robot.data.default_joint_pos[:, hip_ids].clone()
    out_hist, tau_hist, vx_hist, fell = [], [], [], torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    for _ in range(int(args.seconds / dt)):
      cmd_term.vel_command_b[:] = torch.tensor(c, device=dev)
      cmd_term.time_left[:] = 1e9
      with torch.no_grad():
        obs, _, dones, extras = venv.step(policy(obs))
      fell |= dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
      out_hist.append((torch.rad2deg(robot.data.joint_pos[:, hip_ids] - q0) * side).cpu())
      tau_hist.append(robot.data.actuator_force[:, hip_act].cpu())
      vx_hist.append(robot.data.root_link_lin_vel_b[:, 0].cpu())
    O = torch.stack(out_hist).numpy()  # (T, B, 4)
    Tq = torch.stack(tau_hist).numpy()
    VX = torch.stack(vx_hist).numpy()
    ok = ~fell.cpu().numpy()
    w = int(5.0 / dt)
    skip = int(2.0 / dt)
    O, Tq, VX = O[:, ok], Tq[:, ok], VX[:, ok]
    line = f"  cmd {c} vx {VX[skip:].mean():.2f} m/s, fell {(~ok).sum()}/{len(ok)}"
    for nm, msk in (("front", front), ("rear", ~front)):
      first, last = O[skip:skip + w][:, :, msk].mean(), O[-w:][:, :, msk].mean()
      tau = Tq[skip:][:, :, msk]
      line += (f" | {nm}: outward {first:+5.1f} -> {last:+5.1f} deg, |tau| {np.abs(tau).mean():4.2f}, "
               f"rms {np.sqrt((tau ** 2).mean()):4.2f} Nm")
    print(line)
    env.close()
    del env, venv, policy
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="mode", required=True)
  a = sub.add_parser("map3fail")
  a.add_argument("checkpoint")
  a.add_argument("--task", required=True)
  a.add_argument("--scenarios", nargs="+", required=True)
  a.add_argument("--num-envs", type=int, default=64)
  a.add_argument("--time-scale", type=float, default=1.6)
  a.add_argument("--up-vx", type=float, default=0.5)
  a.add_argument("--down-vx", type=float, default=0.4)
  a.add_argument("--operator", action=argparse.BooleanOptionalAction, default=True)
  a.add_argument("--shift-along", type=float, default=0.0, help="Move the spawn this far along the heading (m).")
  a.add_argument("--seed", type=int, default=0)
  b = sub.add_parser("stairbrake")
  b.add_argument("checkpoint")
  b.add_argument("--task", required=True)
  b.add_argument("--wheel-radius", type=float, required=True)
  b.add_argument("--vx", type=float, default=0.5)
  b.add_argument("--envs-per-tile", type=int, default=6)
  b.add_argument("--hold-kp", type=float, default=0.0)
  b.add_argument("--seed", type=int, default=0)
  c = sub.add_parser("hipload")
  c.add_argument("checkpoint")
  c.add_argument("--task", required=True)
  c.add_argument("--cmds", default="0,0,0;0.5,0,0;1.0,0,0;1.5,0,0;1.0,0,0.1")
  c.add_argument("--seconds", type=float, default=60.0)
  c.add_argument("--num-envs", type=int, default=32)
  c.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  {"map3fail": map3fail, "stairbrake": stairbrake, "hipload": hipload}[args.mode](args)


if __name__ == "__main__":
  main()
