"""push_test.py --task <flat task> <checkpoint> (go2-depth 2026-11-16; acceptance item 22, 抗推, the user via go2-home:
"可以都添加").

Play config of a flat task (its own training push removed), one large flat tile, N robots per group, all groups in one
simulation. Each group holds one command -- stand (0, 0, 0), forward (1.0, 0, 0), sideways (0, 0.5, 0); the linear
command slewed at --lin-accel m/s^2 from a standstill, as the acceptance's controllers -- and at --push-at s (3.0) every
robot gets one velocity push: PUSH m/s (0.5) added to its base's linear velocity in one direction of its own base frame
(+x, -x, +y, -y; yaw-aligned, level), written to the simulator as mjlab's push_by_setting_velocity does. The run goes
on --after s (5.0) after the push.

Per group: falls (terminations that are not time-outs; first falls only); the recovery: the first time after the push
from which the base's horizontal velocity (base frame) stays within TOL m/s (0.3) of the command for HOLD s (0.5) -- the time from the push to that moment, p50 / p90 over the robots that recovered and did not fall,
and the share of the robots still standing that never recovered; for reference the error's peak after the push (p50:
the push's size as the robot felt it) and its p90 over the second before the push (a gait whose steady error is near
TOL reads slow recoveries that are the gait's, not the push's).

Judged (accept_judge.spec_push; doc/acceptance_go2_go2w.md item 22): no fall in any group, every condition (U_PAYLOAD
from the environment as the other tests; --payload kg,x,y,z for an old task). The recovery is reported.

  python jobs/unified/u_eval.py scripts/unified/push_test.py --task Unitree-Go2W-U-Teacher-Flat <ckpt> [--num-envs 64]
"""
import argparse
import os
import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "blind", "eval"))   # (run from the repo root, as every test)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blind", "eval"))
import payload_opt  # noqa: E402

COMMANDS = [("stand", (0.0, 0.0, 0.0)), ("fwd", (1.0, 0.0, 0.0)), ("side", (0.0, 0.5, 0.0))]
PUSHES = [("+x", (1.0, 0.0)), ("-x", (-1.0, 0.0)), ("+y", (0.0, 1.0)), ("-y", (0.0, -1.0))]
TOL = 0.3      # m/s: the base back within this of the command
HOLD = 0.5     # s: ... and staying there this long

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--num-envs", type=int, default=64, help="robots per group (command x push direction)")
p.add_argument("--push", type=float, default=0.5, help="m/s added to the base's velocity")
p.add_argument("--push-at", type=float, default=3.0)
p.add_argument("--after", type=float, default=5.0)
p.add_argument("--lin-accel", type=float, default=2.0, help="m/s^2 slew of the linear command (0: a step)")
payload_opt.add(p)
p.add_argument("--seed", type=int, default=None,
               help="the environment's seed (2026-11-16, repeats near a bar); unset: none, as the flat tests")
args = p.parse_args()

groups = [(c, cv, d, dv) for c, cv in COMMANDS for d, dv in PUSHES]
cfg = load_env_cfg(args.task, play=True)
if args.seed is not None:
  cfg.seed = args.seed
payload_opt.apply(cfg, args.payload)
n = args.num_envs
cfg.scene.num_envs = n * len(groups)
T = args.push_at + args.after
cfg.episode_length_s = T + 5.0
cfg.events.pop("push_robot", None)
gen = cfg.scene.terrain.terrain_generator
gen.size = (120.0, 120.0)
gen.num_rows = gen.num_cols = 1
agent = load_rl_cfg(args.task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(args.task)(venv, asdict(agent), device="cuda:0")
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
robot = env.scene["robot"]
cmd = env.command_manager.get_term("twist")
dev = env.device
N = env.num_envs
dt = env.step_dt

target = torch.zeros(N, 3, device=dev)
push_b = torch.zeros(N, 2, device=dev)
for i, (_, cv, _, dv) in enumerate(groups):
  target[i * n:(i + 1) * n] = torch.tensor(cv, device=dev)
  push_b[i * n:(i + 1) * n] = torch.tensor(dv, device=dev) * args.push
cur = torch.zeros(N, 3, device=dev)
fell = torch.zeros(N, dtype=torch.bool, device=dev)
fell_after = torch.zeros(N, dtype=torch.bool, device=dev)   # fell at or after the push
pre_err = torch.zeros(N, device=dev)                       # the error's largest over the second before the push
peak = torch.zeros(N, device=dev)                          # ... and after it
ok_since = torch.full((N,), float("nan"), device=dev)
rec = torch.full((N,), float("nan"), device=dev)            # seconds from the push to the recovery
pushed = False

ramp = f", linear command slewed at {args.lin_accel:g} m/s^2" if args.lin_accel > 0 else ""
print(f"{args.checkpoint}\n{args.task}: {n} robots per group, {len(groups)} groups ({', '.join(c for c, _ in COMMANDS)} x "
      f"pushes {', '.join(d for d, _ in PUSHES)} in the base frame){ramp}; {args.push:g} m/s added to the base's velocity at "
      f"{args.push_at:g} s, then {args.after:g} s; recovered = the base velocity (base frame) within {TOL:g} m/s of the "
      f"command for {HOLD:g} s")
print(f"{'group':>11} | falls | recovered after the push p50 p90 | not recovered | error peak after the push p50 | "
      f"before the push p90")
env.reset()
obs = venv.get_observations()
steps = int(round(T / dt))
for k in range(steps):
  t = k * dt
  if not pushed and t >= args.push_at - 1e-9:
    # the push: the standing robots' base velocity + PUSH in their own (level) base frame, as push_by_setting_velocity
    h = robot.data.heading_w
    c, s = torch.cos(h), torch.sin(h)
    vel = robot.data.root_link_vel_w.clone()
    vel[:, 0] += c * push_b[:, 0] - s * push_b[:, 1]
    vel[:, 1] += s * push_b[:, 0] + c * push_b[:, 1]
    ids = (~fell).nonzero(as_tuple=False).flatten()
    robot.write_root_link_velocity_to_sim(vel[ids], env_ids=ids)
    pushed = True
  if args.lin_accel > 0:
    a = args.lin_accel * dt
    d = target[:, :2] - cur[:, :2]
    dn = d.norm(dim=1, keepdim=True).clamp(min=1e-9)
    cur[:, :2] = cur[:, :2] + d * torch.clamp(a / dn, max=1.0)
  else:
    cur[:, :2] = target[:, :2]
  cur[:, 2] = target[:, 2]
  cmd.vel_command_b[:] = cur
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, dones, extras = venv.step(policy(obs))
  new = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool() & ~fell
  if pushed:
    fell_after |= new
  fell |= new
  err = (robot.data.root_link_lin_vel_b[:, :2] - cur[:, :2]).norm(dim=1)
  t1 = t + dt
  if not pushed and t1 >= args.push_at - 1.0 - 1e-9:
    pre_err = torch.maximum(pre_err, err)
  if pushed:
    peak = torch.where(fell, peak, torch.maximum(peak, err))
    good = (err <= TOL) & ~fell
    ok_since = torch.where(good & torch.isnan(ok_since), torch.full_like(ok_since, t1), ok_since)
    ok_since = torch.where(~good, torch.full_like(ok_since, float("nan")), ok_since)
    done = torch.isnan(rec) & torch.isfinite(ok_since) & (t1 - ok_since >= HOLD - 1e-9)
    rec = torch.where(done, ok_since - args.push_at, rec)

q = torch.tensor((0.5, 0.9), device=dev)
qs = lambda x: x.quantile(q).tolist() if len(x) else [float("nan")] * 2
for i, (c, _, d, _) in enumerate(groups):
  g = torch.zeros(N, dtype=torch.bool, device=dev)
  g[i * n:(i + 1) * n] = True
  up = g & ~fell
  r = rec[up]
  r50, r90 = qs(r[torch.isfinite(r)])
  nr = 100.0 * float(torch.isnan(r).float().mean()) if len(r) else float("nan")
  pe = qs(pre_err[g])[1]
  pk = qs(peak[up])[0]
  print(f"{c + ' ' + d:>11} | falls {int((fell & g).sum())}/{n} | recovered p50 {r50:.2f} s p90 {r90:.2f} s | "
        f"not recovered {nr:.1f} % | error peak p50 {pk:.2f} m/s | before p90 {pe:.2f} m/s")
up = ~fell
r = rec[up]
r50, r90 = qs(r[torch.isfinite(r)])
nr = 100.0 * float(torch.isnan(r).float().mean()) if len(r) else float("nan")
print(f"ALL push (item 22): falls {int(fell.sum())} of {N} robots; recovered within {TOL:g} m/s: p50 {r50:.2f} s, "
      f"p90 {r90:.2f} s, not recovered {nr:.1f} % (falls at or after the push {int(fell_after.sum())})")
env.close()
