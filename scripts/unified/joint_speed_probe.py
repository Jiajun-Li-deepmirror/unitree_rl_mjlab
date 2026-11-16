"""joint_speed_probe.py --task <flat or stairs task> <checkpoint> (go2-depth 2026-11-16, step 1 of U_MOTOR: how often
today's policies run past the GO-M8010-6 motor's torque-speed corner, before the curve is modelled).

The motor (unitree_rl_lab UnitreeActuatorCfg_Go2HV, at the joint for hip / thigh / wheel): full torque up to X1 13.5
rad/s, then linearly down to 0 at X2 30 rad/s; the calf has a linkage ratio R (Go2 1.917, Go2W 1.5): its corner and
no-load speed are X1 / R and X2 / R at the joint. Read-only: the policy runs on the task's own play physics.

Flat task: groups of robots, each holding one command for HOLD s (slewed at 2 m/s^2 / 3 rad/s^2, as the acceptance's
controllers), then a step to zero for 3 s (the brake): forward 1.0 / 1.5, back 1.0, sideways 0.6, a turn in place 1.5,
an arc 1.0 + 0.3, and a stand. Stairs task: every robot walks forward at 0.5 m/s from its spawn for WALK s (the stair
tiles, both directions, every band).

Per joint group (hip, thigh, calf, wheel) over all robot-steps: |joint speed| p50 / p99 / max (rad/s), the share
above the corner and above the no-load speed, and -- where the applied torque can be read -- the share where the
torque would be clipped by the curve (|tau| above the curve's limit at that speed, the torque pushing the way the
joint moves).

  python jobs/unified/u_eval.py scripts/unified/joint_speed_probe.py --task Unitree-Go2W-U-Teacher-Flat <ckpt>
"""
import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

X1, X2, Y1 = 13.5, 30.0, 20.2
CALF_RATIO = {"Go2": 1.917, "Go2W": 1.5}
FLAT = [("fwd 1.0", (1.0, 0.0, 0.0)), ("fwd 1.5", (1.5, 0.0, 0.0)), ("back 1.0", (-1.0, 0.0, 0.0)),
        ("side 0.6", (0.0, 0.6, 0.0)), ("turn 1.5", (0.0, 0.0, 1.5)), ("arc 1.0+0.3", (1.0, 0.0, 0.3)),
        ("stand", (0.0, 0.0, 0.0))]

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--num-envs", type=int, default=128, help="robots per flat group / in all on the stairs")
p.add_argument("--hold", type=float, default=4.0)
p.add_argument("--walk", type=float, default=10.0)
args = p.parse_args()
robot_name = "Go2W" if "Go2W" in args.task else "Go2"
stairs = "Stairs" in args.task

cfg = load_env_cfg(args.task, play=True)
n = args.num_envs
cfg.scene.num_envs = n if stairs else n * len(FLAT)
T = args.walk if stairs else args.hold + 3.0
cfg.episode_length_s = T + 5.0
cfg.events.pop("push_robot", None)
if not stairs:
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
names = robot.joint_names
groups = {g: [i for i, nm in enumerate(names) if g in nm] for g in ("hip", "thigh", "calf", "wheel")}
groups = {g: ids for g, ids in groups.items() if ids}
R = CALF_RATIO.get(robot_name, 1.0)
corner = {g: (X1 / R if g == "calf" else X1) for g in groups}
noload = {g: (X2 / R if g == "calf" else X2) for g in groups}
ylim = {g: (Y1 * R if g == "calf" else Y1) for g in groups}


def torque() -> torch.Tensor | None:
  d = robot.data
  for k in ("actuator_force", "joint_torque", "applied_torque", "qfrc_actuator"):
    v = getattr(d, k, None)
    if v is not None and v.shape[-1] == len(names):
      return v
  return None


speeds = {g: [] for g in groups}
clip = {g: [0, 0] for g in groups}
target = torch.zeros(env.num_envs, 3, device=dev)
if not stairs:
  for i, (_, c) in enumerate(FLAT):
    target[i * n:(i + 1) * n] = torch.tensor(c, device=dev)
else:
  target[:, 0] = 0.5
cur = torch.zeros_like(target)
env.reset()
obs = venv.get_observations()
fell = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
have_tau = None
for k in range(int(round(T / env.step_dt))):
  t = k * env.step_dt
  want = target if (stairs or t < args.hold) else torch.zeros_like(target)
  lin, yaw = 2.0 * env.step_dt, 3.0 * env.step_dt
  d = want - cur
  dn = d[:, :2].norm(dim=1, keepdim=True).clamp(min=1e-9)
  cur[:, :2] = torch.where(want[:, :2].abs().sum(1, keepdim=True) == 0, want[:, :2],
                           cur[:, :2] + d[:, :2] * torch.clamp(lin / dn, max=1.0))   # (slewed rise, step stop)
  cur[:, 2] = torch.where(want[:, 2] == 0, want[:, 2], cur[:, 2] + d[:, 2].clamp(-yaw, yaw))
  cmd.vel_command_b[:] = cur
  cmd.time_left[:] = 1e9
  with torch.no_grad():
    obs, _, dones, extras = venv.step(policy(obs))
  fell |= dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
  v = robot.data.joint_vel
  tau = torque()
  have_tau = tau is not None
  alive = ~fell
  for g, ids in groups.items():
    s = v[alive][:, ids].abs().flatten()
    speeds[g].append(s.float().cpu())
    if tau is not None:
      vv, tt = v[alive][:, ids], tau[alive][:, ids]
      w = vv.abs()
      lim = torch.where(w <= corner[g], torch.full_like(w, ylim[g]),
                        (ylim[g] * (noload[g] - w) / (noload[g] - corner[g])).clamp(min=0.0))
      motoring = (vv * tt) > 0
      clip[g][0] += int(((tt.abs() > lim) & motoring).sum())
      clip[g][1] += int(tt.numel())
print(f"{args.checkpoint}\n{args.task} ({robot_name}): {'stairs walk 0.5 m/s, ' + str(n) + ' robots, ' + str(args.walk) + ' s' if stairs else str(n) + ' robots per group: ' + ', '.join(f for f, _ in FLAT) + f'; {args.hold:g} s, then a step stop, 3 s'}")
print(f"motor GO-M8010-6 (Go2HV curve): corner {X1} rad/s, no-load {X2}, flat torque {Y1} N m at the joint; calf ratio {R} "
      f"(corner {X1 / R:.2f}, no-load {X2 / R:.2f}, {Y1 * R:.1f} N m); applied torque {'read' if have_tau else 'not readable'}")
print(f"{'group':>6} | speed p50 p99 max rad/s | above corner | above no-load | torque over the curve (motoring)")
for g in groups:
  s = torch.cat(speeds[g]) if speeds[g] else torch.zeros(1)
  qs = torch.quantile(s[:200000] if len(s) > 200000 else s, torch.tensor([0.5, 0.99])).tolist()
  over = float((s > corner[g]).float().mean())
  over2 = float((s > noload[g]).float().mean())
  cl = f"{100 * clip[g][0] / max(clip[g][1], 1):.2f} %" if have_tau else "-"
  print(f"{g:>6} | {qs[0]:6.2f} {qs[1]:6.2f} {float(s.max()):6.2f} | {100 * over:9.2f} % | {100 * over2:10.3f} % | {cl}")
print(f"falls {int(fell.sum())}/{env.num_envs}")
env.close()
