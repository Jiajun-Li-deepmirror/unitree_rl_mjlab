"""stop_then_move.py --task <flat task> <checkpoint> [--payload kg,x,y,z] (go2-depth 2026-11-09, the user: in the
Go2W 12 kg flat video the legs stood wrong after the stop from 1.5 m/s and the next moves nearly fell).

Play config of a flat task (no pushes), N robots per group, all groups in one simulation, the command slewed at 2 m/s^2
and 3 rad/s^2 as the controllers do. Each group: stand 2 s, then (groups "after stop") drive at 1.5 m/s for 3 s and
stand 3 s, then a move for 3 s, then stand 2 s; the "fresh" groups make the same move from the first stand. Moves:
strafe left 0.6, strafe right -0.6, turn in place 1.0, forward 0.5.

Per group, over the robots that did not fall: the legs' deviation from the default stance at the end of the stand
before the move (rad: mean |q - q0| over the leg joints, p50 / p90; and the worst joint's mean by name), the base pitch
then (deg), during the move's last 2 s the speed along the commanded axis over the command (p50 / p10) and the yaw rate
not asked for (|wz| mean, rad/s; turns: the speed is wz), and falls.
"""
import argparse
import math
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
import src.tasks.velocity.mdp as mdp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--num-envs", type=int, default=128, help="robots per group")
p.add_argument("--payload", default=None, help="kg,x,y,z fixed point mass on the base (else 0 kg)")
args = p.parse_args()

MOVES = [("strafe left 0.6", (0.0, 0.6, 0.0)), ("strafe right 0.6", (0.0, -0.6, 0.0)), ("turn 1.0", (0.0, 0.0, 1.0)),
         ("forward 0.5", (0.5, 0.0, 0.0))]
GROUPS = [(f"{m} after stop", True, c) for m, c in MOVES] + [(f"{m} fresh", False, c) for m, c in MOVES]
n = args.num_envs
T_STAND, T_DRIVE, T_STOP, T_MOVE, T_END = 2.0, 3.0, 3.0, 3.0, 2.0

cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = n * len(GROUPS)
cfg.episode_length_s = T_STAND + T_DRIVE + T_STOP + T_MOVE + T_END + 5.0
cfg.events.pop("push_robot", None)
if "base_payload" in cfg.events or args.payload:
  m, x, y, z = (float(v) for v in (args.payload or "0,0,0,0").split(","))
  cfg.events["base_payload"] = EventTermCfg(
    mode="reset", func=mdp.base_payload_point,
    params={"asset_cfg": SceneEntityCfg("robot"), "light_range": (m, m), "heavy_max": 0.0, "heavy_share": 0.0,
            "pos_ranges": ((x, x), (y, y), (z, z)), "base_com_range": 0.0, "ramp_steps": 0},
  )
  cfg.events.pop("base_com", None)
gen = cfg.scene.terrain.terrain_generator
gen.size = (200.0, 200.0)
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
dt = env.step_dt
legs = [i for i, nm in enumerate(robot.joint_names) if any(k in nm for k in ("hip", "thigh", "calf"))]
leg_names = [robot.joint_names[i] for i in legs]
after = torch.tensor([g[1] for g in GROUPS], device=dev).repeat_interleave(n)
move = torch.tensor([g[2] for g in GROUPS], device=dev).repeat_interleave(n, dim=0)
t_move = torch.where(after, torch.full_like(after, T_STAND + T_DRIVE + T_STOP, dtype=torch.float),
                     torch.full_like(after, T_STAND + T_DRIVE + T_STOP, dtype=torch.float))
env.reset()
obs = venv.get_observations()
fell = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
cur = torch.zeros(env.num_envs, 3, device=dev)
dev_q = None
pitch = None
spd = torch.zeros(env.num_envs, device=dev)
wz_off = torch.zeros(env.num_envs, device=dev)
k_meas = 0
steps = int(round((T_STAND + T_DRIVE + T_STOP + T_MOVE + T_END) / dt))
for k in range(steps):
  t = k * dt
  target = torch.zeros(env.num_envs, 3, device=dev)
  drive = after & (t >= T_STAND) & (t < T_STAND + T_DRIVE)
  target[drive, 0] = 1.5
  moving = (t >= T_STAND + T_DRIVE + T_STOP) & (t < T_STAND + T_DRIVE + T_STOP + T_MOVE)
  if moving:
    target = move.clone()
  d = target - cur
  lin = d[:, :2]
  nrm = torch.norm(lin, dim=1, keepdim=True)
  lin = torch.where(nrm > 2.0 * dt, lin * (2.0 * dt) / torch.clamp(nrm, min=1e-9), lin)
  cur[:, :2] += lin
  cur[:, 2] += torch.clamp(d[:, 2], -3.0 * dt, 3.0 * dt)
  cmd.vel_command_b[:] = cur
  cmd.time_left[:] = 1e9
  if hasattr(cmd, "march"):
    cmd.march[:] = False
  with torch.no_grad():
    obs, _, dones, extras = venv.step(policy(obs))
  fell |= dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
  if abs(t - (T_STAND + T_DRIVE + T_STOP - 2 * dt)) < dt / 2:   # the end of the stand before the move
    q = robot.data.joint_pos[:, legs] - robot.data.default_joint_pos[:, legs]
    dev_q = q.clone()
    g = robot.data.projected_gravity_b
    pitch = torch.rad2deg(torch.atan2(-g[:, 0], -g[:, 2]))
  if moving and t >= T_STAND + T_DRIVE + T_STOP + 1.0:
    v = robot.data.root_link_lin_vel_b
    w = robot.data.root_link_ang_vel_b[:, 2]
    along = torch.where(move[:, 2] != 0, w / torch.where(move[:, 2] != 0, move[:, 2], torch.ones_like(w)),
                        (v[:, :2] * move[:, :2]).sum(1) / torch.clamp((move[:, :2] ** 2).sum(1), min=1e-9))
    spd += along
    wz_off += torch.where(move[:, 2] != 0, torch.zeros_like(w), w.abs())
    k_meas += 1
spd /= max(k_meas, 1)
wz_off /= max(k_meas, 1)
q = torch.tensor((0.1, 0.5, 0.9), device=dev)
print(f"{args.checkpoint}\n{args.task} payload {args.payload or 0}: {n} robots per group; stand {T_STAND:g} s, "
      f"(after stop: 1.5 m/s {T_DRIVE:g} s, stand {T_STOP:g} s), move {T_MOVE:g} s; command slewed at 2 m/s^2, 3 rad/s^2")
print(f"{'group':>26} | stance dev rad p50 p90 | worst joint (mean rad) | pitch deg | speed ratio p10 p50 | wz off | fell")
for i, (name, _, _) in enumerate(GROUPS):
  sel = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
  sel[i * n:(i + 1) * n] = True
  ok = sel & ~fell
  if not ok.any():
    print(f"{name:>26} | all fell")
    continue
  dq = dev_q[ok].abs()
  per = dq.mean(1)
  jm = dev_q[ok].mean(0)
  wj = int(torch.argmax(jm.abs()))
  print(f"{name:>26} | {float(per.quantile(0.5)):.3f} {float(per.quantile(0.9)):.3f}       | {leg_names[wj]:>16} {float(jm[wj]):+.2f} | "
        f"{float(pitch[ok].mean()):+5.1f}     | {float(spd[ok].quantile(0.1)):.2f} {float(spd[ok].quantile(0.5)):.2f}         | "
        f"{float(wz_off[ok].mean()):.2f}   | {int((fell & sel).sum())}/{n}")
env.close()
