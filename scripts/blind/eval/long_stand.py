"""long_stand.py --task <flat task> <checkpoint> [--seconds 60] [--payload kg,x,y,z] (go2-depth 2026-11-07, the user: Go2W
"一开始零指令站久了也会摔倒"): N robots stand at the zero command from the start for --seconds (play config, no pushes);
--march T marches in place from T s on (11-07, the user: "有训练到负载情况下做原地踏步吗"; touchdowns per foot);
falls (terminations other than time-outs) with their times, base drift (cm) and tilt (deg) at the end."""
import argparse
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
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--seconds", type=float, default=60.0)
p.add_argument("--payload", default=None, help="kg,x,y,z fixed point mass on the base (else 0 kg)")
p.add_argument("--wheel-hold", type=float, default=None, help="braked wheels hold their angle with this stiffness (Nm/rad)")
p.add_argument("--no-leg-hold", action="store_true", help="legs never held (hold_after_s None)")
p.add_argument("--march", type=float, default=None, help="march in place (the controller's march flag) from this time (s) on")
p.add_argument("--handover", type=float, default=0.0, help="for the first T s the joints hold the default stance (zero actions, wheels braked) with the payload on, then the policy takes over (11-08, the user: the robot stands up loaded, then switches to the policy); falls before T are reported apart")
p.add_argument("--trace", type=float, default=0.0, help="print the fallen robots' base height / roll / pitch / velocity every 0.1 s over the first T s")
p.add_argument("--no-splay", action="store_true", help="no reset hip splay (robots start from the default stance +- the joint noise)")
args = p.parse_args()
GROUPS = [("stand", None)]  # one group of all robots, for jobs/fall_diag.py

cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = args.seconds + 5.0
cfg.events.pop("push_robot", None)
if args.no_splay:
  cfg.events.pop("reset_hip_splay", None)
if "base_payload" in cfg.events or args.payload:
  m, x, y, z = (float(v) for v in (args.payload or "0,0,0,0").split(","))
  cfg.events["base_payload"] = EventTermCfg(
    mode="reset", func=mdp.base_payload_point,
    params={"asset_cfg": SceneEntityCfg("robot"), "light_range": (m, m), "heavy_max": 0.0, "heavy_share": 0.0,
            "pos_ranges": ((x, x), (y, y), (z, z)), "base_com_range": 0.0, "ramp_steps": 0},
  )
  cfg.events.pop("base_com", None)
if args.wheel_hold is not None and "wheel_vel" in cfg.actions:
  cfg.actions["wheel_vel"].hold_stiffness = args.wheel_hold
if args.no_leg_hold and hasattr(cfg.actions["joint_pos"], "hold_after_s"):
  cfg.actions["joint_pos"].hold_after_s = None
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
env.reset()
obs = venv.get_observations()
p0 = robot.data.root_link_pos_w[:, :2].clone()
fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
fall_t = []
pre = 0   # falls while the stance was held (--handover)
pre_fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
sensor = env.scene["feet_ground_contact"]
touch = air = 0
nm = 0
steps = int(args.seconds / env.step_dt)
tr = []  # (t, z, roll, pitch, vx, wz, legs |q - q0| max) per step over the first --trace s
# the zero-command stance (2026-11-12, doc/redesign.md 2.2): each robot's largest leg-joint deviation from the default
# stance after POSTURE_SETTLE s (the real robot's posture reset fires at 0.15 rad, ThermalGuard posture_march_rad)
POSTURE_SETTLE = 3.0
leg_ids = [i for i, n in enumerate(robot.joint_names) if "wheel" not in n]
dev_max = torch.zeros(env.num_envs, device=env.device)
for k in range(steps):
  cmd.vel_command_b[:] = 0.0
  cmd.time_left[:] = 1e9
  if hasattr(cmd, "march"):
    cmd.march[:] = args.march is not None and k * env.step_dt >= args.march
  with torch.no_grad():
    act = policy(obs)
    if k * env.step_dt < args.handover:
      act = torch.zeros_like(act)
    obs, _, dones, extras = venv.step(act)
  new = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool() & ~fell
  if k * env.step_dt < args.handover:
    pre += int(new.sum())
    pre_fell |= new
    new = torch.zeros_like(new)
  if new.any():
    fall_t += [round(k * env.step_dt, 1)] * int(new.sum())
  fell |= new
  if k * env.step_dt >= POSTURE_SETTLE:
    d = (robot.data.joint_pos[:, leg_ids] - robot.data.default_joint_pos[:, leg_ids]).abs().amax(dim=1)
    dev_max = torch.where(fell, dev_max, torch.maximum(dev_max, d))
  if k * env.step_dt < args.trace:
    gq = robot.data.projected_gravity_b
    tr.append((k * env.step_dt, robot.data.root_link_pos_w[:, 2].clone(), torch.rad2deg(torch.atan2(gq[:, 1], -gq[:, 2])),
               torch.rad2deg(torch.atan2(-gq[:, 0], -gq[:, 2])), robot.data.root_link_lin_vel_b[:, 0].clone(),
               robot.data.root_link_ang_vel_b[:, 2].clone()))
  if args.march is not None and k * env.step_dt >= args.march + 5.0:   # touchdowns once the march has settled
    touch = touch + sensor.compute_first_contact(dt=env.step_dt)[~fell].float().mean(0)
    air = air + (sensor.data.found[~fell] == 0).float().mean(0)
    nm += 1
ok = ~fell
drift = (robot.data.root_link_pos_w[:, :2] - p0).norm(dim=1)[ok] * 100
g = robot.data.projected_gravity_b[ok]
tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
q = torch.tensor((0.5, 0.9, 1.0), device=env.device)
print(f"{args.checkpoint} {args.task} payload {args.payload or 0} wheel hold {args.wheel_hold} leg hold {not args.no_leg_hold}: {args.seconds:.0f} s at the zero command, "
      f"falls {int(fell.sum())}/{env.num_envs} at t {sorted(fall_t)[:20]} | drift cm p50 p90 max "
      f"{' '.join(f'{v:.1f}' for v in drift.quantile(q).tolist())} | tilt deg p50 p90 max {' '.join(f'{v:.1f}' for v in tilt.quantile(q).tolist())}")
print(f"  posture: leg-joint deviation from the default stance after {POSTURE_SETTLE:g} s, each robot's max: "
      f"p50 {float(dev_max[~fell].median()) if (~fell).any() else float('nan'):.3f} "
      f"p90 {float(dev_max[~fell].quantile(0.9)) if (~fell).any() else float('nan'):.3f} rad")
if args.handover > 0:
  print(f"  handover: the stance held for {args.handover:g} s with the payload, then the policy; "
        f"falls while held {pre}/{env.num_envs} (not counted above; those robots respawn and are judged with the rest)")
if nm:
  print(f"  marching (from {args.march + 5:g} s): touchdowns/s per foot {' '.join(f'{v:.2f}' for v in (touch / (nm * env.step_dt)).tolist())} "
        f"| air % {' '.join(f'{100 * v:.0f}' for v in (air / nm).tolist())}")
if args.trace > 0 and tr:
  every = max(1, int(round(0.1 / env.step_dt)))
  for e in fell.nonzero().flatten().tolist():
    print(f"  trace env {e}: t z(m) roll pitch(deg, + nose down) vx(m/s) wz(rad/s)")
    for row in tr[::every]:
      t, z, rl, pt, vx, wz = row
      print(f"    {t:4.1f} {float(z[e]):.3f} {float(rl[e]):6.1f} {float(pt[e]):6.1f} {float(vx[e]):6.2f} {float(wz[e]):6.2f}")
    zs = torch.stack([r[1] for r in tr])
  print(f"  all robots at {tr[-1][0]:.1f} s: z p50 {float(tr[-1][1].median()):.3f}, |pitch| p90 {float(tr[-1][3].abs().quantile(0.9)):.1f}")
env.close()
