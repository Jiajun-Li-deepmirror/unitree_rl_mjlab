"""term_diag.py --task T CK --cmd vx,vy,wz [--num-envs N] [--seconds S] (go2-depth 2026-11-10): which termination ends
the episodes of robots holding one command on the play config (gait_clock.py counts every done as a fall)."""
import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--cmd", default="0,0.3,0")
p.add_argument("--num-envs", type=int, default=32)
p.add_argument("--seconds", type=float, default=8.0)
p.add_argument("--ramp", type=float, default=0.0, help="m/s^2 linear command ramp (0: a step)")
p.add_argument("--train", action="store_true", help="the training config (DR, noise, pushes) instead of play")
p.add_argument("--settle", type=float, default=0.0, help="zero command for this long after every reset")
p.add_argument("--payload", default=None, help="kg,x,y,z pinned ('0,0,0,0' = none); default: the config's")
args = p.parse_args()
cfg = load_env_cfg(args.task, play=not args.train)
if args.train:
  cfg.events.pop("push_robot", None)
cfg.scene.num_envs = args.num_envs
print("base_payload event in cfg:", "base_payload" in cfg.events, cfg.events["base_payload"].params if "base_payload" in cfg.events else None)
if args.payload is not None:
  import src.tasks.velocity.mdp as mdp
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  pm, px, py, pz = (float(v) for v in args.payload.split(","))
  cfg.events["base_payload"] = EventTermCfg(mode="reset", func=mdp.base_payload_point, params={"asset_cfg": SceneEntityCfg("robot"),
    "light_range": (pm, pm), "heavy_max": 0.0, "heavy_share": 0.0, "pos_ranges": ((px, px), (py, py), (pz, pz)), "base_com_range": 0.0, "ramp_steps": 0})
  cfg.events.pop("base_com", None)
agent = load_rl_cfg(args.task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(args.task)(venv, asdict(agent), device="cuda:0")
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
cmd = env.command_manager.get_term("twist")
c = torch.tensor([float(v) for v in args.cmd.split(",")], device=env.device)
tm = env.termination_manager
print("terminations:", tm.active_terms, "episode_length_s", cfg.episode_length_s)
env.reset()
obs = venv.get_observations()
counts = {}
prev_len = env.episode_length_buf.clone()
first_t = {}
steps = int(args.seconds / env.step_dt)
for k in range(steps):
  cc = c if args.ramp <= 0 else c * min(1.0, (k + 1) * env.step_dt * args.ramp / max(float(c[:2].norm()), 1e-6))
  if args.settle > 0:
    young = (env.episode_length_buf.float() * env.step_dt < args.settle).unsqueeze(1)
    cc = torch.where(young, torch.zeros_like(c).expand(env.num_envs, 3), c.expand(env.num_envs, 3) if cc is c else cc.expand(env.num_envs, 3))
  cmd.vel_command_b[:] = cc
  cmd.time_left[:] = 1e9
  if hasattr(cmd, "march"):
    cmd.march[:] = False
  with torch.no_grad():
    obs, _, dones, extras = venv.step(policy(obs))
  if k in (int(0.2 / env.step_dt), int(1.0 / env.step_dt)):
    g = env.scene["robot"].data.projected_gravity_b
    roll = torch.rad2deg(torch.atan2(g[:, 1], -g[:, 2])); pitch = torch.rad2deg(torch.atan2(-g[:, 0], -g[:, 2]))
    rz0 = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    print(f"t {k * env.step_dt:.1f} s roll p10/p50/p90 {roll.quantile(0.1):+.1f} {roll.quantile(0.5):+.1f} {roll.quantile(0.9):+.1f}  pitch {pitch.quantile(0.1):+.1f} {pitch.quantile(0.5):+.1f} {pitch.quantile(0.9):+.1f}  z p10 {rz0.quantile(0.1):.3f}")
  if k == int(2.0 / env.step_dt):
    rz = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    print(f"t 2 s base height above origin p10 p50 p90 {rz.quantile(0.1):.3f} {rz.quantile(0.5):.3f} {rz.quantile(0.9):.3f}")
  if dones.any() and args.settle > 0:
    old = dones.bool() & (prev_len.float() * env.step_dt >= args.settle)
    counts["after_settle"] = counts.get("after_settle", 0) + int(old.sum())
  prev_len = env.episode_length_buf.clone()
  if dones.any():
    for name in tm.active_terms:
      hit = tm.get_term(name)
      n = int((hit & dones.bool()).sum())
      if n:
        counts[name] = counts.get(name, 0) + n
        first_t.setdefault(name, round(k * env.step_dt, 2))
print(f"cmd {args.cmd}: terminations over {args.seconds} s:", counts, "first at", first_t)
ic = tm.get_term_cfg("illegal_contact") if "illegal_contact" in tm.active_terms else None
if ic is not None:
  sname = ic.params.get("sensor_name")
  sensor = env.scene[sname]
  print("illegal_contact sensor", sname, "params", {k: v for k, v in ic.params.items() if k != "sensor_name"})
  names = getattr(sensor, "body_names", None) or getattr(getattr(sensor, "cfg", None), "primary", None)
  print("sensor bodies / cfg:", names if names is not None else sensor.cfg)
  hits = {}
  env.reset(); obs = venv.get_observations()
  for k in range(int(2.0 / env.step_dt)):
    cmd.vel_command_b[:] = c; cmd.time_left[:] = 1e9
    with torch.no_grad():
      obs, _, dones, _ = venv.step(policy(obs))
    found = sensor.data.found
    if found is not None and found.dim() >= 2:
      tot = (found > 0).float().sum(0).flatten().tolist()
      for i, v in enumerate(tot):
        if v:
          hits[i] = hits.get(i, 0) + v
  print("contact counts per sensor slot over 2 s:", hits)
  print("sensor attrs:", [a for a in dir(sensor) if "name" in a.lower() or "geom" in a.lower()][:20])
  for a in dir(sensor):
    if ("name" in a.lower() or "geom" in a.lower()) and not a.startswith("__"):
      try:
        v = getattr(sensor, a)
        if isinstance(v, (list, tuple)) and len(v) < 40: print(" ", a, v)
      except Exception as e: pass
  print("secondary:", getattr(sensor.cfg, "secondary", None), "reduce:", getattr(sensor.cfg, "reduce", None))
  import inspect
  try:
    print("resolve sig:", inspect.signature(sensor._resolve_primary_names))
  except Exception as e: print("sig err", e)
  m = env.sim.mj_model
  import re, mujoco
  gn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
  pats = ["base\\d_collision", "[FR][LR]_hip_collision"]
  print("matching geoms (first robot):", [g for g in gn if g and any(re.search(pp, g) for pp in pats)][:12])
