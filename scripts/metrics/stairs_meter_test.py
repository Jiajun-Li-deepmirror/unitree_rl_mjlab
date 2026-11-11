#!/usr/bin/env python3
"""stairs_meter_test.py --task T CHECKPOINT [--envs N] [--steps S]: run a checkpoint in its task's training terrain and
print src/metrics.mj.StairsMeter's readings (go2-depth 2026-11-12: the mjlab side of the shared stairs metrics, checked
against plausible values before the teacher uses them). Play config, commands as in training, every env sampled."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("UNIFIED_TASKS", "1")

import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

import src.tasks  # noqa: E402,F401
from src.metrics import mj  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--envs", type=int, default=256)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--wheeled", action="store_true")
args = p.parse_args()

cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = args.envs
mj.add_sensor(cfg)
agent = load_rl_cfg(args.task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(args.task)(venv, asdict(agent), device="cuda:0")
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
meter = mj.StairsMeter(env, torch.arange(env.num_envs), wheeled=args.wheeled, min_steps=20.0)
print(f"grid {meter.grid.H.shape} cells at {meter.grid.res} m, {meter.feet} feet, stairs envs "
      f"{int(env.command_manager.get_term('twist').is_stair_env.sum())} / {env.num_envs}", flush=True)
obs = venv.get_observations()
z0 = env.scene["robot"].data.root_link_pos_w[:, 2].clone()
ep0 = env.episode_length_buf.clone()
with torch.inference_mode():
  for i in range(args.steps):
    obs, _, _, _ = venv.step(policy(obs))
    meter.step(env)
    if (i + 1) == 400:
      stairs = env.command_manager.get_term("twist").is_stair_env
      same = env.episode_length_buf > ep0   # not reset since the start
      dz = env.scene["robot"].data.root_link_pos_w[:, 2] - z0
      m = stairs & same
      print(f"diag: {int(m.sum())} stairs robots not reset in 400 steps: base rose > 10 cm {int((dz[m] > 0.10).sum())}, "
            f"fell > 10 cm {int((dz[m] < -0.10).sum())}", flush=True)
    if (i + 1) % 500 == 0:
      names = ("slide_back", "idle_start", "idle_end") if args.wheeled else ("kicks", "slips", "mid", "edge")
      vals = {k: meter.rate(k) for k in names}
      print(f"step {i + 1}: sums {{{', '.join(f'{k} {v:.0f}' for k, v in meter.sums.items())}}} | "
            + " ".join(f"{k} {'-' if v is None else f'{v:.3f}'}" for k, v in vals.items()), flush=True)
