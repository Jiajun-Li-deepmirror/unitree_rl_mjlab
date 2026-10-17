"""Flat-ground acceptance check for a Go2 depth teacher (or blind) checkpoint.

Runs fixed joystick commands on flat ground (each scenario on its own group of
envs), reports tracking, gait and standing metrics against go2-blind's
stage-1 goals, and optionally records a scripted video of one robot.

  PYTHONPATH=. python scripts/go2_depth/eval_flat.py \
    --checkpoint logs/rsl_rl/go2_depth_teacher/<run>/model_N.pt --video

Exit code 0 = all gates pass, 1 = some gate failed (details printed and in
eval_flat.json next to the checkpoint).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from tensordict import TensorDict

# name: (vx, vy, wz)
SCENARIOS = {
  "stand": (0.0, 0.0, 0.0),
  "vx_+0.5": (0.5, 0.0, 0.0),
  "vx_+1.0": (1.0, 0.0, 0.0),
  "vx_-0.4": (-0.4, 0.0, 0.0),
  "vy_+0.4": (0.0, 0.4, 0.0),
  "vy_-0.4": (0.0, -0.4, 0.0),
  "wz_+1.0": (0.0, 0.0, 1.0),
  "wz_-1.0": (0.0, 0.0, -1.0),
  "wz_+0.4": (0.0, 0.0, 0.4),
  "combo": (0.6, 0.3, 0.6),
}
VIDEO_SCRIPT = [  # (seconds, vx, vy, wz)
  (2.0, 0.0, 0.0, 0.0),
  (3.0, 0.6, 0.0, 0.0),
  (3.0, 0.0, 0.4, 0.0),
  (3.0, 0.0, -0.4, 0.0),
  (3.0, 0.0, 0.0, 1.0),
  (3.0, 0.0, 0.0, -1.0),
  (3.0, 0.6, 0.3, 0.6),
  (3.0, 0.0, 0.0, 0.0),
]
DURATION_S = 6.0
WARMUP_S = 1.5  # Ignore the command transition at the start.

# Gates: (metric, scenario filter, low, high).
GATES = [
  ("fall_rate", "*", None, 0.02),
  ("ratio_vx", "vx_+0.5", 0.8, 1.15),
  ("ratio_vx", "vx_+1.0", 0.8, 1.15),
  ("ratio_vx", "vx_-0.4", 0.7, 1.2),
  ("ratio_vy", "vy_+0.4", 0.75, 1.2),
  ("ratio_vy", "vy_-0.4", 0.75, 1.2),
  ("ratio_wz", "wz_+1.0", 0.8, 1.15),
  ("ratio_wz", "wz_-1.0", 0.8, 1.15),
  ("ratio_wz", "wz_+0.4", 0.7, 1.2),
  # Turning in place must step (touchdowns on every foot), not twist.
  ("min_foot_touchdown_hz", "wz_+1.0", 1.0, None),
  ("min_foot_touchdown_hz", "wz_-1.0", 1.0, None),
  ("min_foot_touchdown_hz", "wz_+0.4", 0.8, None),
  # Zero command: stand still on four feet.
  ("all_feet_contact_frac", "stand", 0.95, None),
  ("speed_xy", "stand", None, 0.05),
  ("abs_wz", "stand", None, 0.1),
  ("max_air_time_s", "stand", None, 0.5),
]
# Wheel-legged robots roll forward and back, and step to move sideways or to
# turn in place (mdp/wheels.py step_mode_mask): add rolling/stepping gates.
WHEELED_GATES = GATES + [
  ("max_foot_touchdown_hz", "vx_+0.5", None, 0.5),
  ("max_foot_touchdown_hz", "vx_+1.0", None, 0.5),
  ("max_foot_touchdown_hz", "vx_-0.4", None, 0.5),
  ("min_foot_touchdown_hz", "vy_+0.4", 0.8, None),
  ("min_foot_touchdown_hz", "vy_-0.4", 0.8, None),
]


def load_teacher(checkpoint: str, obs: TensorDict, device: str):
  from rsl_rl.models import MLPModel

  ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
  weights = [k for k in ck["actor_state_dict"] if k.startswith("mlp.") and k.endswith(".weight")]
  num_actions = ck["actor_state_dict"][max(weights, key=lambda k: int(k.split(".")[1]))].shape[0]
  model = MLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    num_actions,
    hidden_dims=(512, 256, 128),
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  model.load_state_dict(ck["actor_state_dict"])
  return model.to(device).eval()


def make_env(task: str, num_envs: int, render: bool):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  cfg.events.pop("push_robot", None)
  cfg.episode_length_s = 1.0e4
  twist = cfg.commands["twist"]
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.stair_resampling_time_range = (1.0e9, 1.0e9)
  twist.zero_threshold = 0.0
  twist.debug_vis = False
  if render:
    for sensor in cfg.scene.sensors or ():
      if hasattr(sensor, "debug_vis"):
        sensor.debug_vis = False
    cfg.viewer.width, cfg.viewer.height = 640, 480
  return ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None)


def set_commands(env, cmd: torch.Tensor) -> None:
  # Commands are never resampled (huge resampling time); reset envs get a
  # random command on reset, so rewrite every step.
  env.command_manager.get_term("twist").vel_command_b[:] = cmd


def evaluate(args) -> dict:
  names = list(SCENARIOS)
  per = args.envs_per_scenario
  env = make_env(args.task, per * len(names), render=False)
  cmd = torch.tensor([SCENARIOS[n] for n in names], device=env.device).repeat_interleave(per, 0)
  obs, _ = env.reset()
  set_commands(env, cmd)
  obs = TensorDict(env.observation_manager.compute(update_history=False), batch_size=[env.num_envs])
  policy = load_teacher(args.checkpoint, obs, env.device)

  robot = env.scene["robot"]
  contact = env.scene["feet_ground_contact"]
  # Posture deviation over the legs only (wheel angles are unbounded).
  legs = [i for i, name in enumerate(robot.joint_names) if "wheel" not in name]
  default_q = robot.data.default_joint_pos[:, legs]
  n = env.num_envs
  steps = int(DURATION_S / env.step_dt)
  warm = int(WARMUP_S / env.step_dt)
  acc = {k: torch.zeros(n, device=env.device) for k in ("vx", "vy", "wz", "speed_xy", "abs_wz", "all4", "qdev")}
  touchdowns = torch.zeros(n, 4, device=env.device)
  max_air = torch.zeros(n, device=env.device)
  diag_agree = torch.zeros(n, device=env.device)
  fell = torch.zeros(n, dtype=torch.bool, device=env.device)
  prev = None
  for t in range(steps):
    with torch.inference_mode():
      raw, _, terminated, _, _ = env.step(policy(obs))
    set_commands(env, cmd)
    obs = TensorDict(raw, batch_size=[n])
    fell |= terminated
    found = contact.data.found.reshape(n, 4) > 0
    if t >= warm:
      v = robot.data.root_link_lin_vel_b
      w = robot.data.root_link_ang_vel_b[:, 2]
      acc["vx"] += v[:, 0]
      acc["vy"] += v[:, 1]
      acc["wz"] += w
      acc["speed_xy"] += torch.norm(v[:, :2], dim=1)
      acc["abs_wz"] += w.abs()
      acc["all4"] += found.all(dim=1).float()
      acc["qdev"] += (robot.data.joint_pos[:, legs] - default_q).abs().max(dim=1).values
      # Feet order FR, FL, RR, RL: diagonals are (FR, RL) and (FL, RR).
      diag_agree += 0.5 * ((found[:, 0] == found[:, 3]).float() + (found[:, 1] == found[:, 2]).float())
      if prev is not None:
        touchdowns += (found & ~prev).float()
      max_air = torch.maximum(max_air, contact.data.current_air_time.reshape(n, 4).max(dim=1).values)
    prev = found
  m = steps - warm
  span = m * env.step_dt
  results = {}
  for i, name in enumerate(names):
    sl = slice(i * per, (i + 1) * per)
    ok = ~fell[sl]
    vx, vy, wz = SCENARIOS[name]

    def mean(x):
      return float(x[sl][ok].mean()) if ok.any() else float("nan")

    r = {
      "fall_rate": float(fell[sl].float().mean()),
      "vx": mean(acc["vx"] / m),
      "vy": mean(acc["vy"] / m),
      "wz": mean(acc["wz"] / m),
      "speed_xy": mean(acc["speed_xy"] / m),
      "abs_wz": mean(acc["abs_wz"] / m),
      "all_feet_contact_frac": mean(acc["all4"] / m),
      "max_joint_dev_rad": mean(acc["qdev"] / m),
      "diag_contact_agree": mean(diag_agree / m),
      "min_foot_touchdown_hz": mean(touchdowns.min(dim=1).values / span),
      "max_foot_touchdown_hz": mean(touchdowns.max(dim=1).values / span),
      "max_air_time_s": float(max_air[sl][ok].max()) if ok.any() else float("nan"),
    }
    for key, target in (("vx", vx), ("vy", vy), ("wz", wz)):
      if target != 0.0:
        r[f"ratio_{key}"] = r[key] / target
    results[name] = r
  results["_action_terms"] = tuple(env.action_manager.active_terms)
  env.close()
  return results


def check_gates(results: dict, gates: list = GATES) -> list[str]:
  failures = []
  for metric, scope, low, high in gates:
    for name, r in results.items():
      if scope not in ("*", name) or metric not in r:
        continue
      value = r[metric]
      if np.isnan(value) or (low is not None and value < low) or (high is not None and value > high):
        failures.append(f"{name}.{metric}={value:.3f} not in [{low}, {high}]")
  return failures


def record_video(args, out: Path) -> None:
  import imageio

  env = make_env(args.task, 1, render=True)
  obs, _ = env.reset()
  obs = TensorDict(obs, batch_size=[1])
  policy = load_teacher(args.checkpoint, obs, env.device)
  frames = []
  fps = int(round(1.0 / env.step_dt / 2))
  for seconds, vx, vy, wz in VIDEO_SCRIPT:
    cmd = torch.tensor([[vx, vy, wz]], device=env.device)
    for t in range(int(seconds / env.step_dt)):
      set_commands(env, cmd)
      with torch.inference_mode():
        raw = env.step(policy(obs))[0]
      obs = TensorDict(raw, batch_size=[1])
      if t % 2 == 0:
        frames.append(env.render())
  env.close()
  imageio.mimsave(out, frames, fps=fps)
  print(f"[video] {out}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--task", default="Unitree-Go2-Depth-Teacher-Flat")
  parser.add_argument("--envs-per-scenario", type=int, default=16)
  parser.add_argument("--video", action="store_true")
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  results = evaluate(args)
  wheeled = "wheel_vel" in results.pop("_action_terms", ())
  failures = check_gates(results, WHEELED_GATES if wheeled else GATES)
  print(f"\n[eval_flat] {args.checkpoint}")
  keys = ["fall_rate", "ratio_vx", "ratio_vy", "ratio_wz", "speed_xy", "abs_wz",
          "min_foot_touchdown_hz", "max_foot_touchdown_hz", "all_feet_contact_frac", "max_air_time_s",
          "diag_contact_agree", "max_joint_dev_rad"]
  for name, r in results.items():
    print(f"  {name:8s} " + " ".join(f"{k}={r[k]:.2f}" for k in keys if k in r))
  print("[eval_flat] " + ("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures)))
  out_dir = Path(args.checkpoint).parent
  (out_dir / "eval_flat.json").write_text(
    json.dumps({"checkpoint": args.checkpoint, "results": results, "failures": failures}, indent=2)
  )
  if args.video:
    record_video(args, out_dir / f"eval_flat_{Path(args.checkpoint).stem}.mp4")
  raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
  main()
