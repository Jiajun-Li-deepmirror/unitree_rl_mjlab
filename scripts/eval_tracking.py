"""Post-training evaluation: deterministic rollouts of a tracking checkpoint over the
whole reference motion, reported per time bin.

Each env plays the motion once from t=0 (play mode: no RSI, no pushes, no obs
noise). An env stops being recorded at its first termination. Output: JSON with
per-bin tracking errors (the env's own Metrics/motion/* terms), termination
events (which term, when), survival, worst segments; and a PNG plot.

Usage:
  python src/unitree_rl_mjlab/scripts/eval_tracking.py --checkpoint output/<video>/training/<run>/model_30000.pt \
      --motion output/<video>/<video>.npz \
      [--num-envs 1] [--randomize] [--seed 42] [--device cpu]

--randomize keeps the startup domain randomization (com offset, encoder bias,
foot friction) so several envs give a robustness spread instead of identical runs.
Reference time and initial joints remain fixed in both modes. The first policy
step is evaluated, including any real termination on that step.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import torch
from project_paths import motion_output_dir, project_relative

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

TASK = "Unitree-G1-Tracking-No-State-Estimation"
METRICS = ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot",
           "error_joint_pos", "error_joint_vel")


def prepare_eval_cfg(cfg, motion, num_envs, randomize, seed):
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = int(1e9)
  cmd = cfg.commands["motion"]
  cmd.motion_file = motion
  cmd.sampling_mode = "start"
  cmd.pose_range = {}
  cmd.velocity_range = {}
  cmd.joint_position_range = (0.0, 0.0)
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  if not randomize:
    cfg.events = {}
  return cfg


def initialize_reference_targets(cmd):
  """Fill mjlab 1.2's relative targets at t=0 without advancing physics/time.

  MotionCommand.reset() leaves these buffers uninitialized until its first
  update. Match MotionCommand._update_command's alignment, but do not increment
  time_steps, resample the motion, or discard a policy step.
  """
  from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

  anchor = cmd.anchor_pos_w[:, None, :]
  delta_pos = cmd.robot_anchor_pos_w[:, None, :].clone()
  delta_pos[..., 2] = anchor[..., 2]
  delta_quat = yaw_quat(quat_mul(cmd.robot_anchor_quat_w, quat_inv(cmd.anchor_quat_w)))
  delta_quat = delta_quat[:, None, :].expand(-1, len(cmd.cfg.body_names), -1)
  cmd.body_pos_relative_w = delta_pos + quat_apply(delta_quat, cmd.body_pos_w - anchor)
  cmd.body_quat_relative_w = quat_mul(delta_quat, cmd.body_quat_w)
  cmd._update_metrics()


def rollout(venv, policy, on_step=None):
  env = venv.unwrapped
  cmd = env.command_manager.get_term("motion")
  n_frames = int(cmd.motion.joint_pos.shape[0])
  term_names = list(env.termination_manager.active_terms)
  alive = np.ones(env.num_envs, dtype=bool)
  terminations, samples = [], []

  venv.reset()
  initialize_reference_targets(cmd)
  obs = venv.get_observations()
  if on_step:
    on_step(env, cmd, 0.0, alive.copy())
  with torch.inference_mode():
    # Stop at the last reference frame, before mjlab wraps/resamples the clip.
    for step in range(n_frames - 1):
      t = (step + 1) * env.step_dt
      obs, _, dones, _ = venv.step(policy(obs))
      fired = {k: env.termination_manager.get_term(k).cpu().numpy().astype(bool)
               for k in term_names}
      d = dones.cpu().numpy().astype(bool)
      # mjlab updates metrics before advancing the reference. Refresh them so
      # the report compares the post-step robot with the current reference.
      cmd._update_metrics()
      vals = {k: cmd.metrics[k].cpu().numpy() for k in METRICS if k in cmd.metrics}
      if on_step:
        # Terminated environments have already reset; never record their reset
        # pose as if it were the failed policy trajectory.
        on_step(env, cmd, t, alive & ~d)
      for e in np.where(alive & ~d)[0]:
        samples.append((t, int(e), {k: float(v[e]) for k, v in vals.items()}))
      for e in np.where(alive & d)[0]:
        terminations.append({"env": int(e), "t": t,
                             "terms": [k for k in term_names if fired[k][e] and k != "time_out"]
                                      or ["time_out"]})
      alive &= ~d
      if not alive.any():
        break
  return (n_frames - 1) * env.step_dt, int(alive.sum()), terminations, samples


class RolloutArtifacts:
  """Stream policy video and summarize velocities before the first reset."""
  def __init__(self, video=None):
    self.video = pathlib.Path(video) if video else None
    self.writer = None
    self.video_frames = 0
    self.foot_speed_sum = 0.0
    self.planted_samples = 0
    self.foot_speed_max = 0.0
    self.joint_speed_max = 0.0

  def __call__(self, env, cmd, t, valid):
    if t > 0 and valid.any():
      ids = [i for i, name in enumerate(cmd.cfg.body_names) if name in
             ("left_ankle_roll_link", "right_ankle_roll_link")]
      mask = torch.as_tensor(valid, device=env.device)
      height = cmd.body_pos_w[:, ids, 2] - env.scene.env_origins[:, None, 2]
      ref_speed = torch.linalg.vector_norm(cmd.body_lin_vel_w[:, ids, :2], dim=-1)
      planted = (height < .05) & (ref_speed < .3) & mask[:, None]
      speed = torch.linalg.vector_norm(cmd.robot_body_lin_vel_w[:, ids, :2], dim=-1)[planted]
      self.planted_samples += speed.numel()
      self.foot_speed_sum += speed.sum().item()
      if speed.numel():
        self.foot_speed_max = max(self.foot_speed_max, speed.max().item())
      self.joint_speed_max = max(self.joint_speed_max, cmd.robot_joint_vel[mask].abs().max().item())
    if self.video and valid[0]:
      import imageio_ffmpeg
      frame = env.render()
      if frame is None:
        raise RuntimeError("policy video requires rgb_array rendering")
      if self.writer is None:
        self.video.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        self.writer = imageio_ffmpeg.write_frames(str(self.video), (w, h),
                      fps=1 / env.step_dt, codec="libx264", pix_fmt_out="yuv420p",
                      output_params=["-crf", "20", "-movflags", "+faststart"])
        self.writer.send(None)
      self.writer.send(np.ascontiguousarray(frame))
      self.video_frames += 1

  def close(self):
    if self.writer is not None:
      self.writer.close()

  def report(self):
    return {"video": project_relative(self.video) if self.video else None,
            "video_frames": self.video_frames,
            "peak_joint_speed_rad_s": self.joint_speed_max,
            "reference_planted_foot_samples": self.planted_samples,
            "reference_planted_foot_speed_mean_m_s": self.foot_speed_sum / self.planted_samples if self.planted_samples else None,
            "reference_planted_foot_speed_max_m_s": self.foot_speed_max if self.planted_samples else None,
            "foot_metric_definition": "Robot ankle horizontal speed when reference ankle z < 0.05 m and horizontal speed < 0.3 m/s; reference-based contact proxy, not measured contact. Terminal reset states excluded."}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--motion", required=True)
  ap.add_argument("--out", help="output prefix; default: output/<video>/evaluation/<video>")
  ap.add_argument("--task", default=TASK)
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--randomize", action="store_true")
  ap.add_argument("--bin", type=float, default=1.0, help="report bin width (s)")
  ap.add_argument("--device", default=None)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--video", help="stream the trained policy rollout for env 0 to this MP4")
  args = ap.parse_args()
  if args.num_envs < 1 or not np.isfinite(args.bin) or args.bin <= 0:
    ap.error("--num-envs and --bin must be positive")
  if args.video:
    os.environ.setdefault("MUJOCO_GL", "egl")

  import src.tasks  # noqa: F401  (registers tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cfg = prepare_eval_cfg(load_env_cfg(args.task, play=True), args.motion,
                         args.num_envs, args.randomize, args.seed)
  with np.load(args.motion, allow_pickle=False) as motion:
    motion_fps = np.asarray(motion["fps"]).reshape(-1)
    expected_fps = 1.0 / (cfg.sim.mujoco.timestep * cfg.decimation)
    if motion_fps.size != 1 or not np.isclose(motion_fps[0], expected_fps, rtol=0, atol=1e-6):
      ap.error(f"motion fps must match the environment rate ({expected_fps:g} Hz)")
  if args.video:
    cfg.viewer.width, cfg.viewer.height = 1280, 720
    cfg.viewer.distance = 3.0
    cfg.viewer.azimuth, cfg.viewer.elevation = 135, -15
    cfg.commands["motion"].debug_vis = False
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if args.video else None)
  agent_cfg = load_rl_cfg(args.task)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  n = args.num_envs
  artifacts = RolloutArtifacts(args.video)
  try:
    duration, completed, terminations, samples = rollout(venv, policy, artifacts)
  finally:
    artifacts.close()
  bins = {}
  for t, e, m in samples:
    bins.setdefault(int(t // args.bin), []).append((e, m))
  per_bin = []
  for b, ms in sorted(bins.items()):
    row = {"t0": b * args.bin, "envs": len({e for e, _ in ms}), "samples": len(ms)}
    row.update({k: float(np.mean([m[k] for _, m in ms])) for k in ms[0][1]})
    per_bin.append(row)
  report = {
    "checkpoint": project_relative(args.checkpoint), "motion": project_relative(args.motion),
    "path_base": "project_root", "num_envs": n,
    "checkpoint_sha256": hashlib.sha256(pathlib.Path(args.checkpoint).read_bytes()).hexdigest(),
    "motion_sha256": hashlib.sha256(pathlib.Path(args.motion).read_bytes()).hexdigest(),
    "randomize": args.randomize, "seed": args.seed, "motion_duration_s": duration,
    "completed_envs": completed,
    "physical_diagnostics": artifacts.report(),
    "terminations": terminations,
    "mean": {k: float(np.mean([m[k] for _, _, m in samples])) for k in METRICS if samples and k in samples[0][2]},
    "worst_bins": sorted(per_bin, key=lambda r: -r.get("error_body_pos", 0))[:5],
    "per_bin": per_bin,
  }
  out = pathlib.Path(args.out) if args.out else motion_output_dir(args.motion) / "evaluation" / pathlib.Path(args.motion).stem
  out.parent.mkdir(parents=True, exist_ok=True)
  pathlib.Path(str(out) + ".json").write_text(json.dumps(report, indent=1))
  plot(report, pathlib.Path(str(out) + ".png"))

  print(f"completed {report['completed_envs']}/{n} envs over {duration:.1f}s of motion")
  for tr in terminations:
    print(f"  env {tr['env']} terminated at {tr['t']:.2f}s by {','.join(tr['terms'])}")
  print("mean: " + ", ".join(f"{k}={v:.3f}" for k, v in report["mean"].items()))
  print("worst bins (by error_body_pos): " +
        ", ".join(f"{b['t0']:.0f}s={b.get('error_body_pos', 0):.3f}" for b in report["worst_bins"]))
  print(f"wrote {out}.json and .png")
  env.close()


def plot(report, path):
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    return
  rows = report["per_bin"]
  if not rows:
    return
  t = [r["t0"] for r in rows]
  fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
  for k in ("error_anchor_pos", "error_body_pos"):
    if k in rows[0]:
      ax[0].plot(t, [r[k] for r in rows], label=k + " (m)")
  for k in ("error_joint_pos", "error_anchor_rot", "error_body_rot"):
    if k in rows[0]:
      ax[1].plot(t, [r[k] for r in rows], label=k)
  for tr in report["terminations"]:
    for a in ax:
      a.axvline(tr["t"], color="r", ls="--", lw=0.8)
  for a in ax:
    a.legend(loc="upper right")
  ax[1].set_xlabel("time (s)")
  ax[0].set_title(f"{report['completed_envs']}/{report['num_envs']} envs completed "
                  f"{report['motion_duration_s']:.1f}s")
  fig.tight_layout()
  fig.savefig(path, dpi=90)


if __name__ == "__main__":
  main()
