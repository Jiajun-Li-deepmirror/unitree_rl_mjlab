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

Robustness checks before hardware can add what training already sees or what
the real robot adds:
  --pushes      keep the training task's interval pushes (play mode drops them)
  --obs-noise   keep the training task's actor observation noise
  --latency-ms  hold every joint target back by this delay, quantized to physics
                steps, as a fixed control-loop latency
"""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import torch
from project_paths import motion_output_dir, project_relative
from tracking_diagnostics import TrackingDiagnostics, summarize
from tracking_targets import initialize_reference_targets

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

TASK = "Unitree-G1-Tracking-No-State-Estimation"
METRICS = ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot",
           "error_joint_pos", "error_joint_vel")
REFERENCE_RGBA = (0.45, 0.78, 1.0, 0.35)


def configure_video(cfg, reference_overlay=True):
  cfg.viewer.width, cfg.viewer.height = 1280, 720
  cfg.viewer.distance = 3.0
  cfg.viewer.azimuth, cfg.viewer.elevation = 135, -15
  cfg.viewer.max_extra_envs = 0
  cfg.commands["motion"].debug_vis = reference_overlay
  cfg.commands["motion"].viz.mode = "ghost"
  cfg.commands["motion"].viz.ghost_color = REFERENCE_RGBA
  return cfg


def prepare_reference_overlay(env):
  """Prepare mjlab's render-only model; keep the trained motion command intact."""
  cmd = env.command_manager.get_term("motion")
  ghost = copy.deepcopy(env.sim.mj_model)
  visible = ghost.geom_rgba[:, 3] > 0
  ghost.geom_rgba[visible] = REFERENCE_RGBA
  ghost.geom_matid[:] = -1
  # mjlab enables mjVIS_TRANSPARENT; avoid multiplying opacity by its 0.3 default.
  ghost.vis.map.alpha = 1.0
  cmd._ghost_model = ghost


def video_legend(frame, t, cmd):
  """Label the two robots without changing the scene or simulation state."""
  from PIL import Image, ImageDraw, ImageFont
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image, "RGBA")
  font = ImageFont.load_default(size=20)
  draw.rounded_rectangle((18, 18, 494, 125), radius=10, fill=(12, 22, 34, 210))
  draw.rectangle((32, 32, 47, 47), fill=(235, 235, 235, 255))
  draw.text((58, 27), "Policy - solid robot", font=font, fill=(245, 245, 245, 255))
  draw.rectangle((32, 64, 47, 79), fill=(115, 199, 255, 255))
  draw.text((58, 59), "Reference - light blue, 35% opacity", font=font, fill=(155, 220, 255, 255))
  body = cmd.metrics["error_body_pos"][0].item() * 100
  root = cmd.metrics["error_anchor_pos"][0].item() * 100
  draw.text((32, 94), f"t {t:05.2f}s  |  Body {body:4.1f} cm  |  Anchor {root:4.1f} cm",
            font=font, fill=(230, 236, 242, 255))
  return np.asarray(image)


def prepare_eval_cfg(cfg, motion, num_envs, randomize, seed, push=None, obs_noise=False,
                     latency_steps=0):
  """push: the training task's push event to keep (play configs drop it)."""
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = int(1e9)
  cmd = cfg.commands["motion"]
  cmd.motion_file = motion
  cmd.sampling_mode = "start"
  cmd.pose_range = {}
  cmd.velocity_range = {}
  cmd.joint_position_range = (0.0, 0.0)
  cfg.observations["actor"].enable_corruption = obs_noise
  cfg.events.pop("push_robot", None)
  if not randomize:
    cfg.events = {}
  if push is not None:
    cfg.events["push_robot"] = copy.deepcopy(push)
  if latency_steps:
    delay_actuators(cfg, latency_steps)
  return cfg


def delay_actuators(cfg, steps):
  """Apply each joint target `steps` physics steps late, like a slow control loop."""
  from mjlab.actuator import DelayedActuatorCfg
  robot = cfg.scene.entities["robot"]
  # Replace, not edit: the articulation config can be shared with other task configs.
  robot.articulation = dataclasses.replace(robot.articulation, actuators=tuple(
    DelayedActuatorCfg(base_cfg=a, delay_target="position", delay_min_lag=steps, delay_max_lag=steps)
    for a in robot.articulation.actuators))



def restart_delays(env):
  """Start each delayed actuator's history from the first real joint target.

  Resetting writes a zero joint target, and a delay buffer would replay it for
  the whole delay, pulling every joint toward 0 rad at the start. A real robot
  keeps holding its previous command instead.
  """
  from mjlab.actuator import DelayedActuator
  for actuator in env.scene["robot"].actuators:
    if isinstance(actuator, DelayedActuator):
      actuator.reset()


def rollout(venv, policy, on_step=None):
  env = venv.unwrapped
  cmd = env.command_manager.get_term("motion")
  n_frames = int(cmd.motion.joint_pos.shape[0])
  term_names = list(env.termination_manager.active_terms)
  alive = np.ones(env.num_envs, dtype=bool)
  terminations, samples = [], []

  venv.reset()
  restart_delays(env)
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
  def __init__(self, video=None, reference_overlay=True):
    self.video = pathlib.Path(video) if video else None
    self.writer = None
    self.video_frames = 0
    self.foot_speed_sum = 0.0
    self.planted_samples = 0
    self.foot_speed_max = 0.0
    self.joint_speed_max = 0.0
    self.reference_overlay = reference_overlay

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
      if self.reference_overlay:
        frame = video_legend(frame, t, cmd)
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
            "reference_overlay": {"enabled": bool(self.video and self.reference_overlay),
                                  "rgba": list(REFERENCE_RGBA), "coordinates": "world",
                                  "timing": "current reference frame; no per-frame pose alignment"},
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
  ap.add_argument("--trajectory", help="save synchronized joint/foot/force samples to NPZ plus a leg tracking plot")
  ap.add_argument("--reference-overlay", action=argparse.BooleanOptionalAction, default=True,
                  help="overlay the current reference in translucent light blue (default: enabled)")
  ap.add_argument("--pushes", action="store_true", help="keep the training task's interval pushes")
  ap.add_argument("--obs-noise", action="store_true", help="keep the training task's actor observation noise")
  ap.add_argument("--latency-ms", type=float, default=0.0,
                  help="fixed delay of every joint target, quantized to physics steps")
  args = ap.parse_args()
  if args.num_envs < 1 or not np.isfinite(args.bin) or args.bin <= 0:
    ap.error("--num-envs and --bin must be positive")
  if not np.isfinite(args.latency_ms) or args.latency_ms < 0:
    ap.error("--latency-ms must be zero or positive")
  if args.video:
    os.environ.setdefault("MUJOCO_GL", "egl")

  import src.tasks  # noqa: F401  (registers tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cfg = load_env_cfg(args.task, play=True)
  physics_dt = cfg.sim.mujoco.timestep
  latency_steps = int(round(args.latency_ms / 1000 / physics_dt))
  push = None
  if args.pushes:
    push = load_env_cfg(args.task).events.get("push_robot")
    if push is None:
      ap.error(f"{args.task} has no push_robot event to keep")
  cfg = prepare_eval_cfg(cfg, args.motion, args.num_envs, args.randomize, args.seed,
                         push=push, obs_noise=args.obs_noise, latency_steps=latency_steps)
  with np.load(args.motion, allow_pickle=False) as motion:
    motion_fps = np.asarray(motion["fps"]).reshape(-1)
    expected_fps = 1.0 / (cfg.sim.mujoco.timestep * cfg.decimation)
    if motion_fps.size != 1 or not np.isclose(motion_fps[0], expected_fps, rtol=0, atol=1e-6):
      ap.error(f"motion fps must match the environment rate ({expected_fps:g} Hz)")
  if args.video:
    configure_video(cfg, args.reference_overlay)
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if args.video else None)
  if args.video and args.reference_overlay:
    prepare_reference_overlay(env)
  agent_cfg = load_rl_cfg(args.task)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  n = args.num_envs
  artifacts = RolloutArtifacts(args.video, args.reference_overlay)
  diagnostics = TrackingDiagnostics()
  def record(env, cmd, t, valid):
    artifacts(env, cmd, t, valid)
    diagnostics(env, cmd, t, valid)
  try:
    duration, completed, terminations, samples = rollout(venv, policy, record)
  finally:
    artifacts.close()
  if args.trajectory:
    diagnostics.save(args.trajectory)
  bins = {}
  for t, e, m in samples:
    bins.setdefault(int(t // args.bin), []).append((e, m))
  per_bin = []
  for b, ms in sorted(bins.items()):
    row = {"t0": b * args.bin, "envs": len({e for e, _ in ms}), "samples": len(ms)}
    row.update({k: float(np.mean([m[k] for _, m in ms])) for k in ms[0][1]})
    per_bin.append(row)
  report = {
    "checkpoint": project_relative(args.checkpoint), "motion": project_relative(args.motion), "task": args.task,
    "path_base": "project_root", "num_envs": n,
    "checkpoint_sha256": hashlib.sha256(pathlib.Path(args.checkpoint).read_bytes()).hexdigest(),
    "motion_sha256": hashlib.sha256(pathlib.Path(args.motion).read_bytes()).hexdigest(),
    "randomize": args.randomize, "seed": args.seed, "motion_duration_s": duration,
    "perturbations": {"pushes": args.pushes, "obs_noise": args.obs_noise,
                      "latency_physics_steps": latency_steps,
                      "latency_ms": latency_steps * physics_dt * 1000,
                      "push_velocity_range": push.params["velocity_range"] if push else None},
    "completed_envs": completed,
    "physical_diagnostics": artifacts.report(),
    "limb_diagnostics": summarize(diagnostics.data()),
    "trajectory": project_relative(args.trajectory) if args.trajectory else None,
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
