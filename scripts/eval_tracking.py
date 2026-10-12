"""Post-training evaluation: deterministic rollouts of a tracking checkpoint over the
whole reference motion, reported per time bin.

Each env plays the motion once from t=0 (play mode: no RSI, no pushes, no obs
noise). An env stops being recorded at its first termination. Output: JSON with
per-bin tracking errors (the env's own Metrics/motion/* terms), termination
events (which term, when), survival, worst segments; and a PNG plot.

Usage:
  python scripts/eval_tracking.py --checkpoint logs/rsl_rl/<exp>/<run>/model_30000.pt \
      --motion src/assets/motions/g1/dance2_v3.npz --out eval/dance2_v3 \
      [--num-envs 1] [--randomize] [--device cpu]

--randomize keeps the startup domain randomization (com offset, encoder bias,
foot friction) so several envs give a robustness spread instead of identical runs.
"""
import argparse
import json
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

TASK = "Unitree-G1-Tracking-No-State-Estimation"
METRICS = ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot",
           "error_joint_pos", "error_joint_vel")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--motion", required=True)
  ap.add_argument("--out", required=True, help="output path prefix (writes .json and .png)")
  ap.add_argument("--task", default=TASK)
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--randomize", action="store_true")
  ap.add_argument("--bin", type=float, default=1.0, help="report bin width (s)")
  ap.add_argument("--device", default=None)
  args = ap.parse_args()

  import src.tasks  # noqa: F401  (registers tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.commands["motion"].motion_file = args.motion
  if not args.randomize:
    cfg.events = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  agent_cfg = load_rl_cfg(args.task)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  cmd = env.command_manager.get_term("motion")
  fps = float(1.0 / env.step_dt)
  n_frames = int(cmd.motion.joint_pos.shape[0])
  term_names = list(env.termination_manager.active_terms)

  n = args.num_envs
  alive = np.ones(n, dtype=bool)
  died_at = [None] * n
  died_by = [None] * n
  samples = []  # (t, metrics dict of np arrays over alive envs)

  # Right after a reset the command's body_pos_relative_w is still all zeros
  # (filled on its first update), so ee_body_pos fires spuriously on the first
  # step and the env resets to t=0. Take that step and ignore its termination.
  venv.reset()
  with torch.inference_mode():
    obs, *_ = venv.step(policy(venv.get_observations()))
  with torch.inference_mode():
    for _ in range(n_frames - 1):
      t_now = cmd.time_steps.cpu().numpy() / fps
      obs, _, dones, _ = venv.step(policy(obs))
      # Termination terms are evaluated before the auto-reset, so their values
      # still describe the step that just ended.
      fired = {k: env.termination_manager.get_term(k).cpu().numpy().astype(bool)
               for k in term_names}
      d = dones.cpu().numpy().astype(bool)
      vals = {k: cmd.metrics[k].cpu().numpy() for k in METRICS if k in cmd.metrics}
      for e in np.where(alive & ~d)[0]:
        samples.append((float(t_now[e]) + 1.0 / fps, {k: float(v[e]) for k, v in vals.items()}))
      for e in np.where(alive & d)[0]:
        alive[e] = False
        died_at[e] = float(t_now[e]) + 1.0 / fps
        died_by[e] = [k for k in term_names if fired[k][e] and k != "time_out"] or ["time_out"]
      if not alive.any():
        break

  duration = (n_frames - 1) / fps
  bins = {}
  for t, m in samples:
    bins.setdefault(int(t // args.bin), []).append(m)
  per_bin = []
  for b, ms in sorted(bins.items()):
    row = {"t0": b * args.bin, "envs": len(ms)}
    row.update({k: float(np.mean([m[k] for m in ms])) for k in ms[0]})
    per_bin.append(row)
  terminations = [{"env": e, "t": died_at[e], "terms": died_by[e]} for e in range(n) if died_at[e] is not None]
  report = {
    "checkpoint": args.checkpoint, "motion": args.motion, "num_envs": n,
    "randomize": args.randomize, "motion_duration_s": duration,
    "completed_envs": int(sum(1 for e in range(n) if died_at[e] is None)),
    "terminations": terminations,
    "mean": {k: float(np.mean([m[k] for _, m in samples])) for k in METRICS if samples and k in samples[0][1]},
    "worst_bins": sorted(per_bin, key=lambda r: -r.get("error_body_pos", 0))[:5],
    "per_bin": per_bin,
  }
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.with_suffix(".json").write_text(json.dumps(report, indent=1))
  plot(report, out.with_suffix(".png"))

  print(f"completed {report['completed_envs']}/{n} envs over {duration:.1f}s of motion")
  for tr in terminations:
    print(f"  env {tr['env']} terminated at {tr['t']:.2f}s by {','.join(tr['terms'])}")
  print("mean: " + ", ".join(f"{k}={v:.3f}" for k, v in report["mean"].items()))
  print("worst bins (by error_body_pos): " +
        ", ".join(f"{b['t0']:.0f}s={b.get('error_body_pos', 0):.3f}" for b in report["worst_bins"]))
  print(f"wrote {out.with_suffix('.json')} and .png")
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
