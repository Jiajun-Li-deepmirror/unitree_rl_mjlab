"""Per-limb diagnostics using synchronized, pre-termination rollout samples."""
from pathlib import Path

import numpy as np


LEG_PARTS = ("hip", "knee", "ankle")


def summarize(data):
  """Errors use every valid sample; amplitude gains are computed per environment."""
  ref, actual = data["reference_joint_pos"], data["policy_joint_pos"]
  if not len(ref):
    return {"samples": 0, "joint_groups": {}, "joints": {}, "feet": {}}
  names = data["joint_names"].tolist()
  err = actual - ref
  groups = {"legs": [i for i, n in enumerate(names) if any(p in n for p in LEG_PARTS)],
            "arms": [i for i, n in enumerate(names) if any(p in n for p in ("shoulder", "elbow", "wrist"))]}
  report = {"samples": len(ref), "joint_groups": {}, "joints": {}, "feet": {}}
  for group, ids in groups.items():
    if not ids:
      continue
    error = err[:, ids]
    # A motion-only mask: top 20% excursion of these joints from their reference
    # median. It cannot improve by shrinking the policy's movement.
    center = np.median(ref[:, ids], axis=0)
    excursion = np.sqrt(np.mean((ref[:, ids] - center)**2, axis=-1))
    high = excursion >= np.quantile(excursion, .8)
    report["joint_groups"][group] = {
      "mae_rad": float(np.mean(np.abs(error))),
      "rmse_rad": float(np.sqrt(np.mean(error**2))),
      "high_excursion_rmse_rad": float(np.sqrt(np.mean(error[high]**2))),
      "high_excursion_threshold_rad": float(np.quantile(excursion, .8)),
      "high_excursion_samples": int(high.sum()),
    }
  for j, name in enumerate(names):
    gains = []
    for e in np.unique(data["env_id"]):
      mask = data["env_id"] == e
      x, y = ref[mask, j], actual[mask, j]
      x, y = x - x.mean(), y - y.mean()
      if np.mean(x*x) > 1e-5:
        gains.append(float(np.dot(x, y) / np.dot(x, x)))
    report["joints"][name] = {
      "mae_rad": float(np.mean(np.abs(err[:, j]))),
      "rmse_rad": float(np.sqrt(np.mean(err[:, j]**2))),
      "p95_abs_error_rad": float(np.quantile(np.abs(err[:, j]), .95)),
      "reference_range_rad": float(np.ptp(ref[:, j])),
      "policy_range_rad": float(np.ptp(actual[:, j])),
      "centered_tracking_gain": float(np.mean(gains)) if gains else None,
    }
  for i, name in enumerate(data["body_names"]):
    if "ankle_roll" not in name:
      continue
    target = data["reference_body_pos_aligned"][:, i]
    actual = data["policy_body_pos_world"][:, i]
    difference = actual - target
    high = target[:, 2] >= np.quantile(target[:, 2], .8)
    report["feet"][str(name)] = {
      "aligned_position_rmse_m": float(np.sqrt(np.mean(np.sum(difference**2, axis=-1)))),
      "height_rmse_m": float(np.sqrt(np.mean(difference[:, 2]**2))),
      "high_lift_height_bias_m": float(np.mean(difference[high, 2])),
      "high_lift_height_rmse_m": float(np.sqrt(np.mean(difference[high, 2]**2))),
      "reference_peak_height_m": float(target[:, 2].max()),
      "policy_height_at_reference_peak_m": float(actual[target[:, 2].argmax(), 2]),
    }
  if "actuator_force" in data:
    force, limits = np.abs(data["actuator_force"]), data["actuator_force_limits"]
    limited = limits > 0
    report["actuator_saturation"] = {
      str(name): float(np.mean(force[:, i] >= .95 * limits[i]))
      for i, name in enumerate(data["actuator_names"]) if limited[i]}
  report["definitions"] = {
    "angles": "per-joint radians, not the norm of a joint vector",
    "high_excursion": "top 20% reference joint RMS excursion from reference median",
    "centered_tracking_gain": "same-time least-squares gain after removing per-env mean; 1 ideal, <1 attenuation or lag; not proof of a motor limit",
    "feet": "ankle body origins, yaw/XY anchor-aligned; Z stays in world/ground coordinates",
    "actuator_saturation": "fraction of 50 Hz samples at >=95% model force limit; does not capture every physics substep",
    "validity": "initial state and automatically reset terminal states excluded; incomplete rollouts must not be compared as complete motions",
  }
  return report


class TrackingDiagnostics:
  def __init__(self):
    self.rows = []
    self.metadata = {}

  def __call__(self, env, cmd, t, valid):
    if t <= 0 or not valid.any():
      return
    def array(tensor):
      return tensor.detach().cpu().numpy()
    if not self.metadata:
      import mujoco
      ids = array(cmd.robot.indexing.ctrl_ids)
      model = env.sim.mj_model
      self.metadata = {
        "joint_names": np.asarray(cmd.robot.joint_names),
        "body_names": np.asarray(cmd.cfg.body_names),
        "actuator_names": np.asarray([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)) or str(i) for i in ids]),
        "actuator_force_limits": np.max(np.abs(model.actuator_forcerange[ids]), axis=-1) * model.actuator_forcelimited[ids],
      }
    values = {
      "reference_joint_pos": array(cmd.joint_pos),
      "policy_joint_pos": array(cmd.robot_joint_pos),
      "reference_body_pos_aligned": array(cmd.body_pos_relative_w - env.scene.env_origins[:, None, :]),
      "policy_body_pos_world": array(cmd.robot_body_pos_w - env.scene.env_origins[:, None, :]),
      "reference_body_pos_world": array(cmd.body_pos_w - env.scene.env_origins[:, None, :]),
      "actuator_force": array(cmd.robot.data.actuator_force),
    }
    for e in np.flatnonzero(valid):
      self.rows.append({"t": t, "env_id": e, **{k: v[e].copy() for k, v in values.items()}})

  def data(self):
    if not self.rows:
      return {"reference_joint_pos": np.empty((0, 0)), "policy_joint_pos": np.empty((0, 0))}
    return {**self.metadata, **{k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}}

  def save(self, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = self.data()
    np.savez_compressed(path, **data)
    if self.rows:
      plot(data, path.with_name(path.stem + "_legs.png"))


def plot(data, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  mask = data["env_id"] == 0
  t = data["t"][mask]
  if not len(t):
    return
  fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
  for side, column in (("left", 0), ("right", 1)):
    for row, part in enumerate(("hip_pitch", "hip_roll", "knee")):
      name = f"{side}_{part}_joint"
      j = data["joint_names"].tolist().index(name)
      ax = axes[row, column]
      ax.plot(t, np.rad2deg(data["reference_joint_pos"][mask, j]), color="#69b9ee", label="Reference")
      ax.plot(t, np.rad2deg(data["policy_joint_pos"][mask, j]), color="#e27933", label="Policy", lw=1)
      ax.set_ylabel("degrees")
      ax.set_title(name)
    i = data["body_names"].tolist().index(f"{side}_ankle_roll_link")
    ax = axes[3, column]
    ax.plot(t, data["reference_body_pos_aligned"][mask, i, 2]*100, color="#69b9ee", label="Reference")
    ax.plot(t, data["policy_body_pos_world"][mask, i, 2]*100, color="#e27933", label="Policy", lw=1)
    ax.set_title(f"{side} ankle height")
    ax.set_ylabel("cm")
    ax.set_xlabel("time (s)")
  for ax in axes.flat:
    ax.legend(loc="upper right")
    ax.grid(alpha=.2)
  fig.suptitle("Leg tracking — synchronized samples before first termination (env 0)")
  fig.tight_layout()
  fig.savefig(path, dpi=140)
  plt.close(fig)
