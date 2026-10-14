"""Sim2sim: run a deployed Mimic policy the way the C++ controller does, in plain CPU MuJoCo.

Mirrors deploy/robots/g1/src/State_Mimic.cpp + deploy/include/isaaclab/*:
  - observations (order and definitions from deploy.yaml / the C++ terms):
      motion_command      ref joint_pos[frame] ++ ref joint_vel[frame]
      motion_anchor_ori_b first two columns of R(robot_torso^-1 * init_quat * ref_torso)
                          where "torso" = root_quat * Rz(q12) * Rx(q13) * Ry(q14)
      base_ang_vel        IMU gyro (pelvis angular velocity, body frame)
      joint_pos_rel       q - default_joint_pos
      joint_vel_rel       dq
      last_action         previous raw policy output
  - init_quat = yaw(robot torso) * yaw(ref torso)^-1, computed once at state entry
  - motion frame = floor((k * step_dt + time_start) / (1/50)), k = policy steps so far
  - action: q_des = a * scale + offset (+ clip), PD with deploy.yaml stiffness/damping,
    dq_des = 0, held between 50 Hz policy updates; motor effort limits as in training
  - C++ deployment also checks tilt, stale state and policy deadlines; this
    standalone diagnostic is not the C++/DDS deployment acceptance runner.

The physics model is the exact model used for training (compiled from the mjlab
scene config: same armature, collision geoms, solver options), stepped with the
plain MuJoCo C library rather than mujoco-warp.

Usage:
  python src/unitree_rl_mjlab/scripts/sim2sim.py \
    --policy-dir output/<video>/deploy/<video> \
    --motion output/<video>/deploy/<video>/params/dance1_subject2.npz \
    --duration 30 --start default --video out.mp4 --report out.json
"""
import argparse
import json
import math
import os
import pathlib
import sys

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

MOTION_DT = 1.0 / 50.0
TRACKED_BODIES = (
  "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
EE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
             "left_wrist_yaw_link", "right_wrist_yaw_link")


# ---------------------------------------------------------------- quaternions (wxyz)
def qmul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ])


def qconj(q):
  return np.array([q[0], -q[1], -q[2], -q[3]])


def qaxis(axis, angle):
  s = math.sin(angle / 2)
  v = np.zeros(3)
  v["xyz".index(axis)] = s
  return np.array([math.cos(angle / 2), *v])


def qmat(q):
  m = np.zeros(9)
  mujoco.mju_quat2Mat(m, q / np.linalg.norm(q))
  return m.reshape(3, 3)


def yaw_quat(q):
  w, x, y, z = q
  yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def waist_torso(root_quat, joint_pos):
  """C++ robot_quat_w / motion_anchor_quat_w: root * Rz(q12) * Rx(q13) * Ry(q14)."""
  q = qmul(root_quat, qaxis("z", joint_pos[12]))
  q = qmul(q, qaxis("x", joint_pos[13]))
  return qmul(q, qaxis("y", joint_pos[14]))


# ---------------------------------------------------------------- model
def build_model():
  """The training physics model, compiled from the mjlab scene config, on CPU."""
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation
  from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True)
  scene = Scene(cfg.scene, device="cpu")
  sim = Simulation(num_envs=1, cfg=cfg.sim, model=scene.compile(), device="cpu")
  return sim.mj_model, cfg.decimation


class Robot:
  def __init__(self, model, deploy):
    self.m = model
    self.d = mujoco.MjData(model)
    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    self.hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    assert len(self.hinges) == 29, len(self.hinges)
    self.joint_names = [jnames[j].split("/")[-1] for j in self.hinges]
    self.qadr = np.array([model.jnt_qposadr[j] for j in self.hinges])
    self.vadr = np.array([model.jnt_dofadr[j] for j in self.hinges])
    act_of_joint = {model.actuator_trnid[a, 0]: a for a in range(model.nu)}
    self.act = np.array([act_of_joint[j] for j in self.hinges])
    self.body = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b).split("/")[-1]: b
                 for b in range(model.nbody)}
    self.pelvis = self.body["pelvis"]

    # Policy joint i <-> SDK motor joint_ids_map[i]; deploy gains are in SDK order.
    # Here the model's joint order is the SDK order (identity map in deploy.yaml).
    ids = [int(i) for i in deploy["joint_ids_map"]]
    assert ids == list(range(29)), "non-identity joint_ids_map not supported"
    kp = np.asarray(deploy["stiffness"], dtype=np.float64)
    kd = np.asarray(deploy["damping"], dtype=np.float64)
    for i, a in enumerate(self.act):
      # position actuator: tau = kp*(ctrl - q) - kd*dq, clipped to forcerange (effort limit)
      model.actuator_gainprm[a, 0] = kp[i]
      model.actuator_biasprm[a, 1] = -kp[i]
      model.actuator_biasprm[a, 2] = -kd[i]

  # state accessors ------------------------------------------------------
  @property
  def q(self):
    return self.d.qpos[self.qadr].copy()

  @property
  def dq(self):
    return self.d.qvel[self.vadr].copy()

  @property
  def root_quat(self):
    return self.d.qpos[3:7].copy()

  @property
  def gyro(self):
    return self.d.qvel[3:6].copy()  # free joint angular velocity is in the body frame

  def set_targets(self, q_des):
    self.d.ctrl[self.act] = q_des

  def set_state(self, root_pos, root_quat, q, dq=None, root_lin_vel=None, root_ang_vel_w=None):
    self.d.qpos[:3] = root_pos
    self.d.qpos[3:7] = root_quat
    self.d.qpos[self.qadr] = q
    self.d.qvel[:] = 0
    if dq is not None:
      self.d.qvel[self.vadr] = dq
    if root_lin_vel is not None:
      self.d.qvel[:3] = root_lin_vel
    if root_ang_vel_w is not None:
      self.d.qvel[3:6] = qmat(root_quat).T @ root_ang_vel_w
    mujoco.mj_forward(self.m, self.d)


# ---------------------------------------------------------------- controller
class MimicController:
  def __init__(self, policy_dir, motion_path, deploy, time_start=0.0, time_end=1e9):
    onnx = pathlib.Path(policy_dir) / "exported" / "policy.onnx"
    self.sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    self.in_name = self.sess.get_inputs()[0].name
    self.in_dim = self.sess.get_inputs()[0].shape[-1]

    self.step_dt = float(deploy["step_dt"])
    act = deploy["actions"]["JointPositionAction"]
    self.scale = np.asarray(act["scale"], dtype=np.float32)
    self.offset = np.asarray(act["offset"], dtype=np.float32)
    self.clip = None if act.get("clip") is None else np.asarray(act["clip"], dtype=np.float32)
    self.default = np.asarray(deploy["default_joint_pos"], dtype=np.float32)
    self.obs_cfg = deploy["observations"]
    for name, c in self.obs_cfg.items():
      if c.get("history_length", 1) != 1:
        raise NotImplementedError(f"history_length != 1 for {name}")

    mot = np.load(motion_path)
    self.ref_root_pos = mot["body_pos_w"][:, 0].astype(np.float64)
    self.ref_root_quat = mot["body_quat_w"][:, 0].astype(np.float64)  # wxyz
    self.ref_q = mot["joint_pos"].astype(np.float32)
    self.ref_dq = mot["joint_vel"].astype(np.float32)
    self.ref_body_pos = mot["body_pos_w"].astype(np.float64)
    self.num_frames = self.ref_q.shape[0]
    self.duration = self.num_frames * MOTION_DT
    self.time_start = float(np.clip(time_start, 0, self.duration))
    self.time_end = float(np.clip(time_end, 0, self.duration))

  def frame_at(self, t):
    t = min(max(t, 0.0), self.duration)
    return min(int(math.floor(t / MOTION_DT)), self.num_frames - 1)

  def enter(self, robot):
    """State_Mimic::enter: zero actions, yaw-align the reference to the robot."""
    self.k = 0
    self.last_action = np.zeros(29, dtype=np.float32)
    f = self.frame_at(self.time_start)
    ref_yaw = yaw_quat(waist_torso(self.ref_root_quat[f], self.ref_q[f]))
    robot_yaw = yaw_quat(waist_torso(robot.root_quat, robot.q))
    self.init_quat = qmul(robot_yaw, qconj(ref_yaw))

  def observation(self, robot, frame):
    terms = {
      "motion_command": np.concatenate([self.ref_q[frame], self.ref_dq[frame]]),
      "motion_anchor_ori_b": self.anchor_ori_b(robot, frame),
      "base_ang_vel": robot.gyro,
      "joint_pos_rel": robot.q - self.default,
      "joint_vel_rel": robot.dq,
      "last_action": self.last_action,
      "projected_gravity": qmat(robot.root_quat).T @ np.array([0, 0, -1.0]),
    }
    parts = []
    for name, c in self.obs_cfg.items():  # yaml order == C++ order
      v = np.asarray(terms[name], dtype=np.float32)
      if c.get("clip") is not None:
        v = np.clip(v, c["clip"][0], c["clip"][1])
      if c.get("scale") is not None:
        v = v * np.asarray(c["scale"], dtype=np.float32)
      parts.append(v)
    return np.concatenate(parts).astype(np.float32)

  def anchor_ori_b(self, robot, frame):
    real = waist_torso(robot.root_quat, robot.q)
    ref = waist_torso(self.ref_root_quat[frame], self.ref_q[frame])
    rot = qmat(qmul(qconj(qmul(self.init_quat, ref)), real)).T
    return np.array([rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]])

  def step(self, robot):
    """One 50 Hz policy step. Returns (q_des, frame, done)."""
    t = self.k * self.step_dt + self.time_start
    if self.k >= round((self.time_end - self.time_start) / self.step_dt):
      return None, None, True
    frame = min(self.frame_at(self.time_start) + self.k, self.num_frames - 1)
    obs = self.observation(robot, frame)
    assert obs.shape[0] == self.in_dim, (obs.shape, self.in_dim)
    a = self.sess.run(None, {self.in_name: obs[None]})[0][0].astype(np.float32)
    self.last_action = a
    q_des = a * self.scale + self.offset
    if self.clip is not None:
      q_des = np.clip(q_des, self.clip[:, 0], self.clip[:, 1])
    self.k += 1
    return q_des, frame, False


# ---------------------------------------------------------------- metrics
class Tracker:
  """Compares the robot with the reference, expressed in the robot's world frame."""

  def __init__(self, robot, ctrl, frame0):
    self.robot, self.ctrl = robot, ctrl
    self.R = qmat(ctrl.init_quat)
    self.p_ref0 = ctrl.ref_root_pos[frame0]
    self.p_rob0 = robot.d.xpos[robot.pelvis].copy()
    self.idx = [robot.body[n] - robot.body["pelvis"] for n in TRACKED_BODIES]  # npz body index
    self.rows = []

  def ref_world(self, p):
    out = self.R @ (p - self.p_ref0)
    out[:2] += self.p_rob0[:2]
    out[2] = p[2]
    return out

  def record(self, t, frame):
    r, c = self.robot, self.ctrl
    ref_b = np.array([self.ref_world(c.ref_body_pos[frame, i]) for i in self.idx])
    rob_b = np.array([r.d.xpos[r.body[n]] for n in TRACKED_BODIES])
    a = TRACKED_BODIES.index("torso_link")
    anchor_err = float(np.linalg.norm(rob_b[a] - ref_b[a]))
    rel_err = np.linalg.norm((rob_b - rob_b[a]) - (ref_b - ref_b[a]), axis=1)
    ee = [TRACKED_BODIES.index(n) for n in EE_BODIES]
    torso_up = qmat(r.d.xquat[r.body["torso_link"]])[:, 2]
    ref_torso = waist_torso(c.ref_root_quat[frame], c.ref_q[frame])
    ref_up = qmat(qmul(c.init_quat, ref_torso))[:, 2]
    self.rows.append(dict(
      t=t, frame=frame,
      pelvis_z=float(r.d.xpos[r.pelvis][2]),
      anchor_pos_err=anchor_err,
      anchor_z_err=float(abs(rob_b[a][2] - ref_b[a][2])),
      body_pos_err=float(rel_err.mean()),
      ee_z_err=float(np.abs(rob_b[ee, 2] - ref_b[ee, 2]).max()),
      gravity_dir_err=float(np.linalg.norm(torso_up - ref_up)),
      joint_pos_err=float(np.abs(r.q - c.ref_q[frame]).mean()),
      torso_up_z=float(torso_up[2]),
    ))
    return self.rows[-1]


def summarize(rows, fell_at, bin_s=1.0):
  if not rows:
    return {}
  keys = ["anchor_pos_err", "body_pos_err", "joint_pos_err", "ee_z_err", "anchor_z_err"]
  # Thresholds of the training env's tracking terminations (tracking_env_cfg.py).
  train_term = [r["t"] for r in rows
                if r["anchor_z_err"] > 0.25 or r["ee_z_err"] > 0.25 or r["gravity_dir_err"] > 0.8]
  bins = {}
  for r in rows:
    bins.setdefault(int(r["t"] // bin_s), []).append(r)
  per_bin = [{"t0": b * bin_s, **{k: float(np.mean([x[k] for x in v])) for k in keys}}
             for b, v in sorted(bins.items())]
  return dict(
    simulated_s=rows[-1]["t"],
    fell=fell_at is not None,
    fell_at_s=fell_at,
    mean={k: float(np.mean([r[k] for r in rows])) for k in keys},
    max={k: float(np.max([r[k] for r in rows])) for k in keys},
    first_train_termination_s=train_term[0] if train_term else None,
    frac_steps_over_train_termination=len(train_term) / len(rows),
    worst_bins=sorted(per_bin, key=lambda b: -b["body_pos_err"])[:5],
    per_bin=per_bin,
  )


# ---------------------------------------------------------------- main
def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--policy-dir", required=True)
  ap.add_argument("--motion", required=True)
  ap.add_argument("--duration", type=float, default=None, help="seconds of motion (default: whole clip)")
  ap.add_argument("--time-start", type=float, default=0.0)
  ap.add_argument("--start", choices=["default", "reference"], default="default",
                  help="default: settle in the default stance first (as when entering Mimic from "
                       "Velocity on the robot); reference: start exactly in the first reference frame")
  ap.add_argument("--settle", type=float, default=1.0, help="seconds of PD hold before --start default")
  ap.add_argument("--video", default=None)
  ap.add_argument("--report", default=None, help="JSON report path (a .png plot is written next to it)")
  ap.add_argument("--fall-height", type=float, default=0.35)
  args = ap.parse_args()
  for output in (args.report, args.video):
    if output:
      pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)

  os.environ.setdefault("MUJOCO_GL", "egl")
  deploy = yaml.safe_load(open(pathlib.Path(args.policy_dir) / "params" / "deploy.yaml"))
  model, decimation = build_model()
  assert abs(model.opt.timestep * decimation - float(deploy["step_dt"])) < 1e-9
  robot = Robot(model, deploy)
  time_end = 1e9 if args.duration is None else args.time_start + args.duration
  ctrl = MimicController(args.policy_dir, args.motion, deploy, args.time_start, time_end)

  # Sanity check: npz body order == model body order (FK of frame 0 must match).
  f0 = ctrl.frame_at(ctrl.time_start)
  robot.set_state(ctrl.ref_root_pos[f0], ctrl.ref_root_quat[f0], ctrl.ref_q[f0])
  nb = ctrl.ref_body_pos.shape[1]
  fk_err = np.abs(robot.d.xpos[robot.pelvis:robot.pelvis + nb] - ctrl.ref_body_pos[f0]).max()
  assert fk_err < 1e-3, f"npz body order mismatch (FK error {fk_err})"

  mot = np.load(args.motion)
  if args.start == "reference":
    robot.set_state(ctrl.ref_root_pos[f0], ctrl.ref_root_quat[f0], ctrl.ref_q[f0],
                    ctrl.ref_dq[f0], mot["body_lin_vel_w"][f0, 0], mot["body_ang_vel_w"][f0, 0])
    robot.set_targets(ctrl.ref_q[f0])
  else:
    robot.set_state(np.array([0, 0, 0.8]), np.array([1.0, 0, 0, 0]), ctrl.default)
    robot.set_targets(ctrl.default)
    for _ in range(int(args.settle / model.opt.timestep)):
      mujoco.mj_step(model, robot.d)

  renderer = frames = None
  if args.video:
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = robot.body["torso_link"]
    cam.distance, cam.elevation, cam.azimuth = 3.0, -10.0, 120.0
    frames = []

  ctrl.enter(robot)
  tracker = Tracker(robot, ctrl, f0)
  fell_at = None
  while True:
    q_des, frame, done = ctrl.step(robot)
    if done:
      break
    robot.set_targets(q_des)
    for _ in range(decimation):
      mujoco.mj_step(model, robot.d)
    t = ctrl.k * ctrl.step_dt
    row = tracker.record(t, frame)
    if renderer is not None:
      renderer.update_scene(robot.d, cam)
      frames.append(renderer.render())
    if row["pelvis_z"] < args.fall_height or row["torso_up_z"] < 0.3:
      fell_at = t
      break

  summary = summarize(tracker.rows, fell_at)
  print(f"simulated {summary['simulated_s']:.2f}s of {ctrl.time_end - ctrl.time_start:.2f}s | "
        f"fell: {fell_at is not None}" + (f" at {fell_at:.2f}s" if fell_at else "") +
        " | mean errors: " + ", ".join(f"{k}={v:.3f}" for k, v in summary["mean"].items()))
  print(f"steps over the training env's tracking-termination thresholds: "
        f"{summary['frac_steps_over_train_termination'] * 100:.1f}% "
        f"(first at {summary['first_train_termination_s']})")

  if frames:
    import mediapy
    mediapy.write_video(args.video, frames, fps=1.0 / ctrl.step_dt)
    print(f"wrote {args.video}")
  if args.report:
    with open(args.report, "w") as fh:
      json.dump(dict(policy_dir=str(args.policy_dir), motion=str(args.motion), start=args.start,
                     summary=summary, steps=tracker.rows), fh, indent=1)
    plot(tracker.rows, fell_at, str(pathlib.Path(args.report).with_suffix(".png")))
    print(f"wrote {args.report}")
  return 0 if fell_at is None else 1


def plot(rows, fell_at, path):
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    return
  t = [r["t"] for r in rows]
  fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
  ax[0].plot(t, [r["anchor_pos_err"] for r in rows], label="anchor pos err (m)")
  ax[0].plot(t, [r["body_pos_err"] for r in rows], label="body pos err, rel. to anchor (m)")
  ax[1].plot(t, [r["joint_pos_err"] for r in rows], label="mean |joint err| (rad)")
  ax[2].plot(t, [r["pelvis_z"] for r in rows], label="pelvis z (m)")
  for a in ax:
    a.legend(loc="upper right")
    if fell_at is not None:
      a.axvline(fell_at, color="r", ls="--")
  ax[2].set_xlabel("time (s)")
  fig.tight_layout()
  fig.savefig(path, dpi=90)


if __name__ == "__main__":
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
  sys.exit(main())
