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
  - --guard runs the fall guard of deploy/include/FallGuard.h (same signals and
    thresholds): once the robot has left the states training kept it in, the
    policy's actions mean nothing and it would thrash on the floor, so the
    motors go to damping (kp 0, kd 3, the Passive state) for good. --push-at
    knocks the robot over to test it; --after-fall keeps simulating that long
    after a fall (default: stop at the fall) and reports how much it thrashes.

The physics model is the exact model used for training (compiled from the mjlab
scene config: same armature, collision geoms, solver options), stepped with the
plain MuJoCo C library rather than mujoco-warp.

Usage:
  python src/unitree_rl_mjlab/scripts/sim2sim.py \
    --policy-dir output/<video>/deploy/<video> \
    --motion output/<video>/deploy/<video>/params/dance1_subject2.npz \
    --duration 30 --start velocity --video out.mp4 --report out.json
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
# deploy/include/FallGuard.h's defaults (deploy/robots/g1/config/config.yaml may override them).
FALL_GUARD = {"anchor_gravity_z": 0.8, "tilt_rad": 1.0, "joint_error_rad": 0.5, "joint_error_steps": 5,
              "joint_speed_ratio": 1.0, "joint_speed_steps": 2}
# Motor speed limits (rad/s) in SDK joint order: legs, waist, arms (validate_motion.py VELOCITY_LIMITS).
SPEED_LIMITS = np.array([32, 20, 32, 20, 37, 37] * 2 + [32, 37, 37] + [37, 37, 37, 37, 37, 22, 22] * 2, float)
DAMPING_KD = 3.0  # the Passive state's kd
# The policy of the Velocity state, the only one the dances are entered from on the robot.
VELOCITY_POLICY = pathlib.Path(__file__).resolve().parents[1] / "deploy/robots/g1/config/policy/velocity/v0"


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
    self.set_gains(deploy)

  def set_gains(self, deploy):
    """A state's PD gains, as the C++ state writes them to the motors on entry."""
    ids = [int(i) for i in deploy["joint_ids_map"]]
    assert ids == list(range(29)), "non-identity joint_ids_map not supported"
    kp = np.asarray(deploy["stiffness"], dtype=np.float64)
    kd = np.asarray(deploy["damping"], dtype=np.float64)
    for i, a in enumerate(self.act):
      # position actuator: tau = kp*(ctrl - q) - kd*dq, clipped to forcerange (effort limit)
      self.m.actuator_gainprm[a, 0] = kp[i]
      self.m.actuator_biasprm[a, 1] = -kp[i]
      self.m.actuator_biasprm[a, 2] = -kd[i]

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
def assemble(obs_cfg, terms):
  """The policy input: the deploy.yaml terms in their order (== C++ order), clipped and scaled."""
  parts = []
  for name, c in obs_cfg.items():
    v = np.asarray(terms[name], dtype=np.float32)
    if c.get("clip") is not None:
      v = np.clip(v, c["clip"][0], c["clip"][1])
    if c.get("scale") is not None:
      v = v * np.asarray(c["scale"], dtype=np.float32)
    parts.append(v)
  return np.concatenate(parts).astype(np.float32)


class VelocityController:
  """State_RLBase running the Velocity policy with the sticks released: a zero command, which also
  zeroes the gait phase (observations.h gait_phase), so the robot stands still."""

  def __init__(self, policy_dir=VELOCITY_POLICY):
    policy_dir = pathlib.Path(policy_dir)
    self.deploy = yaml.safe_load((policy_dir / "params" / "deploy.yaml").read_text())
    self.sess = ort.InferenceSession(str(policy_dir / "exported" / "policy.onnx"), providers=["CPUExecutionProvider"])
    self.in_name = self.sess.get_inputs()[0].name
    act = self.deploy["actions"]["JointPositionAction"]
    assert act.get("clip") is None
    self.scale = np.asarray(act["scale"], dtype=np.float32)
    self.offset = np.asarray(act["offset"], dtype=np.float32)
    self.default = np.asarray(self.deploy["default_joint_pos"], dtype=np.float32)
    self.last_action = np.zeros(29, dtype=np.float32)

  def step(self, robot):
    obs = assemble(self.deploy["observations"], {
      "base_ang_vel": robot.gyro,
      "projected_gravity": qmat(robot.root_quat).T @ np.array([0, 0, -1.0]),
      "velocity_commands": np.zeros(3),
      "gait_phase": np.zeros(2),
      "joint_pos_rel": robot.q - self.default,
      "joint_vel_rel": robot.dq,
      "last_action": self.last_action,
    })
    self.last_action = self.sess.run(None, {self.in_name: obs[None]})[0][0].astype(np.float32)
    return self.last_action * self.scale + self.offset


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
    return assemble(self.obs_cfg, {
      "motion_command": np.concatenate([self.ref_q[frame], self.ref_dq[frame]]),
      "motion_anchor_ori_b": self.anchor_ori_b(robot, frame),
      "base_ang_vel": robot.gyro,
      "joint_pos_rel": robot.q - self.default,
      "joint_vel_rel": robot.dq,
      "last_action": self.last_action,
      "projected_gravity": qmat(robot.root_quat).T @ np.array([0, 0, -1.0]),
    })

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


class FallGuard:
  """deploy/include/FallGuard.h: has the robot left the states training kept it in?

  The training env ends an episode once the torso's tilt differs from the reference's by more than its
  anchor_ori threshold (0.8 on the gravity z component), or the torso or a hand or foot is 25 cm off in
  height, so the policy never learned what to do beyond that. The robot can measure the tilt (IMU and
  waist encoders) but not heights, so heights are stood in for by the joints: their mean distance from
  the reference's, and their speeds against the motor limits (a thrashing robot shows both).
  """

  def __init__(self, cfg=None):
    self.cfg = {**FALL_GUARD, **(cfg or {})}
    self.errors, self.fast = [], 0

  def check(self, robot, ctrl, frame):
    """The reason to stop the policy, or None."""
    c = self.cfg
    up = qmat(waist_torso(robot.root_quat, robot.q))[2, 2]
    ref_up = qmat(waist_torso(ctrl.ref_root_quat[frame], ctrl.ref_q[frame]))[2, 2]
    if abs(up - ref_up) > c["anchor_gravity_z"]:
      return "anchor_gravity"
    if math.acos(float(np.clip(qmat(robot.root_quat)[2, 2], -1, 1))) > c["tilt_rad"]:
      return "tilt"
    self.errors = (self.errors + [float(np.abs(robot.q - ctrl.ref_q[frame]).mean())])[-c["joint_error_steps"]:]
    if len(self.errors) == c["joint_error_steps"] and min(self.errors) > c["joint_error_rad"]:
      return "joint_error"
    self.fast = self.fast + 1 if (np.abs(robot.dq) > c["joint_speed_ratio"] * SPEED_LIMITS).any() else 0
    if self.fast >= c["joint_speed_steps"]:
      return "joint_speed"
    return None

  def signals(self, robot, ctrl, frame):
    up = qmat(waist_torso(robot.root_quat, robot.q))[2, 2]
    ref_up = qmat(waist_torso(ctrl.ref_root_quat[frame], ctrl.ref_q[frame]))[2, 2]
    return dict(guard_anchor_gravity=float(abs(up - ref_up)),
                guard_tilt=math.acos(float(np.clip(qmat(robot.root_quat)[2, 2], -1, 1))),
                guard_speed_ratio=float((np.abs(robot.dq) / SPEED_LIMITS).max()))


def damp(robot):
  """The Passive state: no stiffness, kd damping, on every motor."""
  for a in robot.act:
    robot.m.actuator_gainprm[a, 0] = 0
    robot.m.actuator_biasprm[a, 1] = 0
    robot.m.actuator_biasprm[a, 2] = -DAMPING_KD


def thrashing(rows, since):
  """How much the joints move after `since` (s): mean and p99 |dq|, and fast reversals per joint-second."""
  after = [r for r in rows if r["t"] >= since]
  if len(after) < 3:
    return None
  dq = np.array([r["dq"] for r in after])
  fast = np.abs(dq) > 1.0
  flips = ((np.sign(dq[1:]) != np.sign(dq[:-1])) & fast[1:] & fast[:-1]).sum()
  seconds = after[-1]["t"] - after[0]["t"]
  return dict(seconds=round(seconds, 2), mean_abs_dq=float(np.abs(dq).mean()), p99_abs_dq=float(np.percentile(np.abs(dq), 99)),
              fast_reversals_per_joint_s=float(flips / max(seconds, 1e-9) / dq.shape[1]))


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
      leg_joint_pos_err=float(np.abs(r.q[:12] - c.ref_q[frame][:12]).mean()),
      torso_up_z=float(torso_up[2]),
      dq=r.dq.tolist(),
    ))
    return self.rows[-1]


def summarize(rows, fell_at, bin_s=1.0, joint_names=None):
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
    peak_joint_speed=peak_joint_speed(rows, fell_at, joint_names),
  )


def peak_joint_speed(rows, fell_at, joint_names=None):
  """The fastest joint against its motor limit before any fall: the fall guard trips past 1.0 for two steps,
  and the motors cannot reach it (the training actuators have no speed-torque curve, so the simulation can)."""
  dancing = [r for r in rows if fell_at is None or r["t"] < fell_at] or rows
  ratio = np.abs(np.array([r["dq"] for r in dancing])) / SPEED_LIMITS
  step, joint = np.unravel_index(ratio.argmax(), ratio.shape)
  return dict(ratio=float(ratio[step, joint]), joint=joint_names[joint] if joint_names else int(joint),
              t=dancing[step]["t"], steps_over_limit=int((ratio > 1).any(axis=1).sum()))


# ---------------------------------------------------------------- main
def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--policy-dir", required=True)
  ap.add_argument("--motion", required=True)
  ap.add_argument("--duration", type=float, default=None, help="seconds of motion (default: whole clip)")
  ap.add_argument("--time-start", type=float, default=0.0)
  ap.add_argument("--start", choices=["velocity", "default", "reference"], default="velocity",
                  help="velocity: stand with the Velocity policy first, as the robot does before a dance "
                       "starts (Velocity is the only state with transitions to the dances); default: settle "
                       "in the Mimic policy's default stance, a deeper crouch the robot never starts from; "
                       "reference: start exactly in the first reference frame")
  ap.add_argument("--settle", type=float, default=1.0, help="seconds of PD hold before --start velocity/default")
  ap.add_argument("--stand", type=float, default=2.0, help="seconds of the Velocity policy before --start velocity")
  ap.add_argument("--video", default=None)
  ap.add_argument("--report", default=None, help="JSON report path (a .png plot is written next to it)")
  ap.add_argument("--fall-height", type=float, default=0.35)
  ap.add_argument("--guard", action="store_true", help="run the deployment's fall guard (FallGuard.h)")
  ap.add_argument("--after-fall", type=float, default=0.0, help="seconds to keep simulating after a fall")
  ap.add_argument("--push-at", type=float, default=None, help="knock the robot over at this time (s)")
  ap.add_argument("--push-speed", type=float, default=2.0, help="horizontal speed the push adds (m/s)")
  ap.add_argument("--push-heading", type=float, default=0.0, help="push direction, degrees from +x")
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
  elif args.start == "default":
    robot.set_state(np.array([0, 0, 0.8]), np.array([1.0, 0, 0, 0]), ctrl.default)
    robot.set_targets(ctrl.default)
    for _ in range(int(args.settle / model.opt.timestep)):
      mujoco.mj_step(model, robot.d)
  else:
    # FixStand's pose is the Velocity policy's default; hold it, then let the policy stand.
    velocity = VelocityController()
    robot.set_gains(velocity.deploy)
    robot.set_state(np.array([0, 0, 0.8]), np.array([1.0, 0, 0, 0]), velocity.default)
    robot.set_targets(velocity.default)
    for _ in range(int(args.settle / model.opt.timestep)):
      mujoco.mj_step(model, robot.d)
    for _ in range(round(args.stand / ctrl.step_dt)):
      robot.set_targets(velocity.step(robot))
      for _ in range(decimation):
        mujoco.mj_step(model, robot.d)
    robot.set_gains(deploy)
    robot.set_targets(robot.q)  # State_Mimic::enter holds the current joint positions

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
  guard = FallGuard() if args.guard else None
  fell_at = guarded_at = guard_reason = None
  pushed = False
  while True:
    t = ctrl.k * ctrl.step_dt
    if fell_at is not None and t >= fell_at + args.after_fall:
      break
    if args.push_at is not None and not pushed and t >= args.push_at:
      heading = math.radians(args.push_heading)
      robot.d.qvel[:2] += args.push_speed * np.array([math.cos(heading), math.sin(heading)])
      pushed = True
    if guarded_at is None:
      q_des, frame, done = ctrl.step(robot)
      if done:
        break
      robot.set_targets(q_des)
    else:
      ctrl.k += 1  # damping: no policy, the clock goes on
      frame = min(frame + 1, ctrl.num_frames - 1)
    for _ in range(decimation):
      mujoco.mj_step(model, robot.d)
    t = ctrl.k * ctrl.step_dt
    row = tracker.record(t, frame)
    if guard is not None:
      row.update(guard.signals(robot, ctrl, frame))
      if guarded_at is None:
        guard_reason = guard.check(robot, ctrl, frame)
        if guard_reason:
          guarded_at = t
          damp(robot)
    if renderer is not None:
      renderer.update_scene(robot.d, cam)
      frames.append(renderer.render())
    if fell_at is None and (row["pelvis_z"] < args.fall_height or row["torso_up_z"] < 0.3):
      fell_at = t

  summary = summarize(tracker.rows, fell_at, joint_names=robot.joint_names)
  summary["guard"] = dict(enabled=guard is not None, triggered_at_s=guarded_at, reason=guard_reason,
                          before_fall_s=None if guarded_at is None or fell_at is None else round(fell_at - guarded_at, 3))
  if fell_at is not None:
    summary["after_fall"] = thrashing(tracker.rows, fell_at)
  if guard is not None:
    summary["guard_signals_max_before_trigger"] = {
      k: float(max(r[k] for r in tracker.rows if guarded_at is None or r["t"] < guarded_at))
      for k in ("guard_anchor_gravity", "guard_tilt", "guard_speed_ratio", "joint_pos_err")}
  print(f"simulated {summary['simulated_s']:.2f}s of {ctrl.time_end - ctrl.time_start:.2f}s | "
        f"fell: {fell_at is not None}" + (f" at {fell_at:.2f}s" if fell_at else "") +
        " | mean errors: " + ", ".join(f"{k}={v:.3f}" for k, v in summary["mean"].items()))
  print(f"steps over the training env's tracking-termination thresholds: "
        f"{summary['frac_steps_over_train_termination'] * 100:.1f}% "
        f"(first at {summary['first_train_termination_s']})")
  peak = summary["peak_joint_speed"]
  print(f"peak joint speed: {peak['ratio']:.2f}x the motor limit ({peak['joint']} at {peak['t']:.2f}s), "
        f"{peak['steps_over_limit']} steps over it")
  if guard is not None:
    print(f"fall guard: {guard_reason or 'not triggered'}" + (f" at {guarded_at:.2f}s" if guarded_at else ""))
  if summary.get("after_fall"):
    print("after the fall: " + ", ".join(f"{k}={v:.2f}" for k, v in summary["after_fall"].items()))

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
