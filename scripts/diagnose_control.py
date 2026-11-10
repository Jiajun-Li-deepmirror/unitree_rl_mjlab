"""Local control/substep diagnosis; results are not training acceptance reports."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from project_paths import project_relative
from eval_tracking import TASK, prepare_eval_cfg
from tracking_targets import initialize_reference_targets
from tracking_diagnostics import TrackingDiagnostics


def retime_motion(motion, speed, dt):
  """Resample reference time only; retain physics/control dt and scale velocities."""
  from scipy.spatial.transform import Rotation, Slerp
  source = {k: v.copy() for k, v in motion.items()}
  if speed == 1:
    return source
  n = len(source['joint_pos'])
  x = np.arange(n) * dt
  t = np.arange(int(np.floor((n - 1) / speed)) + 1) * dt * speed
  result = {}
  for key, value in source.items():
    if key == 'fps' or value.ndim == 0 or len(value) != n:
      result[key] = value
    elif key == 'body_quat_w':
      result[key] = np.stack([Slerp(x, Rotation.from_quat(value[:, b], scalar_first=True))(t)
                             .as_quat(scalar_first=True) for b in range(value.shape[1])], axis=1).astype(value.dtype)
    else:
      flat = value.reshape(n, -1)
      out = np.stack([np.interp(t, x, col) for col in flat.T], axis=-1)
      if key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
        out *= speed
      result[key] = out.reshape((len(t), *value.shape[1:])).astype(value.dtype)
  return result


def array(tensor):
  return tensor.detach().cpu().numpy().copy()


class SubstepRecorder:
  """Observe the unmodified native step without extra forward/solver calls.

  mj_step leaves forces/derived poses at pre-integration time. Joint q/v are
  copied before step, contacts/COM after step, so every row describes that same
  pre-integration instant. Read contact buffers directly to bypass lazy caches.
  """
  def __init__(self, env, speed, window):
    self.env, self.speed, self.window = env, speed, window
    self.cmd = env.command_manager.get_term('motion')
    self.robot = self.cmd.robot
    self.rows = []
    self.tick = 0
    self.original_step = env.sim.step
    model = env.sim.mj_model
    self.ctrl_ids = array(self.robot.indexing.ctrl_ids).astype(int)
    import mujoco
    self.joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                             'robot/' + n) for n in self.robot.joint_names])
    if (self.joint_ids < 0).any():
      raise ValueError('joint names did not resolve')
    self.actuator_joints = model.actuator_trnid[self.ctrl_ids, 0]
    if not np.all(model.actuator_gear[self.ctrl_ids, 0] == 1):
      raise ValueError('diagnostic assumes unit joint transmission gear')
    self.actuator_joint_names = np.array([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)).split('/')[-1]
                                       for j in self.actuator_joints])
    self.metadata = dict(joint_names=np.array(self.robot.joint_names), body_names=np.array(self.robot.body_names),
        actuator_joint_names=self.actuator_joint_names,
        actuator_force_limits=np.max(abs(model.actuator_forcerange[self.ctrl_ids]), axis=-1),
        actuator_kp=model.actuator_gainprm[self.ctrl_ids, 0].copy(),
        actuator_kd=-model.actuator_biasprm[self.ctrl_ids, 2].copy(),
        physics_dt=np.array(env.physics_dt), control_dt=np.array(env.step_dt), speed=np.array(speed))

  def __enter__(self):
    self.env.sim.step = self.step
    return self

  def __exit__(self, *_):
    self.env.sim.step = self.original_step

  def step(self):
    env, robot = self.env, self.robot
    t = self.tick * env.physics_dt
    self.tick += 1
    capture = self.window[0] - 1e-8 <= t * self.speed <= self.window[1] + 1e-8
    if capture:
      pre = dict(t=t, reference_t=t*self.speed,
                 command_reference_t=float(self.cmd.time_steps[0])*env.step_dt*self.speed,
                 joint_pos=array(robot.data.joint_pos[0]), joint_vel=array(robot.data.joint_vel[0]),
                 policy_action=array(env.action_manager.action[0]),
                 pd_target=array(robot.data.joint_pos_target[0]),
                 actuator_ctrl=array(env.sim.data.ctrl[0, self.ctrl_ids]))
    self.original_step()
    if not capture:
      return
    # Current sensor buffers are solver outputs from this substep, before scene.update.
    contacts = [env.scene[f'{s}_foot_ground']._compute_data() for s in ('left', 'right')]
    root = robot.indexing.root_body_id
    pre.update(actuator_force=array(robot.data.actuator_force[0]),
               com_world=array(env.sim.data.subtree_com[0, root]),
               foot_force_n=np.array([float(torch.linalg.vector_norm(c.force[0, 0])) for c in contacts]),
               foot_contact_count=np.array([int(c.found[0, 0]) for c in contacts]),
               foot_contact_pos=np.stack([array(c.pos[0, 0]) for c in contacts]),
               body_pos=array(robot.data.body_link_pos_w[0]))
    self.rows.append(pre)

  def save(self, path):
    data = {**self.metadata, **{k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}}
    np.savez_compressed(path, **data)
    return data


def reference_kinematics(env, motion):
  """CPU forward kinematics only; no dynamic feasibility claim or policy change."""
  import mujoco
  model, robot = env.sim.mj_model, env.scene['robot']
  data = mujoco.MjData(model)
  qids = array(robot.indexing.joint_q_adr).astype(int)
  root = robot.indexing.root_body_id
  body_ids = array(robot.indexing.body_ids).astype(int)
  root_index = robot.body_names.index('pelvis')
  foot_geoms = [[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'robot/{side}_foot{i}_collision')
                 for i in range(1, 8)] for side in ('left', 'right')]
  if any(g < 0 for foot in foot_geoms for g in foot):
    raise ValueError('Expected G1 foot collision geoms')
  if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE for foot in foot_geoms for g in foot):
    raise ValueError('Reference ground-clearance diagnosis expects capsule foot geoms')
  collision_ids = np.flatnonzero(np.isin(model.geom_bodyid, body_ids) &
                                ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
  allowed = (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE)
  if any(model.geom_type[g] not in allowed for g in collision_ids):
    raise ValueError('Expected capsule/sphere robot collision geometry')
  half_length = np.where(model.geom_type[collision_ids] == mujoco.mjtGeom.mjGEOM_CAPSULE,
                         model.geom_size[collision_ids, 1], 0.)
  result = []
  for i in range(len(motion['joint_pos'])):
    data.qpos[:3] = motion['body_pos_w'][i, root_index]
    data.qpos[3:7] = motion['body_quat_w'][i, root_index]
    data.qpos[qids] = motion['joint_pos'][i]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    # Capsule radius plus the vertical projection of its axial half-length.
    gap = [min(data.geom_xpos[g, 2] - model.geom_size[g, 0]
               - abs(data.geom_xmat[g].reshape(3, 3)[2, 2])*model.geom_size[g, 1] for g in foot)
           for foot in foot_geoms]
    all_gap = (data.geom_xpos[collision_ids, 2] - model.geom_size[collision_ids, 0]
               - abs(data.geom_xmat[collision_ids, 8])*half_length)
    result.append((data.subtree_com[root].copy(), data.xpos[body_ids].copy(), gap, all_gap.min()))
  com, bodies, gap, all_gap = map(np.asarray, zip(*result))
  return dict(com_world=com, body_pos=bodies, body_names=np.array(robot.body_names),
              foot_min_collision_height=gap, gravity=np.asarray(model.opt.gravity).copy(),
              robot_mass=np.array(model.body_subtreemass[root]), robot_min_collision_height=all_gap,
              max_contact_margin=np.array(model.geom_margin.max()*2))


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--checkpoint', required=True)
  ap.add_argument('--motion', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--speed', type=float, default=1.)
  ap.add_argument('--window', type=float, nargs=2, default=(4.5, 6.0), metavar=('START', 'END'))
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--seed', type=int, default=42)
  ap.add_argument('--no-substeps', action='store_true', help='control run for instrumentation parity')
  ap.add_argument('--reference-only', action='store_true', help='only calculate reference COM and foot collision clearance')
  args = ap.parse_args()
  if not 0 < args.speed <= 1 or not np.isfinite(args.window).all() or not 0 <= args.window[0] < args.window[1]:
    ap.error('require 0 < speed <= 1 and 0 <= window start < end')
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  cfg = prepare_eval_cfg(load_env_cfg(TASK, play=True), args.motion, 1, False, args.seed)
  dt = cfg.sim.mujoco.timestep * cfg.decimation
  with np.load(args.motion, allow_pickle=False) as src:
    if not np.allclose(src['fps'], 1/dt):
      ap.error('motion/control FPS mismatch')
    original = dict(src)
  motion = retime_motion(original, args.speed, dt)
  if args.window[1] > (len(original['joint_pos'])-1)*dt:
    ap.error('diagnostic window exceeds the reference duration')
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  # Diagnostic references stay beside diagnostic results, never replace the accepted NPZ.
  if args.speed != 1:
    motion_path = out.with_name(out.name + '_reference.npz')
    np.savez_compressed(motion_path, **motion)
    cfg.commands['motion'].motion_file = str(motion_path)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  kin = reference_kinematics(env, motion)
  np.savez_compressed(out.with_name(out.name + '_reference_kinematics.npz'),
                      t=np.arange(len(motion['joint_pos']))*dt*args.speed, **kin)
  from reference_flight_diagnostics import save_flight_diagnostics
  save_flight_diagnostics(kin, dt, args.speed, args.window, out)
  if args.reference_only:
    env.close()
    return
  agent_cfg = load_rl_cfg(TASK)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device=args.device)
  runner.load(args.checkpoint, load_cfg={'actor': True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  cmd = env.command_manager.get_term('motion')
  venv.reset()
  initialize_reference_targets(cmd)
  obs = venv.get_observations()
  diagnostics, controls = TrackingDiagnostics(), []
  recorder = SubstepRecorder(env, args.speed, args.window)
  steps = min(len(motion['joint_pos'])-1, int(np.ceil(args.window[1]/args.speed/dt)))
  terminated = None
  from contextlib import nullcontext
  try:
    with torch.inference_mode(), (nullcontext() if args.no_substeps else recorder):
      for step in range(steps):
        action = policy(obs)
        action_value = array(action[0])
        obs, _, dones, _ = venv.step(action)
        if bool(dones[0]):
          terminated = {'t': (step+1)*dt, 'reference_t': (step+1)*dt*args.speed,
              'terms': [n for n in env.termination_manager.active_terms if bool(env.termination_manager.get_term(n)[0])]}
          break
        diagnostics(env, cmd, (step+1)*dt, np.array([True]))
        controls.append(dict(policy_action=action_value, pd_target=array(cmd.robot.data.joint_pos_target[0]),
                             com_world=array(env.sim.data.subtree_com[0, cmd.robot.indexing.root_body_id])))
    trajectory = diagnostics.data()
    if controls:
      trajectory.update({k: np.array([c[k] for c in controls]) for k in controls[0]})
      trajectory['reference_t'] = trajectory['t']*args.speed
    np.savez_compressed(Path(str(out)+'.npz'), **trajectory)
    substeps = recorder.save(out.with_name(out.name + '_substeps.npz')) if recorder.rows else None
    report = dict(kind='local_control_diagnosis', quality_accepted=False, task=TASK,
        checkpoint=project_relative(args.checkpoint), motion=project_relative(args.motion),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        motion_sha256=hashlib.sha256(Path(args.motion).read_bytes()).hexdigest(),
        seed=args.seed, speed=args.speed, window_reference_s=list(args.window),
        physics_dt=env.physics_dt, control_dt=dt, substep_samples=len(recorder.rows),
        completed_diagnostic_window=terminated is None, termination=terminated,
        final_reference_t=(step+1)*dt*args.speed,
        timing='Control rows: post-step actual/reference; target/action held over the preceding control interval. Substep rows: pre-integration joint q/v and same-time solver forces/derived COM/contacts; reference_t is continuous original-motion time, command_reference_t is the held input frame.',
        limitations=['Partial rollout from time zero; not full-motion acceptance.',
          'Contact force is strongest point per foot, not summed ground reaction force.',
          'COM projection is kinematic evidence, not a dynamic stability certificate.',
          'Slower reference is out-of-distribution for the existing policy; results do not prove feasibility.'])
    if substeps is not None:
      force, limits = abs(substeps['actuator_force']), substeps['actuator_force_limits']
      report['substep_actuators'] = {str(n): dict(max_abs_nm=float(force[:, j].max()),
          limit_nm=float(limits[j]), fraction_at_95_percent=float(np.mean(force[:, j] >= .95*limits[j])))
          for j, n in enumerate(substeps['actuator_joint_names']) if limits[j] > 0}
    Path(str(out)+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
  finally:
    env.close()


if __name__ == '__main__':
  main()
