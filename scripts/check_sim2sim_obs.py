"""Cross-check sim2sim's (i.e. the C++ deploy controller's) observation vector against
the mjlab training env's actor observation, for identical simulator states.

Runs the training env (play mode, randomization off) for a few steps with small
random actions, copies its state into sim2sim.Robot, rebuilds the observation
with sim2sim.MimicController and compares term by term.

Usage:
  python scripts/check_sim2sim_obs.py --motion <npz> \
      [--policy-dir deploy/robots/g1/config/policy/mimic/dance1_subject2]
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import sim2sim  # noqa: E402

TASK = "Unitree-G1-Tracking-No-State-Estimation"
TERMS = [("motion_command", 58), ("motion_anchor_ori_b", 6), ("base_ang_vel", 3),
         ("joint_pos_rel", 29), ("joint_vel_rel", 29), ("last_action", 29)]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--motion", required=True)
  ap.add_argument("--policy-dir", default="deploy/robots/g1/config/policy/mimic/dance1_subject2")
  ap.add_argument("--steps", type=int, default=40)
  args = ap.parse_args()

  import src.tasks  # noqa: F401  (registers tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  cfg.commands["motion"].motion_file = args.motion
  cfg.terminations = {}
  cfg.events = {}  # only startup randomizations (com, encoder bias, friction) live here
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  obs, _ = env.reset()
  cmd = env.command_manager.get_term("motion")

  deploy = yaml.safe_load(open(pathlib.Path(args.policy_dir) / "params" / "deploy.yaml"))
  model, _ = sim2sim.build_model()
  robot = sim2sim.Robot(model, deploy)
  ctrl = sim2sim.MimicController(args.policy_dir, args.motion, deploy)

  # Deploy-side action mapping vs the training env's action term.
  term = env.action_manager.get_term("joint_pos")
  scale_env = term._scale.cpu().numpy().reshape(-1) if hasattr(term, "_scale") else None
  off_env = term._offset.cpu().numpy().reshape(-1) if hasattr(term, "_offset") else None
  if scale_env is not None:
    print(f"action scale  max |deploy.yaml - env| = {np.abs(ctrl.scale - scale_env).max():.2e}")
  if off_env is not None:
    print(f"action offset max |deploy.yaml - env| = {np.abs(ctrl.offset - off_env).max():.2e}")

  def load_state():
    qpos = env.sim.data.qpos[0].cpu().numpy().astype(np.float64)
    qvel = env.sim.data.qvel[0].cpu().numpy().astype(np.float64)
    robot.d.qpos[:] = qpos
    robot.d.qvel[:] = qvel
    import mujoco
    mujoco.mj_forward(robot.m, robot.d)

  # C++: init_quat is fixed when the Mimic state is entered (here: env reset state).
  load_state()
  ctrl.time_start = 0.0
  ctrl.enter(robot)
  init_quat_cpp = ctrl.init_quat.copy()

  rng = np.random.default_rng(0)
  worst = {n: 0.0 for n, _ in TERMS}
  worst_identity_ori = 0.0
  for step in range(args.steps):
    a = torch.tensor(rng.uniform(-0.3, 0.3, size=(1, 29)), dtype=torch.float32)
    obs, *_ = env.step(a)
    o_env = obs["actor"][0].cpu().numpy() if isinstance(obs, dict) else obs[0].cpu().numpy()
    load_state()
    ctrl.last_action = a[0].numpy()
    frame = int(cmd.time_steps[0].item())
    ctrl.init_quat = init_quat_cpp
    o_dep = ctrl.observation(robot, frame)
    ctrl.init_quat = np.array([1.0, 0, 0, 0])
    o_id = ctrl.observation(robot, frame)
    i = 0
    for name, n in TERMS:
      worst[name] = max(worst[name], float(np.abs(o_env[i:i + n] - o_dep[i:i + n]).max()))
      if name == "motion_anchor_ori_b":
        worst_identity_ori = max(worst_identity_ori, float(np.abs(o_env[i:i + n] - o_id[i:i + n]).max()))
      i += n
    assert i == o_env.shape[0], (i, o_env.shape)

  print(f"over {args.steps} env steps, max |env obs - deploy obs| per term:")
  for name, _ in TERMS:
    print(f"  {name:22s} {worst[name]:.2e}")
  print(f"  motion_anchor_ori_b with init_quat=identity instead of the C++ yaw "
        f"alignment: {worst_identity_ori:.2e}")
  yaw = 2 * np.degrees(np.arctan2(init_quat_cpp[3], init_quat_cpp[0]))
  print(f"C++ init_quat yaw at entry = {yaw:.2f} deg "
        f"(robot torso yaw vs reference *pelvis* yaw)")


if __name__ == "__main__":
  main()
