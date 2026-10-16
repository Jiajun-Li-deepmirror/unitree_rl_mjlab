"""Compare the training depth observation with the tinynav-sim depth pipeline.

Builds the Unitree-Go2-Depth-Student play env (no depth noise), steps it with
zero actions, then renders the same MuJoCo model and state with the native
``mujoco.Renderer`` (optical Z) and tinynav-sim's ``preprocess_depth_z``. The
two preprocessed frames should agree to well under 1 cm.

  PYTHONPATH=. python scripts/go2_depth/depth_parity.py \
    --tinynav-sim ~/tinynav-sim --num-envs 32
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch


def load_contract(tinynav_sim: Path):
  path = tinynav_sim / "mujoco" / "sim" / "contract.py"
  spec = importlib.util.spec_from_file_location("tinynav_contract", path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--tinynav-sim", type=Path, default=Path.home() / "tinynav-sim")
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=10)
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  from src.tasks.velocity.config.go2.depth_env_cfgs import DEPTH_SENSOR_NAME
  from src.tasks.velocity.mdp import depth as depth_mod

  contract = load_contract(args.tinynav_sim.expanduser())
  for name in (
    "DEPTH_RAW_WIDTH",
    "DEPTH_HEIGHT",
    "DEPTH_CROP_LEFT",
    "DEPTH_CROP_RIGHT",
    "DEPTH_MIN_M",
    "DEPTH_MAX_M",
    "DEPTH_FOVY_DEG",
    "DEPTH_CAMERA_POS",
    "DEPTH_CAMERA_QUAT",
    "DEPTH_UPDATE_PERIOD_STEPS",
    "DEPTH_HISTORY_LENGTH",
  ):
    ours, theirs = getattr(depth_mod, name), getattr(contract, name)
    assert np.allclose(ours, theirs), f"{name}: {ours} != {theirs}"
  print("[OK] depth constants match tinynav contract")

  cfg = load_env_cfg("Unitree-Go2-Depth-Student", play=True)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  for _ in range(args.steps):
    env.step(actions)
  env.sim.sense()

  camera = env.scene[DEPTH_SENSOR_NAME]
  ours = depth_mod.preprocess_ray_depth(camera.data.depth[..., 0]).cpu().numpy()

  model = env.sim.mj_model
  data = mujoco.MjData(model)
  cam_name = camera.cfg.camera_name
  cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
  assert cam_id >= 0, f"camera {cam_name} not in model"
  assert np.isclose(model.cam_fovy[cam_id], contract.DEPTH_FOVY_DEG)
  # The terrain grid makes model.stat.extent large, so the default near/far
  # planes are far apart and the native z-buffer loses precision. Pin them to
  # 1 cm / 10 m (tinynav's map3 scene has near < 5 cm and far > 3 m as well).
  print(f"default near/far: {model.vis.map.znear * model.stat.extent:.3f} / "
        f"{model.vis.map.zfar * model.stat.extent:.1f} m")
  model.vis.map.znear = 0.01 / model.stat.extent
  model.vis.map.zfar = 10.0 / model.stat.extent
  renderer = mujoco.Renderer(model, height=contract.DEPTH_HEIGHT, width=contract.DEPTH_RAW_WIDTH)
  renderer.enable_depth_rendering()
  option = mujoco.MjvOption()
  option.geomgroup[:] = 0
  option.geomgroup[:3] = 1
  far_clip = float(model.vis.map.zfar * model.stat.extent)

  qpos = env.sim.data.qpos.cpu().numpy()
  qvel = env.sim.data.qvel.cpu().numpy()
  errors = []
  for e in range(env.num_envs):
    data.qpos[:] = qpos[e]
    data.qvel[:] = qvel[e]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam_id, scene_option=option)
    depth_z = np.asarray(renderer.render(), dtype=np.float32)
    depth_z[~np.isfinite(depth_z) | (depth_z >= far_clip * 0.99)] = 0.0
    theirs = contract.preprocess_depth_z(depth_z)[0]
    err = np.abs(ours[e] - theirs) * contract.DEPTH_MAX_M
    errors.append((float(err.mean()), float(np.percentile(err, 99)), float((ours[e] - theirs).mean())))
    if os.environ.get("PARITY_DEBUG") and err.mean() > 0.01:
      signed = (ours[e] - theirs) * contract.DEPTH_MAX_M
      print(f"  env {e} row mean signed err (cm):", np.round(signed.mean(1) * 100, 1).tolist())
      print(f"  env {e} ours row mean (m):", np.round(ours[e].mean(1) * 3, 2).tolist())
  renderer.close()

  errors = np.asarray(errors)
  for e, (mean, p99, bias) in enumerate(errors):
    print(f"env {e:3d}: mean |err| {mean * 100:.2f} cm, p99 {p99 * 100:.2f} cm, bias {bias * 300:+.2f} cm")
  print(
    f"[SUMMARY] mean |err| {errors[:, 0].mean() * 100:.2f} cm, "
    f"worst env mean {errors[:, 0].max() * 100:.2f} cm, "
    f"mean bias {errors[:, 2].mean() * 300:+.2f} cm"
  )


if __name__ == "__main__":
  main()
