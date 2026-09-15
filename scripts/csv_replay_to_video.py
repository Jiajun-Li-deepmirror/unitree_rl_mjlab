"""Replay a retargeted CSV motion and save it directly to an mp4 video
(does not run any RL policy -- just visualizes the reference motion itself,
the same way csv_to_npz.py does, but actually writes the accumulated frames
out with mediapy instead of silently discarding them)."""
from typing import Any

import numpy as np
import torch
import tyro
import os
import mediapy
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from src.tasks.tracking.config.g1_23dof.env_cfgs import unitree_g1_23dof_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


class MotionLoader:
  def __init__(self, motion_file, input_fps, output_fps, device, line_range=None):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()

  def _load_motion(self):
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(np.loadtxt(
        self.motion_file, delimiter=",",
        skiprows=self.line_range[0] - 1,
        max_rows=self.line_range[1] - self.line_range[0] + 1,
      ))
    motion = motion.to(torch.float32).to(self.device)
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
    self.motion_dof_poss_input = motion[:, 7:]
    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(self.motion_base_poss_input[index_0], self.motion_base_poss_input[index_1], blend.unsqueeze(1))
    self.motion_base_rots = self._slerp(self.motion_base_rots_input[index_0], self.motion_base_rots_input[index_1], blend)
    self.motion_dof_poss = self._lerp(self.motion_dof_poss_input[index_0], self.motion_dof_poss_input[index_1], blend.unsqueeze(1))
    print(f"Motion interpolated, input frames: {self.input_frames}, output frames: {self.output_frames}")

  def _lerp(self, a, b, blend):
    return a * (1 - blend) + b * blend

  def _slerp(self, a, b, blend):
    out = torch.zeros_like(a)
    for i in range(a.shape[0]):
      out[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return out

  def _compute_frame_blend(self, times):
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def get_next_pose(self):
    pos = self.motion_base_poss[self.current_idx : self.current_idx + 1]
    rot = self.motion_base_rots[self.current_idx : self.current_idx + 1]
    dof = self.motion_dof_poss[self.current_idx : self.current_idx + 1]
    self.current_idx += 1
    done = self.current_idx >= self.output_frames
    return pos, rot, dof, done


def main(
  robot: str,
  input_file: str,
  output_video: str,
  input_fps: float = 30.0,
  output_fps: float = 30.0,
  device: str = "cuda:0",
  line_range: tuple[int, int] | None = None,
):
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  if robot == "g1":
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    joint_names = [
      "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
      "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
      "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
      "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
      "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ]
  else:
    raise ValueError(f"Unsupported robot: {robot}")

  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  viewer_cfg = ViewerConfig(
    height=480, width=640,
    origin_type=ViewerConfig.OriginType.ASSET_ROOT, entity_name="robot",
    distance=2.5, elevation=-10, azimuth=20,
  )
  renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
  renderer.initialize()

  motion = MotionLoader(input_file, input_fps, output_fps, sim.device, line_range)
  robot_entity: Entity = scene["robot"]
  robot_joint_indexes = robot_entity.find_joints(joint_names, preserve_order=True)[0]

  scene.reset()
  frames = []
  pbar = tqdm(total=motion.output_frames, desc="Rendering", unit="frame")
  done = False
  while not done:
    pos, rot, dof, done = motion.get_next_pose()
    root_states = robot_entity.data.default_root_state.clone()
    root_states[:, 0:3] = pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = rot
    robot_entity.write_root_state_to_sim(root_states)
    joint_pos = robot_entity.data.default_joint_pos.clone()
    joint_pos[:, robot_joint_indexes] = dof
    robot_entity.write_joint_state_to_sim(joint_pos, robot_entity.data.default_joint_vel.clone())
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    renderer.update(sim.data)
    frames.append(renderer.render())
    pbar.update(1)
  pbar.close()

  print(f"\nSaving {len(frames)} frames to {output_video} ...")
  os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
  mediapy.write_video(output_video, frames, fps=output_fps)
  print("Done.")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
