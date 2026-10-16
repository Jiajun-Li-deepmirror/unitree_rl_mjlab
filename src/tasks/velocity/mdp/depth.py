"""Front depth camera matching the tinynav-sim PIE camera contract.

The constants and preprocessing mirror tinynav-sim
``mujoco/sim/contract.py`` (DEPTH_* and ``preprocess_depth_z``); keep them in
sync. mujoco_warp reports Euclidean distance along each pixel ray (0 on a
miss), which is what the contract normalizes, so no optical-Z conversion is
needed here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.utils.spec_config import CameraCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

DEPTH_RAW_WIDTH = 106
DEPTH_HEIGHT = 60
DEPTH_CROP_LEFT = 10
DEPTH_CROP_RIGHT = 10
DEPTH_WIDTH = DEPTH_RAW_WIDTH - DEPTH_CROP_LEFT - DEPTH_CROP_RIGHT
DEPTH_HISTORY_LENGTH = 2
DEPTH_UPDATE_PERIOD_STEPS = 5  # 10 Hz at the 50 Hz control rate.
DEPTH_HORIZONTAL_FOV_DEG = 87.0
DEPTH_FOVY_DEG = 2.0 * math.degrees(
  math.atan(
    math.tan(math.radians(DEPTH_HORIZONTAL_FOV_DEG) * 0.5) * DEPTH_HEIGHT / DEPTH_RAW_WIDTH
  )
)
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 3.0
DEPTH_CAMERA_POS = (0.345, 0.0, 0.07)
DEPTH_CAMERA_QUAT = (0.5792280, 0.4055798, -0.4055798, -0.5792280)

_GAUSSIAN_KERNEL = torch.tensor(
  (
    (0.07511361, 0.12384140, 0.07511361),
    (0.12384140, 0.20417996, 0.12384140),
    (0.07511361, 0.12384140, 0.07511361),
  )
)
_GAUSSIAN_KERNEL = _GAUSSIAN_KERNEL / _GAUSSIAN_KERNEL.sum()


FRONT_DEPTH_CAMERA = "front_depth"
"""Camera name inside the robot entity (``robot/front_depth`` in the scene)."""


def front_depth_camera_spec_cfg() -> CameraCfg:
  """The camera itself, added to the robot entity so DR can address it."""
  return CameraCfg(
    name=FRONT_DEPTH_CAMERA,
    body="base_link",
    pos=DEPTH_CAMERA_POS,
    quat=DEPTH_CAMERA_QUAT,
    fovy=DEPTH_FOVY_DEG,
  )


def front_depth_camera_cfg(name: str = FRONT_DEPTH_CAMERA) -> CameraSensorCfg:
  """Depth sensor wrapping the robot's front camera."""
  return CameraSensorCfg(
    name=name,
    camera_name=f"robot/{FRONT_DEPTH_CAMERA}",
    fovy=DEPTH_FOVY_DEG,
    width=DEPTH_RAW_WIDTH,
    height=DEPTH_HEIGHT,
    data_types=("depth",),
    use_textures=False,
    # tinynav renders groups 0-2: terrain plus the robot's own visual meshes.
    enabled_geom_groups=(0, 1, 2),
  )


def preprocess_ray_depth(ray_depth: torch.Tensor) -> torch.Tensor:
  """(N, H, W_raw) ray distances (0 = miss) -> (N, H, W) normalized frame."""
  depth = torch.where(ray_depth > 0.0, ray_depth, torch.full_like(ray_depth, DEPTH_MAX_M))
  depth = depth[:, :, DEPTH_CROP_LEFT : DEPTH_RAW_WIDTH - DEPTH_CROP_RIGHT]
  depth = F.pad(depth.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
  kernel = _GAUSSIAN_KERNEL.to(depth.device, depth.dtype).view(1, 1, 3, 3)
  depth = F.conv2d(depth, kernel).squeeze(1)
  return depth.clamp(DEPTH_MIN_M, DEPTH_MAX_M) / DEPTH_MAX_M


class depth_history:
  """Two 10 Hz depth frames, oldest first: (N, 2, 60, 86).

  A new frame is captured whenever ``episode_length_buf % 5 == 0``, i.e. on
  the first observation after reset and every 5th control step, like the
  tinynav runtime (``control_step % 5 == 0``). The first frame after a reset
  fills the whole history.

  Domain randomization (training only): each capture uses the render from one
  control step earlier with probability ``stale_prob`` (latency), adds
  Gaussian noise of ``noise_std`` meters, and drops pixels to max range with
  probability ``dropout_prob``.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self._sensor: CameraSensor = env.scene[cfg.params["sensor_name"]]
    n = env.num_envs
    self._frames = torch.zeros(
      n, DEPTH_HISTORY_LENGTH, DEPTH_HEIGHT, DEPTH_WIDTH, device=env.device
    )
    self._prev_raw = torch.zeros(n, DEPTH_HEIGHT, DEPTH_RAW_WIDTH, device=env.device)
    self._needs_fill = torch.ones(n, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._needs_fill[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    stale_prob: float = 0.0,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
  ) -> torch.Tensor:
    del sensor_name  # Resolved in __init__.
    depth = self._sensor.data.depth
    assert depth is not None
    raw = depth[..., 0]

    # Called once per env step (ObservationManager.compute after sim.sense).
    capture = self._needs_fill | (env.episode_length_buf % DEPTH_UPDATE_PERIOD_STEPS == 0)
    if torch.any(capture):
      frame_raw = raw
      if stale_prob > 0.0:
        stale = torch.rand(env.num_envs, 1, 1, device=env.device) < stale_prob
        stale &= ~self._needs_fill.view(-1, 1, 1)
        frame_raw = torch.where(stale, self._prev_raw, raw)
      frame = preprocess_ray_depth(frame_raw)
      if noise_std > 0.0:
        frame = frame + torch.randn_like(frame) * (noise_std / DEPTH_MAX_M)
      if dropout_prob > 0.0:
        frame = torch.where(torch.rand_like(frame) < dropout_prob, 1.0, frame)
      frame = frame.clamp(DEPTH_MIN_M / DEPTH_MAX_M, 1.0)

      shifted = torch.cat([self._frames[:, 1:], frame.unsqueeze(1)], dim=1)
      filled = frame.unsqueeze(1).expand_as(self._frames)
      new = torch.where(self._needs_fill.view(-1, 1, 1, 1), filled, shifted)
      self._frames = torch.where(capture.view(-1, 1, 1, 1), new, self._frames)
      self._needs_fill[:] = False
    if stale_prob > 0.0:
      self._prev_raw = raw.clone()
    return self._frames
