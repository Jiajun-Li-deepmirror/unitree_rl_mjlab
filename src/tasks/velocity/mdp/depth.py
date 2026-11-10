"""Front depth camera of the students: tinynav-sim's PIE camera or navcore's.

A DepthCameraSpec says what the policy sees: a render (ray distances, as
mujoco_warp reports them, 0 on a miss) and the frame cut from it. The PIE
spec mirrors tinynav-sim ``mujoco/sim/contract.py`` (DEPTH_* and
``preprocess_depth_z``; keep them in sync): 106 x 60 cut to 86 x 60, ray
distance, a 3x3 Gaussian. The navcore spec is the real camera measured from
its bag (2026-10-26): optical Z like the sensor's, no blur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DepthCameraSpec:
  """A depth camera as the policy sees it: a render_width x render_height
  render at fovy_deg (principal point at its centre), and the width x height
  frame cut from it at (crop_left, crop_top), normalized to
  [DEPTH_MIN_M, DEPTH_MAX_M] / DEPTH_MAX_M."""

  name: str
  render_width: int
  render_height: int
  fovy_deg: float
  width: int
  height: int
  crop_left: int
  crop_top: int
  optical_z: bool = False
  """Frames hold optical Z (a stereo camera's output) instead of ray distance."""
  blur: bool = False
  """tinynav PIE's 3x3 Gaussian over the frame."""

  @property
  def focal_px(self) -> float:
    return 0.5 * self.render_height / math.tan(math.radians(self.fovy_deg) / 2)


PIE_DEPTH_CAMERA = DepthCameraSpec(
  "pie", DEPTH_RAW_WIDTH, DEPTH_HEIGHT, DEPTH_FOVY_DEG, DEPTH_WIDTH, DEPTH_HEIGHT,
  DEPTH_CROP_LEFT, 0, blur=True,
)
# navcore's depth (its bag of 2026-10-26): the rectified left camera, 544 x 640,
# fx = fy = 310.15 px, principal point (270.06, 322.23), 82.5 x 91.8 deg, optical
# Z in mm at 10 Hz. An eighth of it, 68 x 80 at 38.77 px, cut from a 72 x 84 render
# so the frame can shift 2 px each way (principal point randomization); the
# deploy takes every 8th pixel (block centres) of the sensor's frame.
NAVCORE_FOCAL_PX = 310.15 / 8
NAVCORE_DEPTH_CAMERA = DepthCameraSpec(
  "navcore", 72, 84, 2 * math.degrees(math.atan(42 / NAVCORE_FOCAL_PX)), 68, 80, 2, 2,
  optical_z=True,
)
DEPTH_CAMERAS = {c.name: c for c in (PIE_DEPTH_CAMERA, NAVCORE_DEPTH_CAMERA)}


def front_depth_camera_spec_cfg(
  pos: tuple[float, float, float] = DEPTH_CAMERA_POS,
  quat: tuple[float, float, float, float] = DEPTH_CAMERA_QUAT,
  body: str = "base_link",
  camera: DepthCameraSpec = PIE_DEPTH_CAMERA,
) -> CameraCfg:
  """The camera itself, added to the robot entity so DR can address it.

  Defaults are the Go2 mount of the tinynav PIE contract."""
  return CameraCfg(
    name=FRONT_DEPTH_CAMERA,
    body=body,
    pos=pos,
    quat=quat,
    fovy=camera.fovy_deg,
  )


def front_depth_camera_cfg(
  name: str = FRONT_DEPTH_CAMERA, camera: DepthCameraSpec = PIE_DEPTH_CAMERA
) -> CameraSensorCfg:
  """Depth sensor wrapping the robot's front camera."""
  return CameraSensorCfg(
    name=name,
    camera_name=f"robot/{FRONT_DEPTH_CAMERA}",
    fovy=camera.fovy_deg,
    width=camera.render_width,
    height=camera.render_height,
    data_types=("depth",),
    use_textures=False,
    # tinynav renders groups 0-2: terrain plus the robot's own visual meshes.
    enabled_geom_groups=(0, 1, 2),
  )


_Z_FROM_RAY: dict = {}


def _odd_pad(d: torch.Tensor, w: torch.Tensor, r: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
  """Pad values `d` and weights `w` (1, C, H, W) by `r` along `dim` (2 rows, 3 columns):
  values by odd reflection about the edge pixel (2 edge - mirror: affine values stay
  affine), weights by edge times mirror (a miss on either side is no padding)."""
  size = d.shape[dim]
  lo = torch.arange(r, 0, -1, device=d.device)
  hi = torch.arange(size - 2, size - 2 - r, -1, device=d.device)
  parts_d, parts_w = [], []
  for idx, edge in ((lo, 0), (hi, size - 1)):
    e_d, e_w = d.narrow(dim, edge, 1), w.narrow(dim, edge, 1)
    parts_d.append(2.0 * e_d - d.index_select(dim, idx))
    parts_w.append(e_w * w.index_select(dim, idx))
  return (torch.cat((parts_d[0], d, parts_d[1]), dim=dim), torch.cat((parts_w[0], w, parts_w[1]), dim=dim))


def _z_from_ray(camera: DepthCameraSpec, like: torch.Tensor) -> torch.Tensor:
  """(render H, render W) optical Z per unit ray distance at each pixel centre."""
  key = (camera.name, like.device, like.dtype)
  if key not in _Z_FROM_RAY:
    f = camera.focal_px
    u = (torch.arange(camera.render_width, device=like.device, dtype=like.dtype) + 0.5 - camera.render_width / 2) / f
    v = (torch.arange(camera.render_height, device=like.device, dtype=like.dtype) + 0.5 - camera.render_height / 2) / f
    _Z_FROM_RAY[key] = torch.rsqrt(1.0 + u.square().view(1, -1) + v.square().view(-1, 1))
  return _Z_FROM_RAY[key]


def preprocess_ray_depth(
  ray_depth: torch.Tensor,
  crop_left: torch.Tensor | None = None,
  camera: DepthCameraSpec = PIE_DEPTH_CAMERA,
  crop_top: torch.Tensor | None = None,
) -> torch.Tensor:
  """(N, render H, render W) ray distances (0 = miss) -> (N, H, W) normalized frame.

  `crop_left` / `crop_top` (N,) move each env's frame (the spec's by default).
  Optical-Z cameras read a miss, or anything past DEPTH_MAX_M in Z, as max range."""
  hit = ray_depth > 0.0
  depth = ray_depth * _z_from_ray(camera, ray_depth) if camera.optical_z else ray_depth
  depth = torch.where(hit, depth, torch.full_like(depth, DEPTH_MAX_M))
  if crop_left is None and crop_top is None:
    depth = depth[:, camera.crop_top : camera.crop_top + camera.height,
                  camera.crop_left : camera.crop_left + camera.width]
  else:
    n, dev = depth.shape[0], depth.device
    top = crop_top if crop_top is not None else torch.full((n,), camera.crop_top, device=dev, dtype=torch.long)
    left = crop_left if crop_left is not None else torch.full((n,), camera.crop_left, device=dev, dtype=torch.long)
    rows = top.view(-1, 1, 1) + torch.arange(camera.height, device=dev).view(1, -1, 1)
    depth = torch.gather(depth, 1, rows.expand(-1, -1, depth.shape[2]))
    cols = left.view(-1, 1, 1) + torch.arange(camera.width, device=dev).view(1, 1, -1)
    depth = torch.gather(depth, 2, cols.expand(-1, depth.shape[1], -1))
  if camera.blur:
    depth = F.pad(depth.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
    kernel = _GAUSSIAN_KERNEL.to(depth.device, depth.dtype).view(1, 1, 3, 3)
    depth = F.conv2d(depth, kernel).squeeze(1)
  return depth.clamp(DEPTH_MIN_M, DEPTH_MAX_M) / DEPTH_MAX_M


class depth_history:
  """Two 10 Hz depth frames of the term's camera (``camera``, a DEPTH_CAMERAS
  name; PIE by default: (N, 2, 60, 86)), oldest first.

  A new frame is captured whenever ``episode_length_buf % 5 == 0``, i.e. on
  the first observation after reset and every 5th control step, like the
  tinynav runtime (``control_step % 5 == 0``). The first frame after a reset
  fills the whole history.

  Domain randomization (training only): each capture uses the render from one
  control step earlier with probability ``stale_prob`` (latency), adds
  Gaussian noise of ``noise_std`` meters, and drops pixels to max range with
  probability ``dropout_prob``.

  A real stereo camera (``latency_range`` set; go2-depth 2026-10-27, for the
  navcore deployment, sized from its bag):

  - latency: each episode draws a latency of ``latency_range`` control steps
    and each capture waits up to ``latency_jitter`` steps more: it uses the
    render that old, never one from before the reset. Replaces ``stale_prob``.
  - frame timing: a capture repeats the newest frame with probability
    ``drop_prob`` (no new camera frame), and starts a freeze of
    ``freeze_captures`` captures with probability ``freeze_prob``.
  - stereo error, on the distance z (m): a per-episode scale drawn from
    ``scale_range`` (calibration); per frame a common scale of
    ``frame_scale_std`` and a smooth field of ``field_std`` (bilinear over a
    4 x 4 grid: navcore's frame-to-frame error moves whole surfaces, its pixels
    48 px apart correlate at 0.8); Gaussian pixel noise of ``z_noise * z^2``;
    readings under a per-episode minimum range drawn from ``min_range_m``
    are invalid.
  - wrong patches: with probability ``patch_prob`` a frame gets 1-2 patches of
    ``patch_size`` (height, width) render pixels whose depth is off by a factor
    drawn from ``patch_scale`` (a learned stereo fills textureless or shiny
    surfaces with wrong values rather than leaving holes); with probability
    ``hole_prob`` 1-4 holes of ``hole_size``; pixels at depth edges over
    ``edge_jump_m`` turn into flying pixels between the near and the far side
    with probability ``edge_prob``.
  - the frame moves up to ``crop_jitter`` pixels each way per episode where the
    render has the margin (principal point).
  - edge smoothing (go2-depth 2026-10-30: the real navcore camera's learned stereo
    turns a 0.15-0.19 m stair riser into a 35-50 deg ramp ~14 cm wide, and no other
    term does that): with probability ``edge_blur_prob`` an episode blurs each
    frame's disparity (1 / optical Z) with a Gaussian of ``edge_blur_sigma`` render
    pixels across rows and that times ``edge_blur_aspect`` across columns,
    normalised over the valid pixels, before the other errors. A plane's disparity
    is affine in the image, so planes stay as they are: only edges and corners
    smear. Fitted on real stair frames (B2, 2026-10-30): rows ~1.75 px, columns
    about half of it.
  - occlusions (go2-depth 2026-10-29: in tinynav-sim a frame held at 0.20 m made
    the students flail, Go2 rolling 22-25 deg): with probability
    ``occlusion_prob`` per capture an event of ``occlusion_captures`` captures
    starts, of a kind drawn with weights ``occlusion_kinds`` (patch, near,
    miss): a patch of ``occlusion_patch`` (height, width as fractions of the
    frame) at a distance from ``occlusion_depth_m``, the whole frame at that
    distance, or the whole frame invalid (a covered lens). The teacher's view
    is untouched, so the student learns to walk on what it saw before and on
    its proprioception.

  Invalid readings become misses (max range) as in deployment.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self._sensor: CameraSensor = env.scene[cfg.params["sensor_name"]]
    p = cfg.params
    self.camera = cam = DEPTH_CAMERAS[p.get("camera", PIE_DEPTH_CAMERA.name)]
    n = env.num_envs
    self._device = env.device
    self._frames = torch.zeros(n, DEPTH_HISTORY_LENGTH, cam.height, cam.width, device=env.device)
    self._prev_raw = torch.zeros(n, cam.render_height, cam.render_width, device=env.device)
    self._needs_fill = torch.ones(n, dtype=torch.bool, device=env.device)
    self._real = p.get("latency_range") is not None
    if self._real:
      lo, hi = p["latency_range"]
      self._latency_range = (float(lo), float(hi))
      self._ring_len = int(math.ceil(hi + p.get("latency_jitter", 0.0))) + 1
      self._ring = torch.zeros(self._ring_len, n, cam.render_height, cam.render_width, device=env.device)
      self._ring_pos = 0
      self._steps_since_reset = torch.zeros(n, dtype=torch.long, device=env.device)
      self._latency = torch.zeros(n, device=env.device)
      self._scale = torch.ones(n, device=env.device)
      self._min_range = torch.zeros(n, device=env.device)
      self._crop_left = torch.full((n,), cam.crop_left, dtype=torch.long, device=env.device)
      self._crop_top = torch.full((n,), cam.crop_top, dtype=torch.long, device=env.device)
      self._freeze = torch.zeros(n, dtype=torch.long, device=env.device)
      self._occ_left = torch.zeros(n, dtype=torch.long, device=env.device)
      self._occ_kind = torch.zeros(n, dtype=torch.long, device=env.device)
      self._occ_depth = torch.zeros(n, device=env.device)
      self._occ_rect = torch.zeros(n, 4, dtype=torch.long, device=env.device)
      self._blur_rows = torch.zeros(n, device=env.device)
      self._blur_cols = torch.zeros(n, device=env.device)
      self._params = p
      self.reset()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._needs_fill[env_ids] = True
    if self._real:
      cam, k = self.camera, self._needs_fill[env_ids].numel()
      p, u = self._params, lambda lo, hi: lo + (hi - lo) * torch.rand(k, device=self._device)
      self._steps_since_reset[env_ids] = 0
      self._latency[env_ids] = u(*self._latency_range)
      self._scale[env_ids] = u(*p.get("scale_range", (1.0, 1.0)))
      self._min_range[env_ids] = u(*p.get("min_range_m", (0.0, 0.0)))
      jitter = int(p.get("crop_jitter", 0))
      for offset, home, margin in ((self._crop_left, cam.crop_left, cam.render_width - cam.width),
                                   (self._crop_top, cam.crop_top, cam.render_height - cam.height)):
        j = min(jitter, home, margin - home)
        offset[env_ids] = home + torch.randint(-j, j + 1, (k,), device=self._device)
      self._freeze[env_ids] = 0
      self._occ_left[env_ids] = 0
      blur_prob = float(p.get("edge_blur_prob", 0.0))
      if blur_prob > 0.0:
        on = (torch.rand(k, device=self._device) < blur_prob).float()
        rows = u(*p.get("edge_blur_sigma", (1.0, 2.5)))
        self._blur_rows[env_ids] = on * rows
        self._blur_cols[env_ids] = on * rows * u(*p.get("edge_blur_aspect", (0.3, 0.7)))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    stale_prob: float = 0.0,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    **real: object,
  ) -> torch.Tensor:
    del sensor_name, real  # Resolved in __init__.
    depth = self._sensor.data.depth
    assert depth is not None
    raw = depth[..., 0]
    if self._real:
      self._ring[self._ring_pos] = raw
      self._ring_pos = (self._ring_pos + 1) % self._ring_len

    # Called once per env step (ObservationManager.compute after sim.sense).
    capture = self._needs_fill | (env.episode_length_buf % DEPTH_UPDATE_PERIOD_STEPS == 0)
    if torch.any(capture):
      frame_raw = raw
      crop_left = crop_top = None
      repeat = None
      if self._real:
        frame_raw, repeat = self._camera_frame(env.num_envs, capture)
        crop_left, crop_top = self._crop_left, self._crop_top
      elif stale_prob > 0.0:
        stale = torch.rand(env.num_envs, 1, 1, device=env.device) < stale_prob
        stale &= ~self._needs_fill.view(-1, 1, 1)
        frame_raw = torch.where(stale, self._prev_raw, raw)
      frame = preprocess_ray_depth(frame_raw, crop_left, self.camera, crop_top)
      if noise_std > 0.0:
        frame = frame + torch.randn_like(frame) * (noise_std / DEPTH_MAX_M)
      if dropout_prob > 0.0:
        frame = torch.where(torch.rand_like(frame) < dropout_prob, 1.0, frame)
      frame = frame.clamp(DEPTH_MIN_M / DEPTH_MAX_M, 1.0)
      if repeat is not None:
        frame = torch.where(repeat.view(-1, 1, 1), self._frames[:, -1], frame)

      shifted = torch.cat([self._frames[:, 1:], frame.unsqueeze(1)], dim=1)
      filled = frame.unsqueeze(1).expand_as(self._frames)
      new = torch.where(self._needs_fill.view(-1, 1, 1, 1), filled, shifted)
      self._frames = torch.where(capture.view(-1, 1, 1, 1), new, self._frames)
      self._needs_fill[:] = False
    if self._real:
      self._steps_since_reset += 1
    elif stale_prob > 0.0:
      self._prev_raw = raw.clone()
    return self._frames

  def _rects(self, n: int, prob: float, count: int, size, shape) -> list[torch.Tensor]:
    """Up to `count` random rectangles per env, (N, H, W) masks: the first with
    probability `prob`, each further one half as likely."""
    dev, (height, width) = self._device, shape
    (h_lo, h_hi), (w_lo, w_hi) = size
    rows = torch.arange(height, device=dev).view(1, -1, 1)
    cols = torch.arange(width, device=dev).view(1, 1, -1)
    masks = []
    for i in range(count):
      on = torch.rand(n, device=dev) < prob * 0.5**i
      y = torch.randint(0, height, (n,), device=dev).view(-1, 1, 1)
      x = torch.randint(0, width, (n,), device=dev).view(-1, 1, 1)
      h = torch.randint(h_lo, h_hi + 1, (n,), device=dev).view(-1, 1, 1)
      w = torch.randint(w_lo, w_hi + 1, (n,), device=dev).view(-1, 1, 1)
      masks.append(on.view(-1, 1, 1) & (rows >= y) & (rows < y + h) & (cols >= x) & (cols < x + w))
    return masks

  def _occlude(self, n: int, capture: torch.Tensor, z: torch.Tensor,
               valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """This capture's occlusions (see the class doc): events start on new
    captures (never the reset's), count down in each env's captures."""
    p, dev = self._params, self._device
    height, width = z.shape[1:]
    start = (capture & ~self._needs_fill & (self._occ_left == 0)
             & (torch.rand(n, device=dev) < float(p["occlusion_prob"])))
    if torch.any(start):
      u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev)
      lo, hi = p.get("occlusion_captures", (2, 20))
      kinds = torch.tensor(p.get("occlusion_kinds", (0.5, 0.25, 0.25)), device=dev)
      (h_lo, h_hi), (w_lo, w_hi) = p.get("occlusion_patch", ((0.3, 1.0), (0.3, 1.0)))
      h = (u(h_lo, h_hi) * height).long().clamp(1, height)
      w = (u(w_lo, w_hi) * width).long().clamp(1, width)
      y = (torch.rand(n, device=dev) * (height - h + 1).float()).long()
      x = (torch.rand(n, device=dev) * (width - w + 1).float()).long()
      self._occ_left = torch.where(start, torch.randint(lo, hi + 1, (n,), device=dev), self._occ_left)
      self._occ_kind = torch.where(start, torch.multinomial(kinds, n, replacement=True) + 1, self._occ_kind)
      self._occ_depth = torch.where(start, u(*p.get("occlusion_depth_m", (0.15, 0.4))), self._occ_depth)
      self._occ_rect = torch.where(start.view(-1, 1), torch.stack((y, x, h, w), dim=1), self._occ_rect)
    on = capture & (self._occ_left > 0)
    if torch.any(on):
      rows = torch.arange(height, device=dev).view(1, -1, 1)
      cols = torch.arange(width, device=dev).view(1, 1, -1)
      y, x, h, w = (self._occ_rect[:, i].view(-1, 1, 1) for i in range(4))
      in_rect = (rows >= y) & (rows < y + h) & (cols >= x) & (cols < x + w)
      kind, on = self._occ_kind.view(-1, 1, 1), on.view(-1, 1, 1)
      near = on & ((kind == 2) | ((kind == 1) & in_rect))
      z = torch.where(near, self._occ_depth.view(-1, 1, 1).expand_as(z), z)
      # nearer than the episode's minimum range reads as nothing, as on the robot
      valid = torch.where(near, z >= self._min_range.view(-1, 1, 1), valid) & ~(on & (kind == 3))
    self._occ_left = torch.where(capture & (self._occ_left > 0), self._occ_left - 1, self._occ_left)
    return z, valid

  def _edge_blur(self, z: torch.Tensor, envs: torch.Tensor | None = None) -> torch.Tensor:
    """(N, H, W) ray distances (0 = miss) with each env's disparity blurred (see the
    class doc): separable Gaussians of `_blur_rows` / `_blur_cols` pixels (0: none),
    one depthwise convolution per pass, normalised over the hits. The frame is padded
    by odd reflection (linear extrapolation), so a plane stays exact up to the
    border. `envs` (N,) bool: only these envs (this step's captures); the rest pass."""
    on = self._blur_rows > 0.0
    if envs is not None:
      on = on & envs
    ids = torch.nonzero(on).flatten()
    if ids.numel() == 0:
      return z
    out = z.clone()
    out[ids] = self._edge_blur_rows(z[ids], self._blur_rows[ids], self._blur_cols[ids])
    return out

  def _edge_blur_rows(self, z: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    hit = z > 0.0
    zfac = _z_from_ray(self.camera, z) if self.camera.optical_z else torch.ones_like(z[0])
    disp = torch.where(hit, 1.0 / torch.clamp(z * zfac, min=1e-3), torch.zeros_like(z)).unsqueeze(0)
    weight = hit.to(z.dtype).unsqueeze(0)
    sigma_max = float(self._params.get("edge_blur_sigma", (1.0, 2.5))[1])
    radius = max(1, int(math.ceil(3.0 * sigma_max)))
    taps = torch.arange(-radius, radius + 1, device=z.device, dtype=z.dtype)
    n = z.shape[0]
    for sigma, dim in ((rows, 2), (cols, 3)):
      kernel = torch.exp(-0.5 * (taps.view(1, -1) / torch.clamp(sigma, min=1e-3).view(-1, 1)).square())
      kernel = kernel / kernel.sum(dim=1, keepdim=True)
      shape = (n, 1, -1, 1) if dim == 2 else (n, 1, 1, -1)
      d, w = _odd_pad(disp, weight, radius, dim)
      num = F.conv2d(d * w, kernel.view(shape), groups=n)
      weight = F.conv2d(w, kernel.view(shape), groups=n)
      disp = num / torch.clamp(weight, min=1e-6)
    blurred = 1.0 / torch.clamp(disp[0], min=1e-4) / zfac
    return torch.where(hit, blurred, z)

  def _camera_frame(self, n: int, capture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The real camera's frame for this step's captures (meters, 0 = invalid)
    and which envs repeat their newest frame instead (drops and freezes,
    counted in each env's own captures)."""
    p, dev = self._params, self._device
    age = self._latency + float(p.get("latency_jitter", 0.0)) * torch.rand(n, device=dev)
    age = torch.minimum(age.round().long(), self._steps_since_reset).clamp(0, self._ring_len - 1)
    newest = (self._ring_pos - 1) % self._ring_len
    z = self._ring[(newest - age) % self._ring_len, torch.arange(n, device=dev)]
    shape = z.shape[1:]
    if float(p.get("edge_blur_prob", 0.0)) > 0.0:
      z = self._edge_blur(z, capture)

    valid = z > 0.0
    scale = self._scale.view(-1, 1, 1) * (1.0 + float(p.get("frame_scale_std", 0.0)) * torch.randn(n, 1, 1, device=dev))
    field_std = float(p.get("field_std", 0.0))
    if field_std > 0.0:
      grid = field_std * torch.randn(n, 1, 4, 4, device=dev)
      scale = scale * (1.0 + F.interpolate(grid, size=shape, mode="bilinear", align_corners=True).squeeze(1))
    patch_prob = float(p.get("patch_prob", 0.0))
    if patch_prob > 0.0:
      lo, hi = p.get("patch_scale", (0.7, 1.3))
      for mask in self._rects(n, patch_prob, 2, p.get("patch_size", ((10, 40), (10, 40))), shape):
        factor = lo + (hi - lo) * torch.rand(n, 1, 1, device=dev)
        scale = torch.where(mask, scale * factor, scale)
    z = z * scale
    z = z + torch.randn_like(z) * float(p.get("z_noise", 0.0)) * z.square()
    valid &= z >= self._min_range.view(-1, 1, 1)
    edge_prob = float(p.get("edge_prob", 0.0))
    if edge_prob > 0.0:
      # misses read as max range at an edge, as their neighbors see them
      full = torch.where(valid, z, torch.full_like(z, DEPTH_MAX_M)).unsqueeze(1)
      near = -F.max_pool2d(-full, 3, stride=1, padding=1).squeeze(1)
      far = F.max_pool2d(full, 3, stride=1, padding=1).squeeze(1)
      flying = (far - near > float(p.get("edge_jump_m", 0.25))) & (torch.rand_like(z) < edge_prob)
      z = torch.where(flying, near + (far - near) * torch.rand_like(z), z)
      valid = (valid | flying) & (z < DEPTH_MAX_M)
    hole_prob = float(p.get("hole_prob", 0.0))
    if hole_prob > 0.0:
      for mask in self._rects(n, hole_prob, 4, p.get("hole_size", ((3, 12), (3, 16))), shape):
        valid &= ~mask
    if float(p.get("occlusion_prob", 0.0)) > 0.0:
      z, valid = self._occlude(n, capture, z, valid)
    z = torch.where(valid, z, torch.zeros_like(z))

    # Frame timing: a dropped frame repeats the newest one; a freeze repeats it
    # for its captures (the first frame after a reset is always new).
    timed = capture & ~self._needs_fill
    lo, hi = p.get("freeze_captures", (3, 10))
    start = timed & (self._freeze == 0) & (torch.rand(n, device=dev) < float(p.get("freeze_prob", 0.0)))
    self._freeze = torch.where(start, torch.randint(lo, hi + 1, (n,), device=dev), self._freeze)
    repeat = timed & ((self._freeze > 0) | (torch.rand(n, device=dev) < float(p.get("drop_prob", 0.0))))
    self._freeze = torch.where(capture & (self._freeze > 0), self._freeze - 1, self._freeze)
    return z, repeat
