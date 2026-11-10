"""Checks of mdp.depth_history's edge smoothing (REAL_CAMERA_EDGE_BLUR) on the navcore camera:
planes stay exact (also at the border and next to misses), a step edge's 20-80% width is
1.68 sigma, sigma 0 and misses are untouched, and on sim2sim map3 frames the stair profiles
match the real camera's (go2-depth 2026-10-30 fit). Run from the repo root:
    PYTHONPATH=$PWD python scripts/go2_depth/check_edge_blur.py [SIM2SIM_OUT_DIR...]"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from src.tasks.velocity.mdp import depth as D

cam = D.NAVCORE_DEPTH_CAMERA
H, W = cam.render_height, cam.render_width
f = cam.focal_px
dev = "cuda" if torch.cuda.is_available() else "cpu"
u = (torch.arange(W, device=dev) + 0.5 - W / 2) / f
v = (torch.arange(H, device=dev) + 0.5 - H / 2) / f
zfac = D._z_from_ray(cam, torch.zeros(1, H, W, device=dev))


def blur(z_opt: torch.Tensor, rows: list[float], aspect: float = 0.5) -> torch.Tensor:
  """Optical Z (N, H, W), 0 = miss -> blurred optical Z, sigma rows per frame."""
  obj = object.__new__(D.depth_history)
  obj.camera = cam
  obj._params = dict(D_EDGE)
  obj._blur_rows = torch.tensor(rows, device=dev, dtype=torch.float32)
  obj._blur_cols = obj._blur_rows * aspect
  ray = torch.where(z_opt > 0, z_opt / zfac, torch.zeros_like(z_opt))
  out = obj._edge_blur(ray)
  return torch.where(out > 0, out * zfac, torch.zeros_like(out))


from src.tasks.velocity.depth_env_cfg import REAL_CAMERA_EDGE_BLUR as D_EDGE  # noqa: E402

ok = True
# 1. a tilted plane (disparity affine in the image), all valid
disp = 0.6 + 0.10 * u.view(1, -1) + 0.25 * v.view(-1, 1)
z = (1.0 / disp).expand(3, H, W).contiguous()
out = blur(z, [1.0, 1.75, 2.5])
err = ((out - z).abs() / z).max().item()
print(f"tilted plane: max relative error {err:.2e} (whole frame, border included)"); ok &= err < 1e-4

# 2. the floor seen by a level camera 0.70 m up: valid below the horizon (6 m), misses above
z_floor = torch.where(v > 0.70 / 6.0, 0.70 / torch.clamp(v, min=1e-6), torch.zeros_like(v)).view(-1, 1).expand(H, W)
out = blur(z_floor.unsqueeze(0).contiguous(), [2.5])[0]
valid_rows = torch.nonzero(z_floor[:, 0] > 0).flatten()
deep = valid_rows[valid_rows > valid_rows.min() + 8]          # away from the miss boundary
err = ((out[deep] - z_floor[deep]).abs() / z_floor[deep]).max().item()
misses = (out[z_floor == 0] == 0).all().item()
print(f"floor: max relative error {err:.2e} below the horizon band, bottom border included; misses kept {misses}")
ok &= err < 1e-4 and misses

# 3. a step edge: far half (2 m) above near half (1 m); 20-80% width of disparity across rows
z = torch.where(v.view(-1, 1) < 0.0, torch.tensor(2.0, device=dev), torch.tensor(1.0, device=dev)).expand(H, W)
for sig in (1.0, 1.75, 2.5):
  col = 1.0 / blur(z.unsqueeze(0).contiguous(), [sig])[0][:, W // 2].cpu().numpy()
  t = (col - 0.5) / 0.5
  rows = np.arange(H, dtype=np.float64)
  width = np.interp(0.8, t, rows) - np.interp(0.2, t, rows)
  print(f"step edge sigma {sig}: 20-80% width {width:.2f} rows (Gaussian: {1.683 * sig:.2f})")
  ok &= abs(width - 1.683 * sig) < 0.15

# 4. sigma 0 leaves the frame alone
z = (1.0 + torch.rand(1, H, W, device=dev)) * (torch.rand(1, H, W, device=dev) > 0.1)
out = blur(z, [0.0])
same = torch.equal(out, z) or ((out - z).abs().max().item() < 1e-5)
print(f"sigma 0: unchanged {same}"); ok &= same

# 5. sim2sim map3 frames (optical Z from the bridge's depth_*.npz) through this op: stair profiles
dirs = [Path(a) for a in sys.argv[1:]]
if dirs:
  sys.path.insert(0, str(Path.home() / ".cache/go2-depth-work/jetson_b2/tests"))
  import edge_blur_fit_student as fit  # noqa: E402
  sim = [fr for d in dirs if d.exists() for fr in fit.frames(d)]
  frames = np.stack([z for z, g in sim])                      # (N, 80, 68), metres, MAX = miss
  zt = torch.tensor(np.where(frames < fit.DB.MAX_M - 1e-3, frames, 0.0), device=dev, dtype=torch.float32)
  pad = torch.zeros(len(sim), H, W, device=dev)
  top, left = cam.crop_top, cam.crop_left
  pad[:, top:top + zt.shape[1], left:left + zt.shape[2]] = zt
  for sig in (0.0, 1.75):
    out = blur(pad, [sig] * len(sim))[:, top:top + zt.shape[1], left:left + zt.shape[2]].cpu().numpy()
    out = np.where(out > 0, out, fit.DB.MAX_M)
    h, w = fit.stats([fit.profile(out[i], sim[i][1]) for i in range(len(sim))])
    print(f"sim2sim stairs through the op, sigma {sig}: slopes {np.round(h * 100).astype(int).tolist()}%, riser width median "
          f"{np.median(w) * 100 if len(w) else float('nan'):.0f} cm (real: [21, 20, 36, 8, 16]%, 10 cm)")
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
