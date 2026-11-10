"""dr_edge_check.py TASK N (2026-11-02): build TASK's training env with N robots and print, per randomized term,
the share of robots in the outer 10% band of either end of its range (uniform draws: ~20%)."""
import sys

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

sys.path.insert(0, "jobs")
from fall_diag import _params  # noqa: E402

task, n = sys.argv[1], int(sys.argv[2])
cfg = load_env_cfg(task)
cfg.scene.num_envs = n
for k in ("base_payload", "base_com", "foot_friction"):
  print(k, cfg.events[k].params.get("distribution", "uniform"), cfg.events[k].params["ranges"])
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
par = _params(env)
rng = {"payload_kg": cfg.events["base_payload"].params["ranges"], "friction": cfg.events["foot_friction"].params["ranges"]}
for i, a in enumerate("xyz"):
  r = cfg.events["base_com"].params["ranges"][i]
  rng[f"com_{a}_cm"] = (r[0] * 100, r[1] * 100)
for k, (lo, hi) in rng.items():
  v = par[k].float()
  if k == "friction":
    v, lo, hi = v.log(), torch.tensor(lo).log().item(), torch.tensor(hi).log().item()
  w = hi - lo
  band = ((v < lo + 0.1 * w) | (v > hi - 0.1 * w)).float().mean().item()
  print(f"{k:>12}: in the outer 10% bands {100 * band:5.1f}%  (min {v.min().item():.3f} max {v.max().item():.3f})")
env.close()
