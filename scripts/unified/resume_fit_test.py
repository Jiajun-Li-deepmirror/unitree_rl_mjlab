#!/usr/bin/env python3
"""resume_fit_test.py [CHECKPOINT] -- curriculum.progress._fit_robots (go2-depth 2026-11-17, go2-home's fix 3: a 2048-robot
fine-tune of a 4096-robot run kept no per-robot state). No GPU.
  (1) synthetic 4096 -> 2048 / 8192 / 4096: evaluation robots from evaluation robots, training from training, every
      array with the same draw (a robot's level, column, rows and class together), the per-class row means kept,
      the same count unchanged;
  (2) with CHECKPOINT (a unified teacher's model_K.pt): its saved progress state resampled to 2048 robots, the training
      robots' up / down row means per payload class before and after (B1 s0's model_3000: light ~6.1 / mid ~5.7 /
      heavy ~5.2 up).
Local: PYTHONPATH=$PWD python scripts/unified/resume_fit_test.py [CHECKPOINT]"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from src.tasks.unified.constraints import eval_count  # noqa: E402
from src.tasks.unified.curriculum import PAY_CLASSES, progress  # noqa: E402

ok = True


def check(good, text):
  global ok
  ok &= bool(good)
  print(("  ok    " if good else "  FAIL  ") + text)


def class_means(st, key="up"):
  m = len(st["terrain_levels"])
  e = eval_count(m)
  out = {}
  for ci, c in enumerate(PAY_CLASSES):
    v = [st["rows"][key][i] for i in range(e, m) if st["pay_class"][i] == ci]
    out[c] = sum(v) / len(v) if v else float("nan")
  return out


print("(1) synthetic:")
g = torch.Generator().manual_seed(0)
m = 4096
cls = torch.randint(3, (m,), generator=g)
base = torch.tensor([6.1, 5.7, 5.2])[cls] + torch.randn(m, generator=g) * 0.8
lv = base.round().clamp(0, 9).long()
st = {"p": 0.5, "level": float(lv.float().mean()), "level0": 0.0, "terrain_levels": lv.tolist(),
      "terrain_types": (torch.arange(m) % 20).tolist(), "pay_class": cls.tolist(),
      "rows": {"up": lv.tolist(), "down": (lv + 1).tolist(), "other": (lv - 1).tolist()}}
# tag each robot so the pairing can be checked: level*1000 + column*10 + class
tag = [st["terrain_levels"][i] * 1000 + st["terrain_types"][i] * 10 + st["pay_class"][i] for i in range(m)]
e_old = eval_count(m)
for n in (2048, 8192):
  out = progress._fit_robots(st, n)
  e_new = eval_count(n)
  tags = {t: i for i, t in enumerate(tag)}
  paired = all(out["rows"]["up"][j] == out["terrain_levels"][j] and out["rows"]["down"][j] == out["terrain_levels"][j] + 1
               for j in range(n))
  ev_ok = True
  for j in range(n):
    # find a source robot with this tag on the right side of the split
    src = [i for i in ([tags[t] for t in [out["terrain_levels"][j] * 1000 + out["terrain_types"][j] * 10 + out["pay_class"][j]]])]
    if not src:
      ev_ok = False
  cm0, cm1 = class_means(st), class_means(out)
  close = all(abs(cm0[c] - cm1[c]) < 0.1 for c in PAY_CLASSES)
  check(len(out["terrain_levels"]) == n and len(out["pay_class"]) == n and all(len(v) == n for v in out["rows"].values())
        and paired and ev_ok and close,
        f"4096 -> {n}: every array {n} long, rows paired with the levels, classes' up means "
        + " / ".join(f"{c} {cm0[c]:.2f}->{cm1[c]:.2f}" for c in PAY_CLASSES))
# evaluation robots from evaluation robots: mark the old evaluation robots' columns
st2 = dict(st, terrain_types=[99 if i < e_old else 1 for i in range(m)])
out = progress._fit_robots(st2, 2048)
e_new = eval_count(2048)
check(all(out["terrain_types"][j] == 99 for j in range(e_new)) and all(out["terrain_types"][j] == 1 for j in range(e_new, 2048)),
      f"evaluation robots {e_old} -> {e_new} drawn from the evaluation robots, training from training")
check(progress._fit_robots(st, m) is st, "the same count: the state as saved")
if len(sys.argv) > 1:
  print("(2) the checkpoint:")
  ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
  ps = ck["infos"]["unified"]["progress"]
  check(len(ps.get("terrain_levels") or []) > 0 and ps.get("rows") and ps.get("pay_class") is not None,
        f"{len(ps['terrain_levels'])} robots' levels, rows and classes saved")
  out = progress._fit_robots(ps, 2048)
  for key in ("up", "down"):
    a, b = class_means(ps, key), class_means(out, key)
    check(all(abs(a[c] - b[c]) < 0.15 for c in PAY_CLASSES),
          f"{key} rows by class, saved -> 2048: " + " / ".join(f"{c} {a[c]:.2f}->{b[c]:.2f}" for c in PAY_CLASSES))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
