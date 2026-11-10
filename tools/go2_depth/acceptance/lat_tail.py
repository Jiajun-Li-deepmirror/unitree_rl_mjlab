"""lat_tail.py ROBOT ONNX [RUNS]: accept_map3.py's runs; the distribution of the stance legs' sideways angle (hip -> foot
from the vertical, deg) on stairs, up and down: mean, p90, p99, max, and the share over 5 / 10 deg."""
import os, sys
import numpy as np
C = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(C, "accept_map3.py")).read()
exec(src[:src.index("def posture(")])
out = {"up": {"F": [], "R": []}, "down": {"F": [], "R": []}}
for sc in SCEN:
  model, _b = model_for(sc[1])
  grid = Grid(model, sc[5])
  for s in range(runs):
    m = run(sc, s, grid)
    for side, sag, lat in m["ang"]:
      out["up" if sc[6] else "down"][side].append(lat)
for k, d in out.items():
  for side, v in d.items():
    v = np.array(v)
    if len(v):
      print(f"{robot} {onnx.stem} {k:4s} {side}: mean {v.mean():4.1f} p90 {np.percentile(v, 90):4.1f} p99 {np.percentile(v, 99):4.1f} max {v.max():4.1f}  >5deg {100 * np.mean(v > 5):4.1f}%  >10deg {100 * np.mean(v > 10):4.1f}%")
