#!/usr/bin/env python3
"""last_riser_test.py -- U_LAST_RISER's terrain part (go2-depth 2026-11-17, go2-home's L6): mdp.walled_stairs.WalledCrossStairsTerrainCfg
last_riser_scale. No GPU.
  (1) _last_riser: the last riser in [lo, hi] x the flight's mean rise, the flight's sum kept, the other risers all
      shifted by the same amount, the tile's generator not advanced;
  (2) the whole tile (up and down, every difficulty): the same boxes and origin as without it, the flights' heights kept.
Local: PYTHONPATH=$PWD python scripts/unified/last_riser_test.py"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.getcwd())
from src.tasks.velocity.mdp.walled_stairs import WalledCrossStairsTerrainCfg  # noqa: E402

ok = True


def check(good, text):
  global ok
  ok &= bool(good)
  print(("  ok    " if good else "  FAIL  ") + text)


print("(1) _last_riser:")
cfg = WalledCrossStairsTerrainCfg(proportion=1.0, inverted=True, last_riser_scale=(0.4, 1.0))
lasts, good = [], True
for seed in range(200):
  rng = np.random.default_rng(seed)
  rises = rng.uniform(0.05, 0.23, rng.integers(3, 12))
  before = rng.bit_generator.state
  out = cfg._last_riser(rises, rng, seed % 4)
  mean = rises.sum() / len(rises)
  shift = out[:-1] - rises[:-1]
  good &= (rng.bit_generator.state == before and abs(out.sum() - rises.sum()) < 1e-9
           and 0.4 * mean - 1e-9 <= out[-1] <= 1.0 * mean + 1e-9 and np.allclose(shift, shift[0]))
  lasts.append(out[-1] / mean)
check(good, f"200 flights: sums kept, last risers {min(lasts):.2f}-{max(lasts):.2f} x the mean (bar 0.4-1.0), "
      "the rest shifted alike, the tile's generator not advanced")
off = WalledCrossStairsTerrainCfg(proportion=1.0, inverted=True)
r0 = np.array([0.1, 0.12, 0.15])
check(np.array_equal(off._last_riser(r0, np.random.default_rng(0)), r0), "without the scale: the rises as drawn")
print("(2) the whole tile:")
for inverted in (True, False):
  for d in (0.0, 0.5, 1.0):
    outs = []
    for c in (WalledCrossStairsTerrainCfg(proportion=1.0, inverted=inverted),
              WalledCrossStairsTerrainCfg(proportion=1.0, inverted=inverted, last_riser_scale=(0.4, 1.0))):
      spec = mujoco.MjSpec()
      spec.worldbody.add_body(name="terrain")
      outs.append(c.function(d, spec, np.random.default_rng(7)))
    a, b = outs
    tops = lambda o: np.array([g.geom.pos[2] + g.geom.size[2] for g in o.geometries])
    ta, tb = tops(a), tops(b)
    changed = int((np.abs(ta - tb) > 1e-6).sum()) if len(ta) == len(tb) else -1
    check(len(a.geometries) == len(b.geometries) and np.allclose(a.origin, b.origin) and changed > 0
          and abs(ta.max() - tb.max()) < 1e-6,
          f"{'up' if inverted else 'down'} difficulty {d}: {len(b.geometries)} boxes as without, origin {b.origin.tolist()}, "
          f"the highest surface kept, {changed} step boxes moved")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
