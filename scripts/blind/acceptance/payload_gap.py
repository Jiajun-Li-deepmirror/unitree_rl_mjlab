#!/usr/bin/env python3
"""payload_gap.py R PREFIX [PREFIX ...] -- the unloaded and the loaded readings of the same tests, side by side
(go2-depth 2026-11-11, the user: "带负载的同一组考题（12 kg 下的蠕动、踏步漂移、跑偏、交替原地转）只测量、写进报告，不参与通过与否 ...
每个候选模型并排列出无负载和带负载的读数，能看出重载和轻载差多少，以及这个差距随训练是否在缩小").

PREFIX is a check's file prefix (jobs/fe_check.sh: <PREFIX>_drift3.txt ... and the loaded runs <PREFIX>_drift3_pay.txt,
_stand_pay, _spinalt_pay, _march_pay at the robot's payload limit, jobs/accept_payload.sh). Per checkpoint:
  drift     the worst sideways / straight offset over the distance walked, % (as g_drift judges the unloaded run)
  creep     the worst stop creep, cm p90 (eval_standing)
  spin-alt  the worst end distance after the alternating turns, m p50 (spin_alt)
  march     the march-in-place drift, cm (march_eval)
each as unloaded / loaded / gap (loaded - unloaded). Several prefixes (checkpoints in training order) show whether the
gap shrinks. The numbers come from the judge's own parsers (accept_tables.py), so they are the acceptance's measures.
Informational: nothing here passes or fails a check.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accept_tables as at  # noqa: E402


def drift(p) -> float | None:
  d = at.drift_vals(p)
  worst = None
  for (vx, vy, wz), v in d.items():
    if wz != 0.0:
      continue
    sideways = abs(vy) > abs(vx)
    travel = abs(v["across"][1]) if sideways else abs(v["along"][1])
    off = v["along"] if sideways else v["across"]
    if travel > 0:
      r = 100.0 * max(abs(off[0]), abs(off[2])) / travel
      worst = r if worst is None else max(worst, r)
  return worst


def creep(p) -> float | None:
  v = [x["p90"] for x in at.stand_vals(p).values() if not math.isnan(x["p90"])]
  return max(v) if v else None


def spin(p) -> float | None:
  v = [x["end"][0] for x in (at.spinalt_vals(p) or {}).values() if not math.isnan(x["end"][0])]
  return max(v) if v else None


def march(p) -> float | None:
  m = at.march_vals(p).get("march")
  return m["drift"] if m else None


ITEMS = (("drift", "跑偏 %", drift), ("creep", "停车蠕动 cm", creep), ("spin", "交替原地转 m", spin),
         ("march", "原地踏步漂移 cm", march))
KINDS = ("drift3", "stand", "spinalt", "march")


def files(prefix: str, loaded: bool) -> dict:
  return {k: f"{prefix}_{k}{'_pay' if loaded else ''}.txt" for k in KINDS}


def main(argv: list[str]) -> int:
  if len(argv) < 2:
    print(__doc__)
    return 2
  r, prefixes = argv[0], argv[1:]
  fmt = lambda x: "  -  " if x is None else f"{x:5.2f}"
  print(f"{r}: unloaded / loaded (at the payload limit) / gap = loaded - unloaded")
  print(f"{'checkpoint':38s} " + " | ".join(f"{name:>22s}" for _, name, _ in ITEMS))
  for pre in prefixes:
    cells = []
    for _, _, fn in ITEMS:
      a, b = fn(files(pre, False)), fn(files(pre, True))
      gap = None if a is None or b is None else b - a
      cells.append(f"{fmt(a)} / {fmt(b)} / {'  -  ' if gap is None else f'{gap:+5.2f}'}")
    print(f"{Path(pre).name[-38:]:38s} " + " | ".join(f"{c:>22s}" for c in cells))
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
