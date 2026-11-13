#!/usr/bin/env python3
"""s1_plateau.py LOG MIN WINDOW DROP HOLD EVERY (unitree-rl-mjlab-62 2026-11-13; go2-depth's decision: the students'
stage 1 ends when its behavior loss levels off): reads a stage-1 (distillation) training log -- rsl_rl's blocks
"Learning iteration N/M" ... "Mean behavior loss: X" -- and judges every EVERY iterations K >= MIN: the mean loss of
the last WINDOW iterations (K-WINDOW+1 .. K) against the WINDOW before them; it "levels off" at K when it fell less
than DROP (relative) there. The first K where that held at HOLD consecutive checks is the end.
Prints "STOP K (drops: ...)" or "WAIT <the last iteration read>". Exit 0 either way, 2 on bad arguments.
"""

from __future__ import annotations

import re
import sys

HEAD = re.compile(r"Learning iteration (\d+)/\d+")
LOSS = re.compile(r"Mean behavior loss:\s*([-\d.eE+]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def losses(path: str) -> dict[int, float]:
  out, it = {}, None
  try:
    text = open(path, errors="replace").read()
  except OSError:
    return out
  for line in ANSI.sub("", text).splitlines():
    m = HEAD.search(line)
    if m:
      it = int(m.group(1))
      continue
    m = LOSS.search(line)
    if m and it is not None:
      out[it] = float(m.group(1))
  return out


def judge(loss: dict[int, float], k_min: int, w: int, drop: float, hold: int, every: int) -> tuple[int | None, list]:
  if not loss:
    return None, []
  last = max(loss)
  run, drops = 0, []
  k = max(k_min, 2 * w)
  k += (-k) % every   # the checks at K = multiples of EVERY: the runner saves model_K there
  while k <= last:
    a = [loss[i] for i in range(k - w + 1, k + 1) if i in loss]
    b = [loss[i] for i in range(k - 2 * w + 1, k - w + 1) if i in loss]
    if len(a) >= 0.9 * w and len(b) >= 0.9 * w:
      ma, mb = sum(a) / len(a), sum(b) / len(b)
      rel = (mb - ma) / mb if mb > 0 else 0.0
      drops.append((k, rel))
      run = run + 1 if rel < drop else 0
      if run >= hold:
        return k, drops[-hold:]
    k += every
  return None, drops[-hold:]


def main(argv: list[str]) -> int:
  if len(argv) != 6:
    print(__doc__)
    return 2
  path = argv[0]
  k_min, w, drop, hold, every = int(argv[1]), int(argv[2]), float(argv[3]), int(argv[4]), int(argv[5])
  loss = losses(path)
  k, drops = judge(loss, k_min, w, drop, hold, every)
  txt = ", ".join(f"{i}: {d:.1%}" for i, d in drops)
  print(f"STOP {k} (drops: {txt})" if k is not None else f"WAIT {max(loss) if loss else -1} (drops: {txt})")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
