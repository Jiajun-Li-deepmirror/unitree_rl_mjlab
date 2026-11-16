#!/usr/bin/env python3
"""loaded_top_judge_test.py -- the full payload's stairs falls judged on 5-15 cm for every robot (unitree-rl-mjlab-62
2026-11-16; the user ~13:40, relayed by go2-depth 14:50: "满载在 15–18 cm 档不判，只判 5–15 cm，15–18 cm 只报告", every
robot; scripts/unified/accept_judge.py LOADED_TOP_REPORTED, read by the students' acceptance and the teacher's stairs
gate). No simulator. Checked:
  (1) every robot (Go2, Go2W, As2, As2W, B2, B2W) is in the set;
  (2) loaded_falls on an eval_stairs table (72 robots per band and direction): falls only at 15-18 cm -> judged 0,
      reported 3; one more at 12-15 cm -> judged 1; a table without the per-band robot count (an older output) -> every
      band judged, nothing reported, as before;
  (3) the unloaded and the typical payload's counts are untouched: stairs_falls still counts every band.

Local: python scripts/unified/loaded_top_judge_test.py"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accept_judge as aj  # noqa: E402

ROBOTS = ("Go2", "Go2W", "As2", "As2W", "B2", "B2W")
BANDS = ("5-8cm", "8-12cm", "12-15cm", "15-18cm")


def table(falls: dict[tuple[str, str], int], n: int = 72, per_band_line: bool = True) -> str:
  """An eval_stairs-like output: the per-band table (traverse falls % in the first block, stop falls 0 %), the exact
  "falls:" line and (per_band_line) the robots per (band, direction)."""
  lines = []
  if per_band_line:
    lines.append(f"stairs tiles: {n} robots per (band, direction)")
  for b in BANDS:
    for d in ("up", "down"):
      f = falls.get((b, d), 0)
      lines.append(f"  {b:>7s} {d:>4s} |   100%   {round(100 * f / n)}%      0.0%    6.50 |   100%    0%      0.4   1.00")
  lines.append(f"falls: traverse {sum(falls.values())} of {8 * n}, stop 0 of {8 * n}")
  return "\n".join(lines) + "\n"


def main() -> int:
  fails = 0

  def check(good: bool, text: str) -> None:
    nonlocal fails
    print(("ok   " if good else "FAIL ") + text)
    fails += 0 if good else 1

  os.environ.pop("U_JUDGE_LOADED_TOP_REPORTED", None)
  for r in ROBOTS:
    check(aj.loaded_top_reported(r), f"{r}: the full payload judged on 5-15 cm, 15-18 cm reported")
  with tempfile.TemporaryDirectory() as tmp:
    top = Path(tmp, "top.txt")
    top.write_text(table({("15-18cm", "up"): 3}))
    both = Path(tmp, "both.txt")
    both.write_text(table({("15-18cm", "up"): 3, ("12-15cm", "down"): 1}))
    old = Path(tmp, "old.txt")
    old.write_text(table({("15-18cm", "up"): 3}, per_band_line=False))
    for r in ROBOTS:
      check(aj.loaded_falls(r, str(top)) == (0, 3), f"{r}: 3 falls at 15-18 cm only -> judged 0, reported 3 "
                                                     f"({aj.loaded_falls(r, str(top))})")
      check(aj.loaded_falls(r, str(both)) == (1, 3), f"{r}: + 1 at 12-15 cm -> judged 1, reported 3")
      check(aj.loaded_falls(r, str(old)) == (3, None), f"{r}: no per-band robot count -> every band judged, as before")
    check(aj.stairs_falls(str(top)) == 3, "stairs_falls (the unloaded / typical count) still counts every band")
  print("ALL OK" if not fails else f"FAILED: {fails}")
  return 1 if fails else 0


if __name__ == "__main__":
  sys.exit(main())
