#!/usr/bin/env python3
"""flight_heading_judge_test.py -- the heading on a flight judged for the Go2 (unitree-rl-mjlab-62 2026-11-16,
go2-depth's decision: from the Go2 teacher v2 on; scripts/unified/accept_judge.py FLIGHT_HEADING_*).

descent_probe.py --straight's SUM / SUMLAT lines (tier 2 writes them per condition: ${P}_descent{,_typ,_pay}.txt) are
read per band and direction; the bar: every band and direction's hd_max_p90 at most 8 deg, unloaded and with the typical
payload, judged for the robots in FLIGHT_HEADING_JUDGED (the full payload reported only). Checked, no simulator:
  (1) the bar 8 deg; the Go2 judged, the Go2W / As2 / B2 / As2W / B2W not; U_JUDGE_FLIGHT_HEADING adds a robot;
  (2) flight_path_lines: a band over 8 deg is a fail when judged (its band, direction and value named), none when only
      reported; exactly 8 deg passes; a missing file is "未测" when judged (a tool fault: accept_judge.MISSING), nothing
      when reported; the detail lines say 判 / 只报告;
  (3) the whole judge on a fake prefix (only the descent files there): the Go2's verdict names the typical payload's
      heading over the bar and not the full payload's, the Go2W's names neither, and neither is a judge error.

Local: python scripts/unified/flight_heading_judge_test.py"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accept_judge as aj  # noqa: E402

BANDS = ("5-8cm", "8-12cm", "12-15cm", "15-18cm")


def probe_text(over: dict[tuple[str, str], float] | None = None) -> str:
  """descent_probe --straight lines for every band and direction, hd_max_p90 4.0 unless `over` names another."""
  over = over or {}
  out = []
  for b in BANDS:
    for d in ("up", "down"):
      out.append(f"SUM {b} {d} rear_calf 1.00 rear_thigh 0.00 front_calf 3.00 front_thigh 0.00 roll_std 1.3 falls 0")
      hd = over.get((b, d), 4.0)
      out.append(f"SUMLAT {b} {d} meander 0.6 meander_pp 2.3 sway 0.3 drift 0.7 hd_mean 0.6 hd_max 1.9 vy 0.05 "
                 f"wz 9.4 hd_max_p90 {hd:.3f} n 72")
  return "\n".join(out) + "\n"


def main() -> int:
  fails = 0

  def check(good: bool, text: str) -> None:
    nonlocal fails
    print(("ok   " if good else "FAIL ") + text)
    fails += 0 if good else 1

  os.environ.pop("U_JUDGE_FLIGHT_HEADING", None)
  # (1)
  check(aj.FLIGHT_HEADING_BAR_DEG == 8.0, f"the bar is 8 deg (got {aj.FLIGHT_HEADING_BAR_DEG})")
  check(aj.FLIGHT_HEADING_JUDGED == frozenset({"Go2"}), f"judged robots: Go2 only (got {sorted(aj.FLIGHT_HEADING_JUDGED)})")
  check(aj.flight_heading_judged("Go2"), "the Go2 is judged")
  for r in ("Go2W", "As2", "B2", "As2W", "B2W"):
    check(not aj.flight_heading_judged(r), f"the {r} is not judged")
  os.environ["U_JUDGE_FLIGHT_HEADING"] = "Go2W"
  check(aj.flight_heading_judged("Go2W") and not aj.flight_heading_judged("As2"),
        "U_JUDGE_FLIGHT_HEADING=Go2W adds the Go2W only")
  os.environ.pop("U_JUDGE_FLIGHT_HEADING")

  with tempfile.TemporaryDirectory() as tmp:
    # (2)
    f_over = Path(tmp, "over.txt")
    f_over.write_text(probe_text({("15-18cm", "down"): 9.4}))
    f_edge = Path(tmp, "edge.txt")
    f_edge.write_text(probe_text({("15-18cm", "up"): 8.0}))
    bad, detail = aj.flight_path_lines("Go2", str(f_over), "", True)
    check(len(bad) == 1 and "楼梯朝向最大偏离超过 8°" in bad[0] and "15-18cm 下 9.4°" in bad[0],
          f"judged: a band over 8 deg fails, named ({bad})")
    check(any("（判，上限 8°）" in x for x in detail), "judged: the detail says 判")
    bad, detail = aj.flight_path_lines("Go2W", str(f_over), "", False)
    check(bad == [], f"reported: no fail ({bad})")
    check(any("只报告" in x and "15-18cm 下 9.4" in x for x in detail), "reported: the detail lists the reading")
    bad, _ = aj.flight_path_lines("Go2", str(f_edge), "", True)
    check(bad == [], f"judged: exactly 8 deg passes ({bad})")
    bad, _ = aj.flight_path_lines("Go2", str(Path(tmp, "none.txt")), "常用负载 ", True)
    check(len(bad) == 1 and "未测" in bad[0] and any(w in bad[0] for w in aj.MISSING),
          f"judged: a missing file is 未测, a tool fault ({bad})")
    bad, _ = aj.flight_path_lines("Go2W", str(Path(tmp, "none.txt")), "", False)
    check(bad == [], "reported: a missing file is no fail")

    # (3) the whole judge: unloaded under the bar, the typical payload over it on 15-18 cm down, the full payload over
    # it everywhere (reported only)
    prefix = str(Path(tmp, "u_x"))
    Path(prefix + "_descent.txt").write_text(probe_text())
    Path(prefix + "_descent_typ.txt").write_text(probe_text({("15-18cm", "down"): 9.4}))
    Path(prefix + "_descent_pay.txt").write_text(probe_text({(b, d): 12.0 for b in BANDS for d in ("up", "down")}))
    for r, want in (("Go2", True), ("Go2W", False)):
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        aj.main([r, prefix, "--typical"])
      first = buf.getvalue().splitlines()[0] if buf.getvalue() else ""
      check("judge error" not in first, f"{r}: the judge ran ({first[:80]})")
      typ = "常用负载 楼梯朝向最大偏离超过 8°：15-18cm 下 9.4°" in first
      check(typ == want, f"{r}: the typical payload's heading {'in' if want else 'not in'} the verdict")
      check("满载 楼梯朝向最大偏离超过" not in first, f"{r}: the full payload's heading reported only")
      check("楼梯朝向最大偏离超过" not in first.replace("常用负载 楼梯朝向最大偏离超过", ""),
            f"{r}: the unloaded heading (4 deg) not in the verdict")
  print("ALL OK" if not fails else f"FAILED: {fails}")
  return 1 if fails else 0


if __name__ == "__main__":
  sys.exit(main())
