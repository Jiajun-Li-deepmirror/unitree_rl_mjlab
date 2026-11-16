#!/usr/bin/env python3
"""meander_judge_test.py -- the meander on a flight judged (go2-depth 2026-11-16, the user via go2-home: the stairs
meander becomes a bar; scripts/unified/accept_judge.py MEANDER_BAR_CM / meander_bar / flight_path_lines).

descent_probe.py --straight's SUMLAT lines carry meander_pp (cm, per band and direction); the bar: every band and
direction at most the robot's bar, unloaded and with the typical payload; the full payload reported only. Checked, no
simulator:
  (1) the bars: Go2 8 cm, Go2W 5 cm; As2 / As2W / B2 / B2W report it (no bar); U_JUDGE_MEANDER adds or overrides;
  (2) flight_path_lines: a band over the bar fails (band, direction, value named); exactly the bar passes; no bar ->
      no fail, the detail says 只报告; with a bar the detail says 判 and the bar; a file without meander_pp on a band ->
      未测 (a tool fault: accept_judge.MISSING); a missing file with a bar -> 未测; the heading item is unchanged;
  (3) the whole judge on a fake prefix (only the descent files there): the Go2W's verdict names the typical payload's
      meander over 5 cm and not the full payload's; the Go2's (bar 8) names neither for the same readings; the As2's
      (no bar) names none.

Local: python scripts/unified/meander_judge_test.py"""

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


def probe_text(mpp: dict[tuple[str, str], float | None] | None = None, base: float = 2.3) -> str:
  """descent_probe --straight lines for every band and direction: meander_pp `base` unless `mpp` names another;
  None leaves meander_pp out of that band's line; hd_max_p90 4.0 (under the heading bar) everywhere."""
  mpp = mpp or {}
  out = []
  for b in BANDS:
    for d in ("up", "down"):
      out.append(f"SUM {b} {d} rear_calf 1.00 rear_thigh 0.00 front_calf 3.00 front_thigh 0.00 roll_std 1.3 falls 0")
      v = mpp.get((b, d), base)
      m = "" if v is None else f"meander_pp {v:.3f} "
      out.append(f"SUMLAT {b} {d} meander 0.6 {m}sway 0.3 drift 0.7 hd_mean 0.6 hd_max 1.9 vy 0.05 "
                 f"wz 9.4 hd_max_p90 4.000 n 72")
  return "\n".join(out) + "\n"


def main() -> int:
  fails = 0

  def check(good: bool, text: str) -> None:
    nonlocal fails
    print(("ok   " if good else "FAIL ") + text)
    fails += 0 if good else 1

  os.environ.pop("U_JUDGE_MEANDER", None)
  os.environ.pop("U_JUDGE_FLIGHT_HEADING", None)
  # (1)
  check(aj.MEANDER_BAR_CM == {"Go2": 8.0, "Go2W": 5.0}, f"bars Go2 8 / Go2W 5 cm (got {aj.MEANDER_BAR_CM})")
  check(aj.meander_bar("Go2") == 8.0 and aj.meander_bar("Go2W") == 5.0, "meander_bar reads them")
  for r in ("As2", "As2W", "B2", "B2W"):
    check(aj.meander_bar(r) is None, f"the {r} reports it (no bar)")
  os.environ["U_JUDGE_MEANDER"] = "As2:6,Go2W:4"
  check(aj.meander_bar("As2") == 6.0 and aj.meander_bar("Go2W") == 4.0 and aj.meander_bar("Go2") == 8.0
        and aj.meander_bar("B2") is None, "U_JUDGE_MEANDER=As2:6,Go2W:4 adds the As2, overrides the Go2W only")
  os.environ.pop("U_JUDGE_MEANDER")

  with tempfile.TemporaryDirectory() as tmp:
    # (2)
    f_over = Path(tmp, "over.txt")
    f_over.write_text(probe_text({("15-18cm", "up"): 6.2}))
    f_edge = Path(tmp, "edge.txt")
    f_edge.write_text(probe_text({("15-18cm", "up"): 5.0}))
    f_gap = Path(tmp, "gap.txt")
    f_gap.write_text(probe_text({("12-15cm", "down"): None}))
    bad, detail = aj.flight_path_lines("Go2W", str(f_over), "", False, 5.0)
    check(len(bad) == 1 and "楼梯蛇行峰峰值超过 5 cm" in bad[0] and "15-18cm 上 6.2 cm" in bad[0],
          f"bar 5: a band at 6.2 fails, named ({bad})")
    check(any("（判，上限 5 cm）" in x and "15-18cm 上 6.2" in x for x in detail), "bar 5: the detail says 判 and the bar")
    bad, _ = aj.flight_path_lines("Go2W", str(f_edge), "", False, 5.0)
    check(bad == [], f"bar 5: exactly 5.0 passes ({bad})")
    bad, detail = aj.flight_path_lines("As2", str(f_over), "", False, None)
    check(bad == [] and any("（只报告，cm）" in x and "15-18cm 上 6.2" in x for x in detail),
          f"no bar: no fail, the detail reports it ({bad})")
    bad, _ = aj.flight_path_lines("Go2W", str(f_gap), "", False, 5.0)
    check(len(bad) == 1 and "楼梯蛇行：未测" in bad[0] and "12-15cm 下" in bad[0]
          and any(w in bad[0] for w in aj.MISSING), f"bar 5: a band without meander_pp is 未测 ({bad})")
    bad, _ = aj.flight_path_lines("Go2W", str(Path(tmp, "none.txt")), "常用负载 ", False, 5.0)
    check(bad == ["常用负载 楼梯蛇行：未测"], f"bar 5: a missing file is 未测 ({bad})")
    bad, _ = aj.flight_path_lines("Go2", str(Path(tmp, "none.txt")), "", True, 8.0)
    check(bad == ["楼梯朝向最大偏离：未测", "楼梯蛇行：未测"], f"both judged, missing file: both 未测 ({bad})")
    bad, _ = aj.flight_path_lines("Go2", str(f_over), "", True, 8.0)
    check(bad == [], f"Go2 (heading 8 deg, meander 8 cm): 6.2 cm and 4 deg pass ({bad})")

    # (3) runs the judge before the spec (U_JUDGE_OLD=1): Go2 / Go2W are judged by accept_spec_test.py since
    # 2026-11-16 (doc/acceptance_go2_go2w.md); this keeps the old path the other robots still take
    os.environ["U_JUDGE_OLD"] = "1"
    # (3) the whole judge: unloaded under every bar, the typical payload at 6.2 cm on 15-18 cm up, the full payload at
    # 12 cm everywhere (reported only)
    prefix = str(Path(tmp, "u_x"))
    Path(prefix + "_descent.txt").write_text(probe_text())
    Path(prefix + "_descent_typ.txt").write_text(probe_text({("15-18cm", "up"): 6.2}))
    Path(prefix + "_descent_pay.txt").write_text(probe_text({(b, d): 12.0 for b in BANDS for d in ("up", "down")}))
    for r, want in (("Go2W", True), ("Go2", False), ("As2", False)):
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        aj.main([r, prefix, "--typical"])
      first = buf.getvalue().splitlines()[0] if buf.getvalue() else ""
      check("judge error" not in first, f"{r}: the judge ran ({first[:80]})")
      typ = "常用负载 楼梯蛇行峰峰值超过" in first
      check(typ == want, f"{r}: the typical payload's meander {'in' if want else 'not in'} the verdict")
      check("满载 楼梯蛇行" not in first, f"{r}: the full payload's meander reported only")
      check("楼梯蛇行峰峰值超过" not in first.replace("常用负载 楼梯蛇行峰峰值超过", ""),
            f"{r}: the unloaded meander (2.3 cm) not in the verdict")
  os.environ.pop("U_JUDGE_OLD", None)
  print("ALL OK" if not fails else f"FAILED: {fails}")
  return 1 if fails else 0


if __name__ == "__main__":
  sys.exit(main())
