"""luck_pass.py R PREFIX (go2-depth 2026-11-10, doc/acceptance.md, the user: "运气通过规则化"): whether a failed seed's check
may pass by the luck rule. ELIGIBLE when every failed goal fails on falls alone and exactly one robot fell over all the
tests (doc/acceptance.md §1: "所有测试合计只摔了 1 台"): the flat matrix, drift, stops, alternating turns, the loaded
stand (after its LOADSTAND_SETTLE, as its own goal), the march and its reset, the four-wheels-down drives, and the
stairs; else NO <why>. The chains then need the other seeds to pass (one luck seed per check) and log
the luck pass. (2026-11-11, the user: the loaded stand, march and wheels-down falls are counted too; the code had
counted the first five only, its docstring had named the loaded stand.)"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts/blind/acceptance").resolve()))
import accept_tables as at  # noqa: E402

r, prefix = sys.argv[1:3]
p = {k: f"{prefix}_{k}.txt" for k in at.KINDS}
cells = at.goal_cells(r, 1, "", p, [])
FALL_ONLY = {"g_falls"}   # its tags name where robots fell
bad = []
for k, v in cells.items():
  if not k.startswith("g_") or v in ("通过", at.NA, at.NOT_JUDGED, "不适用", "未测") or not str(v).startswith(at.FAIL):
    continue
  tags = str(v)[len(at.FAIL):].split("、")
  if k not in FALL_ONLY and any(t != "摔倒" for t in tags):
    bad.append(f"{k} {v}")
if bad:
  print("NO " + "; ".join(bad))
  sys.exit(0)
n = 0
a = at.am_vals(p)
n += a["falls"] if a else 0
n += sum(v["fell"] for v in at.drift_vals(p).values())
n += sum(v["fell"] for v in at.stand_vals(p).values())
n += sum(v["fell"] for v in (at.spinalt_vals(p) or {}).values())
# the loaded stand: falls after the settle (accept_tables' g_loadstand rule)
m = re.search(r"falls (\d+)/(\d+) at t \[([^\]]*)\]", at.txt(p.get("loadstand")) or "")
if m:
  ts = [float(t) for t in m.group(3).split(",") if t.strip()]
  n += sum(1 for t in ts if t >= at.LOADSTAND_SETTLE) + max(0, int(m.group(1)) - len(ts))
n += sum(v["fell"] for v in (at.march_vals(p) or {}).values())
rv = at.reset_vals(p)
n += rv["fell"] if rv else 0
nl = at.nolift_vals(p)
n += nl["fell"] if nl else 0
# (not stop_then_move's falls: that test judges the stance after a stop, not falls, and the user's list of the falls
# to add, 2026-11-11, was the loaded stand, the march and the four-wheels-down drives; counted 11-11 by mistake and
# taken out the same day, after gp2 avg13700: 18 of its 19 falls were there)
st = at.stop_row(p)
if st and st.split("|")[1].strip() != "0%":
  n += 1   # stairs falls come as a share of the runs: one incident at least
print("ELIGIBLE 1 fall" if n == 1 else f"NO {n} falls")
