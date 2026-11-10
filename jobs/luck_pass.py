"""luck_pass.py R PREFIX (go2-depth 2026-11-10, doc/acceptance.md, the user: "运气通过规则化"): whether a failed seed's check
may pass by the luck rule. ELIGIBLE when every failed goal fails on falls alone and exactly one robot fell over all the
judged tests (flat matrix, drift, stops, alternating turns, loaded stand, stairs); else NO <why>. The chains then need the
other seeds to pass (one luck seed per check) and log the luck pass."""
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
st = at.stop_row(p)
if st and st.split("|")[1].strip() != "0%":
  n += 1   # stairs falls come as a share of the runs: one incident at least
print("ELIGIBLE 1 fall" if n == 1 else f"NO {n} falls")
