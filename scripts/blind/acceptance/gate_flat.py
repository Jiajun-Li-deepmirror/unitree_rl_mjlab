"""Flat gate of a phased teacher checkpoint (user 2026-10-23): every flat goal of
accept_tables.goal_cells passes - flat speed, straight / sideways drift, turning in
place, stopping, marching, gait clock, and for wheeled robots all four wheels down -
and no robot fell in the flat tests (goal_cells' 不摔 also counts the stairs, which
the gate leaves to the mixed phase). The hip load is no longer a goal (user
2026-10-24 ~15:45); marching in place is judged as a posture reset for every robot (user
~23:20; see goal_cells and march_eval.py).

Usage: python3 gate_flat.py <robot> <results prefix, e.g. fix/ph_Go2_3000>
Prints PASS, or FAIL: <goal>：<tags>；... on one line.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accept_tables as at  # noqa: E402

FLAT = [("g_speed", "平地速度"), ("g_drift", "跑偏"), ("g_spin", "原地转"), ("g_stop", "停车"),
        ("g_march", "原地踏步"), ("g_clock", "步态节奏")]

r, prefix = sys.argv[1:3]
p = {k: f"{prefix}_{k}.txt" for k in at.KINDS}
cells = at.goal_cells(r, 1, "", p, [])
fails = []
for key, name in FLAT + ([("g_wheels", "四轮着地")] if r in at.WHEELED else []):
  v = cells.get(key)
  if v not in ("通过", at.NOT_JUDGED):
    fails.append(f"{name}：{(v or '无结果').replace(at.FAIL, '')}")
a, d, g = at.am_vals(p), at.drift_vals(p), at.stand_vals(p)
falls = (a["falls"] if a else 0) + sum(v["fell"] for v in d.values()) + sum(v["fell"] for v in g.values())
if falls:
  fails.append(f"摔倒：{falls} 台")
print("PASS" if not fails else "FAIL: " + "；".join(fails))
