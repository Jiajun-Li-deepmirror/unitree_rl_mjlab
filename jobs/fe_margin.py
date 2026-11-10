"""fe_margin.py R PREFIX (depth_fe, 2026-10-31): the stairs speed ratios of one check (median speed over the map3
flights with operator, up / down, as the acceptance's g_stairs judges them: 0.8-1.2) and their margin to that band,
min(up - 0.8, down - 0.8, 1.2 - up, 1.2 - down). Prints "up down margin". A teacher needs some margin: its student
lands a little below it (Go2 10-31: teacher descent 0.815 -> students 0.79-0.80, "下楼慢")."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts/blind/acceptance").resolve()))
import accept_tables as at  # noqa: E402

r, prefix = sys.argv[1:3]
s = at.stairs_vals({k: f"{prefix}_{k}.txt" for k in at.KINDS})
up, dn = (s["sp_up"], s["sp_dn"]) if s else (math.nan, math.nan)
m = min(up - 0.8, dn - 0.8, 1.2 - up, 1.2 - dn)
print(f"{up:.3f} {dn:.3f} {m:.3f}")
