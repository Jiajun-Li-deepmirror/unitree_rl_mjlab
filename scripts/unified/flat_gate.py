#!/usr/bin/env python3
"""flat_gate.py R PREFIX (unitree-rl-mjlab-62 2026-11-13; go2-depth's decision 11-13 03:36): does a teacher checkpoint pass
the flat part of its gate? Read from tier-2's own flat tests in both scan modes (jobs/unified/flat_gate.sh: accept.sh's
unloaded flat set on Unitree-<R>-U-Teacher-Flat): PREFIX_<on|off>_<test>.txt.

Why tier 2 and not the training's readings: on Go2 s34 model_6097 the training's flat readings (side_drift 0.08-0.095,
head_med 4.4-4.7 deg, multipliers at their cap) read about half of tier 2's (drift 14.8 / 18.7 %, heading p50 -12.6 /
-17.3 deg sideways right); Go2W s25 model_4997's creep (2.3 / 2.5 cm), wheel lift and falls at 1.5 m/s do not show in
the training's readings at all.

Judged in each scan mode as the students' acceptance judges them (scripts/unified/accept_judge.py: accept_tables'
goal_cells, the same parsers and bars): flat speed, drift and heading, turning in place, alternating spins, stop, march,
gait clock, falls, wheeled: wheels down; the zero-command stance (long_stand 60 s and stop_then_move: p90 deviation
<= accept_judge.POSTURE_BAR, no fall after the settle); and no fall in the gait-clock test either (go2-depth: "平地摔倒
0"; the students' acceptance reads no falls there). Not here: the stairs (stairs_gate.py) and the loaded / typical
conditions (the teacher's loaded behaviour is judged on the stairs gate's loaded run).
Reported per mode: drift %, creep, alternating-spin drift, march drift (payload_gap's readings).
First line: "PASS ...", "FAIL (gap G): <scan mode item: tags; ...>" (G = 0.01 x the failed items: teacher.sh's best
judged at the cap then ranks a stairs pass that fails here before most stairs fails) or "FAULT: <what was not measured>".
Exit 0 / 1 / 4.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "unified"))
import accept_judge as aj  # noqa: E402

at = aj.at
pg = aj.payload_gap
MODES = ("on", "off")
FLAT_GOALS = {k: v for k, v in aj.GOALS.items() if k != "g_stairs"}


def clock_falls(path: str) -> int | None:
  """gait_clock.py's falls column ("| N/M |" per command), summed; None without a row."""
  rows = re.findall(r"\|\s*(\d+)/\d+\s*\|", aj.txt(path))
  return sum(int(n) for n in rows) if rows else None


def judge(r: str, prefix: str) -> tuple[str, int]:
  wheeled = r in at.WHEELED
  at.STOP_FEET_DEPLOYED = {}
  goals = dict(FLAT_GOALS)
  if wheeled:
    goals["g_wheels"] = "四轮着地"
  bad, missing, detail = [], [], []
  for mode in MODES:
    q, tag = f"{prefix}_{mode}", f"scan {mode}"
    p = {k: f"{q}_{k}.txt" for k in at.KINDS}
    cells = at.goal_cells(r, 1, "", p, [])
    for k, name in goals.items():
      c = cells.get(k, "未测")
      if c == "通过":
        continue
      if c.startswith(at.FAIL):
        bad.append(f"{tag} {name}：{c[len(at.FAIL):]}")
      else:
        missing.append(f"{tag} {name}")
    sp, sl = aj.stand_posture(f"{q}_loadstand.txt")
    mp, mf = aj.stopmove(f"{q}_stopmove.txt")
    detail.append(f"{tag}: 零指令站姿 静站 p90 {sp} rad，3 s 后摔 {sl}；急停后 p90 {mp} rad，摔 {mf}")
    if sp is None or mp is None:
      missing.append(f"{tag} 零指令站姿")
    else:
      tags = (["偏离站姿"] if sp > aj.POSTURE_BAR or mp > aj.POSTURE_BAR else []) \
        + (["摔倒"] if (sl or 0) > 0 or (mf or 0) > 0 else [])
      if tags:
        bad.append(f"{tag} 零指令站姿：" + "、".join(tags))
    cf = clock_falls(f"{q}_clock.txt")
    if cf:
      bad.append(f"{tag} 步态节奏测试摔倒 {cf}")
    reads = []
    for _kind, label, fn in pg.ITEMS:
      v = fn(pg.files(q, False))
      reads.append(f"{label} {'-' if v is None else f'{v:.2f}'}")
    detail.append(f"{tag}: " + "，".join(reads))
  if bad:
    head, rc = f"FAIL (gap {0.01 * len(bad):.2f}): " + "；".join(bad + [f"未测 {m}" for m in missing]), 1
  elif missing:
    head, rc = "FAULT: 未测 " + "；".join(missing), 4
  else:
    head, rc = "PASS (the tier-2 flat tests in both scan modes)", 0
  return "\n".join([head] + [f"  {d}" for d in detail]), rc


def main() -> int:
  if len(sys.argv) != 3:
    print(__doc__)
    return 2
  try:
    out, rc = judge(sys.argv[1], sys.argv[2])
  except Exception as e:   # a tool fault, not a fail
    print(f"FAULT: flat gate error: {type(e).__name__}: {e}")
    return 4
  print(out)
  return rc


if __name__ == "__main__":
  sys.exit(main())
