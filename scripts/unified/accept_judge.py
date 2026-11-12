#!/usr/bin/env python3
"""accept_judge.py R PREFIX (go2-depth 2026-11-12, doc/redesign.md 2.2): the mjlab acceptance of one candidate, from
jobs/unified/accept.sh's files (one pooled run per test at 3x the robots: the "3 seeds pooled"). First line PASS or
FAIL: <item: reason; ...>, then the details.

Unloaded (PREFIX_<kind>.txt, fixed 0 kg), judged as scripts/blind/acceptance/accept_tables.py's goal_cells judges them
(the same parsers and bars) except: stop-feet against the fixed 0.9 (no deployed comparison), no luck rule, and the
loaded-stand / posture goals replaced by the zero-command stance below. Items: flat speed, drift, spin, alternating
spins, stop, march, gait clock, stairs (reach with / without operator, speed, stop, stop-feet), wheels down (wheeled),
falls; plus
  zero-command stance  long_stand (60 s) and stop_then_move (after a 1.5 m/s stop): leg joints' deviation from the
                       default stance p90 <= 0.15 rad, 0 falls after 3 s -- unloaded and loaded
  stairs counts        src/metrics (tinynav's counters) while crossing: legged kicks <= 0.245 and slips <= 0.0245 per
                       step, mid >= 80%, edge <= 10%; wheeled slide-back <= 5% of step-ups
Loaded (PREFIX_<kind>_pay.txt, the walking payload): stairs 0 falls; the zero-command stance; drift / creep /
alternating spins / march drift reported beside the unloaded ones (scripts/blind/acceptance/payload_gap.py), not judged
-- but a fall in those four tests fails it (2026-11-12, doc/redesign.md 2.5 item 10).
Typical payload (PREFIX_<kind>_typ.txt, doc/redesign.md 2.2, 2026-11-12; robots without one: not run): the loaded
condition's tests (wheeled: also the wheels down), every item at the unloaded bar, no fall in any of them (typical()).
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "blind" / "acceptance"))
import accept_tables as at  # noqa: E402
import payload_gap  # noqa: E402

POSTURE_BAR = 0.15
STAND_SETTLE = 3.0
LEGGED_BARS = {"kicks": 0.245, "slips": 0.0245, "mid": 0.80, "edge": 0.10}
WHEELED_SLIDE_BAR = 0.05
GOALS = {"g_speed": "平地速度", "g_drift": "跑偏", "g_spin": "原地旋转", "g_spinalt": "交替原地转", "g_stop": "停车",
         "g_march": "原地踏步", "g_clock": "步态节奏", "g_stairs": "楼梯", "g_falls": "摔倒"}


def txt(path: str) -> str:
  try:
    return Path(path).read_text()
  except OSError:
    return ""


def stand_posture(path: str) -> tuple[float | None, int | None]:
  """long_stand: (posture p90 rad, falls after the settle)."""
  t = txt(path)
  m = re.search(r"posture: .*? p90 ([\d.na]+) rad", t)
  p90 = float(m.group(1)) if m and m.group(1) != "nan" else None
  f = re.search(r"falls (\d+)/\d+ at t \[([^\]]*)\]", t)
  late = None
  if f:
    times = [float(x) for x in f.group(2).split(",") if x.strip()]
    late = sum(1 for x in times if x >= STAND_SETTLE) + max(0, int(f.group(1)) - len(times))   # beyond the list's 20
  return p90, late


def stopmove(path: str) -> tuple[float | None, int | None]:
  """stop_then_move: (the largest "after stop" p90 deviation, falls over all groups)."""
  p = {"stopmove": path}
  dev = at.posture_vals(p)
  falls = [int(a) for a, _b in re.findall(r"\|\s*(\d+)/(\d+)\s*$", txt(path), re.M)]
  return dev, (sum(falls) if falls else None)


def stairs_metrics(path: str) -> dict[str, float]:
  t = txt(path)
  out = {}
  m = re.search(r"metrics \(src/metrics\): steps ([\d.]+), kicks ([\d.]+) / step, slips ([\d.]+) / step, mid ([\d.]+), "
                r"edge ([\d.]+)", t)
  if m:
    out.update(steps=float(m.group(1)), kicks=float(m.group(2)), slips=float(m.group(3)), mid=float(m.group(4)),
               edge=float(m.group(5)))
  m = re.search(r"metrics \(src/metrics\): step-ups ([\d.]+), slide-back ([\d.]+)", t)
  if m:
    out.update(ups=float(m.group(1)), slide_back=float(m.group(2)))
  return out


def stairs_falls(path: str) -> int | None:
  """Any fall in eval_stairs' table (traverse or stop), in % of the robots of each row."""
  rows = re.findall(r"^\s*[\d-]+cm\s+(?:up|down)\s*\|\s*\d+%\s+(\d+)%.*\|\s*\d+%\s+(\d+)%", txt(path), re.M)
  if not rows:
    return None
  return sum(int(a) + int(b) for a, b in rows)


# The typical payload's stairs speed: at least this share of the same test's unloaded reading, per band and direction
# (2026-11-12, go2-depth's value in doc/redesign.md 2.2, the user may change it); eval_stairs.py's "speed" lines.
STAIRS_SPEED_REL = 0.8


def flight_speeds(path: str) -> dict[tuple[str, str], float]:
  """eval_stairs.py's speed on the flight per (band, direction), m/s (nan: no robot got past the last step)."""
  out = {}
  for m in re.finditer(r"^speed\s+(\d+-\d+cm)\s+(up|down)\s*\|\s*([\d.]+|nan) m/s", txt(path), re.M):
    out[(m.group(1), m.group(2))] = float(m.group(3))
  return out


def comparison_falls(p: dict) -> dict[str, int | None]:
  """Falls in the tests whose readings are only compared under the walking payload (2026-11-12, doc/redesign.md 2.5
  item 10: they count toward the loaded condition's 0 falls); also the typical payload's falls in the same tests."""
  d, sa, g, rv = at.drift_vals(p), at.spinalt_vals(p), at.stand_vals(p), at.reset_vals(p)
  return {"跑偏": sum(v["fell"] for v in d.values()) if d else None,
          "交替原地转": sum(v["fell"] for v in sa.values()) if sa else None,
          "停车": sum(v["fell"] for v in g.values()) if g else None,
          "原地踏步": rv["fell"] if rv else None}


def typical(r: str, prefix: str, wheeled: bool) -> tuple[list[str], list[str]]:
  """The typical payload (doc/redesign.md 2.2, 2026-11-12): the loaded condition's tests (wheeled: also the wheels
  down) with the typical payload, every item at the unloaded bar, and no fall in any of them. Not run: the 44 flat
  commands, the gait clock and the map3 flights -- so the stairs reach and speed (both map3's) are not judged here;
  the stairs items are eval_stairs' own: stop ok, all feet down when stopped, the counts, no fall."""
  p = {k: f"{prefix}_{k}_typ.txt" for k in at.KINDS}
  cells = at.goal_cells(r, 1, "", p, [])
  bad, detail = [], []
  goals = {"g_drift": "跑偏", "g_spinalt": "交替原地转", "g_stop": "停车", "g_march": "原地踏步"}
  if wheeled:
    goals["g_wheels"] = "四轮着地"
  for k, n in goals.items():
    if cells.get(k) != "通过":
      bad.append(f"常用负载 {n}：{cells.get(k, '未测')}")
  # zero-command stance
  sp, sl = stand_posture(p["loadstand"])
  mp, mf = stopmove(p["stopmove"])
  detail.append(f"零指令站姿（常用负载）：静站 p90 {sp} rad，3 s 后摔 {sl}；急停后 p90 {mp} rad，摔 {mf}")
  if sp is None or mp is None:
    bad.append("零指令站姿（常用负载）：未测")
  else:
    if sp > POSTURE_BAR or mp > POSTURE_BAR:
      bad.append("零指令站姿（常用负载）：偏离站姿")
    if (sl or 0) > 0 or (mf or 0) > 0:
      bad.append("零指令站姿（常用负载）：摔倒")
  # stairs: eval_stairs' stop and stop-feet, the counts, no fall
  st = at.stop_row(p)
  if st is None:
    bad.append("常用负载 楼梯：未测")
  else:
    if float(st.split("|")[2].strip().rstrip("%")) < 95.0:
      bad.append("常用负载 楼梯：停不稳")
    feet = [(m.group(1), m.group(2), float(m.group(3))) for m in re.finditer(
      r"^\s*(\d+-\d+cm)\s+(up|down)\s*\|.*\|\s*\d+%\s+\d+%\s+[\d.]+\s+([\d.]+)\s*$", txt(p["stairs"]), re.M)]
    if any(v < at.stop_feet_bar(r, b, d) for b, d, v in feet):
      bad.append("常用负载 楼梯：停住缺脚")
    if stairs_falls(p["stairs"]):
      bad.append("常用负载 楼梯：摔倒")
  # stairs speed: against the unloaded reading of the same test, band by band
  su, st_ = flight_speeds(f"{prefix}_stairs.txt"), flight_speeds(p["stairs"])
  if not su or not st_:
    bad.append("常用负载 楼梯速度：未测")
  else:
    slow = []
    for key, base in su.items():
      v = st_.get(key, float("nan"))
      if math.isfinite(base) and not (math.isfinite(v) and v >= STAIRS_SPEED_REL * base):
        slow.append(f"{key[0]} {'上' if key[1] == 'up' else '下'} {v:.2f}/{base:.2f}")
    detail.append("常用负载 楼梯速度（常用 / 空载，m/s）：" + "，".join(
      f"{k[0]} {'上' if k[1] == 'up' else '下'} {st_.get(k, float('nan')):.2f}/{v:.2f}" for k, v in su.items()))
    if slow:
      bad.append(f"常用负载 楼梯速度低于空载的 {STAIRS_SPEED_REL:.0%}：" + "、".join(slow))
  sm = stairs_metrics(p["stairs"])
  if wheeled:
    sb = sm.get("slide_back")
    detail.append(f"常用负载 楼梯计数：上台阶 {sm.get('ups')}，滑回 {sb}")
    if sb is None:
      bad.append("常用负载 楼梯计数：未测")
    elif sb > WHEELED_SLIDE_BAR:
      bad.append(f"常用负载 楼梯计数：滑回 {sb:.3f}")
  else:
    detail.append("常用负载 楼梯计数：" + "，".join(f"{k} {sm.get(k)}" for k in ("steps", "kicks", "slips", "mid", "edge")))
    if "kicks" not in sm:
      bad.append("常用负载 楼梯计数：未测")
    else:
      tags = [f"{k} {sm[k]:.3f}" for k in ("kicks", "slips", "edge") if sm[k] > LEGGED_BARS[k]]
      if sm["mid"] < LEGGED_BARS["mid"]:
        tags.append(f"mid {sm['mid']:.3f}")
      if tags:
        bad.append("常用负载 楼梯计数：" + "、".join(tags))
  # no fall in any test run (goal_cells' own fall item reads the 44 flat commands, not run here: per test instead)
  cf = comparison_falls(p)
  if any(v for v in cf.values() if v):
    bad.append("常用负载：摔倒（" + "、".join(k for k, v in cf.items() if v) + "）")
  if wheeled:
    n = at.nolift_vals(p)
    if n and n["fell"]:
      bad.append("常用负载：摔倒（四轮着地）")
  detail.append("常用负载各项：" + "，".join(f"{n} {cells.get(k, '未测')}" for k, n in goals.items())
                + "；摔倒 " + "，".join(f"{k} {v}" for k, v in cf.items()))
  return bad, detail


def main(argv: list[str]) -> int:
  r, prefix = argv[0], argv[1]
  wheeled = r in at.WHEELED
  at.STOP_FEET_DEPLOYED = {}   # the fixed bar (doc/redesign.md 2.2)
  p = {k: f"{prefix}_{k}.txt" for k in at.KINDS}
  cells = at.goal_cells(r, 1, "", p, [])
  goals = dict(GOALS)
  if wheeled:
    goals["g_wheels"] = "四轮着地"
  bad = [f"{n}：{cells.get(k, '未测')}" for k, n in goals.items() if cells.get(k) != "通过"]
  detail = []
  # zero-command stance, both conditions
  for cond, sfx in (("空载", ""), ("负载", "_pay")):
    sp, sl = stand_posture(f"{prefix}_loadstand{sfx}.txt")
    mp, mf = stopmove(f"{prefix}_stopmove{sfx}.txt")
    tags = []
    if sp is None or mp is None:
      tags.append("未测")
    else:
      if sp > POSTURE_BAR or mp > POSTURE_BAR:
        tags.append("偏离站姿")
      if (sl or 0) > 0 or (mf or 0) > 0:
        tags.append("摔倒")
    detail.append(f"零指令站姿（{cond}）：静站 p90 {sp} rad，3 s 后摔 {sl}；急停后 p90 {mp} rad，摔 {mf}")
    if tags:
      bad.append(f"零指令站姿（{cond}）：" + "、".join(tags))
  # stairs counts (unloaded)
  sm = stairs_metrics(p["stairs"])
  if wheeled:
    sb = sm.get("slide_back")
    detail.append(f"楼梯计数：上台阶 {sm.get('ups')}，滑回 {sb}")
    if sb is None:
      bad.append("楼梯计数：未测")
    elif sb > WHEELED_SLIDE_BAR:
      bad.append(f"楼梯计数：滑回 {sb:.3f}")
  else:
    detail.append("楼梯计数：" + "，".join(f"{k} {sm.get(k)}" for k in ("steps", "kicks", "slips", "mid", "edge")))
    if "kicks" not in sm:
      bad.append("楼梯计数：未测")
    else:
      tags = [f"{k} {sm[k]:.3f}" for k in ("kicks", "slips", "edge") if sm[k] > LEGGED_BARS[k]]
      if sm["mid"] < LEGGED_BARS["mid"]:
        tags.append(f"mid {sm['mid']:.3f}")
      if tags:
        bad.append("楼梯计数：" + "、".join(tags))
  # loaded stairs: 0 falls
  lf = stairs_falls(f"{prefix}_stairs_pay.txt")
  detail.append(f"负载楼梯：摔倒合计 {lf}（各档百分比之和）")
  if lf is None:
    bad.append("负载楼梯：未测")
  elif lf > 0:
    bad.append("负载楼梯：摔倒")
  # loaded: the falls in the four comparison-only tests count too (doc/redesign.md 2.5 item 10); their readings do not
  lfalls = comparison_falls({k: f"{prefix}_{k}_pay.txt" for k in at.KINDS})
  detail.append("负载（只出读数的四项）摔倒：" + "，".join(f"{k} {v}" for k, v in lfalls.items()))
  if any(v is None for v in lfalls.values()):
    bad.append("负载对比项：未测")
  elif any(v > 0 for v in lfalls.values()):
    bad.append("负载对比项：摔倒（" + "、".join(k for k, v in lfalls.items() if v) + "）")
  # the typical payload (doc/redesign.md 2.2; robots without one: not run)
  if any(os.path.exists(f"{prefix}_{k}_typ.txt") for k in ("stand", "drift3", "stairs")):
    tb, td = typical(r, prefix, wheeled)
    bad += tb
    detail += td
  else:
    detail.append("常用负载：本机型没有这一档，未测")
  print("PASS" if not bad else "FAIL: " + "；".join(bad))
  for line in detail:
    print("  " + line)
  print("  空载 / 负载 对比（不判）：")
  for kind, label, fn in payload_gap.ITEMS:
    a = fn(payload_gap.files(prefix, False))
    b = fn(payload_gap.files(prefix, True))
    f = lambda x: "-" if x is None else f"{x:.3f}"
    gap = "-" if a is None or b is None else f"{b - a:+.3f}"
    print(f"    {label}: {f(a)} / {f(b)} / {gap}")
  dr = txt(f"{prefix}_dr.txt").strip()
  if dr:
    print("  评估时的随机化：")
    for line in dr.splitlines():
      print("    " + line)
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
