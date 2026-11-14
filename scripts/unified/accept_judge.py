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
  stairs counts        src/metrics (tinynav's counters) while crossing: legged edge <= 10% (hard); the kicks on the
                       ascents, slips and mid are quality targets reported in their own block, not judged (2026-11-13,
                       the user; quality_lines); wheeled slide-back <= 5% of step-ups
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
# The stairs counts are judged only on enough events (2026-11-12, the user: fewer luck passes): at 0.05 a rate from
# ~100 step-ups has a 95% interval of about +-0.044 -- it cannot tell 0.05 from 0.10. Fewer: "未测" (a tool fault,
# not a fail). Go2W P4 11-12 read 103 / 121 step-ups of 576 robots (the counter's level bug, go2-depth's fix).
MIN_STAIR_EVENTS = 400   # wheeled step-ups / legged foot placements on the steps


def stairs_count_verdict(sm: dict, wheeled: bool, label: str) -> tuple[str, str | None]:
  """(detail line, the reason it does not pass or None) for eval_stairs' counts (label: "" or "常用负载 ")."""
  if wheeled:
    sb, n = sm.get("slide_back"), sm.get("ups")
    detail = f"{label}楼梯计数：上台阶 {n}，滑回 {sb}"
    if sb is None or n is None:
      return detail, f"{label}楼梯计数：未测"
    if n < MIN_STAIR_EVENTS:
      return detail, f"{label}楼梯计数：未测（上台阶 {n:.0f} 次，少于 {MIN_STAIR_EVENTS}）"
    return detail, (f"{label}楼梯计数：滑回 {sb:.3f}" if sb > WHEELED_SLIDE_BAR else None)
  detail = f"{label}楼梯计数：" + "，".join(f"{k} {sm.get(k)}" for k in ("steps", "kicks", "slips", "mid", "edge"))
  if "kicks" not in sm:
    return detail, f"{label}楼梯计数：未测"
  if sm.get("steps", 0) < MIN_STAIR_EVENTS:
    return detail, f"{label}楼梯计数：未测（落脚 {sm.get('steps', 0):.0f} 次，少于 {MIN_STAIR_EVENTS}）"
  # hard: the edge only (2026-11-13, the user: mid is a means to keep off the edge; kicks / slips / mid are quality
  # targets -- quality_lines(), outside PASS / FAIL)
  return detail, (f"{label}楼梯计数：edge {sm['edge']:.3f}" if sm["edge"] > LEGGED_BARS["edge"] else None)


# The legged stairs counts' quality targets (2026-11-13; the user ~15:00 CST: "我其实没有明确的数字要求，尽量满足上楼梯不要
# 踢立面，最好踩在中间就行。map3每段踢立面最好能控制在2次左右就行，踩中间其实主要是为了不要踩楼梯边缘导致脚滑，没有硬性要
# 求"; go2-depth's decision): not in PASS / FAIL; each reported beside the user's target and the delivered model's
# reading (达到 / 未达到 / 差于已交付), and the report to the user lists them as they are. Kicks on the ASCENTS only
# (eval_stairs' tile rows "up", step-weighted), also per map3 flight (x FLIGHT_STEPS steps); the descents' kicks and mid
# reported only. The edge stays hard (stairs_count_verdict).
QUALITY_TARGETS = {"kicks_up": ("max", 0.245), "slips": ("max", 0.0245), "mid": ("min", 0.80)}
DELIVERED_REF = {   # the delivered legged blind policy's tier-2 reading, per robot (none: no comparison)
  "Go2": {"kicks_up": 0.356, "slips": 0.068, "mid": 0.778, "src": "go2_blind_v4 model_10000 (0dc3b56 counters)"},
}
FLIGHT_STEPS = 8.15   # steps per map3 flight (go2-depth 11-13): kicks per flight = kicks per step x this


def stairs_dir_counts(path: str) -> dict[str, dict[str, float]]:
  """eval_stairs --metrics' tile rows pooled per direction ("up" / "down"), each count weighted by its tile's steps:
  {dir: {steps, kicks, slips, mid, edge}} (a count a tile printed as '-' is left out of its own average)."""
  acc: dict[str, dict] = {}
  for m in re.finditer(r"^tile \S+ (up|down) \S+: steps ([\d.]+), kicks ([\d.]+|-), slips ([\d.]+|-), mid ([\d.]+|-), "
                       r"edge ([\d.]+|-)", txt(path), re.M):
    d, n = m.group(1), float(m.group(2))
    a = acc.setdefault(d, {"steps": 0.0, **{k: [0.0, 0.0] for k in ("kicks", "slips", "mid", "edge")}})
    a["steps"] += n
    for k, g in (("kicks", 3), ("slips", 4), ("mid", 5), ("edge", 6)):
      if m.group(g) != "-":
        a[k][0] += float(m.group(g)) * n
        a[k][1] += n
  return {d: {"steps": a["steps"], **{k: (a[k][0] / a[k][1] if a[k][1] else None)
                                       for k in ("kicks", "slips", "mid", "edge")}} for d, a in acc.items()}


QUALITY_GATED = ("kicks_up", "slips")   # the targets an early stop needs no worse than delivered (mid: reported)
QUALITY_UNMEASURED_GAP = 9.0             # the gap an unmeasured one counts (finite: the shell sorts and compares it)


def quality_gap(r: str, path: str) -> tuple[float, str]:
  """(gap, text) of one eval_stairs output against the delivered model's readings (DELIVERED_REF): the summed relative
  excess of QUALITY_GATED (the teacher gate's rule, 2026-11-13); 0 when no worse, or when the robot has no reference
  (wheeled ones: none -- their counts are hard items)."""
  ref = DELIVERED_REF.get(r)
  if not ref or r in at.WHEELED:
    return 0.0, "no delivered reference"
  dc, sm = stairs_dir_counts(path), stairs_metrics(path)
  vals = {"kicks_up": (dc.get("up") or {}).get("kicks"), "slips": sm.get("slips")}
  gap, parts = 0.0, []
  for k in QUALITY_GATED:
    v = vals.get(k)
    if v is None:   # not measured is not "no worse": no early stop on it
      parts.append(f"{k} 未测")
      gap += QUALITY_UNMEASURED_GAP
      continue
    parts.append(f"{k} {v:.3f}/{ref[k]}")
    gap += max(0.0, v / ref[k] - 1.0)
  return gap, ", ".join(parts)


def quality_lines(r: str, path: str, label: str = "") -> list[str]:
  """The legged quality-target block for one eval_stairs output (not judged)."""
  dc, sm = stairs_dir_counts(path), stairs_metrics(path)
  ref = DELIVERED_REF.get(r, {})
  vals = {"kicks_up": (dc.get("up") or {}).get("kicks"), "slips": sm.get("slips"), "mid": sm.get("mid")}
  names = {"kicks_up": "上楼 kicks（每步）", "slips": "slips（每步）", "mid": "mid（落在踏面中间）"}
  out = [f"质量目标（非硬性）{label}："]
  for k, (sense, goal) in QUALITY_TARGETS.items():
    v, rv = vals[k], ref.get(k)
    if v is None:
      out.append(f"  {names[k]}：未测")
      continue
    met = v <= goal if sense == "max" else v >= goal
    worse = rv is not None and (v > rv if sense == "max" else v < rv)
    status = "达到" if met else ("差于已交付" if worse else "未达到")
    extra = f"（每段约 {v * FLIGHT_STEPS:.1f} 次；目标每段约 {goal * FLIGHT_STEPS:.0f} 次）" if k == "kicks_up" else ""
    out.append(f"  {names[k]}：{v:.3f}{extra}；目标 {'≤' if sense == 'max' else '≥'} {goal}；"
               f"已交付 {rv if rv is not None else '-'} → {status}")
  dn = (dc.get("down") or {}).get("kicks")
  out.append(f"  下楼 kicks（每步，只报告）：{'-' if dn is None else f'{dn:.3f}'}"
             + (f"；已交付参考：{ref['src']}" if ref else ""))
  return out
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
  """stop_then_move: (the largest "after stop" p90 deviation, falls over all groups; a group printed "all fell"
  counts its robots -- the header's "N robots per group")."""
  p = {"stopmove": path}
  t = txt(path)
  dev = at.posture_vals(p)
  falls = [int(a) for a, _b in re.findall(r"\|\s*(\d+)/(\d+)\s*$", t, re.M)]
  allfell = len(re.findall(r"\|\s*all fell\s*$", t, re.M))
  if allfell:
    m = re.search(r"(\d+) robots per group", t)
    falls.append(allfell * (int(m.group(1)) if m else 1))
  return dev, (sum(falls) if falls else None)


def stairs_fall_counts(path: str) -> int | None:
  """eval_stairs.py's exact fall count, traverse + stop (its "falls:" line, 2026-11-12); None without it."""
  m = re.search(r"^falls: traverse (\d+) of \d+, stop (\d+) of \d+", txt(path), re.M)
  return None if m is None else int(m.group(1)) + int(m.group(2))


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


def stairs_reach(path: str) -> str | None:
  """Per band and direction the traverse's reached / stalled / fell shares (eval_stairs' table and its "stalled" line;
  2026-11-13, go2-depth: under load only the falls are judged -- the user: "尽可能和空载一样，物理上做不到也无所谓，
  但一定不能摔" -- so a refusal to climb must at least be visible beside them). None without the table."""
  t = txt(path)
  rows = re.findall(r"^\s*(\d+-\d+cm)\s+(up|down) \|\s+(\d+)%\s+(\d+)%", t, re.M)
  if not rows:
    return None
  st = stairs_stalled(path) or {}
  return "，".join(f"{b} {'上' if d == 'up' else '下'} 到达 {ok}% 停住 {f'{st[(b, d)]:g}%' if (b, d) in st else '-'} 摔 {fl}%"
                  for b, d, ok, fl in rows)


def stairs_stalled(path: str) -> dict[tuple[str, str], float] | None:
  """eval_stairs' "stalled on the traverse" line per (band, direction): % of that band-direction's robots that neither
  fell nor got past the last step in time; None without the line."""
  m = re.search(r"^stalled on the traverse.*$", txt(path), re.M)
  if m is None:
    return None
  return {(b, d): float(v) for b, d, v in re.findall(r"(\d+-\d+cm) (up|down) ([\d.]+)%", m.group(0))}


# The typical payload's stairs: at most this % of a band-direction's robots stalled (go2-depth 11-13: "常用负载每项按空载
# 门槛" for the reach -- the unloaded stop ok's 95%; typical() read only the flight speed of the robots that got past, so a
# 40% stall at 15-18 cm up (Go2W s33 4997) passed). Also the teacher gate's --typ (scripts/unified/stairs_gate.py).
TYPICAL_STALL_MAX = 5.0


HEADING_WARN_DEG = 3.0   # tinynav's stairs heading bar (judged there, tier 3); tier 2 only warns


def stairs_heading(path: str) -> tuple[float | None, float | None]:
  """eval_stairs' "heading on the flight" line (2026-11-13): (all, 15-18 cm up) in degrees, or Nones."""
  m = re.search(r"heading on the flight: ([\d.]+) deg.*?15-18cm up ([\d.]+)", txt(path))
  return (None, None) if m is None else (float(m.group(1)), float(m.group(2)))


# The full payload's stairs falls per robot (2026-11-14, the user: "约 12:30 G11 的分档数据出来，或者到约 6000 步还卡在这一项
# 时，再改用方案二"; go2-depth: only Go2 switches, Go2W stays): a robot listed here has its full-load falls judged on the
# 5-15 cm bands only -- 15-18 cm (above Go2's rated ~16 cm step) measured and reported, not a fail; unloaded and the
# typical payload are judged on every band as before. Every robot not listed: every band (the user's option 1, the
# default). The teacher's stairs gate (stairs_gate.py) and the students' acceptance (judge below) read the same set.
# U_JUDGE_LOADED_TOP_REPORTED="Go2,..." adds robots for a test run.
LOADED_TOP_REPORTED: frozenset[str] = frozenset()
LOADED_TOP_BAND = "15-18cm"


def loaded_top_reported(r: str) -> bool:
  return r in LOADED_TOP_REPORTED or r in {x for x in os.environ.get("U_JUDGE_LOADED_TOP_REPORTED", "").split(",") if x}


def band_fall_counts(path: str) -> dict[tuple[str, str], int] | None:
  """eval_stairs.py's falls (traverse + stop) per (band, direction), from its table's whole percents and its robots
  per (band, direction): exact while those robots are at most 100 (one robot is then over 1 %); None otherwise, or
  when the sum differs from the exact "falls:" count."""
  t = txt(path)
  m = re.search(r"(\d+) robots per \(band, direction\)", t)
  rows = re.findall(r"^\s*([\d-]+cm)\s+(up|down)\s*\|\s*\d+%\s+(\d+)%.*\|\s*\d+%\s+(\d+)%", t, re.M)
  if m is None or not rows or int(m.group(1)) > 100:
    return None
  n = int(m.group(1))
  out = {(b, d): round(int(a) * n / 100) + round(int(c) * n / 100) for b, d, a, c in rows}
  total = stairs_fall_counts(path)
  if total is not None and sum(out.values()) != total:
    return None
  return out


def loaded_falls(r: str, path: str) -> tuple[int | None, int | None]:
  """The full payload's stairs falls as judged, and those on the top band only reported (None: none reported): every
  band, unless the robot is in LOADED_TOP_REPORTED and the per-band counts can be read (else every band, as before)."""
  total = stairs_falls(path)
  if total is None or not loaded_top_reported(r):
    return total, None
  bands = band_fall_counts(path)
  if bands is None:
    return total, None
  top = sum(v for (b, _d), v in bands.items() if b == LOADED_TOP_BAND)
  return total - top, top


def stairs_falls(path: str) -> int | None:
  """Any fall in eval_stairs (traverse or stop): the exact count of its "falls:" line, else (a file from before it)
  the table's whole percents of each row summed -- which read one or two robots of a row as 0%."""
  n = stairs_fall_counts(path)
  if n is not None:
    return n
  rows = re.findall(r"^\s*[\d-]+cm\s+(?:up|down)\s*\|\s*\d+%\s+(\d+)%.*\|\s*\d+%\s+(\d+)%", txt(path), re.M)
  if not rows:
    return None
  return sum(int(a) + int(b) for a, b in rows)


# What the judge cannot read is a tool fault, not a fail (the user's "fault != fail"): an item whose reason is one of
# these, when every reason is one, makes the verdict FAULT (accept.sh: exit 4, no fallback to another teacher seed).
MISSING = ("未测", "评测中")


def verdict_line(bad: list[str]) -> str:
  if not bad:
    return "PASS"
  if all(any(w in b for w in MISSING) for b in bad):
    return "FAULT: " + "；".join(bad)
  return "FAIL: " + "；".join(bad)


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
  commands, the gait clock and the map3 flights, so map3's stairs reach is not judged here; the stairs items are
  eval_stairs' own: stop ok, all feet down when stopped, the counts, no fall, the speed on the flight at least
  STAIRS_SPEED_REL of the same test's unloaded reading in every band and direction, and at most TYPICAL_STALL_MAX %
  stalled in every band and direction (11-13)."""
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
    if not feet:
      bad.append("常用负载 楼梯停住四脚着地：未测")
    elif any(v < at.stop_feet_bar(r, b, d) for b, d, v in feet):
      bad.append("常用负载 楼梯：停住缺脚")
    tf = stairs_falls(p["stairs"])
    if tf is None:
      bad.append("常用负载 楼梯摔倒：未测")
    elif tf > 0:
      bad.append(f"常用负载 楼梯：摔倒 {tf}")
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
  # stairs reach: the stalled share per band and direction (the speed above reads only the robots that got past)
  ss = stairs_stalled(p["stairs"])
  if not ss:
    bad.append("常用负载 楼梯停住：未测")
  else:
    detail.append("常用负载 楼梯每档停住：" + "，".join(f"{b} {'上' if d == 'up' else '下'} {v:g}%" for (b, d), v in ss.items()))
    over = [f"{b} {'上' if d == 'up' else '下'} {v:g}%" for (b, d), v in ss.items() if v > TYPICAL_STALL_MAX]
    if over:
      bad.append(f"常用负载 楼梯停住超过 {TYPICAL_STALL_MAX:g}%：" + "、".join(over))
  line, why = stairs_count_verdict(stairs_metrics(p["stairs"]), wheeled, "常用负载 ")
  detail.append(line)
  if why:
    bad.append(why)
  if not wheeled:
    detail.extend(quality_lines(r, p["stairs"], "（常用负载）"))
  # no fall in any test run (goal_cells' own fall item reads the 44 flat commands, not run here: per test instead)
  cf = comparison_falls(p)
  if any(v is None for v in cf.values()):
    bad.append("常用负载摔倒：未测（" + "、".join(k for k, v in cf.items() if v is None) + "）")
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
  """accept_judge.py R PREFIX [--typical]: --typical (accept.sh, when the robot has a typical payload and the
  condition ran) makes its files required (missing: 未测, a tool fault), instead of looked for."""
  try:
    return judge(argv)
  except Exception as e:   # a malformed line or file: what the judge cannot read is a tool fault, not a fail
    import traceback
    print(f"FAULT: judge error: {type(e).__name__}: {e}")
    traceback.print_exc(file=sys.stdout)
    return 0


def judge(argv: list[str]) -> int:
  expect_typ = "--typical" in argv
  argv = [a for a in argv if a != "--typical"]
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
  line, why = stairs_count_verdict(stairs_metrics(p["stairs"]), wheeled, "")
  detail.append(line)
  if why:
    bad.append(why)
  if not wheeled:
    detail.extend(quality_lines(r, p["stairs"]))
  # the heading on the flight: a reading here, judged in tinynav (the user: "heading需要验收"; go2-depth 11-13: tier 2
  # warns only -- a judged tier-2 bar would be a new acceptance rule)
  ha, hu = stairs_heading(p["stairs"])
  if ha is not None:
    warn = ha > HEADING_WARN_DEG or hu > HEADING_WARN_DEG
    detail.append(f"楼梯航向（只出读数，tinynav 判 {HEADING_WARN_DEG:.0f}°）：合计 {ha:.1f}°，15–18 cm 上楼 {hu:.1f}°"
                  + ("  WARN：超过 3°，tinynav 可能不过" if warn else ""))
  # unloaded stairs: 0 falls, counted exactly (traverse and stop; accept_tables' fall item reads the traverse's whole
  # percent only -- doc/redesign.md 2.2 "空载测试全部 0 摔", go2-depth's review 11-12 item 15)
  uf = stairs_falls(p["stairs"])
  detail.append(f"空载楼梯：摔倒 {uf}" + ("（精确计数）" if stairs_fall_counts(p["stairs"]) is not None else "（各档百分比之和）"))
  if uf is None:
    bad.append("空载楼梯摔倒：未测")
  elif uf > 0:
    bad.append(f"空载楼梯：摔倒 {uf}")
  # loaded stairs: 0 falls (a robot in LOADED_TOP_REPORTED: on 5-15 cm; its 15-18 cm falls reported)
  lf, ltop = loaded_falls(r, f"{prefix}_stairs_pay.txt")
  detail.append(f"负载楼梯：摔倒 {lf}" + ("（精确计数）" if stairs_fall_counts(f"{prefix}_stairs_pay.txt") is not None
                                          else "（各档百分比之和）")
                + (f"（只判 5–15 cm；15–18 cm 满载摔倒 {ltop}，只报告）" if ltop is not None else ""))
  lr = stairs_reach(f"{prefix}_stairs_pay.txt")
  if lr:
    detail.append(f"负载楼梯 各档（只出读数；停住不判）：{lr}")
  if lf is None:
    bad.append("负载楼梯：未测")
  elif lf > 0:
    bad.append(f"负载楼梯：摔倒 {lf}")
  # loaded: the falls in the four comparison-only tests count too (doc/redesign.md 2.5 item 10); their readings do not
  lfalls = comparison_falls({k: f"{prefix}_{k}_pay.txt" for k in at.KINDS})
  detail.append("负载（只出读数的四项）摔倒：" + "，".join(f"{k} {v}" for k, v in lfalls.items()))
  if any(v is None for v in lfalls.values()):
    bad.append("负载对比项：未测")
  elif any(v > 0 for v in lfalls.values()):
    bad.append("负载对比项：摔倒（" + "、".join(k for k, v in lfalls.items() if v) + "）")
  # the typical payload (doc/redesign.md 2.2; robots without one: not run)
  if expect_typ or any(os.path.exists(f"{prefix}_{k}_typ.txt") for k in ("stand", "drift3", "stairs")):
    tb, td = typical(r, prefix, wheeled)
    bad += tb
    detail += td
  else:
    detail.append("常用负载：本机型没有这一档，不测")
  print(verdict_line(bad))
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
