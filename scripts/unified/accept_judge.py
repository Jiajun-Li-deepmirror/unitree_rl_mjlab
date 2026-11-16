#!/usr/bin/env python3
"""accept_judge.py R PREFIX (go2-depth 2026-11-12, doc/redesign.md 2.2): the mjlab acceptance of one candidate, from
jobs/unified/accept.sh's files (one pooled run per test at 3x the robots: the "3 seeds pooled"). First line PASS or
FAIL: <item: reason; ...>, then the details.

Unloaded (PREFIX_<kind>.txt, fixed 0 kg), judged as scripts/blind/acceptance/accept_tables.py's goal_cells judges them
(the same parsers and bars) except: stop-feet against the fixed 0.9 (no deployed comparison), no luck rule, the creep
after a stop at the robot's own bar where it has one (STOP_CREEP_CM: the Go2W 1.5 cm), and the loaded-stand / posture
goals replaced by the zero-command stance below. Items: flat speed, drift, spin, alternating
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
# The zero-command stance's static-stand p90, per robot (2026-11-15, the user: "接受你的建议" -- the first delivery's
# bar 0.17 rad for the Go2 and the Go2W, from their teachers' readings: the Go2's 0.154 / 0.167 rad with / without the
# scan with its weight at the cap and the reading flat, the Go2W's probes 0.151-0.157; no flat fall, the stance after
# an emergency stop 0.085-0.091). Every robot not listed, and the stance after an emergency stop for every robot:
# POSTURE_BAR. Read by every unified judge -- the teacher's flat gate (flat_gate.py) and the students' acceptance
# (judge, typical below); not the blind acceptance, not tinynav's tier (no stance item).
STAND_P90_RAD: dict[str, float] = {"Go2": 0.17, "Go2W": 0.17}


def stand_bar(r: str) -> float:
  return STAND_P90_RAD.get(r, POSTURE_BAR)

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
# 2026-11-16 (the user, ~13:40 answering go2-home's As2 sheet, question 1 (a), relayed by go2-depth 14:50: every robot):
# the full payload's 15-18 cm falls are reported on every robot, judged on 5-15 cm only -- Go2 since 11-14, now Go2W,
# As2, As2W, B2 and B2W too. Unloaded and the typical payload stay judged on every band.
LOADED_TOP_REPORTED: frozenset[str] = frozenset({"Go2", "Go2W", "As2", "As2W", "B2", "B2W"})
LOADED_TOP_BAND = "15-18cm"


def loaded_top_reported(r: str) -> bool:
  return r in LOADED_TOP_REPORTED or r in {x for x in os.environ.get("U_JUDGE_LOADED_TOP_REPORTED", "").split(",") if x}


# The creep in the 4 s after a stop, per robot: the largest group mean in cm (2026-11-15, go2-depth -- the choice among
# relax / accept / another direction for the Go2W's stand the user left to it; doc/redesign.md 2.5 item 16). Every robot
# not listed: accept_tables.STOP_CREEP_BAR, 1.0 cm (doc/acceptance.md). The Go2W 1.5 cm: its stand has two forms by
# the base's tilt (recipe.py's U_REST_OLD=tilt,0.20) -- the one that holds a stop on a 15-18 cm flight costs the flat
# stops a few millimetres (uprobe_s150 / s157: the worst groups, the stops from 1.5 m/s and from an arc, 1.0-1.2 cm
# with the scan and 1.2-1.45 cm with it masked; the form that reads 0.9 cm rolls on down a steep descent after the
# stop command, uprobe_s114: 71 % / 46 % stay). Read by every unified judge -- the teacher's flat gate (flat_gate.py),
# the students' acceptance without a load and with the typical payload (judge, typical below); the loaded creep is
# reported, not judged, as before. Not the blind acceptance (accept_tables' own table is empty), not tinynav's tier
# (it has no creep item).
STOP_CREEP_CM: dict[str, float] = {"Go2W": 1.5}
at.STOP_CREEP_ROBOT = dict(STOP_CREEP_CM)   # at import: every unified judge reads accept_tables through this module


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


# The path on a flight (scripts/unified/descent_probe.py --straight, 72 robots per band and direction, spawned square to
# the steps, a straight command): the user 11-15 on the Go2 teacher's descents, "不是旋转晃动，是左右晃动，像是喝醉了走不了
# 直线", then "需要训练蛇行幅度，但是不用门槛", "Go2W 上楼的蛇行也要算在内", "朝向最大偏离也要训练，门槛你定". The heading
# item: per robot the largest |heading - spawn heading| on the flight, its p90 over the robots of a band and direction
# at most FLIGHT_HEADING_BAR_DEG in every band and direction, unloaded and with the typical payload (the full payload:
# reported); unitree-rl-mjlab-62's bar 8 deg (go2-depth agreed): the Go2's 5-8 cm steps 2.6-5.5 deg and the Go2W's
# descents ~2 deg reach it, the Go2 model_5200's 15-18 cm 9.4-12.0 / the Go2W model_3000's climbs 10-15 deg do not;
# a peak of ~8 deg goes with tinynav's mean bar of 3 deg. Judged from the next version on: the first deliveries (the
# Go2 students of 93a857f, the Go2W of 2c808e7) were not trained for it and report it -- FLIGHT_HEADING_JUDGED lists
# the robots judged (U_JUDGE_FLIGHT_HEADING="Go2,..." adds robots for a test run). The meander (the slow weaving,
# peak-to-peak) and the calf contacts per flight: reported, no bar.
# 2026-11-16 (go2-depth): the Go2 judged from its teacher v2 on (the user's option 甲 by default: U_DESCENT_LEG=6 +
# U_FLIGHT_HEADING=5,0.07 trained; uprobe_s178's avg4 read <= 5.6 deg up and down). The Go2W not: its version of
# 77a29ea was trained without the heading cost (the user's option C by default for problem 1) and reports it.
FLIGHT_HEADING_BAR_DEG = 8.0
FLIGHT_HEADING_JUDGED: frozenset[str] = frozenset({"Go2"})


def flight_heading_judged(r: str) -> bool:
  return r in FLIGHT_HEADING_JUDGED or r in {x for x in os.environ.get("U_JUDGE_FLIGHT_HEADING", "").split(",") if x}


# The meander on a flight judged (2026-11-16, the user via go2-home: the stairs meander becomes a bar, reversing 11-15's
# "需要训练蛇行幅度，但是不用门槛"): descent_probe --straight's meander_pp (the robots' p50 of the lateral weave's peak-to-
# peak on the flight, cm) at most the robot's bar in every band and direction, unloaded and with the typical payload;
# the full payload reported only. Bars per robot: go2-home proposed them from the deployed references (max p-p over
# the bands, unloaded / typical: Go2 v3_fe 7.2 / 8.6, go2_blind_v4 5.7 / 7.6, unified students 6.6-7.2 / 6.4; Go2W
# w27d up 3.9, down 0.7), the user said "可以" (~15:20 CST). Robots not listed report it. A reading within 0.5 cm of
# the bar gets 3 runs, judged pooled (go2-home's measurement rule; not in this judge). U_JUDGE_MEANDER="As2:6,Go2W:4"
# adds or overrides robots for a test run.
MEANDER_BAR_CM: dict[str, float] = {"Go2": 8.0, "Go2W": 5.0}


def meander_bar(r: str) -> float | None:
  extra = {}
  for tok in os.environ.get("U_JUDGE_MEANDER", "").split(","):
    if ":" in tok:
      k, v = tok.split(":", 1)
      extra[k.strip()] = float(v)
  return extra.get(r, MEANDER_BAR_CM.get(r))


def flight_path(path: str) -> dict[tuple[str, str], dict[str, float]]:
  """descent_probe.py's SUM / SUMLAT / SUMQ lines: (band, direction) -> their readings ({} if none)."""
  out: dict[tuple[str, str], dict[str, float]] = {}
  for m in re.finditer(r"^SUM(?:LAT|Q)? (\d+-\d+cm) (up|down) (.*)$", txt(path), re.M):
    toks = m.group(3).split()
    out.setdefault((m.group(1), m.group(2)), {}).update(
      {toks[i]: float(toks[i + 1]) for i in range(0, len(toks) - 1, 2)})
  return out


def flight_path_lines(r: str, path: str, label: str, judged: bool,
                      meander: float | None = None) -> tuple[list[str], list[str]]:
  """(bad, detail) for one condition's descent_probe file: the heading item (judged when `judged`), the meander
  (judged against `meander` cm when given) and the calf contacts (reported)."""
  fp = flight_path(path)
  rows = {k: v for k, v in fp.items() if "hd_max_p90" in v}
  tag = "判" if judged else "只报告，下一版起判"
  if not rows:
    bad = ([f"{label}楼梯朝向最大偏离：未测"] if judged else []) + ([f"{label}楼梯蛇行：未测"] if meander is not None else [])
    return bad, [f"{label}楼梯走直线（descent_probe）：未测"]
  ud = lambda d: "上" if d == "up" else "下"
  over = [f"{b} {ud(d)} {v['hd_max_p90']:.1f}°" for (b, d), v in rows.items() if v["hd_max_p90"] > FLIGHT_HEADING_BAR_DEG]
  mtag = f"判，上限 {meander:g} cm" if meander is not None else "只报告，cm"
  detail = [f"{label}楼梯朝向最大偏离 p90（{tag}，上限 {FLIGHT_HEADING_BAR_DEG:g}°）：" + "，".join(
              f"{b} {ud(d)} {v['hd_max_p90']:.1f}" for (b, d), v in rows.items()),
            f"{label}楼梯蛇行峰峰值 p50（{mtag}）：" + "，".join(
              f"{b} {ud(d)} {v.get('meander_pp', float('nan')):.1f}" for (b, d), v in rows.items()),
            f"{label}楼梯小腿磕台阶（只报告，每段 前 / 后）：" + "，".join(
              f"{b} {ud(d)} {v.get('front_calf', float('nan')):.1f}/{v.get('rear_calf', float('nan')):.1f}"
              for (b, d), v in fp.items())]
  bad = [f"{label}楼梯朝向最大偏离超过 {FLIGHT_HEADING_BAR_DEG:g}°：" + "、".join(over)] if judged and over else []
  if meander is not None:
    missing = [f"{b} {ud(d)}" for (b, d), v in rows.items() if "meander_pp" not in v]
    if missing:
      bad.append(f"{label}楼梯蛇行：未测（" + "、".join(missing) + "）")
    m_over = [f"{b} {ud(d)} {v['meander_pp']:.1f} cm" for (b, d), v in rows.items()
              if "meander_pp" in v and v["meander_pp"] > meander]
    if m_over:
      bad.append(f"{label}楼梯蛇行峰峰值超过 {meander:g} cm：" + "、".join(m_over))
  return bad, detail


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
  detail.append(f"零指令站姿（常用负载）：静站 p90 {sp} rad（上限 {stand_bar(r):.2f}），3 s 后摔 {sl}；急停后 p90 {mp} rad，摔 {mf}")
  if sp is None or mp is None:
    bad.append("零指令站姿（常用负载）：未测")
  else:
    if sp > stand_bar(r) or mp > POSTURE_BAR:
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


# ==== The Go2 / Go2W acceptance specification (2026-11-16, signed by the user; doc/acceptance_go2_go2w.md) ====
# Three conditions with the same items, the values per condition: unloaded (files PREFIX_<test>.txt), the typical
# payload (_typ) and the full payload (_pay, 8 kg at the acceptance point). Concrete numbers only. The full payload's
# stairs (every eval_stairs item, descent_probe's heading and meander) judged on 5-15 cm, 15-18 cm reported. Read by
# the students' acceptance (judge below, for the robots in SPEC_ROBOTS) and by the teacher gates (stairs_gate.py,
# flat_gate.py: the same items and values where they measure them, the teacher's fall tolerance).
SPEC_ROBOTS: frozenset[str] = frozenset({"Go2", "Go2W"})
CONDS = ("unl", "typ", "full")
COND_LABEL = {"unl": "", "typ": "常用负载 ", "full": "满载 "}
COND_SFX = {"unl": "", "typ": "_typ", "full": "_pay"}
FULL_TOP_BAND = "15-18cm"   # the full payload's stairs band that is reported, not judged (the Go2)
# The robots whose full-payload stairs items are judged on every band, 5-18 cm (2026-11-16, the user via go2-home:
# "可以" -- map3's risers are 15-18.6 cm almost everywhere and the full-load map3 reach is judged already): every
# eval_stairs item, descent_probe's heading / meander, items 26 / 27, and the teacher gates' full-load falls
FULL_ALL_BANDS: frozenset[str] = frozenset({"Go2W"})


def full_top(r: str, cond: str) -> str | None:
  """The band reported, not judged, in robot r's condition cond: 15-18 cm under the full payload for a robot not in
  FULL_ALL_BANDS (the Go2), else none."""
  return FULL_TOP_BAND if cond == "full" and r not in FULL_ALL_BANDS else None
SPEC: dict[str, dict] = {
  "flat_speed": {"unl": (0.80, 1.20), "typ": (0.75, None), "full": (0.65, None)},   # share of the command per axis
  "drift": {"unl": 0.05, "typ": 0.05, "full": 0.08},                     # off-axis offset / travel (p10 and p90)
  "drift_head": {c: (3.0, 8.0) for c in CONDS},                            # heading p50 / p10-p90, deg
  "spin_end_p90": {"unl": 0.20, "typ": 0.20, "full": 0.30},               # m
  "spinalt_end_p50": {"unl": 0.25, "typ": 0.25, "full": 0.35},            # m
  "stop_lift": {c: 1.0 for c in CONDS},                                    # % of the rest group's time in the air
  "stop_creep": {"Go2": {"unl": 1.0, "typ": 1.0, "full": 2.0}, "Go2W": {"unl": 1.5, "typ": 1.5, "full": 2.5}},   # cm
  "stop_brake": {"unl": 30.0, "typ": 30.0, "full": 40.0},                 # cm, the stop from 1.0 m/s fwd / back
  "march_td": {c: 0.5 for c in CONDS},                                     # touchdowns / s per foot
  "march_reset": {"unl": 2.0, "typ": 2.0, "full": 3.0},                   # deg, the worst hip's mean after
  "march_drift": {"unl": 5.0, "typ": 5.0, "full": 8.0},                   # cm
  "clock": {"unl": 80.0, "typ": 80.0, "full": 70.0},                      # % match, every judged command
  "stance_stand": {"unl": 0.17, "typ": 0.17, "full": 0.20},               # rad, p90 standing 60 s
  "stance_stop": {"unl": 0.15, "typ": 0.15, "full": 0.18},                # rad, p90 after an emergency stop
  "wheels_down": {"unl": 98.0, "typ": 98.0, "full": 95.0},                # % (wheeled)
  "map3_op": {"unl": 95.0, "typ": 95.0, "full": 90.0},                    # % reached, with the operator
  "map3_noop": {"unl": 90.0, "typ": 90.0, "full": 85.0},                  # % reached, without
  "stairs_speed": {"unl": (0.80, 1.20), "typ": (0.75, None), "full": (0.65, None)},   # share of the command per band
  "stop_held": {"unl": 95.0, "typ": 95.0, "full": 90.0},                  # % within 0.5 m, per band and direction
  "four_feet": {"unl": 0.90, "typ": 0.90, "full": 0.80},                  # per band and direction
  "heading": {"unl": 8.0, "typ": 8.0, "full": 10.0},                      # deg, max p90 per band and direction
  "meander": {"Go2": {"unl": 8.0, "typ": 8.0, "full": 10.0}, "Go2W": {"unl": 5.0, "typ": 5.0, "full": 7.0}},   # cm
  "edge": {"unl": 0.10, "typ": 0.10, "full": 0.12},                       # legged
  "kicks_up": {"unl": 0.35, "typ": 0.35, "full": 0.40},                   # legged, per step on the climbs
  "slips": {"unl": 0.06, "typ": 0.06, "full": 0.07},                      # legged, per step
  "slide_back": {"unl": 0.039, "typ": 0.039, "full": 0.05},               # wheeled
  "mid": {"unl": 0.76, "typ": 0.76, "full": 0.72},                        # legged
  "stalled": {c: 5.0 for c in CONDS},                                      # % per band and direction
  "wall_lift": {c: 5.0 for c in CONDS},                                    # % of the robots on map3's top landings
  "pitch_down": {c: 45.0 for c in CONDS},                                  # deg/s, descents' pitch rate RMS p50
  "roll_up": {c: 45.0 for c in CONDS},                                     # deg/s, climbs' roll rate RMS p50
  "strikes": {c: 2.0 for c in CONDS},                                      # % of a flight's steps, front / rear
  "top_steps": {c: 1.0 for c in CONDS},                                    # per climb, map3's top landings
  "push_falls": {c: 0 for c in CONDS},                                     # falls, 1.0 m/s base pushes (item 22)
  "push_mps": {c: 1.0 for c in CONDS},                                     # the judged push (the user 11-16; was 0.5)
  "slope_reach": {"unl": 95.0, "typ": 95.0, "full": 90.0},                # %, the worst of the 6 cells (item 23)
}
# items 29 and 30 are reported until go2-home fixes their bars from w27d's readings (the proposals below)
SPEC_REPORT = {"bob_height_cm": 1.0, "bob_pitch_deg": 2.0, "speed_cv": 0.3}
SPEC_LONG_TERM = {"kicks_up": 0.245, "slips": 0.0245, "mid": 0.80}   # reported beside the readings


def spec_robot(r: str) -> bool:
  return r in SPEC_ROBOTS


class Sheet:
  """The judged items of one judgment: bad (item strings "<label><name>：<reason>"), detail lines, and the gap (the
  summed relative shortfall, for the teacher's pick at the cap)."""

  def __init__(self) -> None:
    self.bad: list[str] = []
    self.detail: list[str] = []
    self.gap = 0.0

  def fail(self, item: str, gap: float = 1.0) -> None:
    self.bad.append(item)
    self.gap += max(gap, 0.01)

  def missing(self, item: str) -> None:
    self.bad.append(item if "未测" in item else item + "：未测")

  def missing_only(self) -> bool:
    return bool(self.bad) and all(any(w in b for w in MISSING) for b in self.bad)


def _ud(d: str) -> str:
  return "上" if d == "up" else "下"


def _files(prefix: str, cond: str) -> dict[str, str]:
  sfx = COND_SFX[cond]
  return {k: f"{prefix}_{k}{sfx}.txt" for k in at.KINDS + ("loadstand", "stopmove", "descent", "wall", "stairsedge",
                                                              "noopedge", "opedge", "push", "slope")}


def am_rows(path: str) -> list[tuple[str, list[int], int]] | None:
  """accept_matrix's 44 rows: (command, the commanded axes' actual / command in %, falls); None if not all 44."""
  rows = re.findall(r"^\s*(\([^)]*\)) \|\s*(.*?) \|\s*(\d+)/(\d+) \| (?:ok|FAIL)", txt(path), re.M)
  if len(rows) != 44:
    return None
  return [(c, [int(x) for x in re.findall(r"[xyw](-?\d+)%", a)], int(k)) for c, a, k, _n in rows]


def clock_rows(path: str) -> list[tuple[str, float, int, str]] | None:
  """gait_clock's rows: (command, match %, falls, verdict) -- the judged ones carry ok / FAIL; None without rows."""
  rows = re.findall(r"^\s*(\([^)]*\))\s*\|\s*([\d.]+|nan)%\s*\|\s*[\d.na]+\s*\|\s*(\d+)/\d+\s*\|\s*(\S+)", txt(path), re.M)
  if not rows:
    return None
  return [(c, float(m) if m != "nan" else float("nan"), int(f), v) for c, m, f, v in rows]


def spec_flat(r: str, p: dict[str, str], cond: str, sh: Sheet, label: str | None = None, falls_max: int = 0) -> None:
  """Items 1-9 (flat) of one condition's files p (kind -> path) into sh; label: the item prefix (default the
  condition's)."""
  L = COND_LABEL[cond] if label is None else label
  wheeled = r in at.WHEELED
  # 1 flat speed, 44 commands
  lo, hi = SPEC["flat_speed"][cond]
  rows = am_rows(p["am"])
  if rows is None:
    sh.missing(f"{L}平地速度")
  else:
    slow = [(c, v) for c, rs, _k in rows for v in rs if v < 100 * lo]
    fast = [(c, v) for c, rs, _k in rows for v in rs if hi is not None and v > 100 * hi]
    falls = sum(k for _c, _rs, k in rows)
    sh.detail.append(f"{L}平地速度（44 条指令，{'%.0f-%.0f%%' % (100 * lo, 100 * hi) if hi else '>= %.0f%%' % (100 * lo)}）："
                     f"最低 {min((v for _c, rs, _k in rows for v in rs), default=float('nan'))}%，摔 {falls}")
    if slow:
      sh.fail(f"{L}平地速度：偏慢 " + "、".join(f"{c} {v}%" for c, v in slow[:6]), sum(lo - v / 100 for _c, v in slow))
    if fast:
      sh.fail(f"{L}平地速度：超速 " + "、".join(f"{c} {v}%" for c, v in fast[:6]), sum(v / 100 - hi for _c, v in fast))
    if falls > falls_max:
      sh.fail(f"{L}平地速度：摔倒 {falls}", 1.0)
  # 2 drift and heading; 3 spin in place (drift_all's spins)
  d = at.drift_vals(p)
  bar = SPEC["drift"][cond]
  h50, h90 = SPEC["drift_head"][cond]
  bar_spin = SPEC["spin_end_p90"][cond]
  if not d:
    sh.missing(f"{L}跑偏")
    sh.missing(f"{L}原地旋转")
  else:
    drift_bad, head_bad, worst = [], [], 0.0
    for c, v in d.items():
      if c[2] != 0.0:
        continue
      sideways = c[1] != 0.0
      travel = abs(v["across"][1]) if sideways else abs(v["along"][1])
      off = v["along"] if sideways else v["across"]
      share = max(abs(off[0]), abs(off[2])) / travel if travel > 0 else float("nan")
      worst = max(worst, share) if share == share else worst
      if travel > 0 and share > bar:
        drift_bad.append(f"{c} {100 * share:.1f}%")
      if abs(v["head"][1]) > h50 or max(abs(v["head"][0]), abs(v["head"][2])) > h90:
        head_bad.append(f"{c} {v['head'][1]:+.1f}°（{v['head'][0]:+.1f}~{v['head'][2]:+.1f}）")
    sh.detail.append(f"{L}跑偏：最大 {100 * worst:.1f}%（上限 {100 * bar:g}%），朝向 p50 <= {h50:g}°、p10 / p90 <= {h90:g}°")
    if drift_bad:
      sh.fail(f"{L}跑偏：超过 {100 * bar:g}% " + "、".join(drift_bad), worst / bar - 1.0)
    if head_bad:
      sh.fail(f"{L}跑偏：转头 " + "、".join(head_bad))
    dfalls = sum(v["fell"] for v in d.values())
    if dfalls > falls_max:
      sh.fail(f"{L}跑偏：摔倒 {dfalls}")
    spins = {c: v for c, v in d.items() if c[0] == c[1] == 0.0 and c[2] != 0.0}
    if not spins:
      sh.missing(f"{L}原地旋转")
    else:
      far = [(c, v["end"][-1]) for c, v in spins.items() if v["end"][-1] > bar_spin]
      sh.detail.append(f"{L}原地旋转：终点 p90 最大 {max(v['end'][-1] for v in spins.values()):.2f} m（上限 {bar_spin:g}）")
      if far:
        sh.fail(f"{L}原地旋转：漂移 " + "、".join(f"{c} {x:.2f} m" for c, x in far), max(x for _c, x in far) / bar_spin - 1.0)
  # 4 alternating spins
  sa = at.spinalt_vals(p)
  bar = SPEC["spinalt_end_p50"][cond]
  if not sa:
    sh.missing(f"{L}交替原地转")
  else:
    far = [(k, v["end"][0]) for k, v in sa.items() if v["end"][0] > bar]
    sh.detail.append(f"{L}交替原地转：终点 p50 最大 {max(v['end'][0] for v in sa.values()):.2f} m（上限 {bar:g}）")
    if far:
      sh.fail(f"{L}交替原地转：漂移 " + "、".join(f"±{k[0]:g} {x:.2f} m" for k, x in far), max(x for _k, x in far) / bar - 1.0)
    sf = sum(v["fell"] for v in sa.values())
    if sf > falls_max:
      sh.fail(f"{L}交替原地转：摔倒 {sf}")
  # 5 stop: lift, creep, brake from 1.0 m/s
  g = at.stand_vals(p)
  if "rest" not in g:
    sh.missing(f"{L}停车")
  else:
    lift, creep = SPEC["stop_lift"][cond], SPEC["stop_creep"].get(r, SPEC["stop_creep"]["Go2"])[cond]
    brake = SPEC["stop_brake"][cond]
    worst_creep = max(v["creep"] for v in g.values())
    brakes = {k: g[k]["stop_cm"] for k in ("fwd 1.0", "back 1.0") if k in g}
    sh.detail.append(f"{L}停车：离地 {g['rest']['air']:.1f}%（上限 {lift:g}），蠕动最大 {worst_creep:.2f} cm（上限 {creep:g}），"
                     f"1.0 m/s 刹停 " + "、".join(f"{k} {v:.1f} cm" for k, v in brakes.items()) + f"（上限 {brake:g}）")
    if g["rest"]["air"] > lift:
      sh.fail(f"{L}停车：离地 {g['rest']['air']:.1f}%", g["rest"]["air"] / lift - 1.0)
    if worst_creep > creep:
      sh.fail(f"{L}停车：蠕动 {worst_creep:.2f} cm", worst_creep / creep - 1.0)
    over = {k: v for k, v in brakes.items() if v > brake}
    if over:
      sh.fail(f"{L}停车：刹停远 " + "、".join(f"{k} {v:.1f} cm" for k, v in over.items()), max(over.values()) / brake - 1.0)
    if len(brakes) < 2:
      sh.missing(f"{L}停车 1.0 m/s 刹停")
    gf = sum(v["fell"] for v in g.values())
    if gf > falls_max:
      sh.fail(f"{L}停车：摔倒 {gf}")
  # 6 march: touchdowns, hip reset, drift
  rv = at.reset_vals(p)
  if rv is None:
    sh.missing(f"{L}原地踏步")
  else:
    td, rs, dr = SPEC["march_td"][cond], SPEC["march_reset"][cond], SPEC["march_drift"][cond]
    sh.detail.append(f"{L}原地踏步：踏步 {rv['td']:.2f}/s（>= {td:g}），复位后 {rv['after']:.1f}°（<= {rs:g}），漂移 {rv['drift']:.1f} cm（<= {dr:g}）")
    if rv["td"] < td:
      sh.fail(f"{L}原地踏步：不踏步 {rv['td']:.2f}/s", 1.0 - rv["td"] / td)
    if rv["after"] > rs:
      sh.fail(f"{L}原地踏步：没复位 {rv['after']:.1f}°", rv["after"] / rs - 1.0)
    if rv["drift"] > dr:
      sh.fail(f"{L}原地踏步：漂移 {rv['drift']:.1f} cm", rv["drift"] / dr - 1.0)
    if rv["fell"] > falls_max:
      sh.fail(f"{L}原地踏步：摔倒 {rv['fell']}")
  # 7 gait clock: every judged command at least the condition's match
  cr = clock_rows(p["clock"])
  bar = SPEC["clock"][cond]
  if cr is None:
    sh.missing(f"{L}步态节奏")
  else:
    judged = [(c, m, f) for c, m, f, v in cr if v in ("ok", "FAIL")]
    low = [(c, m) for c, m, _f in judged if not m >= bar]
    cf = sum(f for _c, _m, f, _v in cr)
    sh.detail.append(f"{L}步态节奏：判 {len(judged)} 条，最低 {min((m for _c, m, _f in judged), default=float('nan')):.0f}%（>= {bar:g}%），摔 {cf}")
    if not judged:
      sh.missing(f"{L}步态节奏")
    if low:
      sh.fail(f"{L}步态节奏：不合拍 " + "、".join(f"{c} {m:.0f}%" for c, m in low[:6]),
              sum(1.0 - (m / bar if m == m else 0.0) for _c, m in low))
    if cf > falls_max:
      sh.fail(f"{L}步态节奏：摔倒 {cf}")
  # 8 zero-command stance: standing 60 s, after an emergency stop
  sp, sl = stand_posture(p["loadstand"])
  mp, mf = stopmove(p["stopmove"])
  b1, b2 = SPEC["stance_stand"][cond], SPEC["stance_stop"][cond]
  sh.detail.append(f"{L}零指令站姿：静站 p90 {sp} rad（<= {b1:g}），3 s 后摔 {sl}；急停后 p90 {mp} rad（<= {b2:g}），摔 {mf}")
  if sp is None or mp is None:
    sh.missing(f"{L}零指令站姿")
  else:
    if sp > b1:
      sh.fail(f"{L}零指令站姿：静站 {sp:.3f} rad", sp / b1 - 1.0)
    if mp > b2:
      sh.fail(f"{L}零指令站姿：急停后 {mp:.3f} rad", mp / b2 - 1.0)
    if (sl or 0) + (mf or 0) > falls_max:
      sh.fail(f"{L}零指令站姿：摔倒 {(sl or 0) + (mf or 0)}")
  # 9 wheels down (wheeled)
  if wheeled:
    n = at.nolift_vals(p)
    bar = SPEC["wheels_down"][cond]
    if n is None:
      sh.missing(f"{L}四轮着地")
    else:
      sh.detail.append(f"{L}四轮着地：最低 {min(n['down']):.1f}%（>= {bar:g}），摔 {n['fell']}")
      if min(n["down"]) < bar:
        sh.fail(f"{L}四轮着地：抬轮 {min(n['down']):.1f}%", 1.0 - min(n["down"]) / bar)
      if n["fell"] > falls_max:
        sh.fail(f"{L}四轮着地：摔倒 {n['fell']}")


def tile_counts(path: str, skip_band: str | None = None) -> dict[str, float | None]:
  """eval_stairs --metrics' tile rows pooled, without skip_band's tiles: legged steps / kicks on the climbs / slips /
  mid / edge (step-weighted), wheeled step-ups / slide-back (step-up-weighted). None for a count with no tile."""
  t = txt(path)
  legged = re.findall(r"^tile (\S+) (up|down) \S+: steps ([\d.]+), kicks ([\d.]+|-), slips ([\d.]+|-), mid ([\d.]+|-), "
                      r"edge ([\d.]+|-)", t, re.M)
  wheeled = re.findall(r"^tile (\S+) (up|down) \S+: step-ups ([\d.]+), slide-back ([\d.]+|-)", t, re.M)
  out: dict[str, float | None] = {}
  if legged:
    acc = {k: [0.0, 0.0] for k in ("kicks_up", "slips", "mid", "edge")}
    steps = 0.0
    for b, d, n, k, s, m, e in legged:
      if b == skip_band:
        continue
      n = float(n)
      steps += n
      for key, v, use in (("kicks_up", k, d == "up"), ("slips", s, True), ("mid", m, True), ("edge", e, True)):
        if use and v != "-":
          acc[key][0] += float(v) * n
          acc[key][1] += n
    out["steps"] = steps
    out.update({k: (a / w if w else None) for k, (a, w) in acc.items()})
  if wheeled:
    ups, sb = 0.0, [0.0, 0.0]
    for b, d, n, s in wheeled:
      if b == skip_band:
        continue
      ups += float(n)
      if s != "-":
        sb[0] += float(s) * float(n)
        sb[1] += float(n)
    out["ups"] = ups
    out["slide_back"] = sb[0] / sb[1] if sb[1] else None
  return out


def spec_stairs(r: str, path: str, cond: str, sh: Sheet, label: str | None = None, falls_max: int = 0) -> None:
  """Items 11-13 and 16-20 and the falls of one eval_stairs output into sh. The full payload: judged on 5-15 cm,
  15-18 cm reported."""
  L = COND_LABEL[cond] if label is None else label
  wheeled = r in at.WHEELED
  t = txt(path)
  if not t:
    sh.missing(f"{L}楼梯")
    return
  top = full_top(r, cond)
  judged = lambda b: b != top
  rep = (f"（只判 5–15 cm，{top} 只报告）" if top else "")
  # 11 speed per band and direction, share of the command
  lo, hi = SPEC["stairs_speed"][cond]
  sp = [(m.group(1), m.group(2), float(m.group(3)) if m.group(3) != "nan" else float("nan"))
        for m in re.finditer(r"^speed\s+(\d+-\d+cm)\s+(up|down) \| (?:[\d.]+|nan) m/s x([\d.]+|nan) of the command", t, re.M)]
  if not sp:
    sh.missing(f"{L}楼梯速度")
  else:
    off = [(b, d, v) for b, d, v in sp if judged(b) and not (v >= lo and (hi is None or v <= hi))]
    sh.detail.append(f"{L}楼梯速度（命令的倍数，{f'{lo:g}-{hi:g}' if hi else f'>= {lo:g}'}）{rep}：" +
                     "，".join(f"{b} {_ud(d)} {v:.2f}" for b, d, v in sp))
    if off:
      sh.fail(f"{L}楼梯速度：" + "、".join(f"{b} {_ud(d)} x{v:.2f}" for b, d, v in off),
              sum(1.0 if v != v else (lo - v) if v < lo else (v - hi) for _b, _d, v in off))
  # 12 stop held within 0.5 m, per band and direction
  bar = SPEC["stop_held"][cond]
  ss = [(m.group(1), m.group(2), float(m.group(3)), int(m.group(4))) for m in re.finditer(
    r"^stop-stay\s+(\d+-\d+cm)\s+(up|down) \| stayed (\d+)% \(within [\d.]+ m\).*?n (\d+)", t, re.M)]
  if not ss:
    sh.missing(f"{L}楼梯停住")
  else:
    away = [(b, d, v) for b, d, v, n in ss if judged(b) and n > 0 and v < bar]
    sh.detail.append(f"{L}楼梯停住（0.5 m 内，>= {bar:g}%）{rep}：" + "，".join(f"{b} {_ud(d)} {v:.0f}%" for b, d, v, _n in ss))
    if away:
      sh.fail(f"{L}楼梯停住：" + "、".join(f"{b} {_ud(d)} {v:.0f}%" for b, d, v in away),
              sum((bar - v) / bar for _b, _d, v in away))
  # 13 four feet at the stop, per band and direction
  bar = SPEC["four_feet"][cond]
  ft = [(m.group(1), m.group(2), float(m.group(3))) for m in re.finditer(
    r"^\s*(\d+-\d+cm)\s+(up|down)\s*\|.*\|\s*\d+%\s+\d+%\s+[\d.]+\s+([\d.]+)\s*$", t, re.M)]
  if not ft:
    sh.missing(f"{L}停住四脚")
  else:
    short = [(b, d, v) for b, d, v in ft if judged(b) and v < bar]
    sh.detail.append(f"{L}停住四脚（>= {bar:g}）{rep}：" + "，".join(f"{b} {_ud(d)} {v:.2f}" for b, d, v in ft))
    if short:
      sh.fail(f"{L}停住缺脚：" + "、".join(f"{b} {_ud(d)} {v:.2f}" for b, d, v in short), sum(bar - v for _b, _d, v in short))
  # 16-19 the counts (judged on enough events)
  tc = tile_counts(path, top)
  if wheeled:
    n, v, bar = tc.get("ups"), tc.get("slide_back"), SPEC["slide_back"][cond]
    sh.detail.append(f"{L}楼梯计数{rep}：上台阶 {n}，滑回 {v}（<= {bar:g}）")
    if n is None or v is None:
      sh.missing(f"{L}楼梯计数")
    elif n < MIN_STAIR_EVENTS:
      sh.missing(f"{L}楼梯计数：未测（上台阶 {n:.0f} 次，少于 {MIN_STAIR_EVENTS}）")
    elif v > bar:
      sh.fail(f"{L}楼梯计数：滑回 {v:.3f}", v / bar - 1.0)
  else:
    n = tc.get("steps")
    sh.detail.append(f"{L}楼梯计数{rep}：落脚 {n}，" + "，".join(
      f"{k} {'-' if tc.get(k) is None else f'{tc[k]:.3f}'}" for k in ("edge", "kicks_up", "slips", "mid")) +
      "；长期目标（只报告）" + "，".join(f"{k} {v}" for k, v in SPEC_LONG_TERM.items()))
    if n is None:
      sh.missing(f"{L}楼梯计数")
    elif n < MIN_STAIR_EVENTS:
      sh.missing(f"{L}楼梯计数：未测（落脚 {n:.0f} 次，少于 {MIN_STAIR_EVENTS}）")
    else:
      for k, name, sense in (("edge", "踩边", "max"), ("kicks_up", "上楼踢立面", "max"), ("slips", "打滑", "max"),
                             ("mid", "踩中间", "min")):
        v, bar = tc.get(k), SPEC[k][cond]
        if v is None:
          sh.missing(f"{L}楼梯计数 {name}")
        elif (sense == "max" and v > bar) or (sense == "min" and v < bar):
          sh.fail(f"{L}楼梯计数：{name} {v:.3f}", (v / bar - 1.0) if sense == "max" else (bar / max(v, 1e-6) - 1.0))
  # 20 stalled, per band and direction
  st = stairs_stalled(path)
  bar = SPEC["stalled"][cond]
  if not st:
    sh.missing(f"{L}楼梯停滞")
  else:
    over = [(b, d, v) for (b, d), v in st.items() if judged(b) and v > bar]
    if over:
      sh.fail(f"{L}楼梯停滞：" + "、".join(f"{b} {_ud(d)} {v:g}%" for b, d, v in over), sum((v - bar) / 100 for _b, _d, v in over))
  # falls: every band (the full payload of a robot not in FULL_ALL_BANDS: 5-15 cm, 15-18 cm reported)
  if top:
    total = stairs_falls(path)
    bands = band_fall_counts(path)
    if total is None:
      sh.missing(f"{L}楼梯摔倒")
      return
    topf = sum(v for (b, _d), v in bands.items() if b == FULL_TOP_BAND) if bands else None
    falls = total - topf if topf is not None else total
    sh.detail.append(f"{L}楼梯摔倒：{falls}（5–15 cm" + (f"；{FULL_TOP_BAND} {topf}，只报告）" if topf is not None else "；分档读不出，全部计入）"))
  else:
    falls = stairs_falls(path)
    if falls is None:
      sh.missing(f"{L}楼梯摔倒")
      return
    sh.detail.append(f"{L}楼梯摔倒：{falls}")
  if falls > falls_max:
    sh.fail(f"{L}楼梯：摔倒 {falls}", 1.0 + 0.1 * (falls - falls_max))


def spec_map3(r: str, p: dict[str, str], cond: str, sh: Sheet, label: str | None = None) -> None:
  """Item 10: map3 reach with / without the operator; map3's speed reported."""
  L = COND_LABEL[cond] if label is None else label
  s = at.stairs_vals(p)
  if s is None:
    sh.missing(f"{L}楼梯到达（map3）")
    return
  bo, bn = SPEC["map3_op"][cond], SPEC["map3_noop"][cond]
  sh.detail.append(f"{L}楼梯到达（map3）：带操作 {s['op']:.1f}%（>= {bo:g}），无操作 {s['no']:.1f}%（>= {bn:g}）；"
                   f"速度（只报告）上 {s['sp_up']:.2f} 下 {s['sp_dn']:.2f}")
  if s["op"] < bo - 1e-6:   # (a mean of 18 scenarios: 0.95 x 18 / 18 reads 94.999...)
    sh.fail(f"{L}楼梯到达（map3）：带操作 {s['op']:.1f}%", (bo - s["op"]) / bo)
  if s["no"] < bn - 1e-6:
    sh.fail(f"{L}楼梯到达（map3）：无操作 {s['no']:.1f}%", (bn - s["no"]) / bn)


def wall_lift(path: str) -> tuple[int, int] | None:
  """map3_stairs_probe.py's item-21 line: (robots over the lift, robots on the landing); None without the line."""
  m = re.search(r"^ALL wall lift \(item 21\).*?: (\d+) of (\d+) robots on the landing", txt(path), re.M)
  return (int(m.group(1)), int(m.group(2))) if m else None


def spec_wall(r: str, path: str, cond: str, sh: Sheet) -> None:
  """Item 21 (2026-11-16, the user via go2-home: "面前是墙不抬腿"): on map3's top landings facing the end wall
  (map3_stairs_probe.py: stand, turn in place, stand), the share of the robots on the landing whose front foot / wheel
  rose more than 10 cm above the first stand, in the turn or the last stand: <= 5 % in every condition. No robot on a
  landing: not judgeable here -- a fail (the climbs' items say why)."""
  L = COND_LABEL[cond]
  v = wall_lift(path)
  if v is None:
    sh.missing(f"{L}面前是墙不抬腿（map3_stairs_probe）")
    return
  k, n = v
  bar = SPEC["wall_lift"][cond]
  if n == 0:
    sh.fail(f"{L}面前是墙不抬腿：没有机器人到达顶层平台")
    return
  share = 100.0 * k / n
  sh.detail.append(f"{L}面前是墙不抬腿：前脚抬高超过 10 cm {k}/{n} = {share:.1f}%（<= {bar:g}%）")
  if share > bar + 1e-9:
    sh.fail(f"{L}面前是墙不抬腿：{share:.1f}%", share / bar - 1.0)


def top_steps(path: str) -> tuple[float, float, float, int] | None:
  """map3_stairs_probe.py's item-28 line: (per climb, front, rear, climbs); None without the line."""
  m = re.search(r"^ALL top-landing steps \(item 28\): \d+ steps .*? over (\d+) climbs = ([\d.]+|nan) per climb "
                r"\(front ([\d.]+|nan), rear ([\d.]+|nan)", txt(path), re.M)
  return (float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(1))) if m else None


def bob_rows(path: str) -> dict[str, tuple[float, float, float, float]]:
  """drift_all.py's item-29 rows: command -> (height std cm p50, p90, pitch std deg p50, p90)."""
  out = {}
  for m in re.finditer(r"^BOB (\([^)]*\)) \| height std cm p50 (\S+) p90 (\S+) \| pitch std deg p50 (\S+) p90 (\S+)",
                       txt(path), re.M):
    out[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
  return out


def push_vals(path: str) -> dict[str, float] | None:
  """push_test.py's item-22 line: falls, robots, recovery p50 / p90 (s), not recovered (%); None without it."""
  m = re.search(r"^ALL push \(item 22\): falls (\d+) of (\d+) robots; recovered within [\d.]+ m/s: p50 (\S+) s, "
                r"p90 (\S+) s, not recovered (\S+) %", txt(path), re.M)
  if not m:
    return None
  return {"falls": int(m.group(1)), "n": int(m.group(2)), "r50": float(m.group(3)), "r90": float(m.group(4)),
          "nr": float(m.group(5))}


def spec_push(r: str, path: str, cond: str, sh: Sheet) -> None:
  """Item 22 (2026-11-16, the user via go2-home): one base push of SPEC['push_mps'] (1.0 m/s; the user first approved
  0.5, then picked 1.0) in 4 directions, standing, 1.0 m/s forward and 0.5 m/s sideways (push_test.py --push): no fall
  in any condition; the recovery (the base back within 0.3 m/s of the command for 0.5 s) reported; the other push
  sizes of the run (0.5) reported."""
  L = COND_LABEL[cond]
  v = push_vals(path)
  if v is None:
    sh.missing(f"{L}抗推（push_test）")
    return
  sh.detail.append(f"{L}抗推 {SPEC['push_mps'][cond]:g} m/s：摔倒 {v['falls']}/{v['n']}（= {SPEC['push_falls'][cond]}）；恢复到指令 0.3 m/s 内 p50 "
                   f"{v['r50']:.2f} s、p90 {v['r90']:.2f} s，未恢复 {v['nr']:.1f}%（只报告）")
  for m in re.finditer(r"^ALL push ([\d.]+) \(item 22, reported\): falls (\d+) of (\d+) robots; recovered within [\d.]+ "
                       r"m/s: p50 (\S+) s, p90 (\S+) s, not recovered (\S+) %", txt(path), re.M):
    sh.detail.append(f"{L}抗推 {m.group(1)} m/s（只报告，不判）：摔倒 {m.group(2)}/{m.group(3)}，恢复 p50 {float(m.group(4)):.2f} s、"
                     f"p90 {float(m.group(5)):.2f} s，未恢复 {float(m.group(6)):.1f}%")
  if v["falls"] > SPEC["push_falls"][cond]:
    sh.fail(f"{L}抗推：摔倒 {v['falls']}")


def slope_vals(path: str) -> tuple[float, int, int, list[str]] | None:
  """slope_rough_test.py: (the worst cell's reach %, falls, robots, the cell rows); None without its ALL line."""
  t = txt(path)
  m = re.search(r"^ALL slopes \(item 23\): reach min ([\d.]+|nan) % over \d+ cells, falls (\d+) of (\d+) robots", t, re.M)
  if not m:
    return None
  rows = re.findall(r"^ (\S+ ?\S*)\s+([\d.]+) \| reached\s+([\d.]+) % \| falls (\d+)/\d+", t, re.M)
  return float(m.group(1)), int(m.group(2)), int(m.group(3)), [f"{a.strip()} {b} {float(c):.1f}%" for a, b, c, _f in rows]


def spec_slope(r: str, path: str, cond: str, sh: Sheet) -> None:
  """Item 23 (2026-11-16, the user via go2-home): a 15 deg ramp up and down and rough ground +-3 cm, at 0.5 and 1.0 m/s
  (slope_rough_test.py): every cell's reach >= 95 % (unloaded, typical) / 90 % (full), 0 falls."""
  L = COND_LABEL[cond]
  v = slope_vals(path)
  if v is None:
    sh.missing(f"{L}斜坡和不平地面（slope_rough_test）")
    return
  worst, falls, n, rows = v
  bar = SPEC["slope_reach"][cond]
  sh.detail.append(f"{L}斜坡和不平地面：到达最低 {worst:.1f}%（>= {bar:g}%），摔倒 {falls}/{n}；" + "，".join(rows))
  if worst < bar - 1e-9:
    sh.fail(f"{L}斜坡和不平地面：到达 {worst:.1f}%", (bar - worst) / bar)
  if falls:
    sh.fail(f"{L}斜坡和不平地面：摔倒 {falls}")


def spec_motion(r: str, p: dict[str, str], cond: str, sh: Sheet) -> None:
  """Items 26-30 (2026-11-16, the user via go2-home: "都加"). From descent_probe's SUM / SUMQ lines (the descent file,
  already run): 26 the descents' pitch rate and the climbs' roll rate (RMS per robot, p50) per band; 27 the share of
  the flight's steps with a front / rear calf or thigh on the steps; 30 the forward speed's coefficient of variation
  (reported). From map3_stairs_probe (the wall file): 28 the top-landing steps / wheel lifts per climb. From drift_all
  (drift3): 29 the base height / pitch std over the steady cruise (reported). The full payload: 26 and 27 judged on
  5-15 cm, 15-18 cm reported."""
  L = COND_LABEL[cond]
  top = full_top(r, cond)
  rep_ = f"（只判 5–15 cm，{top} 只报告）" if top else ""
  fp = flight_path(p["descent"])
  rows = {k: v for k, v in fp.items() if "pitch_rate" in v and "roll_rate" in v}
  if not rows:
    sh.missing(f"{L}楼梯机身角速度（descent_probe）")
  else:
    pb, rb = SPEC["pitch_down"][cond], SPEC["roll_up"][cond]
    sh.detail.append(f"{L}下楼俯仰角速度 RMS p50（<= {pb:g}°/s）/ 上楼横滚角速度 RMS p50（<= {rb:g}°/s）{rep_}：" + "，".join(
      f"{b} {_ud(d)} {v['pitch_rate'] if d == 'down' else v['roll_rate']:.1f}" for (b, d), v in rows.items()))
    over = [(b, d, v["pitch_rate"] if d == "down" else v["roll_rate"]) for (b, d), v in rows.items() if b != top]
    od = [(b, x) for b, d, x in over if d == "down" and x > pb]
    ou = [(b, x) for b, d, x in over if d == "up" and x > rb]
    if od:
      sh.fail(f"{L}下楼点头：" + "、".join(f"{b} {x:.1f}°/s" for b, x in od), sum(x / pb - 1.0 for _b, x in od))
    if ou:
      sh.fail(f"{L}上楼晃：" + "、".join(f"{b} {x:.1f}°/s" for b, x in ou), sum(x / rb - 1.0 for _b, x in ou))
  q = {k: v for k, v in fp.items() if "front_touch" in v}
  if not q:
    sh.missing(f"{L}小腿大腿磕台阶（descent_probe SUMQ）")
  else:
    sb = SPEC["strikes"][cond]
    sh.detail.append(f"{L}小腿大腿磕台阶，飞段步数占比 前 / 后（<= {sb:g}%）{rep_}：" + "，".join(
      f"{b} {_ud(d)} {v['front_touch']:.2f}/{v['rear_touch']:.2f}" for (b, d), v in q.items()))
    so = [(b, d, s, v[f"{s}_touch"]) for (b, d), v in q.items() if b != top for s in ("front", "rear") if v[f"{s}_touch"] > sb]
    if so:
      sh.fail(f"{L}小腿大腿磕台阶：" + "、".join(f"{b} {_ud(d)} {'前' if s == 'front' else '后'} {x:.2f}%" for b, d, s, x in so),
              sum(x / sb - 1.0 for _b, _d, _s, x in so))
    cvb = SPEC_REPORT["speed_cv"]
    sh.detail.append(f"{L}楼梯走走停停（只报告，提议 <= {cvb:g}）：速度变异系数 p50 " + "，".join(
      f"{b} {_ud(d)} {v.get('speed_cv_p50', float('nan')):.2f}" for (b, d), v in q.items()))
  ts = top_steps(p["wall"])
  if ts is None:
    sh.missing(f"{L}顶部平台多余踏步（map3_stairs_probe）")
  else:
    per, fr, rr, nc = ts
    tb = SPEC["top_steps"][cond]
    if nc == 0:
      sh.fail(f"{L}顶部平台多余踏步：没有机器人到达顶层平台")
    else:
      sh.detail.append(f"{L}顶部平台多余踏步：每次爬楼 {per:.2f}（前 {fr:.2f}，后 {rr:.2f}；<= {tb:g}；{nc} 次）")
      if per > tb + 1e-9:
        sh.fail(f"{L}顶部平台多余踏步：每次 {per:.2f}", per / tb - 1.0)
  bob = bob_rows(p["drift3"])
  if not bob:
    sh.missing(f"{L}平地颠晃（drift_all BOB）")
  else:
    hb, pb2 = SPEC_REPORT["bob_height_cm"], SPEC_REPORT["bob_pitch_deg"]
    sh.detail.append(f"{L}平地颠晃（只报告，提议高度 <= {hb:g} cm、俯仰 <= {pb2:g}°；稳态段 std p50）：" + "，".join(
      f"{c} {v[0]:.2f} cm / {v[2]:.2f}°" for c, v in bob.items()))


def spec_flight(r: str, path: str, cond: str, sh: Sheet) -> None:
  """Items 14 and 15: the heading's max p90 and the meander p-p p50 per band and direction (descent_probe
  --straight), and its falls. The full payload: 5-15 cm, 15-18 cm reported."""
  L = COND_LABEL[cond]
  fp = flight_path(path)
  rows = {k: v for k, v in fp.items() if "hd_max_p90" in v}
  if not rows:
    sh.missing(f"{L}楼梯走直线（descent_probe）")
    return
  top = full_top(r, cond)
  hb, mb = SPEC["heading"][cond], SPEC["meander"].get(r, SPEC["meander"]["Go2"])[cond]
  rep = (f"（只判 5–15 cm，{top} 只报告）" if top else "")
  sh.detail.append(f"{L}楼梯朝向最大偏离 p90（<= {hb:g}°）{rep}：" + "，".join(f"{b} {_ud(d)} {v['hd_max_p90']:.1f}" for (b, d), v in rows.items()))
  sh.detail.append(f"{L}楼梯蛇行峰峰值 p50（<= {mb:g} cm）{rep}：" + "，".join(
    f"{b} {_ud(d)} {v.get('meander_pp', float('nan')):.1f}" for (b, d), v in rows.items()))
  sh.detail.append(f"{L}楼梯小腿磕台阶（只报告，每段 前 / 后）：" + "，".join(
    f"{b} {_ud(d)} {v.get('front_calf', float('nan')):.1f}/{v.get('rear_calf', float('nan')):.1f}" for (b, d), v in fp.items()))
  hj = {k: v for k, v in rows.items() if k[0] != top}
  over = [(b, d, v["hd_max_p90"]) for (b, d), v in hj.items() if v["hd_max_p90"] > hb]
  if over:
    sh.fail(f"{L}楼梯朝向最大偏离超过 {hb:g}°：" + "、".join(f"{b} {_ud(d)} {x:.1f}°" for b, d, x in over),
            sum(x / hb - 1.0 for _b, _d, x in over))
  nom = [f"{b} {_ud(d)}" for (b, d), v in hj.items() if "meander_pp" not in v]
  if nom:
    sh.missing(f"{L}楼梯蛇行：未测（" + "、".join(nom) + "）")
  mo = [(b, d, v["meander_pp"]) for (b, d), v in hj.items() if "meander_pp" in v and v["meander_pp"] > mb]
  if mo:
    sh.fail(f"{L}楼梯蛇行峰峰值超过 {mb:g} cm：" + "、".join(f"{b} {_ud(d)} {x:.1f} cm" for b, d, x in mo),
            sum(x / mb - 1.0 for _b, _d, x in mo))
  ff = sum(int(v.get("falls", 0)) for (b, _d), v in fp.items() if b != top and "falls" in v)
  if ff:
    sh.fail(f"{L}楼梯走直线：摔倒 {ff}")


EDGE_LABEL = "相机误差 "   # item 24's label after the condition's


def spec_judge(r: str, prefix: str, expect_typ: bool, expect_edge: bool = False) -> list[str]:
  """The students' acceptance of a robot in SPEC_ROBOTS: the verdict line and the detail lines. expect_edge (the
  depth student, accept.sh --edge): item 24, the stairs and map3 items again on the -RealEdge task (the real camera's
  errors and edge smoothing), at the same bars; its files are then required (missing: 未测), else read when present."""
  sh = Sheet()
  conds = ["unl"] + (["typ"] if expect_typ or os.path.exists(f"{prefix}_stairs_typ.txt") else []) + ["full"]
  for cond in conds:
    p = _files(prefix, cond)
    spec_flat(r, p, cond, sh)
    spec_map3(r, p, cond, sh)
    spec_stairs(r, p["stairs"], cond, sh)
    spec_flight(r, p["descent"], cond, sh)
    spec_wall(r, p["wall"], cond, sh)
    spec_motion(r, p, cond, sh)
    spec_push(r, p["push"], cond, sh)
    spec_slope(r, p["slope"], cond, sh)
    if expect_edge or any(os.path.exists(p[k]) for k in ("stairsedge", "noopedge", "opedge")):
      E = COND_LABEL[cond] + EDGE_LABEL
      spec_stairs(r, p["stairsedge"], cond, sh, label=E)
      spec_map3(r, {"noop": p["noopedge"], "op": p["opedge"]}, cond, sh, label=E)
  if "typ" not in conds:
    sh.detail.append("常用负载：本机型没有这一档，不测")
  return [verdict_line(sh.bad)] + sh.detail


def main(argv: list[str]) -> int:
  """accept_judge.py R PREFIX [--typical] [--edge]: --typical (accept.sh, when the robot has a typical payload and
  the condition ran) makes its files required (missing: 未测, a tool fault), instead of looked for; --edge (accept.sh,
  the depth student) the same for item 24's files."""
  try:
    return judge(argv)
  except Exception as e:   # a malformed line or file: what the judge cannot read is a tool fault, not a fail
    import traceback
    print(f"FAULT: judge error: {type(e).__name__}: {e}")
    traceback.print_exc(file=sys.stdout)
    return 0


def judge(argv: list[str]) -> int:
  expect_typ = "--typical" in argv
  expect_edge = "--edge" in argv
  argv = [a for a in argv if a not in ("--typical", "--edge")]
  r, prefix = argv[0], argv[1]
  wheeled = r in at.WHEELED
  at.STOP_FEET_DEPLOYED = {}   # the fixed bar (doc/redesign.md 2.2)
  if spec_robot(r) and os.environ.get("U_JUDGE_OLD") != "1":   # (U_JUDGE_OLD=1: the judge before the spec, a comparison)
    lines = spec_judge(r, prefix, expect_typ, expect_edge)
    print(lines[0])
    for line in lines[1:]:
      print("  " + line)
    _tail(prefix)
    return 0
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
      if sp > stand_bar(r) or mp > POSTURE_BAR:
        tags.append("偏离站姿")
      if (sl or 0) > 0 or (mf or 0) > 0:
        tags.append("摔倒")
    detail.append(f"零指令站姿（{cond}）：静站 p90 {sp} rad（上限 {stand_bar(r):.2f}），3 s 后摔 {sl}；急停后 p90 {mp} rad，摔 {mf}")
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
  # the path on a flight (descent_probe --straight): the heading judged unloaded and typical, the full payload reported
  hj = flight_heading_judged(r)
  mb = meander_bar(r)
  has_typ = expect_typ or os.path.exists(f"{prefix}_descent_typ.txt")
  for label, sfx, judged, mbar in (("", "", hj, mb), ("常用负载 ", "_typ", hj, mb), ("满载 ", "_pay", False, None)):
    if sfx == "_typ" and not has_typ:
      continue
    fb, fd = flight_path_lines(r, f"{prefix}_descent{sfx}.txt", label, judged, mbar)
    bad += fb
    detail += fd
  print(verdict_line(bad))
  for line in detail:
    print("  " + line)
  _tail(prefix)
  return 0


def _tail(prefix: str) -> None:
  """The report lines after the items: the unloaded / loaded readings side by side, the evaluation's randomization."""
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


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
