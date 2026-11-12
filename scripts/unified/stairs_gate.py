#!/usr/bin/env python3
"""stairs_gate.py R PREFIX [--pay N] [--typ] (unitree-rl-mjlab-62 2026-11-13; go2-depth's gate design, option (b) of doc/redesign.md's
teacher gate): does a teacher checkpoint pass the stairs part of its gate? Read from tier-2's own stairs test
(scripts/go2_blind/eval_stairs.py --metrics, Unitree-<R>-U-Teacher-Stairs, unloaded) in both scan modes:
PREFIX_on.txt and PREFIX_off.txt. jobs/unified/stairs_gate.sh runs them; jobs/unified/teacher.sh calls it on a seed's
"PRE-PASS K" (scripts/unified/teacher_gate.py --pre: the curriculum done and the flat items passed).

Why tier 2 and not the training's readings: on one checkpoint the training's stairs readings ran 1.5-4x the
acceptance's (Go2 kicks 0.72 / 0.46, Go2W slides 0.094 / 0.028, 11-13), the ratio moving with the policy.

Judged, in each scan mode (the students' bars unless said):
  falls        traverse + stop, exact count: <= FALLS_MAX (2) per mode
  stop ok      the mid-flight stop's share: >= 95%
  four feet    the stop's all-feet-down share per band and direction: >= accept_tables.stop_feet_bar
  speed        the flight speed over the command (eval_stairs' "speed" lines, robots weighted), up and down: 0.8-1.2
  counts       TEACHER_COUNT_BARS: "no worse than the delivered model" on tier 2 (option (b): the students' 0.245 kicks
               is beyond even the delivered Go2 blind; a teacher worse than delivered does not go on); a robot without
               an entry: the students' bars. Fewer than MIN_EVENTS counted: untested (a tool fault)
--pay N: also PREFIX_pay.txt (the seen mode with the walking payload, 8 kg at the acceptance's point), judged on its
falls only (traverse + stop <= N); its speed, four feet and counts reported.
--typ: also PREFIX_typ_on.txt / PREFIX_typ_off.txt (both scan modes with the typical payload, jobs/accept_payload.sh R
typical), judged as the students' typical condition judges its stairs (accept_judge.typical()) at the teacher's
tolerances: falls <= FALLS_MAX, stop ok, four feet and the counts as unloaded, the flight speed at least
accept_judge.STAIRS_SPEED_REL of the same mode's unloaded reading in every band and direction, and at most
accept_judge.TYPICAL_STALL_MAX % stalled in every band and direction (11-13: Go2W s33 4997 stalled 40% at 15-18 cm up
with 4 kg, which the speed -- read on the robots that got past -- let through). Why (11-13): Go2 s38
6097 fell 43 / 27 and Go2W s28 4997 (every unloaded stairs item passed) 11 / 24 at the typical 4 kg, 10-36% on the
15-18 cm climbs; the students' acceptance takes no fall there, so such a teacher's students fail 7000 iterations later.
Reported only: the unloaded runs' stalled share, the heading on the flight.
First line: "PASS ...", "FAIL (gap G): <reasons>" (G: the summed relative shortfall, for picking the best judged at the
cap) or "FAULT: <what could not be read>". Exit 0 / 1 / 4.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "unified"))
import accept_judge as aj  # noqa: E402

at = aj.at
FALLS_MAX = 2   # per scan mode, traverse + stop, of 576 (go2-depth 11-13: one number for every robot; the students'
                # acceptance keeps 0 + the luck rule)
STOP_OK_MIN = 95.0
SPEED_RANGE = (0.8, 1.2)
MIN_EVENTS = aj.MIN_STAIR_EVENTS
STUDENT_COUNT_BARS = {"legged": {"kicks": ("max", aj.LEGGED_BARS["kicks"]), "slips": ("max", aj.LEGGED_BARS["slips"]),
                                 "mid": ("min", aj.LEGGED_BARS["mid"]), "edge": ("max", aj.LEGGED_BARS["edge"])},
                      "wheeled": {"slide_back": ("max", aj.WHEELED_SLIDE_BAR)}}
# The teacher's stairs counts: the delivered model's tier-2 reading (11-13, the counters of 0dc3b56 / 28edf91, every
# robot counted; go2-depth's choice (b)). edge: the students' 0.10, not the delivered 0.083 (unitree-rl-mjlab-62: a
# teacher-only stricter edge would block on an item the students' own bar allows; the Go2 candidates read 0.090-0.097).
TEACHER_COUNT_BARS = {
  "Go2": {"bars": {"kicks": ("max", 0.31), "slips": ("max", 0.069), "mid": ("min", 0.778), "edge": ("max", 0.10)},
          "source": "go2_blind_v4 model_10000 (logs/pkg/go2_fe_polish_10000) on t1, 0dc3b56: kicks 0.308, slips 0.068, "
                    "mid 0.778, edge 0.083 (logs/checks/u_go2_v4_10000_stairs_tiles.txt)"},
  "Go2W": {"bars": {"slide_back": ("max", 0.039)},
           "source": "go2w_bl2_g14_13600 (logs/pkg/go2w_bl2_g14_13600) on t2, 28edf91: slide-back 0.039 over 17116 "
                     "step-ups (logs/checks/u_go2w_deployed_13600_stairs_tiles.txt)"},
}


def speeds(t: str) -> dict[str, float]:
  """The flight speed over the command per direction, the bands' robots pooled."""
  out = {}
  for d in ("up", "down"):
    rows = re.findall(rf"^speed\s+\S+\s+{d} \| [\d.]+ m/s x([\d.]+) of the command, n (\d+)", t, re.M)
    n = sum(int(c) for _, c in rows)
    if n:
      out[d] = sum(float(x) * int(c) for x, c in rows) / n
  return out


def stop_ok(t: str) -> float | None:
  m = re.search(r"overall: traverse ok \d+%, fell \d+%; stop ok (\d+)%", t)
  return None if m is None else float(m.group(1))


def feet(t: str) -> list[tuple[str, str, float]]:
  return [(m.group(1), m.group(2), float(m.group(3))) for m in re.finditer(
    r"^\s*(\d+-\d+cm)\s+(up|down)\s*\|.*\|\s*\d+%\s+\d+%\s+[\d.]+\s+([\d.]+)\s*$", t, re.M)]


def mode_checks(r: str, path: str, t: str, tag: str, bars: dict, bad: list, missing: list, detail: list) -> float:
  """One run's falls, stop ok, four feet and counts (both the unloaded and the typical runs); returns its gap."""
  wheeled = r in at.WHEELED
  gap = 0.0
  falls = aj.stairs_fall_counts(path)
  if falls is None:
    missing.append(f"{tag}: no falls line")
  else:
    detail.append(f"{tag}: falls {falls}")
    if falls > FALLS_MAX:
      bad.append(f"{tag} 摔倒 {falls}")
      gap += 1.0 + 0.1 * (falls - FALLS_MAX)
  so = stop_ok(t)
  if so is None:
    missing.append(f"{tag}: no stop ok")
  else:
    detail.append(f"{tag}: stop ok {so:.0f}%")
    if so < STOP_OK_MIN:
      bad.append(f"{tag} 停不稳 {so:.0f}%")
      gap += (STOP_OK_MIN - so) / STOP_OK_MIN
  short = [(b, d, v) for b, d, v in feet(t) if v < at.stop_feet_bar(r, b, d)]
  if short:
    bad.append(f"{tag} 停住缺脚 " + "，".join(f"{b} {d} {v:.2f}" for b, d, v in short))
    gap += sum(at.stop_feet_bar(r, b, d) - v for b, d, v in short)
  sm = aj.stairs_metrics(path)
  n = sm.get("ups") if wheeled else sm.get("steps")
  if n is None:
    missing.append(f"{tag}: no stairs counts")
  elif n < MIN_EVENTS:
    missing.append(f"{tag}: {n:.0f} events counted (< {MIN_EVENTS})")
  else:
    for k, (sense, bar) in bars.items():
      v = sm.get(k)
      if v is None:
        missing.append(f"{tag}: no {k}")
        continue
      detail.append(f"{tag}: {k} {v:.3f} ({'<=' if sense == 'max' else '>='} {bar})")
      if (sense == "max" and v > bar) or (sense == "min" and v < bar):
        bad.append(f"{tag} {k} {v:.3f}")
        gap += (v / bar - 1.0) if sense == "max" else (bar / max(v, 1e-6) - 1.0)
  st = re.search(r"^stalled on the traverse .*?: ([\d.]+)% of", t, re.M)
  ha, hu = aj.stairs_heading(path)
  detail.append(f"{tag}: stalled {st.group(1) if st else '-'}%, heading {ha if ha is not None else '-'} deg"
                f" (15-18cm up {hu if hu is not None else '-'}) (reported)")
  return gap


def judge(r: str, prefix: str, pay_falls: int | None = None, typ: bool = False) -> tuple[str, int]:
  wheeled = r in at.WHEELED
  kind = "wheeled" if wheeled else "legged"
  entry = TEACHER_COUNT_BARS.get(r)
  bars = entry["bars"] if entry else STUDENT_COUNT_BARS[kind]
  bad, missing, detail, gap = [], [], [], 0.0
  for mode in ("on", "off"):
    path = f"{prefix}_{mode}.txt"
    t = aj.txt(path)
    tag = f"scan {mode}"
    if not t:
      missing.append(f"{tag}: no output ({path})")
      continue
    sp = speeds(t)
    for d in ("up", "down"):
      if d not in sp:
        missing.append(f"{tag}: no {d} speed")
      elif not SPEED_RANGE[0] <= sp[d] <= SPEED_RANGE[1]:
        bad.append(f"{tag} {'上楼' if d == 'up' else '下楼'}速度 x{sp[d]:.2f}")
        gap += min(abs(sp[d] - SPEED_RANGE[0]), abs(sp[d] - SPEED_RANGE[1]))
    if sp:
      detail.append(f"{tag}: speed " + " ".join(f"{d} x{v:.2f}" for d, v in sp.items()))
    gap += mode_checks(r, path, t, tag, bars, bad, missing, detail)
  # the typical payload (--typ): both modes as the students' typical condition judges its stairs, the speed against
  # the same mode's unloaded run band by band
  if typ:
    for mode in ("on", "off"):
      path = f"{prefix}_typ_{mode}.txt"
      t = aj.txt(path)
      tag = f"常用负载 scan {mode}"
      if not t:
        missing.append(f"{tag}: no output ({path})")
        continue
      su, st_ = aj.flight_speeds(f"{prefix}_{mode}.txt"), aj.flight_speeds(path)
      if not su or not st_:
        missing.append(f"{tag}: no flight speeds")
      else:
        slow = [(k, st_.get(k, float("nan")), b) for k, b in su.items()
                if b == b and not st_.get(k, float("nan")) >= aj.STAIRS_SPEED_REL * b]
        if slow:
          bad.append(f"{tag} 楼梯速度低于空载的 {aj.STAIRS_SPEED_REL:.0%}："
                     + "、".join(f"{k[0]} {'上' if k[1] == 'up' else '下'} {v:.2f}/{b:.2f}" for k, v, b in slow))
          gap += sum(1.0 - (v / b if v == v else 0.0) / aj.STAIRS_SPEED_REL for _k, v, b in slow)
        detail.append(f"{tag}: speed (typical / unloaded, m/s) " + " ".join(
          f"{k[0]} {k[1]} {st_.get(k, float('nan')):.2f}/{b:.2f}" for k, b in su.items()))
      gap += mode_checks(r, path, t, tag, bars, bad, missing, detail)
      ss = aj.stairs_stalled(path)
      if not ss:
        missing.append(f"{tag}: no stalled line")
      else:
        over = [(b, d, v) for (b, d), v in ss.items() if v > aj.TYPICAL_STALL_MAX]
        if over:
          bad.append(f"{tag} 停住超过 {aj.TYPICAL_STALL_MAX:g}%：" + "、".join(f"{b} {'上' if d == 'up' else '下'} {v:g}%"
                                                                     for b, d, v in over))
          gap += sum((v - aj.TYPICAL_STALL_MAX) / 100.0 for _b, _d, v in over)
      lr = aj.stairs_reach(path)
      if lr:
        detail.append(f"{tag} per band (reported): {lr}")
  # the walking payload (go2-depth 11-13, the user's hard rule "带负载一定不能摔"): PREFIX_pay.txt -- the seen mode,
  # 8 kg at the acceptance's point -- judged on its falls only (pay_falls); speed, four feet and the counts reported
  if pay_falls is not None:
    path = f"{prefix}_pay.txt"
    t = aj.txt(path)
    pf = aj.stairs_fall_counts(path) if t else None
    if pf is None:
      missing.append(f"loaded: no falls line ({path})")
    else:
      sp, so = speeds(t), stop_ok(t)
      sm = aj.stairs_metrics(path)
      detail.append(f"loaded (scan on): falls {pf} (<= {pay_falls}); stop ok {so}; speed "
                    + " ".join(f"{d} x{v:.2f}" for d, v in sp.items()) + "; "
                    + ", ".join(f"{k} {v:.3f}" for k, v in sm.items() if k in ("kicks", "slips", "mid", "slide_back"))
                    + " (reported)")
      lr = aj.stairs_reach(path) if hasattr(aj, "stairs_reach") else None   # (an older accept_judge: none)
      if lr:
        detail.append(f"loaded per band (reported; a stall is not judged): {lr}")
      if pf > pay_falls:
        bad.append(f"满载摔倒 {pf}")
        gap += 1.0 + 0.1 * (pf - pay_falls)
  src = entry["source"] if entry else "the students' bars (no delivered reading for this robot)"
  if bad:
    head, rc = f"FAIL (gap {gap:.2f}): " + "；".join(bad + [f"未测 {m}" for m in missing]), 1
  elif missing:
    head, rc = "FAULT: " + "；".join(missing), 4
  else:
    head, rc = "PASS (the tier-2 stairs in both scan modes)", 0
  return "\n".join([head] + [f"  {d}" for d in detail] + [f"  count bars: {bars}; from {src}"]), rc


def main() -> int:
  args = sys.argv[1:]
  typ = "--typ" in args
  args = [a for a in args if a != "--typ"]
  pay = None
  if "--pay" in args:
    i = args.index("--pay")
    pay = int(args[i + 1])
    args = args[:i] + args[i + 2:]
  if len(args) != 2:
    print(__doc__)
    return 2
  try:
    out, rc = judge(args[0], args[1], pay, typ)
  except Exception as e:   # a tool fault, not a fail
    print(f"FAULT: stairs gate error: {type(e).__name__}: {e}")
    return 4
  print(out)
  return rc


if __name__ == "__main__":
  sys.exit(main())
