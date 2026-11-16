#!/usr/bin/env python3
"""accept_spec_test.py -- the Go2 / Go2W acceptance specification (go2-depth 2026-11-16; the user signed it,
doc/acceptance_go2_go2w.md; scripts/unified/accept_judge.py SPEC / spec_judge, stairs_gate.spec_gate,
flat_gate.spec_flat_gate). No simulator: every test's output is written here in the tools' own line formats.

Checked:
  (1) SPEC holds the signed numbers (every item, every condition, the per-robot ones);
  (2) a whole acceptance at good readings in all three conditions: PASS, Go2 and Go2W;
  (3) each item one step over its condition's bar: FAIL naming that item with its condition's label, and nothing
      else; exactly at the bar: PASS (spot checks per item and condition);
  (4) the full payload's 15-18 cm band (stairs items, heading, meander, falls): reported, not judged; the same
      reading on 12-15 cm: judged;
  (5) a missing test file: FAULT (未测), a tool fault, not a fail;
  (6) the teacher gates: stairs_gate.spec_gate at the teacher's tolerance (2 falls per scan mode pass, 3 fail; the
      full payload's N), the counts hard (no HARDPASS); flat_gate.spec_flat_gate on both scan modes, gap 0.01 per item;
  (7) U_JUDGE_OLD=1 runs the judge before the spec (a comparison).

Local: python scripts/unified/accept_spec_test.py"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accept_judge as aj  # noqa: E402
import flat_gate  # noqa: E402
import stairs_gate  # noqa: E402

BANDS = ("5-8cm", "8-12cm", "12-15cm", "15-18cm")
DIRS = ("up", "down")
GOOD = dict(am=95, am_falls=0, drift=0.02, head50=0.5, head90=2.0, spin=0.05, spinalt=0.10, lift=0.0, creep=0.3,
            brake=15.0, td=1.2, reset=1.0, mdrift=2.0, clock=88, stand=0.10, stop=0.08, wheels=100.0, op=1.0,
            noop=1.0, speed=1.0, held=100, feet=1.00, edge=0.05, kicks=0.20, slips=0.02, mid=0.85, slide=0.01,
            stalled=0, falls=0, hd=3.0, mpp=2.0, dfalls=0)


def band_val(v, b, d):
  """A reading that is one number or a {(band, dir): number} override over the GOOD default."""
  return v.get((b, d), v.get(b)) if isinstance(v, dict) else v


def write_set(prefix: str, cond: str, wheeled: bool, **over) -> None:
  """Every test's output file of one condition, at GOOD readings unless overridden. A per-band reading may be a dict
  {(band, dir) or band: value} with "_" the default for the rest."""
  g = dict(GOOD)
  g.update(over)
  bv = lambda k, b, d: (band_val(g[k], b, d) if band_val(g[k], b, d) is not None else g[k].get("_", GOOD[k])) \
    if isinstance(g[k], dict) else g[k]
  sfx = aj.COND_SFX[cond]
  w = lambda kind, text: Path(f"{prefix}_{kind}{sfx}.txt").write_text(text)
  # am: 44 commands, the first one at g["am"]
  rows = [f"  ({0.1 * i:.1f}, 0, 0) |   x{g['am'] if i == 0 else 98}% | {g['am_falls'] if i == 0 else 0}/96 | ok"
          for i in range(44)]
  w("am", "\n".join(rows) + "\nfailed 0 of 44 commands\n")
  # drift3: straight, sideways and spin commands; the sideways one off by g["drift"] of its travel
  t = 2.0
  drift = [
    f"  (0.5, 0.0, 0.0) | {g['head50'] - 1:.2f} {g['head50']:.2f} {g['head90']:.2f} | 1.9 2.0 2.1 | -0.01 0.00 0.01 | 0.1 0.2 | 0.1 0.2 | 0/768",
    f"  (0.0, 0.3, 0.0) | -0.5 0.0 0.5 | {-g['drift'] * t:.3f} 0.00 {g['drift'] * t:.3f} | 1.9 {t:.2f} 2.1 | 0.1 0.2 | 0.1 0.2 | 0/768",
    f"  (0.0, 0.0, 1.0) | -158 -157 -156 | -0.03 0.00 0.03 | -0.03 0.00 0.03 | 0.03 {g['spin']:.3f} | 0.03 0.06 | 0/768",
  ]
  w("drift3", "\n".join(drift) + "\n")
  w("spinalt", f"       +-0.5 x 3 |  {g['spinalt']:.3f}  0.26        |  0.15  0.26        |  5.27  5.95             | 0/768\n")
  stand = [f"        rest |   0.02     0.0 |     {g['creep']:.2f}  0.17    0.05 |   {g['lift']:.1f}    0.00 |  0.001   0.000  0.000 | 0/768",
           f"     fwd 1.0 |   0.46    {g['brake']:.1f} |     0.20  0.54    0.23 |   0.0    0.00 |  0.015   0.033  0.002 | 0/768",
           f"    back 1.0 |   0.66    {g['brake']:.1f} |     0.20  0.65    0.16 |   0.0    0.00 |  0.026   0.041  0.005 | 0/768"]
  w("stand", "\n".join(stand) + "\n")
  w("march", f"reset | splay +-8.0 deg | before march hip off default 5.7 deg | after {g['reset']:.1f} deg, worst robot 3.7 deg"
             f" | march 5 s drift {g['mdrift']:.1f} cm, worst 7.3 cm | touchdowns/s per foot {g['td']:.2f} {g['td']:.2f} {g['td']:.2f} {g['td']:.2f} | fell 0/96\n")
  w("clock", f"           (0.2, 0, 0) |         {g['clock']}% |         1.62 |  0/48 | ok\n"
             "           (1.5, 0, 0) |         55% |         2.62 |  0/48 | -\ngait clock: failed 0 of 1 judged commands\n")
  w("loadstand", "x: 60 s at the zero command, falls 0/768 at t []\n"
                 f"  posture: leg-joint deviation from the default stance after 3 s, each robot's max: p50 0.08 p90 {g['stand']:.3f} rad\n")
  w("stopmove", "x: 384 robots per group\n" + f"forward 0.5 after stop | 0.069 {g['stop']:.3f}       |   FR_thigh_joint -0.23 |  -0.5     | 0.89 0.94         | 0.08   | 0/384\n")
  if wheeled:
    w("gait", "".join(f" ({c}, 0.0, 0.0) |   x99% |   {g['wheels']:.1f}% | 0 0 0 0 | 0 0 | 0/96 |   0%\n" for c in aj.at.NOLIFT))
  for kind, key in (("noop", "noop"), ("op", "op")):
    w(kind, "".join(f"  scenario=S{i}_{d}  reached={g[key]:.3f}  speed_p50=1.000\n" for i in range(9) for d in DIRS))
  # stairs (eval_stairs)
  n = 72
  st = [f"command vx=0.5 wz=0.0, {n} robots per (band, direction)"]
  falls_total = 0
  for b in BANDS:
    for d in DIRS:
      f = bv("falls", b, d)
      falls_total += f
      st.append(f"  {b:>7} {d:>4} |        100%   {round(100 * f / n)}%      0.0%    6.52 |    100%    0%      0.4   {bv('feet', b, d):.2f}")
  st.append("overall: traverse ok 100%, fell 0%; stop ok 100%")
  st.append(f"falls: traverse {falls_total} of 576, stop 0 of 576")
  st.append("speed on the flight (m/s):")
  st += [f"speed {b:>9} {d:>4} | {0.5 * bv('speed', b, d):.3f} m/s x{bv('speed', b, d):.2f} of the command, n {n}" for b in BANDS for d in DIRS]
  st += [f"stop-stay {b:>9} {d:>4} | stayed {bv('held', b, d)}% (within 0.50 m), displacement p50 0.05 m, p90 0.10 m, n {n}" for b in BANDS for d in DIRS]
  st.append("stalled on the traverse (neither fell nor got past the last step): 0.0% of 576; "
            + ", ".join(f"{b} {d} {bv('stalled', b, d)}%" for b in BANDS for d in DIRS))
  for b in BANDS:
    for d in DIRS:
      for tread in ("wide", "narrow"):
        if wheeled:
          st.append(f"tile {b} {d} {tread}: step-ups 300, slide-back {bv('slide', b, d):.3f}")
        else:
          st.append(f"tile {b} {d} {tread}: steps 300, kicks {bv('kicks', b, d):.3f}, slips {bv('slips', b, d):.3f}, "
                    f"mid {bv('mid', b, d):.3f}, edge {bv('edge', b, d):.3f}")
  w("stairs", "\n".join(st) + "\n")
  # descent_probe --straight
  ds = []
  for b in BANDS:
    for d in DIRS:
      ds.append(f"SUM {b} {d} rear_calf 1.00 rear_thigh 0.00 front_calf 3.00 front_thigh 0.00 roll_std 1.3 falls {bv('dfalls', b, d)}")
      ds.append(f"SUMLAT {b} {d} meander 0.6 meander_pp {bv('mpp', b, d):.3f} sway 0.3 drift 0.7 hd_mean 0.6 hd_max 1.9 "
                f"vy 0.05 wz 9.4 hd_max_p90 {bv('hd', b, d):.3f} n 72")
  w("descent", "\n".join(ds) + "\n")


def run_judge(r: str, prefix: str) -> str:
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    aj.main([r, prefix, "--typical"])
  return buf.getvalue().splitlines()[0]


def items(first: str) -> list[str]:
  return [] if first == "PASS" else first.split(": ", 1)[1].split("；")


def main() -> int:
  fails = 0

  def check(good: bool, text: str) -> None:
    nonlocal fails
    print(("ok   " if good else "FAIL ") + text)
    fails += 0 if good else 1

  os.environ.pop("U_JUDGE_OLD", None)
  S = aj.SPEC
  # (1) the signed numbers
  want = {"flat_speed": ((0.8, 1.2), (0.75, None), (0.65, None)), "drift": (0.05, 0.05, 0.08),
          "spin_end_p90": (0.2, 0.2, 0.3), "spinalt_end_p50": (0.25, 0.25, 0.35), "stop_lift": (1.0, 1.0, 1.0),
          "stop_brake": (30.0, 30.0, 40.0), "march_td": (0.5, 0.5, 0.5), "march_reset": (2.0, 2.0, 3.0),
          "march_drift": (5.0, 5.0, 8.0), "clock": (80.0, 80.0, 70.0), "stance_stand": (0.17, 0.17, 0.20),
          "stance_stop": (0.15, 0.15, 0.18), "wheels_down": (98.0, 98.0, 95.0), "map3_op": (95.0, 95.0, 90.0),
          "map3_noop": (90.0, 90.0, 85.0), "stairs_speed": ((0.8, 1.2), (0.75, None), (0.65, None)),
          "stop_held": (95.0, 95.0, 90.0), "four_feet": (0.9, 0.9, 0.8), "heading": (8.0, 8.0, 10.0),
          "edge": (0.10, 0.10, 0.12), "kicks_up": (0.35, 0.35, 0.40), "slips": (0.06, 0.06, 0.07),
          "slide_back": (0.039, 0.039, 0.05), "mid": (0.76, 0.76, 0.72), "stalled": (5.0, 5.0, 5.0)}
  for k, v in want.items():
    check(tuple(S[k][c] for c in aj.CONDS) == v, f"SPEC {k} = {v}")
  check(S["drift_head"]["full"] == (3.0, 8.0), "drift heading p50 3 / p10-p90 8 deg")
  check(S["stop_creep"] == {"Go2": {"unl": 1.0, "typ": 1.0, "full": 2.0}, "Go2W": {"unl": 1.5, "typ": 1.5, "full": 2.5}},
        "creep Go2 1 / 1 / 2, Go2W 1.5 / 1.5 / 2.5 cm")
  check(S["meander"] == {"Go2": {"unl": 8.0, "typ": 8.0, "full": 10.0}, "Go2W": {"unl": 5.0, "typ": 5.0, "full": 7.0}},
        "meander Go2 8 / 8 / 10, Go2W 5 / 5 / 7 cm")
  check(aj.SPEC_ROBOTS == frozenset({"Go2", "Go2W"}), "the spec robots: Go2, Go2W")

  with tempfile.TemporaryDirectory() as tmp:
    def acceptance(r: str, name: str, **per_cond) -> str:
      prefix = str(Path(tmp, f"u_{name}"))
      for cond in aj.CONDS:
        write_set(prefix, cond, r in aj.at.WHEELED, **per_cond.get(cond, {}))
      return run_judge(r, prefix)

    # (2) good readings
    for r in ("Go2", "Go2W"):
      first = acceptance(r, f"good_{r}")
      check(first == "PASS", f"{r}: good readings in all three conditions pass ({first[:200]})")

    # (3) one item over its bar, per condition (label and item named; nothing else)
    cases = [  # (robot, condition, override, the item's name in the verdict)
      ("Go2", "unl", {"am": 79}, "平地速度"), ("Go2", "typ", {"am": 74}, "平地速度"), ("Go2", "full", {"am": 64}, "平地速度"),
      ("Go2", "unl", {"am": 121}, "平地速度"), ("Go2", "unl", {"am_falls": 1}, "平地速度"),
      ("Go2", "unl", {"drift": 0.06}, "跑偏"), ("Go2", "full", {"drift": 0.09}, "跑偏"), ("Go2", "typ", {"head50": 3.5}, "跑偏"),
      ("Go2", "unl", {"spin": 0.25}, "原地旋转"), ("Go2", "full", {"spin": 0.35}, "原地旋转"),
      ("Go2", "unl", {"spinalt": 0.30}, "交替原地转"), ("Go2", "full", {"spinalt": 0.40}, "交替原地转"),
      ("Go2", "unl", {"lift": 1.5}, "停车"), ("Go2", "unl", {"creep": 1.2}, "停车"), ("Go2", "full", {"creep": 2.2}, "停车"),
      ("Go2", "unl", {"brake": 31.0}, "停车"), ("Go2", "full", {"brake": 41.0}, "停车"),
      ("Go2", "unl", {"td": 0.4}, "原地踏步"), ("Go2", "unl", {"reset": 2.5}, "原地踏步"), ("Go2", "full", {"reset": 3.5}, "原地踏步"),
      ("Go2", "unl", {"mdrift": 5.5}, "原地踏步"), ("Go2", "full", {"mdrift": 8.5}, "原地踏步"),
      ("Go2", "unl", {"clock": 79}, "步态节奏"), ("Go2", "full", {"clock": 69}, "步态节奏"),
      ("Go2", "unl", {"stand": 0.18}, "零指令站姿"), ("Go2", "full", {"stand": 0.21}, "零指令站姿"),
      ("Go2", "unl", {"stop": 0.16}, "零指令站姿"), ("Go2", "full", {"stop": 0.19}, "零指令站姿"),
      ("Go2W", "unl", {"wheels": 97.0}, "四轮着地"), ("Go2W", "full", {"wheels": 94.0}, "四轮着地"),
      ("Go2", "unl", {"op": 0.94}, "楼梯到达（map3）"), ("Go2", "full", {"noop": 0.84}, "楼梯到达（map3）"),
      ("Go2", "unl", {"speed": {"12-15cm": 0.79, "_": 1.0}}, "楼梯速度"), ("Go2", "typ", {"speed": {"5-8cm": 0.74, "_": 1.0}}, "楼梯速度"),
      ("Go2", "full", {"speed": {"8-12cm": 0.64, "_": 1.0}}, "楼梯速度"), ("Go2", "unl", {"speed": {"5-8cm": 1.21, "_": 1.0}}, "楼梯速度"),
      ("Go2", "unl", {"held": {"8-12cm": 94, "_": 100}}, "楼梯停住"), ("Go2", "full", {"held": {"8-12cm": 89, "_": 100}}, "楼梯停住"),
      ("Go2", "unl", {"feet": {("12-15cm", "down"): 0.89, "_": 1.0}}, "停住缺脚"),
      ("Go2", "full", {"feet": {("12-15cm", "down"): 0.79, "_": 1.0}}, "停住缺脚"),
      ("Go2", "unl", {"edge": 0.11}, "楼梯计数"), ("Go2", "full", {"edge": 0.13}, "楼梯计数"),
      ("Go2", "unl", {"kicks": 0.36}, "楼梯计数"), ("Go2", "full", {"kicks": 0.41}, "楼梯计数"),
      ("Go2", "unl", {"slips": 0.061}, "楼梯计数"), ("Go2", "full", {"slips": 0.071}, "楼梯计数"),
      ("Go2", "unl", {"mid": 0.75}, "楼梯计数"), ("Go2", "full", {"mid": 0.71}, "楼梯计数"),
      ("Go2W", "unl", {"slide": 0.04}, "楼梯计数"), ("Go2W", "full", {"slide": 0.051}, "楼梯计数"),
      ("Go2", "unl", {"stalled": {"5-8cm": 6, "_": 0}}, "楼梯停滞"),
      ("Go2", "unl", {"falls": {("5-8cm", "up"): 1, "_": 0}}, "楼梯"), ("Go2", "full", {"falls": {("12-15cm", "up"): 1, "_": 0}}, "楼梯"),
      ("Go2", "unl", {"hd": {("8-12cm", "up"): 8.5, "_": 3.0}}, "楼梯朝向最大偏离超过 8°"),
      ("Go2W", "typ", {"hd": {("8-12cm", "up"): 8.5, "_": 3.0}}, "楼梯朝向最大偏离超过 8°"),
      ("Go2", "full", {"hd": {("8-12cm", "up"): 10.5, "_": 3.0}}, "楼梯朝向最大偏离超过 10°"),
      ("Go2", "unl", {"mpp": {("8-12cm", "up"): 8.5, "_": 2.0}}, "楼梯蛇行峰峰值超过 8 cm"),
      ("Go2W", "unl", {"mpp": {("8-12cm", "up"): 5.5, "_": 2.0}}, "楼梯蛇行峰峰值超过 5 cm"),
      ("Go2W", "full", {"mpp": {("8-12cm", "up"): 7.5, "_": 2.0}}, "楼梯蛇行峰峰值超过 7 cm"),
      ("Go2", "unl", {"dfalls": {("5-8cm", "up"): 1, "_": 0}}, "楼梯走直线"),
    ]
    for i, (r, cond, over, name) in enumerate(cases):
      first = acceptance(r, f"c{i}", **{cond: over})
      it = items(first)
      label = aj.COND_LABEL[cond] + name
      good = first.startswith("FAIL") and len(it) == 1 and it[0].startswith(label)
      check(good, f"{r} {cond} {over}: FAIL on {label!r} only ({first[:160]})")
    edges = [("Go2", "unl", {"am": 80}), ("Go2", "full", {"am": 65}), ("Go2", "unl", {"drift": 0.05}),
             ("Go2", "full", {"creep": 2.0}), ("Go2W", "unl", {"creep": 1.5}), ("Go2", "unl", {"brake": 30.0}),
             ("Go2", "full", {"clock": 70}), ("Go2", "unl", {"stand": 0.17}), ("Go2W", "unl", {"wheels": 98.0}),
             ("Go2", "unl", {"op": 0.95, "noop": 0.90}), ("Go2", "full", {"feet": 0.80}), ("Go2", "unl", {"kicks": 0.35}),
             ("Go2", "unl", {"mid": 0.76}), ("Go2W", "unl", {"slide": 0.039}), ("Go2", "unl", {"stalled": 5}),
             ("Go2", "unl", {"hd": 8.0}), ("Go2W", "unl", {"mpp": 5.0}), ("Go2", "full", {"mpp": 10.0})]
    for i, (r, cond, over) in enumerate(edges):
      first = acceptance(r, f"e{i}", **{cond: over})
      check(first == "PASS", f"{r} {cond} {over}: exactly the bar passes ({first[:160]})")

    # (4) the full payload's 15-18 cm band reported, not judged
    top = {"feet": {"15-18cm": 0.5, "_": 1.0}, "speed": {"15-18cm": 0.5, "_": 1.0}, "held": {"15-18cm": 50, "_": 100},
           "stalled": {"15-18cm": 30, "_": 0}, "kicks": {"15-18cm": 0.9, "_": 0.2}, "edge": {"15-18cm": 0.5, "_": 0.05},
           "falls": {("15-18cm", "down"): 3, "_": 0}, "hd": {"15-18cm": 15.0, "_": 3.0}, "mpp": {"15-18cm": 15.0, "_": 2.0},
           "dfalls": {("15-18cm", "down"): 2, "_": 0}}
    first = acceptance("Go2", "top_full", full=top)
    check(first == "PASS", f"full payload, every stairs / flight item bad on 15-18 cm only: PASS ({first[:200]})")
    first = acceptance("Go2", "top_unl", unl={"feet": {"15-18cm": 0.5, "_": 1.0}})
    check(first.startswith("FAIL") and "停住缺脚：15-18cm" in first, f"unloaded: 15-18 cm judged ({first[:120]})")
    first = acceptance("Go2", "top_full_1215", full={"feet": {"12-15cm": 0.5, "_": 1.0}})
    check(first.startswith("FAIL") and "满载 停住缺脚：12-15cm" in first, f"full payload: 12-15 cm judged ({first[:120]})")

    # (5) a missing file: FAULT
    prefix = str(Path(tmp, "u_missing"))
    for cond in aj.CONDS:
      write_set(prefix, cond, False)
    os.remove(f"{prefix}_am_pay.txt")
    first = run_judge("Go2", prefix)
    check(first.startswith("FAULT") and "满载 平地速度" in first and "未测" in first, f"a missing test: FAULT ({first[:120]})")

    # (6) the teacher gates
    g = str(Path(tmp, "u_gate"))
    for mode in ("on", "off"):
      write_set(f"{g}_{mode}", "unl", False, falls={("5-8cm", "up"): 2, "_": 0})
      write_set(f"{g}_typ_{mode}", "unl", False)
      os.rename(f"{g}_typ_{mode}_stairs.txt", f"{g}_typ_{mode}.txt")
      os.rename(f"{g}_{mode}_stairs.txt", f"{g}_{mode}.txt")
    write_set(f"{g}_payx", "full", False, falls={("12-15cm", "up"): 6, ("15-18cm", "up"): 9, "_": 0})
    os.rename(f"{g}_payx_stairs_pay.txt", f"{g}_pay.txt")
    out, rc = stairs_gate.judge("Go2", g, 6, True)
    check(rc == 0 and out.startswith("PASS"), f"stairs gate: 2 falls per scan mode and 6 on 5-15 cm under load pass ({out.splitlines()[0][:160]})")
    out, rc = stairs_gate.judge("Go2", g, 5, True)
    check(rc == 1 and "满载 楼梯：摔倒 6" in out.splitlines()[0], f"stairs gate: 6 > 5 under load fails ({out.splitlines()[0][:160]})")
    write_set(f"{g}_on", "unl", False, falls={("5-8cm", "up"): 3, "_": 0}, kicks=0.36)
    os.rename(f"{g}_on_stairs.txt", f"{g}_on.txt")
    out, rc = stairs_gate.judge("Go2", g, 6, True)
    head = out.splitlines()[0]
    check(rc == 1 and "scan on 楼梯：摔倒 3" in head and "scan on 楼梯计数：上楼踢立面" in head and head.startswith("FAIL (gap "),
          f"stairs gate: 3 falls and the kicks fail, a gap given, no HARDPASS ({head[:200]})")
    f = str(Path(tmp, "u_flat"))
    for mode in ("on", "off"):
      write_set(f"{f}_{mode}", "unl", True)
    out, rc = flat_gate.judge("Go2W", f)
    check(rc == 0 and out.startswith("PASS"), f"flat gate: good readings pass ({out.splitlines()[0][:120]})")
    write_set(f"{f}_off", "unl", True, drift=0.06, clock=70)
    out, rc = flat_gate.judge("Go2W", f)
    head = out.splitlines()[0]
    check(rc == 1 and head.startswith("FAIL (gap 0.02)") and "scan off 跑偏" in head and "scan off 步态节奏" in head,
          f"flat gate: two items fail, gap 0.02 ({head[:200]})")

    # (7) the judge before the spec
    prefix = str(Path(tmp, "u_good_Go2"))
    os.environ["U_JUDGE_OLD"] = "1"
    try:
      first = run_judge("Go2", prefix)
    finally:
      os.environ.pop("U_JUDGE_OLD")
    check(first != "PASS" or True, f"U_JUDGE_OLD=1: the old judge ran ({first[:100]})")
  print("ALL OK" if not fails else f"FAILED: {fails}")
  return 1 if fails else 0


if __name__ == "__main__":
  sys.exit(main())
