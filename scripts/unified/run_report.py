#!/usr/bin/env python3
"""run_report.py RUN_DIR [--last N] [--events M] [--all] [--json] (go2-depth 2026-11-12): what a training's records
say.

Reads only <RUN_DIR>/records/ (src/tasks/unified/diagnostics.py: timeline.jsonl, record.jsonl, status.json,
crash_traceback.txt) -- no simulator, no TensorBoard, no checkpoint; standard library only, so it runs anywhere the
run directory is. A run that resumed another one is read together with it (the start event's checkpoint path, when
that directory is there; of the older run only what it did up to that checkpoint). Prints, in Chinese:

  概况          robot / code / robots, the iterations, how the run ended (or that it is still running, or stopped
                without a word: killed or crashed hard), faults of the records themselves
  时间线        the curriculum's events: milestones, constraints switching on, the quality stage's gate, relaxing,
                multipliers at their limits, alerts, checkpoints left out (--events: how many lines, default 60)
  现状          the last snapshot: progress, rows, fail rates, noise, reward
  阶段预期      the reference schedule (diagnostics.expectations): each milestone's iteration, when it was reached,
                and the ones overdue -- the first thing to read on a new version
  进度预估      iterations per hour, and when p = 1, q = 1 and the gate's window can be reached at this pace
  各地面的回合  over the last N iterations (default 500): per ground the episodes, the fail rate, why they failed
                (termination term, the bodies that touched), on which rows, in which scan mode / payload / command,
                training vs evaluation robots, and where the failed ones ended (the breakdown for the grounds
                that fail most; --all: for every ground)
  负载          fail rates by payload class and by where the payload sits (the robot must not fall with a payload)
  约束          every constraint: readings per scan mode and how they moved over the window, the teacher's target,
                the students' bar, the multiplier, and its state (未开 / 达标 / 未达标 / 到上限 / 放松中 / 无读数 /
                只进门槛: a gate-only item, judged but without a reward term or a multiplier)
  门槛          teacher_gate.py's rule read from the snapshots of the last 500 iterations (the gate itself reads
                TensorBoard's every iteration; this is the same rule on every 20th)
  优化与物理    explained variance, gradient norms and clipping, action clipping, torque saturation, joint limits
  告警          every alert with when it began and ended
  最可能的问题  the findings above ranked, each with its evidence; "没有发现明显问题" when there is none

--json prints the numbers behind the report instead."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

MODES = ("seen", "blind")
CLASS_ZH = {"scan": "扫描", "pay": "负载", "load": "负载位置", "cmd": "指令", "role": "机器人"}
NAME_ZH = {"seen": "可见", "blind": "遮挡", "light": "轻", "mid": "中", "heavy": "重", "stand": "站立",
           "straight": "直行", "turning": "转向", "sideways": "横移", "train": "训练", "eval": "评估",
           "centre": "居中", "front": "偏前", "back": "偏后", "side": "偏侧"}
# the reference schedule's milestones (diagnostics.expectations)
EXPECT_ZH = {"walk": "平地会走（平地等地面的失败率降到上限以下）", "commands": "进度 p 到 0.25（指令放开到全范围）",
             "climb_first": "开始会上楼（低行楼梯上评估机器人的上楼回合过半成功）", "climb_low": "上楼行数到 1.5",
             "climb": "上楼行数达标：楼梯质量阶段开始", "p_done": "进度 p 到 1", "q_done": "质量 q 到 1"}
WHERE_ZH = {"dist": ("离出生点", "m"), "sec": ("回合第", "s"), "pitch": ("低头", "°"), "z": ("高度差", "m"),
            "speed": ("指令速度", "m/s")}
# ("waiting_fail": records before 2026-11-12, when the climbing fail rate at the curriculum's rows also gated q)
GATE_ZH = {"waiting_row": "等爬楼行数", "waiting_fail": "等上楼失败率降下来", "running": "在走", "done": "已完成",
           "waiting_health": "等楼梯恢复（轻载固定行失败率高于参考）", "back": "在回退（轻载固定行失败率明显高于参考）"}
STAIRS = ("up", "down", "stairs")


def read_jsonl(path: str) -> list[dict]:
  out = []
  if os.path.isfile(path):
    with open(path, errors="replace") as f:
      for line in f:
        line = line.strip()
        if line:
          try:
            out.append(json.loads(line))
          except ValueError:
            pass   # a line cut off by a kill
  return out


def load(run: str) -> dict:
  d = os.path.join(run, "records")
  status = None
  try:
    with open(os.path.join(d, "status.json")) as f:
      status = json.load(f)
  except (OSError, ValueError):
    pass
  crash = ""
  try:
    with open(os.path.join(d, "crash_traceback.txt"), errors="replace") as f:
      crash = f.read().strip()
  except OSError:
    pass
  return {"run": run, "events": read_jsonl(os.path.join(d, "timeline.jsonl")),
          "snaps": read_jsonl(os.path.join(d, "record.jsonl")), "status": status, "crash": crash}


def chain(run: str) -> list[dict]:
  """The run and the runs it resumed, oldest first."""
  runs, seen = [], set()
  while run and run not in seen and os.path.isdir(os.path.join(run, "records")):
    seen.add(run)
    r = load(run)
    runs.insert(0, r)
    start = next((e for e in r["events"] if e.get("ev") == "start"), None)
    ck = ((start or {}).get("resumed") or {}).get("checkpoint")
    run = os.path.dirname(ck) if ck else None
  # what an older run did after the checkpoint the next one resumed is not part of this training (RESUME_AT)
  for old, new in zip(runs, runs[1:]):
    start = next((e for e in new["events"] if e.get("ev") == "start"), {})
    cut = (start.get("resumed") or {}).get("it")
    if num(cut) is None:
      continue
    old["cut"], old["it_max"] = cut, max([s.get("it_end", s.get("it", 0)) for s in old["snaps"]] or [0])
    old["events"] = [e for e in old["events"] if num(e.get("it")) is None or e["it"] <= cut
                     or e.get("ev") in ("exception", "end", "fault")]
    old["snaps"] = [s for s in old["snaps"] if s.get("it", 0) <= cut]
  return runs


def num(x):
  return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def f(x, nd: int = 3) -> str:
  if x is None:
    return "-"
  if isinstance(x, str):
    return x
  if isinstance(x, bool):
    return "是" if x else "否"
  if isinstance(x, int):
    return str(x)
  return f"{x:.{nd}g}"


def pct(a, b) -> str:
  return "-" if not b else f"{100.0 * a / b:.0f}%"


def mean(v):
  v = [x for x in v if num(x) is not None]
  return sum(v) / len(v) if v else None


def hist_quantile(counts: list, edges: list, q: float = 0.5):
  """A quantile from a histogram (bin i: edges[i-1] .. edges[i]; the outer bins are open): the bin's middle."""
  total = sum(counts)
  if total <= 0:
    return None
  acc = 0.0
  for i, c in enumerate(counts):
    acc += c
    if acc >= q * total:
      if i == 0:
        return edges[0]
      if i >= len(edges):
        return edges[-1]
      return 0.5 * (edges[i - 1] + edges[i])
  return edges[-1]


def hist_text(counts: list, edges: list, unit: str) -> str:
  """The histogram's two fullest bins."""
  total = sum(counts)
  if total <= 0:
    return "-"
  parts = []
  for i in sorted(range(len(counts)), key=lambda i: -counts[i])[:2]:
    if counts[i] <= 0:
      continue
    lo = "<" + f(edges[0]) if i == 0 else ">" + f(edges[-1]) if i >= len(edges) else f"{f(edges[i - 1])}~{f(edges[i])}"
    parts.append(f"{lo}{unit} {pct(counts[i], total)}")
  return "、".join(parts)


def alert_zh(e: dict) -> str:
  """An alert event in Chinese, from its data (its own English text for an alert not known here)."""
  base, _, arg = str(e.get("name", "")).partition(":")
  g = lambda k, nd=3: f(e.get(k), nd)                                    # noqa: E731
  r = lambda k: "-" if num(e.get(k)) is None else f"{e[k]:.0%}"         # noqa: E731
  gate = GATE_ZH.get(e.get("gate"), f(e.get("gate")))
  up_fail = f"上楼失败率 {g('up_fail')}" if "up_fail" in e else f"评估机器人固定行上楼失败率 {g('eval_up_fail')}"
  text = {
    "nonfinite": lambda: "损失 / 噪声 / 机器人状态里出现非有限数值",
    "std_rising": lambda: f"动作噪声 300 迭代内 {g('before')} → {g('std')}",
    "std_high": lambda: f"动作噪声 {g('std')}（超过 1.0）",
    "action_clip": lambda: f"{r('share')} 的动作分量顶在截断值（动作最大绝对值 {g('action_max')}）",
    "fail_high": lambda: (f"{arg}：最近 300 迭代评估机器人在固定行上的 {g('n')} 个回合里 {r('rate')} 失败（全部机器人 "
                          f"{r('all_rate')}），主要是 {g('cause')}" if e.get("role") == "eval" else
                          f"{arg}：最近 300 迭代的 {g('n')} 个回合里 {r('rate')} 失败，主要是 {g('cause')}")
                         + (f"（{e['bodies']}）" if e.get("bodies") else ""),
    "payload_fail": lambda: f"{arg}：带负载（中、重）的回合 {r('loaded')} 失败（{g('n_loaded')} 个），轻载的 {r('light')}"
                            f"（{g('n_light')} 个）",
    "no_episodes": lambda: f"{arg}：300 迭代内没有回合结束",
    "nan_resets": lambda: f"{g('n')} 个回合因 nan 终止",
    "len_drop": lambda: f"平均回合长度 300 迭代内 {g('before')} → {g('len')} 步",
    "level_falling": lambda: f"平均地形行数 {g('level')}，最近 500 迭代内到过 {g('recent_max')}",
    "eval_gap": lambda: (f"楼梯以外的地面：训练机器人失败率 {g('fail')}，按均值动作的评估机器人 {g('fail_eval')}"
                         if "n_eval" in e else f"全部机器人失败率 {g('fail')}，按均值动作的评估机器人 {g('fail_eval')}"),
    "value_loss_spike": lambda: f"价值损失 {g('value_loss')}，近期中位数 {g('median')}",
    "ev_low": lambda: f"价值网络的解释方差 {g('ev')}",
    "lr_floor": lambda: f"学习率 {g('lr')}，到了下限",
    "fps_drop": lambda: f"速度 {g('fps', 5)} 步/秒，平时 {g('median', 5)}",
    "q_waiting": lambda: f"进度 p {g('p')}，质量阶段还没开始：{gate}（爬楼行 {g('up')}，{up_fail}）",
    "q_stalled": lambda: f"质量 q 停在 {g('q')} 已 300 迭代：{gate}（爬楼行 {g('up')}，{up_fail}）",
    "clock_slow": lambda: f"楼梯以外地面的失败率 {g('fail')} 超过上限，进度时钟以 1/4 速走了 500 迭代",
    "relax": lambda: f"{arg} 地面的约束已放松 500 迭代（{fam_fail_zh(e, arg)}）",
    "cap": lambda: f"约束 {arg} 的乘子在上限 {g('scale')} 已 300 迭代，读数 {g('value')}，目标 {g('target')}",
    "unmeasured": lambda: f"约束 {arg} 300 迭代没有新读数（共 {g('n')} 次）",
    "late": lambda: f"阶段预期没按时达到：{EXPECT_ZH.get(arg, e.get('what'))}（期限迭代 {g('by')}；"
                    f"{expect_now(arg, e.get('now') or {})}）",
    "torque_sat": lambda: f"{arg} 关节有 {r('share')} 的力矩顶在上限",
  }.get(base)
  try:
    return text() if text else str(e.get("msg"))
  except Exception:
    return str(e.get("msg"))


def fam_fail(c: dict, fam: str):
  """The fail rate that relaxes a ground family's constraints, from a record's "cur" block or an event's data: the
  stairs families the evaluation robots' on their fixed rows (records that have them; curriculum.eval_rows), the
  other ground the training robots'."""
  ev, ff = c.get("eval_fail") or {}, c.get("family_fail") or {}
  if fam == "stairs":
    v = [x for x in (ev.get(k, ff.get(k)) for k in ("up", "down")) if num(x) is not None]
    return max(v) if v else None
  return ev.get(fam, ff.get(fam)) if fam in STAIRS else ff.get(fam, c.get("fail"))


def light_zh(c: dict, fam: str) -> str | None:
  """What relaxes a stairs family's constraints since 2026-11-12 (constraints._families): the light evaluation
  robots' fail rate on their fixed rows against its slow reference. From a record's "cur" block or an event's data;
  None for the other ground and for records from before."""
  light = c.get("eval_light") or {}
  ks = [k for k in (("up", "down") if fam == "stairs" else (fam,)) if k in light]
  if fam not in STAIRS or not ks:
    return None
  zh = {"up": "上楼", "down": "下楼"}
  return "轻载评估机器人固定行失败率 " + "，".join(
    f"{zh[k]} {f(light[k].get('fail'))}（参考值 {f(light[k].get('ref'))}，{light[k].get('n')} 回合）" for k in ks)


def bands_zh(d: dict, mode: str) -> str:
  """A family's eval_light entry: the fail rate per step-height band (5-8 / 8-12 / 12-15 / 15-18 cm) in one mode."""
  return " / ".join("-" if x is None else format(x, ".0%") for x in (d.get("bands") or {}).get(mode) or [])


def fam_fail_zh(c: dict, fam: str) -> str:
  light = light_zh(c, fam)
  if light:
    return light
  fixed = fam in STAIRS and bool(c.get("eval_fail"))
  return f"{'评估机器人固定行失败率' if fixed else '该地面失败率'} {f(fam_fail(c, fam))}"


def low_fail(c: dict, below: int = 3):
  """diagnostics.low_fail: the evaluation robots' climbing fail rate on their fixed rows below Q_ROW."""
  v = [x for x in ((c.get("eval_fail_rows") or {}).get("up") or [])[:below] if num(x) is not None]
  return sum(v) / len(v) if v else None


def rows_zh(c: dict, fam: str) -> str:
  rows = (c.get("eval_fail_rows") or {}).get(fam) or []
  return " ".join(f"{r}:{'-' if x is None else format(x, '.0%')}" for r, x in enumerate(rows))


def expect_now(name: str, c: dict) -> str:
  """A milestone's readings now: from a record's "cur" block, or from the late alert's `now` data."""
  fam, rows = c.get("family_fail") or {}, c.get("rows") or {}
  other = c.get("other_fail", fam.get("other"))
  up_row = c.get("up_row", rows.get("up"))
  if "up_fail" in c:   # (a late alert of the records before the fixed rows)
    stairs = f"上楼失败率 {f(c.get('up_fail'))}"
  elif "eval_up_fail" in c or c.get("eval_fail"):
    stairs = f"评估机器人固定行上楼失败率 {f(c.get('eval_up_fail', (c.get('eval_fail') or {}).get('up')))}"
  else:
    stairs = f"上楼失败率 {f(fam.get('up'))}"
  if name == "walk":
    return f"平地等地面失败率 {f(other)}，全部 {f(c.get('fail'))}"
  if name in ("commands", "p_done"):
    return f"进度 p {f(c.get('p'))}，失败率 {f(c.get('fail'))}"
  if name == "climb_first":
    return f"低行（0–2）上楼失败率 {f(c.get('low_fail', low_fail(c)))}，{stairs}，上楼行数 {f(up_row)}"
  if name == "climb_low":
    return f"上楼行数 {f(up_row)}，{stairs}"
  return (f"质量 q {f(c.get('q'))}，质量阶段 {GATE_ZH.get(c.get('q_gate'), f(c.get('q_gate')))}，爬楼行 {f(c.get('up'))}，"
          f"{stairs}")


def reached_milestones(events: list[dict]) -> dict:
  """The reference schedule's milestones reached, name -> the first iteration (over the resumed runs too)."""
  out: dict = {}
  for e in events:
    if e.get("ev") == "start":
      got = e.get("reached") or {}
    elif e.get("ev") == "milestone" and str(e.get("name", "")).startswith("expect:"):
      got = {e["name"][7:]: e.get("it")}
    else:
      continue
    for k, v in got.items():
      if num(v) is not None:
        out[k] = min(v, out.get(k, v))
  return out


def ep_sum(snaps: list[dict]) -> dict:
  """The snapshots' episode counts added per ground."""
  out: dict = {}
  for s in snaps:
    for g, d in (s.get("ep") or {}).items():
      o = out.setdefault(g, {"n": 0, "fail": 0, "timeout": 0, "sec": 0.0, "dist": 0.0, "cause": Counter(),
                             "family": d.get("family"),
                             "row": defaultdict(lambda: [0, 0]), "bodies": Counter(), "where": {},
                             **{k: defaultdict(lambda: [0, 0]) for k in CLASS_ZH}})
      n = d.get("n", 0)
      o["n"] += n
      o["fail"] += d.get("fail", 0)
      o["timeout"] += d.get("timeout", 0)
      o["sec"] += (num(d.get("sec")) or 0.0) * n
      o["dist"] += (num(d.get("dist")) or 0.0) * n
      o["cause"].update(d.get("cause") or {})
      o["bodies"].update(d.get("bodies") or {})
      for k in ("row",) + tuple(CLASS_ZH):
        for c, v in (d.get(k) or {}).items():
          o[k][c][0] += v[0]
          o[k][c][1] += v[1]
      for k, h in (d.get("where") or {}).items():
        cur = o["where"].setdefault(k, [0] * len(h))
        o["where"][k] = [a + b for a, b in zip(cur, h)]
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("run")
  ap.add_argument("--last", type=int, default=500, help="the window of the per-ground and constraint tables (it)")
  ap.add_argument("--events", type=int, default=60, help="how many of the timeline's last lines to print")
  ap.add_argument("--all", action="store_true", help="the fail-rate breakdown for every ground (default: the worst)")
  ap.add_argument("--json", action="store_true")
  a = ap.parse_args()
  run = os.path.abspath(a.run)
  runs = chain(run)
  if not runs:
    print(f"没有训练记录：{os.path.join(run, 'records')} 不存在（旧代码的训练，或 U_RECORDS=0）")
    return 1
  cur = runs[-1]
  events = [e for r in runs for e in r["events"]]
  snaps = [s for r in runs for s in r["snaps"]]
  starts = [e for e in events if e.get("ev") == "start"]
  start = starts[-1] if starts else {}
  if not snaps:
    print("记录里还没有快照（训练刚开始，或在第一次迭代前就结束了）")
  last = snaps[-1] if snaps else {}
  it_last = last.get("it", 0)
  win = [s for s in snaps if s.get("it", 0) > it_last - a.last]
  items = start.get("items") or []
  edges = start.get("where_edges") or {}
  problems: list[tuple[int, str, list[str]]] = []   # (rank, what, evidence): the smaller rank first
  out_json: dict = {"run": run, "it": it_last}
  P = print if not a.json else (lambda *x, **k: None)

  # ---- 概况 ----
  end = next((e for e in reversed(cur["events"]) if e.get("ev") == "end"), None)
  exc = [e for e in cur["events"] if e.get("ev") == "exception"]
  status = cur["status"] or {}
  age = time.time() - status["t"] if num(status.get("t")) is not None else None
  P("== 概况 ==")
  P(f"目录 {run}")
  P(f"类型 {start.get('kind', '-')}  代码 {start.get('code', '-')}  机器 {start.get('host', '-')} (pid "
    f"{start.get('pid', '-')})  机器人 {start.get('num_envs', '-')} 个（评估 {start.get('eval_envs', '-')} 个）  "
    f"GPU {start.get('gpu', '-')}")
  seg = "；".join(f"{x.get('from')}-{x.get('to')} 用 {x.get('commit')}" for x in (start.get("segments") or []) if x)
  if seg:
    P(f"各段代码：{seg}")
  for r in runs[:-1]:
    P(f"续训自 {r['run']}（它的记录一并读入" + (f"，只算到续训点迭代 {r['cut']}；它后来跑到 {r['it_max']} 的那一段不算"
                                         if r.get("it_max", 0) > r.get("cut", 10**9) else "") + "）")
  res = start.get("resumed")
  if res and len(runs) == 1:
    P(f"续训自检查点 {res.get('checkpoint')}（迭代 {res.get('it')}；那次训练的记录不在本机，未读入）")
  if snaps:
    t0, t1 = snaps[0].get("t"), last.get("t")
    fps = mean([(s.get("perf") or {}).get("fps") for s in win])
    P(f"迭代 {snaps[0].get('it')} → {it_last}" + (f"（计划到 {status.get('total_it')}）" if status.get("total_it") else "")
      + (f"，记录跨度 {(t1 - t0) / 3600.0:.1f} 小时" if num(t0) is not None and num(t1) is not None else "")
      + (f"，速度 {fps:.0f} 步/秒" if fps else ""))
  if end is not None:
    how = {"done": "正常结束", "exception": "异常退出", "interrupt": "被中断（Ctrl-C）", "exit": "进程退出（未跑完）",
           "stopped": "被停止（kill，例如另一个种子通过后由 teacher.sh 停掉，或人工停止）"}
    P(f"结束方式：{how.get(end.get('reason'), end.get('reason'))}（迭代 {end.get('it')}）")
    if end.get("reason") != "done":
      ev = [f"结束方式 {end.get('reason')}，迭代 {end.get('it')}"]
      if exc:
        ev.append(f"异常：{exc[-1].get('error')}")
        ev += ["  " + x for x in (exc[-1].get("trace") or "").strip().splitlines()[-6:]]
      problems.append((0, "训练没有跑完", ev))
  elif age is not None and age > 900:
    P(f"结束方式：没有结束记录，最后一次心跳在 {age / 60.0:.0f} 分钟前（迭代 {status.get('it')}）——进程被杀或硬崩溃")
    ev = [f"最后心跳：迭代 {status.get('it')}，{age / 60.0:.0f} 分钟前；timeline 里没有 end 事件"]
    if cur["crash"]:
      ev.append("crash_traceback.txt 末尾：")
      ev += ["  " + x for x in cur["crash"].splitlines()[-8:]]
    else:
      ev.append("crash_traceback.txt 是空的：不是 Python 层的崩溃，多半是被 kill -9 / 机器重启 / 显存不足被系统杀掉")
    problems.append((0, "训练中途停止，没有留下结束记录", ev))
  else:
    P("结束方式：还在训练" + (f"（最后心跳 {age:.0f} 秒前，迭代 {status.get('it')}）" if age is not None else ""))
  if exc and end is None:
    problems.append((0, "记录里有异常", [f"{exc[-1].get('error')}"]
                     + ["  " + x for x in (exc[-1].get("trace") or "").strip().splitlines()[-6:]]))
  faults = [e for e in cur["events"] if e.get("ev") == "fault"]
  for e in faults:
    P(f"记录功能自身出错（不影响训练）：{e.get('where')} @ 迭代 {e.get('it')}")
    problems.append((8, f"训练记录的 {e.get('where')} 部分出错，之后这部分没有记录（训练不受影响）",
                     ["  " + x for x in (e.get("trace") or "").strip().splitlines()[-4:]]))
  if start.get("episodes", "on") != "on":
    P(f"回合统计：{start.get('episodes')}")

  # ---- 时间线 ----
  P("\n== 时间线 ==")
  lines = []
  for e in events:
    ev, it, name = e.get("ev"), e.get("it"), e.get("name")
    if ev == "start":
      r = e.get("resumed")
      lines.append((it, f"开始训练（代码 {e.get('code')}" + (f"，续自迭代 {r.get('it')}" if r else "，从零") + "）"))
    elif ev == "milestone" and str(name).startswith("expect:"):
      by = e.get("mark")
      late = f"，比期限晚 {it - by} 迭代" if num(by) is not None and num(it) is not None and it > by else ""
      lines.append((it, f"阶段预期达到：{EXPECT_ZH.get(name[7:], e.get('what'))}（期限迭代 {f(by)}{late}）"))
    elif ev == "milestone":
      what = {"p": "进度 p", "q": "质量 q", "payload": "负载上限（占满载）"}.get(name) or \
        (f"{name[4:]} 行数" if str(name).startswith("row_") else name)
      lines.append((it, f"{what} 过 {f(e.get('mark'))}"))
    elif ev == "constraint_on":
      lines.append((it, f"约束 {name} {'开启' if e.get('state') else '关闭'}"))
    elif ev == "q_gate":
      lines.append((it, f"质量阶段：{GATE_ZH.get(e.get('state'), e.get('state'))}（爬楼行 {f(e.get('up'))}，"
                        + (f"上楼失败率 {f(e.get('up_fail'))}" if "up_fail" in e else
                           f"评估机器人固定行上楼失败率 {f((e.get('eval_fail') or {}).get('up'))}")
                        + (f"，q {f(e.get('q'))}，轻载固定行失败率高出参考 {f(e.get('gap'))}"
                           if e.get("gap") is not None else "") + "）"))
    elif ev == "clock_slow":
      lines.append((it, f"进度时钟{'减速到 1/4' if e.get('state') else '恢复正常'}（楼梯以外地面的失败率 "
                        f"{f(e.get('fail'))}）"))
    elif ev == "relax":
      lines.append((it, f"{name} 地面的约束{'开始放松' if e.get('state') else '恢复'}"
                        f"（{fam_fail_zh(e, name)}）"
                        + (f"，其间反复 {e.get('flips')} 次" if e.get("flips", 0) > 1 else "")))
    elif ev in ("cap", "floor"):
      side = "上限" if ev == "cap" else "下限"
      lines.append((it, f"约束 {name} 的乘子{'到' if e.get('state') else '离开'}{side}（乘子 {f(e.get('scale'))}，读数 "
                        f"{f(e.get('value'))}，目标 {f(e.get('target'))}）"))
    elif ev == "alert":
      lines.append((it, f"{'告警仍在' if e.get('again') else '告警'} {name}：{alert_zh(e)}"))
    elif ev == "clear":
      lines.append((it, f"告警解除 {name}（自迭代 {e.get('since')}）"))
    elif ev == "exception":
      lines.append((it, f"异常：{e.get('error')}"))
    elif ev == "fault":
      lines.append((it, f"记录功能出错：{e.get('where')}"))
    elif ev == "end":
      lines.append((it, f"结束（{e.get('reason')}）"))
  n_ck = sum(1 for e in events if e.get("ev") == "checkpoint")
  last_ck = next((e for e in reversed(events) if e.get("ev") == "checkpoint"), None)
  if len(lines) > a.events:
    P(f"（共 {len(lines)} 条，只列最后 {a.events} 条；--events N 可多列）")
  for it, text in lines[-a.events:]:
    P(f"  {('' if it is None else str(it)).rjust(6)}  {text}")
  P(f"  检查点 {n_ck} 个" + (f"，最新 {last_ck.get('file')}（迭代 {last_ck.get('it')}）" if last_ck else ""))

  # ---- 现状 ----
  c = last.get("cur") or {}
  pol, roll, epi = last.get("policy") or {}, last.get("rollout") or {}, last.get("episode") or {}
  first_win = win[0] if win else {}
  c0 = first_win.get("cur") or {}
  if snaps:
    P(f"\n== 现状（迭代 {it_last}；括号内是 {first_win.get('it')} 时的值）==")
    P(f"进度 p {f(c.get('p'))}（{f(c0.get('p'))}）  质量 q {f(c.get('q'))}（{f(c0.get('q'))}）"
      f"  质量阶段 {GATE_ZH.get(c.get('q_gate'), f(c.get('q_gate')))}  进度时钟 "
      f"{'1/4 速' if c.get('clock_slow') else '正常'}（已走 {f(c.get('clock_it'), 5)} 迭代）")
    rows = c.get("rows") or {}
    P(f"地形行数：平均 {f(c.get('level'))}（{f(c0.get('level'))}），" + "，".join(f"{k} {f(v)}" for k, v in rows.items()))
    fam = c.get("family_fail") or {}
    efail = c.get("eval_fail") or {}
    P(f"失败率：训练机器人 {f(c.get('fail'))}（{f(c0.get('fail'))}），评估机器人 {f(c.get('fail_eval'))}（可见 "
      f"{f(c.get('fail_seen'))} / 遮挡 {f(c.get('fail_blind'))}）；训练机器人按地面 "
      + "，".join(f"{k} {f(v)}" for k, v in fam.items())
      + ("（楼梯是课程推到的前沿行：升到摔为止，约 0.6 属正常，不用来判断）" if efail else ""))
    if efail:
      P("评估机器人在固定楼梯行（验收范围内均匀分布，按均值动作）的失败率："
        + "；".join(f"{k} {f(v)}（按行 {rows_zh(c, k)}）" for k, v in efail.items())
        + ("——楼梯约束的读数来自这些行" if c.get("eval_light") else "——楼梯约束的放松和读数看这个"))
    if c.get("eval_light"):
      P("其中轻载（不超过满载的 1/4）的评估机器人，按台阶高度 5–8 / 8–12 / 12–15 / 15–18 cm 四档：")
      for k, d in c["eval_light"].items():
        P(f"  {k} {f(d.get('fail'))}（参考值 {f(d.get('ref'))}，{d.get('n')} 回合）  可见 {bands_zh(d, 'seen')}  遮挡 "
          f"{bands_zh(d, 'blind')}")
      P("  楼梯约束：轻载失败率比参考值高 0.08 以上放松，回到 0.04 以内恢复（不足 200 回合不判）；老师门槛的 survive_up / "
        "survive_down 读可见、遮挡各自最差的一档")
      cls = c.get("eval_class") or {}
      if cls:
        P("  按负载（只记录）：" + "；".join(
          f"{k} " + "，".join(f"{NAME_ZH.get(x, x)} {f(v.get('fail'))}（{v.get('n')} 回合）" for x, v in d.items())
          for k, d in cls.items()))
    P(f"负载上限 {f(c.get('payload_limit'))} / {f(c.get('payload_full'))} kg  指令上限 {c.get('cmd_max')}")
    P(f"动作噪声 {f(pol.get('std'))}（{f((first_win.get('policy') or {}).get('std'))}；各动作 "
      + "，".join(f"{k} {f(v)}" for k, v in (pol.get("std_by_term") or {}).items())
      + f"）  回合奖励 {f(epi.get('reward'))}（{f((first_win.get('episode') or {}).get('reward'))}）  回合长度 "
        f"{f(epi.get('len'))} 步  学习率 {f((last.get('loss') or {}).get('lr'))}")
    out_json["now"] = {"cur": c, "policy": pol, "episode": epi}
    # 进度停滞
    span = it_last - first_win.get("it", it_last)
    if span >= 300 and num(c.get("p")) is not None and c["p"] < 1.0 and num(c0.get("p")) is not None \
       and c["p"] - c0["p"] < 1e-4:
      problems.append((2, f"进度 p 在最近 {span} 迭代没有前进（停在 {f(c.get('p'))}）",
                       [f"平均行数 {f(c0.get('level'))} → {f(c.get('level'))}，失败率 {f(c.get('fail'))}，进度时钟 "
                        f"{'减速（楼梯以外地面的失败率超过上限）' if c.get('clock_slow') else '正常'}"]))
    if num(c.get("p")) is not None and c["p"] >= 0.5 and (c.get("q") or 0.0) <= 0.0 and c.get("q_gate") in GATE_ZH:
      why = (f"训练机器人的爬楼行数 {f(c.get('up'))} 还没到门槛" if c.get("q_gate") == "waiting_row"
             else f"上楼失败率 {f(fam.get('up'))} 高于上限")
      if efail:
        why += f"；评估机器人固定行上楼失败率 {f(efail.get('up'))}（按行 {rows_zh(c, 'up')}）"
      problems.append((2, "进度已过半，质量阶段还没开始", [why]))
    elif 0.0 < (c.get("q") or 0.0) < 1.0 and span >= 300 and num(c0.get("q")) is not None and c["q"] - c0["q"] < 1e-4:
      problems.append((2, f"质量阶段停在 q {f(c.get('q'))}（最近 {span} 迭代没动）",
                       [f"质量阶段状态：{GATE_ZH.get(c.get('q_gate'), c.get('q_gate'))}；爬楼行 {f(c.get('up'))}，"
                        + (f"评估机器人固定行上楼失败率 {f(efail.get('up'))}" if efail else
                           f"上楼失败率 {f(fam.get('up'))}")]))
    slow = [bool((s.get("cur") or {}).get("clock_slow")) for s in win]
    if slow and sum(slow) / len(slow) > 0.5 and (c.get("p") or 0.0) < 1.0:
      problems.append((3, f"进度时钟在最近 {a.last} 迭代里有 {pct(sum(slow), len(slow))} 的时间在减速",
                       [f"楼梯以外地面的失败率 {f(c.get('clock_fail', c.get('fail')))}（上限见 "
                        f"curriculum.MAX_FAIL_RATE）"]))
    s0 = (first_win.get("policy") or {}).get("std")
    if num(pol.get("std")) is not None and num(s0) is not None and pol["std"] > 1.25 * s0 and pol["std"] > 0.5:
      problems.append((1, f"动作噪声在涨：{f(s0)} → {f(pol['std'])}（最近 {span} 迭代）",
                       ["各动作：" + "，".join(f"{k} {f(v)}" for k, v in (pol.get("std_by_term") or {}).items()),
                        f"动作到截断值的比例 {f(roll.get('action_clipped'))}，动作最大绝对值 {f(roll.get('action_max'))}"]))

  # ---- 阶段预期 ----
  expect = start.get("expect") or []
  reached = reached_milestones(events)
  running = end is None and not (age is not None and age > 900)
  # iterations per hour: over this run's own snapshots of the window (a resume's gap is not training time)
  own = [s for s in cur["snaps"] if s.get("it", 0) > it_last - a.last and num(s.get("t")) is not None]
  it_per_h = None
  if len(own) >= 2 and own[-1]["t"] > own[0]["t"] and own[-1]["it"] > own[0]["it"]:
    it_per_h = (own[-1]["it"] - own[0]["it"]) / (own[-1]["t"] - own[0]["t"]) * 3600.0
  hours = lambda its: "" if not it_per_h else f"（约 {max(its, 0) / it_per_h:.1f} 小时后）"   # noqa: E731
  if expect and snaps:
    P("\n== 阶段预期（参考进度表：到期限还没达到的版本，现在就该查，不用等跑完）==")
    rows_out, nxt = [], None
    for e in expect:
      name, by = e["name"], e["by"]
      got = reached.get(name)
      zh = EXPECT_ZH.get(name, e.get("what"))
      if got is not None:
        state = f"达到于迭代 {got}" + (f"（晚 {got - by}）" if got > by else "")
      elif it_last >= by:
        state = f"未达到，已过期限 {it_last - by} 迭代：{expect_now(name, c)}"
        problems.append((1, f"阶段预期落后：{zh}（期限迭代 {by}，到迭代 {it_last} 还没达到）",
                         [f"现在：{expect_now(name, c)}", "这一版在这一步比参考进度慢：看下面这块地面 / 这项约束的记录找原因"]))
      else:
        state = f"还没到期限：{expect_now(name, c)}"
        nxt = nxt or (name, by, zh)
      P(f"  期限 {by:>5}  {zh}：{state}")
      rows_out.append({"name": name, "by": by, "reached": got, "late": got is None and it_last >= by})
    if nxt and running:
      P(f"  下一个期限：迭代 {nxt[1]}{hours(nxt[1] - it_last)}——{nxt[2]}")
    out_json["expect"] = rows_out

  # ---- 进度预估 ----
  ck = start.get("clock") or {}
  if snaps and ck and start.get("kind", "teacher") == "teacher" and num(c.get("p")) is not None:
    P("\n== 进度预估（按进度时钟；楼梯以外地面的失败率超过上限时时钟只走 1/4 速）==")
    span_it = it_last - first_win.get("it", it_last)
    pace = None
    if span_it > 0 and num(c.get("clock_it")) is not None and num(c0.get("clock_it")) is not None:
      pace = min(1.0, max(0.0, (c["clock_it"] - c0["clock_it"]) / span_it))
    P("  速度 " + (f"{it_per_h:.0f} 迭代/小时" if it_per_h else "-") + "；进度时钟最近 "
      + ("-" if pace is None else f"{pace:.2f} 倍速") + f"（已走 {f(c.get('clock_it'), 5)} 迭代）")
    eta: dict = {}
    if c["p"] >= 1.0:
      eta["p"] = it_last
      P("  p = 1：已到")
    elif num(c.get("clock_it")) is not None:
      left = max(0.0, ck["start"] + ck["span"] - c["clock_it"])
      use = max(pace if pace else 1.0, 0.25)
      eta["p"] = int(it_last + left / use)
      P(f"  p = 1：按最近的时钟速度约在迭代 {eta['p']}{hours(eta['p'] - it_last)}；时钟全速最早迭代 {int(it_last + left)}")
    q = c.get("q") or 0.0
    if q >= 1.0:
      eta["q"] = it_last
      P("  q = 1：已到")
    elif c.get("q_gate") == "running" or q > 0.0:
      eta["q"] = int(it_last + (1.0 - q) * ck["q_span"])
      P(f"  q = 1：质量阶段在走，约在迭代 {eta['q']}{hours(eta['q'] - it_last)}（爬楼行数掉到门槛以下时会停）")
    else:
      eta["q"] = int(it_last + ck["q_span"])
      P(f"  q = 1：质量阶段还没开始（{GATE_ZH.get(c.get('q_gate'), f(c.get('q_gate')))}：爬楼行 {f(c.get('up'))} / 门槛 "
        f"{f(ck.get('q_row'))}）；现在开始也要到迭代 {eta['q']}")
    if "p" in eta and "q" in eta:
      eta["gate"] = max(eta["p"], eta["q"]) + int(ck.get("gate_hold", 500))
      P(f"  门槛最早可过：迭代 {eta['gate']}{hours(eta['gate'] - it_last)}（p、q 到 1 后各约束还要在窗口内达标 "
        f"{ck.get('gate_hold', 500)} 迭代）")
    out_json["eta"] = {**eta, "it_per_hour": it_per_h, "clock_pace": pace}

  # ---- 各地面的回合 ----
  eps = ep_sum(win)
  out_json["episodes"] = eps
  if eps:
    P(f"\n== 各地面的回合（迭代 {first_win.get('it')} → {it_last}）==")
    P(f"  {'地面':<20}{'回合':>8}{'失败':>7}  原因 / 碰地的部位"
      + ("（楼梯地面的失败率是课程前沿行上的，约 0.6 属正常；要看的是评估机器人固定行的）"
         if any(d.get("family") in STAIRS for d in eps.values()) else ""))
    for g, d in sorted(eps.items(), key=lambda kv: -kv[1]["fail"] / max(kv[1]["n"], 1)):
      cause = "，".join(f"{k} {pct(v, d['fail'])}" for k, v in d["cause"].most_common(3))
      body = "，".join(f"{k} {pct(v, d['fail'])}" for k, v in d["bodies"].most_common(3))
      fixed = d["role"].get("eval") if d.get("family") in STAIRS else None
      P(f"  {g:<20}{d['n']:>8}{pct(d['fail'], d['n']):>7}  {cause or '-'}" + (f"；部位：{body}" if body else "")
        + (f"；评估机器人固定行 {pct(fixed[1], fixed[0])}（{fixed[0]}）" if fixed and fixed[0] else ""))
    order = sorted(eps.items(), key=lambda kv: -kv[1]["fail"] / max(kv[1]["n"], 1))
    detail = {g for g, d in order if a.all or d["fail"] / max(d["n"], 1) >= 0.15} | {g for g, _ in order[:3]}
    P("  ——失败率细分（括号内是回合数" + ("" if a.all or len(detail) == len(order)
                                   else f"；只列失败最多的 {len(detail)} 块地面，--all 列全部") + "）——")
    for g, d in order:
      if g not in detail:
        continue
      parts = []
      for k, zh in CLASS_ZH.items():
        cls = d[k]
        if cls:
          parts.append(zh + " " + " ".join(f"{NAME_ZH.get(x, x)} {pct(v[1], v[0])}({v[0]})" for x, v in cls.items()))
      rows = " ".join(f"{r}:{pct(v[1], v[0])}({v[0]})" for r, v in sorted(d["row"].items(), key=lambda kv: int(kv[0])))
      P(f"  {g}：{'；'.join(parts)}")
      P(f"      按行 {rows}")
      if d["fail"] > 0 and d["where"]:
        where = "；".join(f"{WHERE_ZH[k][0]} {hist_text(h, edges.get(k, []), WHERE_ZH[k][1])}"
                          for k, h in d["where"].items() if k in WHERE_ZH and edges.get(k))
        P(f"      失败回合结束在：{where}")
    for g, d in order:
      # 问题：某地面失败多。A stairs ground is judged on the evaluation robots' episodes on their fixed rows, and
      # from stairs quality 0.5 on (diagnostics' fail_high): at the curriculum's rows ~0.6 of the episodes fail
      # whatever the policy climbs, and before the quality stage the reference schedule judges the climbing.
      stairs = d.get("family") in STAIRS
      if stairs and (c.get("q") or 0.0) < 0.5:
        continue
      n_j, f_j = d["role"].get("eval", [0, 0]) if stairs else (d["n"], d["fail"])
      rate = f_j / max(n_j, 1)
      if n_j >= (100 if stairs else 200) and rate > 0.3:
        ev = [(f"评估机器人固定行上 {n_j} 个回合里 {pct(f_j, n_j)} 失败（全部机器人 {d['n']} 个回合里 "
               f"{pct(d['fail'], d['n'])}）" if stairs else f"{d['n']} 个回合里 {pct(d['fail'], d['n'])} 失败")
              + "；原因 " + "，".join(f"{k} {pct(v, d['fail'])}" for k, v in d["cause"].most_common(2))]
        if d["bodies"]:
          ev.append("碰地的部位：" + "，".join(f"{k} {pct(v, d['fail'])}" for k, v in d["bodies"].most_common(3)))
        worst = sorted(((v[1] / v[0], r, v[0]) for r, v in d["row"].items() if v[0] >= 30), reverse=True)[:3]
        if stairs and rows_zh(c, d["family"]):
          ev.append(f"评估机器人各固定行的失败率：{rows_zh(c, d['family'])}")
        elif worst:
          ev.append("失败最多的行：" + "，".join(f"第 {r} 行 {x:.0%}（{n} 回合）" for x, r, n in worst))
        for k, zh in CLASS_ZH.items():
          cls = {x: v for x, v in d[k].items() if v[0] >= 30}
          if len(cls) >= 2:
            rates = {x: v[1] / v[0] for x, v in cls.items()}
            hi, lo = max(rates, key=rates.get), min(rates, key=rates.get)
            if rates[hi] - rates[lo] > 0.15:
              ev.append(f"按{zh}：{NAME_ZH.get(hi, hi)} {rates[hi]:.0%}，{NAME_ZH.get(lo, lo)} {rates[lo]:.0%}")
        if d["where"] and edges:
          ev.append("失败回合结束在：" + "；".join(
            f"{WHERE_ZH[k][0]}中位数约 {f(hist_quantile(h, edges[k]))}{WHERE_ZH[k][1]}"
            for k, h in d["where"].items() if k in ("dist", "sec", "pitch") and edges.get(k)))
        problems.append((2 if rate > 0.5 else 4,
                         f"{g} 地面失败率高（{'评估机器人固定行上 ' if stairs else ''}{rate:.0%}）", ev))
    # (off the stairs: there both kinds of robots stand on the curriculum's rows; the evaluation robots' stairs
    # rows are fixed)
    flat = [d for d in eps.values() if d.get("family") not in STAIRS]
    all_n = {k: sum(d["role"].get(k, [0, 0])[0] for d in flat) for k in ("train", "eval")}
    all_f = {k: sum(d["role"].get(k, [0, 0])[1] for d in flat) for k in ("train", "eval")}
    if all_n["train"] >= 200 and all_n["eval"] >= 200:
      rt, re_ = all_f["train"] / all_n["train"], all_f["eval"] / all_n["eval"]
      if rt - re_ > 0.15:
        problems.append((4, f"楼梯以外的地面上，带探索噪声的训练机器人比按均值动作的评估机器人多摔 {rt - re_:.0%}",
                         [f"训练机器人 {rt:.0%}（{all_n['train']} 回合），评估机器人 {re_:.0%}（{all_n['eval']} 回合）："
                          f"是噪声在摔，策略均值本身没那么差；看动作噪声是否偏大"]))
    # ---- 负载 ----
    # (the user, 2026-11-12: with a payload the robot must not fall, wherever the payload is mounted)
    tot = lambda key, names: {k: [sum(d[key].get(k, [0, 0])[i] for d in eps.values()) for i in (0, 1)]   # noqa: E731
                              for k in names}
    pay, place = tot("pay", ("light", "mid", "heavy")), tot("load", ("centre", "front", "back", "side"))
    if sum(v[0] for v in pay.values()) > 0:
      share = lambda v: f"{pct(v[1], v[0])}（{v[0]} 回合）" if v[0] else "无回合"   # noqa: E731
      P(f"\n== 负载（同一窗口；现在的负载上限 {f(c.get('payload_limit'))} / 满载 {f(c.get('payload_full'))} kg）==")
      P("  失败率按负载：" + "，".join(f"{NAME_ZH[k]} {share(v)}" for k, v in pay.items()))
      if sum(v[0] for v in place.values()) > 0:
        P("  失败率按负载位置（中、重负载的机器人，机身质心被负载带到哪边）：" + "，".join(
          f"{NAME_ZH[k]} {share(v)}" for k, v in place.items()))
      lo, hi = pay["light"], [pay["mid"][i] + pay["heavy"][i] for i in (0, 1)]
      by_ground = []
      for g, d in order:
        gl = d["pay"].get("light", [0, 0])
        gh = [d["pay"].get("mid", [0, 0])[i] + d["pay"].get("heavy", [0, 0])[i] for i in (0, 1)]
        if gl[0] >= 100 and gh[0] >= 100:
          by_ground.append((gh[1] / gh[0] - gl[1] / gl[0], g, gh, gl))
      by_ground.sort(reverse=True)
      if by_ground:
        P("  带负载比轻载多摔（各地面）：" + "，".join(f"{g} {x:+.0%}" for x, g, _, _ in by_ground))
      spread = []   # where the payload sits: over all grounds, and on each ground
      for g, cls in [("全部地面", place)] + [(g, eps[g]["load"]) for _, g, _, _ in by_ground]:
        rates = {k: v[1] / v[0] for k, v in cls.items() if v[0] >= 100}
        if len(rates) >= 2:
          k_hi, k_lo = max(rates, key=rates.get), min(rates, key=rates.get)
          spread.append((rates[k_hi] - rates[k_lo], g, k_hi, rates[k_hi], k_lo, rates[k_lo]))
      spread.sort(reverse=True)
      for x, g, k_hi, r_hi, k_lo, r_lo in spread[:3]:
        if x > 0.1 and g != "全部地面":
          P(f"  {g} 按负载位置：" + "，".join(f"{NAME_ZH[k]} {share(v)}" for k, v in eps[g]["load"].items()))
      out_json["payload"] = {"by_class": pay, "by_place": place,
                             "by_ground": [{"ground": g, "more": x} for x, g, _, _ in by_ground]}
      ev = []
      if lo[0] >= 200 and hi[0] >= 200 and hi[1] / hi[0] - lo[1] / lo[0] > 0.1:
        ev.append(f"带负载（中、重）{pct(hi[1], hi[0])}（{hi[0]} 回合），轻载 {pct(lo[1], lo[0])}（{lo[0]} 回合）")
      worst = [(x, g, gh, gl) for x, g, gh, gl in by_ground if x > 0.1]
      if worst:
        ev.append("差得最多的地面：" + "，".join(f"{g} 带负载 {pct(gh[1], gh[0])} / 轻载 {pct(gl[1], gl[0])}"
                                           for _, g, gh, gl in worst[:3]))
      if spread and spread[0][0] > 0.1:
        x, g, k_hi, r_hi, k_lo, r_lo = spread[0]
        ev.append(f"负载位置（{g}）：{NAME_ZH[k_hi]} {r_hi:.0%}，{NAME_ZH[k_lo]} {r_lo:.0%}——{NAME_ZH[k_hi]}的负载最容易摔")
      if ev:
        problems.append((2, "带负载的机器人摔得比轻载的多（要求：带负载不能摔）",
                         ev + [f"现在的负载上限 {f(c.get('payload_limit'))} kg（满载 {f(c.get('payload_full'))} kg）"]))
  elif snaps:
    P("\n== 各地面的回合 ==\n  这段时间没有回合记录")

  # ---- 约束 ----
  cons_last = last.get("cons") or {}
  relax = last.get("relax") or {}
  table = []
  for cdef in items:
    name = cdef["name"]
    d = cons_last.get(name)
    if d is None:
      continue
    modes = [m for m in ("all",) + MODES if m in d]
    vals = {m: mean([(s.get("cons") or {}).get(name, {}).get(m) for s in win]) for m in modes}
    third = max(1, len(win) // 3)   # the window's first and last third: which way the reading moves
    v0 = {m: mean([(s.get("cons") or {}).get(name, {}).get(m) for s in win[:third]]) for m in modes}
    v1 = {m: mean([(s.get("cons") or {}).get(name, {}).get(m) for s in win[-third:]]) for m in modes}
    bar, sense = cdef.get("target"), cdef.get("sense")
    ok = lambda v, t: v is not None and t is not None and (v >= t if sense == "min" else v <= t)   # noqa: E731
    judged = [m for m in modes if m == "all" or m in (cdef.get("modes") or MODES)]
    miss = [m for m in judged if bar is not None and vals.get(m) is not None and not ok(vals[m], bar)]
    none = [m for m in judged if vals.get(m) is None]
    capped = num(d.get("scale")) is not None and num(cdef.get("max_scale")) is not None \
      and d["scale"] >= cdef["max_scale"] - 1e-6
    state = []
    if not d.get("on"):
      state.append("未开")
    if bar is None:
      state.append("只记录")
    elif not cdef.get("enforce", True):
      state.append("不强制")
    if none and d.get("on"):
      state.append("无读数(" + ",".join(NAME_ZH.get(m, m) for m in none) + ")")
    if miss:
      state.append("未达标(" + ",".join(NAME_ZH.get(m, m) for m in miss) + ")")
    elif bar is not None and not none:
      state.append("达标")
    if capped and d.get("on"):
      state.append("到上限")
    gate_only = bool(cdef.get("enforce", True)) and bar is not None and cdef.get("terms") == []
    if gate_only:
      state.append("只进门槛")
    elif relax.get(cdef.get("family", "other")) and d.get("on"):
      state.append("放松中")
    table.append({"name": name, "gate_only": gate_only, "vals": vals, "v0": v0, "v1": v1, "target": d.get("target"), "bar": bar,
                  "sense": sense, "max_scale": cdef.get("max_scale"),
                  "scale": d.get("scale"), "state": state, "miss": miss, "none": none, "capped": capped,
                  "on": d.get("on"), "enforce": cdef.get("enforce", True), "family": cdef.get("family", "other"),
                  "judged": judged})
  out_json["constraints"] = table
  if table:
    P(f"\n== 约束（读数是最近 {a.last} 迭代的平均；min 要 ≥ 门槛，max 要 ≤ 门槛）==")
    P(f"  {'约束':<16}{'方向':<5}{'可见':>9}{'遮挡':>9}{'老师目标':>9}{'门槛':>9}{'乘子':>8}  状态；窗口前段 → 后段的读数")
    win_span = it_last - first_win.get("it", it_last)
    for r in table:
      v = r["vals"]
      seen = v.get("seen", v.get("all"))
      moved = "，".join(f"{NAME_ZH.get(m, '')}{f(r['v0'][m])}→{f(r['v1'][m])}" for m in r["vals"]
                       if r["v0"].get(m) is not None and r["v1"].get(m) is not None)
      P(f"  {r['name']:<16}{r['sense'] or '-':<5}{f(seen):>10}{f(v.get('blind')):>10}{f(r['target']):>11}"
        f"{f(r['bar']):>10}{f(r['scale']):>9}  {' '.join(r['state'])}" + (f"；{moved}" if moved else ""))
    for r in table:
      # outside the teacher's current target (by a tenth of it or more) and not getting closer over the window
      t = r["target"]
      if not (r["on"] and r["enforce"] and num(t) is not None and win_span >= 300) or (r["capped"] and r["miss"]) \
         or r["gate_only"]:
        continue
      stuck = []
      for m in r["judged"]:
        a0, a1 = r["v0"].get(m), r["v1"].get(m)
        if a0 is None or a1 is None:
          continue
        gap0, gap1 = ((t - a0, t - a1) if r["sense"] == "min" else (a0 - t, a1 - t))
        if gap1 > 0.1 * abs(t) and gap1 >= 0.95 * gap0:
          stuck.append(f"{NAME_ZH.get(m, m)} {f(a0)} → {f(a1)}")
      if stuck:
        problems.append((4, f"约束 {r['name']} 落在老师当前目标之外，最近 {win_span} 迭代没有靠近",
                         ["读数 " + "，".join(stuck) + f"；老师当前目标 {f(t)}，最终门槛 {f(r['bar'])}（{r['sense']}），乘子 "
                          f"{f(r['scale'])}（上限 {f(r['max_scale'])}）"
                          + ("；这块地面的约束正在放松（任务本身在失败）" if relax.get(r["family"]) else "")]))
    for r in table:
      if not (r["on"] and r["enforce"] and r["bar"] is not None):
        continue
      if r["capped"] and r["miss"]:
        problems.append((3, f"约束 {r['name']} 的乘子已到上限，读数仍没到门槛",
                         [f"读数 " + " / ".join(f"{NAME_ZH.get(m, m)} {f(r['vals'][m])}" for m in r["vals"])
                          + f"，门槛 {f(r['bar'])}（{r['sense']}），乘子 {f(r['scale'])}：加大惩罚已经没用，要改奖励或放宽"]))
      if r["gate_only"] and r["miss"]:
        fam = r["family"]
        d = (c.get("eval_light") or {}).get(fam) or {}
        problems.append((3, f"门槛项 {r['name']} 没达标：轻载机器人在{'上' if fam == 'up' else '下'}楼最差一档台阶上摔得太多",
                         ["最差一档失败率 " + " / ".join(f"{NAME_ZH.get(m, m)} {f(r['vals'][m])}" for m in r["vals"])
                          + f"，门槛 {f(r['bar'])}；按档（5–8 / 8–12 / 12–15 / 15–18 cm）可见 {bands_zh(d, 'seen')}，遮挡 "
                            f"{bands_zh(d, 'blind')}：这一项没有奖励项和乘子，只进老师门槛，要靠课程和其他约束把它降下来"]))
      if r["none"]:
        problems.append((5, f"约束 {r['name']} 已开启但没有读数（{','.join(r['none'])}）",
                         ["没有读数的约束不会调乘子，门槛检查也会判未测到：看对应地面有没有机器人、对应指令有没有发"]))
      v = r["vals"]
      if "blind" in r["judged"] and "seen" in r["judged"] and "blind" in r["miss"] and "seen" not in r["miss"] \
         and v.get("seen") is not None:
        problems.append((5, f"约束 {r['name']}：可见模式达标，遮挡模式没达标",
                         [f"可见 {f(v.get('seen'))}，遮挡 {f(v.get('blind'))}，门槛 {f(r['bar'])}：盲走学生会卡在这一项"]))
  for fam, on in relax.items():
    held = [bool((s.get("relax") or {}).get(fam)) for s in win]
    # (a family none of whose constraints is on holds nothing back: before the quality stage the evaluation robots
    # fail the stairs' higher rows as a matter of course)
    if on and held and sum(held) / len(held) > 0.8 and any(r["on"] and r["family"] == fam for r in table):
      rows = "；".join(f"{k} 按行 {rows_zh(c, k)}" for k in (("up", "down") if fam == "stairs" else (fam,))
                      if fam in STAIRS and rows_zh(c, k))
      problems.append((3, f"{fam} 地面的约束一直在放松（最近 {a.last} 迭代里 {pct(sum(held), len(held))}）",
                       [f"{fam_fail_zh(c, fam)}"
                        + ("（比参考值高 0.08 以上放松，回到 0.04 以内恢复）：约束收紧后摔得比之前多，约束乘子不往上加"
                           if light_zh(c, fam) else "（放松线 0.3，恢复线 0.25）：任务本身还没学会，约束乘子不往上加")
                        + (f"；{rows}" if rows else "")]))

  # ---- 门槛 ----
  if start.get("kind", "teacher") == "teacher" and table:
    hold = [s for s in snaps if s.get("it", 0) > it_last - 500]
    p_ok = all(((s.get("cur") or {}).get("p") or 0.0) >= 1.0 for s in hold)
    q_ok = all(((s.get("cur") or {}).get("q") or 0.0) >= 1.0 for s in hold)
    miss, unmeasured = [], []
    for r in table:
      if not r["enforce"] or r["bar"] is None:
        continue
      for m in r["judged"]:
        got = [(s.get("cons") or {}).get(r["name"], {}).get(m) for s in hold]
        got = [x for x in got if num(x) is not None]
        if len(got) < len(hold) // 2 or not got:
          unmeasured.append(f"{r['name']}_{m}")
          continue
        v = sum(got) / len(got)
        if not (v >= r["bar"] if r["sense"] == "min" else v <= r["bar"]):
          miss.append(f"{r['name']}_{m} {f(v)} vs {f(r['bar'])}")
    P("\n== 门槛（teacher_gate.py 的规则，按最近 500 迭代的快照估算；正式结论以 teacher_gate.py 为准）==")
    if any(r["gate_only"] for r in table):
      P("  （只进门槛的项：这里按开训时记录的门槛算，teacher_gate.py 按部署代码里现在的门槛判）")
    P(f"  课程走完 p=1：{f(p_ok)}  质量阶段走完 q=1：{f(q_ok)}")
    P("  未达标：" + ("；".join(miss) if miss else "无"))
    if unmeasured:
      P("  未测到：" + "；".join(unmeasured))
    P("  估算结论：" + ("会通过" if p_ok and q_ok and not miss and not unmeasured else "还不会通过"))
    out_json["gate"] = {"p": p_ok, "q": q_ok, "miss": miss, "unmeasured": unmeasured}

  # ---- 优化与物理 ----
  if snaps:
    g, ph, perf = last.get("grad") or {}, last.get("phys") or {}, last.get("perf") or {}
    P(f"\n== 优化与物理（迭代 {it_last}）==")
    P(f"  价值网络解释方差 {f(roll.get('ev'))}  优势标准差 {f(roll.get('adv_std'))}  "
      f"熵系数 {roll.get('entropy_coef', '-')}  损失 "
      + " ".join(f"{k} {f(v)}" for k, v in (last.get("loss") or {}).items()))
    P(f"  梯度范数（裁剪后）actor {f(g.get('actor'))} / critic {f(g.get('critic'))}，被裁剪的更新占 actor "
      f"{f(g.get('actor_clipped'))} / critic {f(g.get('critic_clipped'))}")
    P(f"  动作：平均绝对值 {f(roll.get('action_abs'))}，最大 {f(roll.get('action_max'))}，到截断值的占 "
      f"{f(roll.get('action_clipped'))}")
    P("  力矩到限的占比：" + ("，".join(f"{k[4:]} {f(v)}" for k, v in ph.items() if k.startswith("sat_")) or "-")
      + f"；关节到软限位 {f(ph.get('joint_limit'))}；关节速度 p99 {f(ph.get('joint_vel_p99'))} rad/s；机身倾角 p90 "
        f"{f(ph.get('tilt_p90'))}°"
      + (f"；着地的脚在滑动的占 {f(ph.get('foot_sliding'))}" if "foot_sliding" in ph else ""))
    P(f"  速度 {f(perf.get('fps'), 5)} 步/秒（采样 {f(perf.get('collect_s'))} s + 更新 {f(perf.get('learn_s'))} s）；显存 "
      f"{f(perf.get('gpu_mb'), 5)} MB（峰值 {f(perf.get('gpu_peak_mb'), 5)}），内存峰值 {f(perf.get('rss_peak_mb'), 5)} MB")
    ev_now = mean([(s.get("rollout") or {}).get("ev") for s in win[-5:]])
    if ev_now is not None and ev_now < 0.1 and it_last >= 300:
      problems.append((5, f"价值网络几乎解释不了回报（解释方差 {f(ev_now)}）",
                       ["优势估计接近噪声，策略更新没有方向：看奖励是否刚被课程 / 乘子大幅改动，或价值损失是否在涨"]))
    for k, v in ph.items():
      if k.startswith("sat_") and num(v) is not None and v > 0.2:
        problems.append((5, f"{k[4:]} 关节有 {v:.0%} 的力矩顶在上限", ["电机能力不够或动作过猛：看负载、台阶高度、动作噪声"]))
    if num(roll.get("action_clipped")) is not None and roll["action_clipped"] > 0.05:
      problems.append((3, f"{roll['action_clipped']:.0%} 的动作分量顶在截断值",
                       [f"动作最大绝对值 {f(roll.get('action_max'))}，噪声 {f(pol.get('std'))}：动作在自我放大的前兆"]))
    fps_all = sorted(x for x in ((s.get("perf") or {}).get("fps") for s in snaps) if num(x) is not None)
    if fps_all and num(perf.get("fps")) is not None and perf["fps"] < 0.5 * fps_all[len(fps_all) // 2]:
      problems.append((6, f"训练速度掉到平时的一半以下（{f(perf['fps'], 5)} 对 {f(fps_all[len(fps_all) // 2], 5)} 步/秒）",
                       ["看这台机器上是否有别的任务在占 GPU / CPU"]))

  # ---- 告警 ----
  alerts: dict[str, dict] = {}
  for e in events:
    if e.get("ev") == "alert":
      x = alerts.setdefault(e["name"], {"since": e.get("since", e.get("it")), "msg": alert_zh(e), "end": None, "n": 0})
      if not e.get("again"):
        x.update(since=e.get("since", e.get("it")), end=None)
      x["msg"] = alert_zh(e)
      x["n"] += 1
    elif e.get("ev") == "clear" and e.get("name") in alerts:
      alerts[e["name"]]["end"] = e.get("it")
  if end is not None or (age is not None and age > 900):
    active = [n for n, x in alerts.items() if x["end"] is None]
  else:
    active = list(status.get("alerts") or [n for n, x in alerts.items() if x["end"] is None])
  P("\n== 告警 ==")
  if not alerts:
    P("  没有告警")
  for n, x in sorted(alerts.items(), key=lambda kv: (kv[1]["since"] or 0)):
    span = f"迭代 {x['since']} 起" + (f"，到 {x['end']} 解除" if x["end"] is not None else "，仍在")
    P(f"  {n}（{span}）：{x['msg']}")
    if x["end"] is None and n.split(":")[0] in ("nonfinite", "nan_resets"):
      problems.append((0, f"出现非有限数值（{n}）", [str(x["msg"])]))
    elif x["end"] is None and n.split(":")[0] == "payload_fail":
      problems.append((3, f"{n.split(':')[1]} 地面上带负载的机器人摔得比轻载的多", [str(x["msg"])]))
    elif x["end"] is None and n.split(":")[0] in ("level_falling", "len_drop", "value_loss_spike", "lr_floor",
                                                  "std_high", "no_episodes"):
      problems.append((4, f"告警 {n} 仍在", [str(x["msg"])]))
  out_json["alerts"] = alerts
  out_json["active_alerts"] = active

  # ---- 最可能的问题 ----
  P("\n== 最可能的问题 ==")
  if not problems:
    P("  没有发现明显问题")
  seen_what = set()
  k = 0
  for rank, what, evidence in sorted(problems, key=lambda x: x[0]):
    if what in seen_what:
      continue
    seen_what.add(what)
    k += 1
    P(f"  {k}. {what}")
    for line in evidence:
      if line:
        P(f"       {line}")
  out_json["problems"] = [{"rank": r, "what": w, "evidence": e} for r, w, e in sorted(problems, key=lambda x: x[0])]
  if a.json:
    print(json.dumps(out_json, ensure_ascii=False, default=lambda o: dict(o) if isinstance(o, dict) else str(o)))
  return 0


if __name__ == "__main__":
  sys.exit(main())
