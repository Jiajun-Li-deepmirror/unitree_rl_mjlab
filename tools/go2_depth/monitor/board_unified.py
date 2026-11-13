#!/usr/bin/env python3
"""board_unified.py OUTDIR [--since 2026-11-13_17-15-00] [--manual FILE] -- the progress board's data: the unified
flow's FORMAL trainings only (go2-depth 2026-11-12, the user: "看板重置，只记录united训练"; 2026-11-13, the user:
"重置看板，之后只要放正式训练的数据就行" -- a probe's run, its alerts, its chain-log lines and the notes about probes
are left out; --since is the formal run's start). Writes OUTDIR/board.json for the board artifact
https://claude.ai/artifact/EYUJwttsJF7dW5cQdbjrXY (its page is board_unified.html beside this file; the page's
datasets all read this one file, each through a loader that takes one list):

  robots      one row per robot: where its flow stands (teacher -> blind / depth students -> acceptance -> delivery)
  runs        one row per formal training run started at or after --since (the teacher seeds, later the students)
  milestones  the reference schedule (diagnostics.expectations) of each formal teacher seed: the deadline iteration,
              the clock time the run gets there at its measured speed, reached / open / late, the readings now
  curves      p, q, the climbing row and the fail rates (flat ground; the evaluation robots' stairs rows) over the
              iterations (<= 150 points a run; the fail rates from iteration FAIL_FROM_IT on)
  alerts      one row per alert active now
  events      the runs' timeline events (start, milestones, alerts, faults, ends) and the chain logs' lines, in
              Chinese, newest first

Read over ssh from t1 (Go2) and t2 (Go2W): logs/rsl_rl/<r>_u_*/<run>/records/{status.json,record.jsonl,
timeline.jsonl} (src/tasks/unified/diagnostics.py), logs/jobs/unified_<r>{,_students,_accept}.txt and the pick
files. A run renamed after a void start (_discard ...), a smoke run and a probe (uprobe_*, *_probe_*) are left out by
their names; a chain-log line older than --since, of a smoke run or of a probe (PROBE_LINE) is left out too.
--manual: a JSON file {"events": [{at, machine, robot, run, text}]} of notes to add (default ~/.cache/go2-depth-work/board/manual.json when it exists).

A stairs milestone "reached" before the robots walk is the starting rows' value, not climbing (the 11-12 Go2 run:
"climbing row at 1.5" at iteration 300, with the rows still falling from where they started): it counts here only
from the iteration the walk milestone was reached, judged on the curve (the climbing row only at its deadline).

Standard library only on both sides. Then (the loop's board step): upload board.json to the artifact as an asset,
run `board_unified.py OUTDIR --docs /_blob/<id>` (writes OUTDIR/docs/<dataset>.json, the six dataset documents
pointing at the new blob, from the board.json already in OUTDIR) and write them to the store's `datasets`
collection in one batch, each pinned to the version last seen.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

SSH = "/home/dm/.cache/go2-depth-work/ssh/r"
D = "/root/gpufree-data/go2-depth/depth_fe"
ROBOTS = {"go2": ("Go2", "t1", "30714"), "go2w": ("Go2W", "t2", "31365")}   # key -> (name, box, ssh port)
MANUAL = Path("/home/dm/.cache/go2-depth-work/board/manual.json")
CST = timezone(timedelta(hours=8))
STALE_MIN = 15          # a run whose status is older than this while its process lives: stalled
SPEED_WINDOW_S = 1800   # the speed is measured over the run's last half hour
STAIRS_MILESTONES = ("climb_first", "climb_low", "climb")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "unified"))
import run_report as rr  # noqa: E402  (the report's Chinese wording: alerts, milestones, their readings)

# the reference schedule's milestones (diagnostics.expectations; curriculum.MAX_FAIL_RATE 0.3, Q_ROW 3.0): short
# Chinese name, the curve it is read on, the value to reach there, from which side
MILE = {
  "walk": ("平地会走", "flat", 0.3, "<="), "commands": ("进度 p 到 0.25", "p", 0.25, ">="),
  "climb_first": ("低台阶过半成功", "low", 0.5, "<="), "climb_low": ("上楼行到 1.5", "up", 1.5, ">="),
  "climb": ("上楼行达标，质量阶段开始", "up", 3.0, ">="), "p_done": ("进度 p 到 1", "p", 1.0, ">="),
  "q_done": ("质量 q 到 1", "q", 1.0, ">="),
}
COL = {"p": 2, "q": 3, "up": 4, "low": 5, "flat": 10}   # a measure's place in a run's series rows
FAIL_FROM_IT = 50   # the failure rates' curves start here: before it too few episodes have ended
SLOTS = Path("/home/dm/.cache/go2-depth-work/board/slots.json")   # run -> its colour's place on the charts, kept
KIND_ZH = {"blind": "盲走", "depth": "深度"}

REMOTE = r'''
import glob, json, os, re, subprocess, sys, time
a = json.loads(sys.argv[1])
since = a["since"]
since_log = since[5:10] + " " + since[11:].replace("-", ":")
NAME = re.compile(r"^(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)_((?:u[a-z]+|blind\d|depth\d)(?:_t\d+)?_s\d+)$")
out = {"now": time.time(), "runs": [], "logs": {}, "picks": {}}
def jl(path):
    rows = []
    try:
        with open(path, errors="replace") as f:
            for x in f:
                x = x.strip()
                if x:
                    try:
                        rows.append(json.loads(x))
                    except ValueError:
                        pass
    except OSError:
        pass
    return rows
def low(c):
    v = [x for x in ((c.get("eval_fail_rows") or {}).get("up") or [])[:3] if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else None
for r in a["robots"]:
    for d in sorted(glob.glob("logs/rsl_rl/%s_u_*/*/records/status.json" % r)):
        run_dir = os.path.dirname(os.path.dirname(d))
        m = NAME.match(os.path.basename(run_dir))
        if not m or m.group(1) < since:
            continue
        try:
            st = json.load(open(d))
        except (OSError, ValueError):
            continue
        rec = jl(run_dir + "/records/record.jsonl")
        step = max(1, -(-len(rec) // 150))
        keep = rec[::step] + ([rec[-1]] if rec and (len(rec) - 1) % step else [])
        series = []
        for x in keep:
            c = x.get("cur") or {}
            ef = c.get("eval_fail") or {}
            series.append([x.get("it"), x.get("t"), c.get("p"), c.get("q"), (c.get("rows") or {}).get("up", c.get("up")),
                           low(c), ef.get("up"), ef.get("down"), c.get("fail"), (x.get("policy") or {}).get("std"),
                           (c.get("family_fail") or {}).get("other", c.get("fail"))])
        tl = [e for e in jl(run_dir + "/records/timeline.jsonl")
              if e.get("ev") in ("start", "milestone", "q_gate", "alert", "clear", "fault", "exception", "end")]
        for e in tl:
            for k in ("items", "argv", "env_vars", "algorithm", "rl", "columns", "trace", "weights", "segments"):
                e.pop(k, None)
        last = rec[-1] if rec else {}
        pid = st.get("pid")
        out["runs"].append({
            "robot": r, "exp": os.path.basename(os.path.dirname(run_dir)), "run": os.path.basename(run_dir), "status": st,
            "alive": bool(pid) and os.path.exists("/proc/%s" % pid), "series": series, "timeline": tl[-400:],
            "cur": last.get("cur"), "gate": last.get("gate")})
    for name in ("unified_%s.txt" % r, "unified_%s_students.txt" % r, "unified_%s_accept.txt" % r):
        try:
            lines = open("logs/jobs/" + name, errors="replace").read().splitlines()
        except OSError:
            lines = []
        out["logs"][name] = [l[:400] for l in lines if re.match(r"\d\d-\d\d \d\d:\d\d:\d\d ", l) and l[:14] >= since_log and "NOT YET" not in l][-600:]
    for name in ("unified_%s_full.pick" % r, "unified_%s_blind.pick" % r, "unified_%s_depth.pick" % r):
        try:
            out["picks"][name] = open("logs/jobs/" + name).read().strip()[:300]
        except OSError:
            pass
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
out["procs"] = {k: sum(1 for l in ps if re.search(p, l)) for k, p in (
    ("teacher", r"jobs/unified/teacher\.sh"), ("students", r"jobs/unified/students\.sh"),
    ("accept", r"jobs/unified/accept\.sh"), ("train", r"scripts/train\.py"))}
try:
    g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20).stdout.splitlines()[0]
    out["gpu"] = [int(x) for x in g.replace(" ", "").split(",")[:2]]
except Exception:
    out["gpu"] = None
print(json.dumps(out))
'''


def clock(t, sec: bool = False) -> str:
  """An epoch time as Beijing time, MM-DD HH:MM[:SS]."""
  return "" if t is None else datetime.fromtimestamp(float(t), CST).strftime("%m-%d %H:%M:%S" if sec else "%m-%d %H:%M")


def r3(x, nd: int = 3):
  return None if not isinstance(x, (int, float)) else round(float(x), nd)


def label(run: str) -> tuple[str, str, str]:
  """A run's name -> (role, kind, Chinese label): role teacher / probe / student."""
  tail = re.sub(r"^\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_", "", run)
  m = re.match(r"(blind|depth)(\d)(?:_t(\d+))?_s(\d+)$", tail)
  if m:
    stage = {"1": "阶段 1（蒸馏）", "2": "阶段 2（强化）"}.get(m.group(2), f"阶段 {m.group(2)}")
    return "student", m.group(1), f"{KIND_ZH[m.group(1)]}学生 {stage}" + (f"，老师种子 {m.group(3)}" if m.group(3) else "")
  m = re.match(r"u([a-z]+)_s(\d+)$", tail)
  if m and m.group(1) == "probe":
    return "probe", "teacher", f"探针 种子 {m.group(2)}"
  if m:
    return "teacher", "teacher", f"老师 种子 {m.group(2)}" + ("" if m.group(1) == "full" else f"（{m.group(1)}）")
  return "teacher", "teacher", tail


def speed(series: list, now_t: float | None) -> float | None:
  """Iterations per hour over the run's last SPEED_WINDOW_S seconds (the whole run when it is shorter)."""
  pts = [(s[0], s[1]) for s in series if isinstance(s[0], (int, float)) and isinstance(s[1], (int, float))]
  if len(pts) < 2:
    return None
  it1, t1 = pts[-1]
  it0, t0 = next(((i, t) for i, t in pts if t >= t1 - SPEED_WINDOW_S), pts[0])
  if t1 - t0 < 300:
    it0, t0 = pts[0]
  return (it1 - it0) / (t1 - t0) * 3600.0 if t1 > t0 and it1 > it0 else None


def reached(run: dict) -> dict:
  """The reference schedule's milestones reached, name -> (iteration, time); a stairs one counts from the walk on."""
  got: dict = {}
  for e in run["timeline"]:
    if e.get("ev") == "start":
      for k, v in (e.get("reached") or {}).items():
        if isinstance(v, (int, float)):
          got.setdefault(k, (v, None))
    elif e.get("ev") == "milestone" and str(e.get("name", "")).startswith("expect:"):
      got.setdefault(e["name"][7:], (e.get("it"), e.get("t")))
  walk = got.get("walk")
  return {k: v for k, v in got.items()
          if k not in STAIRS_MILESTONES or (walk is not None and v[0] is not None and v[0] >= walk[0])}


def fold_resumes(runs: list[dict]) -> list[dict]:
  """A run that resumed another run of the list takes that run's curve points and events up to the resume point and
  stands for both (2026-11-12: a move to new code is a resume into a new run directory; the board shows one training
  per seed, not the stopped part beside the running one)."""
  by = {f"{r['exp']}/{r['run']}": r for r in runs}
  gone: set[int] = set()
  for r in sorted(runs, key=lambda x: x["run"]):   # oldest first: a chain folds link by link
    start = next((e for e in r["timeline"] if e.get("ev") == "start"), None)
    res = (start or {}).get("resumed") or {}
    parent = by.get("/".join(str(res.get("checkpoint") or "").split("/")[-3:-1]))
    if parent is None or parent is r:
      continue
    at = res.get("it")
    upto = lambda it: not isinstance(at, (int, float)) or not isinstance(it, (int, float)) or it <= at   # noqa: E731
    r["series"] = [s for s in parent["series"] if upto(s[0])] + r["series"]
    r["timeline"] = [e for e in parent["timeline"] if e.get("ev") != "end" and upto(e.get("it"))] + r["timeline"]
    gone.add(id(parent))
  return [r for r in runs if id(r) not in gone]


# a chain-log line of a probe: the students' / acceptance scripts run as a probe (PROBE ...), a baseline reading
# (BASELINE ..., "(baseline)"), anything on a probe's run directory
PROBE_LINE = re.compile(r"PROBE|BASELINE|\(baseline\)|_probe_|uprobe_")


def tier2_zh(m) -> str:
  """A formal teacher's tier-2 verdict line (teacher.sh's gate: the stairs, then the flat tests) in Chinese."""
  rest = re.sub(r"\(gap ([\d.]+)\)", r"（差距 \1）", m[4])
  for a, b in (("HARDPASS", "硬项通过（质量目标没全到）"), ("PASS", "通过"), ("FAIL", "不通过"), ("scan on", "扫描开"), ("scan off", "扫描关")):
    rest = rest.replace(a, b)
  return f"种子 {m[1]} 第 {m[2]} 步，第 2 层{'楼梯' if m[3] == 'stairs' else '平地'}检查：{rest}"


def chain_zh(text: str) -> str | None:
  """A chain-log line's text in Chinese (None: not an event for the board); an unknown line stays as written."""
  base = lambda p: "/".join(p.rstrip("/").split("/")[-2:])   # noqa: E731
  rules = [
    (r"seed \d+: (NOT YET|no verdict|its new run dir|passes at \d+, but|PRE-PASS at \d+[;,])", None),
    (r"gate: PRE-PASS on the curriculum", lambda m: "通过条件：先课程预通过（进度、质量走完并保持），再第 2 层楼梯检查，再第 2 层平地检查"),
    (r"seed (\d+): PRE-PASS (\d+)$", lambda m: f"种子 {m[1]} 课程预通过（第 {m[2]} 步）；接着由第 2 层检查楼梯和平地"),
    (r"seed (\d+): model_(\d+) tier-2 (stairs|flat): (.*)", tier2_zh),
    (r"seed (\d+): TOOL FAULT in the tier-2 (stairs|flat) gate \(exit (\d+)",
     lambda m: f"种子 {m[1]}：第 2 层{'楼梯' if m[2] == 'stairs' else '平地'}检查工具故障（exit {m[3]}；不算未通过，下次再判）"),
    (r"STOP: needs a human -- no seed passed and nothing was judged", lambda m: "停下等人工处理：到上限没有种子通过，第 2 层也没有判过任何检查点"),
    (r"seed \d+ started \(pid", None),
    (r"STOPPED by ", lambda m: "正式训练被手动停止：这一版不交付（原因见同一时刻的说明）；检查点都保留"),
    (r"FROM ZERO on (\w+)", lambda m: f"从零开训（代码 {m[1]}）；这之前的运行作废"),
    (r"seeds (.+?), (\d+) robots each, (side by side|one after the other), cap (\d+) it, code (\w+)",
     lambda m: f"老师链启动：种子 {m[1]}，每个 {m[2]} 台机器人，{'并行' if m[3] == 'side by side' else '依次'}，上限 {m[4]} 步，代码 {m[5]}"),
    (r"resume from (\S+) model_(\d+); seeds (.+?),", lambda m: f"老师链从 {m[1]} 第 {m[2]} 步续训（种子 {m[3]}）"),
    (r"quality-stage probe ", None),
    (r"seed (\d+) resumes from (\S+) model_(\d+)", lambda m: f"种子 {m[1]} 从 {m[2]} 第 {m[3]} 步续训"),
    (r"seed (\d+): already at the cap", lambda m: f"种子 {m[1]} 已到上限，不再启动"),
    (r"TEACHER PICK (\S+) \(seed (\d+)(.*)",
     lambda m: (f"老师到上限：硬项全过、质量目标没全到，选定种子 {m[2]}，{base(m[1])}" if "every hard item passed" in m[3]
                else f"老师到上限没过门：选定判过的检查点里最好的，种子 {m[2]}，{base(m[1])}" if "below the gate" in m[3]
                else f"老师通过：选定种子 {m[2]}，{base(m[1])}")),
    (r"PASS \(seed (\d+): .*?\); kept for a fallback: seeds (.+?) \(", lambda m: f"老师链完成：种子 {m[1]} 通过；备用种子 {m[2]}"),
    (r"seeds (.+?) stopped \(seed (\d+) passed", lambda m: f"种子 {m[1]} 已停（种子 {m[2]} 通过；检查点都保留）"),
    (r"seed (\d+) stopped at its pick", lambda m: f"种子 {m[1]} 在选定的检查点停下"),
    (r"KEEP_OTHERS=1", lambda m: "其余种子继续跑到上限，作备用"),
    (r"seed (\d+) ended \(exit (\d+)\) without passing", lambda m: f"种子 {m[1]} 结束（exit {m[2]}），没有通过"),
    (r"no seed passed", lambda m: "老师链结束：没有种子通过"),
    (r"waiting for the teacher's pick", lambda m: "学生等待进程已启动，等老师通过"),
    (r"the waiter started (\S+ \S+) \(old code\) stopped", lambda m: f"旧版学生等待进程（{m[1]} 启动）已停，用户同意；新版在学生冒烟测试通过后启动"),
    (r"teacher (\S+) \(seed (\d+), code (\w+)\); kinds (.+?); stage 1 (\d+) it, stage 2 (\d+) it, (\d+) robots each",
     lambda m: f"学生开训：老师种子 {m[2]}（{base(m[1])}），{m[4]}；阶段 1 {m[5]} 步，阶段 2 {m[6]} 步，每种 {m[7]} 台机器人"),
    (r"(blind|depth) stage (\d) done", lambda m: f"{KIND_ZH[m[1]]}学生阶段 {m[2]} 完成"),
    (r"TOOL FAULT: (.*)", lambda m: f"工具故障（不算未通过）：{m[1]}"),
    (r"tool fault:(.*?) \(", lambda m: f"工具故障（不算未通过，没有重试）：{m[1].strip()}"),
    (r"done with teacher seed (\w+) \((.*?)\s*\)", lambda m: f"学生流程结束（老师种子 {m[1]}；{m[2]}）"),
    (r"fallback: kinds(.*?) failed their acceptance on teacher seed (\d+); resuming teacher seed (\d+)",
     lambda m: f"回退：{m[1].strip()} 学生验收没过（老师种子 {m[2]}），改用老师种子 {m[3]} 续训"),
    (r"STOP: needs a human -- kinds(.*?) have no accepted student", lambda m: f"停下等人工处理：{m[1].strip()} 没有通过验收的学生"),
    (r"every kind has a pick", lambda m: "两种学生都通过验收并选定"),
    (r"candidates \(avg: (\d+) checkpoints each.*?\): (.*)", lambda m: f"验收候选（每个由 {m[1]} 个检查点平均）：{m[2]}"),
    (r"candidate (\S+): tier 2 .*started", lambda m: f"第 2 层验收（mjlab）开始：{base(m[1])}"),
    (r"(\S+) tier 2 \((\d+) min\): (.*)", lambda m: f"第 2 层验收（mjlab，{m[2]} 分钟）：{m[3]}"),
    (r"(\S+) tier 3 \(tinynav\): (.*)", lambda m: f"第 3 层验收（tinynav）：{m[2]}"),
    (r"(\S+) tier 3 requested", lambda m: "第 3 层验收（tinynav）已提交"),
    (r"(\S+) tier 3 skipped", lambda m: "第 3 层验收跳过（TN_GATE=skip）"),
    (r"^PICK (\S+)", lambda m: f"验收通过，选定 {base(m[1])}"),
    (r"UNIFIED DELIVERY (\S+)", lambda m: f"交付包已生成：{m[1]}"),
    (r"no candidate of (\S+) passed", lambda m: "验收未通过：没有候选通过"),
    (r"done: (\d+) candidates read, no pick \(baseline\)", lambda m: f"基线测量完成：读了 {m[1]} 个候选，不选定"),
  ]
  for pat, fn in rules:
    m = re.search(pat, text)
    if m:
      return fn(m) if fn else None
  return text


def timeline_zh(e: dict, got: dict) -> str | None:
  """A timeline event in Chinese (None: not shown)."""
  ev, it = e.get("ev"), e.get("it")
  if ev == "start":
    res = e.get("resumed") or {}
    if res.get("checkpoint"):
      return f"从 {'/'.join(str(res['checkpoint']).split('/')[-2:])} 续训（代码 {e.get('code')}，到第 {e.get('iterations')} 步）"
    return f"从零开训（代码 {e.get('code')}，{e.get('num_envs')} 台机器人，上限 {e.get('iterations')} 步）"
  if ev == "milestone":
    name = str(e.get("name", ""))
    if not name.startswith("expect:"):
      return None
    k = name[7:]
    if k not in got or got[k][0] != it:   # (a stairs milestone before the walk: the starting rows, not shown)
      return None
    return f"到达时间点「{MILE.get(k, (k,))[0]}」：第 {it} 步（期限第 {e.get('mark')} 步）"
  if ev == "q_gate":
    if e.get("state") == "waiting_row" and not e.get("flips"):   # (where every run starts)
      return None
    return f"楼梯质量阶段：{rr.GATE_ZH.get(e.get('state'), e.get('state'))}（第 {it} 步，上楼行 {rr.f(e.get('up'))}）"
  if ev == "alert":
    return None if e.get("again") else f"告警 {e.get('name')}：{rr.alert_zh(e)}（第 {it} 步）"
  if ev == "clear":
    return f"告警解除：{e.get('name')}（第 {it} 步）"
  if ev == "fault":
    return f"训练记录的一部分出错停了（训练继续）：{e.get('where')}"
  if ev == "exception":
    return f"训练异常退出：{str(e.get('error'))[:160]}（第 {it} 步）"
  if ev == "end":
    how = {"done": "跑完", "interrupt": "被停止", "exception": "异常退出"}.get(e.get("how", e.get("ended")), "结束")
    return f"训练{how}（第 {it} 步）"
  return None


def fetch(key: str, since: str) -> dict | None:
  name, box, port = ROBOTS[key]
  cmd = f"cd {D} && python3 - '{json.dumps({'robots': [key], 'since': since})}'"
  try:
    res = subprocess.run([SSH, port, cmd], input=REMOTE, capture_output=True, text=True, timeout=120)
    return json.loads(res.stdout.strip().splitlines()[-1])
  except Exception as e:   # a box that does not answer: its robot says so
    print(f"board_unified: {box} did not answer ({type(e).__name__})", file=sys.stderr)
    return None


# the board's dataset documents: name -> (title, description); all six read the one attached file
DATASETS = {
  "robots": ("各机型的流程", "每个机型一行：老师、盲走学生、深度学生、验收、交付各到哪一步，以及这台机器的 GPU。"
                             "只记录正式训练（一个冻结提交、从零开始；Go2 / Go2W）。"),
  "runs": ("训练运行", "每个正式训练运行一行（老师的每个种子，之后是学生）：步数、进度 p、质量 q、上楼行、"
                       "各失败率、动作噪声、速度、下一个参考时间点、通过条件、告警数。取自各运行的训练记录。"),
  "milestones": ("参考进度的时间点", "每个正式老师种子的参考进度：到第几步应该达到什么、按现在速度几点到、"
                                     "已达到 / 未到 / 迟到，以及现在的读数。"),
  "curves": ("训练曲线", "每个正式老师种子随迭代步数的 p、q、上楼行和各失败率（每个运行最多 150 个点；"
                         "失败率从第 50 步起）。"),
  "alerts": ("当前告警", "正式训练的记录里现在没解除的告警，每条一行。"),
  "events": ("最近事件", "各正式运行的时间线事件（开训、时间点、告警、故障、结束）、流程日志里的结论和人工记下的判断，新的在前，最多 80 条。"),
}
SCHEDULE = "训练期间每 30 分钟由 Claude 的循环重新生成并上传"
STALE_HOURS = 2


def docs(out: Path, blob: str) -> int:
  """The six dataset documents for the board's store, pointing at the uploaded file `blob` (/_blob/<id>)."""
  board = json.loads((out / "board.json").read_text())
  (out / "docs").mkdir(parents=True, exist_ok=True)
  for name, (title, description) in DATASETS.items():
    doc = {"title": title, "description": description, "schedule": SCHEDULE, "staleAfterHours": STALE_HOURS,
           "source": {"kind": "file", "format": "json", "name": "board.json", "url": blob},
           "updated": {"at": board["updated"], "by": "Claude"}, "edit": {"at": board["updated"], "via": "claude"}}
    (out / "docs" / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
  print(f"{out / 'docs'}: {', '.join(DATASETS)} -> {blob} (data of {board['updated']})")
  return 0


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("outdir")
  ap.add_argument("--since", default="2026-11-13_17-15-00")   # the formal run's start (Go2W 17:15:52, Go2 17:32:10)
  ap.add_argument("--manual", default=str(MANUAL))
  ap.add_argument("--docs", metavar="BLOB", help="write OUTDIR/docs/*.json for the uploaded file /_blob/<id> and stop")
  a = ap.parse_args()
  if a.docs:
    return docs(Path(a.outdir), a.docs)
  now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  robots, runs, miles, curves, events, alerts_out = [], [], [], [], [], []
  slots = json.loads(SLOTS.read_text()) if SLOTS.exists() else {}   # (a run keeps its colour when others come or go)

  for key, (R, box, _) in ROBOTS.items():
    d = fetch(key, a.since)
    if d is None:
      robots.append({"robot": R, "machine": box, "teacher": "连不上训练机", "blind": "", "depth": "", "accept": "",
                     "delivery": "", "gpu": "", "n_training": 0, "teacher_ok": 0, "students_ok": 0, "n_alerts": 0,
                     "n_late": 0, "n_bad": 1, "updated": now_iso})
      continue
    now_t = d["now"]
    mine = []
    # the formal trainings only (the user, 2026-11-13). The probes go before the folding: a probe continued from a
    # formal run's checkpoint is a resume too, and folded first it took the formal run off the board with itself
    # (11-13 22:40: uprobe_s58 from the Go2W seed 0's model_3200)
    formal = [r for r in d["runs"] if label(r["run"])[0] != "probe"]
    for run in sorted(fold_resumes(formal), key=lambda x: (x["exp"], x["run"])):
      st, series = run["status"], run["series"]
      role, kind, lab = label(run["run"])
      it, cap = st.get("it"), st.get("total_it")
      age = (now_t - float(st.get("t_end") or st.get("t") or now_t)) / 60.0
      if st.get("ended"):
        status = {"done": "ended", "interrupt": "stopped", "exception": "crashed"}.get(st["ended"], "ended")
      elif not run["alive"]:
        status = "dead"
      else:
        status = "stalled" if age >= STALE_MIN else "training"
      sp = speed(series, now_t)
      eta = lambda n: (None if sp is None or it is None or not st.get("t") else float(st["t"]) + (n - it) / sp * 3600.0)  # noqa: E731
      got = reached(run) if role == "teacher" else {}
      expect = st.get("expect") or {} if role == "teacher" else {}
      for k in STAIRS_MILESTONES:   # one the records marked before the walk: judged on the curve from the walk on, and
        if k in expect and k not in got and "walk" in got and MILE[k][1] in COL:   # the climbing row only at its deadline
          _, measure, value, cmp = MILE[k]
          first = max(got["walk"][0], expect[k][0] if k == "climb_low" else 0)
          hit = next((s for s in series if isinstance(s[0], (int, float)) and s[0] >= first
                      and isinstance(s[COL[measure]], (int, float))
                      and (s[COL[measure]] >= value if cmp == ">=" else s[COL[measure]] <= value)), None)
          if hit and k != "climb":   # (the quality stage's start has its own event)
            got[k] = (hit[0], hit[1])
      order = sorted(expect, key=lambda k: (expect[k][0], list(MILE).index(k) if k in MILE else 99))
      todo = [k for k in order if k not in got]
      late = [k for k in todo if it is not None and it >= expect[k][0]]
      cur = run.get("cur") or {}
      names = list(st.get("alerts") or [])
      last_alert = {e.get("name"): e for e in run["timeline"] if e.get("ev") == "alert"}
      alerts = "；".join(names)
      for n in names:
        e = last_alert.get(n) or {}
        alerts_out.append({"key": f"{R} {lab} {n}", "robot": R, "machine": box, "run": lab, "name": n,
                           "text": rr.alert_zh(e) if e else "", "since": e.get("since", e.get("it")),
                           "at": clock(e.get("t")) if e.get("t") else ""})
      gate = run.get("gate") or {}
      if role != "student" and gate:
        miss = gate.get("miss") or []
        gate_zh = ("进度 p " + ("已满" if gate.get("p_done") else "未满") + "，质量 q " + ("已满" if gate.get("q_done") else "未满")
                   + (f"，没达标或还没测到的约束 {len(miss)} 项" if miss else "，约束全部达标"))
      else:
        gate_zh = ""
      nxt = ""
      if todo and status in ("training", "stalled"):
        k = todo[0]
        nxt = f"{MILE.get(k, (k,))[0]}（期限第 {expect[k][0]} 步" + (f"，约 {clock(eta(expect[k][0]))[6:]}）" if eta(expect[k][0]) and k not in late else "，已迟到）")
      row = {
        "key": f"{R} {lab}", "robot": R, "machine": box, "run": lab, "role": role, "kind": kind, "status": status,
        "iter": it, "max": cap, "p": r3(st.get("p")), "q": r3(st.get("q")), "up": r3(cur.get("rows", {}).get("up", st.get("up")), 2),
        "low_fail": r3(rr.low_fail(cur), 2), "up_fail": r3((cur.get("eval_fail") or {}).get("up"), 2),
        "down_fail": r3((cur.get("eval_fail") or {}).get("down"), 2), "fail": r3(st.get("fail"), 2),
        "std": r3(st.get("std"), 2), "speed": None if sp is None else int(round(sp)),
        "cap_eta": clock(eta(cap)) if cap and status == "training" else "", "next": nxt, "late": len(late), "gate": gate_zh,
        "alerts": alerts, "n_alerts": len(names), "faults": "、".join(st.get("faults") or []), "code": st.get("code") or "",
        "dir": f"{run['exp']}/{run['run']}", "at": clock(st.get("t_end") or st.get("t")), "updated": now_iso,
      }
      runs.append(row)
      mine.append((run, row))
      if role == "teacher":
        for i, k in enumerate(order):
          by = expect[k][0]
          g = got.get(k)
          state = "reached" if g else ("late" if k in late else "open")
          zh, measure, value, cmp = MILE.get(k, (k, "", None, ""))
          miles.append({
            "key": f"{R} {lab} {k}", "robot": R, "run": lab, "order": i, "name": zh, "by": by,
            "due": "" if (it or 0) >= by else clock(eta(by)), "reached_it": g[0] if g else None,
            "reached_at": clock(g[1]) if g and g[1] else "", "state": state, "now": rr.expect_now(k, cur) if cur else "",
            "measure": measure, "value": value, "cmp": cmp})
        name = f"{R} {lab.replace('老师 ', '')}"
        slot = slots.setdefault(name, len(slots))
        for s in series:
          if isinstance(s[0], (int, float)):
            rate = (lambda v: r3(v) if s[0] >= FAIL_FROM_IT else None)   # noqa: E731
            curves.append({"run": name, "slot": slot, "robot": R, "iter": s[0], "p": r3(s[2]), "q": r3(s[3]),
                           "up": r3(s[4]), "low": rate(s[5]), "up_fail": rate(s[6]), "down_fail": rate(s[7]),
                           "fail": rate(s[8]), "std": r3(s[9]), "flat": rate(s[10])})
      for e in run["timeline"]:
        text = timeline_zh(e, got)
        if text and e.get("t") is not None:
          events.append({"at": clock(e["t"], sec=True), "machine": box, "robot": R, "run": lab, "text": text})

    # the chain logs: the teacher chain, the students, the acceptance
    last_line: dict = {}
    prepass: set = set()
    for fname, lines in d["logs"].items():
      for l in lines:
        head, _, text = l[15:].partition(": ")
        if "SMOKE" in l or PROBE_LINE.search(l):
          continue
        m = re.search(r"seed (\d+): PRE-PASS \d+$", text)   # every gate round repeats it: the first one is the event
        if m:
          if (fname, m[1]) in prepass:
            continue
          prepass.add((fname, m[1]))
        what = "流程"
        m = re.match(r"\S+ (blind|depth)$", head)
        if m:
          what = f"{KIND_ZH[m.group(1)]}学生验收"
          last_line[("accept", m.group(1))] = text
        elif head.endswith("students"):
          what = "学生"
        zh = chain_zh(text)
        if zh:
          events.append({"at": l[:14], "machine": box, "robot": R, "run": what, "text": zh[:300]})

    # the robot's flow: teacher -> students -> acceptance -> delivery
    teachers = [(run, row) for run, row in mine if row["role"] == "teacher"]
    pick = d["picks"].get(f"unified_{key}_full.pick")
    if pick:
      t_cell = "已通过，选定 " + "/".join(pick.split("/")[-2:])
    elif any(row["status"] in ("training", "stalled") for _, row in teachers):
      t_cell = "训练中：" + "；".join(f"{row['run'].replace('老师 ', '')} 第 {row['iter']} 步" for _, row in teachers
                                  if row["status"] in ("training", "stalled")) + f"（上限 {teachers[0][1]['max']} 步）"
    elif teachers:
      t_cell = "已停，没有通过"
    else:
      t_cell = "没有开始"
    cells = {}
    for kind in ("blind", "depth"):
      sp_pick = d["picks"].get(f"unified_{key}_{kind}.pick")
      mine_k = [row for _, row in mine if row["role"] == "student" and row["kind"] == kind]
      live = [row for row in mine_k if row["status"] in ("training", "stalled")]
      if sp_pick:
        cells[kind] = "已通过验收，选定 " + "/".join(sp_pick.split("/")[-2:])
      elif live:
        cells[kind] = f"{live[-1]['run'].split('学生 ')[-1]}：第 {live[-1]['iter']} / {live[-1]['max']} 步"
      elif mine_k:
        cells[kind] = "训练结束，验收中" if d["procs"].get("accept") else "训练结束"
      else:
        cells[kind] = "等老师通过" if d["procs"].get("students") else "没有开始（没有等待进程）"
    acc = "；".join(f"{KIND_ZH[k]}：{(chain_zh(last_line[('accept', k)]) or last_line[('accept', k)])[:120]}"
                   for k in ("blind", "depth") if ("accept", k) in last_line)
    deliv = [e["text"] for e in events if e["robot"] == R and e["text"].startswith("交付包已生成")]
    gpu = d.get("gpu")
    formal = [row for _, row in mine]
    robots.append({"robot": R, "machine": box, "teacher": t_cell, "blind": cells["blind"], "depth": cells["depth"],
                   "accept": acc or "没有开始", "delivery": "；".join(deliv) or "没有",
                   "gpu": "" if not gpu else f"{gpu[0]}%，{gpu[1] / 1024:.1f} GB",
                   "n_training": sum(1 for row in formal if row["status"] == "training"), "teacher_ok": int(bool(pick)),
                   "students_ok": sum(1 for k in ("blind", "depth") if d["picks"].get(f"unified_{key}_{k}.pick")),
                   "n_alerts": sum(row["n_alerts"] for row in formal), "n_late": sum(row["late"] for row in formal),
                   "n_bad": sum(1 for row in formal if row["status"] in ("stalled", "dead", "crashed") or row["faults"]),
                   "updated": now_iso})

  manual = Path(a.manual)
  if manual.exists():
    events += json.loads(manual.read_text()).get("events", [])
  events.sort(key=lambda e: e["at"], reverse=True)
  seen, uniq = set(), []
  for e in events:
    k = (e["at"], e.get("robot"), e["text"][:80])
    if k not in seen:
      seen.add(k)
      uniq.append({"id": f"{e['at']} {e.get('robot', '')} {zlib.crc32(e['text'].encode()):08x}", "at": e["at"],
                   "machine": e.get("machine", ""),
                   "robot": e.get("robot", ""), "run": e.get("run", ""), "text": e["text"]})
  SLOTS.parent.mkdir(parents=True, exist_ok=True)
  SLOTS.write_text(json.dumps(slots, ensure_ascii=False))
  out = Path(a.outdir)
  out.mkdir(parents=True, exist_ok=True)
  board = {"updated": now_iso, "since": a.since, "robots": robots, "runs": runs, "milestones": miles, "curves": curves,
           "alerts": alerts_out, "events": uniq[:80]}
  (out / "board.json").write_text(json.dumps(board, ensure_ascii=False, indent=1))
  print(f"{out / 'board.json'}: {len(robots)} robots, {len(runs)} runs, {len(miles)} milestones, {len(curves)} curve points, "
        f"{len(alerts_out)} alerts, {len(uniq[:80])} events")
  for row in runs:
    print(f"  {row['key']}: {row['status']}, it {row['iter']} / {row['max']}, p {row['p']}, q {row['q']}, "
          f"{row['speed']} it/h, next {row['next'] or '-'}, alerts {row['n_alerts']}, late {row['late']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
