"""Unified acceptance tables of the blind locomotion policies (self-contained copy).

Same sections for legged and wheeled robots, in a fixed order; the first is the goal
table (通过 or 不通过：<short tags>). Run `python accept_tables.py` for markdown, or
`python accept_tables.py --json dash` for the dashboard datasets (one JSON per section
plus sections.json; then `python dash_page.py dash/sections.json dash/index.html`).
See README.md.
"""
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
KINDS = ("noop", "op", "am", "stairs", "drift3", "stand", "climb", "hip", "stairhip", "gait", "march")
WHEELED = ("Go2W", "As2W", "B2W")
NA = "—"


def kv(line):
  return {k: float(v) for k, v in re.findall(r"(\w+)=([-\d.]+)", line)}


def read(path):
  return open(path).read() if path and os.path.exists(path) else None


def files(model_id):
  return {k: str(RESULTS / f"{model_id}_{k}.txt") for k in KINDS}


# manifest.json: {robot: [{"label", "id"}, ...]}, the first row the deployed final
# (the baseline of the percentages and of the hip torque goal); result files are
# results/<id>_<kind>.txt.
MANIFEST = json.load(open(HERE / "manifest.json"))
ROWS = {r: [(m["label"], files(m["id"])) for m in rows] for r, rows in MANIFEST.items()}


def txt(p):
  return read(p) if p and os.path.exists(p) else None


# ---- 1. stairs reached ----
def scen(p):
  t = txt(p)
  if not t:
    return None
  d = {re.search(r"scenario=(\S+)", l).group(1): kv(l) for l in t.splitlines() if "scenario=" in l}
  return d if len(d) == 18 else None


def stairs_vals(p):
  no, op = scen(p.get("noop")), scen(p.get("op"))
  if not no or not op:
    return None
  m = lambda d, f: 100 * sum(v["reached"] for k, v in d.items() if f(k)) / sum(1 for k in d if f(k))
  sp = lambda d, f: sum(v["speed_p50"] for k, v in d.items() if f(k)) / sum(1 for k in d if f(k))
  up, dn, al = (lambda k: k.endswith("_up")), (lambda k: k.endswith("_down")), (lambda k: True)
  return dict(op=m(op, al), op_up=m(op, up), op_dn=m(op, dn), no=m(no, al), no_up=m(no, up), no_dn=m(no, dn),
              sp_up=sp(op, up), sp_dn=sp(op, dn))


# ---- 2. stopping mid-flight ----
def stop_row(p):
  t = txt(p.get("stairs")) or ""
  m = re.search(r"overall: traverse ok (\d+)%, fell (\d+)%; stop ok (\d+)%", t)
  if not m:
    return None
  bands = re.findall(r"^\s*(\d+-\d+cm)\s+(up|down) \|\s+(\d+)%\s+(\d+)%\s+[\d.]+%\s+[\d.]+ \|\s+(\d+)%\s+(\d+)%\s+([\d.]+)", t, re.M)
  worst = min(bands, key=lambda b: int(b[4])) if bands else None
  w = f"{worst[0]} {'上' if worst[1] == 'up' else '下'} {worst[4]}%" if worst else NA
  drift = max(float(b[6]) for b in bands) if bands else float("nan")
  return f"{m.group(1)}% | {m.group(2)}% | {m.group(3)}% | {w} | {drift:.1f}"


# ---- 3. 44 flat commands ----
def am_vals(p):
  t = txt(p.get("am")) or ""
  rows = re.findall(r"^\s*(\([^)]*\)) \|\s*(.*?) \|\s*(\d+)/(\d+) \| (ok|FAIL)", t, re.M)
  if len(rows) != 44:
    return None
  falls = sum(int(k) for _, _, k, _, _ in rows)
  fails = [f"{c} {a.strip()}" + (f" 摔{k}" if int(k) else "") for c, a, k, n, v in rows if v == "FAIL"]
  return dict(ok=sum(v == "ok" for *_, v in rows), falls=falls, fails=fails)


# ---- 4. drift ----
def drift_vals(p):
  out = {}
  for l in (txt(p.get("drift3")) or "").splitlines():
    m = re.match(r"\s*\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)\s*\|(.*)", l)
    if not m:
      continue
    cells = m.group(4).split("|")
    nums = [[float(x) for x in c.split()] for c in cells[:5]]
    fell = re.match(r"\s*(\d+)/(\d+)", cells[5])
    out[(float(m.group(1)), float(m.group(2)), float(m.group(3)))] = dict(
      head=nums[0], along=nums[1], across=nums[2], end=nums[3], far=nums[4], fell=int(fell.group(1)))
  return out


def pq(x, unit, fmt="{:+.1f}"):
  """p50 (p10 ~ p90)."""
  return f"{fmt.format(x[1])}{unit} ({fmt.format(x[0])}~{fmt.format(x[2])})"


# ---- 5. standing still after letting go ----
def stand_vals(p):
  g = {}
  for l in (txt(p.get("stand")) or "").splitlines():
    m = re.match(r"\s*([a-z][a-z0-9 .+]*?) \|\s+([\w.]+)\s+([\w.]+) \|\s+([\w.]+)\s+([\w.]+)\s+([\w.]+) \|\s+([\w.]+)\s+([\w.]+) \|.*\| (\d+)/(\d+)", l)
    if m:
      f = lambda s: float(s) if s != "nan" else float("nan")
      g[m.group(1).strip()] = dict(stop_s=f(m.group(2)), stop_cm=f(m.group(3)), creep=f(m.group(4)),
                                   p90=f(m.group(5)), yaw=f(m.group(6)), air=f(m.group(7)), fell=int(m.group(9)))
  return g


# ---- 6. climbing posture ----
def climb_vals(p):
  m = re.search(r"climbing L0 \(0.5 m/s\): pitch ([-+\d.]+) deg, roll ([-+\d.]+) deg, vx ([-+\d.]+)", txt(p.get("climb")) or "")
  return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None


# ---- 7. hip load ----
def hip_vals(p):
  out = {}
  for l in (txt(p.get("hip")) or "").splitlines():
    m = re.match(r"\s*cmd \(([-\d.]+), 0.0, 0.0\).*front: outward\s+([-+\d.]+) ->\s+([-+\d.]+) deg.*rms ([\d.]+) Nm.*rear: outward\s+([-+\d.]+) ->\s+([-+\d.]+) deg.*rms ([\d.]+) Nm", l)
    if m:
      g = m.groups()
      out[float(g[0])] = dict(f=float(g[3]), r=float(g[6]), fo=(float(g[1]), float(g[2])), ro=(float(g[4]), float(g[5])))
  return out


def stairhip_vals(p):
  out = {}
  for l in (txt(p.get("stairhip")) or "").splitlines():
    m = re.match(r"all\s+(up|down): hip rms front ([\d.]+) rear ([\d.]+) Nm, fell (\d+)%", l)
    if m:
      out[m.group(1)] = dict(f=float(m.group(2)), r=float(m.group(3)), fell=int(m.group(4)))
  return out


# ---- 8. wheeled: all four wheels down ----
NOLIFT = ("0.2", "0.5", "1.0", "1.5", "-0.5", "-1.0")


def nolift_vals(p):
  t = txt(p.get("gait")) or ""
  vals, fell = [], 0
  for c in NOLIFT:
    m = re.search(rf"^\s*\({c}, 0.0, 0.0\)\s*\|[^|]*\|\s*([\d.]+)%[^|]*\|[^|]*\|[^|]*\|\s*(\d+)/(\d+)", t, re.M)
    if not m:
      return None
    vals.append(float(m.group(1)))
    fell += int(m.group(2))
  return dict(down=vals, fell=fell, ok=min(vals) >= 98.0 and fell == 0)


def march_vals(p):
  """march_eval.py: per mode (stand / march) touchdowns per second per foot (mean of the
  four), drift (cm), hip RMS torque front / rear (Nm) and falls."""
  out = {}
  for l in (txt(p.get("march")) or "").splitlines():
    m = re.match(r"(stand|march) \| touchdowns/s per foot ([\d. ]+) \| air % ([\d ]+) \| drift ([\d.]+) cm.*rms tau \+?([-\d.]+) / \+?([-\d.]+) Nm \| fell (\d+)/(\d+)", l)
    if m:
      g = m.groups()
      out[g[0]] = dict(td=sum(map(float, g[1].split())) / 4, drift=float(g[3]), f=float(g[4]), r=float(g[5]),
                       fell=int(g[6]), n=int(g[7]))
  return out


LEGGED = ("Go2", "As2", "B2")


def rel(a, b):
  return "" if b is None or a is None or b == 0 else f" ({100 * (a / b - 1):+.0f}%)"


def gates(r, p):
  s, a = stairs_vals(p), am_vals(p)
  g = []
  g.append(NA if s is None else ("通过" if s["op"] >= 95.0 else "不通过"))
  g.append(NA if a is None else ("通过" if a["falls"] == 0 else "不通过"))
  if r in WHEELED:
    n = nolift_vals(p)
    g.append(NA if n is None else ("通过" if n["ok"] else "不通过"))
  else:
    g.append("不适用")
  return g


PENDING = "评测中"


def march_cell(r, p):
  if r not in LEGGED:
    return "未训练"
  v = march_vals(p)
  if "stand" not in v or "march" not in v:
    return PENDING
  s, m = v["stand"], v["march"]
  hot = 100 * (max(m["f"], m["r"]) / max(s["f"], s["r"]) - 1)
  return f"{m['drift']:.0f} cm / {hot:+.0f}%"


FAIL = "不通过："


def goal_cells(r, i, label, p, ms):
  """One verdict per goal (user 2026-10-22 ~21:30): 通过, or 不通过：<short tags joined by 、>
  (two or three characters each, ~21:35); the numbers are in the detail tables."""
  out = {}
  def verdict(key, tags, missing=False):
    tags = list(dict.fromkeys(tags))
    out[key] = PENDING if missing else ("通过" if not tags else FAIL + "、".join(tags))
  a, d, g = am_vals(p), drift_vals(p), stand_vals(p)
  # Flat speed, small commands included.
  tags = []
  for item in (a or {}).get("fails", []):
    ratios = [int(x) for x in re.findall(r"[xyw](\d+)%", item)]
    if any(v > 120 for v in ratios):
      tags.append("超速")
    if any(v < 80 for v in ratios):
      tags.append("偏慢")
    if "摔" in item:
      tags.append("摔倒")
  verdict("g_speed", tags, a is None)
  # Straight and sideways drift.
  tags = []
  for c, v in d.items():
    if c[2] != 0.0:
      continue
    sideways = c[1] != 0.0
    travel = abs(v["across"][1]) if sideways else abs(v["along"][1])
    off = v["along"] if sideways else v["across"]
    if abs(v["head"][1]) > 3.0 or max(abs(v["head"][0]), abs(v["head"][2])) > 8.0:
      tags.append("转头")
    if travel > 0 and max(abs(off[0]), abs(off[2])) > 0.05 * travel:
      tags.append("跑偏")
  verdict("g_drift", tags, not d)
  # Turning in place.
  spins = [v for c, v in d.items() if c[0] == c[1] == 0.0 and c[2] != 0.0]
  verdict("g_spin", ["漂移"] if any(v["end"][1] > 0.2 for v in spins) else [], not spins)
  # Stopping.
  tags = []
  if "rest" in g:
    if g["rest"]["air"] > 1.0:
      tags.append("离地")
    if max(v["creep"] for v in g.values()) > 1.0:
      tags.append("蠕动")
    if any(k in g and g[k]["stop_cm"] > 30.0 for k in ("fwd 1.0", "back 1.0")):
      tags.append("刹停远")
  verdict("g_stop", tags, "rest" not in g)
  # Marching in place.
  mv = march_vals(p)
  if "stand" in mv and "march" in mv:
    s, m = mv["stand"], mv["march"]
    tags = (["漂移"] if m["drift"] > 10.0 else []) + (["更热"] if max(m["f"], m["r"]) >= max(s["f"], s["r"]) else []) \
      + (["摔倒"] if s["fell"] + m["fell"] else [])
    verdict("g_march", tags)
  else:
    out["g_march"] = "未训练" if r not in LEGGED and i < 2 else PENDING
  # Hip abduction torque, at least 30% below the deployed final.
  h, h0 = hip_vals(p), hip_vals(ms[0][1])
  if i == 0:
    out["g_hip"] = "基准"
  else:
    tags = []
    for c, n in ((0.0, "站立高"), (1.0, "行走高")):
      if c in h and c in h0 and any(1 - h[c][sd] / h0[c][sd] < 0.3 for sd in ("f", "r")):
        tags.append(n)
    verdict("g_hip", tags, 0.0 not in h)
  # Stairs.
  s, st = stairs_vals(p), stop_row(p)
  tags = []
  if s:
    if s["op"] < 95.0:
      tags.append("带操作")
    if s["no"] < 90.0:
      tags.append("无操作")
    for k, n in (("sp_up", "上楼"), ("sp_dn", "下楼")):
      if s[k] < 0.8:
        tags.append(n + "慢")
      elif s[k] > 1.2:
        tags.append(n + "快")
  if st and float(st.split("|")[2].strip().rstrip("%")) < 95.0:
    tags.append("停不稳")
  verdict("g_stairs", tags, not s)
  # Wheeled: all four wheels down driving forward / backward.
  if r in WHEELED:
    n = nolift_vals(p)
    if n:
      verdict("g_wheels", (["抬轮"] if min(n["down"]) < 98.0 else []) + (["摔倒"] if n["fell"] else []))
    else:
      out["g_wheels"] = PENDING
  else:
    out["g_wheels"] = "不适用"
  # No falls anywhere: the tests that had falls.
  tags = []
  if a and a["falls"]:
    tags.append("平地")
  if st and st.split("|")[1].strip() != "0%":
    tags.append("楼梯")
  if sum(v["fell"] for v in d.values()):
    tags.append("跑偏")
  if sum(v["fell"] for v in g.values()):
    tags.append("停车")
  verdict("g_falls", tags, a is None)
  out["g_push"] = "未测"
  return out


def build():
  """The acceptance sections: [{id, title, note, cols: [(field, header)], rows: [dict]}]. Every row
  has `model` (the key: robot · label), `robot` and `kind` (deployed / start / new)."""
  secs = []

  def rows_of(fn, only=None):
    out = []
    for r, ms in ROWS.items():
      if only and r not in only:
        continue
      for i, (label, p) in enumerate(ms):
        k = "deployed" if i == 0 else ("start" if i == 1 else "new")
        row = {"model": f"{r} · {label}", "robot": r, "kind": k}
        row.update(fn(r, i, label, p, ms))
        out.append(row)
    return out

  secs.append(dict(id="goals", title="目标达成", note="每条目标一格，绿色为达到，红色标出没达到的方面（具体数字见后面各表）。目标：平地全向速度（含小指令）在指令的 80–120%；直线和横移 10 s 航向中位数 ≤ 3°、最差 10% ≤ 8°，偏移 ≤ 行走距离的 5%；原地转 10 s 离开原地 ≤ 0.2 m（p90）；停止时四足 / 四轮着地、蠕动 ≤ 1 cm / 4 s、从 1 m/s 松杆刹停 ≤ 30 cm；原地踏步 20 s 漂移 ≤ 10 cm、最热髋力矩低于站立、不摔；平地站立和 1 m/s 的髋外展力矩比现部署低 ≥ 30%；上下楼带操作员 ≥ 95%、不带 ≥ 90%、速度在 0.8–1.2 倍、中途停稳 ≥ 95%；轮式前进后退四轮同时着地每档 ≥ 98%；所有测试 0 摔倒；被推后能恢复（推力测试待加）。",
    cols=[("model", "模型"), ("g_speed", "平地速度"), ("g_drift", "直线 / 横移跑偏"), ("g_spin", "原地旋转"), ("g_stop", "停止"),
          ("g_march", "原地踏步"), ("g_hip", "髋外展力矩"), ("g_stairs", "上下楼梯"), ("g_wheels", "轮式四轮着地"),
          ("g_falls", "不摔"), ("g_push", "抗推")],
    rows=rows_of(goal_cells)))

  def summary(r, i, label, p, ms):
    s, a, d, h, sh = stairs_vals(p), am_vals(p), drift_vals(p), hip_vals(p).get(0.0), stairhip_vals(p)
    h0 = hip_vals(ms[0][1]).get(0.0) if i else None
    g = gates(r, p)
    line = [v["head"][1] for k, v in d.items() if k[0] != 0.0]
    side = [v["head"][1] for k, v in d.items() if k[1] != 0.0]
    spin = [v["end"][1] for k, v in d.items() if k[2] != 0.0 and k[0] == k[1] == 0.0]
    pend = lambda x: PENDING if x is None else x
    return {
      "gate_stairs": g[0] if g[0] != NA else PENDING, "gate_flat": g[1] if g[1] != NA else PENDING,
      "gate_wheels": g[2] if g[2] != NA else PENDING,
      "reach": pend(f"{s['op']:.1f} / {s['no']:.1f}" if s else None),
      "down_speed": pend(f"{s['sp_dn']:.2f}×" if s else None),
      "flat_ok": pend(f"{a['ok']}/44" if a else None),
      "drift": pend(f"{max(line, key=abs):+.1f}° / {max(side, key=abs):+.1f}° / {max(spin):.2f} m" if line and side and spin else None),
      "stand_hip": pend(f"{h['f']:.2f}{rel(h['f'], h0['f'] if h0 else None)} / {h['r']:.2f}{rel(h['r'], h0['r'] if h0 else None)}" if h else None),
      "march": march_cell(r, p),
    }
  secs.append(dict(id="summary", title="数值总览", note="括号里是对比现部署。跑偏列：直线和横移取航向变化中位数偏得最多的一条指令，原地转取离开原地距离的 p90 最大值。原地踏步目前只训练了腿式；轮式的策略没有步态时钟输入，还不会踏步，标“未训练”。",
    cols=[("model", "模型"),
          ("reach", "上下楼到达 % 带 / 不带操作员"), ("down_speed", "下楼速度"), ("flat_ok", "平地 44 条"),
          ("drift", "跑偏 直线 / 横移 / 原地转"), ("stand_hip", "平地站立髋力矩 前 / 后 Nm"),
          ("march", "原地踏步 20 s 漂移 / 最热髋力矩变化")],
    rows=rows_of(summary)))

  def stairs(r, i, label, p, ms):
    s = stairs_vals(p)
    if not s:
      return {k: PENDING for k in ("op", "op_updn", "no", "no_updn", "sp_up", "sp_dn")}
    return {"op": f"{s['op']:.1f}", "op_updn": f"{s['op_up']:.1f} / {s['op_dn']:.1f}", "no": f"{s['no']:.1f}",
            "no_updn": f"{s['no_up']:.1f} / {s['no_dn']:.1f}", "sp_up": f"{s['sp_up']:.2f}×", "sp_dn": f"{s['sp_dn']:.2f}×"}
  secs.append(dict(id="stairs", title="1. 楼梯到达", note="地图 3 的 18 段楼梯，每段 64 台，时限 2 倍。不带操作员时上下楼都是 0.5 m/s 直走；带操作员时上楼 0.5、下楼 0.4 m/s，操作员修正方向。速度是实际 / 指令的中位数。",
    cols=[("model", "模型"), ("op", "带操作员 到达 %"), ("op_updn", "带操作员 上楼 / 下楼"), ("no", "不带操作员 到达 %"),
          ("no_updn", "不带操作员 上楼 / 下楼"), ("sp_up", "上楼速度"), ("sp_dn", "下楼速度")],
    rows=rows_of(stairs)))

  def stop(r, i, label, p, ms):
    v = stop_row(p)
    if not v:
      return {k: PENDING for k in ("traverse", "fell", "stop_ok", "worst", "slip")}
    a, b, c, w, d = [x.strip() for x in v.split("|")]
    return {"traverse": a, "fell": b, "stop_ok": c, "worst": w, "slip": d}
  secs.append(dict(id="stop", title="2. 楼梯中途停稳", note="4 档台阶高度（5–18 cm）× 上 / 下，每组 24 台。走到楼梯中间松杆，停 4 s，后 2 s 滑移小于 5 cm 且不摔算停稳。",
    cols=[("model", "模型"), ("traverse", "通过楼梯"), ("fell", "摔倒"), ("stop_ok", "中途停稳"), ("worst", "停稳最差的一组"), ("slip", "最大平均滑移 cm")],
    rows=rows_of(stop)))

  def flat(r, i, label, p, ms):
    a = am_vals(p)
    if not a:
      return {"ok": PENDING, "falls": PENDING, "fails": PENDING}
    return {"ok": f"{a['ok']}/44", "falls": str(a["falls"]), "fails": "；".join(a["fails"]) if a["fails"] else NA}
  secs.append(dict(id="flat44", title="3. 平地 44 条指令", note="每条指令 32 台，跑 8 s，取后 4 s 平均。只判有指令的轴，实际 / 指令在 80–120% 之内；32 台里有一台摔倒即不通过。",
    cols=[("model", "模型"), ("ok", "通过"), ("falls", "摔倒台数"), ("fails", "未通过的指令（实际 / 指令）")],
    rows=rows_of(flat)))

  def dline(r, i, label, p, ms):
    d = drift_vals(p)
    keys = {"f05": (0.5, 0.0, 0.0), "f10": (1.0, 0.0, 0.0), "b05": (-0.5, 0.0, 0.0)}
    return {f: (f"{pq(d[c]['head'], '°')} / {pq(d[c]['across'], ' m', '{:+.2f}')}" if c in d else PENDING) for f, c in keys.items()}
  secs.append(dict(id="drift_line", title="4a. 直线跑偏", note="平地，固定指令 10 s，不带操作员，64 台。每格：航向变化 / 侧向偏移（+ 为向左），中位数（10%~90%）。",
    cols=[("model", "模型"), ("f05", "前进 0.5 m/s"), ("f10", "前进 1.0 m/s"), ("b05", "后退 0.5 m/s")],
    rows=rows_of(dline)))

  def dside(r, i, label, p, ms):
    d = drift_vals(p)
    keys = {"l03": (0.0, 0.3, 0.0), "r03": (0.0, -0.3, 0.0), "l06": (0.0, 0.6, 0.0), "r06": (0.0, -0.6, 0.0)}
    return {f: (f"{pq(d[c]['head'], '°')} / {pq(d[c]['along'], ' m', '{:+.2f}')}" if c in d else PENDING) for f, c in keys.items()}
  secs.append(dict(id="drift_side", title="4b. 横向跑偏", note="平地，固定指令 10 s，不带操作员，64 台。每格：航向变化 / 前后偏移（+ 为向前），中位数（10%~90%）。",
    cols=[("model", "模型"), ("l03", "向左 0.3 m/s"), ("r03", "向右 0.3 m/s"), ("l06", "向左 0.6 m/s"), ("r06", "向右 0.6 m/s")],
    rows=rows_of(dside)))

  def dspin(r, i, label, p, ms):
    d = drift_vals(p)
    out = {}
    for f, w in (("w05", 0.5), ("w10", 1.0), ("w15", 1.5)):
      vs = [d[c] for c in ((0.0, 0.0, w), (0.0, 0.0, -w)) if c in d]
      out[f] = (f"{max(v['end'][0] for v in vs):.2f} / {max(v['end'][1] for v in vs):.2f}，最远 {max(v['far'][1] for v in vs):.2f}" if vs else PENDING)
    return out
  secs.append(dict(id="drift_spin", title="4c. 原地旋转跑偏", note="平地，固定指令 10 s，64 台。每格：10 s 结束时机身离开原地的距离 m（中位数 / p90），过程中最远（p90）；左右两个方向取较大值。",
    cols=[("model", "模型"), ("w05", "0.5 rad/s"), ("w10", "1.0 rad/s"), ("w15", "1.5 rad/s")],
    rows=rows_of(dspin)))

  def park(r, i, label, p, ms):
    g = stand_vals(p)
    if "rest" not in g:
      return {k: PENDING for k in ("rest", "stop", "creep", "yaw", "falls")}
    moving = [v for k, v in g.items() if k != "rest"]
    return {"rest": f"{g['rest']['creep']:.2f}",
            "stop": " / ".join(f"{g[k]['stop_cm']:.1f}" if k in g else NA for k in ("fwd 1.0", "back 1.0", "fwd 1.5")),
            "creep": f"{max(v['creep'] for v in moving):.2f}", "yaw": f"{max(v['yaw'] for v in moving):.2f}",
            "falls": str(sum(v["fell"] for v in g.values()))}
  secs.append(dict(id="parking", title="5. 停车", note="每组 16 台：给指令 3 s 后松杆，零指令 6 s。刹停距离 = 松杆到速度一直低于 0.05 m/s 走过的距离；蠕动 = 最后 4 s 的位移。",
    cols=[("model", "模型"), ("rest", "静止蠕动 cm"), ("stop", "刹停距离 cm：前进 1.0 / 后退 1.0 / 前进 1.5"), ("creep", "松杆后蠕动 cm（最大组）"),
          ("yaw", "松杆后航向漂 °（最大组）"), ("falls", "摔倒")],
    rows=rows_of(park)))

  def climb(r, i, label, p, ms):
    c = climb_vals(p)
    return {"pitch": f"{c[0]:+.1f}", "roll": f"{c[1]:+.1f}", "vx": f"{c[2]:.2f}"} if c else {"pitch": PENDING, "roll": PENDING, "vx": PENDING}
  secs.append(dict(id="climb", title="6. 爬楼姿态", note="地图 3 的 L0 段上楼，0.5 m/s，不带操作员，16 台，走到中段时的平均。",
    cols=[("model", "模型"), ("pitch", "俯仰 °"), ("roll", "横滚 °"), ("vx", "实际速度 m/s")],
    rows=rows_of(climb)))

  def hip(r, i, label, p, ms):
    b, bs = hip_vals(ms[0][1]), stairhip_vals(ms[0][1])
    h, sh = hip_vals(p), stairhip_vals(p)
    def fr(x, y):
      return f"{x['f']:.2f}{rel(x['f'], y['f']) if i and y else ''} / {x['r']:.2f}{rel(x['r'], y['r']) if i and y else ''}"
    return {"flat": fr(h[0.0], b.get(0.0)) if 0.0 in h else PENDING, "run": fr(h[1.0], b.get(1.0)) if 1.0 in h else PENDING,
            "splay": f"{h[1.0]['fo'][0]:+.1f}→{h[1.0]['fo'][1]:+.1f} / {h[1.0]['ro'][0]:+.1f}→{h[1.0]['ro'][1]:+.1f}" if 1.0 in h else PENDING}
  secs.append(dict(id="hip", title="7. 髋外展力矩", note="RMS，Nm，前腿 / 后腿；括号里是对比现部署。平地站立和 1 m/s 前进各 60 s；",
    cols=[("model", "模型"), ("flat", "平地站立"), ("run", "1 m/s 前进"), ("splay", "1 m/s 外撇 ° 前 / 后（开始→60 s）")],
    rows=rows_of(hip)))

  def march(r, i, label, p, ms):
    v = march_vals(p)
    if "stand" not in v or "march" not in v:
      return {k: PENDING for k in ("stand", "march", "hot", "steps", "drift", "falls")}
    s, m = v["stand"], v["march"]
    hot = 100 * (max(m["f"], m["r"]) / max(s["f"], s["r"]) - 1)
    return {"stand": f"{s['f']:.2f} / {s['r']:.2f}", "march": f"{m['f']:.2f} / {m['r']:.2f}", "hot": f"{hot:+.0f}%",
            "steps": f"{m['td']:.2f}", "drift": f"{m['drift']:.0f}", "falls": str(s["fell"] + m["fell"])}
  secs.append(dict(id="march", title="8. 腿式：原地踏步", note="目前只有腿式训练了原地踏步；轮式的策略没有步态时钟输入，还不会踏步。平地零指令，32 台，跑 30 s 统计后 20 s：站立和原地踏步各一遍。部署时由温控在髋电机过热或空闲时触发踏步。最热髋力矩变化 = 踏步时最热那侧髋的 RMS 力矩对比站立。",
    cols=[("model", "模型"), ("stand", "站立 髋力矩 前 / 后 Nm"), ("march", "踏步 髋力矩 前 / 后 Nm"), ("hot", "最热髋力矩变化"),
          ("steps", "每只脚步频 次/s"), ("drift", "踏步 20 s 漂移 cm"), ("falls", "摔倒")],
    rows=rows_of(march, only=LEGGED)))

  def nolift(r, i, label, p, ms):
    n = nolift_vals(p)
    if not n:
      return {"down": PENDING, "falls": PENDING, "result": PENDING}
    return {"down": " / ".join(f"{v:.1f}" for v in n["down"]), "falls": str(n["fell"]), "result": "通过" if n["ok"] else "不通过"}
  secs.append(dict(id="nolift", title="9. 轮式：前进后退四轮着地", note="只测轮式。前进 0.2 / 0.5 / 1.0 / 1.5、后退 0.5 / 1.0 m/s，每档 32 台，8 s 取后 4 s；四轮同时着地每档 ≥ 98% 且无摔倒为通过。",
    cols=[("model", "模型"), ("down", "四轮同时着地 %（按上面的顺序）"), ("falls", "摔倒"), ("result", "结果")],
    rows=rows_of(nolift, only=WHEELED)))
  inputs = [
    ("base_ang_vel", "IMU 角速度（机体系，rad/s）", "3", "3", "±0.2"),
    ("projected_gravity", "重力在机体系的方向（姿态）", "3", "3", "±0.05"),
    ("command", "指令 vx / vy / wz；摇杆按速度范围映射，三项合计小于 0.1 时置零", "3", "3", "—"),
    ("phase", "步态时钟 sin / cos，周期 0.6 s；站立时为 0，原地踏步时继续转", "2", "无（第七轮加）", "—"),
    ("joint_pos", "腿关节角减默认站姿（不含轮子转角）", "12", "12", "±0.01"),
    ("joint_vel", "关节速度，轮式含 4 个轮速", "12", "16", "±1.5"),
    ("actions", "上一步策略输出", "12", "16", "—"),
    ("heading_error", "目标航向（指令 wz 的积分）减 IMU 航向，限幅 ±0.5 rad", "1", "1", "±0.02"),
    ("单帧合计", "", "48", "54", ""),
    ("策略输入合计", "每项带最近 10 帧（0.2 s）", "480", "540", ""),
    ("策略输出", "12 个腿关节位置目标；轮式另加 4 个轮速目标", "12", "16", ""),
    ("Critic 输入（只在训练时用）", "上面各项的单帧，加真实线速度 3、地形高度 187、脚相对机身高度 4、腾空时间 4、触地 4、触地力 12", "262", "268", ""),
  ]
  secs.append(dict(id="inputs", key="name", title="附：模型输入", note="六台机型用同一套输入，只用本体感知：没有视觉、地形高度和温度。原地踏步由部署端温控决定，做法是零指令时让步态时钟继续转。第二到第六轮没有改过输入。",
    cols=[("name", "输入"), ("meaning", "含义"), ("legged", "腿式 维度"), ("wheeled", "轮式 维度"), ("noise", "训练噪声")],
    rows=[dict(name=a, meaning=b, legged=c, wheeled=d, noise=e) for a, b, c, d, e in inputs]))
  return secs


def to_markdown(secs):
  out = []
  for s in secs:
    out.append(f"## {s['title']}\n\n{s['note']}\n")
    out.append("| " + " | ".join(h for _, h in s["cols"]) + " |")
    out.append("|" + "---|" * len(s["cols"]))
    for r in s["rows"]:
      out.append("| " + " | ".join(str(r[f]) for f, _ in s["cols"]) + " |")
    out.append("")
  return "\n".join(out)


if __name__ == "__main__":
  import argparse
  import json
  ap = argparse.ArgumentParser()
  ap.add_argument("--json", help="write <id>.json per section and sections.json (specs) here")
  a = ap.parse_args()
  secs = build()
  if a.json:
    os.makedirs(a.json, exist_ok=True)
    for s in secs:
      json.dump(s["rows"], open(os.path.join(a.json, f"{s['id']}.json"), "w"), ensure_ascii=False, indent=0)
    json.dump([{"key": s.get("key", "model"), **{k: s[k] for k in ("id", "title", "note", "cols")}} for s in secs],
              open(os.path.join(a.json, "sections.json"), "w"), ensure_ascii=False, indent=1)
  else:
    print(to_markdown(secs))
