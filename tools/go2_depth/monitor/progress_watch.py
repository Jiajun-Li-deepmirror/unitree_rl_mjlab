#!/usr/bin/env python3
"""progress_watch.py [--reset] -- is anything stuck? (go2-depth 2026-11-11, the user: "loop里需要监控训练进度，如果长时间没有
任何变化就要开始排查问题了"). For the 30-minute loop: reads the trainings' state on t1 / t2, compares it with the last
call's (~/.cache/go2-depth-work/central/watch_state.json) and prints one line per watched value with how long it has
not changed, then the alarms:

  STALL      a training's iteration has not moved for ITER_STALL_MIN (15) min: hung, crashed, or waiting
  SLOW       the unified teacher's progress p has not risen for P_STALL_H (3) h once past P_GRACE_IT (2000) it, or
             its gate verdict has not changed for GATE_STALL_H (6) h
  NOCHECK    a blind chain's phase has had no new check line for CHECK_STALL_H (2.5) h while its training runs
  IDLE       a box's GPU under IDLE_UTIL (10) % utilization with less than IDLE_MIB (3000) MiB used
  NODAEMON   the central queue's daemon is not running
Exit 0 always; the alarms are the output's lines starting with those words.

Watched: t1 the unified Go2 teacher (logs/jobs/unified_go2_ufull_s0.log, logs/jobs/unified_go2.txt), t2 the Go2
payload fine-tune gp2 (logs/jobs/blind_go2_bl2P_gp2.log, logs/jobs/blind_go2_gp2.txt; gp1 until 11-11 18:5x). WATCH in the code lists them.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

SSH = "/home/dm/.cache/go2-depth-work/ssh/r"
D = "/root/gpufree-data/go2-depth/depth_fe"
STATE = Path("/home/dm/.cache/go2-depth-work/central/watch_state.json")
ITER_STALL_MIN, P_STALL_H, P_GRACE_IT, GATE_STALL_H, CHECK_STALL_H = 15, 3.0, 2000, 6.0, 2.5
IDLE_UTIL, IDLE_MIB = 10, 3000

# name -> (port, training log, chain log, kind)
WATCH = {
  "t1 unified Go2": ("30714", "logs/jobs/unified_go2_ufull_s0.log", "logs/jobs/unified_go2.txt", "unified"),
  # 11-11 21:2x: gp3 (from gp2 model_13700 + stairstand4,standlong) trains on t2
  "t2 gp3 Go2": ("31365", "logs/jobs/blind_go2_bl2P_gp3.log", "logs/jobs/blind_go2_gp3.txt", "blind"),
}


def remote(port: str, cmd: str) -> str:
  try:
    return subprocess.run([SSH, port, cmd], capture_output=True, text=True, timeout=60,
                          stdin=subprocess.DEVNULL).stdout
  except subprocess.TimeoutExpired:
    return ""


def read(port: str, train_log: str, chain_log: str) -> dict:
  cmd = (f"cd {D}; echo ITER $(grep -oE 'Learning iteration [0-9]+' {train_log} 2>/dev/null | tail -n 1 | grep -oE '[0-9]+$'); "
         f"echo P $(tail -n 400 {train_log} 2>/dev/null | grep -E 'u_progress/p' | tail -n 1 | grep -oE '[0-9.]+$'); "
         f"echo LEVEL $(tail -n 400 {train_log} 2>/dev/null | grep -E 'u_progress/level' | tail -n 1 | grep -oE '[0-9.]+$'); "
         f"echo GATE $(grep -E 'NOT YET|PASS|PICK' {chain_log} 2>/dev/null | tail -n 1 | cut -c16-); "
         f"echo CHECK $(grep -E ' avg[0-9]+ s[0-9]|tinynav' {chain_log} 2>/dev/null | tail -n 1 | cut -c1-140); "
         f"echo ALIVE $(ps -eo args | grep -c '[s]cripts/train.py'); "
         "echo GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')")
  out = {}
  for line in remote(port, cmd).splitlines():
    k, _, v = line.partition(" ")
    out[k] = v.strip()
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--reset", action="store_true")
  a = ap.parse_args()
  now = time.time()
  state = {} if a.reset or not STATE.exists() else json.loads(STATE.read_text())
  alarms = []

  def track(key: str, value) -> float:
    """Hours since `value` last changed (0 when it just did)."""
    old = state.get(key)
    if old is None or old["value"] != value:
      state[key] = {"value": value, "since": now}
      return 0.0
    return (now - old["since"]) / 3600.0

  for name, (port, train_log, chain_log, kind) in WATCH.items():
    r = read(port, train_log, chain_log)
    if not r:
      alarms.append(f"STALL {name}: the box does not answer")
      continue
    it = int(r["ITER"]) if r.get("ITER", "").isdigit() else None
    h_it = track(f"{name}/iter", it)
    line = f"{name}: iteration {it} ({h_it * 60:.0f} min unchanged)"
    if kind != "checks" and it is not None and h_it * 60 >= ITER_STALL_MIN:
      alarms.append(f"STALL {name}: iteration {it} unchanged for {h_it * 60:.0f} min (trainings alive: {r.get('ALIVE')})")
    if kind == "unified":
      p = r.get("P") or None
      h_p = track(f"{name}/p", p)
      h_gate = track(f"{name}/gate", r.get("GATE"))
      line += f", p {p} ({h_p:.1f} h), level {r.get('LEVEL')}, gate '{r.get('GATE')}' ({h_gate:.1f} h)"
      if it is not None and it >= P_GRACE_IT and h_p >= P_STALL_H and (p is None or float(p) < 1.0):
        alarms.append(f"SLOW {name}: progress p {p} has not risen for {h_p:.1f} h (iteration {it}, level {r.get('LEVEL')})")
      if h_gate >= GATE_STALL_H:
        alarms.append(f"SLOW {name}: gate verdict unchanged for {h_gate:.1f} h: {r.get('GATE')}")
    else:
      h_ck = track(f"{name}/check", r.get("CHECK"))
      line += f", last check '{(r.get('CHECK') or '')[:90]}' ({h_ck:.1f} h)"
      if (r.get("ALIVE", "0") != "0" or kind == "checks") and h_ck >= CHECK_STALL_H:
        alarms.append(f"NOCHECK {name}: no new check line for {h_ck:.1f} h")
    gpu = r.get("GPU", "")
    if "," in gpu:
      util, mib = (int(x) for x in gpu.split(",")[:2])
      line += f", GPU {util}% {mib} MiB"
      if util < IDLE_UTIL and mib < IDLE_MIB:
        alarms.append(f"IDLE {name.split()[0]}: GPU {util}% / {mib} MiB")
    print(line)
  daemon = subprocess.run(["bash", "-c", "ps -eo args | grep -c '[p]ython3 -u tools/go2_depth/queue/central_queue.py'"],
                          capture_output=True, text=True).stdout.strip()
  if daemon in ("", "0"):
    alarms.append("NODAEMON the central queue's daemon is not running")
  STATE.parent.mkdir(parents=True, exist_ok=True)
  STATE.write_text(json.dumps(state))
  for x in alarms:
    print(x)
  if not alarms:
    print("OK nothing stuck")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
