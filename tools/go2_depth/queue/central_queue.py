#!/usr/bin/env python3
"""central_queue.py [--every S] [--once] -- the training boxes' one queue (go2-depth 2026-11-11, the user: "中心队列：任何一块
GPU 空出来就自动接手"; go2-home keeps its own local queue on the same code and reports by message).

Runs on the workstation, the one machine that reaches both boxes. Every S seconds (default 60):
  1. For each job it started: done when the box has logs/jobs/central/<ID>.rc (jobs/central_job.sh writes it); a job
     whose process is gone without one died (a tool fault: its box is held).
  2. For each box that is free -- none of our chain scripts running, its GPU under FREE_MIB used (another team's run
     keeps it busy), no hold -- it starts the first pending job allowed there: jobs/central_job.sh (the smoke run unless
     one passed with this code and recipe, then jobs/full_chain.sh).
Files in ~/.cache/go2-depth-work/central/ (runtime state, not in the repo):
  queue.txt   the jobs in order, one per line: R TAG STAGE [box=t1|t2] [smoke=0] [VAR=value ...]; '#' comments.
              A line is pending until a job with its R TAG STAGE was started (state.json); edit the file freely --
              a rerun goes in front, with box= the box that holds the reused picks (jobs/full_chain.sh's
              REPRO_BLIND_FROM / REPRO_FROM_TAG read that box's logs).
  state.json  the started jobs (id, box, R, TAG, STAGE, start, end, exit).
  central.log what it did, one line each (tools/go2_depth/monitor/events.py shows it).
  hold        no new job anywhere while it exists; hold_t1 / hold_t2 only that box. Written by the queue itself when a
              smoke run fails (hold: the pipeline is broken) or a job dies or ends with a setup error (that box);
              removed by hand once fixed.
Exits: a failed round (1) is a result: the queue goes on and the rerun is queued by hand (the rerun rule,
doc/acceptance.md §5). One robot never runs on two boxes at once.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BOXES = {"t1": "30714", "t2": "31365"}
SSH = "/home/dm/.cache/go2-depth-work/ssh/r"
D = "/root/gpufree-data/go2-depth/depth_fe"
HOME = Path("/home/dm/.cache/go2-depth-work/central")
FREE_MIB = 3000
CHAIN = r"jobs/(full_chain|smoke_chain|repro_queue|central_job|blind_chain|fe_chain|v2_chain|blind_videos)\.sh"
ROBOTS = ("Go2", "Go2W", "As2", "As2W", "B2", "B2W")


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    with open(HOME / "central.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def ssh(box: str, cmd: str, timeout: int = 40) -> str | None:
    try:
        p = subprocess.run([SSH, BOXES[box], cmd], capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return out or None
    return p.stdout if p.returncode == 0 else None


def parse(line: str) -> dict | None:
    t = line.split("#", 1)[0].split()
    if not t:
        return None
    if len(t) < 3 or t[0] not in ROBOTS:
        raise ValueError(f"queue.txt: not 'R TAG STAGE ...': {line.strip()}")
    j = {"R": t[0], "TAG": t[1], "STAGE": t[2], "box": None, "smoke": True, "env": []}
    for o in t[3:]:
        k, _, v = o.partition("=")
        if k == "box":
            if v not in BOXES:
                raise ValueError(f"queue.txt: box {v}?")
            j["box"] = v
        elif k == "smoke":
            j["smoke"] = v != "0"
        elif k and k == k.upper() and v != "":
            j["env"].append(o)
        else:
            raise ValueError(f"queue.txt: option {o}?")
    return j


def key(j: dict) -> str:
    return f"{j['R']} {j['TAG']} {j['STAGE']}"


def box_state(box: str) -> dict | None:
    """busy: our chain scripts' pids; mib: GPU memory used; rcs: the ended jobs' exits."""
    cmd = (f"cd {D} || exit 1; echo PIDS $(ps -eo pid,args | grep -E '{CHAIN}' | grep -v -E 'grep|ps -eo' "
           "| awk '{print $1}' | tr '\\n' ' '); "
           "echo MIB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1); "
           "for f in logs/jobs/central/*.rc; do [ -f \"$f\" ] && echo RC $(basename $f .rc) $(cat $f); done; "
           "for f in logs/jobs/central/*.pid; do [ -f \"$f\" ] || continue; i=$(basename $f .pid); "
           "kill -0 $(cat $f) 2> /dev/null && echo ALIVE $i; done; true")
    out = ssh(box, cmd)
    if out is None or "MIB" not in out:
        return None
    st = {"busy": [], "mib": 0, "rcs": {}, "alive": set()}
    for l in out.splitlines():
        w = l.split()
        if not w:
            continue
        if w[0] == "PIDS":
            st["busy"] = w[1:]
        elif w[0] == "MIB" and len(w) > 1 and w[1].isdigit():
            st["mib"] = int(w[1])
        elif w[0] == "RC" and len(w) == 3:
            st["rcs"][w[1]] = w[2]
        elif w[0] == "ALIVE" and len(w) == 2:
            st["alive"].add(w[1])
    return st


def start(box: str, j: dict, jid: str) -> bool:
    env = " ".join(j["env"] + ([] if j["smoke"] else ["CENTRAL_SMOKE=0"]))
    pre = "CENTRAL_SMOKE=0 " if not j["smoke"] else ""
    cmd = (f"cd {D} && mkdir -p logs/jobs/central && {pre}setsid nohup bash jobs/central_job.sh {j['R']} {j['TAG']} "
           f"{j['STAGE']} {jid} {' '.join(j['env'])} > logs/jobs/central/{jid}.out 2>&1 < /dev/null & disown; "
           f"sleep 3; echo STARTED $(cat logs/jobs/central/{jid}.pid 2> /dev/null)")
    ssh(box, cmd, timeout=25)
    time.sleep(2)
    out = ssh(box, f"cd {D} && p=$(cat logs/jobs/central/{jid}.pid 2> /dev/null) && kill -0 $p 2> /dev/null && echo $p")
    ok = bool(out and out.strip().isdigit())
    log(f"[{box}] {'started' if ok else 'COULD NOT START'} {jid}: {key(j)}{' ' + env if env else ''}"
        f"{' (pid ' + out.strip() + ')' if ok else ''}")
    return ok


MEANING = {"0": "PASSED", "1": "FAILED (a result: queue the rerun by the rerun rule)", "3": "not ready",
           "5": "SMOKE FAILED (the pipeline is broken: everything held)"}


def tick(state: dict) -> None:
    if not (HOME / "queue.txt").exists():
        return
    jobs = []
    for l in (HOME / "queue.txt").read_text().splitlines():
        try:
            j = parse(l)
        except ValueError as e:
            log(f"queue.txt not read: {e}")
            return
        if j:
            jobs.append(j)
    boxes = {b: box_state(b) for b in BOXES}
    for b, st in boxes.items():
        if st is None:
            if not state["down"].get(b):
                log(f"[{b}] does not answer (ssh)")
            state["down"][b] = True
            continue
        if state["down"].get(b):
            log(f"[{b}] answers again")
        state["down"][b] = False
        for s in state["jobs"]:
            if s["box"] != b or s.get("end"):
                continue
            rc = st["rcs"].get(s["id"])
            if rc is None and s["id"] not in st["alive"] and time.time() - s["start"] > 120:
                rc = "died"
            if rc is None:
                continue
            s["end"], s["exit"] = time.time(), rc
            hrs = (s["end"] - s["start"]) / 3600
            log(f"[{b}] {s['id']} {s['key']}: {MEANING.get(rc, 'ended with exit ' + rc)} after {hrs:.1f} h")
            if rc == "5":
                (HOME / "hold").write_text(f"{s['id']} smoke failed\n")
            elif rc not in ("0", "1", "3"):
                (HOME / f"hold_{b}").write_text(f"{s['id']} exit {rc}\n")
                log(f"[{b}] held (remove {HOME}/hold_{b} once fixed)")
    started = {s["key"] for s in state["jobs"]}
    running = {s["key"].split()[0] for s in state["jobs"] if not s.get("end")}
    if (HOME / "hold").exists():
        return
    for b, st in boxes.items():
        if st is None or (HOME / f"hold_{b}").exists():
            continue
        if any(s["box"] == b and not s.get("end") for s in state["jobs"]) or st["busy"] or st["mib"] >= FREE_MIB:
            continue
        for j in jobs:
            if key(j) in started or j["R"] in running or (j["box"] and j["box"] != b):
                continue
            jid = f"c{time.strftime('%m%d%H%M')}{j['R'].lower()}"
            if start(b, j, jid):
                state["jobs"].append({"id": jid, "box": b, "key": key(j), "start": time.time()})
                started.add(key(j))
                running.add(j["R"])
            else:
                (HOME / f"hold_{b}").write_text(f"could not start {key(j)}\n")
            break


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    HOME.mkdir(parents=True, exist_ok=True)
    sf = HOME / "state.json"
    state = json.loads(sf.read_text()) if sf.exists() else {"jobs": [], "down": {}}
    log(f"central queue up (every {a.every} s; {HOME}/queue.txt)")
    while True:
        tick(state)
        tmp = sf.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(sf)
        if a.once:
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
