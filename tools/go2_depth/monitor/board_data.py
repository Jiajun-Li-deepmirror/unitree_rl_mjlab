#!/usr/bin/env python3
"""board_data.py OUTDIR [--manual FILE] -- the progress board's data (go2-depth 2026-11-11, the user: "配一个简单的进度看板"):
OUTDIR/robots.json (one row per robot), OUTDIR/events.json (the latest chain events) and OUTDIR/board.json (both, for the
board artifact https://claude.ai/artifact/EYUJwttsJF7dW5cQdbjrXY, uploaded as its one data file), read from the training boxes
over their ssh helper. Which robot runs where and under which TAG comes from the central queue
(~/.cache/go2-depth-work/central: queue.txt, state.json; tools/go2_depth/queue/central_queue.py): a robot's row reads
the box of its latest job, and only the blind-log lines after that TAG's "blind chain <TAG>:" header.
--manual FILE: rows (and events) for robots on machines without a route (go2-home's As2), merged as they are:
{"robots": [{"robot": "As2", "machine": "go2-home", ...}], "events": [...]}.
Row: robot, machine, tag, round, stage (T / M / P / package), iter, max, check (the last one: avgK sN, part),
verdict, status (queued / smoke / training / checking / passed / failed / fault / paused / held), alarm (an open curve
alarm), note, at (box time of the last event), updated (UTC).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SSH = "/home/dm/.cache/go2-depth-work/ssh/r"
PORTS = {"t1": "30714", "t2": "31365"}
CENTRAL = Path("/home/dm/.cache/go2-depth-work/central")
ORDER = ["Go2", "As2W", "Go2W", "B2", "As2", "B2W"]   # the user's order (11-11)

REMOTE = r'''
import json, os, re, sys
os.chdir("/root/gpufree-data/go2-depth/depth_fe")
jobs = json.loads(sys.argv[1])   # [{"robot", "tag", "round", "id"}]
EVENT = re.compile(r"started|starts \(|PICK|PASS|FAIL|TOOL FAULT|PAUSED|DELIVERY|tinynav:|widened|CURVE ALARM|curve back|BLIND VIDEOS|smoke")
def tail(path, n=600):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 300000))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []
out, events = [], []
central = tail("logs/jobs/central/central.txt", 400)
for j in jobs:
    R, tag, rnd, jid = j["robot"], j["tag"], j["round"], j["id"]
    r = R.lower()
    row = {"robot": R, "tag": tag, "round": rnd, "stage": "", "iter": None, "max": None, "check": "", "verdict": "",
           "status": "training", "alarm": "", "note": "", "at": ""}
    cj = [l for l in central if f"central {jid} " in l]
    smoke = bool(cj) and "smoke run first" in cj[-1]
    if cj:
        row["at"] = cj[-1][:14]
        last = cj[-1]
        if "SMOKE FAILED" in last: row.update(status="failed", note="smoke run: " + last.split("SMOKE FAILED", 1)[1][:160])
        elif "PASSED" in last: row.update(status="passed")
        elif "FAILED" in last: row.update(status="failed", note=last.split(": ", 1)[1][:160])
    blind = tail(f"logs/jobs/smoke/blind_{r}.txt" if smoke else f"logs/jobs/blind_{r}.txt")
    run_tag = f"smoke{jid}" if smoke else tag
    start = max([i for i, l in enumerate(blind) if f"blind chain {run_tag}:" in l] or [len(blind)])
    lines = blind[start:]
    alarms = {}
    for l in lines:
        m = re.match(r"(\S+ \S+) %s (T|M|P) started \(pid (\d+); (\S+), (\d+) it from (\S+)(?: model_(\d+))?" % R, l)
        if m:
            row.update(stage=m.group(2), at=m.group(1), check="", verdict="")
            row["max"] = int(m.group(7) or 0) + int(m.group(5)); row["_pid"] = m.group(3)
        m = re.match(r"(\S+ \S+) %s (T|M|P) avg(\d+) s(\d) \(([\w ]+)\): (.*)" % R, l)
        if m: row.update(check=f"avg{m.group(3)} s{m.group(4)} ({m.group(5)})", verdict=m.group(6)[:160], at=m.group(1))
        m = re.match(r"(\S+ \S+) %s (T|M|P) CURVE ALARM (\w+) (.*)" % R, l)
        if m: alarms[m.group(3)] = f"{m.group(2)} {m.group(3)} {m.group(4)}"[:200]
        m = re.match(r"(\S+ \S+) %s (T|M|P) curve back: (\w+)" % R, l)
        if m: alarms.pop(m.group(3), None)
        if "TOOL FAULT" in l: row.update(status="fault", note=l[15:200], at=l[:14])
        if "PAUSED" in l: row.update(status="paused", note=l[15:200], at=l[:14])
        if "BLIND DELIVERY" in l: row.update(stage="package", note=l.split("DELIVERY ", 1)[1][:120], at=l[:14])
        if "BLIND VIDEOS" in l: row["note"] = (row["note"] + " | " if row["note"] else "") + l[15:160]
    row["alarm"] = "; ".join(alarms.values())
    pid = row.pop("_pid", None)
    if row["status"] == "training":
        if smoke:
            row["status"] = "smoke"
        elif pid and not os.path.exists(f"/proc/{pid}") and row["stage"] in ("T", "M", "P"):
            row["status"] = "checking"
    if row["stage"] in ("T", "M", "P"):
        lf = f"logs/jobs/blind_{r}_bl2{row['stage']}_{run_tag}.log"
        for x in reversed(tail(lf, 300)):
            m = re.search(r"Learning iteration (\d+)/(\d+)", x)
            if m: row["iter"], row["max"] = int(m.group(1)), int(m.group(2)); break
    out.append(row)
    for l in lines[-300:] + cj[-20:]:
        if EVENT.search(l) and re.match(r"\d\d-\d\d \d\d:\d\d:\d\d", l):
            events.append({"at": l[:14], "robot": R, "text": l[15:260]})
print(json.dumps({"robots": out, "events": events}))
'''


def queue_jobs() -> tuple[dict[str, dict], list[str]]:
    """robot -> its latest started job (box, tag, round, id); the robots still queued, in order."""
    state = json.loads((CENTRAL / "state.json").read_text()) if (CENTRAL / "state.json").exists() else {"jobs": []}
    latest: dict[str, dict] = {}
    for s in state["jobs"]:
        R, tag, rnd = s["key"].split()[:3]
        latest[R] = {"robot": R, "tag": tag, "round": rnd, "id": s["id"], "box": s["box"], "exit": s.get("exit")}
    queued = []
    if (CENTRAL / "queue.txt").exists():
        for l in (CENTRAL / "queue.txt").read_text().splitlines():
            t = l.split("#", 1)[0].split()
            if len(t) >= 3 and t[0] not in latest and t[0] not in queued:
                queued.append(t[0])
    return latest, queued


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--manual", default=None)
    a = ap.parse_args()
    latest, queued = queue_jobs()
    robots, events = [], []
    for box, port in PORTS.items():
        jobs = [j for j in latest.values() if j["box"] == box]
        if not jobs:
            continue
        cmd = f"python3 - '{json.dumps(jobs)}' <<'PYEOF'\n{REMOTE}\nPYEOF"
        try:
            res = subprocess.run([SSH, port, cmd], capture_output=True, text=True, timeout=90, stdin=subprocess.DEVNULL)
            d = json.loads(res.stdout.strip().splitlines()[-1])
        except Exception as e:  # a box that does not answer: its robots say so
            d = {"robots": [{"robot": j["robot"], "tag": j["tag"], "status": "unknown",
                             "note": f"{box} did not answer ({type(e).__name__})"} for j in jobs], "events": []}
        for row in d["robots"]:
            row["machine"] = box
        robots += d["robots"]
        events += [dict(e, machine=box) for e in d["events"]]
    for R in queued:
        robots.append({"robot": R, "machine": "", "status": "queued", "note": "waiting for a free GPU (central queue)"})
    if a.manual and Path(a.manual).exists():
        m = json.loads(Path(a.manual).read_text())
        robots += m.get("robots", [])
        events += m.get("events", [])
    robots.sort(key=lambda r: ORDER.index(r["robot"]) if r["robot"] in ORDER else 99)
    events.sort(key=lambda e: e["at"], reverse=True)
    seen, uniq = set(), []
    for e in events:
        k = (e["at"], e["robot"], e["text"][:80])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for row in robots:
        for k in ("machine", "tag", "round", "stage", "check", "verdict", "alarm", "note", "at"):
            row.setdefault(k, "")
        for k in ("iter", "max"):
            row.setdefault(k, None)
        row["updated"] = now
    (out / "robots.json").write_text(json.dumps(robots, ensure_ascii=False, indent=1))
    (out / "events.json").write_text(json.dumps(uniq[:40], ensure_ascii=False, indent=1))
    # one file for the board artifact (its two tables read it through loaders): one upload per update
    (out / "board.json").write_text(json.dumps({"robots": robots, "events": uniq[:40]}, ensure_ascii=False, indent=1))
    print(f"{len(robots)} robots, {len(uniq[:40])} events -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
