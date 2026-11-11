#!/usr/bin/env python3
"""events.py [--every S] -- the training boxes' chain events as one line each on stdout (go2-depth 2026-11-11, the user:
"全部改成事件驱动：每个阶段开始、检查出结果、训练崩溃，都主动通知"). Meant for Claude Code's Monitor: every line is a
notification. Polls the chain logs of t1 / t2 every S seconds (default 60) over the boxes' ssh helper and prints the new
lines that are events:
  a stage started / picked / passed; a check verdict; a tool fault, pause or crash sign; a delivery; a tinynav verdict;
  the queue starting / ending a job; the input check; smoke results
(not the "wrote ... averaged" lines). The first poll only takes the files' sizes: nothing old is replayed. A box that
does not answer is reported once, and again when it is back. State: logs/monitor/events_state.json in this repo.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

BOXES = {"t1": "30714", "t2": "31365"}
SSH = "/home/dm/.cache/go2-depth-work/ssh/r"
D = "/root/gpufree-data/go2-depth/depth_fe"
FILES = "logs/jobs/blind_*.txt logs/jobs/fe_*.txt logs/jobs/repro_*.txt logs/jobs/smoke/*.txt logs/jobs/central/central.txt"
# the workstation's central queue (tools/go2_depth/queue/central_queue.py): every line of its log is an event
CENTRAL_LOG = Path("/home/dm/.cache/go2-depth-work/central/central.log")
EVENT = re.compile(
    r"started|PICK|PASS|FAIL|TOOL FAULT|PAUSED|removed, going on|DELIVERY|tinynav|widened|inputs \(actor|inputs: P adds|BLIND VIDEOS"
    r"|STAGE-1|STUDENT|TEACHER|repro queue .*(starts|passed|FAILED|skipped|stopped|all done)|no new run dir"
    r"|ended before|no checkpoint|SMOKE|smoke|inserted|runs as|reused from|Traceback|Error|central ")
SKIP = re.compile(r"^wrote |actor averaged over")
STATE = Path(__file__).resolve().parents[3] / "logs/monitor/events_state.json"


def poll(box: str, offsets: dict[str, int]) -> tuple[dict[str, int], list[str]] | None:
    """New lines of every chain log on BOX since OFFSETS (path -> bytes); None when the box does not answer."""
    script = (f"cd {D} || exit 1; for f in {FILES}; do [ -f \"$f\" ] || continue; "
              "s=$(stat -c %s \"$f\"); echo \"@@ $f $s\"; done")
    try:
        out = subprocess.run([SSH, BOXES[box], script], capture_output=True, text=True, timeout=60).stdout
    except subprocess.TimeoutExpired:
        return None
    sizes = {p: int(s) for _, p, s in (l.split() for l in out.splitlines() if l.startswith("@@ ") and len(l.split()) == 3)}
    if not sizes:
        return None
    lines: list[str] = []
    grow = [p for p, s in sizes.items() if p in offsets and s > offsets[p]]
    if grow:
        cmd = f"cd {D}; " + "; ".join(f"echo '@@@ {p}'; tail -c +{offsets[p] + 1} '{p}' | head -c {sizes[p] - offsets[p]}" for p in grow)
        try:
            text = subprocess.run([SSH, BOXES[box], cmd], capture_output=True, text=True, timeout=90).stdout
        except subprocess.TimeoutExpired:
            return None
        cur = ""
        for l in text.splitlines():
            if l.startswith("@@@ "):
                cur = l[4:].replace("logs/jobs/", "")
                continue
            if l.strip() and EVENT.search(l) and not SKIP.search(l):
                lines.append(f"[{box} {cur}] {l[:300]}")
    # a file new since the last poll counts from its start next time only if it is small (a fresh log)
    new = {p: (sizes[p] if p in offsets or not offsets else 0) for p in sizes}
    for p in grow:
        new[p] = sizes[p]
    return new, lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=60.0)
    a = ap.parse_args()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    down: set[str] = set()
    first = {b: b not in state for b in BOXES}
    while True:
        for box in BOXES:
            r = poll(box, state.get(box, {}))
            if r is None:
                if box not in down:
                    print(f"[{box}] does not answer (ssh)", flush=True)
                    down.add(box)
                continue
            if box in down:
                print(f"[{box}] answers again", flush=True)
                down.discard(box)
            offsets, lines = r
            if first[box]:
                first[box] = False
            else:
                for l in lines:
                    print(l, flush=True)
            state[box] = offsets
        if CENTRAL_LOG.exists():
            size = CENTRAL_LOG.stat().st_size
            off = state.get("central", size)
            if size > off:
                with open(CENTRAL_LOG, "rb") as f:
                    f.seek(off)
                    for l in f.read(size - off).decode(errors="replace").splitlines():
                        if l.strip():
                            print(f"[queue] {l[:300]}", flush=True)
            state["central"] = size
        STATE.write_text(json.dumps(state))
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
