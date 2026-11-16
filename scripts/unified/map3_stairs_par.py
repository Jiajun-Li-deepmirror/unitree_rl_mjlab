#!/usr/bin/env python3
"""map3_stairs_par.py --out FILE [--shards 3] -- TASK CHECKPOINT [map3_stairs_probe.py arguments] (go2-depth 2026-11-16,
go2-home: the acceptance's wall probe (items 21 / 28) in parallel shards, as map3_par.py runs eval_map3): the up flights
(map3_stairs_probe.py --list) split round robin over SHARDS processes, each through jobs/unified/u_eval.py with
--flights <its share> --dump <its file>, then one --merge of the dumps writes the whole report to FILE -- the same
lines a single run prints, the per-robot readings of every flight pooled. A shard's own output: FILE.shard<i>.log.
Exit 0 when every shard and the merge ran; else 1 (FILE says which).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--shards", type=int, default=3)
p.add_argument("rest", nargs=argparse.REMAINDER)
a = p.parse_args()
rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
probe = [sys.executable, "jobs/unified/u_eval.py", "scripts/unified/map3_stairs_probe.py"]
lst = subprocess.run(probe + rest + ["--list"], capture_output=True, text=True)
names = [ln.strip() for ln in lst.stdout.splitlines() if ln.strip() and " " not in ln.strip() and ln.strip()[0].isalnum()
         and not ln.startswith(("checkpoint", "task"))]
if lst.returncode != 0 or not names:
  open(a.out, "w").write(f"map3_stairs_par: the flight list failed (rc {lst.returncode})\n{lst.stdout[-2000:]}\n{lst.stderr[-2000:]}\n")
  sys.exit(1)
n = max(1, min(a.shards, len(names)))
parts = [names[i::n] for i in range(n)]
procs = []
for i, part in enumerate(parts):
  dump = f"{a.out}.shard{i}.pkl"
  if os.path.exists(dump):
    os.remove(dump)
  log = open(f"{a.out}.shard{i}.log", "w")
  procs.append((subprocess.Popen(probe + rest + ["--flights", *part, "--dump", dump], stdout=log, stderr=subprocess.STDOUT), log, dump, part))
bad = []
for pr, log, dump, part in procs:
  rc = pr.wait()
  log.close()
  if rc != 0 or not os.path.exists(dump):
    bad.append(f"shard {part}: rc {rc} (see {log.name})")
if bad:
  open(a.out, "w").write("map3_stairs_par: " + "; ".join(bad) + "\n")
  # the failed shards' tails on stdout too: accept.sh's run() reads its OUT for "out of memory" to wait and retry
  for pr, log, dump, part in procs:
    if pr.returncode != 0 or not os.path.exists(dump):
      print(f"== shard {part} (rc {pr.returncode}):")
      print("".join(open(log.name, errors="replace").readlines()[-15:]))
  sys.exit(1)
with open(a.out, "w") as f:
  rc = subprocess.run(probe + rest + ["--merge", *[d for _p, _l, d, _part in procs]], stdout=f, stderr=subprocess.STDOUT).returncode
sys.exit(0 if rc == 0 else 1)
