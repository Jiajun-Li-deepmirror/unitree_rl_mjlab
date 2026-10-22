"""Run scripts/go2_blind/eval_map3.py with its scenarios split over parallel processes
on one GPU and merge their outputs into one file of the same format.

Each scenario still runs alone in its own environment with the same seed, exactly as
in eval_map3.py; only the waiting is gone (one scenario of 64 robots uses ~330 MB of
GPU memory and leaves the GPU mostly idle).

Usage: python map3_par.py --out FILE [--shards 6] -- <eval_map3.py arguments>
(run from the repo root, like eval_map3.py).
"""
import argparse
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--shards", type=int, default=6)
p.add_argument("rest", nargs=argparse.REMAINDER)
a = p.parse_args()
rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
cmd = [sys.executable, "scripts/go2_blind/eval_map3.py", *rest]
listing = subprocess.run(cmd + ["--list"], capture_output=True, text=True, check=True).stdout
names = [line.split()[0] for line in listing.splitlines() if line.strip() and not line.startswith(" ")]
shards = [names[i :: a.shards] for i in range(a.shards) if names[i :: a.shards]]
t0 = time.time()
procs = []
for i, shard in enumerate(shards):
  f = open(f"{a.out}.part{i}", "w")
  procs.append((subprocess.Popen(cmd + ["--scenarios", *shard], stdout=f, stderr=subprocess.STDOUT), f))
codes = [pr.wait() for pr, _ in procs]
for _, f in procs:
  f.close()
lines, header = {}, None
for i in range(len(shards)):
  for line in open(f"{a.out}.part{i}"):
    if line.startswith("[map3]"):
      header = header or line
    elif "scenario=" in line:
      lines[line.split("scenario=")[1].split()[0]] = line
with open(a.out, "w") as out:
  out.write(f"merged from {len(shards)} parallel shards in {time.time() - t0:.0f} s, exit codes {codes}\n")
  out.write(header or "[map3] (no header)\n")
  for n in names:
    out.write(lines.get(n, f"  scenario={n}  MISSING\n"))
print(f"{a.out}: {sum(n in lines for n in names)}/{len(names)} scenarios in {time.time() - t0:.0f} s")
sys.exit(0 if all(c == 0 for c in codes) and len(lines) == len(names) else 1)
