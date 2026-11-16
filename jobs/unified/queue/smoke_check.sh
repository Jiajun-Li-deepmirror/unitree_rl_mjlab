#!/bin/bash
# smoke_check.sh PID NAME (go2-depth 2026-11-16): wait for a smoke's PID (at most 15 min) and judge it: PASS when its
# records say ended done with no faults and its log has no traceback; prints "PASS ..." or "FAIL ...". From the tree
# the smoke ran in (its logs are the main tree's).
PID=$1 NAME=$2
for i in $(seq 180); do [ -d /proc/$PID ] || break; sleep 5; done
[ -d /proc/$PID ] && { echo "FAIL: pid $PID still runs after 15 min"; exit 1; }
d=$(ls -d logs/rsl_rl/go2w_u_teacher/*_$NAME 2>/dev/null | tail -n 1)
[ -n "$d" ] || { echo "FAIL: no run directory for $NAME"; exit 1; }
st=$(python3 -c "import json; s=json.load(open('$d/records/status.json')); print(s.get('ended'), s.get('faults'))" 2>&1)
tb=$(grep -c Traceback logs/jobs/unified_go2w_$NAME.log)
[ "$st" = "done []" ] && [ "$tb" = 0 ] && { echo "PASS ($d: $st)"; exit 0; }
echo "FAIL ($d: $st; tracebacks $tb)"; exit 1
