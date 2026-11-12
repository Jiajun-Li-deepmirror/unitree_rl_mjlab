#!/bin/bash
# stop_check.sh R TEACHER_CKPT [OUT_PREFIX] (unitree-rl-mjlab-62 2026-11-12, go2-depth's decision 11-12): the teacher's
# stops on flat ground, checked before its students train. The probe teacher (uprobe_s0 12799) fell in 185-193 of 768
# robots after a step release from 1.5 m/s (0.5-1.25 s after it, the body on the ground, pitched ~36 deg), while the
# deployed go2_blind_v4 fell in none; the teacher gate judges the training constraints only, and a student copies
# its teacher.
# scripts/blind/eval/eval_standing.py on Unitree-<R>-U-Teacher-Flat (the acceptance's stop test: 7 groups, the
# command 3 s ramped in at 2 m/s^2 / 3 rad/s^2, then a step to zero for 6 s), unloaded, once per scan mode (U_SCAN=on,
# off), STOP_ENVS (768) robots per group. A recipe check, not the acceptance (the students' acceptance judges 0
# falls), on each group's falls pooled over the two scan modes (2 x 768 robots): a group's pooled share over STOP_MAX
# (0.01) -> FAIL, exit 1; falls in (0, STOP_MAX] -> WARN, exit 0; none -> PASS, exit 0; a run without the table
# (twice; out of memory: 5 min apart) -> FAULT, exit 4. Pooled, not per mode: s0 model_2300 twice gave fwd 1.5
# 2 + 1 and 8 + 0 of 768 per mode (~0.5% true, a 1% bar per 768-robot cell flipped on the draw), the probe 193 + 179.
# Prints one line "STOPCHECK PASS|WARN|FAIL|FAULT <ckpt>: pooled <worst group>; scan on <groups>; scan off <groups>"
# (each group: falls and the creep mean), also appended to logs/checks/u_<r>_tstop.txt. Files
# <OUT_PREFIX>_scan{on,off}.txt (default logs/checks/u_<r>_tstop_<run>_<model>).
set -u
R=$1 CK=$2 r=${1,,}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1 U_PAYLOAD=0,0,0,0
N=${STOP_ENVS:-768} MAX=${STOP_MAX:-0.01}
OUT=${3:-logs/checks/u_${r}_tstop_$(basename $(dirname $CK))_$(basename $CK .pt)}
SUM=logs/checks/u_${r}_tstop.txt
mkdir -p logs/checks $(dirname $OUT)
say() { echo "$*"; echo "$(date '+%m-%d %H:%M:%S') $*" >> $SUM; }
parts=() rows=""
for m in on off; do
  f=${OUT}_scan$m.txt
  for try in 1 2 3 4 5 6 7; do
    U_SCAN=$m $PY jobs/unified/u_eval.py scripts/blind/eval/eval_standing.py --task Unitree-$R-U-Teacher-Flat $CK \
      --envs-per-group $N --lin-accel 2.0 --yaw-accel 3.0 > $f 2>&1
    grep -qE '^ *fwd 1\.5 .*\| *[0-9]+/[0-9]+ *$' $f && break
    if grep -qiE "out of memory" $f && [ $try -lt 7 ]; then sleep 300; continue; fi
    [ $try -ge 2 ] && { say "STOPCHECK FAULT $CK: scan $m gave no table ($f: $(grep -E 'Error' $f | tail -n 1 | cut -c1-120))"; exit 4; }
  done
  # the rows: "group | stop s stop cm | creep cm p90 yaw | ... | falls a/b" -> "group;a;b" lines, and the report
  r1=$(awk -F'|' '$NF ~ /^ *[0-9]+\/[0-9]+ *$/ { g = $1; gsub(/^ +| +$/, "", g); split($NF, x, "/"); print g ";" x[1]+0 ";" x[2]+0 }' $f)
  rows="$rows$r1"$'\n'
  parts+=("scan $m: $(awk -F'|' '$NF ~ /^ *[0-9]+\/[0-9]+ *$/ { g = $1; gsub(/^ +| +$/, "", g); split($NF, x, "/"); split($3, c, " ")
      printf "%s%s %d/%d %.2fcm", (n++ ? ", " : ""), g, x[1], x[2], c[1] }' $f)")
done
# each group's falls pooled over the two modes; the verdict on the worst pooled share
out=$(printf '%s' "$rows" | awk -F';' -v max=$MAX 'NF == 3 { a[$1] += $2; b[$1] += $3; if (!($1 in o)) { o[$1] = ++k; nm[k] = $1 } }
  END { for (i = 1; i <= k; i++) { g = nm[i]; s = a[g] / b[g]; if (s > worst || i == 1) { worst = s; wg = g } }
        v = (worst > max ? "FAIL" : (worst > 0 ? "WARN" : "PASS")); printf "%s|%s %d/%d (%.2f%%)\n", v, wg, a[wg], b[wg], 100 * worst }')
verdict=${out%%|*}
say "STOPCHECK $verdict $CK (a group's falls over ${MAX} of both modes pooled fail): pooled worst ${out#*|}; ${parts[0]}; ${parts[1]}"
[ $verdict = FAIL ] && exit 1
exit 0
