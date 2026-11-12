#!/bin/bash
# smoke_students.sh R TEACHER_CKPT (unitree-rl-mjlab-62 2026-11-12, the review's item 13): the students' chain end to
# end in minutes, once, right before a students' training starts (the user's rule 11-12: no smoke test after each code
# change; one fastest smoke before a formal training). What it runs, at small sizes, for both kinds side by side:
#   jobs/unified/students.sh SMOKE=1   stage 1 (DAgger) and stage 2 (PPO + BC) S_ITERS (default 20) iterations each at
#                                      NUM_ENVS / 2 robots (default 512 in all), a checkpoint every 10 iterations;
#                                      runs *_smoke (not listed by tools/go2_depth/monitor/tb.sh)
#   jobs/unified/accept.sh smoke mode  CANDIDATES=avg AVG_N=2: the averaging (scripts/unified/avg_ckpt.py), the check
#                                      tasks loading the average, the candidate / tier-3 loop (TN_GATE default skip;
#                                      TN_GATE=request sends one real tinynav request: tn_gate_unified.sh must be up)
#   scripts/unified/export_student.py  the pick as a package (policy.onnx, deploy.yaml, manifest)
# and checks: each stage's log has Loss/priv_mse (the privileged-latent probe), each kind has a smoke pick and a
# package with policy.onnx. Prints SMOKE PASS / SMOKE FAIL: ...; log logs/jobs/unified_<r>_students.txt.
set -u
R=$1 T=$2 r=${1,,}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
[ -f "$T" ] || { echo "smoke_students.sh: no teacher checkpoint $T" >&2; exit 2; }
N=${S_ITERS:-20}
rm -f logs/jobs/unified_${r}_blind_smoke.pick logs/jobs/unified_${r}_depth_smoke.pick
STAMP=$(mktemp) && touch $STAMP   # (the pair test's verdict must be newer)
trap 'rm -f $STAMP' EXIT
SL=logs/jobs/unified_${r}_students.txt
L0=$(cat $SL 2> /dev/null | wc -l)   # (the students log's lines from this smoke on)
SMOKE=1 TEACHER=$T S1_ITERS=$N S2_ITERS=$N NUM_ENVS=${NUM_ENVS:-512} SAVE_EVERY=10 \
  CANDIDATES=avg AVG_N=2 AVG_CANDS=1 TN_GATE=${TN_GATE:-skip} bash jobs/unified/students.sh $R
fails=()
tail -n +$((L0 + 1)) $SL 2> /dev/null | grep -qE "SMOKE teacher stop check: STOPCHECK (PASS|WARN|FAIL)" \
  || fails+=("the teacher stop check gave no reading ($SL)")
for kind in blind depth; do
  for st in 1 2; do
    f=logs/jobs/unified_${r}_${kind}${st}_smoke.log
    grep -q "priv_mse" $f 2> /dev/null || fails+=("$kind stage $st: no priv_mse in $f")
  done
  pick=logs/jobs/unified_${r}_${kind}_smoke.pick
  [ -s $pick ] || { fails+=("$kind: no smoke pick (logs/jobs/unified_${r}_accept.txt)"); continue; }
  ck=$(cut -d' ' -f1 $pick)
  [[ $(basename $ck) == avg2_* ]] || fails+=("$kind: the pick $ck is not an averaged candidate")
  pkg=logs/probe/smoke_${r}_$kind
  rm -rf $pkg
  UNIFIED_TASKS=1 $PY scripts/unified/export_student.py $R $kind $ck $pkg > $pkg.export.log 2>&1
  [ -s $pkg/policy.onnx ] || fails+=("$kind: export failed ($pkg.export.log)")
done
# the camera-failure pair test (doc/redesign.md 2.2.1) ran on the two smoke picks and gave a verdict (not judged; its
# own file, not the formal one), and both kinds were delivered (smoke: marked, not packaged)
V=logs/jobs/unified_${r}_pair_smoke.verdict
for kind in blind depth; do
  [ -f logs/jobs/unified_${r}_${kind}_smoke.pick.delivered ] || fails+=("$kind: no delivery marker")
done
if [ ! -f $V ] || [ $V -ot $STAMP ]; then fails+=("pair test: no verdict from this smoke ($V)")
elif ! grep -qE "^(PASS|FAIL)" $V; then fails+=("pair test: $(head -c 160 $V)"); fi
if [ ${#fails[@]} = 0 ]; then echo "SMOKE PASS ($R, teacher $T)"; exit 0; fi
printf 'SMOKE FAIL: %s\n' "${fails[@]}"
exit 1
