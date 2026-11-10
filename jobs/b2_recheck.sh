#!/bin/bash
# b2_recheck.sh (go2-depth 2026-11-02): the B2 RL student checks that lost evaluations to GPU memory on t1 (beside its
# training and dance's: avg1000 s0 11:43 "跑偏 / 原地旋转：评测中", judged FAIL by the chain, which went on). Once the
# B2 training has ended: avg1000 s0 drift_all again and re-judged (its other items passed), then on PASS the full
# seed 1 and 2 checks; avg1500 the same only if the chain's own s0 verdict is incomplete. Logs to fe_b2.txt; the pick
# and the export are by hand.
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
export PATH=$(dirname $PY):$PATH
export BLIND_PILOT=$(cat logs/pkg/b2_fe_mixed_5000/blind_pilot.txt) BLIND_ANCHOR_COEFF=0
export FE_LOCK=/root/gpufree-data/go2-blind/check.lock FE_SERIAL=1
L=logs/jobs/fe_b2.txt C=logs/checks
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
incomplete() { [[ -z "$1" || "$1" == *评测中* || "$1" == *INVALID* || "$1" == *"judge error"* ]]; }
mean_reached() { python3 -c "import re,sys; v=[float(x) for x in re.findall(r'reached=([0-9.]+)', open(sys.argv[1]).read())]; print(f'{100*sum(v)/len(v):.1f}' if v else 'nan')" $1; }
while ps -eo args | grep -qE '^[^ ]*python[^ ]* scripts/train[.]py Unitree-B2-Depth-'; do sleep 60; done
log "B2 recheck: the training has ended"
for K in 1000 1500; do
  A=logs/rsl_rl/b2_depth_student/fe_edge_v4rl_avg$K/model_$K.pt
  P=$C/fe_b2_student_v4rl_avg${K}_s0
  until [ -f ${P}_verdict.txt ]; do sleep 60; done     # the chain's own s0 check first
  v=$(cat ${P}_verdict.txt)
  incomplete "$v" || { log "B2 recheck avg$K: the chain's s0 verdict stands ($v)"; continue; }
  # drift_all (跑偏 / 原地旋转) again under the check lock, then the judge as fe_check.sh does
  exec 9>> $FE_LOCK; flock 9
  for attempt in 1 2 3; do
    $PY jobs/student_eval.py jobs/seed_run.py 0 scripts/blind/eval/drift_all.py --task Unitree-B2-Depth-Student-Flat $A > ${P}_drift3.txt 2>&1 && break
    grep -q "out of memory" ${P}_drift3.txt || break
    log "B2 recheck avg$K s0: drift_all out of GPU memory (attempt $attempt), waiting"; sleep 300
  done
  flock -u 9; exec 9>&-
  v=$($PY jobs/fe_pass.py B2 $P 2>&1 | tail -n 1)
  en=$(mean_reached ${P}_noopedge.txt) eo=$(mean_reached ${P}_opedge.txt)
  python3 -c "import sys; n,o=map(float,sys.argv[1:]); sys.exit(0 if n >= 90 and o >= 95 else 1)" $en $eo || v="$v；棱抹平楼梯：不通过"
  echo "$v" > ${P}_verdict.txt
  log "B2 recheck avg$K s0 (drift_all again): | $v"
  [[ "$v" == PASS ]] || continue
  ok=1
  for SEED in 1 2; do
    Q=$C/fe_b2_student_v4rl_avg${K}_s$SEED
    bash jobs/fe_check.sh student B2 $A $SEED $Q
    incomplete "$(cat ${Q}_verdict.txt 2> /dev/null)" && bash jobs/fe_check.sh student B2 $A $SEED $Q
    v=$(cat ${Q}_verdict.txt)
    log "B2 recheck avg$K s$SEED: noop $(mean_reached ${Q}_noop.txt) op $(mean_reached ${Q}_op.txt) edge noop $(mean_reached ${Q}_noopedge.txt) op $(mean_reached ${Q}_opedge.txt) speed up/down/margin $($PY jobs/fe_margin.py B2 $Q) | $v"
    [[ "$v" == PASS ]] || { ok=0; break; }
  done
  [ $ok = 1 ] && { log "B2 RECHECK PASS avg$K (seeds 0 1 2): pick and export by hand"; exit 0; }
done
log "B2 recheck: done, no average passed here"
