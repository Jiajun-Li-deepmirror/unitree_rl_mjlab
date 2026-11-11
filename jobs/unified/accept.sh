#!/bin/bash
# accept.sh R KIND RUN [K ...] (go2-depth 2026-11-12, doc/redesign.md 2): the acceptance of robot R's KIND (blind |
# depth) student, stage-2 run RUN (logs/rsl_rl/<r>_u_student_<kind>_rl/<run>). Candidates: the checkpoints K given, else
# the last one and then every 500 iterations back to 1000; the first that passes is the pick.
# Tier 2 (mjlab), one run per test at 3x the old per-seed robots (the "3 seeds pooled"), on the check tasks
# Unitree-<R>-U-Student-<Kind>-Flat / -Stairs (jobs/unified/u_eval.py), unloaded (U_PAYLOAD 0 kg) and loaded (the
# walking payload, jobs/accept_payload.sh): files logs/checks/u_<r>_<kind>_<run>_<K>_<test>[_pay].txt, judged by
# scripts/unified/accept_judge.py. Tier 3 (tinynav, the aligned robot models) follows a tier-2 pass: TN_GATE
# (default: the request file logs/accept/u_<r>_<kind>_<run>_<K>.ureq for tn_gate_unified.sh; "skip" until the
# aligned tinynav-sim models are in). Log logs/jobs/unified_<r>_accept.txt; the pick logs/jobs/unified_<r>_<kind>.pick.
set -u
R=$1 KIND=$2 RUN=$3; shift 3
r=${R,,} K=${KIND^}
cd /root/gpufree-data/go2-depth/depth_fe || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1 TRAIN_LOG_LAYOUT=rsl_rl
L=logs/jobs/unified_${r}_accept.txt PICK=logs/jobs/unified_${r}_$KIND.pick
mkdir -p logs/checks logs/accept
log() { echo "$(date '+%m-%d %H:%M:%S') $R $KIND: $*" >> $L; }
W="$PY jobs/unified/u_eval.py" S=scripts/blind/eval
TF=Unitree-$R-U-Student-$K-Flat TE=Unitree-$R-U-Student-$K-Stairs
PAY=$(bash jobs/accept_payload.sh $R) || exit 2
LIN="--lin-accel 2.0" YAW="--yaw-accel 3.0"
CLOCK=""; [[ $R == *W ]] && CLOCK="--clock-task Unitree-$R-U-Teacher"
RUN=${RUN%/}
name=$(basename $RUN)
if [ $# -gt 0 ]; then KS="$*"
else
  last=$(ls $RUN/model_*.pt 2> /dev/null | grep -oE 'model_[0-9]+' | cut -d_ -f2 | sort -n | tail -n 1)
  [ -n "$last" ] || { log "no checkpoints in $RUN"; exit 2; }
  KS="$last $(seq $(( (last / 500) * 500 )) -500 1000 | grep -vx $last | tr '\n' ' ')"
fi
run() {   # OUT CMD...: one test. Out of GPU memory (a training beside it): wait 5 min and retry, up to 6 times; any
  # other failure: retried once. A tool fault is not a result: logged, and the judge reports the test as not measured.
  local out=$1 i; shift
  for i in 1 2 3 4 5 6 7; do
    "$@" > $out 2>&1 && return 0
    grep -qiE "out of memory|CUDA error 2" $out || break
    [ $i -lt 7 ] && sleep 300
  done
  "$@" > $out 2>&1 && return 0
  log "TOOL FAULT: $(basename $out): $(grep -E 'Error' $out | tail -n 1 | cut -c1-160)"
}
for k in $KS; do
  CK=$RUN/model_$k.pt
  [ -f $CK ] || continue
  P=logs/checks/u_${r}_${KIND}_${name}_$k
  t0=$(date +%s)
  log "candidate $CK: tier 2 (mjlab, pooled) started"
  for cond in unloaded loaded; do
    if [ $cond = unloaded ]; then export U_PAYLOAD=0,0,0,0 X=""; else export U_PAYLOAD=$PAY X=_pay; fi
    $PY scripts/unified/dr_report.py $TF > ${P}_dr$X.txt 2>&1
    run ${P}_stairs$X.txt $W scripts/go2_blind/eval_stairs.py --task $TE $CK --envs-per-tile 18 --metrics
    run ${P}_loadstand$X.txt $W $S/long_stand.py --task $TF $CK --seconds 60 --num-envs 768
    run ${P}_stopmove$X.txt $W $S/stop_then_move.py --task $TF $CK --num-envs 384
    run ${P}_drift3$X.txt $W $S/drift_all.py --task $TF $CK --num-envs 768 $YAW $LIN
    run ${P}_spinalt$X.txt $W $S/spin_alt.py --task $TF $CK --num-envs 768 $YAW
    run ${P}_stand$X.txt $W $S/eval_standing.py --task $TF $CK --envs-per-group 768 $LIN $YAW
    run ${P}_march$X.txt $W $S/march_eval.py --task $TF $CK --envs 96
    if [ $cond = unloaded ]; then
      run ${P}_am.txt $W $S/accept_matrix.py --task $TF --envs-per-cmd 96 $CK $LIN $YAW
      run ${P}_clock.txt $W $S/gait_clock.py --task $TF $CLOCK $CK --envs-per-cmd 48
      [[ $R == *W ]] && run ${P}_gait.txt $W $S/gait_fb.py --task $TF $CK --envs-per-cmd 96
      run ${P}_noop.log $W $S/map3_par.py --out ${P}_noop.txt --shards 3 -- --task $TE $CK --set flights --no-operator \
        --down-vx 0.5 --num-envs 192 --time-scale 1.6 --seed 0
      run ${P}_op.log $W $S/map3_par.py --out ${P}_op.txt --shards 3 -- --task $TE $CK --set flights --num-envs 192 \
        --time-scale 1.6 --seed 0
    fi
  done
  unset U_PAYLOAD X
  $PY scripts/unified/accept_judge.py $R $P > ${P}_verdict.txt 2>&1
  v=$(head -n 1 ${P}_verdict.txt)
  [[ "$v" == PASS || "$v" == FAIL* ]] || v="FAIL: judge error: $v"
  log "$CK tier 2 ($(( ($(date +%s) - t0) / 60 )) min): $v"
  [[ "$v" == PASS ]] || continue
  # tier 3: tinynav on the aligned models
  if [ "${TN_GATE:-request}" = skip ]; then
    log "$CK tier 3 skipped (TN_GATE=skip: the aligned tinynav-sim models are not in yet) -> candidate pick"
    echo "$CK tier2-only" > $PICK
    exit 0
  fi
  Q=logs/accept/u_${r}_${KIND}_${name}_$k
  rm -f $Q.verdict
  echo "$R $CK U${KIND^^}" > $Q.ureq   # .ureq: tools/go2_depth/acceptance/tn_gate_unified.sh (the older .req watcher ignores it)
  w=0
  until [ -f $Q.verdict ] || [ $w -ge 43200 ]; do sleep 60; w=$((w + 60)); done
  tv=$(cat $Q.verdict 2> /dev/null || echo "no verdict in 12 h (tool fault)")
  log "$CK tier 3 (tinynav): $tv"
  if [[ "$tv" == PASS* ]]; then
    echo "$CK" > $PICK
    log "PICK $CK"
    # the delivery: the tinynav gate's export (logs/probe/<name>, export_student.py) and the checkpoint, then the two
    # acceptance videos (flat and map3, unloaded above / loaded below; jobs/accept_videos.sh)
    PKG=logs/pkg/${r}_u_${KIND}_${name}_$k
    mkdir -p $PKG
    cp logs/probe/$(basename $Q)/* $PKG/ 2> /dev/null
    cp $CK $PKG/
    if [ -s $PKG/policy.onnx ]; then
      log "UNIFIED DELIVERY $PKG (policy.onnx $(md5sum < $PKG/policy.onnx | cut -c1-8))"
    else
      $PY scripts/unified/export_student.py $R $KIND $CK $PKG >> $L 2>&1 && log "UNIFIED DELIVERY $PKG (exported here)"
    fi
    LOG=$L bash jobs/accept_videos.sh u$KIND $R $CK $PKG/videos
    exit 0
  fi
done
log "no candidate of $RUN passed"
exit 1
