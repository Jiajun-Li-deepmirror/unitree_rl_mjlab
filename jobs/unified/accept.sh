#!/bin/bash
# accept.sh R KIND RUN [K ...] (go2-depth 2026-11-12, doc/redesign.md 2): the acceptance of robot R's KIND (blind |
# depth) student, stage-2 run RUN (logs/rsl_rl/<r>_u_student_<kind>_rl/<run>).
# Candidates (CANDIDATES):
#   avg    (default since 2026-11-12, the user: "尽量减少验收的耗时以及尽量减少运气通过的问题"; doc/accept_stats_proposal.md)
#          AVG_CANDS (default 1) candidates fixed in advance, each the average of the AVG_N (default 8) newest
#          checkpoints at or before an end iteration (scripts/unified/avg_ckpt.py -> RUN/avg<N>_<end>.pt): the ends
#          are the K given, else the last checkpoint and then AVG_SPACING (default 1000) iterations apart. One
#          candidate tested once: a pass is not the luckiest of several tries, and the average evens out the
#          save-to-save swings of single checkpoints (doc/blind_phased_training.md 8.1). A fail is the run's result
#          (jobs/unified/students.sh then falls back to another teacher seed), not a reason to try older checkpoints.
#   first  (the rule before) the checkpoints K given, else the last one and then every 500 iterations back to 1000;
#          the first, in that order, that passes both tiers is the pick
# Tier 2 (mjlab), one run per test at 3x the old per-seed robots (the "3 seeds pooled"), on the check tasks
# Unitree-<R>-U-Student-<Kind>-Flat / -Stairs (jobs/unified/u_eval.py), unloaded (U_PAYLOAD 0 kg) and loaded (the
# walking payload, jobs/accept_payload.sh): files logs/checks/u_<r>_<kind>_<run>_<tag>_<test>[_pay].txt, judged by
# scripts/unified/accept_judge.py. The tests run ACCEPT_JOBS (default 2) at a time on the GPU (both payload
# conditions in one pool; the map3 flights, three shards each already, after them), out-of-memory retried as before.
# Tier 3 (tinynav, the aligned robot models) follows a tier-2 pass: TN_GATE (default:
# the request file logs/accept/u_<r>_<kind>_<run>_<tag>.ureq for tn_gate_unified.sh; "skip": tier 2 only). Tier 3
# runs beside the next candidates' tier 2 (2026-11-12, the review's item 11: the gate's 12 h no longer idle the GPU);
# the pick is still the first candidate in order that passes both, so a later candidate waits for the earlier ones'
# verdicts, and testing stops once an earlier candidate is known to pass.
# BASELINE=1: tier 2 only, no pick, no delivery -- for a reading of another policy under the same tests (the review's
# item 12, jobs/unified/baseline.sh). ACCEPT_SMOKE=1: two small tests per candidate, tier 2 taken as passed (not judged),
# the pick in logs/jobs/unified_<r>_<kind>_smoke.pick, no delivery (jobs/unified/smoke_students.sh).
# Log logs/jobs/unified_<r>_accept.txt; the pick logs/jobs/unified_<r>_<kind>.pick. Exit 0 picked (or BASELINE / a
# smoke done), 1 no candidate passed, 2 setup error.
set -u
R=$1 KIND=$2 RUN=$3; shift 3
r=${R,,} K=${KIND^}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1 TRAIN_LOG_LAYOUT=rsl_rl
SMOKE=${ACCEPT_SMOKE:-0} BASE=${BASELINE:-0} MODE=${CANDIDATES:-avg} JOBS=${ACCEPT_JOBS:-2}
L=logs/jobs/unified_${r}_accept.txt PICK=logs/jobs/unified_${r}_$KIND.pick
[ $SMOKE = 1 ] && PICK=logs/jobs/unified_${r}_${KIND}_smoke.pick
mkdir -p logs/checks logs/accept logs/jobs
TAG=""; [ $SMOKE = 1 ] && TAG="SMOKE "; [ $BASE = 1 ] && TAG="BASELINE "
log() { echo "$(date '+%m-%d %H:%M:%S') $R $KIND: $TAG$*" >> $L; }
W="$PY jobs/unified/u_eval.py" S=scripts/blind/eval
TF=Unitree-$R-U-Student-$K-Flat TE=Unitree-$R-U-Student-$K-Stairs
PAY=$(bash jobs/accept_payload.sh $R) || exit 2
LIN="--lin-accel 2.0" YAW="--yaw-accel 3.0"
CLOCK=""; [[ $R == *W ]] && CLOCK="--clock-task Unitree-$R-U-Teacher"
RUN=${RUN%/}
name=$(basename $RUN)
iters() { ls $RUN/model_*.pt 2> /dev/null | grep -oE 'model_[0-9]+\.pt$' | grep -oE '[0-9]+' | sort -n; }
last=$(iters | tail -n 1)
[ -n "$last" ] || { log "no checkpoints in $RUN"; exit 2; }

# ---- the candidates: tags (first: the iteration K, file model_K.pt; avg: a<N>_<end>, file avg<N>_<end>.pt) ----
CANDS=()
declare -A CKOF
if [ $MODE = avg ]; then
  N=${AVG_N:-8} SP=${AVG_SPACING:-1000}
  if [ $# -gt 0 ]; then ENDS="$*"; else ENDS=$(for c in $(seq 0 $(( ${AVG_CANDS:-1} - 1 ))); do echo $(( last - c * SP )); done); fi
  for e in $ENDS; do
    ks=$(iters | awk -v e=$e '$1 <= e' | tail -n $N)
    if [ $(echo $ks | wc -w) -lt $N ]; then log "avg candidate at $e: only $(echo $ks | wc -w) checkpoints at or before it (need $N), skipped"; continue; fi
    e=$(echo $ks | awk '{print $NF}')
    out=$RUN/avg${N}_$e.pt
    $PY scripts/unified/avg_ckpt.py $out $(for k in $ks; do echo $RUN/model_$k.pt; done) >> $L 2>&1 \
      || { log "avg candidate at $e: averaging failed (tool fault)"; continue; }
    CANDS+=(a${N}_$e); CKOF[a${N}_$e]=$out
  done
  log "candidates (avg: $N checkpoints each, fixed in advance): ${CANDS[*]:-none}"
else
  if [ $# -gt 0 ]; then KS="$*"; else KS="$last $(seq $(( (last / 500) * 500 )) -500 1000 | grep -vx $last | tr '\n' ' ')"; fi
  for k in $KS; do [ -f $RUN/model_$k.pt ] && { CANDS+=($k); CKOF[$k]=$RUN/model_$k.pt; }; done
fi
[ ${#CANDS[@]} -gt 0 ] || { log "no candidate checkpoints in $RUN"; exit 1; }

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

tier2() {   # TAG -> prints PASS / FAIL: ...
  local tag=$1 CK=${CKOF[$1]} P=logs/checks/u_${r}_${KIND}_${name}_$1 cond X
  if [ $SMOKE = 1 ]; then
    export U_PAYLOAD=0,0,0,0
    run ${P}_stand.txt $W $S/eval_standing.py --task $TF $CK --envs-per-group 8 $LIN $YAW
    run ${P}_stairs.txt $W scripts/go2_blind/eval_stairs.py --task $TE $CK --envs-per-tile 1 --metrics
    unset U_PAYLOAD
    echo "PASS"
    return
  fi
  # JOBS tests at a time (both payload conditions in one pool), each in its own process with its own U_PAYLOAD
  pool() {   # PAYLOAD OUT CMD...
    while [ $(jobs -rp | wc -l) -ge $JOBS ]; do wait -n; done
    ( export U_PAYLOAD=$1; shift; run "$@" ) &
  }
  # the conditions: unloaded, the walking payload, and the typical payload where the robot has one (2026-11-12,
  # doc/redesign.md 2.2: kg,x,y,z,dx,dy,dz -- each robot's mount point drawn in the box; jobs/accept_payload.sh exits
  # 2 for a robot without one)
  local conds="unloaded loaded" typ
  typ=$(bash jobs/accept_payload.sh $R typical 2> /dev/null) && conds="$conds typical"
  for cond in $conds; do
    case $cond in unloaded) pay=0,0,0,0 X="" ;; loaded) pay=$PAY X=_pay ;; typical) pay=$typ X=_typ ;; esac
    U_PAYLOAD=$pay $PY scripts/unified/dr_report.py $TF > ${P}_dr$X.txt 2>&1
    pool $pay ${P}_stairs$X.txt $W scripts/go2_blind/eval_stairs.py --task $TE $CK --envs-per-tile 18 --metrics
    pool $pay ${P}_loadstand$X.txt $W $S/long_stand.py --task $TF $CK --seconds 60 --num-envs 768
    pool $pay ${P}_stopmove$X.txt $W $S/stop_then_move.py --task $TF $CK --num-envs 384
    pool $pay ${P}_drift3$X.txt $W $S/drift_all.py --task $TF $CK --num-envs 768 $YAW $LIN
    pool $pay ${P}_spinalt$X.txt $W $S/spin_alt.py --task $TF $CK --num-envs 768 $YAW
    pool $pay ${P}_stand$X.txt $W $S/eval_standing.py --task $TF $CK --envs-per-group 768 $LIN $YAW
    pool $pay ${P}_march$X.txt $W $S/march_eval.py --task $TF $CK --envs 96
    if [ $cond = unloaded ]; then
      pool $pay ${P}_am.txt $W $S/accept_matrix.py --task $TF --envs-per-cmd 96 $CK $LIN $YAW
      pool $pay ${P}_clock.txt $W $S/gait_clock.py --task $TF $CLOCK $CK --envs-per-cmd 48
    fi
    # wheels down: unloaded and typical (doc/redesign.md 2.2)
    if [[ $R == *W ]] && [ $cond != loaded ]; then
      pool $pay ${P}_gait$X.txt $W $S/gait_fb.py --task $TF $CK --envs-per-cmd 96
    fi
  done
  wait
  # the map3 flights (unloaded): three shards each already, so one at a time after the pool
  ( export U_PAYLOAD=0,0,0,0
    run ${P}_noop.log $W $S/map3_par.py --out ${P}_noop.txt --shards 3 -- --task $TE $CK --set flights --no-operator \
      --down-vx 0.5 --num-envs 192 --time-scale 1.6 --seed 0
    run ${P}_op.log $W $S/map3_par.py --out ${P}_op.txt --shards 3 -- --task $TE $CK --set flights --num-envs 192 \
      --time-scale 1.6 --seed 0 )
  $PY scripts/unified/accept_judge.py $R $P > ${P}_verdict.txt 2>&1
  local v=$(head -n 1 ${P}_verdict.txt)
  [[ "$v" == PASS || "$v" == FAIL* ]] || v="FAIL: judge error: $v"
  echo "$v"
}

deliver() {   # TAG: the pick's package and videos (jobs/unified/deliver.sh) -- the blind student at once; the depth
  # student only after the camera-failure test of the pair (doc/redesign.md 2.2.1, pair_step)
  local tag=$1 CK=${CKOF[$1]} Q=logs/accept/u_${r}_${KIND}_${name}_$1
  local PKG=logs/pkg/${r}_u_${KIND}_${name}_$tag
  if [ $KIND = depth ]; then
    echo "$R depth $CK $Q $PKG" > $DPICK.pending
    log "depth delivery waits for the camera-failure pair test ($DPICK.pending)"
  elif [ $SMOKE = 1 ]; then
    log "(smoke: no delivery; would package $PKG)"
  else
    LOG=$L bash jobs/unified/deliver.sh $R $KIND $CK $Q $PKG
  fi
  pair_step
}

# The camera-failure test of the pair (2026-11-12; doc/redesign.md 2.2.1): the depth and the blind student of this
# robot, each accepted on its own. The two kinds' acceptances run side by side, so whichever is accepted second runs
# it (under a lock; the depth student's line moves from .pending to .pairing): PASS -> the depth student is delivered;
# FAIL -> it is not (its pick moves to <pick>.pairfail; the blind student is not affected); a tool fault leaves it
# pending. A depth student without an accepted blind one waits (pending) for a human or a later blind acceptance.
# Exit status of this script: 1 when this run's own depth student failed the pair test (it ran it itself).
DPICK=logs/jobs/unified_${r}_depth$([ $SMOKE = 1 ] && echo _smoke).pick
BPICK=logs/jobs/unified_${r}_blind$([ $SMOKE = 1 ] && echo _smoke).pick
PAIR_RC=0
pair_step() {
  local rc
  exec 9> logs/jobs/unified_${r}_pair$([ $SMOKE = 1 ] && echo _smoke).lock
  flock 9
  if [ -f $DPICK.pending ] && [ -s $BPICK ] && [ -s $DPICK ]; then
    mv -f $DPICK.pending $DPICK.pairing
    read -r _ _ DCK DQ DPKG < $DPICK.pairing
    local BCK=$(cut -d' ' -f1 $BPICK)
    PAIR_SMOKE=$SMOKE bash jobs/unified/pair_accept.sh $R $DCK $BCK; rc=$?
    # smoke: untrained students fall, the verdict is not judged -- only that the test ran and gave one
    [ $SMOKE = 1 ] && [ $rc = 1 ] && { log "(smoke: the pair test's FAIL not judged)"; rc=0; }
    case $rc in
      0) log "camera-failure pair test PASS (depth $DCK, blind $BCK)"
         if [ $SMOKE = 1 ]; then log "(smoke: no delivery; would package $DPKG)"
         else LOG=$L bash jobs/unified/deliver.sh $R depth $DCK $DQ $DPKG; fi
         rm -f $DPICK.pairing ;;
      1) mv -f $DPICK $DPICK.pairfail; mv -f $DPICK.pairing $DPICK.pairing.failed
         log "DEPTH NOT DELIVERED: the camera-failure pair test failed ($(cat logs/jobs/unified_${r}_pair.verdict 2> /dev/null | cut -c1-200)); the blind student is not affected"
         [ $KIND = depth ] && PAIR_RC=1 ;;
      *) mv -f $DPICK.pairing $DPICK.pending
         log "camera-failure pair test: tool fault (exit $rc); the depth delivery stays pending" ;;
    esac
  elif [ -f $DPICK.pending ]; then
    log "camera-failure pair test waits: no accepted blind student yet ($BPICK)"
  fi
  flock -u 9
}

# ---- tier 2 per candidate in order; tier 3 alongside ----
declare -A T2 T3 ASKED
ORDER=()
TN=${TN_GATE:-request}
[ $SMOKE = 1 ] && TN=${TN_GATE:-skip}
WAIT_S=${TN_WAIT_S:-43200}
poll() {   # the tier-3 verdicts that have come back (or timed out)
  local tag f
  for tag in "${ORDER[@]}"; do
    [ "${T2[$tag]}" = PASS ] && [ -z "${T3[$tag]:-}" ] || continue
    f=logs/accept/u_${r}_${KIND}_${name}_$tag.verdict
    if [ -f $f ]; then T3[$tag]=$(cat $f); log "${CKOF[$tag]} tier 3 (tinynav): ${T3[$tag]}"
    elif [ $(( $(date +%s) - ${ASKED[$tag]} )) -ge $WAIT_S ]; then
      T3[$tag]="no verdict in $(( WAIT_S / 3600 )) h (tool fault)"; log "${CKOF[$tag]} tier 3 (tinynav): ${T3[$tag]}"
    fi
  done
}
PICKED=""
decide() {   # 0: PICKED is the pick; 1: an earlier candidate's tier 3 is still out; 2: none of the tested ones passes
  local tag
  for tag in "${ORDER[@]}"; do
    [ "${T2[$tag]}" = PASS ] || continue
    [ -n "${T3[$tag]:-}" ] || return 1
    [[ "${T3[$tag]}" == PASS* ]] && { PICKED=$tag; return 0; }
  done
  return 2
}
for tag in "${CANDS[@]}"; do
  if [ $BASE != 1 ]; then poll; decide && break; fi
  t0=$(date +%s)
  log "candidate ${CKOF[$tag]}: tier 2 (mjlab, pooled) started"
  v=$(tier2 $tag)
  log "${CKOF[$tag]} tier 2 ($(( ($(date +%s) - t0) / 60 )) min): $v$([ $SMOKE = 1 ] && echo ' (smoke: not judged)')"
  ORDER+=($tag)
  [[ "$v" == PASS ]] && T2[$tag]=PASS || T2[$tag]=FAIL
  [ $BASE = 1 ] && continue
  [ ${T2[$tag]} = PASS ] || continue
  if [ "$TN" = skip ]; then
    T3[$tag]="PASS (tier 3 skipped: TN_GATE=skip)"
    log "${CKOF[$tag]} tier 3 skipped (TN_GATE=skip)"
  else
    Q=logs/accept/u_${r}_${KIND}_${name}_$tag
    rm -f $Q.verdict
    echo "$R ${CKOF[$tag]} U${KIND^^}" > $Q.ureq   # .ureq: tools/go2_depth/acceptance/tn_gate_unified.sh
    ASKED[$tag]=$(date +%s)
    log "${CKOF[$tag]} tier 3 requested ($Q.ureq); the next candidates' tier 2 runs meanwhile"
  fi
done
if [ $BASE = 1 ]; then log "done: ${#ORDER[@]} candidates read, no pick (baseline)"; exit 0; fi
while true; do
  poll
  decide; rc=$?
  [ $rc = 1 ] || break
  sleep ${POLL_S:-60}
done
if [ $rc = 0 ]; then
  CK=${CKOF[$PICKED]}
  if [ "$TN" = skip ]; then echo "$CK tier2-only" > $PICK; else echo "$CK" > $PICK; fi
  log "PICK $CK"
  rm -f $PICK.pairfail   # (a pair failure of an earlier pick is not this one's)
  deliver $PICKED
  exit $PAIR_RC
fi
log "no candidate of $RUN passed"
exit 1
