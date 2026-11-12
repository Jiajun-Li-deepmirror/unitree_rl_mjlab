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
# Unitree-<R>-U-Student-<Kind>-Flat / -Stairs (jobs/unified/u_eval.py), unloaded (U_PAYLOAD 0 kg), loaded (the walking
# payload, jobs/accept_payload.sh) and, for a robot with one (src/robot_table.py's typical_payload), the typical
# payload (doc/redesign.md 2.2): files logs/checks/u_<r>_<kind>_<run>_<tag>_<test>[_pay|_typ].txt, judged by
# scripts/unified/accept_judge.py (PASS / FAIL / FAULT: what it cannot read is a tool fault). The tests run ACCEPT_JOBS
# (default 2) at a time on the GPU (all conditions in one pool; the map3 flights, three shards each already, after
# them), out-of-memory retried. Tier 3 (tinynav, the aligned robot models) follows a tier-2 pass: TN_GATE (default:
# the request file logs/accept/u_<r>_<kind>_<run>_<tag>.ureq for tn_gate_unified.sh; "skip": tier 2 only). Tier 3
# runs beside the next candidates' tier 2; the pick is still the first candidate in order that passes both.
# Delivery (jobs/unified/deliver.sh; <pick>.delivered when done): the blind student at once; the depth student after
# the camera-failure test of the pair (pair_step below), its line kept in <pick>.pending meanwhile.
# BASELINE=1: tier 2 only, no pick, no delivery (jobs/unified/baseline.sh). ACCEPT_SMOKE=1: two small tests per
# candidate, tier 2 taken as passed (not judged), the picks and markers *_smoke.pick*, no packaging
# (jobs/unified/smoke_students.sh). PAIR_ONLY=1 accept.sh R depth -: only the pair test and the depth delivery again
# (after a pair test's tool fault: <pick>.pairfault).
# Log logs/jobs/unified_<r>_accept.txt; the pick logs/jobs/unified_<r>_<kind>.pick. Exit 0 picked (delivered, or the
# depth student pending the pair test; BASELINE / a smoke done), 1 no candidate passed, 2 setup error, 4 a tool fault
# (a candidate the judge could not read, a tier-3 tool fault, this kind's delivery failed, the typical payload's lookup
# failed) -- a fault is not a fail. The pair test's outcome is in markers on the depth pick (.pairfail, .pairfault),
# which jobs/unified/students.sh reads, whichever kind's run ran it.
set -u
R=$1 KIND=$2 RUN=$3; shift 3
r=${R,,} K=${KIND^}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1 TRAIN_LOG_LAYOUT=rsl_rl
source jobs/unified/formal.env   # the formal run's parameters, frozen with the commit
SMOKE=${ACCEPT_SMOKE:-0} BASE=${BASELINE:-0} MODE=${CANDIDATES:-$F_ACCEPT_CANDIDATES} JOBS=${ACCEPT_JOBS:-$F_ACCEPT_JOBS}
SFX=$([ $SMOKE = 1 ] && echo _smoke)
L=logs/jobs/unified_${r}_accept.txt PICK=logs/jobs/unified_${r}_$KIND$SFX.pick
mkdir -p logs/checks logs/accept logs/jobs
TAG=""; [ $SMOKE = 1 ] && TAG="SMOKE "; [ $BASE = 1 ] && TAG="BASELINE "
log() { echo "$(date '+%m-%d %H:%M:%S') $R $KIND: $TAG$*" >> $L; }
PAY=$(bash jobs/accept_payload.sh $R) || exit 2
OWN_RC=0

# ---- delivery and the pair test (doc/redesign.md 2.2.1) ----
DPICK=logs/jobs/unified_${r}_depth$SFX.pick
BPICK=logs/jobs/unified_${r}_blind$SFX.pick
deliver_now() {   # KIND CKPT Q PKG PICKFILE -> 0 delivered (<pick>.delivered written)
  local kind=$1 ck=$2 q=$3 pkg=$4 pf=$5
  if [ $SMOKE = 1 ]; then
    log "(smoke: no packaging; would package $pkg)"
    echo "smoke $ck" > $pf.delivered
    return 0
  fi
  if LOG=$L bash jobs/unified/deliver.sh $R $kind $ck $q $pkg; then
    echo "$ck $pkg" > $pf.delivered
    return 0
  fi
  log "TOOL FAULT: the $kind delivery of $ck failed (deliver.sh); not delivered"
  return 4
}
# The camera-failure test of the pair: the depth and the blind student of this robot, each accepted on its own. The
# two kinds' acceptances run side by side, so whichever is accepted second runs it (under a lock; the depth student's
# line moves from .pending to .pairing while it runs): PASS -> the depth student is delivered; FAIL -> it is not (its
# pick moves to <pick>.pairfail; the blind student is not affected); a tool fault -> <pick>.pairfault, the line back
# to .pending (PAIR_ONLY=1 reruns it). A depth student without an accepted blind one stays pending.
pair_step() {
  local rc
  exec 9> logs/jobs/unified_${r}_pair$SFX.lock
  flock 9
  if [ -f $DPICK.pairing ]; then   # a pair test that never finished (the run was killed): it did not happen
    mv -f $DPICK.pairing $DPICK.pending
    log "a stale $DPICK.pairing (an interrupted pair test): back to pending"
  fi
  if [ -f $DPICK.pending ] && [ -s $BPICK ] && [ -s $DPICK ]; then
    rm -f $DPICK.pairfault
    mv -f $DPICK.pending $DPICK.pairing
    read -r _ _ DCK DQ DPKG < $DPICK.pairing
    local BCK=$(cut -d' ' -f1 $BPICK)
    PAIR_SMOKE=$SMOKE bash jobs/unified/pair_accept.sh $R $DCK $BCK; rc=$?
    # smoke: untrained students fall, the verdict is not judged -- only that the test ran and gave one
    [ $SMOKE = 1 ] && [ $rc = 1 ] && { log "(smoke: the pair test's FAIL not judged)"; rc=0; }
    case $rc in
      0) log "camera-failure pair test PASS (depth $DCK, blind $BCK)"
         if deliver_now depth $DCK $DQ $DPKG $DPICK; then rm -f $DPICK.pairing
         else mv -f $DPICK.pairing $DPICK.pending; fi ;;
      1) mv -f $DPICK $DPICK.pairfail; mv -f $DPICK.pairing $DPICK.pairing.failed
         log "DEPTH NOT DELIVERED: the camera-failure pair test failed ($(head -c 200 logs/jobs/unified_${r}_pair$SFX.verdict 2> /dev/null)); the blind student is not affected" ;;
      *) mv -f $DPICK.pairing $DPICK.pending
         echo "pair_accept.sh exit $rc" > $DPICK.pairfault
         log "TOOL FAULT: camera-failure pair test (exit $rc): the depth delivery stays pending ($DPICK.pairfault; PAIR_ONLY=1 reruns it)" ;;
    esac
  elif [ -f $DPICK.pending ]; then
    log "camera-failure pair test waits: no accepted blind student yet ($BPICK)"
  fi
  flock -u 9
}
if [ "${PAIR_ONLY:-0}" = 1 ]; then
  [ $KIND = depth ] || { echo "accept.sh: PAIR_ONLY is for the depth student" >&2; exit 2; }
  [ -f $DPICK.pending ] || [ -f $DPICK.pairing ] || { log "PAIR_ONLY: nothing pending ($DPICK.pending)"; exit 2; }
  pair_step
  if [ -f $DPICK.pairfault ]; then exit 4; elif [ -f $DPICK.pairfail ]; then exit 1; elif [ -f $DPICK.delivered ]; then exit 0; fi
  exit 4
fi

# ---- the candidates ----
W="$PY jobs/unified/u_eval.py" S=scripts/blind/eval
TF=Unitree-$R-U-Student-$K-Flat TE=Unitree-$R-U-Student-$K-Stairs
LIN="--lin-accel 2.0" YAW="--yaw-accel 3.0"
CLOCK=""; [[ $R == *W ]] && CLOCK="--clock-task Unitree-$R-U-Teacher"
RUN=${RUN%/}
name=$(basename $RUN)
iters() { ls $RUN/model_*.pt 2> /dev/null | grep -oE 'model_[0-9]+\.pt$' | grep -oE '[0-9]+' | sort -n; }
last=$(iters | tail -n 1)
[ -n "$last" ] || { log "no checkpoints in $RUN"; exit 2; }
# tags (first: the iteration K, file model_K.pt; avg: a<N>_<end>, file avg<N>_<end>.pt)
CANDS=()
declare -A CKOF
if [ $MODE = avg ]; then
  N=${AVG_N:-$F_ACCEPT_AVG_N} SP=${AVG_SPACING:-1000}
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

# Does the robot have a typical payload? The table's own answer (a robot without the key: no; a lookup that fails --
# an unknown robot, an old table -- is a tool fault, not "none": go2-depth's review 11-12)
TYPICAL=$(python3 - $R <<'PYEOF' 2> /dev/null
import sys
sys.path.insert(0, "src")
import robot_table as t
try:
  print("yes" if "typical_payload" in t.get(sys.argv[1]) else "no")
except Exception:
  print("error")
PYEOF
)
TYPICAL=${TYPICAL:-error}

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

tier2() {   # TAG -> prints PASS / FAIL: ... / FAULT: ...
  local tag=$1 CK=${CKOF[$1]} P=logs/checks/u_${r}_${KIND}_${name}_$1 cond X
  if [ $SMOKE = 1 ]; then
    export U_PAYLOAD=0,0,0,0
    run ${P}_stand.txt $W $S/eval_standing.py --task $TF $CK --envs-per-group 8 $LIN $YAW
    run ${P}_stairs.txt $W scripts/go2_blind/eval_stairs.py --task $TE $CK --envs-per-tile 1 --metrics
    unset U_PAYLOAD
    echo "PASS"
    return
  fi
  local conds="unloaded loaded" typ="" jflag=""
  case $TYPICAL in
    yes) typ=$(bash jobs/accept_payload.sh $R typical) || { echo "FAULT: the typical payload's lookup failed (jobs/accept_payload.sh $R typical)"; return; }
         conds="$conds typical" jflag="--typical" ;;
    no) ;;
    *) echo "FAULT: whether $R has a typical payload could not be read (src/robot_table.py)"; return ;;
  esac
  # JOBS tests at a time (all conditions in one pool), each in its own process with its own U_PAYLOAD
  pool() {   # PAYLOAD OUT CMD...
    while [ $(jobs -rp | wc -l) -ge $JOBS ]; do wait -n; done
    ( export U_PAYLOAD=$1; shift; run "$@" ) &
  }
  for cond in $conds; do
    case $cond in unloaded) pay=0,0,0,0 X="" ;; loaded) pay=$PAY X=_pay ;; typical) pay=$typ X=_typ ;; esac
    U_PAYLOAD=$pay $PY scripts/unified/dr_report.py $TF > ${P}_dr$X.txt 2>&1
    pool $pay ${P}_stairs$X.txt $W scripts/go2_blind/eval_stairs.py --task $TE $CK --envs-per-tile 18 --metrics
    pool $pay ${P}_loadstand$X.txt $W $S/long_stand.py --task $TF $CK --seconds 60 --num-envs 768
    pool $pay ${P}_stopmove$X.txt $W $S/stop_then_move.py --task $TF $CK --num-envs 384
    # the depth student: 384 robots per command (P3 11-12: 13 commands x 768 rendering robots asked torch for 9.71 GiB
    # at once and did not fit a 24.5 GB card beside another test; 384 still above the user's 256 per test)
    pool $pay ${P}_drift3$X.txt $W $S/drift_all.py --task $TF $CK --num-envs $([ $KIND = depth ] && echo 384 || echo 768) $YAW $LIN
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
  $PY scripts/unified/accept_judge.py $R $P $jflag > ${P}_verdict.txt 2>&1
  local v=$(head -n 1 ${P}_verdict.txt)
  [[ "$v" == PASS || "$v" == FAIL* || "$v" == FAULT* ]] || v="FAULT: the judge gave no verdict: $(echo $v | cut -c1-120)"
  echo "$v"
}

deliver() {   # TAG: the pick's package and videos -- the blind student at once; the depth student after the pair test
  local tag=$1 CK=${CKOF[$1]} Q=logs/accept/u_${r}_${KIND}_${name}_$1
  local PKG=logs/pkg/${r}_u_${KIND}_${name}_$tag
  if [ $KIND = depth ]; then
    echo "$R depth $CK $Q $PKG" > $DPICK.pending
    log "depth delivery waits for the camera-failure pair test ($DPICK.pending)"
  else
    deliver_now $KIND $CK $Q $PKG $PICK || OWN_RC=4
  fi
  pair_step
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
  case "$v" in PASS) T2[$tag]=PASS ;; FAULT*) T2[$tag]=FAULT ;; *) T2[$tag]=FAIL ;; esac
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
  # an earlier pick's markers of this kind are not this one's
  rm -f $PICK.pairfail $PICK.pairfault $PICK.delivered $PICK.pending $PICK.pairing $PICK.pairing.failed
  if [ "$TN" = skip ]; then echo "$CK tier2-only" > $PICK; else echo "$CK" > $PICK; fi
  log "PICK $CK"
  deliver $PICKED
  exit $OWN_RC
fi
# no candidate passed: a tool fault when one of them was not judged (tier 2 unreadable, tier 3's tool fault)
for tag in "${ORDER[@]}"; do
  if [ "${T2[$tag]}" = FAULT ] || [[ "${T3[$tag]:-}" == *"tool fault"* ]]; then
    log "no candidate of $RUN passed, and ${CKOF[$tag]} was not judged (tool fault, not a fail)"
    exit 4
  fi
done
log "no candidate of $RUN passed"
exit 1
