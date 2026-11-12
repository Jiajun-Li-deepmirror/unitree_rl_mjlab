#!/bin/bash
# students.sh R [VARIANT] (go2-depth 2026-11-12, doc/redesign.md 3.6): robot R's blind and depth students from its
# unified teacher. Waits for jobs/unified/teacher.sh's pick (logs/jobs/unified_<r>_<variant>.pick: the checkpoint that
# passed scripts/unified/teacher_gate.py), then trains both students side by side on this GPU (NUM_ENVS / 2 robots
# each, default 4096 in all):
#   stage 1  Unitree-<R>-U-Student-<Kind>      DAgger distillation, S1_ITERS (default 4000) iterations
#   stage 2  Unitree-<R>-U-Student-<Kind>-RL   PPO + behavior cloning from stage 1's last checkpoint, S2_ITERS (3000)
# Each kind's stage-2 run is then judged by the acceptance (jobs/unified/accept.sh: the mjlab tiers, 3 seeds pooled,
# and tinynav). Log logs/jobs/unified_<r>_students.txt; runs logs/rsl_rl/<r>_u_student_<kind>[_rl]/<date>_<kind><stage>
# _t<teacher seed>_s<SEED> (the "_s<seed>" ending: tools/go2_depth/monitor/tb.sh lists runs by it; SEED default 0).
# TEACHER=<checkpoint> skips the wait. SAVE_EVERY: the checkpoint interval (default the runner's, 100).
# Fallback (2026-11-12, the review's item 10; FALLBACK=0 turns it off): when a kind's acceptance finds no candidate
# (accept.sh exit 1; a tool fault is not a fail and stops here), the next seed teacher.sh kept (<pick>.others) is
# resumed from its latest checkpoint at the same robot count, cap (TEACHER_CAP, default 30000, as a total) and code;
# once it passes its gate, the kinds that failed are trained again from it. Every teacher seed is tried once.
# No seed left to try (one seed trained, no or an empty <pick>.others, every kept seed tried, or the next one did not
# pass its gate): "STOP: needs a human" in the log and exit 3 -- not an error. Off by default with TEACHER= (that
# checkpoint is not the pick's, so the pick's .others are not its).
# Exit 0 every kind has a pick, 3 a kind failed and nothing is left to try (a human decides), 4 a tool fault.
# SMOKE=1 (jobs/unified/smoke_students.sh): runs named *_smoke, the acceptance in its smoke mode, no fallback.
set -u
R=$1 VARIANT=${2:-full}
r=${R,,}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1 TRAIN_LOG_LAYOUT=rsl_rl
SMOKE=${SMOKE:-0}
L=logs/jobs/unified_${r}_students.txt PICK=logs/jobs/unified_${r}_$VARIANT.pick
log() { echo "$(date '+%m-%d %H:%M:%S') $R students: $([ $SMOKE = 1 ] && echo 'SMOKE ')$*" >> $L; }
S1=${S1_ITERS:-4000} S2=${S2_ITERS:-3000} ENVS=$(( ${NUM_ENVS:-4096} / 2 )) SEED=${SEED:-0}
SAVE=""; [ -n "${SAVE_EVERY:-}" ] && SAVE="--agent.save-interval=$SAVE_EVERY"
FALLBACK=${FALLBACK:-$([ -n "${TEACHER:-}" ] && echo 0 || echo 1)}; [ $SMOKE = 1 ] && FALLBACK=0
T=${TEACHER:-}
if [ -z "$T" ]; then
  log "waiting for the teacher's pick ($PICK)"
  until [ -s $PICK ]; do sleep 600; done
  T=$(cut -d' ' -f1 $PICK)
fi
tseed_of() { grep -oE 'seed [0-9]+' $PICK 2> /dev/null | grep -oE '[0-9]+' | head -n 1; }
TSEED=$([ -n "${TEACHER:-}" ] && echo x || tseed_of)
TSEED=${TSEED:-x}
newest() { ls -d logs/rsl_rl/$1/*_$2 2> /dev/null | sort | tail -n 1; }
one() {   # KIND: stage 1, stage 2, then the acceptance; returns accept.sh's exit (4: a tool fault in training)
  local kind=$1 K=${1^} before run ck n1 n2
  if [ $SMOKE = 1 ]; then n1=${kind}1_smoke n2=${kind}2_smoke; else n1=${kind}1_t${TSEED}_s$SEED n2=${kind}2_t${TSEED}_s$SEED; fi
  before=$(newest ${r}_u_student_$kind $n1)
  $PY scripts/train.py Unitree-$R-U-Student-$K --env.scene.num-envs=$ENVS --agent.run-name=$n1 --agent.seed=$SEED \
    --agent.max-iterations=$S1 --agent.teacher-checkpoint=$T $SAVE > logs/jobs/unified_${r}_$n1.log 2>&1
  local rc=$?
  run=$(newest ${r}_u_student_$kind $n1)
  ck=$(ls $run/model_*.pt 2> /dev/null | sort -t_ -k2 -n | tail -n 1)
  if [ $rc != 0 ] || [ "$run" = "$before" ] || [ -z "$ck" ]; then
    log "TOOL FAULT: $kind stage 1 ended with exit $rc, run ${run:-none} ($(grep -E 'Error' logs/jobs/unified_${r}_$n1.log | tail -n 1 | cut -c1-160))"
    return 4
  fi
  log "$kind stage 1 done: $ck"
  before=$(newest ${r}_u_student_${kind}_rl $n2)
  $PY scripts/train.py Unitree-$R-U-Student-$K-RL --env.scene.num-envs=$ENVS --agent.run-name=$n2 --agent.seed=$SEED \
    --agent.max-iterations=$S2 --agent.teacher-checkpoint=$T --agent.student-checkpoint=$ck $SAVE \
    > logs/jobs/unified_${r}_$n2.log 2>&1
  rc=$?
  run=$(newest ${r}_u_student_${kind}_rl $n2)
  if [ $rc != 0 ] || [ -z "$run" ] || [ "$run" = "$before" ]; then
    log "TOOL FAULT: $kind stage 2 ended with exit $rc ($(grep -E 'Error' logs/jobs/unified_${r}_$n2.log | tail -n 1 | cut -c1-160))"
    return 4
  fi
  log "$kind stage 2 done: $run"
  ACCEPT_SMOKE=$SMOKE bash jobs/unified/accept.sh $R $kind $run >> $L 2>&1
}
KINDS="${KINDS:-blind depth}"
TRIED=" $TSEED " WHY=""
declare -A FINAL=()
while true; do
  [ -f "$T" ] || { log "no teacher checkpoint $T"; exit 2; }
  T=$(cd $(dirname $T) && pwd)/$(basename $T)
  log "teacher $T (seed $TSEED, code $(cat REPO_COMMIT 2>/dev/null)); kinds $KINDS; stage 1 $S1 it, stage 2 $S2 it, $ENVS robots each"
  declare -A PIDS=() RC=()
  for kind in $KINDS; do one $kind & PIDS[$kind]=$!; done
  failed=""
  for kind in $KINDS; do
    wait ${PIDS[$kind]}; RC[$kind]=$?
    FINAL[$kind]=${RC[$kind]}
    [ ${RC[$kind]} = 1 ] && failed="$failed $kind"
  done
  log "done with teacher seed $TSEED ($(for kind in $KINDS; do echo -n "$kind exit ${RC[$kind]} "; done))"
  [ -n "$failed" ] || break
  [ $FALLBACK = 1 ] || { WHY="fallback off$([ -n "${TEACHER:-}" ] && echo ' (TEACHER= given)')"; break; }
  # the fallback: the next teacher seed teacher.sh kept (read once: the fallback's own teacher.sh run rewrites it)
  if [ -z "${KEPT+x}" ]; then
    [ -f $PICK.others ] || { WHY="no $PICK.others (one teacher seed trained)"; break; }
    envs=$(grep -oE 'envs=[0-9]+' $PICK.others | cut -d= -f2)
    KEPT=$(sed -n 's/.*seeds=//p' $PICK.others)
  fi
  next=""
  for s in $KEPT; do [[ "$TRIED" == *" $s "* ]] || { next=$s; break; }; done
  [ -n "$next" ] || { WHY="no kept teacher seed left (tried:$TRIED; kept: ${KEPT:-none})"; break; }
  [ -n "${envs:-}" ] || { WHY="$PICK.others has no robot count"; break; }
  cp $PICK $PICK.seed$TSEED 2> /dev/null
  log "fallback: kinds$failed failed their acceptance on teacher seed $TSEED; resuming teacher seed $next (its latest checkpoint, $envs robots, cap ${TEACHER_CAP:-30000})"
  TRIED="$TRIED$next "
  NUM_ENVS=$envs PARALLEL=0 RESUME_LATEST=1 ITERS_TOTAL=1 bash jobs/unified/teacher.sh $R $VARIANT "$next" ${TEACHER_CAP:-30000}
  if [ ! -s $PICK ] || ! grep -q "seed $next" $PICK; then WHY="teacher seed $next did not pass its gate"; break; fi
  T=$(cut -d' ' -f1 $PICK) TSEED=$next KINDS=$failed
done
# the outcome over every round: a kind whose last acceptance failed with nothing left to try is for a human
nopick="" fault=""
for kind in "${!FINAL[@]}"; do
  case ${FINAL[$kind]} in 0) ;; 1) nopick="$nopick $kind" ;; *) fault="$fault $kind(exit ${FINAL[$kind]})" ;; esac
done
if [ -n "$nopick" ]; then
  log "STOP: needs a human -- kinds$nopick have no accepted student (${WHY:-no fallback}); the acceptance reports are in logs/jobs/unified_${r}_accept.txt"
  exit 3
fi
[ -z "$fault" ] || { log "tool fault:$fault (not a fail; nothing retried)"; exit 4; }
log "every kind has a pick"
exit 0
