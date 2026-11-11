#!/bin/bash
# students.sh R [VARIANT] (go2-depth 2026-11-12, doc/redesign.md 3.6): robot R's blind and depth students from its
# unified teacher. Waits for jobs/unified/teacher.sh's pick (logs/jobs/unified_<r>_<variant>.pick: the checkpoint that
# passed scripts/unified/teacher_gate.py), then trains both students side by side on this GPU (NUM_ENVS / 2 robots
# each, default 4096 in all):
#   stage 1  Unitree-<R>-U-Student-<Kind>      DAgger distillation, S1_ITERS (default 4000) iterations
#   stage 2  Unitree-<R>-U-Student-<Kind>-RL   PPO + behavior cloning from stage 1's last checkpoint, S2_ITERS (3000)
# Each kind's stage-2 run is then judged by the acceptance (scripts/unified/accept.sh when present: the mjlab tiers,
# 3 seeds pooled, and tinynav); without it the run's checkpoints wait for it. Log logs/jobs/unified_<r>_students.txt;
# runs logs/rsl_rl/<r>_u_student_<kind>[_rl]/<date>_<kind>s{1,2}. TEACHER=<checkpoint> skips the wait.
set -u
R=$1 VARIANT=${2:-full}
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1 TRAIN_LOG_LAYOUT=rsl_rl
L=logs/jobs/unified_${r}_students.txt PICK=logs/jobs/unified_${r}_$VARIANT.pick
log() { echo "$(date '+%m-%d %H:%M:%S') $R students: $*" >> $L; }
S1=${S1_ITERS:-4000} S2=${S2_ITERS:-3000} ENVS=$(( ${NUM_ENVS:-4096} / 2 ))
T=${TEACHER:-}
if [ -z "$T" ]; then
  log "waiting for the teacher's pick ($PICK)"
  until [ -s $PICK ]; do sleep 600; done
  T=$(cut -d' ' -f1 $PICK)
fi
[ -f "$T" ] || { log "no teacher checkpoint $T"; exit 2; }
T=$(cd $(dirname $T) && pwd)/$(basename $T)
log "teacher $T (code $(cat REPO_COMMIT 2>/dev/null)); stage 1 $S1 it, stage 2 $S2 it, $ENVS robots each"
newest() { ls -d logs/rsl_rl/$1/*_$2 2> /dev/null | sort | tail -n 1; }
one() {   # KIND: stage 1 then stage 2
  local kind=$1 K=${1^} before run ck
  before=$(newest ${r}_u_student_$kind ${kind}s1)
  $PY scripts/train.py Unitree-$R-U-Student-$K --env.scene.num-envs=$ENVS --agent.run-name=${kind}s1 \
    --agent.max-iterations=$S1 --agent.teacher-checkpoint=$T > logs/jobs/unified_${r}_${kind}s1.log 2>&1
  local rc=$?
  run=$(newest ${r}_u_student_$kind ${kind}s1)
  ck=$(ls $run/model_*.pt 2> /dev/null | sort -t_ -k2 -n | tail -n 1)
  if [ $rc != 0 ] || [ "$run" = "$before" ] || [ -z "$ck" ]; then
    log "TOOL FAULT: $kind stage 1 ended with exit $rc, run ${run:-none} ($(grep -E 'Error' logs/jobs/unified_${r}_${kind}s1.log | tail -n 1 | cut -c1-160))"
    return 4
  fi
  log "$kind stage 1 done: $ck"
  $PY scripts/train.py Unitree-$R-U-Student-$K-RL --env.scene.num-envs=$ENVS --agent.run-name=${kind}s2 \
    --agent.max-iterations=$S2 --agent.teacher-checkpoint=$T --agent.student-checkpoint=$ck \
    > logs/jobs/unified_${r}_${kind}s2.log 2>&1
  rc=$?
  run=$(newest ${r}_u_student_${kind}_rl ${kind}s2)
  if [ $rc != 0 ] || [ -z "$run" ]; then
    log "TOOL FAULT: $kind stage 2 ended with exit $rc ($(grep -E 'Error' logs/jobs/unified_${r}_${kind}s2.log | tail -n 1 | cut -c1-160))"
    return 4
  fi
  log "$kind stage 2 done: $run"
  if [ -f jobs/unified/accept.sh ]; then
    bash jobs/unified/accept.sh $R $kind $run >> $L 2>&1
  else
    log "$kind: no jobs/unified/accept.sh yet -- the checkpoints of $run wait for the acceptance"
  fi
}
one blind &
pb=$!
one depth &
pd=$!
wait $pb; rb=$?
wait $pd; rd=$?
log "done (blind exit $rb, depth exit $rd)"
