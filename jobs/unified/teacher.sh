#!/bin/bash
# teacher.sh R [VARIANT] [SEEDS] [ITERS] (go2-depth 2026-11-11, the user's redesign: "一个带特权信息的老师 ... 每次同时跑多个种子";
# src/tasks/unified): the unified privileged teacher of robot R (Go2, Go2W) from zero, one run through the whole
# curriculum, on several seeds.
#   VARIANT  full (default: the recipe's shaping terms kept) or lean (core rewards + constrained costs only)
#   SEEDS    F_TEACHER_SEEDS (jobs/unified/formal.env: "0 1"); PARALLEL (F_TEACHER_PARALLEL: 1) trains them side by
#            side on this GPU at NUM_ENVS / seeds robots each (NUM_ENVS default F_TEACHER_ENVS_PER_SEED x seeds), else
#            one after the other at NUM_ENVS
#   ITERS    the cap (F_TEACHER_ITERS: 12000); a seed stops at its cap or when another seed has passed
# Every GATE_EVERY s (default 900) scripts/unified/teacher_gate.py judges each running seed on its constraints in
# training; the first to pass is the pick: logs/jobs/unified_<r>_<variant>.pick (checkpoint path), the other seeds
# are stopped (KEEP_OTHERS=1 lets them run on). Smoke: U_SMOKE=1 (few iterations, every switch-on forced).
# Log logs/jobs/unified_<r>.txt; runs logs/rsl_rl/<r>_u_teacher/<date>_u<variant>_s<seed>.
# RESUME_RUN (a run dir) + RESUME_K: every seed resumes from that run's model_K (the runner restores the curriculum's
# progress, multipliers and meter; 2026-11-11, after the progress clock was added); for one seed. RESUME_LATEST=1: each seed
# resumes from its own newest run (*_u<variant>_s<seed>) at its latest checkpoint (2026-11-12).
# ITERS_TOTAL=1: ITERS is the total (a resumed seed trains ITERS - K more), so a seed resumed later ends where it would
# have; default off (ITERS more iterations, as before).
# The seeds a pass stops keep their runs and checkpoints; <pick>.others records them ("envs=<robots per seed>
# seeds=<the others>", 2026-11-12, the review's item 10): jobs/unified/students.sh resumes one of them, at the same
# robot count and code, when a student from the pick fails its acceptance.
set -u
R=$1
r=${R,,}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
# the formal run's parameters (jobs/unified/formal.env, frozen with the commit): the defaults below
source jobs/unified/formal.env
VARIANT=${2:-full} SEEDS=${3:-$F_TEACHER_SEEDS} ITERS=${4:-$F_TEACHER_ITERS}
PARALLEL=${PARALLEL:-$F_TEACHER_PARALLEL} GATE_EVERY=${GATE_EVERY:-$F_GATE_EVERY_S}
NUM_ENVS=${NUM_ENVS:-$(( F_TEACHER_ENVS_PER_SEED * $(echo $SEEDS | wc -w) ))}
export UNIFIED_TASKS=1 TRAIN_LOG_LAYOUT=rsl_rl
case $VARIANT in full) TASK=Unitree-$R-U-Teacher ;; lean) TASK=Unitree-$R-U-Teacher-Lean ;; *) echo "teacher.sh: VARIANT full or lean" >&2; exit 2 ;; esac
L=logs/jobs/unified_$r.txt PICK=logs/jobs/unified_${r}_$VARIANT.pick
mkdir -p logs/jobs
log() { echo "$(date '+%m-%d %H:%M:%S') $R unified teacher $VARIANT: $*" >> $L; }
n=$(echo $SEEDS | wc -w)
ENVS=${NUM_ENVS:-4096}; [ "${PARALLEL:-0}" = 1 ] && ENVS=$(( ${NUM_ENVS:-4096} / n ))
log "${RESUME_RUN:+resume from $(basename $RESUME_RUN) model_$RESUME_K; }seeds $SEEDS, $ENVS robots each, $([ "${PARALLEL:-0}" = 1 ] && echo side by side || echo one after the other), cap $ITERS it, code $(cat REPO_COMMIT 2>/dev/null), formal.env $(md5sum jobs/unified/formal.env | cut -c1-8)"
declare -A PID RUN BEFORE
start() {   # SEED
  local s=$1 name=u${VARIANT}_s$1
  local res="" before from=0
  # the run dirs of this name already there (a resume's source, old runs): the new run is the one that appears after;
  # gate() never judges this one (2026-11-12, go2-depth's review: a fallback to the newest dir judged an old run)
  before=$(ls -d logs/rsl_rl/${r}_u_teacher/*_$name 2> /dev/null | sort | tail -n 1)
  BEFORE[$s]=$before
  [ -n "${RESUME_RUN:-}" ] && res="--agent.resume=True --agent.load-run=^$(basename $RESUME_RUN)\$ --agent.load-checkpoint=^model_${RESUME_K}\.pt\$" \
    && from=$RESUME_K
  if [ "${RESUME_LATEST:-0}" = 1 ]; then
    # the newest run of this seed that has a checkpoint (a run stopped before its first save has none)
    local rr="" rk="" d
    for d in $(ls -d logs/rsl_rl/${r}_u_teacher/*_$name 2> /dev/null | sort -r); do
      rk=$(ls $d/model_*.pt 2> /dev/null | grep -oE 'model_[0-9]+' | cut -d_ -f2 | sort -n | tail -n 1)
      # RESUME_AT=K: that run's model_K instead of its latest (going back before a run went wrong); RESUME_AT_<seed>=K
      # for one seed only (the others from their latest)
      local at=RESUME_AT_$s; at=${!at:-${RESUME_AT:-}}
      [ -n "$at" ] && { [ -f $d/model_$at.pt ] && rk=$at || rk=""; }
      [ -n "$rk" ] && { rr=$d; break; }
    done
    [ -n "$rk" ] && res="--agent.resume=True --agent.load-run=^$(basename $rr)\$ --agent.load-checkpoint=^model_${rk}\.pt\$" \
      && from=$rk && log "seed $s resumes from $(basename $rr) model_$rk"
  fi
  local more=$ITERS
  if [ "${ITERS_TOTAL:-0}" = 1 ]; then
    more=$(( ITERS - from ))
    [ $more -gt 0 ] || { log "seed $s: already at the cap ($from >= $ITERS), not started"; PID[$s]=""; RUN[$s]=""; return; }
  fi
  $PY scripts/train.py $TASK --env.scene.num-envs=$ENVS --agent.run-name=$name --agent.seed=$s \
    --agent.max-iterations=$more $res > logs/jobs/unified_${r}_$name.log 2>&1 &
  PID[$s]=$!
  RUN[$s]=""
  for _ in $(seq ${RUN_WAIT_TRIES:-80}); do
    d=$(ls -d logs/rsl_rl/${r}_u_teacher/*_$name 2> /dev/null | sort | tail -n 1)
    [ -n "$d" ] && [ "$d" != "$before" ] && { RUN[$s]=$d; break; }
    sleep 15
  done
  log "seed $s started (pid ${PID[$s]}, run ${RUN[$s]:-none yet})"
}
gate() {    # SEED -> 0 when it passed (writes the pick)
  local s=$1 v d
  if [ -z "${RUN[$s]:-}" ]; then
    # the run dir did not show while start() waited: look again, but only for one newer than the runs before this start
    d=$(ls -d logs/rsl_rl/${r}_u_teacher/*_u${VARIANT}_s$s 2> /dev/null | sort | tail -n 1)
    if [ -z "$d" ] || [ "$d" = "${BEFORE[$s]:-}" ]; then log "seed $s: its new run dir is not there yet"; return 1; fi
    RUN[$s]=$d
  fi
  v=$($PY scripts/unified/teacher_gate.py ${RUN[$s]} --hold $F_GATE_HOLD --every $F_GATE_EVERY_IT 2> /dev/null | head -n 1)
  log "seed $s: ${v:-no verdict}"
  if [[ "$v" == PASS* ]]; then
    local k=$(echo $v | cut -d' ' -f2)
    # the events can be ahead of the save: the pick only once its checkpoint is on disk (else the next gate)
    [ -f ${RUN[$s]}/model_$k.pt ] || { log "seed $s: passes at $k, but model_$k.pt is not saved yet (next gate)"; return 1; }
    echo "${RUN[$s]}/model_$k.pt seed $s" > $PICK
    log "TEACHER PICK ${RUN[$s]}/model_$k.pt (seed $s)"
    return 0
  fi
  return 1
}
stop() {   # PID...: all at once. SIGTERM (a background job of a non-interactive shell ignores SIGINT, and Python then
  # installs no KeyboardInterrupt handler; the unified runner's training records end on SIGTERM as "stopped" with a
  # last entry, diagnostics.py), then SIGKILL for whatever still runs after STOP_WAIT_S (60). The checkpoints are on
  # disk already; nothing is saved on the way out.
  local p alive="$*"
  kill -TERM "$@" 2> /dev/null
  for _ in $(seq $(( ${STOP_WAIT_S:-60} / 2 ))); do
    alive=""; for p in "$@"; do kill -0 $p 2> /dev/null && alive="$alive $p"; done
    [ -n "$alive" ] || return 0
    sleep 2
  done
  [ -z "$alive" ] || kill -KILL $alive 2> /dev/null
}
rm -f $PICK $PICK.others   # (a previous run's kept seeds are not this run's)
passed=""
if [ "${PARALLEL:-0}" = 1 ]; then
  for s in $SEEDS; do start $s; done
  while [ -z "$passed" ]; do
    alive=0
    for s in $SEEDS; do kill -0 ${PID[$s]} 2> /dev/null && alive=1; done
    sleep ${GATE_EVERY:-900}
    for s in $SEEDS; do gate $s && { passed=$s; break; }; done
    [ $alive = 0 ] && break
  done
  if [ -n "$passed" ]; then
    # the pick's seed stops too, as in the sequential branch (go2-depth's review 11-12: it trained on to its cap, and
    # this script's exit, the .others and anything waiting for it with it); its checkpoints stay
    if [ "${KEEP_OTHERS:-0}" != 1 ]; then
      stop $(for s in $SEEDS; do echo ${PID[$s]}; done)
      log "seeds $SEEDS stopped (seed $passed passed; every seed's checkpoints stay)"
    else
      stop ${PID[$passed]}; log "seed $passed stopped at its pick (its checkpoints stay)"
      log "KEEP_OTHERS=1: the other seeds run on to their caps; <pick>.others is written when they end"
    fi
  fi
  wait
else
  for s in $SEEDS; do
    start $s
    while kill -0 ${PID[$s]} 2> /dev/null; do
      sleep ${GATE_EVERY:-900}
      gate $s && { passed=$s; break; }
    done
    if [ -n "$passed" ]; then stop ${PID[$s]}; break; fi
    wait ${PID[$s]}; log "seed $s ended (exit $?) without passing"
    gate $s && { passed=$s; break; }
  done
fi
if [ -n "$passed" ]; then
  others=$(for s in $SEEDS; do [ $s = $passed ] || echo -n "$s "; done)
  echo "envs=$ENVS seeds=${others% }" > $PICK.others
  log "PASS (seed $passed: $(cat $PICK)); kept for a fallback: seeds ${others:-none} ($PICK.others)"
  exit 0
fi
log "no seed passed"; exit 1
