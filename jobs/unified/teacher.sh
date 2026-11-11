#!/bin/bash
# teacher.sh R [VARIANT] [SEEDS] [ITERS] (go2-depth 2026-11-11, the user's redesign: "一个带特权信息的老师 ... 每次同时跑多个种子";
# src/tasks/unified): the unified privileged teacher of robot R (Go2, Go2W) from zero, one run through the whole
# curriculum, on several seeds.
#   VARIANT  full (default: the recipe's shaping terms kept) or lean (core rewards + constrained costs only)
#   SEEDS    "0 1" (default); PARALLEL=1 trains them side by side on this GPU at NUM_ENVS / seeds robots each
#            (default 4096 in all), else one after the other at NUM_ENVS (4096)
#   ITERS    the cap (default 30000); a seed stops at its cap or when another seed has passed
# Every GATE_EVERY s (default 900) scripts/unified/teacher_gate.py judges each running seed on its constraints in
# training; the first to pass is the pick: logs/jobs/unified_<r>_<variant>.pick (checkpoint path), the other seeds
# are stopped (KEEP_OTHERS=1 lets them run on). Smoke: U_SMOKE=1 (few iterations, every switch-on forced).
# Log logs/jobs/unified_<r>.txt; runs logs/rsl_rl/<r>_u_teacher/<date>_u<variant>_s<seed>.
# RESUME_RUN (a run dir) + RESUME_K: every seed resumes from that run's model_K (the runner restores the curriculum's
# progress, multipliers and meter; 2026-11-11, after the progress clock was added); for one seed. RESUME_LATEST=1: each seed
# resumes from its own newest run (*_u<variant>_s<seed>) at its latest checkpoint (2026-11-12).
set -u
R=$1 VARIANT=${2:-full} SEEDS=${3:-0 1} ITERS=${4:-30000}
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1 TRAIN_LOG_LAYOUT=rsl_rl
case $VARIANT in full) TASK=Unitree-$R-U-Teacher ;; lean) TASK=Unitree-$R-U-Teacher-Lean ;; *) echo "teacher.sh: VARIANT full or lean" >&2; exit 2 ;; esac
L=logs/jobs/unified_$r.txt PICK=logs/jobs/unified_${r}_$VARIANT.pick
mkdir -p logs/jobs
log() { echo "$(date '+%m-%d %H:%M:%S') $R unified teacher $VARIANT: $*" >> $L; }
n=$(echo $SEEDS | wc -w)
ENVS=${NUM_ENVS:-4096}; [ "${PARALLEL:-0}" = 1 ] && ENVS=$(( ${NUM_ENVS:-4096} / n ))
log "${RESUME_RUN:+resume from $(basename $RESUME_RUN) model_$RESUME_K; }seeds $SEEDS, $ENVS robots each, $([ "${PARALLEL:-0}" = 1 ] && echo side by side || echo one after the other), cap $ITERS it, code $(cat REPO_COMMIT 2>/dev/null)"
declare -A PID RUN
start() {   # SEED
  local s=$1 name=u${VARIANT}_s$1
  local res="" before
  [ -n "${RESUME_RUN:-}" ] && res="--agent.resume=True --agent.load-run=^$(basename $RESUME_RUN)\$ --agent.load-checkpoint=^model_${RESUME_K}\.pt\$"
  if [ "${RESUME_LATEST:-0}" = 1 ]; then
    # the newest run of this seed that has a checkpoint (a run stopped before its first save has none)
    local rr="" rk="" d
    for d in $(ls -d logs/rsl_rl/${r}_u_teacher/*_$name 2> /dev/null | sort -r); do
      rk=$(ls $d/model_*.pt 2> /dev/null | grep -oE 'model_[0-9]+' | cut -d_ -f2 | sort -n | tail -n 1)
      [ -n "$rk" ] && { rr=$d; break; }
    done
    [ -n "$rk" ] && res="--agent.resume=True --agent.load-run=^$(basename $rr)\$ --agent.load-checkpoint=^model_${rk}\.pt\$" \
      && log "seed $s resumes from $(basename $rr) model_$rk"
  fi
  # the run dirs of this name already there (a resume's source among them): the new run is the one that appears after
  before=$(ls -d logs/rsl_rl/${r}_u_teacher/*_$name 2> /dev/null | sort | tail -n 1)
  $PY scripts/train.py $TASK --env.scene.num-envs=$ENVS --agent.run-name=$name --agent.seed=$s \
    --agent.max-iterations=$ITERS $res > logs/jobs/unified_${r}_$name.log 2>&1 &
  PID[$s]=$!
  RUN[$s]=""
  for _ in $(seq 80); do
    d=$(ls -d logs/rsl_rl/${r}_u_teacher/*_$name 2> /dev/null | sort | tail -n 1)
    [ -n "$d" ] && [ "$d" != "$before" ] && { RUN[$s]=$d; break; }
    sleep 15
  done
  log "seed $s started (pid ${PID[$s]}, run ${RUN[$s]:-none yet})"
}
gate() {    # SEED -> 0 when it passed (writes the pick)
  local s=$1 v
  [ -n "${RUN[$s]:-}" ] || RUN[$s]=$(ls -d logs/rsl_rl/${r}_u_teacher/*_u${VARIANT}_s$s 2> /dev/null | sort | tail -n 1)   # the newest
  [ -n "${RUN[$s]:-}" ] || return 1
  v=$($PY scripts/unified/teacher_gate.py ${RUN[$s]} 2> /dev/null | head -n 1)
  log "seed $s: ${v:-no verdict}"
  if [[ "$v" == PASS* ]]; then
    local k=$(echo $v | cut -d' ' -f2)
    echo "${RUN[$s]}/model_$k.pt seed $s" > $PICK
    log "TEACHER PICK ${RUN[$s]}/model_$k.pt (seed $s)"
    return 0
  fi
  return 1
}
stop() { kill -INT $1 2> /dev/null; for _ in $(seq 60); do kill -0 $1 2> /dev/null || return; sleep 2; done; kill $1 2> /dev/null; }
rm -f $PICK
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
  if [ -n "$passed" ] && [ "${KEEP_OTHERS:-0}" != 1 ]; then
    for s in $SEEDS; do [ $s = $passed ] || { stop ${PID[$s]}; log "seed $s stopped (seed $passed passed)"; }; done
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
if [ -n "$passed" ]; then log "PASS (seed $passed: $(cat $PICK))"; exit 0; fi
log "no seed passed"; exit 1
