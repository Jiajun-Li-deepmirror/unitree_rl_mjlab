#!/bin/bash
# queue_runner.sh BOX [--once] (go2-depth 2026-11-16, go2-home's automatic chaining; the user: two runs per card): the
# box's Go2W run queue. Input logs/jobs/go2w_queue_<BOX>.txt -- one run per line, edited by go2-depth on go2-home's
# word only ("nothing is queued without my word"):
#     NAME COMMIT SEED ITERS ENVS [FT=SRC_RUN@K] [GATE_TESTS=key,key,...] [VAR=VALUE ...]
# ('#' starts a comment). Every 60 s:
#   1. a run it launched that has ended (records/status.json "ended") gets its end gate -- accept.sh Go2W teacher in
#      the MAIN tree (the newest tools), BASELINE on the avg4 at its last checkpoint, U_SCAN=on, the run's switches;
#      GATE_TESTS: a screening gate (accept.sh ACCEPT_TESTS, the judge's --screen) -- one gate at a time on the box
#      (none while another accept.sh teacher gate runs); its verdict line goes to logs/jobs/go2w_queue_results.txt;
#   2. while fewer than QUEUE_MAX_RUNS (2) Go2W teacher trainings run and QUEUE_FREE_MB (8000) are free on the card,
#      the next line with no state: its commit's tree trees/<COMMIT> (tools/go2_depth/tree_commit.sh; missing: the
#      line waits, logged once), a smoke there (64 x 20, the run's switches, FT too), and only on PASS the run.
# State: logs/jobs/go2w_queue_<BOX>.state ("NAME state ..."; the last line of a name counts: smoking, launched,
# gating, done, failed). A stop file logs/jobs/go2w_queue_<BOX>.stop ends the runner. QUEUE_DRY=1: nothing started,
# the actions logged ("dry: ..."). Lines into logs/jobs/unified_go2w_arms.txt ("Go2W queue <BOX>: ...").
set -u
BOX=$1 ONCE=0; [ "${2:-}" = --once ] && ONCE=1
M=/root/gpufree-data/go2-depth/depth_fe TREES=/root/gpufree-data/go2-depth/trees
cd $M || exit 2
Q=logs/jobs/go2w_queue_$BOX.txt S=logs/jobs/go2w_queue_$BOX.state RES=logs/jobs/go2w_queue_results.txt
A=logs/jobs/unified_go2w_arms.txt H=jobs/unified/queue
MAXR=${QUEUE_MAX_RUNS:-2} FREE=${QUEUE_FREE_MB:-8000} DRY=${QUEUE_DRY:-0}
touch $S
log() { echo "$(date '+%m-%d %H:%M:%S') Go2W queue $BOX: $*" >> $A; }
state() { awk -v n="$1" '$1 == n {s = $2} END {print s}' $S; }
setst() { echo "$*" >> $S; }
line_of() { grep -v '^\s*#' $Q 2>/dev/null | awk -v n="$1" '$1 == n' | head -n 1; }
gate_free() { ! pgrep -f "accept.sh Go2W teacher" > /dev/null; }
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -n 1 | awk -F', *' '{print $1 - $2}'; }
log "runner up (max $MAXR runs, $FREE MiB free to start one$([ $DRY = 1 ] && echo ', a dry run'))"
declare -A WAITLOG
while :; do
  [ -f logs/jobs/go2w_queue_$BOX.stop ] && { log "stop file: the runner ends"; exit 0; }
  # 1. the end gate of a launched run that ended
  if gate_free; then
    for name in $(awk '{print $1}' $S | sort -u); do
      [ "$(state $name)" = launched ] || continue
      d=$(ls -d logs/rsl_rl/go2w_u_teacher/*_$name 2>/dev/null | tail -n 1)
      e=$(python3 -c "import json; print(json.load(open('$d/records/status.json')).get('ended') or '')" 2>/dev/null)
      [ -n "$e" ] || continue
      K=$(ls $d | grep -oE '^model_[0-9]+\.pt$' | grep -oE '[0-9]+' | sort -n | tail -n 1)
      read -r _n _c _s _i _e rest <<< "$(line_of $name)"
      sw=() gt=""
      for kv in $rest; do case $kv in FT=*) ;; GATE_TESTS=*) gt=${kv#GATE_TESTS=}; gt=${gt//,/ } ;; U_SMOKE=*) ;; *) sw+=("$kv") ;; esac; done
      if [ $DRY = 1 ]; then log "dry: would gate $name (ended $e) on avg4 at $K${gt:+, screening: $gt}, switches ${sw[*]:-none}"; setst "$name done dry"; break; fi
      setst "$name gating $K"
      log "$name ended ($e): its gate on avg4 at $K${gt:+ (screening: $gt)} starts"
      (
        source env.sh > /dev/null 2>&1; source jobs/unified/formal.env
        for v in $(env | grep -oE '^U_[A-Z0-9_]+'); do unset $v; done
        export UNIFIED_TASKS=1 U_SCAN=on
        for kv in "${sw[@]}"; do export "$kv"; done
        NEED="$F_S2_GATE_NEED_MB push:1500 slope:1800 wall:4500 stairsedge:2400 noopedge:2400 opedge:2400"
        t0=$(date +%s)
        ACCEPT_TESTS="$gt" BASELINE=1 CANDIDATES=avg AVG_N=4 TN_GATE=skip ACCEPT_JOBS=1 ACCEPT_DRIFT_SPLIT=1 \
          ACCEPT_NEED_MB="$NEED" ACCEPT_FREE_WAIT_S=3600 bash jobs/unified/accept.sh Go2W teacher $d $K > /dev/null 2>&1
        v=$(head -n 1 logs/checks/u_go2w_teacher_$(basename $d)_a4_${K}_verdict.txt 2>/dev/null | cut -c1-500)
        echo "$(date '+%m-%d %H:%M') $BOX $name | $(line_of $name | cut -d' ' -f2-) | avg4 $K${gt:+ (screening: $gt)} | ${v:-no verdict} | $(( ($(date +%s) - t0) / 60 )) min | logs/checks/u_go2w_teacher_$(basename $d)_a4_${K}_verdict.txt" >> $RES
        setst "$name done"
        log "$name: gate done: ${v:-no verdict}"
      ) &
      break
    done
  fi
  # 2. the next run
  running=$(pgrep -f "scripts/train.py Unitree-Go2W-U-Teacher " | wc -l)
  busy=$(awk '{print $1}' $S | sort -u | while read -r n; do [ "$(state $n)" = smoking ] && echo $n; done | wc -l)
  f=$(free_mb)
  if [ $running -lt $MAXR ] && [ $busy = 0 ] && [ "${f:-0}" -ge $FREE ]; then
    next=""
    while read -r n c s i e rest; do
      [ -z "${n:-}" ] && continue
      [ -z "$(state $n)" ] || continue
      if [ ! -f $TREES/$c/REPO_COMMIT ] || [ "$(cat $TREES/$c/REPO_COMMIT)" != "$c" ]; then
        [ -z "${WAITLOG[$n]:-}" ] && log "$n waits for its tree $TREES/$c (tools/go2_depth/tree_commit.sh $BOX $c)"; WAITLOG[$n]=1; continue
      fi
      next="$n $c $s $i $e $rest"; break
    done < <(grep -v '^\s*#' $Q 2>/dev/null)
    if [ -n "$next" ]; then
      read -r n c s i e rest <<< "$next"
      args=(); for kv in $rest; do case $kv in GATE_TESTS=*) ;; *) args+=("$kv") ;; esac; done
      if [ $DRY = 1 ]; then
        log "dry: would smoke and launch $n on $c ($TREES/$c): seed $s, $i it, $e robots, ${args[*]:-no switches}"
        setst "$n done dry"
      else
        setst "$n smoking"
        out=$(UNIFIED_ROOT=$TREES/$c bash $H/launch_run.sh usmoke_q_$n 0 20 64 U_SMOKE=1 "${args[@]}" 2>&1 | tail -n 1)
        pid=$(echo "$out" | grep -oE 'pid [0-9]+' | grep -oE '[0-9]+')
        res=$([ -n "$pid" ] && bash $H/smoke_check.sh $pid usmoke_q_$n 2>&1 || echo "FAIL: the smoke did not start ($out)")
        if [[ $res == PASS* ]]; then
          out=$(UNIFIED_ROOT=$TREES/$c bash $H/launch_run.sh $n $s $i $e "${args[@]}" 2>&1 | tail -n 1)
          pid=$(echo "$out" | grep -oE 'pid [0-9]+' | grep -oE '[0-9]+')
          if [ -n "$pid" ]; then setst "$n launched $pid"; log "smoke $res; $out"; else setst "$n failed launch"; log "$n NOT launched: $out"; fi
        else
          setst "$n failed smoke"; log "$n NOT launched: smoke $res"
        fi
      fi
    fi
  fi
  [ $ONCE = 1 ] && { wait; exit 0; }
  sleep 60
done
