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
# Before each teacher's students: its stops on flat (jobs/unified/stop_check.sh; STOP_CHECK=0 skips it): a group over
# 1% falls -> "STOP: needs a human", exit 3; 0-1% -> a warning in the log.
# Exit 0 every kind has a pick, 3 a kind failed and nothing is left to try (a human decides), 4 a tool fault.
# SMOKE=1 (jobs/unified/smoke_students.sh): runs named *_smoke, the acceptance in its smoke mode, no fallback.
# STUDENT_PROBE=1 (2026-11-13, go2-depth: a student from a probe teacher, TEACHER= required): runs named *_probe, the
# picks left alone, no acceptance, no fallback -- the stage-2 run is the output; read it with
# BASELINE=1 CANDIDATES=first TN_GATE=skip bash jobs/unified/accept.sh R KIND RUN (tier 2 only: no pick, no delivery).
# Stopping a formal run by hand: stop this script first (its PID, TERM), then the trainings. A stage 2 whose training
# alone is TERMed counts as a fault and is started over from the stage-1 end (F_S2_RETRIES; 2026-11-13: stage 2 is
# never resumed -- student_ppo's resume restores neither the BC progress nor num_updates).
set -u
R=$1 VARIANT=${2:-full}
r=${R,,}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1 TRAIN_LOG_LAYOUT=rsl_rl
SMOKE=${SMOKE:-0} PROBE=${STUDENT_PROBE:-0}
L=logs/jobs/unified_${r}_students.txt PICK=logs/jobs/unified_${r}_$VARIANT.pick
log() { echo "$(date '+%m-%d %H:%M:%S') $R students: $([ $SMOKE = 1 ] && echo 'SMOKE ')$([ $PROBE = 1 ] && echo 'PROBE ')$*" >> $L; }
if [ $PROBE = 1 ]; then
  [ -n "${TEACHER:-}" ] || { echo "students.sh: STUDENT_PROBE=1 needs TEACHER=<checkpoint>" >&2; exit 2; }
  [ $SMOKE = 1 ] && { echo "students.sh: STUDENT_PROBE=1 and SMOKE=1 together" >&2; exit 2; }
fi
source jobs/unified/formal.env   # the formal run's parameters, frozen with the commit
# the students' straight-line flat items at least F_STUDENT_LINE_FLOOR times their weight (2026-11-14, go2-depth's
# decision on probe S2K; src/tasks/unified/students.py line_floor): exported for the trainings, so each run's timeline
# records it; an explicit U_STUDENT_LINE_FLOOR (a probe's) wins
export U_STUDENT_LINE_FLOOR=${U_STUDENT_LINE_FLOOR:-$F_STUDENT_LINE_FLOOR}
S1=${S1_ITERS:-$F_STUDENT_S1_ITERS} S2=${S2_ITERS:-$F_STUDENT_S2_ITERS} SEED=${SEED:-$F_STUDENT_SEED}
# the stage-2 gates' candidate is the acceptance's (its verdict reused by name): the same average, or no run
if [ $PROBE != 1 ] && [ $SMOKE != 1 ] && [ "${F_S2_GATE_EVERY_IT:-0}" -gt 0 ] && [ "$F_S2_GATE_AVG_N" != "${AVG_N:-$F_ACCEPT_AVG_N}" ]; then
  log "setup error: F_S2_GATE_AVG_N ($F_S2_GATE_AVG_N) differs from the acceptance's AVG_N (${AVG_N:-$F_ACCEPT_AVG_N}) -- the gates' verdicts would not be reused (jobs/unified/formal.env)"
  echo "students.sh: F_S2_GATE_AVG_N != the acceptance's AVG_N" >&2
  exit 2
fi
ENVS=$([ -n "${NUM_ENVS:-}" ] && echo $(( NUM_ENVS / 2 )) || echo $F_STUDENT_ENVS_EACH)
SAVE=""; [ -n "${SAVE_EVERY:-}" ] && SAVE="--agent.save-interval=$SAVE_EVERY"
# stage 2's saves (formal runs; SAVE_EVERY, the smoke's, wins; probes keep the runner's 100)
SAVE2=$SAVE; [ -z "$SAVE2" ] && [ $PROBE != 1 ] && [ $SMOKE != 1 ] && SAVE2="--agent.save-interval=$F_STUDENT_S2_SAVE_EVERY"
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
stopw() {   # PID: SIGTERM, then SIGKILL after 60 s (teacher.sh's stop; the checkpoints are on disk already)
  kill -TERM $1 2> /dev/null
  for _ in $(seq 30); do kill -0 $1 2> /dev/null || return 0; sleep 2; done
  kill -KILL $1 2> /dev/null
}
tl() {   # RUN EV IT KEY=VALUE...: one event line in RUN's records/timeline.jsonl (beside the runner's own events; a
  # small line appended at once), so the run's record says why a stage ended where it did
  local run=$1; shift
  [ -d $run/records ] || return 0
  $PY - "$@" >> $run/records/timeline.jsonl 2> /dev/null <<'PYEOF'
import json, sys, time
ev, it, *kv = sys.argv[1:]
d = {"ev": ev, "it": int(it), "t": round(time.time(), 1)}
for x in kv:
  k, v = x.split("=", 1)
  d[k] = v
print(json.dumps(d, ensure_ascii=False))
PYEOF
}
s2_gate_run() {   # -n|-x KIND RUN K [FAILFAST]: the students' tier 2 on the average at K, one gate per card (-n: skip,
  # exit 75, when another runs; -x: wait for it), one test at a time with its memory need free (jobs/unified/accept.sh);
  # FAILFAST 1 (the gates during training, F_S2_GATE_FAILFAST; never the cap's): accept.sh ACCEPT_FAILFAST=1 -- the first
  # group judged first, the gate ending there when it does not pass
  local how=$1 kind=$2 run=$3 k=$4 ff=${5:-0}
  ( flock $how 9 || { log "STUDENT $kind S2 GATE $k: skipped (another gate runs on this card)"; exit 75; }
    ACCEPT_FAILFAST=$ff BASELINE=1 CANDIDATES=avg AVG_N=$F_S2_GATE_AVG_N TN_GATE=skip ACCEPT_JOBS=1 ACCEPT_PEAK=1 ACCEPT_DRIFT_SPLIT=1 \
      ACCEPT_NEED_MB="$([ $kind = depth ] && echo "$F_S2_GATE_NEED_MB_DEPTH" || echo "$F_S2_GATE_NEED_MB")" ACCEPT_FREE_WAIT_S=$F_S2_GATE_WAIT_S \
      ACCEPT_SMALL="$([ $kind = depth ] && echo "${F_S2_GATE_SMALL_DEPTH:-}")" \
      bash jobs/unified/accept.sh $R $kind $run $k >> $L 2>&1 ) 9> logs/jobs/unified_s2gate.lock
}
s2_gate_read() {   # KIND RUN K END -> GATE_OUT (pass | hard | fail | fault), GATE_GAP; logs the event
  local kind=$1 run=$2 k=$3 e=$4 P v q
  P=logs/checks/u_${r}_${kind}_$(basename $run)_a${F_S2_GATE_AVG_N}_$e
  v=$(head -n 1 ${P}_verdict.txt 2> /dev/null)
  local sz=$(grep -m 1 '尺寸（门' ${P}_verdict.txt 2> /dev/null | sed 's/^ *//') note=""
  [ -n "$sz" ] && note=" [$sz]"
  GATE_GAP=0
  case "$v" in
    PASS)
      q=$($PY -c "import sys; sys.path.insert(0, 'scripts/unified'); import accept_judge as aj; g, t = aj.quality_gap('$R', '${P}_stairs.txt'); print(f'{g:.3f} {t}')" 2> /dev/null)
      GATE_GAP=${q%% *}
      if [ -z "$q" ] || awk -v g="${GATE_GAP:-1}" 'BEGIN { exit !(g > 0) }'; then
        GATE_OUT=hard
        log "STUDENT $kind S2 GATE $k: HARDPASS quality gap ${GATE_GAP:-?} (${q#* }) -- kept, training goes on${note}"
        tl $run s2_gate $k verdict=HARDPASS end=$e gap=${GATE_GAP:-?} "quality=${q#* }"
      else
        GATE_OUT=pass
        log "STUDENT $kind S2 GATE $k: PASS, the quality no worse than delivered (${q#* })${note}"
        tl $run s2_gate $k verdict=PASS end=$e gap=$GATE_GAP "quality=${q#* }"
      fi ;;
    FAIL*) GATE_OUT=fail; log "STUDENT $kind S2 GATE $k: $(echo "$v" | cut -c1-300)${note}"
           tl $run s2_gate $k verdict=FAIL end=$e failfast=$([ -f ${P}_failfast.txt ] && echo 1 || echo 0) "why=$(echo "$v" | cut -c1-300)" ;;
    *) GATE_OUT=fault; log "STUDENT $kind S2 GATE $k: no verdict (${v:-none}; a tool fault, training goes on)${note}"
       tl $run s2_gate $k verdict=FAULT end=$e "why=${v:-none}" ;;
  esac
}
one() {   # KIND: stage 1, stage 2, then the acceptance; returns accept.sh's exit (4: a tool fault in training)
  local kind=$1 K=${1^} before run ck n1 n2
  if [ $SMOKE = 1 ]; then n1=${kind}1_smoke n2=${kind}2_smoke
  elif [ $PROBE = 1 ]; then n1=${kind}1_probe_s$SEED n2=${kind}2_probe_s$SEED
  else n1=${kind}1_t${TSEED}_s$SEED n2=${kind}2_t${TSEED}_s$SEED; fi
  local progress=$([ $PROBE != 1 ] && [ $SMOKE != 1 ] && echo 1 || echo 0) pid stopk=""
  before=$(newest ${r}_u_student_$kind $n1)
  $PY scripts/train.py Unitree-$R-U-Student-$K --env.scene.num-envs=$ENVS --agent.run-name=$n1 --agent.seed=$SEED \
    --agent.max-iterations=$S1 --agent.teacher-checkpoint=$T $SAVE > logs/jobs/unified_${r}_$n1.log 2>&1 &
  pid=$!
  # stage 1 ends when its behavior loss levels off (2026-11-13, go2-depth; scripts/unified/s1_plateau.py): S1 is the cap
  if [ $progress = 1 ]; then
    local v k1 run1
    while kill -0 $pid 2> /dev/null; do
      sleep ${S1_POLL_S:-60}
      v=$($PY scripts/unified/s1_plateau.py logs/jobs/unified_${r}_$n1.log $F_S1_MIN_ITERS $F_S1_PLATEAU_WINDOW \
        $F_S1_PLATEAU_DROP $F_S1_PLATEAU_HOLD $F_S1_CHECK_EVERY_IT 2> /dev/null)
      [[ "$v" == STOP* ]] || continue
      k1=$(echo $v | cut -d' ' -f2) run1=$(newest ${r}_u_student_$kind $n1)
      for _ in $(seq 60); do [ -f $run1/model_$k1.pt ] && break; sleep 10; done
      [ -f $run1/model_$k1.pt ] || continue
      stopk=$k1
      tl $run1 s1_end $k1 why=plateau "drops=$(echo $v | cut -d' ' -f3- | tr -d '()')" min=$F_S1_MIN_ITERS \
        window=$F_S1_PLATEAU_WINDOW drop=$F_S1_PLATEAU_DROP hold=$F_S1_PLATEAU_HOLD
      log "STUDENT $kind S1 END at $k1: plateau -- the behavior loss's last $F_S1_PLATEAU_WINDOW-iteration mean fell less than $F_S1_PLATEAU_DROP at $F_S1_PLATEAU_HOLD checks in a row ($(echo $v | cut -d' ' -f3-)); stage 2 from model_$k1"
      stopw $pid
      break
    done
  fi
  wait $pid
  local rc=$?
  run=$(newest ${r}_u_student_$kind $n1)
  if [ -n "$stopk" ]; then
    ck=$run/model_$stopk.pt rc=0
  else
    # (by number: sort -t_ -k2 -n read the path's "rsl_rl" as the key, so model_900 sorted after model_3999)
    ck=$(ls $run/model_*.pt 2> /dev/null | sort -V | tail -n 1)
    if [ $progress = 1 ] && [ $rc = 0 ] && [ -n "$ck" ]; then
      log "STUDENT $kind S1 END at $(basename $ck .pt | cut -d_ -f2): cap ($S1 iterations, no plateau)"
      tl $run s1_end $(basename $ck .pt | cut -d_ -f2) why=cap
    fi
  fi
  if [ $rc != 0 ] || [ "$run" = "$before" ] || [ -z "$ck" ]; then
    log "TOOL FAULT: $kind stage 1 ended with exit $rc, run ${run:-none} ($(grep -E 'Error' logs/jobs/unified_${r}_$n1.log | tail -n 1 | cut -c1-160))"
    return 4
  fi
  log "$kind stage 1 done: $ck"
  local attempt=0
  while :; do
  attempt=$((attempt + 1))
    before=$(newest ${r}_u_student_${kind}_rl $n2)
    $PY scripts/train.py Unitree-$R-U-Student-$K-RL --env.scene.num-envs=$ENVS --agent.run-name=$n2 --agent.seed=$SEED \
      --agent.max-iterations=$S2 --agent.teacher-checkpoint=$T --agent.student-checkpoint=$ck $SAVE2 \
      > logs/jobs/unified_${r}_$n2.log 2>&1 &
    pid=$!
    # stage 2's gates (2026-11-13, go2-depth): every F_S2_GATE_EVERY_IT iterations the students' own tier 2 on the
    # average of the newest F_S2_GATE_AVG_N checkpoints (accept.sh BASELINE=1: no pick, no tinynav, no delivery), one gate
    # at a time on the card (a point that comes while one runs, or without the memory, is skipped). PASS with the quality
    # readings no worse than delivered: stage 2 stops there and that candidate goes on (its tier 2 reused); PASS worse:
    # kept, training goes on; at the cap the kept one with the smallest quality gap, else the last average as before
    local run2="" gpid="" gk="" ge="" pick_end="" HP="" Kg last n d gfirst=1
    declare -A GDONE=()
    if [ $progress = 1 ] && [ "${F_S2_GATE_EVERY_IT:-0}" -gt 0 ]; then
      while kill -0 $pid 2> /dev/null || [ -n "$gpid" ]; do
        sleep ${S2_POLL_S:-60}
        if [ -z "$run2" ]; then
          d=$(newest ${r}_u_student_${kind}_rl $n2); [ -n "$d" ] && [ "$d" != "$before" ] && run2=$d
          [ -n "$run2" ] || continue
        fi
        if [ -n "$gpid" ] && ! kill -0 $gpid 2> /dev/null; then   # a gate ended: its verdict
          wait $gpid 2> /dev/null
          if [ $? = 75 ]; then GATE_OUT=skip; else s2_gate_read $kind $run2 $gk $ge; gfirst=0; fi
          gpid=""
          case $GATE_OUT in
            pass) pick_end=$ge
                  log "STUDENT $kind S2 PICK a${F_S2_GATE_AVG_N}_$ge (gate at $gk): stage 2 stopped"
                  tl $run2 s2_pick $gk end=$ge why=gate
                  kill -0 $pid 2> /dev/null && stopw $pid
                  break ;;
            hard) HP="$HP$GATE_GAP $ge
" ;;
          esac
        fi
        kill -0 $pid 2> /dev/null || continue   # (training ended: only the running gate is waited for)
        last=$(ls $run2/model_*.pt 2> /dev/null | grep -oE 'model_[0-9]+\.pt$' | grep -oE '[0-9]+' | sort -n | tail -n 1)
        [ -n "$last" ] || continue
        for Kg in $(seq $F_S2_GATE_EVERY_IT $F_S2_GATE_EVERY_IT $last); do
          [ -n "${GDONE[$Kg]:-}" ] && continue
          GDONE[$Kg]=1
          n=$(ls $run2/model_*.pt | grep -oE 'model_[0-9]+\.pt$' | grep -oE '[0-9]+' | awk -v k=$Kg '$1 <= k' | wc -l)
          if [ -n "$gpid" ]; then log "STUDENT $kind S2 GATE $Kg: skipped (the gate at $gk still runs)"
            tl $run2 s2_gate $Kg verdict=skipped "why=the gate at $gk still runs"; continue; fi
          if [ $n -lt $F_S2_GATE_AVG_N ]; then log "STUDENT $kind S2 GATE $Kg: skipped ($n checkpoints, the average takes $F_S2_GATE_AVG_N)"
            tl $run2 s2_gate $Kg verdict=skipped "why=$n checkpoints"; continue; fi
          ge=$(ls $run2/model_*.pt | grep -oE 'model_[0-9]+\.pt$' | grep -oE '[0-9]+' | sort -n | awk -v k=$Kg '$1 <= k' | tail -n 1)
          # a student's first gate waits for the card (go2-depth: each student one full reading early); later ones skip
          s2_gate_run $([ $gfirst = 1 ] && printf %s -x || printf %s -n) $kind $run2 $Kg ${F_S2_GATE_FAILFAST:-0} &   # (echo -n prints nothing)
          gpid=$! gk=$Kg
          log "STUDENT $kind S2 GATE $Kg: started on a${F_S2_GATE_AVG_N}_$ge$([ $gfirst = 1 ] && echo ' (its first: waits for the card if another gate runs)')"
          break
        done
      done
    fi
    wait $pid
    rc=$?
    [ -n "$pick_end" ] && rc=0   # (stopped by its gate)
    run=$(newest ${r}_u_student_${kind}_rl $n2)
    if [ $rc != 0 ] || [ -z "$run" ] || [ "$run" = "$before" ]; then
      if [ $progress = 1 ] && [ $attempt -le ${F_S2_RETRIES:-1} ]; then
        # stage 2 is never resumed (2026-11-13, go2-depth): student_ppo's resume restores neither the BC progress (each
        # learn() decays from the configured bc_coef over its own iterations) nor num_updates (the critic warm-up again)
        # -- a stage 2 that ends in a fault starts over from the stage-1 end, a new run; the failed one's log is kept
        log "STUDENT $kind S2 FAULT (exit $rc, attempt $attempt, run ${run2:-none}): stage 2 again from $ck -- a new run, never a resume ($(grep -E 'Error' logs/jobs/unified_${r}_$n2.log | tail -n 1 | cut -c1-160))"
        [ -n "$run2" ] && tl $run2 s2_fault 0 attempt=$attempt rc=$rc
        [ -n "$gpid" ] && wait $gpid 2> /dev/null
        mv -f logs/jobs/unified_${r}_$n2.log logs/jobs/unified_${r}_$n2.fault$attempt.log
        continue
      fi
      log "TOOL FAULT: $kind stage 2 ended with exit $rc ($(grep -E 'Error' logs/jobs/unified_${r}_$n2.log | tail -n 1 | cut -c1-160))"
      return 4
    fi
    break
  done
  log "$kind stage 2 done: $run"
  if [ $PROBE = 1 ]; then
    log "$kind: probe -- no acceptance (read $run with BASELINE=1 CANDIDATES=first TN_GATE=skip jobs/unified/accept.sh)"
    return 0
  fi
  local ends=""
  if [ $progress = 1 ] && [ "${F_S2_GATE_EVERY_IT:-0}" -gt 0 ] && [ -z "$pick_end" ]; then
    # the cap: the last average is one of the candidates compared (go2-depth) -- its tier 2 (which the acceptance runs
    # anyway) as a gate when no gate judged it yet; the other kind may still be training beside it
    last=$(ls $run/model_*.pt 2> /dev/null | grep -oE 'model_[0-9]+\.pt$' | grep -oE '[0-9]+' | sort -n | tail -n 1)
    # (a gate that stopped early on the first group -- its _failfast.txt -- is no judgment of the whole table: run again)
    local cP=logs/checks/u_${r}_${kind}_$(basename $run)_a${F_S2_GATE_AVG_N}_${last}
    if [ -n "$last" ] && { [ ! -s ${cP}_verdict.txt ] || [ -f ${cP}_failfast.txt ]; }; then
      log "STUDENT $kind S2 GATE $last (the cap's last average): started"
      s2_gate_run -x $kind $run $last
      s2_gate_read $kind $run $last $last
      case $GATE_OUT in
        pass) pick_end=$last; log "STUDENT $kind S2 PICK a${F_S2_GATE_AVG_N}_$last (the cap's last average)"
              tl $run s2_pick $last end=$last why=cap_last ;;
        hard) HP="$HP$GATE_GAP $last
" ;;
      esac
    fi
  fi
  if [ -n "$pick_end" ]; then
    ends=$pick_end
  elif [ -n "$HP" ]; then   # the cap: the gate-passed one with the smallest quality gap
    ends=$(echo "$HP" | grep -v '^$' | sort -g | head -n 1 | cut -d' ' -f2)
    tl $run s2_pick $ends end=$ends why=cap_smallest_gap
    log "STUDENT $kind S2 PICK a${F_S2_GATE_AVG_N}_$ends at the cap: hard items passed, quality gap $(echo "$HP" | grep -v '^$' | sort -g | head -n 1 | cut -d' ' -f1) (the smallest of the kept)"
  fi
  # the final acceptance in a formal run (2026-11-14, go2-depth: the other kind's student may still be training beside
  # it -- an unguarded acceptance, two tests at once, the drift's 13 x 768 robots at ~9-10 GB, could take that training
  # down): one test at a time, the drift in halves, each test waiting for its need (the gates' table, per kind) up to
  # F_FINAL_ACCEPT_WAIT_S, then a tool fault (not a fail: rerun); full size always (no ACCEPT_SMALL)
  local guard=()
  [ $progress = 1 ] && guard=(ACCEPT_JOBS=1 ACCEPT_DRIFT_SPLIT=1 ACCEPT_FREE_WAIT_S=$F_FINAL_ACCEPT_WAIT_S "PAIR_NEED_MB=$F_PAIR_NEED_MB"
    "ACCEPT_NEED_MB=$([ $kind = depth ] && echo "$F_S2_GATE_NEED_MB_DEPTH" || echo "$F_S2_GATE_NEED_MB")")
  env "${guard[@]}" REUSE_TIER2=$progress ACCEPT_SMOKE=$SMOKE bash jobs/unified/accept.sh $R $kind $run $ends >> $L 2>&1
}
KINDS="${KINDS:-blind depth}"
TRIED=" $TSEED " WHY=""
declare -A FINAL=()
# a previous run's student picks are not this run's (accept.sh pairs the depth and the blind pick it finds)
SFX=$([ $SMOKE = 1 ] && echo _smoke)
# (a probe makes no pick: the formal run's picks are not touched)
[ $PROBE = 1 ] || for kind in $KINDS; do rm -f logs/jobs/unified_${r}_$kind$SFX.pick logs/jobs/unified_${r}_$kind$SFX.pick.*; done
while true; do
  [ -f "$T" ] || { log "no teacher checkpoint $T"; exit 2; }
  T=$(cd $(dirname $T) && pwd)/$(basename $T)
  log "teacher $T (seed $TSEED, code $(cat REPO_COMMIT 2>/dev/null), formal.env $(md5sum jobs/unified/formal.env | cut -c1-8)); kinds $KINDS; stage 1 $S1 it, stage 2 $S2 it, $ENVS robots each"
  # the teacher's stops on flat before its students (jobs/unified/stop_check.sh; go2-depth's decision 11-12): a
  # recipe check -- a group over 1% falls stops here for the recipe to be fixed; smoke: 32 robots per group, not judged
  if [ "${STOP_CHECK:-1}" = 1 ]; then
    sc=$(STOP_ENVS=$([ $SMOKE = 1 ] && echo 32 || echo ${STOP_ENVS:-768}) bash jobs/unified/stop_check.sh $R $T); src=$?
    log "teacher stop check: $sc"
    if [ $src = 4 ] || { [ $src != 0 ] && [ $src != 1 ]; }; then
      log "TOOL FAULT: the teacher stop check (exit $src; not a fail)"; exit 4
    elif [ $src = 1 ] && [ $SMOKE != 1 ]; then
      log "STOP: needs a human -- the teacher $T falls on flat after a step release (jobs/unified/stop_check.sh, a recipe check; logs/checks/u_${r}_tstop.txt): its recipe is to be fixed and the teacher resumed; no student trained on it"
      exit 3
    fi
  fi
  declare -A PIDS=() RC=()
  for kind in $KINDS; do one $kind & PIDS[$kind]=$!; done
  for kind in $KINDS; do
    wait ${PIDS[$kind]}; RC[$kind]=$?
    FINAL[$kind]=${RC[$kind]}
  done
  log "done with teacher seed $TSEED ($(for kind in $KINDS; do echo -n "$kind exit ${RC[$kind]} "; done))"
  if [ $PROBE = 1 ]; then   # no acceptance ran: training's own exit is the outcome
    for kind in $KINDS; do [ "${RC[$kind]}" = 0 ] || { log "probe: $kind training ended with exit ${RC[$kind]}"; exit 4; }; done
    log "probe done: the stage-2 runs above, no pick, no delivery"
    exit 0
  fi
  # the markers accept.sh left, read the same way whichever kind's run ran the pair test (doc/redesign.md 2.2.1;
  # go2-depth's review 11-12): over every kind accepted so far, this round's and earlier rounds'
  for kind in "${!FINAL[@]}"; do
    [ "${FINAL[$kind]}" = 0 ] || continue
    pf=logs/jobs/unified_${r}_$kind$SFX.pick
    if [ $kind = depth ] && [ -f $pf.pairfail ]; then
      FINAL[$kind]=1; log "depth: the camera-failure pair test failed ($(head -c 160 logs/jobs/unified_${r}_pair$SFX.verdict 2> /dev/null)) -- counted as not accepted"
    elif [ $kind = depth ] && [ -f $pf.pairfault ]; then
      FINAL[$kind]=4; log "depth: the camera-failure pair test had a tool fault ($pf.pairfault; PAIR_ONLY=1 accept.sh reruns it)"
    elif [ ! -f $pf.delivered ] && ! { [ $kind = depth ] && [ -f $pf.pending ]; }; then
      FINAL[$kind]=4; log "$kind: accepted, but not delivered ($pf.delivered missing) -- a tool fault"
    fi
  done
  failed=""
  for kind in "${!FINAL[@]}"; do [ "${FINAL[$kind]}" = 1 ] && failed="$failed $kind"; done
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
  log "fallback: kinds$failed failed their acceptance on teacher seed $TSEED; resuming teacher seed $next (its latest checkpoint, $envs robots, cap ${TEACHER_CAP:-$F_FALLBACK_TEACHER_CAP})"
  TRIED="$TRIED$next "
  NUM_ENVS=$envs PARALLEL=0 RESUME_LATEST=1 ITERS_TOTAL=1 bash jobs/unified/teacher.sh $R $VARIANT "$next" ${TEACHER_CAP:-$F_FALLBACK_TEACHER_CAP}
  if [ ! -s $PICK ] || ! grep -q "seed $next" $PICK; then WHY="teacher seed $next did not pass its gate"; break; fi
  T=$(cut -d' ' -f1 $PICK) TSEED=$next KINDS=$failed
done
# a depth student still waiting for an accepted blind student to pair with is not delivered
if [ "${FINAL[depth]:-x}" = 0 ] && [ -f logs/jobs/unified_${r}_depth$SFX.pick.pending ]; then
  FINAL[depth]=1; WHY="${WHY:+$WHY; }depth: accepted, but no accepted blind student to pair it with"
fi
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
