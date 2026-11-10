#!/bin/bash
# fe_chain.sh R (depth_fe, 2026-10-31; user 10-30/31: depth on go2-home's newest blind models, started as soon as each
# is delivered; "训练过程中如果有checkpoint可以提前通过验收，就直接开始下一个阶段的训练"; the student warm-starts from the
# deployed student). Legged (feetedge) and wheeled deliveries (jobs/fe_robot.sh). Progress in logs/jobs/fe_<r>.txt,
# check lines in logs/checks/checks.out.
#  1. Widen the delivered blind model for the privileged scan (scripts/go2_depth/widen_scan.py) and check the widened
#     model as a teacher (= the blind policy on the scan teacher's map3: the bar the teacher has to keep).
#  2. Teacher: Unitree-<R>-Depth-<PH> (the delivery's phase) from it with the delivery's BLIND_PILOT
#     (BLIND_ANCHOR_COEFF=0), up to 3000 iterations. At +1000, +1500, ... +3000 the actor average of the last 8
#     checkpoints (jobs/avg_ckpt.py, go2-home's: 10-31 single checkpoints failed one item at random, averages passed)
#     goes through jobs/fe_check.sh teacher on seed 0. It is picked when it passes every judged goal, keeps map3 no worse
#     than the widened model (within MAP3_TOL), and its stairs speed ratios keep MARGIN to the 0.8-1.2 band (jobs/fe_margin.py; its
#     student lands a little below it). The first such average ends the teacher. If none has the margin by +3000, the
#     passing average with the widest margin is picked.
#  3. Student: Unitree-<R>-Depth-Student-Edge (real camera, occlusions, edge smoothing), 1024 envs, warm-started from
#     OCC with that teacher, lr 5e-4 -> 1e-4, up to 3000 iterations. At 1000, 1500, ... 2999 the student average of the
#     last 8 checkpoints (jobs/avg_student.py) goes through fe_check.sh student on seeds 0, 1, 2 (stopping at the first
#     failure); passing all three ends the student and delivers: logs/picks/<r>_{teacher,student},
#     logs/exports/<r>_fe_student_avg<k>.onnx (scripts/go2_depth/export_student.py).
#  Every launch waits for free GPU memory (12 GB, 3 minutes in a row).
set -u
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
R=$1 r=${1,,}
source jobs/fe_robot.sh $R || exit 2
# A fine-tune round (2026-11-03, wheeled stairs: pilot "wheelstep"): FE_PICKS (picks dir, so the delivered picks stay),
# FE_PILOT_EXTRA (",key,..." appended to the delivery's pilot), FE_CONS_INIT (BLIND_CONS_INIT instead of fe_robot.sh's:
# the multipliers where the picked teacher's run left them), FE_TEACHER_FROM (a teacher checkpoint model_<N>.pt to
# fine-tune instead of the widened delivery), FE_TEACHER_RUN (run name), FE_TEACHER_KS (checked iterations after the
# start), FE_TEACHER_PICK=last (the last passing average with the margin instead of the first; else the widest).
PK=${FE_PICKS:-logs/picks}
[ -n "${FE_CONS_INIT+x}" ] && { if [ -n "$FE_CONS_INIT" ]; then export BLIND_CONS_INIT=$FE_CONS_INIT; else unset BLIND_CONS_INIT; fi; }
TRUN=${FE_TEACHER_RUN:-fe_v1} TTAG=${FE_TEACHER_RUN:+_$FE_TEACHER_RUN}
TKS=${FE_TEACHER_KS:-1000 1500 2000 2500 3000}
# FE_MARGIN (2026-11-02): B2W students descend 0.04-0.05 slower than their teachers, so its teacher wants ~0.07.
MARGIN=${FE_MARGIN:-0.05}
# map3 "no worse than the widened model" within this many points of reached (1 robot of 64 in one of the 18 scenarios
# is 0.09; 10-31 the Go2W widened model read op 100.0 and its first teacher average 99.9).
MAP3_TOL=0.5
L=logs/jobs/fe_$r.txt C=logs/checks
mkdir -p logs/jobs $C $PK logs/exports
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
free_mib() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )); }
wait_free() { local n=0; until [ $n -ge 3 ]; do if [ $(free_mib) -ge $1 ]; then n=$((n + 1)); else n=0; fi; sleep 60; done; }
# A check that did not finish (2026-11-02, t1: an evaluation out of GPU memory left its items "评测中" and the chain
# took it for a FAIL) is run once more instead of judged.
incomplete() { [[ -z "$1" || "$1" == *评测中* || "$1" == *INVALID* || "$1" == *"judge error"* ]]; }
reached() { python3 -c "import re,sys; v=[float(x) for x in re.findall(r'reached=([0-9.]+)', open(sys.argv[1]).read())]; print(f'{100*sum(v)/len(v):.1f}' if v else 'nan')" $1; }
wait_ck() {  # RUN_DIR K PID: until model_K.pt is saved (0) or the training has ended without it (1)
  until [ -f $1/model_$2.pt ]; do kill -0 $3 2> /dev/null || { sleep 5; [ -f $1/model_$2.pt ]; return; }; sleep 30; done
  sleep 10
}
stop_train() {  # PID
  kill -INT $1 2> /dev/null; for _ in $(seq 60); do kill -0 $1 2> /dev/null || return 0; sleep 2; done; kill $1 2> /dev/null
}
new_run() {  # EXP NAME: the newest run dir
  ls -d logs/rsl_rl/$1/*_$2 2> /dev/null | sort | tail -n 1
}
last8() {  # RUN K: the last 8 checkpoints up to model_K (every 100; a run's final save is K = ...99)
  local k0=$(( $2 / 100 * 100 )) out=""
  for k in $(seq $((k0 - 700 + ($2 % 100 ? 100 : 0))) 100 $k0); do [ -f $1/model_$k.pt ] && out="$out $1/model_$k.pt"; done  # short runs: only those saved
  [ $(( $2 % 100 )) -ne 0 ] && out="$out $1/model_$2.pt"
  echo $out
}
PILOT=$(cat $PKG/blind_pilot.txt)${FE_PILOT_EXTRA:-}
export BLIND_PILOT=$PILOT BLIND_ANCHOR_COEFF=0
TD=logs/rsl_rl/${r}_depth_teacher SD=logs/rsl_rl/${r}_depth_student
INIT=$TD/blind_init_fe/model_$N.pt
export BLIND_ANCHOR_CHECKPOINT=$PWD/$INIT
log "$R chain: $PKG model_$N${FE_TEACHER_FROM:+, teacher fine-tuned from $FE_TEACHER_FROM (run $TRUN, checks at +${TKS// /,+}, pick ${FE_TEACHER_PICK:-first})}${FE_PILOT_EXTRA:+, pilot + $FE_PILOT_EXTRA}, picks $PK, Depth-$PH, teacher speed margin $MARGIN, phase start ${BLIND_PHASE_START_IT:-none}, constraints init ${BLIND_CONS_INIT:-1x}, student from $OCC"
# FE_AFTER_PID (2026-11-05, the new pipeline's scheduling): wait for that process (a training on this GPU) to end
# first; free memory alone let two students overlap and OOM.
if [ -n "${FE_AFTER_PID:-}" ]; then
  while kill -0 $FE_AFTER_PID 2> /dev/null; do sleep 30; done
  log "$R chain: pid $FE_AFTER_PID ended, going on"
fi

# 1. Widen and check the widened model.
if [ ! -f $INIT ]; then
  [ "$(md5sum < $PKG/model_$N.pt | cut -d' ' -f1)" = "$(grep -o '^[0-9a-f]*' $PKG/checkpoint.md5 | head -n 1)" ] || { log "$R: md5 of $PKG/model_$N.pt does not match checkpoint.md5"; exit 1; }
  mkdir -p $TD/blind_init_fe && cp -r $PKG/params $PKG/blind_pilot*.txt $PKG/checkpoint.md5 $TD/blind_init_fe/
  log "$R widen $PKG/model_$N.pt (pilot $PILOT)"
  $PY scripts/go2_depth/widen_scan.py --task Unitree-$R-Depth-$PH $PKG/model_$N.pt --out $INIT > logs/jobs/fe_${r}_widen.log 2>&1 \
    || { log "$R widen failed: $(tail -n 3 logs/jobs/fe_${r}_widen.log)"; exit 1; }
fi

# 2. Teacher.
if [ ! -f $PK/${r}_teacher ]; then
  T0=$N LRUN=blind_init_fe TITERS=3001
  if [ -n "${FE_TEACHER_FROM:-}" ]; then
    T0=$(basename $FE_TEACHER_FROM .pt | cut -d_ -f2) LRUN=${TRUN}_init TITERS=$(( ${TKS##* } + 1 ))
    mkdir -p $TD/$LRUN && cp $FE_TEACHER_FROM $TD/$LRUN/model_$T0.pt || { log "$R: cannot copy $FE_TEACHER_FROM"; exit 1; }
  fi
  wait_free 12000
  $PY scripts/train.py Unitree-$R-Depth-$PH --env.scene.num-envs=4096 --agent.run-name=$TRUN --agent.resume=True \
    --agent.load-run="^$LRUN\$" --agent.load-checkpoint="^model_$T0\.pt\$" --agent.max-iterations=$TITERS \
    > logs/jobs/fe_${r}_teacher$TTAG.log 2>&1 &
  TP=$!
  log "$R teacher started (pid $TP; from $LRUN/model_$T0.pt, $TITERS it, pilot $BLIND_PILOT, constraints init ${BLIND_CONS_INIT:-1x}; free $(free_mib) MiB before)"
  sleep 120
  RUN=$(new_run ${r}_depth_teacher $TRUN)
  log "$R teacher run $RUN"
  # one baseline per blind package and checkpoint (11-07: the B2 chains b2n1-b2n3 had compared their teachers with
  # fe_b2_init of 11-01, an older blind, whatever package they widened)
  B=$C/fe_${r}_init_$(basename $PKG)_$N
  [ -f ${B}_verdict.txt ] || bash jobs/fe_check.sh teacher $R $INIT 0 $B
  BN=$(reached ${B}_noop.txt) BO=$(reached ${B}_op.txt)
  log "$R widened blind (teacher map3): noop $BN op $BO speed up/down/margin $($PY jobs/fe_margin.py $R $B) | $(cat ${B}_verdict.txt)"
  pick="" best="" bestm=-1
  for dK in $TKS; do
    K=$((T0 + dK))
    wait_ck $RUN $K $TP || { log "$R teacher ended before model_$K ($(tail -n 1 logs/jobs/fe_${r}_teacher$TTAG.log))"; break; }
    A=$TD/${TRUN}_avg$K/model_$K.pt
    mkdir -p $(dirname $A)
    $PY jobs/avg_ckpt.py $A $(last8 $RUN $K) >> $L 2>&1 || { log "$R teacher average $K failed"; continue; }
    P=$C/fe_${r}_teacher${TTAG}_avg$K
    bash jobs/fe_check.sh teacher $R $A 0 $P
    if incomplete "$(cat ${P}_verdict.txt 2> /dev/null)"; then
      log "$R teacher avg$K: check incomplete ($(cat ${P}_verdict.txt 2> /dev/null)), checking again"
      bash jobs/fe_check.sh teacher $R $A 0 $P
    fi
    v=$(cat ${P}_verdict.txt) n=$(reached ${P}_noop.txt) o=$(reached ${P}_op.txt) sp=$($PY jobs/fe_margin.py $R $P)
    m=${sp##* }
    log "$R teacher avg$K: noop $n op $o speed up/down/margin $sp | $v"
    if [[ "$v" == PASS ]] && python3 -c "import sys; a=list(map(float,sys.argv[1:])); sys.exit(0 if a[0] >= a[2] - a[4] and a[1] >= a[3] - a[4] else 1)" $n $o $BN $BO $MAP3_TOL; then
      if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" $m $MARGIN; then
        # FE_TEACHER_TOPFLAT (2026-11-09, Go2W): also at most that share (%) of the top-landing wheel time lifted
        # (jobs/top_flat_v2.sh, 1024 robots): the students copy it (Go2W w27d 21300 26% -> its stage 1 / 2 students
        # 2.5-3.1 top-landing steps per climb in tinynav, bar 1.0; B2W tf_v3 17-18% -> 0.8).
        if [ -n "${FE_TEACHER_TOPFLAT:-}" ]; then
          tn=${r}_teacher${TTAG}_avg$K
          bash jobs/top_flat_v2.sh $R $A $tn "${FE_PILOT_EXTRA:-}" > /dev/null 2>&1
          tf=$(grep -a '^top_flat_eval' logs/checks/top_flat_$tn.log 2> /dev/null | tail -n 1 | sed -n 's/.*top_flat lifted \([0-9.]*\)%.*/\1/p')
          if [ -z "$tf" ] || ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" $tf $FE_TEACHER_TOPFLAT; then
            log "$R teacher avg$K passes with the margin but lifts ${tf:-?}% of its top-landing time (> $FE_TEACHER_TOPFLAT): training goes on"
            continue
          fi
          log "$R teacher avg$K top landing lifted $tf% (<= $FE_TEACHER_TOPFLAT)"
        fi
        pick=$A
        [ "${FE_TEACHER_PICK:-first}" = last ] || break
        log "$R teacher avg$K passes with the margin; FE_TEACHER_PICK=last: training goes on"
        continue
      fi
      if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" $m $bestm; then best=$A bestm=$m; fi
      log "$R teacher avg$K passes but its speed margin $m < $MARGIN: training goes on"
    fi
  done
  stop_train $TP
  if [ -z "$pick" ] && [ -n "$best" ]; then
    pick=$best
    log "$R no teacher average with speed margin $MARGIN: taking the passing one with the widest ($bestm)"
  fi
  if [ -z "$pick" ]; then
    log "$R NO TEACHER PASSED: chain stops here (run $RUN kept)"
    exit 1
  fi
  echo "$pick" > $PK/${r}_teacher
  log "$R TEACHER PICK $pick (md5 $(md5sum < $pick | cut -c1-8)); training stopped"
fi
TEACHER=$PWD/$(cat $PK/${r}_teacher)

# 3. Student. A second round (2026-11-01) sets FE_STUDENT_FROM (the warm start instead of OCC), FE_STUDENT_ITERS,
# FE_STUDENT_LR / FE_STUDENT_LR_FINAL, FE_STUDENT_RUN (run name), DEPTH_STUDENT_HARD (more of the hard commands in the
# student's training, depth_env_cfg.apply_depth_student) and FE_CHECK_AFTER_TRAIN=1 (checks only once no training runs
# on this GPU: two students held ~22 of 24 GB on t2 and a depth student's evaluations did not fit beside them).
SFROM=${FE_STUDENT_FROM:-$OCC} SITERS=${FE_STUDENT_ITERS:-3000} SRUN=${FE_STUDENT_RUN:-fe_edge_v1}
SLR=${FE_STUDENT_LR:-5e-4} SLRF=${FE_STUDENT_LR_FINAL:-1e-4}
# FE_STUDENT_ARGS (2026-11-02): more train.py arguments for the student (e.g. --agent.algorithm.bc-jump-weight=8).
# FE_STUDENT_TASK (2026-11-02): the training task, e.g. Unitree-B2-Depth-Student-Edge-RL (the PPO fine-tune of a
# distilled student, rl/student_ppo.py; same arguments). The checks and the export keep -Student-Edge.
STASK=${FE_STUDENT_TASK:-Unitree-$R-Depth-Student-Edge}
# DEPTH_STUDENT_HARD goes to the student's training only: student_eval.py builds its -RealEdge map3 task as a training
# config (play=False), where the switch would change the evaluated command mix.
SHARD=${DEPTH_STUDENT_HARD:-}; unset DEPTH_STUDENT_HARD
# DEPTH_STUDENT_DR_EDGE (2026-11-02: more of the student's robots at the ends of the payload / CoM / friction ranges)
# likewise.
SDRE=${DEPTH_STUDENT_DR_EDGE:-}; unset DEPTH_STUDENT_DR_EDGE
# DEPTH_STUDENT_FRICTION=<lo>,<hi> (2026-11-02: the student's foot friction drawn uniform over that range) likewise.
SFRIC=${DEPTH_STUDENT_FRICTION:-}; unset DEPTH_STUDENT_FRICTION
# DEPTH_STUDENT_SPIN_LEFT=<share> (spin-ups to the left, B2) and DEPTH_STUDENT_STAND_LONG=1 (stands 4-8 s, As2W),
# 2026-11-02, likewise.
SSPIN=${DEPTH_STUDENT_SPIN_LEFT:-}; unset DEPTH_STUDENT_SPIN_LEFT
SSTAND=${DEPTH_STUDENT_STAND_LONG:-}; unset DEPTH_STUDENT_STAND_LONG
[ -f $SFROM ] || { log "$R: no warm-start student $SFROM"; exit 1; }
# FE_STUDENT_FREE (MiB, 2026-11-04): B2W's hard-command student OOM'd twice at start with 14.3 GB free beside a teacher
# (it holds 14.4 GB); two students do not fit one 24 GB box.
wait_free ${FE_STUDENT_FREE:-12000}  # a 1024-env depth student holds ~11.1 GB (Go2W, 11-01); 14000 never came beside a teacher
env ${SHARD:+DEPTH_STUDENT_HARD=$SHARD} ${SDRE:+DEPTH_STUDENT_DR_EDGE=$SDRE} ${SFRIC:+DEPTH_STUDENT_FRICTION=$SFRIC} \
  ${SSPIN:+DEPTH_STUDENT_SPIN_LEFT=$SSPIN} ${SSTAND:+DEPTH_STUDENT_STAND_LONG=$SSTAND} $PY scripts/train.py $STASK --env.scene.num-envs=1024 --agent.max-iterations=$SITERS \
  --agent.run-name=$SRUN --agent.teacher-checkpoint=$TEACHER --agent.student-checkpoint=$SFROM \
  --agent.algorithm.learning-rate=$SLR --agent.learning-rate-final=$SLRF ${FE_STUDENT_ARGS:-} > logs/jobs/fe_${r}_student_$SRUN.log 2>&1 &
SP=$!
log "$R student started (pid $SP; $STASK, run name $SRUN, $SITERS it, lr $SLR -> $SLRF, hard commands ${SHARD:-off}, DR edges ${SDRE:-off}, foot friction ${SFRIC:-as the task}, spin left ${SSPIN:-0.5}, long stands ${SSTAND:-off}, args ${FE_STUDENT_ARGS:-none}; teacher $TEACHER; from $SFROM; free $(free_mib) MiB before)"
sleep 120
RUN=$(new_run ${r}_depth_student $SRUN)
log "$R student run $RUN"
# This robot's depth trainings only (2026-11-02: dance's G1 trainings share the GPUs and run scripts/train.py too,
# and a B2W teacher beside an As2W student must not hold its checks for 10 h).
trainings() { ps -eo args | grep -cE "^[^ ]*python[^ ]* scripts/train[.]py Unitree-$R-Depth-"; }
pick=""
# FE_STUDENT_CHECK=0 (2026-11-05, the new pipeline's two-stage student): stage one (distillation) is not checked;
# once its training ends, the last 8 checkpoints are averaged into $SD/${SRUN}_avg<last> and named in
# $PK/${r}_student_stage1; stage two (PPO+BC, FE_STUDENT_FROM = that average) runs as its own chain and is checked.
if [ "${FE_STUDENT_CHECK:-1}" = 0 ]; then
  K=$((SITERS - 1))
  wait_ck $RUN $K $SP || { log "$R stage-1 student ended before model_$K ($(tail -n 1 logs/jobs/fe_${r}_student_$SRUN.log))"; exit 1; }
  while kill -0 $SP 2> /dev/null; do sleep 10; done
  A=$SD/${SRUN}_avg$K/model_$K.pt
  mkdir -p $(dirname $A)
  $PY jobs/avg_student.py $A $(last8 $RUN $K) >> $L 2>&1 || { log "$R stage-1 student average $K failed"; exit 1; }
  echo "$A" > $PK/${r}_student_stage1
  log "$R STAGE-1 STUDENT $A (no checks; FE_STUDENT_CHECK=0)"
  exit 0
fi
# FE_STUDENT_KS (2026-11-05): the checked iterations (default every 500 from 1000, and the last).
KLIST=${FE_STUDENT_KS:-$(seq 1000 500 $((SITERS - 1))) $((SITERS - 1))}
for K in $KLIST; do
  wait_ck $RUN $K $SP || { log "$R student ended before model_$K ($(tail -n 1 logs/jobs/fe_${r}_student_$SRUN.log))"; break; }
  if [ "${FE_CHECK_AFTER_TRAIN:-0}" = 1 ]; then
    until [ "$(trainings)" = 0 ]; do sleep 60; done
  fi
  A=$SD/${SRUN}_avg$K/model_$K.pt
  mkdir -p $(dirname $A)
  $PY jobs/avg_student.py $A $(last8 $RUN $K) >> $L 2>&1 || { log "$R student average $K failed"; continue; }
  ok=1
  for SEED in 0 1 2; do
    P=$C/fe_${r}_student_${SRUN#fe_edge_}_avg${K}_s$SEED
    bash jobs/fe_check.sh student $R $A $SEED $P
    if incomplete "$(cat ${P}_verdict.txt 2> /dev/null)"; then
      log "$R student avg$K s$SEED: check incomplete ($(cat ${P}_verdict.txt 2> /dev/null)), checking again"
      bash jobs/fe_check.sh student $R $A $SEED $P
    fi
    v=$(cat ${P}_verdict.txt)
    log "$R student avg$K s$SEED: noop $(reached ${P}_noop.txt) op $(reached ${P}_op.txt) edge noop $(reached ${P}_noopedge.txt) op $(reached ${P}_opedge.txt) speed up/down/margin $($PY jobs/fe_margin.py $R $P) | $v"
    [[ "$v" == PASS ]] || { ok=0; break; }
  done
  # FE_TN_GATE=1 (2026-11-05, the new pipeline's early stop: the user, "如果提前通过验收，就不要继续训练了"): the 3 seeds
  # passed; the tinynav acceptance (stairs gate, the user's stairs requirements, the payload limit) runs on the
  # workstation (scratchpad check/tn_gate_watch.sh): request logs/accept/<run>_<K>.req, wait (training goes on) for
  # its .verdict up to FE_TN_WAIT s (default 5400); PASS picks, anything else checks the next K.
  if [ $ok = 1 ] && [ "${FE_TN_GATE:-0}" = 1 ]; then
    mkdir -p logs/accept
    Q=logs/accept/${SRUN}_$K
    echo "$R $A $FE_PILOT_EXTRA" > $Q.req
    log "$R student avg$K: mjlab 3 seeds pass, waiting for the tinynav acceptance ($Q.verdict)"
    w=0
    until [ -f $Q.verdict ] || [ $w -ge ${FE_TN_WAIT:-5400} ]; do sleep 30; w=$((w + 30)); done
    v=$(cat $Q.verdict 2> /dev/null || echo "no verdict in ${FE_TN_WAIT:-5400} s")
    log "$R student avg$K tinynav: $v"
    [[ "$v" == PASS* ]] || ok=0
  fi
  if [ $ok = 1 ]; then pick=$A; break; fi
done
stop_train $SP
if [ -z "$pick" ]; then
  log "$R NO STUDENT PASSED: chain stops here (run $RUN kept)"
  exit 1
fi
echo "$pick" > $PK/${r}_student
K=$(basename $pick .pt | cut -d_ -f2)
X=logs/exports/${r}_fe_student_${SRUN#fe_edge_}_avg$K.onnx
$PY scripts/go2_depth/export_student.py --task Unitree-$R-Depth-Student-Edge --checkpoint $pick --out $X > logs/jobs/fe_${r}_export.log 2>&1 \
  && log "$R STUDENT PICK $pick (seeds 0 1 2 pass); exported $X (md5 $(md5sum < $X | cut -c1-8))" \
  || log "$R STUDENT PICK $pick; export FAILED: $(tail -n 3 logs/jobs/fe_${r}_export.log)"
echo "$(date '+%m-%d %T') $R fe round done" >> $C/checks.out
