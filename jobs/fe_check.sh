#!/bin/bash
# fe_check.sh KIND R CK SEED PREFIX (depth_fe, 2026-10-31; wheeled too): go2-home's delivery check (final_seeds.sh one(): the
# stairs-phase waves, judged by fix/mx_pass.py = jobs/fe_pass.py) for a depth teacher or student checkpoint CK under
# env seed SEED. KIND teacher: the Depth-Scratch / Depth-Flat-Scratch tasks, map3 as a scan teacher reads it
# (--teacher-scan). KIND student: through jobs/student_eval.py on Depth-Student-Stairs / -Flat (clean depth), map3 as
# built, plus map3 with the real camera's errors and edge smoothing (-RealEdge), which must meet the same stairs bars
# (with operator >= 95%, without >= 90%). Writes PREFIX_<kind>.txt, PREFIX_verdict.txt (PASS / FAIL: ...) and one line
# in logs/checks/checks.out. One check at a time per GPU (flock FE_LOCK; on t2 go2-home's check.lock, shared with
# their checks); inside, one evaluation at a time while two or more trainings run here (go2-home's ser: under CUDA MPS
# one faulting process takes every process on the GPU down), and always one at a time for a student.
set -u
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
export PATH=$(dirname $PY):$PATH
KIND=$1 R=$2 CK=$3 SEED=$4 P=$5
: ${BLIND_PILOT:?fe_check.sh needs BLIND_PILOT (the delivery blind_pilot.txt): without it the tasks build another recipe}
S=scripts/blind/eval O=logs/checks
mkdir -p $O $(dirname $P)
if [ $KIND = teacher ]; then
  W="$PY" TE=Unitree-$R-Depth-Scratch TF=Unitree-$R-Depth-Flat-Scratch SCAN=--teacher-scan
elif [ $KIND = blind ]; then
  # KIND blind (2026-11-05, the new pipeline's blind stage: the user, "先训练轮式的盲狗"): go2-home's own eval tasks
  W="$PY" TE=Unitree-$R-Blind-Scratch TF=Unitree-$R-Blind-Flat-Scratch SCAN=""
else
  W="$PY jobs/student_eval.py" TE=Unitree-$R-Depth-Student-Stairs TF=Unitree-$R-Depth-Student-Flat SCAN=""
fi
RS="$W jobs/seed_run.py $SEED"
# Wheeled flat tasks have no foot_gait reward: the clock comes from the training task (go2-home's final_seeds.sh:
# --clock-task Unitree-<R>-Blind-Teacher; the clock observation is the same).
CLOCK=""
[[ $R == *W ]] && CLOCK="--clock-task Unitree-$R-$([ $KIND = blind ] && echo Blind || echo Depth)-Teacher"
# B2's drift spins as deployed (user 2026-11-02): its controller brings a turn in at 3 rad/s² (config max_yaw_accel),
# and every B2 student's falls were stand -> +1.5 rad/s steps (go2-home: 0 falls on 3 seeds with the ramp).
# The B2W's and As2-W's too since their controllers ramp the same way (2026-11-02, jetson_ctrl b2w / as2w
# State_RLBase + config.yaml).
DRIFTA=""
# and the linear part too (same day, go2-home: a sideways step start, never seen on these controllers).
# Every robot since 2026-11-10 (the user: "改成所有机型都按这个加速度给"): commands come in at 2 m/s² / 3 rad/s²,
# as tinynav-sim's sim/accel.py and the B2 / B2-W / As2-W controllers bring the stick in.
DRIFTA="--yaw-accel 3.0 --lin-accel 2.0"
# and its stop test's start as deployed (user 2026-11-02): the command comes in at max_accel 2.0 m/s² / 3 rad/s², the
# release stays a step (the brake key); go2-home: B2 students' back-1.0 stop falls came from the stand -> -1.0 step.
STANDA=""
STANDA="--lin-accel 2.0 --yaw-accel 3.0"   # every robot (2026-11-10, as DRIFTA)
# and the flat acceptance matrix's starts (same day: B2W s1 fell spinning up to 1.5 from rest there).
AMA=$STANDA
[ -f "$CK" ] || { echo "$(date '+%m-%d %T') $KIND $R $CK s$SEED: no checkpoint" >> $O/checks.out; exit 1; }
# Remote mode (2026-11-01; logs/checks/REMOTE exists): this GPU has no room for the evaluations (t2: two trainings
# held ~21 GB and a depth student's map3 shards ran out of memory), so the check runs on another box. This writes
# logs/checks/requests/<name>.req (the arguments and the recipe's environment) and waits for PREFIX_verdict.txt; the
# workstation's relay (scratchpad wh2_port/relay.sh) copies the checkpoint over, runs this script there and copies
# the results back.
# REMOTE may list robots, one per line (only those are relayed); empty = every robot.
if [ -f $O/REMOTE ] && [ -z "${FE_NO_REMOTE:-}" ] && { [ ! -s $O/REMOTE ] || grep -qx "$R" $O/REMOTE; }; then
  mkdir -p $O/requests
  q=$O/requests/$(basename $P).req
  rm -f ${P}_verdict.txt
  { printf 'KIND=%q R=%q CK=%q SEED=%q P=%q\n' "$KIND" "$R" "$CK" "$SEED" "$P"
    for v in BLIND_PILOT BLIND_PARAMS BLIND_ANCHOR_COEFF BLIND_ANCHOR_CHECKPOINT BLIND_PHASE_START_IT BLIND_CONS_INIT; do
      if [ -n "${!v+x}" ]; then printf 'export %s=%q\n' $v "${!v}"; fi
    done; } > $q.new && mv $q.new $q
  echo "$(date '+%m-%d %T') $KIND $R $CK s$SEED: sent to the relay ($q)" >> $O/checks.out
  until [ -f ${P}_verdict.txt ]; do sleep 30; done
  exit 0
fi
mean_reached() {   # "-" when the run did not happen (the flat smoke has no map3)
  [ -f "$1" ] || { echo -; return; }
  python3 -c "import re,sys; v=[float(x) for x in re.findall(r'reached=([0-9.]+)', open(sys.argv[1]).read())]; print(f'{100*sum(v)/len(v):.1f}' if v else 'nan')" $1
}
trainings() { ps -eo args | grep -cE '^[^ ]*python[^ ]* scripts/train[.]py '; }
ser() { if [ $KIND = student ] || [ "$(trainings)" -ge 2 ]; then wait; pids=""; fi; }
# map3 runs (3 shards of 64 robots, ~0.6-1.5 GB each) go 4 at once on a GPU without trainings, 2 at once beside one
# (2026-10-31: the student's 4 map3 runs one after another took 21 of a check's 28 minutes). Under CUDA MPS (FE_MPS=1,
# t2: one faulting process takes every process on the GPU down) one at a time beside any training: 4 student runs at
# once held ~19 GB on t2 and go2-home's B2 polish could not start (12:04 10-31).
nmap=0
ser_map3() {
  nmap=$((nmap + 1))
  local t=$(trainings)
  # FE_SERIAL=1 (a borrowed GPU, e.g. t1 through the relay): always one at a time, whatever runs there.
  if [ "${FE_SERIAL:-0}" = 1 ] || [ $t -ge 2 ] || { [ $t -eq 1 ] && { [ "${FE_MPS:-0}" = 1 ] || [ $((nmap % 2)) -eq 0 ]; }; }; then wait; pids=""; fi
}
# One flat evaluation (2026-11-02): out of GPU memory on a shared GPU (t1 11:43 11-02: the B2 student's drift_all
# died beside its training and dance's, and its missing results judged "评测中" = FAIL) it waits until FE_ROOM_MIB
# are free for 2 minutes and runs again, three times at most.
free_mib() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )); }
wait_room() { local n=0; until [ $n -ge 2 ]; do if [ $(free_mib) -ge $1 ]; then n=$((n + 1)); else n=0; fi; sleep 60; done; }
evalrun() {  # OUT CMD...
  local out=$1 attempt; shift
  for attempt in 1 2 3; do
    "$@" > $out 2>&1 && return 0
    grep -q "out of memory" $out || return 1
    echo "$(date '+%m-%d %T') $KIND $R $CK s$SEED: $(basename $out) out of GPU memory (attempt $attempt), waiting for ${FE_ROOM_MIB:-12000} MiB" >> $O/checks.out
    wait_room ${FE_ROOM_MIB:-12000}
  done
  return 1
}
# CHAIN_SMOKE=1 (2026-11-11, jobs/smoke_chain.sh): a map3 run without the operator (4 robots, one shard) and the flat
# matrix (one robot per command) start and finish on this checkpoint; the verdict is PASS, or a judge error when one of
# them broke (blind_chain / fe_chain count that as a tool fault).
SPART=${FE_CHECK_PART:-all}; [ "${FE_CHECK_FLAT:-0}" = 1 ] && SPART=flat
if [ "${CHAIN_SMOKE:-0}" = 1 ]; then
  if { [ $SPART = flat ] || $W $S/map3_par.py --out ${P}_noop.txt --shards 1 -- --task $TE $CK --set flights --no-operator --down-vx 0.5 --num-envs 4 \
       --time-scale 1.6 --seed $SEED $SCAN > ${P}_noop.log 2>&1; } \
     && { [ $SPART = stairs ] || $RS $S/accept_matrix.py --task $TF --envs-per-cmd 1 $CK $AMA > ${P}_am.txt 2>&1; }; then v=PASS
  else v="FAIL: judge error: smoke evaluation ($(tail -n 1 ${P}_noop.log ${P}_am.txt 2> /dev/null | tr '\n' ' ' | cut -c1-200))"; fi
  echo "$v" > ${P}_verdict.txt
  echo "$(date '+%m-%d %T') SMOKE $KIND $R $CK s$SEED: noop $(mean_reached ${P}_noop.txt) | $v" >> $O/checks.out
  exit 0
fi
exec 9>> ${FE_LOCK:-$O/check.lock}
flock 9
t0=$(date +%s)
pids=""
map3() {  # OUT TASK OPERATOR(op|noop)
  local extra=""
  [ $3 = noop ] && extra="--no-operator --down-vx 0.5"
  $W $S/map3_par.py --out $1.txt --shards 3 -- --task $2 $CK --set flights $extra --num-envs 64 --time-scale 1.6 --seed $SEED $SCAN > $1.log 2>&1 &
  pids="$pids $!"; ser_map3
}
# FE_CHECK_FLAT=1 (2026-11-05, the blind teacher phase's early-stop gate): the flat goals only (no map3, no stairs).
# Drift is not judged there either (FE_FLAT_SKIP, 11-05 18:00): the Go2W blind teacher passed every other flat goal
# at 3000 with sideways creep 5.5% (bar 5%); the phase only decides when stairs start, and the mixed / polish
# phases and tinynav judge drift at the full bar on every seed (go2-home's drift guards are polish-stage terms).
FLAT=${FE_CHECK_FLAT:-0}
# FE_CHECK_PART (2026-11-11, the user: "检查分两档（先快速筛查，过了再跑 map3、3 个种子和 tinynav）"): flat = the flat
# evaluations only (~5 min beside a training; map3 and eval_stairs not run, g_stairs not judged), stairs = map3 and
# eval_stairs only, then the whole judgement over both parts' files (this PREFIX's flat part must have run), all =
# both at once (default). FE_CHECK_FLAT=1 is flat with drift not judged either (the T phase).
PART=${FE_CHECK_PART:-all}
[ $FLAT = 1 ] && PART=flat
if [ $PART != flat ]; then
map3 ${P}_noop $TE noop
map3 ${P}_op $TE op
fi
if [ $KIND = student ] && [ $PART != flat ]; then
  map3 ${P}_noopedge $TE-RealEdge noop
  map3 ${P}_opedge $TE-RealEdge op
fi
wait $pids; pids=""
# A shard that died (e.g. out of GPU memory: 2 of 3 shards at 12:05 10-31) leaves scenarios MISSING; such a run is
# repeated once on its own, and a run still short of 18 scenarios voids the check.
complete() { [ "$(grep -c 'reached=' $1.txt 2>/dev/null)" -ge 18 ]; }
short=""
for run in $([ $PART != flat ] && echo noop:$TE:noop op:$TE:op) $([ $KIND = student ] && [ $PART != flat ] && echo noopedge:$TE-RealEdge:noop opedge:$TE-RealEdge:op); do
  IFS=: read -r tag task mode <<< "$run"
  if ! complete ${P}_$tag; then
    echo "$(date '+%m-%d %T') $KIND $R $CK s$SEED: map3 $tag incomplete ($(grep -c 'reached=' ${P}_$tag.txt 2>/dev/null)/18), repeated" >> $O/checks.out
    nmap=100; map3 ${P}_$tag $task $mode; wait; pids=""
    complete ${P}_$tag || short="$short $tag"
  fi
done
if [ $PART != stairs ]; then
evalrun ${P}_am.txt $RS $S/accept_matrix.py --task $TF --envs-per-cmd 32 $CK $AMA &
pids="$pids $!"; ser
fi
if [ $PART != flat ]; then
evalrun ${P}_stairs.txt $W scripts/go2_blind/eval_stairs.py --task $TE $CK --seed $SEED &
pids="$pids $!"; ser
fi
if [ $PART != stairs ]; then
evalrun ${P}_drift3.txt $RS $S/drift_all.py --task $TF $CK $DRIFTA &
pids="$pids $!"; ser
# turning in place left / right in turn (go2-depth 2026-11-06, the user: "左右交替转多次后的累积位移"; g_spinalt), the yaw
# command slewed as the robot's controller brings it in (DRIFTA's robots)
evalrun ${P}_spinalt.txt $RS $S/spin_alt.py --task $TF $CK $(echo $DRIFTA | grep -o -- "--yaw-accel [0-9.]*") &
pids="$pids $!"; ser
evalrun ${P}_stand.txt $RS $S/eval_standing.py --task $TF $CK $STANDA &
pids="$pids $!"; ser
# standing 60 s at the zero command with the robot's payload limit (go2-depth 2026-11-07, the user: Go2W "负重的时候"
# fell standing; g_loadstand): Go2 / Go2W 12 kg, As2 10, As2W 13, B2 / B2W 40 (tn_gate_watch.sh's PAY)
LPAY=$(bash jobs/accept_payload.sh $R) || exit 2   # the one payload table
# accept_tables judges the falls after 3 s only (LOADSTAND_SETTLE, 2026-11-08, the user approved). long_stand.py
# --handover (zero actions = the policy's own PD at the default stance) is NOT used: those gains are far softer than
# the robot's FixStand (Go2W kp 60 / 80) and 89 / 256 Go2W fell while held with 12 kg.
evalrun ${P}_loadstand.txt $RS $S/long_stand.py --task $TF $CK --seconds 60 --payload $LPAY &
pids="$pids $!"   # waited for like the rest (2026-11-11: it was not, and the judge could read it unfinished)
# Stop then move at the payload limit (2026-11-09, the user: in the Go2W 12 kg flat video the legs stood wrong after the
# stop from 1.5 m/s and the next moves nearly fell): judged since 11-10 (g_posture, doc/acceptance.md).
evalrun ${P}_stopmove.txt $RS $S/stop_then_move.py --task $TF $CK --payload $LPAY &
pids="$pids $!"; ser
wait $pids; pids=""
evalrun ${P}_march.txt $RS $S/march_eval.py --task $TF $CK &
pids="$pids $!"; ser
evalrun ${P}_clock.txt $RS $S/gait_clock.py --task $TF $CLOCK $CK &
pids="$pids $!"; ser
if [[ $R == *W ]]; then  # wheeled: all four wheels down driving forward / backward (the acceptance's g_wheels)
  evalrun ${P}_gait.txt $RS $S/gait_fb.py --task $TF $CK --envs-per-cmd 32 &
  pids="$pids $!"; ser
fi
# The loaded runs of the same tests (2026-11-11, the user: "带负载的同一组考题（12 kg 下的蠕动、踏步漂移、跑偏、交替原地转）
# 只测量、写进报告，不参与通过与否"): drift, alternating turns, stop creep, march at the payload limit, into *_pay.txt;
# scripts/blind/acceptance/payload_gap.py reports them beside the unloaded runs. Never judged (FE_PAY_INFO=0: skipped).
if [ "${FE_PAY_INFO:-1}" = 1 ]; then
  evalrun ${P}_drift3_pay.txt $RS $S/drift_all.py --task $TF $CK $DRIFTA --payload $LPAY &
  pids="$pids $!"; ser
  evalrun ${P}_spinalt_pay.txt $RS $S/spin_alt.py --task $TF $CK $(echo $DRIFTA | grep -o -- "--yaw-accel [0-9.]*") --payload $LPAY &
  pids="$pids $!"; ser
  evalrun ${P}_stand_pay.txt $RS $S/eval_standing.py --task $TF $CK $STANDA --payload $LPAY &
  pids="$pids $!"; ser
  evalrun ${P}_march_pay.txt $RS $S/march_eval.py --task $TF $CK --payload $LPAY &
  pids="$pids $!"; ser
fi
fi   # the flat part
wait $pids
v=$(FE_PASS_SKIP=$({ [ $FLAT = 1 ] && echo ${FE_FLAT_SKIP:-g_stairs,g_drift}; } || { [ $PART = flat ] && echo g_stairs; }) $PY jobs/fe_pass.py $R $P 2>&1 | tail -n 1)
[ -n "$short" ] && v="FAIL: INVALID: map3 incomplete:$short"
[[ "$v" == PASS || "$v" == FAIL* ]] || v="FAIL: judge error: $v"
edge=""
if [ $KIND = student ] && [ $PART != flat ]; then
  en=$(mean_reached ${P}_noopedge.txt) eo=$(mean_reached ${P}_opedge.txt)
  edge=" edge noop $en op $eo"
  if ! python3 -c "import sys; n,o=map(float,sys.argv[1:]); sys.exit(0 if n >= 90 and o >= 95 else 1)" $en $eo; then
    if [[ "$v" == PASS ]]; then v="FAIL: 棱抹平楼梯：不通过"; else v="$v；棱抹平楼梯：不通过"; fi
  fi
fi
echo "$v" > ${P}_verdict.txt
# the unloaded / loaded readings side by side (informational; payload_gap.py)
[ $PART != stairs ] && [ "${FE_PAY_INFO:-1}" = 1 ] && $PY scripts/blind/acceptance/payload_gap.py $R $P > ${P}_payload_gap.txt 2>&1
echo "$(date '+%m-%d %T') $KIND $R $CK s$SEED ($(( $(date +%s) - t0 )) s): noop $(mean_reached ${P}_noop.txt) op $(mean_reached ${P}_op.txt)$edge | $(grep failed ${P}_am.txt) | $(tail -n 1 ${P}_stairs.txt) | $(tail -n 1 ${P}_clock.txt) | $v" >> $O/checks.out
