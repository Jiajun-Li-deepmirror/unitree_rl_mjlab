#!/bin/bash
# students_progress_test.sh (unitree-rl-mjlab-62 2026-11-13): stub tests of jobs/unified/students.sh's stages by
# progress -- stage 1 ending on its behavior loss levelling off (scripts/unified/s1_plateau.py), stage 2's gates and
# early stop, the pick at the cap, the reuse of the gate's tier 2 by the acceptance, the timeline events -- with a fake
# training (logs, checkpoints, at a few ms per iteration) and a fake acceptance (verdicts by a rule on the end
# iteration). No GPU, no simulator; ~3 min. Run from the repo root: bash jobs/unified/students_progress_test.sh
# Prints one line per check and "ALL OK" / "FAILED: n".
set -u
REPO=$(cd $(dirname $0)/../.. && pwd)
W=$(mktemp -d ${TMPDIR:-/tmp}/sp_test.XXXXXX)
NF=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; NF=$((NF + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1  [$2]"; fi; }

setup() {   # a fresh root
  local d=$1
  rm -rf $d && mkdir -p $d/jobs/unified $d/scripts/unified $d/logs/jobs $d/logs/checks $d/logs/rsl_rl $d/teacher
  cp $REPO/jobs/unified/students.sh $REPO/jobs/unified/formal.env $d/jobs/unified/
  cp $REPO/scripts/unified/s1_plateau.py $d/scripts/unified/
  touch $d/teacher/model_100.pt
  echo "PY=$d/stubpy" > $d/env.sh
  # the python: train.py faked, everything else the real interpreter
  cat > $d/stubpy <<'EOF'
#!/bin/bash
if [ "$1" = scripts/train.py ]; then shift; exec bash $(dirname $0)/stub_train.sh "$@"; fi
exec python3 "$@"
EOF
  chmod +x $d/stubpy
  # a fake training: model_0 at once, a log block per iteration, a checkpoint every 100 and at the end
  cat > $d/stub_train.sh <<'EOF'
#!/bin/bash
task=$1; shift
SV=100
for a in "$@"; do case $a in --agent.run-name=*) name=${a#*=} ;; --agent.max-iterations=*) M=${a#*=} ;; --agent.save-interval=*) SV=${a#*=} ;; esac; done
echo "$name $*" >> logs/train_args.txt
case $task in *-RL) dir=go2_u_student_${task##*Student-}; dir=${dir,,}; dir=${dir%-rl}_rl; st=2 ;;
  *) dir=go2_u_student_${task##*Student-}; dir=${dir,,}; st=1 ;; esac
run=logs/rsl_rl/$dir/$(date +%Y-%m-%d_%H-%M-%S)_$name
mkdir -p $run/records
echo '{"ev":"start","it":0}' > $run/records/timeline.jsonl
touch $run/model_0.pt
for ((i = 0; i < M; i++)); do
  echo "Learning iteration $i/$M"
  if [ $st = 1 ]; then echo "  Mean behavior loss: $(awk -v i=$i "BEGIN { print ${STUB_LOSS:-0.05 + 1.3 * exp(-i / 60)} }")"; fi
  [ $((i % SV)) = 0 ] && [ $i -gt 0 ] && touch $run/model_$i.pt
  if [ $st = 2 ] && [ "${STUB_CRASH_S2:-x}" = $i ]; then   # a stage-2 crash, STUB_CRASH_TIMES times over all runs
    c=$(cat logs/crashes 2> /dev/null || echo 0)
    if [ $c -lt ${STUB_CRASH_TIMES:-1} ]; then echo $((c + 1)) > logs/crashes; echo "RuntimeError: stub crash"; exit 1; fi
  fi
  [ $st = 2 ] && sleep ${STUB_IT_S:-0.01}
done
touch $run/model_$((M - 1)).pt   # (the runner's last save: model_<max - 1>)
EOF
  # a fake acceptance: the end = the newest checkpoint at or before K (else the last); the verdict by STUB_RULE
  # "fail<E1 hard<E2 pass" (FAIL below E1, HARDPASS with gap (E2-end)/1000 below E2, PASS above); every call logged
  cat > $d/jobs/unified/accept.sh <<'EOF'
#!/bin/bash
R=$1 KIND=$2 RUN=$3; shift 3
r=${R,,}
its=$(ls $RUN/model_*.pt | grep -oE '[0-9]+\.pt$' | grep -oE '[0-9]+' | sort -n)
if [ $# -gt 0 ]; then e=$(echo "$its" | awk -v k=$1 '$1 <= k' | tail -n 1); else e=$(echo "$its" | tail -n 1); fi
P=logs/checks/u_${r}_${KIND}_$(basename $RUN)_a8_$e
echo "call kind=$KIND base=${BASELINE:-0} reuse=${REUSE_TIER2:-0} e=$e jobs=${ACCEPT_JOBS:-} peak=${ACCEPT_PEAK:-} split=${ACCEPT_DRIFT_SPLIT:-} need=${ACCEPT_NEED_MB:+set}" >> logs/calls.txt
[ "${BASELINE:-0}" = 1 ] && echo "need kind=$KIND ${ACCEPT_NEED_MB:-}" >> logs/needs.txt
if [ "${REUSE_TIER2:-0}" = 1 ] && [ -s ${P}_verdict.txt ]; then
  v=$(head -n 1 ${P}_verdict.txt); echo "reused kind=$KIND e=$e $v" >> logs/calls.txt
else
  t0=$(date +%s.%N)
  sleep ${STUB_GATE_S:-2}
  [ "${BASELINE:-0}" = 1 ] && echo "$t0 $(date +%s.%N)" >> logs/spans.txt
  read e1 e2 <<< "$(echo ${STUB_RULE:-fail<1500 hard<2500 pass} | grep -oE '[0-9]+' | tr '\n' ' ')"
  if [ $e -lt $e1 ]; then v="FAIL: 楼梯 (stub)"; g=0
  elif [ $e -lt $e2 ]; then v=PASS; g=$(( e2 - e ))
  else v=PASS; g=0; fi
  echo "$v" > ${P}_verdict.txt
  echo "gap $g" > ${P}_stairs.txt
fi
[ "${BASELINE:-0}" = 1 ] && exit 0
if [[ "$v" == PASS ]]; then echo "$RUN/avg8_$e.pt" > logs/jobs/unified_${r}_$KIND.pick; touch logs/jobs/unified_${r}_$KIND.pick.delivered; exit 0; fi
exit 1
EOF
  # the judge's quality gap: the stub stairs file's "gap G" (/1000)
  cat > $d/scripts/unified/accept_judge.py <<'EOF'
def quality_gap(r, path):
  g = int(open(path).read().split()[1]) / 1000
  return g, f"stub gap {g}"
EOF
}

runit() {   # DIR ENV...: students.sh Go2 with the stub root
  local d=$1; shift
  ( cd $d && env UNIFIED_ROOT=$d TEACHER=$d/teacher/model_100.pt STOP_CHECK=0 KINDS=blind S1_POLL_S=1 S2_POLL_S=1 \
      F_S2_GATE_WAIT_S=5 "$@" timeout 600 bash jobs/unified/students.sh Go2 > $d/out.txt 2>&1; echo $? > $d/rc )
}
L() { cat $1/logs/jobs/unified_go2_students.txt 2> /dev/null; }
s2run() { ls -d $1/logs/rsl_rl/go2_u_student_blind_rl/*_blind2_* 2> /dev/null | tail -n 1; }
s1run() { ls -d $1/logs/rsl_rl/go2_u_student_blind/*_blind1_* 2> /dev/null | tail -n 1; }
lastck() { ls $1/model_*.pt | grep -oE '[0-9]+\.pt$' | grep -oE '[0-9]+' | sort -n | tail -n 1; }

# --- s1_plateau.py on its own
python3 - $REPO/scripts/unified <<'EOF' && ok "s1_plateau.judge: plateau / not / min / hold" || bad "s1_plateau.judge"
import math, sys
sys.path.insert(0, sys.argv[1])
import s1_plateau as s
flat = {i: 0.05 + 1.3 * math.exp(-i / 60) for i in range(3000)}
k, d = s.judge(flat, 1000, 200, 0.05, 2, 100)
assert k == 1100, (k, d)            # 1000 is the first check, 1100 the second in a row
steep = {i: 1.0 * 0.9995 ** i for i in range(4000)}   # ~9.5% per 200: never
assert s.judge(steep, 1000, 200, 0.05, 2, 100)[0] is None
slow = {i: 1.0 * 0.99985 ** i for i in range(4000)}   # ~3% per 200: from the minimum on
assert s.judge(slow, 1000, 200, 0.05, 2, 100)[0] == 1100
short = {i: 0.05 for i in range(900)}
assert s.judge(short, 1000, 200, 0.05, 2, 100)[0] is None
rise = {i: (0.05 if i < 1050 or i > 1150 else 0.2) for i in range(3000)}   # a rise is no fall: levelled off
assert s.judge(rise, 1000, 200, 0.05, 2, 100)[0] == 1100
step = {i: (0.05 if i < 1050 else 0.04) for i in range(3000)}   # a fall of 5% or more resets the run of checks
k, d = s.judge(step, 1000, 200, 0.05, 2, 100)
assert k == 1600, (k, d)
EOF

# --- A: stage 1 levels off; stage 2 fails at 1000, hard-passes at 1500, passes at 2000 -> stopped there, tier 2 reused
A=$W/A; setup $A
runit $A STUB_RULE="fail<1500 hard<2000 pass"
r1=$(s1run $A) r2=$(s2run $A)
check "A exit 0" '[ "$(cat $A/rc)" = 0 ]'
check "A S1 END plateau at 1100" 'L $A | grep -q "S1 END at 1100: plateau"'
check "A S1 stopped before its cap" '[ $(lastck $r1) -lt 4000 ]'
check "A S2 from model_1100" 'L $A | grep -q "stage 1 done: .*model_1100.pt"'
check "A stage 2 saves every 50 (formal), stage 1 at the runner's 100" 'grep -q "blind2_tx_s0 .*--agent.save-interval=50" $A/logs/train_args.txt && ! grep -q "blind1_tx_s0 .*save-interval" $A/logs/train_args.txt'
check "A gate 500 judged (11 checkpoints at a save every 50)" 'L $A | grep -q "S2 GATE 500: FAIL"'
check "A gate 1000 FAIL" 'L $A | grep -q "S2 GATE 1000: FAIL"'
check "A gate 1500 HARDPASS" 'L $A | grep -q "S2 GATE 1500: HARDPASS quality gap 0.500"'
check "A gate 2000 PASS + PICK, stage 2 stopped" 'L $A | grep -q "S2 GATE 2000: PASS" && L $A | grep -q "S2 PICK a8_2000 (gate at 2000): stage 2 stopped"'
check "A S2 stopped before its cap" '[ $(lastck $r2) -lt 3000 ]'
check "A gates ran with the gate settings" 'grep "base=1" $A/logs/calls.txt | grep -v "jobs=1 peak=1 split=1 need=set" | wc -l | grep -qx 0'
check "A acceptance reused the gate's tier 2 at 2000" 'grep -q "reused kind=blind e=2000 PASS" $A/logs/calls.txt'
check "A pick is a8_2000" 'grep -q "avg8_2000.pt" $A/logs/jobs/unified_go2_blind.pick'
check "A timeline: s1_end plateau" 'grep -q "\"ev\": \"s1_end\", \"it\": 1100.*\"why\": \"plateau\"" $r1/records/timeline.jsonl'
check "A timeline: gates + pick" 'grep -c "\"ev\": \"s2_gate\"" $r2/records/timeline.jsonl | grep -qvx 0 && grep -q "\"ev\": \"s2_pick\", \"it\": 2000.*\"why\": \"gate\"" $r2/records/timeline.jsonl'
check "A timeline still JSON lines" 'python3 -c "import json,sys; [json.loads(l) for f in sys.argv[1:] for l in open(f)]" $r1/records/timeline.jsonl $r2/records/timeline.jsonl'

# --- B: no plateau (S1 to its cap); every gate a hard pass, the gap smallest near the end -> the cap's pick; the
# cap's last average gated (it was not)
B=$W/B; setup $B
runit $B STUB_LOSS='0.9995 ^ i' F_STUDENT_S1_ITERS=1500 F_STUDENT_S2_ITERS=1700 STUB_RULE="fail<0 hard<99999 pass" STUB_GATE_S=1
r2=$(s2run $B)
check "B exit 0" '[ "$(cat $B/rc)" = 0 ]'
check "B S1 END cap" 'L $B | grep -q "S1 END at 1499: cap"'
check "B the cap last average gated" 'L $B | grep -q "S2 GATE 1699 (the cap.s last average): started"'
check "B pick at the cap = the smallest gap (the last, 1699)" 'L $B | grep -q "S2 PICK a8_1699 at the cap: hard items passed, quality gap 98.300"'
check "B acceptance reused 1700" 'grep -q "reused kind=blind e=1699 PASS" $B/logs/calls.txt'

# --- C: every gate fails -> the last average, its tier 2 from the cap's gate, reused: FAIL, exit 3 (no fallback)
C=$W/C; setup $C
runit $C F_STUDENT_S2_ITERS=1600 STUB_RULE="fail<99999 hard<99999 pass" STUB_GATE_S=1
check "C exit 3 (no accepted student, fallback off)" '[ "$(cat $C/rc)" = 3 ]'
check "C no S2 PICK" '! L $C | grep -q "S2 PICK"'
check "C acceptance on the last average, reused FAIL" 'grep -q "reused kind=blind e=1599 FAIL" $C/logs/calls.txt'

# --- D: a probe keeps the fixed stages (no plateau stop, no gates)
D=$W/D; setup $D
runit $D STUDENT_PROBE=1 S1_ITERS=1300 S2_ITERS=600
check "D probe exit 0" '[ "$(cat $D/rc)" = 0 ]'
check "D probe S1 to 1300, no S1 END" '[ $(lastck $(s1run $D)) = 1299 ] && ! L $D | grep -q "S1 END"'
check "D probe no gates" '! L $D | grep -q "S2 GATE" && [ ! -f $D/logs/calls.txt ]'
check "D probe keeps the runner's saves" '! grep -q "save-interval" $D/logs/train_args.txt'

# --- E: F_S2_GATE_EVERY_IT=0 -> stage 2 to its cap, the acceptance as before (no reuse file)
E=$W/E; setup $E
runit $E F_S2_GATE_EVERY_IT=0 F_STUDENT_S2_ITERS=900 STUB_RULE="fail<0 hard<0 pass"
check "E exit 0, no gates, accept on the last" '[ "$(cat $E/rc)" = 0 ] && ! L $E | grep -q "S2 GATE" && grep -q "base=0 reuse=1 e=899" $E/logs/calls.txt'

# --- F: two students side by side (one card): each one's first gate waits for the card -- both judged early
F=$W/F; setup $F
runit $F KINDS="blind depth" F_STUDENT_S2_ITERS=2600 STUB_RULE="fail<99999 hard<99999 pass" STUB_GATE_S=6
check "F blind's first gate judged" 'L $F | grep -qE "STUDENT blind S2 GATE 500: FAIL"'
check "F depth's first gate judged (waited, not skipped)" 'L $F | grep -qE "STUDENT depth S2 GATE 500: FAIL" && ! L $F | grep -q "STUDENT depth S2 GATE 500: skipped"'
check "F one gate on the card at a time (no two gate runs overlap)" 'sort -n $F/logs/spans.txt | awk "{ if (NR > 1 && \$1 < e) bad = 1; e = \$2 } END { exit bad }"'
check "F the depth gates use the depth table, the blind ones the blind table" 'grep -q "^need kind=depth \*:7500\$" $F/logs/needs.txt && grep "^need kind=blind" $F/logs/needs.txt | grep -q "stand:5800" && ! grep "^need kind=depth" $F/logs/needs.txt | grep -q stand'
check "F every gate run judged" '[ $(wc -l < $F/logs/spans.txt) = $(L $F | grep -cE "S2 GATE [0-9]+: (FAIL|PASS|HARDPASS)") ]'

# --- G: the gates' average differs from the acceptance's -> setup error, nothing trained
G=$W/G; setup $G
runit $G F_S2_GATE_AVG_N=6
check "G AVG_N mismatch: exit 2, no training" '[ "$(cat $G/rc)" = 2 ] && L $G | grep -q "setup error: F_S2_GATE_AVG_N (6)" && [ -z "$(s1run $G)" ]'

# --- H: stage 2 crashes once at 700 -> started over from the stage-1 end (a new run, not a resume), then accepted
H=$W/H; setup $H
runit $H F_STUDENT_S2_ITERS=1200 STUB_RULE="fail<0 hard<0 pass" STUB_CRASH_S2=700 STUB_CRASH_TIMES=1 F_S2_GATE_EVERY_IT=0
check "H exit 0 after one stage-2 restart" '[ "$(cat $H/rc)" = 0 ] && L $H | grep -q "STUDENT blind S2 FAULT (exit 1, attempt 1"'
check "H two stage-2 runs, both from the stage-1 end (no resume)" '[ $(grep -c "blind2_tx_s0 .*--agent.student-checkpoint=.*model_1100.pt" $H/logs/train_args.txt) = 2 ] && ! grep -q "blind2_tx_s0 .*resume" $H/logs/train_args.txt'
check "H the failed attempt's log kept, the acceptance on the second run" '[ -f $H/logs/jobs/unified_go2_blind2_tx_s0.fault1.log ] && grep -q "e=1199" $H/logs/calls.txt'
# --- I: stage 2 crashes twice -> a tool fault (exit 4), no third attempt
I=$W/I; setup $I
runit $I F_STUDENT_S2_ITERS=1200 STUB_CRASH_S2=700 STUB_CRASH_TIMES=5 F_S2_GATE_EVERY_IT=0
check "I exit 4 after the retry also crashed" '[ "$(cat $I/rc)" = 4 ] && L $I | grep -q "TOOL FAULT: blind stage 2 ended with exit 1" && [ $(grep -c "blind2_tx_s0" $I/logs/train_args.txt) = 2 ]'

echo "(work dir $W)"
[ $NF = 0 ] && { echo "ALL OK"; rm -rf $W; exit 0; }
echo "FAILED: $NF"
exit 1
