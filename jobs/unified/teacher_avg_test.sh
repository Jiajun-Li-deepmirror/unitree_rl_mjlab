#!/bin/bash
# teacher_avg_test.sh (unitree-rl-mjlab-62 2026-11-15, go2-depth's request): stub tests of jobs/unified/teacher.sh's
# candidate at a PRE-PASS (F_TEACHER_AVG_N) -- N=1 as before (model_K), N=4 the average of the seed's newest 4
# checkpoints <= K (its path in the gates, the pick, <pick>.judged / .hardpass and the log), fewer than 4 saved (the next
# gate), two seeds apart, the cap's two pick rules on averaged candidates, a failed average (a tool fault, no half file)
# -- with a fake training (tiny torch checkpoints), a fake teacher_gate.py (PRE-PASS at the newest checkpoint) and fake
# tier-2 gates; the real scripts/unified/avg_ckpt.py averages. No GPU, no simulator; ~2-3 min. Needs a python with
# torch: $PY, else env.sh's, else python3. Run from the repo root: bash jobs/unified/teacher_avg_test.sh
# Prints one line per check and "ALL OK" / "FAILED: n".
set -u
REPO=$(cd $(dirname $0)/../.. && pwd)
TPY=${PY:-}
[ -n "$TPY" ] || TPY=$(source $REPO/env.sh > /dev/null 2>&1; echo ${PY:-})
[ -n "$TPY" ] && [ -x "$TPY" ] || TPY=python3
$TPY -c "import torch" 2> /dev/null || { echo "teacher_avg_test.sh: $TPY has no torch (set PY=<a python with torch>)"; exit 2; }
export TPY
W=$(mktemp -d ${TMPDIR:-/tmp}/tavg_test.XXXXXX)
NF=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; NF=$((NF + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1  [$2]"; fi; }

setup() {   # a fresh root
  local d=$1
  rm -rf $d && mkdir -p $d/jobs/unified $d/scripts/unified $d/logs/jobs $d/logs/rsl_rl
  cp $REPO/jobs/unified/teacher.sh $REPO/jobs/unified/formal.env $d/jobs/unified/
  cp $REPO/scripts/unified/avg_ckpt.py $d/scripts/unified/
  echo "PY=$d/stubpy" > $d/env.sh
  # the python: train.py and teacher_gate.py faked, everything else (avg_ckpt.py) the real interpreter
  cat > $d/stubpy <<'EOF'
#!/bin/bash
case $1 in
  scripts/train.py) shift; exec bash $(dirname $0)/stub_train.sh "$@" ;;
  scripts/unified/teacher_gate.py) shift; exec bash $(dirname $0)/stub_gate.sh "$@" ;;
esac
exec $TPY "$@"
EOF
  chmod +x $d/stubpy
  # a fake training: its run dir, the checkpoints FAKE_CKS at once, FAKE_LATER once the gate has looked (logs/release_<seed>),
  # then it ends (FAKE_END=1) or runs until stopped
  cat > $d/stub_train.sh <<'EOF'
#!/bin/bash
task=$1; shift
for a in "$@"; do case $a in --agent.run-name=*) name=${a#*=} ;; esac; done
r=${task#Unitree-}; r=${r%%-U-*}; r=${r,,}
run=logs/rsl_rl/${r}_u_teacher/$(date +%Y-%m-%d_%H-%M-%S)_$name
mkdir -p $run
seed=${name##*_s}
$TPY mkck.py $run $FAKE_CKS
if [ -n "${FAKE_LATER:-}" ]; then
  until [ -f logs/release_$seed ]; do sleep 0.2; done
  $TPY mkck.py $run $FAKE_LATER
fi
[ "${FAKE_END:-0}" = 1 ] && exit 0
exec sleep 300
EOF
  # tiny checkpoints shaped like a teacher's: every float tensor at value k, the normalizer's count k, infos marked k;
  # FAKE_BAD=<k>: that one has an extra tensor (avg_ckpt.py: exit 2)
  cat > $d/mkck.py <<'EOF'
import os, sys, torch
run, ks = sys.argv[1], [int(x) for x in sys.argv[2:]]
bad = int(os.environ.get("FAKE_BAD") or -1)
for k in ks:
  a = {"actor.0.weight": torch.full((2, 2), float(k)), "priv_encoder.0.weight": torch.full((2,), float(k)),
       "obs_normalizer._mean": torch.full((3,), float(k)), "obs_normalizer.count": torch.tensor(k)}
  if k == bad:
    a["extra.weight"] = torch.zeros(1)
  ck = {"actor_state_dict": a, "critic_state_dict": {"v": torch.full((1,), float(k))}, "optimizer_state_dict": {},
        "iter": k, "infos": {"unified": {"progress": float(k)}, "env_state": None}}
  torch.save(ck, f"{run}/model_{k}.pt.tmp")
  os.replace(f"{run}/model_{k}.pt.tmp", f"{run}/model_{k}.pt")
EOF
  # teacher_gate.py: --help names --flat-tier2; a run: PRE-PASS at its newest checkpoint (and the training may go on)
  cat > $d/stub_gate.sh <<'EOF'
#!/bin/bash
[ "$1" = --help ] && { echo "--pre --flat-tier2"; exit 0; }
run=$1
k=$(ls $run/model_*.pt 2> /dev/null | grep -oE 'model_[0-9]+\.pt$' | grep -oE '[0-9]+' | sort -n | tail -n 1)
touch logs/release_${run##*_s}
[ -n "$k" ] && echo "PRE-PASS $k (stub)" || echo "NOT YET (stub)"
EOF
  # tier 2: FAKE_STAIRS=pass | hard | fail, or fail for a path matching FAKE_STAIRS_FAIL; the flat gate passes
  cat > $d/jobs/unified/stairs_gate.sh <<'EOF'
#!/bin/bash
echo "$2" >> logs/stairs_calls.txt
[ -f "$2" ] || { echo "FAULT: no checkpoint $2"; exit 4; }
case "$2" in *${FAKE_STAIRS_FAIL:-@none@}*) echo "FAIL: gap 0.30 (stub)"; exit 1 ;; esac
case ${FAKE_STAIRS:-pass} in
  hard) echo "HARDPASS: target gap 0.20 (stub)"; exit 5 ;;
  fail) echo "FAIL: gap 0.40 (stub)"; exit 1 ;;
esac
echo "PASS (stub)"; exit 0
EOF
  cat > $d/jobs/unified/flat_gate.sh <<'EOF'
#!/bin/bash
echo "$2" >> logs/flat_calls.txt
[ -f "$2" ] || { echo "FAULT: no checkpoint $2"; exit 4; }
echo "PASS (stub)"; exit 0
EOF
}

run_teacher() {   # ROOT SEEDS [VAR=VALUE ...]: teacher.sh Go2W full on the stubs; its exit code in RC
  local d=$1 seeds=$2; shift 2
  ( cd $d && env UNIFIED_ROOT=$d GATE_EVERY=4 STOP_WAIT_S=4 NUM_ENVS=64 PARALLEL=1 "$@" \
      bash jobs/unified/teacher.sh Go2W full "$seeds" 100 > logs/teacher_out.txt 2>&1 )
  RC=$?
}
avg_reads() {   # FILE -> "w=<mean> count=<count> iter=<iter> progress=<progress> sources=<first k>..<last k>/<n>"
  $TPY - "$1" <<'EOF'
import re, sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
a = c["actor_state_dict"]
src = [int(re.search(r"model_(\d+)\.pt$", p).group(1)) for p in c["infos"]["averaged"]]
print(f"w={a['actor.0.weight'].mean().item():g} priv={a['priv_encoder.0.weight'].mean().item():g} "
      f"count={int(a['obs_normalizer.count'])} iter={c['iter']} progress={c['infos']['unified']['progress']:g} "
      f"critic={c['critic_state_dict']['v'].item():g} sources={src[0]}..{src[-1]}/{len(src)}")
EOF
}
CKS10="100 200 300 400 500 600 700 800 900 1000"
PICK=logs/jobs/unified_go2w_full.pick
LOG=logs/jobs/unified_go2w.txt

# A: N=1 -- model_K itself, as before
d=$W/a; setup $d
run_teacher $d 0 F_TEACHER_AVG_N=1 FAKE_CKS="$CKS10"
check "A N=1: exit 0" '[ $RC = 0 ]'
check "A N=1: the pick is the run's model_1000" 'grep -qE "^logs/rsl_rl/go2w_u_teacher/[0-9_-]+_ufull_s0/model_1000\.pt seed 0$" $d/$PICK'
check "A N=1: the stairs gate judged model_1000" 'grep -qE "_ufull_s0/model_1000\.pt$" $d/logs/stairs_calls.txt'
check "A N=1: the flat gate judged the same file" 'diff -q $d/logs/stairs_calls.txt $d/logs/flat_calls.txt > /dev/null'
check "A N=1: nothing averaged" '[ ! -d $d/logs/avg ]'
check "A N=1: the log names model_1000" 'grep -q "seed 0: model_1000 tier-2 stairs: PASS" $d/$LOG'

# B: N=4 (formal.env's default), two seeds, seed 0's stairs fail
d=$W/b; setup $d
run_teacher $d "0 1" FAKE_CKS="$CKS10" FAKE_STAIRS_FAIL=_s0_avg4
check "B N=4: formal.env's default is 4" 'grep -q "the average of the seed.s newest 4 checkpoints" $d/$LOG'
check "B N=4: exit 0" '[ $RC = 0 ]'
P1=$(cut -d' ' -f1 $d/$PICK 2> /dev/null)
check "B N=4: the pick is seed 1's average at 1000" '[[ "$P1" =~ ^logs/avg/go2w_[0-9_-]+_ufull_s1_avg4/model_1000\.pt$ ]] && grep -q " seed 1$" $d/$PICK'
check "B N=4: the picked file exists" '[ -s $d/$P1 ]'
check "B N=4: the average = the mean of model_700..1000, count / iter / infos / critic from model_1000" \
  '[ "$(avg_reads $d/$P1)" = "w=850 priv=850 count=1000 iter=1000 progress=1000 critic=1000 sources=700..1000/4" ]'
check "B N=4: the stairs and flat gates judged the averages" '! grep -qv "^logs/avg/" $d/logs/stairs_calls.txt && grep -qx "$P1" $d/logs/flat_calls.txt'
check "B N=4: the two seeds' averages in two dirs" '[ $(xargs -n1 dirname < $d/logs/stairs_calls.txt | sort -u | wc -l) = 2 ]'
check "B N=4: seed 0's failed average in <pick>.judged" 'grep -qE "^0\.30 logs/avg/go2w_[0-9_-]+_ufull_s0_avg4/model_1000\.pt seed 0: FAIL" $d/$PICK.judged'
check "B N=4: the log says avg 4 of model_700..1000" 'grep -q "seed 1: avg 4 of model_700..1000 tier-2 stairs: PASS" $d/$LOG && grep -q "TEACHER PICK $P1 (seed 1; avg 4 of model_700..1000;" $d/$LOG'
check "B N=4: <pick>.others keeps seed 0" 'grep -qx "envs=32 seeds=0" $d/$PICK.others'
# what students.sh reads from the pick: the checkpoint (made absolute) and the seed
T=$(cut -d' ' -f1 $d/$PICK); TS=$(grep -oE 'seed [0-9]+' $d/$PICK | grep -oE '[0-9]+' | head -n 1)
check "B N=4: students.sh's reading -- checkpoint and seed 1" '( cd $d && [ -f "$T" ] ) && [ "$TS" = 1 ]'
check "B N=4: no half file left" '[ -z "$(find $d/logs/avg -name "*.part")" ]'

# C: N=4, only model_100..300 at the first PRE-PASS: waits; model_400 next gate -> the average of 100..400
d=$W/c; setup $d
run_teacher $d 0 FAKE_CKS="100 200 300" FAKE_LATER=400
check "C fewer: exit 0" '[ $RC = 0 ]'
check "C fewer: the first PRE-PASS waited" 'grep -q "seed 0: PRE-PASS at 300; fewer than 4 checkpoints <= 300" $d/$LOG'
check "C fewer: nothing judged at 300" '! grep -q "model_300" $d/logs/stairs_calls.txt'
P1=$(cut -d' ' -f1 $d/$PICK 2> /dev/null)
check "C fewer: the pick is the average at 400" '[[ "$P1" =~ _ufull_s0_avg4/model_400\.pt$ ]]'
check "C fewer: of model_100..400" '[[ "$(avg_reads $d/$P1)" == "w=250 "*"sources=100..400/4" ]]'

# D: N=4, every gate a hard pass (stairs + flat), the training ends -> the cap's pick from <pick>.hardpass
d=$W/d; setup $d
run_teacher $d 0 FAKE_CKS="$CKS10" FAKE_STAIRS=hard FAKE_END=1
check "D hardpass: exit 0" '[ $RC = 0 ]'
check "D hardpass: <pick>.hardpass has the average" 'grep -qE "^0\.20 logs/avg/go2w_[0-9_-]+_ufull_s0_avg4/model_1000\.pt seed 0: HARDPASS" $d/$PICK.hardpass'
check "D hardpass: the cap's pick is that average" 'grep -qE "^logs/avg/go2w_[0-9_-]+_ufull_s0_avg4/model_1000\.pt seed 0$" $d/$PICK'
check "D hardpass: the log says at the cap" 'grep -q "at the cap: every hard item passed" $d/$LOG'

# E: N=4, every stairs gate fails, the training ends -> the cap's pick below the gate, from <pick>.judged
d=$W/e; setup $d
run_teacher $d 0 FAKE_CKS="$CKS10" FAKE_STAIRS=fail FAKE_END=1
check "E judged: exit 0" '[ $RC = 0 ]'
check "E judged: the cap's pick below the gate is the average" 'grep -qE "^logs/avg/go2w_[0-9_-]+_ufull_s0_avg4/model_1000\.pt seed 0$" $d/$PICK && grep -q "at the cap: below the gate" $d/$LOG'

# F: N=4, model_800 differs (avg_ckpt.py exit 2): a tool fault, no pick, no half file; nothing judged -> needs a human
d=$W/f; setup $d
run_teacher $d 0 FAKE_CKS="$CKS10" FAKE_BAD=800 FAKE_END=1
check "F fault: exit 3 (nothing judged)" '[ $RC = 3 ]'
check "F fault: the log names the averaging fault" 'grep -q "seed 0: TOOL FAULT averaging the candidate at 1000" $d/$LOG'
check "F fault: no gate ran" '[ ! -f $d/logs/stairs_calls.txt ]'
check "F fault: no average and no half file" '[ -z "$(find $d/logs/avg -name "model_*.pt*" ! -name "*.log")" ]'

# G: a bad F_TEACHER_AVG_N stops at once
d=$W/g; setup $d
run_teacher $d 0 F_TEACHER_AVG_N=0 FAKE_CKS="$CKS10"
check "G F_TEACHER_AVG_N=0: exit 2" '[ $RC = 2 ] && grep -q "F_TEACHER_AVG_N must be a whole number" $d/$LOG'

echo
if [ $NF = 0 ]; then echo "ALL OK"; rm -rf $W; exit 0; fi
echo "FAILED: $NF (roots kept in $W)"; exit 1
