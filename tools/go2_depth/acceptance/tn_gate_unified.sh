#!/bin/bash
# tn_gate_unified.sh PORT (go2-depth 2026-11-12, doc/redesign.md 2.3): the tinynav tier of the unified acceptance, run
# on the workstation for the box at PORT. For each logs/accept/<name>.ureq ("R CKPT U<KIND>", jobs/unified/accept.sh)
# without a .verdict:
#   1. on the box: scripts/unified/export_student.py R kind CKPT logs/probe/<name> (policy.onnx, deploy.yaml,
#      blind_pilot.txt, manifest.json); pulled here
#   2. blind: tinynav-sim's import_blind_policy.py (as the blind gate); depth: as exported (legged: the registered
#      depth student's deploy metadata when the export has none; wheeled: slide_gate.sh's)
#   3. tinynav-sim with TNSIM_MODEL=uros (the unitree_ros-aligned robot models, as the training's):
#      unloaded  accept_stairs.py, 4 runs per route: legged kicks <= 2 / run, mid >= 80%, edge <= 10%, slips <= 0.2
#                / run, 0 falls; wheeled slide-back <= 5%, start idle <= 0.5 / run, heading <= 3 deg, 0 falls, plus
#                slide_gate.sh's L0 slide gate and top landing
#      loaded    the walking payload (jobs/accept_payload.sh), 7 runs per route (42): falls <= 5% of the runs;
#                wheeled slide-back <= 5% too
#   roll, map3 flights and the corner payload are not run (doc/redesign.md 2.4: the records are gone).
# The verdict (PASS / FAIL: ...) goes back as <name>.verdict; log $C/accept_unified.txt.
set -u
P=$1
SSH=/home/dm/.cache/go2-depth-work/ssh/r
D=/root/gpufree-data/go2-depth/depth_fe
HERE=$(cd "$(dirname "$0")" && pwd)
C=${GO2_DEPTH_STATE:-$HOME/.cache/go2-depth-work/check}
WT=${TINYNAV_MUJOCO:-/home/dm/.cache/go2-depth-work/tnsim-wheeled/mujoco}
PY=/home/dm/miniconda3/envs/unitree_rl_mjlab/bin/python
TNL=${TN_LOCK:-$C/tinynav_$P.lock}
OUT=$C/accept_unified.txt
LOADED_RUNS=7 FALL_RATE_BAR=0.05
mkdir -p $C/accept_unified $C/gate_pull $C/slide_gate_onnx
export TNSIM_MODEL=uros ROBOT_ASSETS=/home/dm/.cache/go2-depth-work/model_stage MUJOCO_GL=egl OMP_NUM_THREADS=2
verdict() { echo "$2" | timeout 30 $SSH $P "cat > $D/${1%.ureq}.verdict"; echo "$(date '+%m-%d %H:%M') $1 -> $2" >> $OUT; }
while true; do
  reqs=$(timeout 30 $SSH $P "cd $D && for q in logs/accept/*.ureq; do [ -f \"\$q\" ] && [ ! -f \"\${q%.ureq}.verdict\" ] && echo \$q; done" < /dev/null 2> /dev/null)
  for q in $reqs; do
    read R CK KINDTAG <<< "$(timeout 30 $SSH $P "cat $D/$q" < /dev/null)"
    r=${R,,} kind=$(echo ${KINDTAG#U} | tr A-Z a-z) base=$(basename $q .ureq)
    PD=logs/probe/$base n=$base
    timeout 1800 $SSH $P "cd $D && source env.sh > /dev/null 2>&1 && UNIFIED_TASKS=1 \$PY scripts/unified/export_student.py $R $kind $CK $PD > $PD.export.log 2>&1; tail -n 1 $PD.export.log" < /dev/null > /dev/null
    mkdir -p $C/gate_pull/$n
    for f in policy.onnx deploy.yaml blind_pilot.txt manifest.json; do
      timeout 300 $SSH $P "cat $D/$PD/$f" < /dev/null > $C/gate_pull/$n/$f 2> /dev/null
    done
    [ -s $C/gate_pull/$n/policy.onnx ] && [ -s $C/gate_pull/$n/deploy.yaml ] || { verdict $q "FAIL: export / pull (tool fault)"; continue; }
    if [ $kind = blind ]; then
      (cd $WT && $PY scripts/import_blind_policy.py $C/gate_pull/$n $C/gate_pull/$n.onnx > $C/gate_pull/$n.import.log 2>&1)
      [ -s $C/gate_pull/$n.onnx ] || { verdict $q "FAIL: blind import (tool fault)"; continue; }
      X=$C/gate_pull/$n.onnx
    else
      cp $C/gate_pull/$n/policy.onnx $C/gate_pull/$n.onnx
      if [[ $R == *W ]]; then
        X=$C/gate_pull/$n.onnx   # slide_gate.sh writes its deploy-metadata copy to slide_gate_onnx/
      else
        REF=$(cd $WT && $PY -c "from sim.robots import ROBOTS; print(ROBOTS['$r'].policy_path('perception'))" 2> /dev/null)
        $PY - $C/gate_pull/$n.onnx $REF $C/slide_gate_onnx/$n.onnx <<'PYEOF'
import sys, onnx
src, ref, out = sys.argv[1:4]
m = onnx.load(src)
if {p.key: p.value for p in m.metadata_props}.get("sim_cone") != "elliptic":
    R = {p.key: p.value for p in onnx.load(ref, load_external_data=False).metadata_props}
    keep = {p.key: p.value for p in m.metadata_props}
    del m.metadata_props[:]
    for k, v in {**R, **{k: v for k, v in keep.items() if k == "contract"}}.items():
        e = m.metadata_props.add(); e.key, e.value = k, v
onnx.save(m, out)
PYEOF
        X=$C/slide_gate_onnx/$n.onnx
      fi
    fi
    PAY=$(bash "$HERE/../../../jobs/accept_payload.sh" $R) || PAY=0,0,0,0
    fails=()
    ( flock 9; cd $WT
      [[ $R == *W ]] && bash $HERE/slide_gate.sh $r $X > $C/accept_unified/$n.slide.log 2>&1
      [[ $R == *W ]] && [ -s $C/slide_gate_onnx/$n.onnx ] && X=$C/slide_gate_onnx/$n.onnx
      $PY $HERE/accept_stairs.py $r $X 4 > $C/accept_unified/$n.log 2>&1
      PAYLOAD=$PAY $PY $HERE/accept_stairs.py $r $X $LOADED_RUNS > $C/accept_unified/${n}_pay.log 2>&1 ) 9> $TNL
    st=$(grep "^TOTAL" $C/accept_unified/$n.log | tail -n 1)
    pay=$(grep "^TOTAL" $C/accept_unified/${n}_pay.log | tail -n 1)
    [ -n "$st" ] || fails+=("楼梯：未出结果")
    IFS=',' read -ra items <<< "$(echo "$st" | sed -n 's/.*-> FAIL //p')"
    for item in "${items[@]}"; do
      item=$(echo $item)
      [ -z "$item" ] || [ "$item" = roll ] || [ "$item" = "end idle (info)" ] || fails+=("楼梯:${item// /_}")
    done
    if [[ $R == *W ]]; then
      gate=$(grep "$n " $C/slide_gate.txt 2> /dev/null | tail -n 1)   # slide_gate.sh's record, as the blind gate reads it
      [[ "$gate" == *"-> PASS"* ]] || fails+=("滑回门")
    fi
    if [ -z "$pay" ]; then fails+=("负载：未出结果")
    else
      pf=$(echo "$pay" | grep -o "falls [0-9]*" | cut -d" " -f2)
      routes=$(grep -cE '^  [a-z_0-9]+ +(steps|slide-back|kicks)' $C/accept_unified/${n}_pay.log)
      total=$((routes * LOADED_RUNS))
      python3 -c "import sys; f, t = int(sys.argv[1]), int(sys.argv[2]); sys.exit(0 if t > 0 and f / t <= $FALL_RATE_BAR else 1)" ${pf:-999} $total \
        || fails+=("负载：摔倒 ${pf:-?}/$total")
      ps=$(echo "$pay" | grep -o "slide-back [0-9.]*%" | grep -o "[0-9.]*")
      [ -z "$ps" ] || python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 5.0 else 1)" $ps || fails+=("负载：滑回 $ps%")
    fi
    if [ ${#fails[@]} = 0 ]; then v="PASS"; else v="FAIL: ${fails[*]}"; fi
    verdict $q "$v"
    { echo "  stairs: $st"; echo "  payload ($PAY, $LOADED_RUNS runs per route): $pay"; } >> $OUT
  done
  sleep 120
done
