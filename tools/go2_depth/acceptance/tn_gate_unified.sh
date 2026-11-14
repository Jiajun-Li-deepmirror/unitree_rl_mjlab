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
#      loaded    the walking payload (jobs/accept_payload.sh), two stages (2026-11-12, doc/accept_stats_proposal.md:
#                no pass by luck): LOADED_RUNS (10) runs per route; it passes when the falls' one-sided 95% upper
#                bound (Clopper-Pearson) is <= 5% (0 of 60), fails at 2 falls; 1 fall: LOADED_MORE (6) more runs per
#                route on fresh seeds (accept_stairs.py SEED0), and it passes when the bound over both is <= 5%
#                (1 of 96). (Was 7 runs per route, falls <= 5%: 2 of 42 passed, a bound of 14%.)
#                Wheeled: slide-back <= 5% in each stage too
#      STEEP_TYPICAL (2026-11-15, the user's option b; empty = off, every route under the walking payload as above):
#                a comma list of robots (Go2,Go2W) whose loaded stage judges the 15-18 cm routes (STEEP_ROUTES: the
#                two map3 routes and the pyramid's two 18 cm ones) under the typical payload's mass at the walking
#                payload's mount point (Go2 / Go2-W: 4 kg at x +0.15, z +0.20 -- inside the typical payload's mount
#                box, at its front / top end; the two loads differ in mass only) and the pyramid's two 14 cm routes
#                under the walking payload as before: the same count of runs, one bound over all of them (wheeled:
#                slide-back <= 5% in each of the two judged parts). The 15-18 cm routes under the walking payload
#                are run too, LOADED_RUNS each: reported in the verdict and the log, not judged. (The user's
#                wording, relayed 2026-11-15 00:12: every 15-18 cm route at 4 kg, the pyramid's 14 cm ones at 8 kg
#                -- as plan 2 does in the mjlab gate: the full load on 15-18 cm is reported, not judged.)
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
LOADED_RUNS=${LOADED_RUNS:-10} LOADED_MORE=${LOADED_MORE:-6} FALL_RATE_BAR=0.05
STEEP_TYPICAL=${STEEP_TYPICAL:-} STEEP_ROUTES=map3_up,map3_down,pit_18n_up,top_18n_down
# FALLS RUNS: 0 when the falls' one-sided 95% upper bound (Clopper-Pearson) is within FALL_RATE_BAR
cp_ok() { $PY -c "import sys; from scipy.stats import beta; f, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); \
sys.exit(0 if n > 0 and (1.0 if f >= n else beta.ppf(0.95, f + 1, n - f)) <= b else 1)" $1 $2 $FALL_RATE_BAR; }
cp_up() { $PY -c "import sys; from scipy.stats import beta; f, n = int(sys.argv[1]), int(sys.argv[2]); \
print(f'{100 * (1.0 if f >= n else beta.ppf(0.95, f + 1, n - f)):.1f}%')" $1 $2; }
falls_of() { grep "^TOTAL" $1 2> /dev/null | tail -n 1 | grep -o "falls [0-9]*" | cut -d" " -f2; }
routes_of() { grep -cE '^  [a-z_0-9]+ +(steps|slide-back|kicks)' $1 2> /dev/null; }
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
    # the loaded stage's judged parts "<log suffix>|<payload>|<accept_stairs.py's route choice>|<label>", and the
    # reported one (STEEP_TYPICAL)
    parts=("pay|$PAY||") info="" PAYT=""
    if [[ ",$STEEP_TYPICAL," == *",$R,"* ]]; then
      TYP=$(bash "$HERE/../../../jobs/accept_payload.sh" $R typical 2> /dev/null) || TYP=""
      [ -n "$TYP" ] || { verdict $q "FAIL: STEEP_TYPICAL names $R, which has no typical payload (tool fault)"; continue; }
      PAYT=${TYP%%,*},${PAY#*,}
      parts=("pay|$PAY|SKIP=$STEEP_ROUTES|" "payt|$PAYT|ONLY=$STEEP_ROUTES|15-18 cm ${TYP%%,*} kg")
      info="payinfo|$PAY|ONLY=$STEEP_ROUTES|"
    fi
    fails=()
    ( flock 9; cd $WT
      [[ $R == *W ]] && bash $HERE/slide_gate.sh $r $X > $C/accept_unified/$n.slide.log 2>&1
      [[ $R == *W ]] && [ -s $C/slide_gate_onnx/$n.onnx ] && X=$C/slide_gate_onnx/$n.onnx
      $PY $HERE/accept_stairs.py $r $X 4 > $C/accept_unified/$n.log 2>&1
      f1=0 t1=0 got=1
      for part in "${parts[@]}"; do
        IFS='|' read -r sfx pl sel lab <<< "$part"
        env PAYLOAD=$pl $sel $PY $HERE/accept_stairs.py $r $X $LOADED_RUNS > $C/accept_unified/${n}_$sfx.log 2>&1
        rm -f $C/accept_unified/${n}_${sfx}2.log
        f=$(falls_of $C/accept_unified/${n}_$sfx.log)
        [ -n "$f" ] || { got=0; continue; }
        f1=$((f1 + f)) t1=$((t1 + $(routes_of $C/accept_unified/${n}_$sfx.log) * LOADED_RUNS))
      done
      # the second loaded stage: only when the first leaves it open (1 fall, or a bound not yet within the bar)
      if [ $got = 1 ] && [ $f1 -lt 2 ] && ! cp_ok $f1 $t1; then
        for part in "${parts[@]}"; do
          IFS='|' read -r sfx pl sel lab <<< "$part"
          env PAYLOAD=$pl SEED0=$LOADED_RUNS $sel $PY $HERE/accept_stairs.py $r $X $LOADED_MORE > $C/accept_unified/${n}_${sfx}2.log 2>&1
        done
      fi
      rm -f $C/accept_unified/${n}_payinfo.log
      if [ -n "$info" ]; then
        IFS='|' read -r sfx pl sel lab <<< "$info"
        env PAYLOAD=$pl $sel $PY $HERE/accept_stairs.py $r $X $LOADED_RUNS > $C/accept_unified/${n}_$sfx.log 2>&1
      fi ) 9> $TNL
    st=$(grep "^TOTAL" $C/accept_unified/$n.log | tail -n 1)
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
    pf=0 total=0 lines=() missing=""
    for part in "${parts[@]}"; do
      IFS='|' read -r sfx pl sel lab <<< "$part"
      for stage in "" 2; do
        L=$C/accept_unified/${n}_$sfx$stage.log
        [ -z "$stage" ] || [ -f $L ] || continue
        line=$(grep "^TOTAL" $L 2> /dev/null | tail -n 1) f=$(falls_of $L)
        if [ -z "$line" ] || [ -z "$f" ]; then
          fails+=("负载${stage:+第二段}${lab:+（${lab// /_}）}：未出结果"); missing=1; continue
        fi
        runs=$LOADED_RUNS; [ -z "$stage" ] || runs=$LOADED_MORE
        pf=$((pf + f)) total=$((total + $(routes_of $L) * runs)) lines+=("${stage:+second stage, }$line")
        ps=$(echo "$line" | grep -o "slide-back [0-9.]*%" | grep -o "[0-9.]*")
        [ -z "$ps" ] || python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 5.0 else 1)" $ps || fails+=("负载：滑回 $ps%")
      done
    done
    [ -n "$missing" ] || cp_ok $pf $total || fails+=("负载：摔倒 $pf/$total（95% 上界 $(cp_up $pf $total)）")
    # STEEP_TYPICAL: the 15-18 cm routes under the walking payload -- in the verdict and the log, not judged
    note="" infoline=""
    if [ -n "$info" ]; then
      infoline=$(grep "^TOTAL" $C/accept_unified/${n}_payinfo.log 2> /dev/null | tail -n 1)
      fi_=$(falls_of $C/accept_unified/${n}_payinfo.log) ri=$(routes_of $C/accept_unified/${n}_payinfo.log)
      note="（15–18 cm 的路线（map3 两条、金字塔 18 cm 两条）按常用负载 ${PAYT%%,*} kg 判；这些路线上 ${PAY%%,*} kg 只报告：摔倒 ${fi_:-未出结果}/$(( ${ri:-0} * LOADED_RUNS ))）"
    fi
    if [ ${#fails[@]} = 0 ]; then v="PASS$note"; else v="FAIL: ${fails[*]}$note"; fi
    verdict $q "$v"
    { echo "  stairs: $st"
      echo "  payload ($PAY$([ -z "$info" ] || echo "; STEEP_TYPICAL: $STEEP_ROUTES at $PAYT"); falls $pf/$total, bound $(cp_up $pf $total)):"
      for line in "${lines[@]}"; do echo "    $line"; done
      [ -z "$info" ] || echo "    reported, not judged: ${infoline:-no result}"; } >> $OUT
  done
  sleep 120
done
