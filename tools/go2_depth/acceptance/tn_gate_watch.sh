#!/bin/bash
# tn_gate_watch.sh PORT (go2-depth 2026-11-05, the new pipeline's early stop): serves fe_chain.sh's FE_TN_GATE requests
# on the box at PORT. For each logs/accept/<run>_<K>.req ("R CKPT PILOT_EXTRA") without a .verdict: probe_export.sh the
# checkpoint, pull it, run the tinynav acceptance here, and write logs/accept/<run>_<K>.verdict back:
#   PASS when slide_gate.sh (map3 L0) passes, accept_stairs.py (map3 + pyramid, the user's stairs requirements) has no
#   failed item except roll (its bar is not calibrated yet: every deployed model is over 4 / 6 deg; recorded, not
#   judged; "heading" = the mean heading off the flights, wheeled, 11-07), and at the robot's payload limit accept_stairs.py shows 0 falls and slide-back <= 5% (wheeled; Go2W 10% 11-08 .. 11-09, then 5% again);
#   else FAIL: <items>. Details in $GO2_DEPTH_STATE/accept_watch.txt. Runs until killed.
# Paths (tools/go2_depth/README.md): code next to this script; state (pulled exports, logs, verdicts) in
# GO2_DEPTH_STATE (default ~/.cache/go2-depth-work/check); WT = the tinynav-sim tree that runs the acceptance.
set -u
P=$1
SSH=/home/dm/.cache/go2-depth-work/ssh/r
D=/root/gpufree-data/go2-depth/depth_fe
HERE=$(cd "$(dirname "$0")" && pwd)
C=${GO2_DEPTH_STATE:-$HOME/.cache/go2-depth-work/check}
mkdir -p $C
WT=${TINYNAV_MUJOCO:-/home/dm/.cache/go2-depth-work/tnsim-wheeled/mujoco}
PY=/home/dm/miniconda3/envs/unitree_rl_mjlab/bin/python
OUT=$C/accept_watch.txt
mkdir -p $C/accept_base $C/gate_pull
while true; do
  reqs=$(timeout 30 $SSH $P "cd $D && for q in logs/accept/*.req; do [ -f \"\$q\" ] && [ ! -f \"\${q%.req}.verdict\" ] && echo \$q; done" < /dev/null 2> /dev/null)
  for q in $reqs; do
    read R CK EXTRA BPILOT <<< "$(timeout 30 $SSH $P "cat $D/$q" < /dev/null)"
    r=${R,,}
    base=$(basename $q .req)
    X=logs/probe/${r}_$base.onnx
    n=$(basename $X .onnx)
    BLIND=0; [ "$EXTRA" = BLIND ] && BLIND=1
    if [ $BLIND = 1 ]; then
      # a blind policy (blind_chain.sh): go2-home's export (policy.onnx from the Flat-Scratch play config, deploy.yaml
      # from the Polish task), imported here as tinynav-sim's blind models are (scripts/import_blind_policy.py)
      # fe_robot.sh first (11-06): the pilot's "payload" key needs BLIND_PAYLOAD (B2 l4 9000 failed here on it)
      PD=logs/probe/${r}_$base
      timeout 900 $SSH $P "cd $D && source env.sh && source jobs/fe_robot.sh $R > /dev/null && export BLIND_PILOT=$BPILOT BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=\$PWD/$CK && mkdir -p $PD && \$PY scripts/go2_blind/export_onnx.py --task Unitree-$R-Blind-Flat-Scratch $CK --out $PD > $PD/export.log 2>&1 && \$PY scripts/blind/export_deploy_yaml.py Unitree-$R-Blind-Polish --robot $r --out $PD/deploy.yaml >> $PD/export.log 2>&1" < /dev/null > /dev/null
      mkdir -p $C/gate_pull/$n
      timeout 300 $SSH $P "cat $D/$PD/policy.onnx" < /dev/null > $C/gate_pull/$n/policy.onnx
      timeout 300 $SSH $P "cat $D/$PD/deploy.yaml" < /dev/null > $C/gate_pull/$n/deploy.yaml
      # the pilot beside them (2026-11-09): import_blind_policy.py takes the contact solver from it ("grip": elliptic,
      # impratio 10); without it every blind candidate ran on tinynav's default pyramidal cone at impratio 1, the
      # deployed blind models (and the depth exports) on elliptic / 10
      echo "$BPILOT" > $C/gate_pull/$n/blind_pilot.txt
      (cd $WT && $PY scripts/import_blind_policy.py $C/gate_pull/$n $C/gate_pull/$n.onnx > /dev/null 2>&1)
      [ -s $C/gate_pull/$n.onnx ] || { echo "FAIL: blind export / import" | timeout 30 $SSH $P "cat > $D/${q%.req}.verdict"; continue; }
      X2=$C/gate_pull/$n.onnx
      # wheeled blind too (2026-11-10, doc/acceptance.md): the L0 slide gate is the one measure of the top landing
      [[ $R == *W ]] && flock $C/tinynav.lock bash $HERE/slide_gate.sh $r $X2 > /dev/null 2>&1
    else
      timeout 900 $SSH $P "cd $D && bash jobs/probe_export.sh $R $CK $X $EXTRA 2>&1 | tail -n 1" < /dev/null > /dev/null
      if ! timeout 300 $SSH $P "cat $D/$X" < /dev/null > $C/gate_pull/$n.onnx.new || [ ! -s $C/gate_pull/$n.onnx.new ]; then
        echo "FAIL: export / pull" | timeout 30 $SSH $P "cat > $D/${q%.req}.verdict" ; continue
      fi
      mv $C/gate_pull/$n.onnx.new $C/gate_pull/$n.onnx
      if [[ $R == *W ]]; then
        flock $C/tinynav.lock bash $HERE/slide_gate.sh $r $C/gate_pull/$n.onnx > /dev/null 2>&1
      else
        # legged depth (11-06): no wheeled slide gate; the deploy metadata of the robot's registered depth student when the
        # export has none (slide_gate.sh's rule for the wheeled), then accept_stairs at the deployed depth's level
        mkdir -p $C/slide_gate_onnx
        REF=$(cd $WT && $PY -c "from sim.robots import ROBOTS; print(ROBOTS['$r'].policy_path('perception'))" 2> /dev/null)
        $PY - $C/gate_pull/$n.onnx $REF $C/slide_gate_onnx/$n.onnx <<'PYEOF'
import sys, onnx
src, ref, out = sys.argv[1:4]
m = onnx.load(src)
if {p.key: p.value for p in m.metadata_props}.get("sim_cone") != "elliptic":
    R = {p.key: p.value for p in onnx.load(ref, load_external_data=False).metadata_props}
    del m.metadata_props[:]
    for k, v in R.items():
        e = m.metadata_props.add(); e.key, e.value = k, v
onnx.save(m, out)
PYEOF
      fi
      X2=$C/slide_gate_onnx/$n.onnx
    fi
    # legged bars: the initial ones for every robot (2026-11-10, doc/acceptance.md, the user: "统一用原始门槛"); B2 had
    # its deployed depth's level (kicks 3.1 / mid 76.6; loaded 4.2 / 73.9 / slips 0.6) since 11-06
    LB="" LBP=""
    case $R in
      Go2|Go2W) PAY=12,0.15,0,0.20 ;; As2) PAY=10,0.15,0,0.20 ;; As2W) PAY=13,0.15,0,0.20 ;;
      # B2 (11-06): 40 kg at x 0.15 / z 0.30, 4 runs. At the training range's corner (x 0.30) the deployed b2_depth_v3_fe
      # and blind fe5000 fell 8 / 8 on top_18n_down (8 runs), the delivered blind l4_9000 2 / 8; at x 0.15 all 0 / 8.
      # The corner stays measured (8 runs, top_18n_down), reported, not judged.
      B2) PAY=40,0.15,0,0.30 ;;
      *) PAY=40,0.30,0,0.30 ;;
    esac
    # one tinynav run at a time on this workstation (11-08): the watchers of both boxes compose the scene into the same
    # assets/.compose/scene.xml (two at once: "empty scene.xml")
    ( flock 9; cd $WT && export ROBOT_ASSETS=/home/dm/.cache/go2-depth-work/model_stage MUJOCO_GL=egl OMP_NUM_THREADS=2
      LEGGED_BARS=$LB $PY $HERE/accept_stairs.py $r $X2 4 > $C/accept_base/$n.log 2>&1
      PR=2; [ $R = B2 ] && PR=4
      LEGGED_BARS=$LBP PAYLOAD=$PAY $PY $HERE/accept_stairs.py $r $X2 $PR > $C/accept_base/${n}_pay.log 2>&1
      [ $R = B2 ] && ONLY=top_18n_down PAYLOAD=40,0.30,0,0.30 $PY $HERE/accept_stairs.py $r $X2 8 > $C/accept_base/${n}_corner.log 2>&1
      # info only (11-06): every map3 flight with the stairs posture (accept_map3.py), not judged
      $PY $HERE/accept_map3.py $r $X2 2 > $C/accept_base/${n}_map3.log 2>&1 ) 9> $C/tinynav.lock
    gate=$(grep "$n " $C/slide_gate.txt | tail -n 1)
    st=$(grep "^TOTAL" $C/accept_base/$n.log | tail -n 1)
    pay=$(grep "^TOTAL" $C/accept_base/${n}_pay.log | tail -n 1)
    fails=()
    [[ $R != *W ]] || [[ "$gate" == *"-> PASS"* ]] || fails+=("滑回门")   # every wheeled policy (blind too since 2026-11-10)
    [ -n "$st" ] || fails+=("楼梯要求:未出结果")
    IFS=',' read -ra items <<< "$(echo "$st" | sed -n 's/.*-> FAIL //p')"
    for item in "${items[@]}"; do
      item=$(echo $item)
      [ -z "$item" ] || [ "$item" = roll ] || fails+=("楼梯要求:${item// /_}")
    done
    pf=$(echo "$pay" | grep -o "falls [0-9]*" | cut -d" " -f2)
    ps=$(echo "$pay" | grep -o "slide-back [0-9.]*%" | grep -o "[0-9.]*")
    [ "${pf:-1}" = 0 ] || fails+=("载荷:摔倒")
    # payload slide-back bar (11-08, the user: "如果实在很难降低那就下降门槛"): Go2W 10% -- no Go2W ever met 5% at 12 kg
    # (g14 / g20b / g21 / g23 / g25: 7.8-11.9%, through noslide / noslide2x), the deployed blind_v2 / depth_v5_ws 36-50%
    # back to 5% for Go2W too (2026-11-09, the user: "改回5%"): the 7.8-11.9% that justified 10% were blind candidates
    # on tinynav's default pyramidal cone (the importer had no pilot); on the trained elliptic / 10 g27b reads 5.3%
    PSB=5.0
    [ -n "$ps" ] && python3 -c "import sys; sys.exit(0 if float('$ps') <= $PSB else 1)" || [ -z "$ps" ] || fails+=("载荷:滑回")
    if [ ${#fails[@]} = 0 ]; then v="PASS"; else v="FAIL: ${fails[*]}"; fi
    echo "$v" | timeout 30 $SSH $P "cat > $D/${q%.req}.verdict"
    { echo "$(date '+%m-%d %H:%M') $R $base -> $v"; echo "  gate: $gate"; echo "  stairs: $st"; echo "  payload: $pay"
      [ -f $C/accept_base/${n}_corner.log ] && echo "  corner (info, 40 kg x 0.30 z 0.30): $(grep top_18n_down $C/accept_base/${n}_corner.log | grep -o 'falls [0-9]*/[0-9]*')"
      echo "  map3 flights (info): $(grep '^TOTAL ' $C/accept_base/${n}_map3.log | sed 's/^[^:]*: //')"
      echo "  posture up (info): $(grep '^POSTURE-up' $C/accept_base/${n}_map3.log | sed 's/^[^:]*: //')"; } >> $OUT
  done
  sleep 120
done
