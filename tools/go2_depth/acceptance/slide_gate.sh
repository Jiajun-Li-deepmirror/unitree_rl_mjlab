#!/bin/bash
# slide_gate.sh ROBOT ONNX (go2-depth 2026-11-04, the user: "添加滑回作为新验收标准"; also the landing-step faults the
# user reported: rear wheels on the start landing 11-03, front wheels on the top landing 11-04). The wheeled stairs
# acceptance items, in tinynav-sim on map3 L0 (stair_wheels.py: spawn (-10.34, 5.75) yaw 270 +- seed, 0.5 m/s for 10 s,
# 12 climbs, ~620 step-ups), with the policy's ONNX. The ONNX must carry the deploy metadata (tinynav takes the contact
# solver from it; 11-03 raw probes read 43-44% slid back): an export without it gets the deployed
# <robot>_depth_v4_fe metadata (same contract) first.
#   滑回 (slid back)              <= SLIDE_MAX (5.0) % of the step-ups
#   起步平台踏步 (start landing)    <= START_MAX (0.5) per climb (rear wheels lifted with the riser > 0.3 m ahead)
#   顶部平台踏步 (top landing)      <= TOP_MAX (1.0) per climb (front wheels lifted on the landing above, flat ahead)
#   摔倒                           0
# Luck rule (the user 11-04): slid back within 1.5 points over the bar on seeds 0-11 -> seeds 12-23 run too and the
# 24 climbs decide. Result line appended to $GO2_DEPTH_STATE/slide_gate.txt.
# Paths (tools/go2_depth/README.md): code next to this script; state (pulled exports, logs, verdicts) in
# GO2_DEPTH_STATE (default ~/.cache/go2-depth-work/check); WT = the tinynav-sim tree that runs the acceptance.
set -u
R=$1 X=$2
SLIDE_MAX=${SLIDE_MAX:-5.0} START_MAX=${START_MAX:-0.5} TOP_MAX=${TOP_MAX:-1.0}
HERE=$(cd "$(dirname "$0")" && pwd)
C=${GO2_DEPTH_STATE:-$HOME/.cache/go2-depth-work/check}
mkdir -p $C
WT=${TINYNAV_MUJOCO:-/home/dm/.cache/go2-depth-work/tnsim-wheeled/mujoco}
PY=/home/dm/miniconda3/envs/unitree_rl_mjlab/bin/python
TMP=$C/slide_gate_onnx; mkdir -p $TMP $C/stairw
N=$(basename $X .onnx)
REF=$(cd $WT && $PY -c "from sim.robots import ROBOTS; print(ROBOTS['$R'].policy_path('perception'))" 2> /dev/null)
$PY - "$X" "$REF" "$TMP/$N.onnx" <<'EOF'
import sys, onnx
src, ref, out = sys.argv[1:4]
m = onnx.load(src)
md = {p.key: p.value for p in m.metadata_props}
if md.get("sim_cone") != "elliptic":
    R = {p.key: p.value for p in onnx.load(ref, load_external_data=False).metadata_props}
    del m.metadata_props[:]
    for k, v in R.items():
        e = m.metadata_props.add(); e.key, e.value = k, v
    print("slide_gate: no deploy metadata in the export, took the deployed model's", flush=True)
onnx.save(m, out)
EOF
cd $WT
run() {  # SEED0 -> log
  SEED0=$1 ROBOT_ASSETS=/home/dm/.cache/go2-depth-work/model_stage MUJOCO_GL=egl \
    $PY $HERE/stair_wheels.py $R $TMP/$N.onnx 12 > $C/stairw/gate_${R}_${N}_s$1.log 2>&1
}
run 0
LOGS="$C/stairw/gate_${R}_${N}_s0.log"
judge() {  # logs... -> "ups slid start top falls climbs"
  $PY - "$@" <<'EOF'
import re, sys
ups = slid = start = top = falls = climbs = 0
for p in sys.argv[1:]:
    s = open(p).read()
    t = re.search(r"TOTAL (\d+) climbs: landing steps (\d+) .*steps up (\d+):.*slid back (\d+) .*top-landing steps (\d+) ", s)
    climbs += int(t.group(1)); start += int(t.group(2)); ups += int(t.group(3)); slid += int(t.group(4)); top += int(t.group(5))
    falls += len(re.findall(r"fell True", s))
print(ups, slid, start, top, falls, climbs)
EOF
}
read ups slid start top falls climbs <<< "$(judge $LOGS)"
pct=$($PY -c "print(f'{100*$slid/$ups:.1f}')")
luck=""
if $PY -c "import sys; sys.exit(0 if $SLIDE_MAX < $pct <= $SLIDE_MAX + 1.5 else 1)"; then
  run 12; LOGS="$LOGS $C/stairw/gate_${R}_${N}_s12.log"
  read ups slid start top falls climbs <<< "$(judge $LOGS)"
  luck=" (seeds 0-11 at $pct%, rerun 12-23)"
  pct=$($PY -c "print(f'{100*$slid/$ups:.1f}')")
fi
read v items <<< "$($PY - <<EOF
f = []
if $pct > $SLIDE_MAX: f.append("滑回")
if $start / $climbs > $START_MAX: f.append("起步平台踏步")
if $top / $climbs > $TOP_MAX: f.append("顶部平台踏步")
if $falls: f.append("摔倒")
print("PASS -" if not f else "FAIL " + "、".join(f))
EOF
)"
line="$(date '+%m-%d %H:%M') $R $N ($climbs climbs$luck): 滑回 $pct% (<= $SLIDE_MAX) | 起步平台踏步 $($PY -c "print(f'{$start/$climbs:.1f}')") / climb (<= $START_MAX) | 顶部平台踏步 $($PY -c "print(f'{$top/$climbs:.1f}')") / climb (<= $TOP_MAX) | 摔倒 $falls -> $v $items"
echo "$line" | tee -a $C/slide_gate.txt
