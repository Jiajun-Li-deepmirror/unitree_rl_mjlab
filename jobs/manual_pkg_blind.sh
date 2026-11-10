#!/bin/bash
# manual_pkg_blind.sh R TAG CK (go2-depth 2026-11-08): blind_chain.sh's delivery package for a P checkpoint picked by hand
# (its chain had stopped at a luck-like seed-0 miss; manual_seeds_log.sh passed the other seeds and tinynav): the
# recipe env rebuilt from the chain log as manual_seeds_log.sh does, then logs/pkg/<r>_bl2_<TAG>_<K> as blind_chain.sh
# writes it (model, params, blind_pilot.txt, checkpoint.md5, policy.onnx, deploy.yaml).
# PKG_NAME (2026-11-09) names the package instead (e.g. an average of two checkpoints of TAG's lineage), PICK_NOTE
# says why it was picked by hand.
set -u
R=$1 TAG=$2 CK=$3
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh $R || exit 2
export $V2_PKG_VAR=$V2_PKG
source jobs/fe_robot.sh $R > /dev/null || exit 2
L=logs/jobs/blind_$r.txt
seg=$(awk -v t="blind chain $TAG:" 'index($0, t) {f=1} f' $L)
TP=$(echo "$seg" | grep -o "blind T PICK[^/]*\(logs/[^ ]*\.pt\)" | head -n 1 | grep -o "logs/.*\.pt")
PP=$(echo "$seg" | grep -o "blind P pilot .*" | head -n 1 | sed "s/blind P pilot //")
[ -n "$TP" ] && [ -n "$PP" ] || { echo "no T pick / P pilot for $TAG in $L"; exit 2; }
export BLIND_PILOT=$PP BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/$TP
PK=$(basename $CK .pt | cut -d_ -f2)
OUT=logs/pkg/${PKG_NAME:-${r}_bl2_${TAG}_$PK}
[ -e $OUT ] && { echo "$OUT exists"; exit 2; }
mkdir -p $OUT
cp $CK $OUT/model_$PK.pt && cp -r $V2_PKG/params $OUT/ 2> /dev/null
echo "$BLIND_PILOT" | sed 's/^,//' > $OUT/blind_pilot.txt
md5sum $OUT/model_$PK.pt | sed "s|$OUT/||" > $OUT/checkpoint.md5
echo "$(date '+%m-%d %H:%M:%S') manual pick $TAG: $CK (by hand: ${PICK_NOTE:-seed-0 luck miss, other seeds + tinynav pass})" >> $L
$PY scripts/go2_blind/export_onnx.py --task Unitree-$R-Blind-Flat-Scratch $OUT/model_$PK.pt --out $OUT > logs/jobs/blind_${r}_export.log 2>&1 \
  && $PY scripts/blind/export_deploy_yaml.py Unitree-$R-Blind-Polish --robot $r --out $OUT/deploy.yaml >> logs/jobs/blind_${r}_export.log 2>&1 \
  && echo "$(date '+%m-%d %H:%M:%S') $R BLIND DELIVERY $OUT (model md5 $(md5sum < $OUT/model_$PK.pt | cut -c1-8), policy.onnx $(md5sum < $OUT/policy.onnx | cut -c1-8))" >> $L \
  || echo "$(date '+%m-%d %H:%M:%S') $R blind export FAILED: $(tail -n 3 logs/jobs/blind_${r}_export.log)" >> $L
tail -n 1 $L
