#!/bin/bash
# record_blind.sh R PKG [OUT] (go2-depth 2026-11-06, the user: acceptance videos with go2-home's scripts, adding the
# tests they miss): scripts/blind/record_flat.py (--reset-demo; + march in place) and record_map3_tour.py
# (+ --stair-stops), each also at the robot's payload limit (tn_gate_watch.sh's PAY), for a blind delivery package
# (model_N.pt + blind_pilot.txt). Videos and logs in OUT (default PKG/videos).
# The payload flat clip marches in place too (2026-11-09, the user: "负载验收视频添加一个原地踏步"; was --no-march).
# The payload clips are the plain clips with the payload and nothing else (2026-11-09, the user: "负载的验收视频完全按照
# 原版验收视频制作"): the flat one with --reset-demo and the default first stand (was --first-stand 30, no reset demo),
# the tour without --contact-goes-on (a body / thigh contact ends the episode there, as in the plain tour).
set -u
R=$1 BPKG=$2 OUT=${3:-$2/videos}
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
CK=$(ls $BPKG/model_*.pt | head -n 1)
PILOT=$(cat $BPKG/blind_pilot.txt)
# fe_robot.sh only for BLIND_PAYLOAD: it sets PKG / N of its own (11-06: with PKG here it recorded go2-home's
# go2w_polish_11300 instead of the package)
source jobs/fe_robot.sh $R > /dev/null
export BLIND_PILOT=$PILOT BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/$CK
case $R in Go2|Go2W) PAY=12,0.15,0,0.20 ;; As2) PAY=10,0.15,0,0.20 ;; As2W) PAY=13,0.15,0,0.20 ;; *) PAY=40,0.15,0,0.30 ;; esac
# B2 (11-06): strafing at R1's top y 5.0 its body still on the last step brushed the well-side rail (illegal contact,
# both tours); it walks onto the landing first (R1_down's spawn is at y 5.78)
# Go2W too (11-07, the user: the turn at the top was on the stairs): every robot walks onto R1's top landing first
TX=""; case $R in B2) TX="--r1-top-y 5.6 --r1-turn-x -9.3" ;; B2W) TX="--r1-top-y 5.6 --r1-turn-x -9.3" ;; *) TX="--r1-top-y 5.6" ;; esac
mkdir -p $OUT
CLIPS=${CLIPS:-flat,tour,tour_pay,flat_pay}   # CLIPS: which clips to record
want() { [[ ",$CLIPS," == *",$1,"* ]]; }
T="$R $(basename $BPKG)"
echo "recording $CK (pilot $(echo $PILOT | md5sum | cut -c1-8))"
want flat && $PY scripts/blind/record_flat.py --task Unitree-$R-Blind-Flat-Scratch $CK --out $OUT/${r}_flat.mp4 --title "$T" --reset-demo > $OUT/flat.log 2>&1
want tour && $PY scripts/blind/record_map3_tour.py $CK --task Unitree-$R-Blind-Scratch --out $OUT/${r}_map3_tour.mp4 --title "$T" --stair-stops $TX > $OUT/tour.log 2>&1
want tour_pay && $PY scripts/blind/record_map3_tour.py $CK --task Unitree-$R-Blind-Scratch --out $OUT/${r}_map3_tour_payload.mp4 --title "$T" --stair-stops --payload $PAY $TX > $OUT/tour_pay.log 2>&1
want flat_pay && $PY scripts/blind/record_flat.py --task Unitree-$R-Blind-Flat-Scratch $CK --out $OUT/${r}_flat_payload.mp4 --title "$T" --reset-demo --payload $PAY > $OUT/flat_pay.log 2>&1
ls -la $OUT
