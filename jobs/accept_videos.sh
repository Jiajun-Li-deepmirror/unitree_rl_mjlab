#!/bin/bash
# accept_videos.sh KIND R SRC [OUT [TITLE]] (go2-depth 2026-11-11, the user: "统一下验收视频脚本，不管是盲走还是感知，验收视频有两个，
# 一个是平地内容，一个是楼梯内容 ... 两个视频里分上下，上面是没有负载，下面是有负载。其他的验收视频生成脚本可以删除了，统一使用一个";
# "机器的初始位置和画面上下两个（负重和无负重）最好统一角度"). The one acceptance-video script, blind and depth alike:
#   KIND blind  SRC = a blind package dir (model_N.pt, blind_pilot.txt, blind_params.txt); OUT default SRC/videos
#   KIND ublind / udepth / uteacher  SRC = a unified student's / teacher's checkpoint; OUT required
#   KIND depth  SRC = a depth student checkpoint, its environment as its chain built it (jobs/v2_robot.sh: the
#               package's pilot + the V2 extras - V2_DROP + V2_MORE); OUT required
# Two videos in OUT, each two clips stacked (tools/go2_depth/video/stack_compare.py): on top without a payload, below
# at the payload limit (jobs/accept_payload.sh):
#   <r>_flat_compare.mp4  scripts/blind/record_flat.py --reset-demo: the posture reset (hips splayed out / in, march
#                         to reset), stand, march in place, the speed and turn commands, the stops from 1.5 m/s and
#                         from backing up
#   <r>_map3_compare.mp4  scripts/blind/record_map3_tour.py --stair-stops: map3 up L0, across, up R1, turn, down R1,
#                         back, down L0, with the operator (heading and centreline held), stopping 2 s halfway up L0
#                         and halfway down R1
# Both clips of a pair start from the same seed, spawn and joint pose, and their cameras follow the commanded / route
# heading, so they differ only by the payload. Logs "R BLIND VIDEOS ..." / "R DEPTH VIDEOS ..." (or "... VIDEOS
# FAILED ...") to LOG (default logs/jobs/blind_<r>.txt / fe_<r>.txt); OUT.pid while it runs. Exit 0, or 1 on a failure.
# ACCEPT_ROOT: the tree to run in (default the boxes' depth_fe; the workstation: the repo, with PY set).
set -u
KIND=$1 R=$2 SRC=$3
r=${R,,}
cd "${ACCEPT_ROOT:-/root/gpufree-data/go2-depth/depth_fe}" || exit 2
# A PY given by the caller (the workstation, with its own PYTHONPATH) skips env.sh, which points PYTHONPATH and PY at
# the boxes' tree and env.
if [ -z "${PY:-}" ] && [ -f env.sh ]; then source env.sh > /dev/null 2>&1; fi
PY=${PY:-python3}
case $KIND in
  blind)
    OUT=${4:-$SRC/videos}
    CK=$(ls $SRC/model_*.pt | head -n 1)
    T=${5:-$R 盲走 $(basename $SRC)}
    L=${LOG:-logs/jobs/blind_$r.txt}
    if E=$(bash jobs/run_env.sh $SRC); then
      source $E   # the environment the package's model trained with (scripts/train.py, from 11-11)
    else          # an older package: its pilot and parameters, the robot's payload / wall stop / foot radius
      export BLIND_PILOT=$(sed 's/^,//' $SRC/blind_pilot.txt)
      export BLIND_PARAMS=$(cat $SRC/blind_params.txt 2> /dev/null)
      source jobs/fe_robot.sh $R > /dev/null || exit 2   # BLIND_PAYLOAD, BLIND_WALL_STOP, BLIND_FOOT_R
    fi
    export BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$([[ $CK = /* ]] && echo $CK || echo $PWD/$CK)
    RUN=("$PY") TF=Unitree-$R-Blind-Flat-Scratch TS=Unitree-$R-Blind-Scratch TAGW=BLIND ;;
  depth)
    CK=$SRC OUT=${4:?accept_videos.sh depth: OUT} T=${5:-$R 感知 $(basename $(dirname $SRC))}
    L=${LOG:-logs/jobs/fe_$r.txt}
    source jobs/v2_robot.sh $R || exit 2
    if E=$(bash jobs/run_env.sh $CK); then V2_EXTRA=""; else
    if [ $V2_KIND = wheeled ]; then V2_EXTRA=$V2_WHEELED_EXTRA; else V2_EXTRA=$V2_LEGGED_EXTRA; fi
    for k in ${V2_DROP:+${V2_DROP//,/ }}; do V2_EXTRA=$(echo "$V2_EXTRA" | tr ',' '\n' | grep -v -x "$k" | paste -sd,); done
    export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=0
    source jobs/fe_robot.sh $R > /dev/null || exit 2
    if [ -n "$V2_CONS" ]; then export BLIND_CONS_INIT=$V2_CONS; else unset BLIND_CONS_INIT; fi
    export BLIND_PILOT=$(cat $PKG/blind_pilot.txt)$V2_EXTRA$V2_ROBOT_EXTRA${V2_MORE:+,$V2_MORE} BLIND_ANCHOR_COEFF=0 \
      BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/${FE_INIT_RUN:-blind_init_fe}/model_$N.pt
    [ -f $PKG/blind_params.txt ] && export BLIND_PARAMS=$(cat $PKG/blind_params.txt)
    fi
    [ -n "${E:-}" ] && source $E   # the student's own run (scripts/train.py, from 11-11) over the rebuilt one
    RUN=("$PY" jobs/student_eval.py) TF=Unitree-$R-Depth-Student-Flat TS=Unitree-$R-Depth-Student-Stairs TAGW=DEPTH ;;
  ublind|udepth)   # a unified student (2026-11-12, doc/redesign.md): SRC = its stage-2 checkpoint, OUT required
    K=${KIND#u}
    CK=$SRC OUT=${4:?accept_videos.sh $KIND: OUT} T=${5:-$R $([ $K = blind ] && echo 盲走 || echo 感知)（新流程） $(basename $(dirname $SRC))}
    L=${LOG:-logs/jobs/unified_${r}_students.txt}
    export UNIFIED_TASKS=1 BLIND_PILOT="(the unified recipe)"
    unset U_PAYLOAD   # the check tasks' payload: 0 kg for the unloaded clips; --payload for the loaded ones
    RUN=("$PY" jobs/unified/u_eval.py) TF=Unitree-$R-U-Student-${K^}-Flat TS=Unitree-$R-U-Student-${K^}-Stairs
    TAGW=U${K^^} ;;
  uteacher)   # a unified teacher (2026-11-14, the user: "如果老师模型通过的话，给我看一下他的验收视频"): SRC = its
    # checkpoint, OUT required; its own check tasks (the privileged teacher's observations), the same two videos
    CK=$SRC OUT=${4:?accept_videos.sh $KIND: OUT} T=${5:-$R 老师（新流程） $(basename $(dirname $SRC)) $(basename $SRC .pt)}
    L=${LOG:-logs/jobs/unified_${r}_videos.txt}
    export UNIFIED_TASKS=1 BLIND_PILOT="(the unified recipe)"
    unset U_PAYLOAD
    RUN=("$PY" jobs/unified/u_eval.py) TF=Unitree-$R-U-Teacher-Flat TS=Unitree-$R-U-Teacher-Stairs TAGW=UTEACHER ;;
  *) echo "accept_videos.sh: KIND blind, depth, ublind, udepth or uteacher" >&2; exit 2 ;;
esac
# a training's payload ramp (BLIND_PAYLOAD ...,ramp=N in run_env.sh) is not the robot's: evaluations see the full payload
export BLIND_PAYLOAD=$(echo "${BLIND_PAYLOAD:-}" | sed -E 's/,ramp=[0-9]+//')
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
PAY=$(bash jobs/accept_payload.sh $R) || { log "$R $TAGW VIDEOS FAILED no payload for $R"; exit 1; }
TX="--r1-top-y 5.6"; case $R in B2|B2W) TX="--r1-top-y 5.6 --r1-turn-x -9.3" ;; esac
mkdir -p $OUT
echo $$ > $OUT.pid
trap 'rm -f $OUT.pid' EXIT
LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock}
locked() { if [ -e "$(dirname $LOCK)" ]; then flock $LOCK "$@"; else "$@"; fi; }   # one GPU job at a time with the checks
t0=$(date +%s)
echo "recording $KIND $CK (pilot md5 $(echo ${BLIND_PILOT:-} | md5sum | cut -c1-8), payload $PAY)" > $OUT/recording.txt
locked "${RUN[@]}" scripts/blind/record_flat.py --task $TF $CK --out $OUT/${r}_flat.mp4 --title "$T" --reset-demo \
  > $OUT/flat.log 2>&1
locked "${RUN[@]}" scripts/blind/record_flat.py --task $TF $CK --out $OUT/${r}_flat_payload.mp4 --title "$T" --reset-demo \
  --payload $PAY > $OUT/flat_pay.log 2>&1
locked "${RUN[@]}" scripts/blind/record_map3_tour.py $CK --task $TS --out $OUT/${r}_map3.mp4 --title "$T" --stair-stops $TX \
  > $OUT/map3.log 2>&1
locked "${RUN[@]}" scripts/blind/record_map3_tour.py $CK --task $TS --out $OUT/${r}_map3_payload.mp4 --title "$T" --stair-stops \
  --payload $PAY $TX > $OUT/map3_pay.log 2>&1
out="" bad=""
for c in "flat:${r}_flat:平地" "map3:${r}_map3:map3 楼梯"; do
  IFS=: read -r k base what <<< "$c"
  a=$OUT/$base.mp4 b=$OUT/${base}_payload.mp4 o=$OUT/${base}_compare.mp4
  if [ ! -s $a ] || [ ! -s $b ]; then
    bad="$bad $k: no clip ($(tail -n 1 $OUT/$k.log $OUT/${k}_pay.log 2> /dev/null | grep -v '^==' | tr '\n' ' ' | cut -c1-120))"; continue
  fi
  j=$("$PY" tools/go2_depth/video/stack_compare.py $a $b $o --title "$T $what" --top-label "无负载" \
      --bottom-label "负载 ${PAY%%,*} kg（前 $(echo $PAY | cut -d, -f2) m、上 $(echo $PAY | cut -d, -f4) m）" 2> $OUT/stack_$k.log | tail -n 1)
  if [ -s $o ]; then out="$out $(basename $o) ($(echo "$j" | grep -oE '"seconds": [0-9.]+' | grep -oE '[0-9.]+$') s, $(du -m $o | cut -f1) MB)"
  else bad="$bad $k: stacking failed ($(tail -n 1 $OUT/stack_$k.log | cut -c1-120))"; fi
done
m=$(( ($(date +%s) - t0) / 60 ))
if [ -n "$bad" ]; then log "$R $TAGW VIDEOS FAILED after $m min:$bad${out:+; made$out}"; exit 1; fi
log "$R $TAGW VIDEOS $OUT:$out (after $m min)"
