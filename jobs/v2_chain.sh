#!/bin/bash
# v2_chain.sh R short|full|fullft TAG [AFTER_PID] (go2-depth 2026-11-05, the new pipeline; the user: one GPU to shake it
# down first, then both for the full runs; keep it generic). For any robot with inputs in v2_robot.sh (legged from 11-06):
#   A. teacher: fine-tune V2_TEACHER_FROM on the v2 recipe, depth_fe checks at FE_TEACHER_KS with a 0.08 speed
#      margin (FE_MARGIN), the last passing average picked; then stage-1 student: distillation (Student-Edge) from
#      V2_STUDENT_FROM under that teacher, not checked (FE_STUDENT_CHECK=0)
#   B. stage-2 student: PPO+BC (Student-Edge-RL) from the stage-1 average, checked on 3 seeds and exported
# short: teacher fine-tuned +400 from V2_TEACHER_FROM (one check), stage 1 800 it, stage 2 800 it (one check at 799),
# payload ramp 200 it -- a dry run of the mechanics; full (the user 11-05, "为什么不需要从盲狗开始训练" -> yes): the
# teacher from the widened blind delivery (blind_init_fe, fe_chain's default), 3000 it with checks at +1000..3000, so
# every robot starts from the same kind of point on the whole v2 recipe; stage 1 2000 it, stage 2 1500 it (checks
# 1000, 1499), payload ramp 1000 it; fullft: as full but the teacher fine-tuned +800/1200/1600 from V2_TEACHER_FROM
# (the comparison: does starting over pay).
# Picks logs/picks_v2_<TAG>, runs v2_<TAG> / v2s1_<TAG> / v2s2_<TAG>. AFTER_PID: wait for that training to end.
set -u
R=$1 MODE=$2 TAG=$3 AFTER=${4:-}
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe
source jobs/v2_robot.sh $R || exit 2
# V2_TF / V2_SF (11-06) override the robot's teacher fine-tune start and stage-1 warm start (B2: the v2 teacher
# avg10000 fine-tuned with feetedgew3, the students from the best stage-2 average so far).
V2_TEACHER_FROM=${V2_TF:-$V2_TEACHER_FROM} V2_STUDENT_FROM=${V2_SF:-$V2_STUDENT_FROM}
# V2_PKG_FROM / V2_CONS_FORCE (2026-11-10, jobs/full_chain.sh): the blind delivery and constraint start of the
# reproducible run; V2_SF=none distils the stage-1 student from scratch
[ -n "${V2_PKG_FROM:-}" ] && V2_PKG=$V2_PKG_FROM
[ -n "${V2_CONS_FORCE+x}" ] && V2_CONS=$V2_CONS_FORCE
case $MODE in
  short) TFROM=$V2_TEACHER_FROM TKS="400" S1=800 S2=800 S2KS="799" RAMP=4800 ;;
  full) TFROM= TKS="500 1000 1500 2000 2500 3000" S1=2000 S2=1500 S2KS="500 1000 1499" RAMP=24000 ;;
  fullft) TFROM=$V2_TEACHER_FROM TKS="400 800 1200 1600" S1=2000 S2=1500 S2KS="500 1000 1499" RAMP=24000 ;;
  *) echo "v2_chain.sh: mode short|full|fullft" >&2; exit 2 ;;
esac
# V2_RAMP overrides the payload ramp (11-06: 0 for a stage-2 rerun, whose fresh step counter would ramp from 0 kg
# again); V2_ONLY_S2=1 skips stage A (teacher and stage 1 picks reused); V2_S2_SUFFIX names the stage-2 rerun.
RAMP=${V2_RAMP:-$RAMP}
# V2_MORE adds pilot keys (11-06: feetedgew3 for the B2 student's edge / slip misses); V2_S2_FROM, V2_S2_ITERS and
# V2_S2_KS start stage 2 from another checkpoint (a stage-2 average) for that many iterations, checked at those K.
S1=${V2_S1_ITERS:-$S1} S2=${V2_S2_ITERS:-$S2} S2KS=${V2_S2_KS:-$S2KS}
TKS=${V2_TKS:-$TKS}   # V2_TKS (2026-11-10): the teacher's checked iterations (a mechanics smoke of a new package)
# V2_S2_BC_FINAL (2026-11-09): stage 2's final behavior-cloning weight (log-linear from 1.0; default 1.0 all along), to
# let the RL terms move a habit the teacher passed on (Go2W top-landing steps: stage 1 -> stage 2 2.7 -> 2.5-3.1 at 1.0)
PK=logs/picks_v2_$TAG
export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=$RAMP
# Legged (11-06): the legged recipe (V2_LEGGED_EXTRA) and fe_chain's own speed margin (0.05; the wheeled 0.08 is for
# the B2W students' slower descents).
if [ $V2_KIND = wheeled ]; then V2_EXTRA=$V2_WHEELED_EXTRA MARGIN=0.08; else V2_EXTRA=$V2_LEGGED_EXTRA MARGIN=0.05; fi
[ -n "${V2_EXTRA_FORCE:-}" ] && V2_EXTRA=$V2_EXTRA_FORCE   # the recipe's whole depth extras (2026-11-10)
# V2_DROP (2026-11-09): keys taken out of the v2 extras (e.g. endwall for Go2W: its teacher slowed every map3 climb
# 0.03-0.08 learning to slow at the end walls, up 0.874 -> 0.814; Go2W never had it, the deploy caps the command at walls)
for k in ${V2_DROP:+${V2_DROP//,/ }}; do V2_EXTRA=$(echo "$V2_EXTRA" | tr ',' '\n' | grep -v -x "$k" | paste -sd,); done
export FE_PICKS=$PK FE_CONS_INIT=$V2_CONS FE_PILOT_EXTRA=$V2_EXTRA$V2_ROBOT_EXTRA${V2_MORE:+,$V2_MORE} FE_TEACHER_RUN=v2_$TAG \
  FE_MARGIN=$MARGIN FE_TEACHER_PICK=first FE_LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock}
# Early stop (the user 11-05: "如果提前通过验收，就不要继续训练了，节省时间"): the teacher stops at its first check that
# passes with the margin (checks every 500); stage 2 at its first K whose 3 seeds and tinynav acceptance pass
# (FE_TN_GATE: scratchpad check/tn_gate_watch.sh on the workstation).
echo "$(date '+%m-%d %H:%M:%S') v2_chain $R $MODE $TAG: pilot +$FE_PILOT_EXTRA, teacher from ${TFROM:-the widened blind}, student from $V2_STUDENT_FROM" >> logs/jobs/fe_$r.txt
# V2_STAGE (2026-11-10, the reproducible run in four rounds -- blind, teacher, stage-1 student, stage-2 student; the user:
# each round passes before the robot's next): teacher = stop at the teacher pick; s1 = the stage-1 student under the
# picked teacher (exit 3 when there is none); s2 = stage 2 from the stage-1 average (exit 3 when there is none); all.
V2_STAGE=${V2_STAGE:-all}
case $V2_STAGE in
  all|teacher) ;;
  s1) [ -f $PK/${r}_teacher ] || { echo "v2_chain: no teacher pick in $PK"; exit 3; } ;;
  s2) [ -f $PK/${r}_student_stage1 ] || { echo "v2_chain: no stage-1 student in $PK"; exit 3; }; V2_ONLY_S2=1 ;;
  *) echo "v2_chain.sh: V2_STAGE all|teacher|s1|s2" >&2; exit 2 ;;
esac
[ -n "${V2_ONLY_S2:-}" ] || env ${TFROM:+FE_TEACHER_FROM=$TFROM} FE_TEACHER_KS="$TKS" \
  FE_STUDENT_RUN=v2s1_$TAG FE_STUDENT_TASK=Unitree-$R-Depth-Student-Edge FE_STUDENT_FROM=$V2_STUDENT_FROM \
  FE_STUDENT_ITERS=$S1 FE_STUDENT_LR=${V2_S1_LR:-2e-4} FE_STUDENT_LR_FINAL=1e-4 FE_STUDENT_FREE=14000 FE_STUDENT_CHECK=0 \
  $([ $V2_STAGE = teacher ] && echo FE_TEACHER_ONLY=1) \
  DEPTH_STUDENT_HARD=1 ${AFTER:+FE_AFTER_PID=$AFTER} bash jobs/fe_chain.sh $R || { echo "v2_chain: stage A failed"; exit 1; }
case $V2_STAGE in teacher) [ -f $PK/${r}_teacher ]; exit $? ;; s1) [ -f $PK/${r}_student_stage1 ]; exit $? ;; esac
S1CK=${V2_S2_FROM:-$(cat $PK/${r}_student_stage1)}
env FE_STUDENT_RUN=v2s2_$TAG${V2_S2_SUFFIX:-} FE_STUDENT_TASK=Unitree-$R-Depth-Student-Edge-RL FE_STUDENT_FROM=$PWD/$S1CK \
  FE_STUDENT_ITERS=$S2 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=${V2_S2_BC_FINAL:-1.0} --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  ${S2KS:+FE_STUDENT_KS="$S2KS"} FE_TN_GATE=1 DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 \
  DEPTH_STUDENT_STAND_LONG=1 bash jobs/fe_chain.sh $R
