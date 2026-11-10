#!/bin/bash
# walled_cmp.sh (go2-depth 2026-11-05 10:45, t1): top_flat_eval on map3-like flights only (walled_up, difficulty 0.7-1.0,
# with longexit + endwall) for teachers and students, two at a time; lines to logs/checks/top_flat.txt (names *_wu).
cd /root/gpufree-data/go2-depth/depth_fe
A="--only walled_up --difficulty 0.7 1.0"
T=logs/rsl_rl
run() { "$@" > /dev/null 2>&1; }
# B2W pair first: tinynav top 0.8 (tf3e) vs 2.0 (tf3d) validates the measure
TF_ARGS="$A" TF_EXTRA=,topair,longexit,endwall TASK=Unitree-B2W-Depth-Student-Edge run bash jobs/top_flat_student.sh B2W $T/b2w_depth_student/fe_edge_tf3e_avg1000/model_1000.pt b2w_s_tf3e_1000_wu &
TF_ARGS="$A" TF_EXTRA=,topair,longexit,endwall TASK=Unitree-B2W-Depth-Student-Edge run bash jobs/top_flat_student.sh B2W $T/b2w_depth_student/fe_edge_tf3d_avg1500/model_1500.pt b2w_s_tf3d_1500_wu &
wait
TF_ARGS="$A" TF_EXTRA=,topair,longexit,endwall run bash jobs/top_flat.sh Go2W $T/go2w_depth_teacher/tf_v4_avg24600/model_24600.pt go2w_t4_24600_wu &
TF_ARGS="$A" TF_EXTRA=,topair,longexit,endwall run bash jobs/top_flat.sh Go2W $T/go2w_depth_teacher/tf_v5_avg26200/model_26200.pt go2w_t5_26200_wu &
wait
TF_ARGS="$A" TF_EXTRA=,topair,longexit,endwall TASK=Unitree-Go2W-Depth-Student-Edge run bash jobs/top_flat_student.sh Go2W $T/go2w_depth_student/fe_edge_tf4p_avg1499/model_1499.pt go2w_s_tf4p_1499_wu &
TF_ARGS="$A" TF_EXTRA=,topairup,longexit,endwall run bash jobs/top_flat.sh As2W $T/as2w_depth_teacher/tf_v6_avg16200/model_16200.pt as2w_t6_16200_wu &
wait
TF_ARGS="$A" TF_EXTRA=,topairup,longexit,endwall TASK=Unitree-As2W-Depth-Student-Edge run bash jobs/top_flat_student.sh As2W $T/as2w_depth_student/fe_edge_tf6p_avg1000/model_1000.pt as2w_s_tf6p_1000_wu &
wait
echo "$(date '+%m-%d %H:%M') walled_cmp done" >> logs/checks/top_flat.txt
