#!/bin/bash
# tf3e_s2.sh (go2-depth 2026-11-05 10:30, t2): the tf3e chain's last step by hand. Its tmux server was killed by mistake
# during the seed-2 check of avg1000 (s0 / s1 PASS); training had ended (model_1499). Same environment as
# tf3e_b2w.sh -> fe_chain.sh: seed 2 again (the killed check's partial outputs moved to logs/checks/stale_tf3e_s2/),
# then on PASS the pick and the export, logged as the chain does (the gate watcher keys on that line).
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
export B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc BLIND_CONS_INIT=back_drift:1.7,descent:2.3
R=B2W r=b2w
source jobs/fe_robot.sh $R || exit 2
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt),wheelstep,wheelstep4,stairspeed2,noslide,noslide2x,wheelahead,bdhold,topair,longexit,endwall
export BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/blind_init_fe/model_$N.pt
L=logs/jobs/fe_$r.txt C=logs/checks
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
reached() { python3 -c "import re,sys; v=[float(x) for x in re.findall(r'reached=([0-9.]+)', open(sys.argv[1]).read())]; print(f'{100*sum(v)/len(v):.1f}' if v else 'nan')" $1; }
A=logs/rsl_rl/b2w_depth_student/fe_edge_tf3e_avg1000/model_1000.pt
P=$C/fe_${r}_student_tf3e_avg1000_s2
mkdir -p $C/stale_tf3e_s2 && mv ${P}_* $C/stale_tf3e_s2/ 2> /dev/null
log "$R student avg1000 s2: checking again by hand (tf3e_s2.sh; the chain's tmux server was killed by mistake)"
bash jobs/fe_check.sh student $R $A 2 $P
v=$(cat ${P}_verdict.txt)
log "$R student avg1000 s2: noop $(reached ${P}_noop.txt) op $(reached ${P}_op.txt) edge noop $(reached ${P}_noopedge.txt) op $(reached ${P}_opedge.txt) speed up/down/margin $($PY jobs/fe_margin.py $R $P) | $v"
[[ "$v" == PASS ]] || { log "$R NO STUDENT PASSED: avg1000 s2 failed (run logs/rsl_rl/b2w_depth_student/2026-11-05_08-38-45_fe_edge_tf3e kept)"; exit 1; }
echo "$A" > logs/picks_tb3e/${r}_student
X=logs/exports/${r}_fe_student_tf3e_avg1000.onnx
$PY scripts/go2_depth/export_student.py --task Unitree-$R-Depth-Student-Edge --checkpoint $A --out $X > logs/jobs/fe_${r}_export.log 2>&1 \
  && log "$R STUDENT PICK $A (seeds 0 1 2 pass); exported $X (md5 $(md5sum < $X | cut -c1-8))" \
  || log "$R STUDENT PICK $A; export FAILED: $(tail -n 3 logs/jobs/fe_${r}_export.log)"
