#!/bin/bash
# m_probe.sh TAG DROP_KEYS ITERS (go2-depth 2026-11-10): a short Go2 mixed-phase probe from T avg4000 straight through
# scripts/train.py (blind_chain.sh kills a training whose checks lie past its end), the env as blind_chain.sh sets it
# for M (package pilot + KEEP + BL_EXTRA, minus DROP_KEYS), then term_diag.py on the sideways start at 4400 / 4600 / 4700.
set -u
TAG=$1 DROP=$2 IT=${3:-700}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh Go2 > /dev/null; export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=0
source jobs/fe_robot.sh Go2 > /dev/null
PP=$(grep 'blind chain o2: pilot ' logs/jobs/blind_go2.txt | head -1 | sed 's/.*chain o2: pilot //; s/; polish adds.*//')
for k in ${DROP//,/ }; do PP=$(echo "$PP" | tr ',' '\n' | grep -v -x "$k" | paste -sd,); done
export BLIND_PILOT=$PP BLIND_ANCHOR_COEFF=0 BLIND_PHASE_START_IT=4000 BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/go2_blind/bl2T_o2_avg4000/model_4000.pt
L=logs/jobs/blind_go2.txt
echo "$(date '+%m-%d %H:%M:%S') Go2 M probe $TAG: $IT it from T avg4000 (bl2T_o2 model_4000), pilot o2 minus [$DROP]" >> $L
$PY scripts/train.py Unitree-Go2-Blind-Mixed --env.scene.num-envs=4096 --agent.run-name=bl2Mp_$TAG --agent.max-iterations=$IT \
  --agent.resume=True --agent.load-run='^2026-11-09_19-17-41_bl2T_o2$' --agent.load-checkpoint='^model_4000\.pt$' > logs/jobs/blind_go2_bl2Mp_$TAG.log 2>&1
RD=$(ls -td logs/rsl_rl/go2_blind/*_bl2Mp_$TAG | head -1)
for k in 4400 4600 4700; do
  [ -f $RD/model_$k.pt ] || continue
  r=$(flock /root/gpufree-data/go2-depth/ws_check.lock $PY jobs/term_diag.py --task Unitree-Go2-Blind-Flat-Scratch $RD/model_$k.pt \
      --cmd 0,0.3,0 --seconds 4 --ramp 2 --settle 1.5 --num-envs 16 2>&1 | grep '^cmd' | sed 's/^cmd 0,0.3,0: //')
  echo "$(date '+%m-%d %H:%M:%S') Go2 M probe $TAG model_$k sideways start: $r" >> $L
done
