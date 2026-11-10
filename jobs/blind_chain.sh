#!/bin/bash
# blind_chain.sh R TAG [AFTER_PID] (go2-depth 2026-11-05, the new pipeline's blind stage; the user: "先训练轮式的盲狗",
# from zero ("为什么不从0开始训练"), one robot first, early stop ("如果提前通过验收，就不要继续训练了")). go2-home's phased
# blind recipe (blind_env_cfg.apply_blind_phased) from scratch with the delivery's pilot (v2_robot.sh V2_PKG) plus the
# blind-safe v2 keys (V2_BLIND_EXTRA: payload to the documented limit, per-wheel stairs terms; no endwall / teachblur,
# which need eyes / a scan). Each phase stops at its first passing check (avg of the last 8 checkpoints, seed 0):
#   T  Unitree-<R>-Blind-Teacher from zero, <= 5000 it; flat goals only, drift not judged (FE_CHECK_FLAT=1) every 500
#      from 2000;
#      payload ramped in over the first 3000 it
#   M  Unitree-<R>-Blind-Mixed from T's pick (BLIND_PHASE_START_IT = its iteration), stairs 8 -> 65% over 3000 it,
#      <= +5000; the whole acceptance every 500 from +3000. Not anchored (BLIND_ANCHOR_COEFF=0, go2-home's recipe; the
#      runner still gets T's pick as BLIND_ANCHOR_CHECKPOINT). Go2W climbs late: 0% up at +2000 / +3000 is normal there
#      (go2-home: onset ~3000-4000 it into mixed), so judge M by wheel_riser_lift / foothold_height_gain, not one check.
#   P  Unitree-<R>-Blind-Polish from M's pick, <= +2000, not anchored either; the whole acceptance on seeds 0 1 2 every
#      500 from +500, then the tinynav acceptance (blind ONNX; FE_TN_GATE via check/tn_gate_watch.sh)
# Restarts: BL_T_PICK (+ BL_T_RUN) skips T; BL_T_FROM_RUN + BL_T_FROM_K resume T; BL_M_FROM_RUN + BL_M_FROM_K (with
# BL_T_PICK) resume M from a raw checkpoint; BL_M_PICK + BL_M_RUN (with BL_T_PICK) skip M; BL_M_END sets M's last iteration;
# BL_M_ITERS / BL_P_ITERS (2026-11-11, jobs/recipes RECIPE_M_ITERS / RECIPE_P_ITERS): M's / P's length after the pick
# before them (default 5000 / 2000);
# BL_P_EXTRA adds polish keys; BL_EXTRA adds keys to every phase run; BL_DROP removes keys; BL_ANCHOR_COEFF overrides 0;
# BL_NUM_ENVS (4096).
# then the delivery package logs/pkg/<r>_bl2_<TAG>_<K> (model, policy.onnx, deploy.yaml, blind_pilot.txt, md5s) for
# the depth chain (v2_robot.sh V2_PKG). Log: logs/jobs/blind_<r>.txt.
set -u
R=$1 TAG=$2 AFTER=${3:-}
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
source jobs/v2_robot.sh $R || exit 2
# BL_CONS (2026-11-10, jobs/full_chain.sh): the recipe's constraint multipliers at the T and M starts ("" = all at 1)
[ -n "${BL_CONS+x}" ] && V2_CONS=$BL_CONS
export $V2_PKG_VAR=$V2_PKG
V2_BLIND_EXTRA=$([ $V2_KIND = wheeled ] && echo $V2_WHEELED_BLIND_EXTRA || echo $V2_LEGGED_BLIND_EXTRA)
# T / M without the stairs-quality keys (V2_STAIR_KEYS), P with them (see v2_robot.sh)
KEEP="" POLISH_KEYS=""
for k in ${V2_BLIND_EXTRA//,/ } ${V2_ROBOT_EXTRA//,/ }; do
  if [[ " $V2_STAIR_KEYS " == *" $k "* ]]; then POLISH_KEYS="$POLISH_KEYS,$k"; else KEEP="$KEEP,$k"; fi
done
# BL_EXTRA: more keys for every phase this run trains (11-06: sidecmd for Go2W's M / P, the B2W fix of the sideways
# overshoot: with the side-drift multiplier at its cap the robots walked sideways 125-128% of the command)
export BLIND_PILOT=$(cat $V2_PKG/blind_pilot.txt)$KEEP${BL_EXTRA:+,$BL_EXTRA}
# BL_PILOT_TM / BL_PILOT_P (2026-11-10, jobs/full_chain.sh with jobs/recipes/<r>.sh): the T / M and the P pilot given
# whole, not built from an earlier package's blind_pilot.txt (the reproducible run)
[ -n "${BL_PILOT_TM:-}" ] && { export BLIND_PILOT=$BL_PILOT_TM; POLISH_KEYS=""; }
# BL_DROP (11-07): keys taken out of the package's recipe (e.g. the Go2W leg hold: loaded robots fell standing with it)
drop_keys() { local p=$1 k; for k in ${BL_DROP:+${BL_DROP//,/ }}; do p=$(echo "$p" | tr ',' '\n' | grep -v -x "$k" | paste -sd,); done; echo "$p"; }
export BLIND_PILOT=$(drop_keys "$BLIND_PILOT")
export FE_LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock}
# fe_robot.sh: the payload limit / wall stop (BLIND_PAYLOAD with the teacher phase's ramp: 3000 it of 24 steps)
export PAYLOAD_RAMP=72000
source jobs/fe_robot.sh $R > /dev/null || exit 2
unset BLIND_CONS_INIT BLIND_PHASE_START_IT BLIND_ANCHOR_CHECKPOINT BLIND_ANCHOR_COEFF
[ -n "$V2_CONS" ] && export BLIND_CONS_INIT=$V2_CONS
L=logs/jobs/blind_$r.txt C=logs/checks TD=logs/rsl_rl/${r}_blind
mkdir -p logs/jobs $C logs/pkg
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
new_run() { ls -d $TD/*_$1 2> /dev/null | sort | tail -n 1; }
# new_run_wait RUN PID BEFORE: the run dir the training just launched writes (other than BEFORE, the newest one of that
# name before the launch), waiting up to 20 min while it lives. 2026-11-10: a fixed 120 s missed B2 rp3's M, whose dir
# came later; the empty dir then ended the phase ("$3: unbound variable") and the training went on unsupervised.
new_run_wait() { local d w=0; while [ $w -lt 1200 ]; do d=$(new_run $1); [ -n "$d" ] && [ "$d" != "$3" ] && { echo $d; return 0; }; kill -0 $2 2> /dev/null || return 1; sleep 15; w=$((w + 15)); done; return 1; }
stop_train() { kill -INT $1 2> /dev/null; for _ in $(seq 60); do kill -0 $1 2> /dev/null || return 0; sleep 2; done; kill $1 2> /dev/null; }
last8() {
  local k0=$(( $2 / 100 * 100 )) out=""
  for k in $(seq $((k0 - 700 + ($2 % 100 ? 100 : 0))) 100 $k0); do [ -f $1/model_$k.pt ] && out="$out $1/model_$k.pt"; done
  [ $(( $2 % 100 )) -ne 0 ] && out="$out $1/model_$2.pt"
  echo $out
}
wait_ck() { until [ -f $1/model_$2.pt ]; do kill -0 $3 2> /dev/null || { sleep 5; [ -f $1/model_$2.pt ]; return; }; sleep 30; done; sleep 10; }
log "$R blind chain $TAG: pilot $BLIND_PILOT; polish adds $POLISH_KEYS"
if [ -n "$AFTER" ]; then while kill -0 $AFTER 2> /dev/null; do sleep 30; done; log "$R pid $AFTER ended"; fi

# phase NAME TASK RUN FROM_RUN FROM_K MAX_IT FIRST_CHECK MODE SEEDS -> PICK (avg checkpoint path) on stdout's last line
phase() {
  local name=$1 task=$2 run=$3 frun=$4 fk=$5 maxit=$6 first=$7 mode=$8 seeds=$9
  # BL_NUM_ENVS (11-06): fewer robots to share a GPU (Go2W g16 at 2048 beside B2 l5's 12.4 GB on t1)
  local args="--env.scene.num-envs=${BL_NUM_ENVS:-4096} --agent.run-name=$run --agent.max-iterations=$maxit"
  [ -n "$frun" ] && args="$args --agent.resume=True --agent.load-run=^$(basename $frun)\$ --agent.load-checkpoint=^model_$fk\.pt\$"
  local before=$(new_run $run) rd=""
  $PY scripts/train.py $task $args > logs/jobs/blind_${r}_$run.log 2>&1 &
  local tp=$!
  log "$R $name started (pid $tp; $task, $maxit it from ${frun:-zero}${fk:+ model_$fk}, checks $mode from +$first every 500, seeds $seeds)"
  rd=$(new_run_wait $run $tp "$before") || {
    log "$R $name: no new run dir for $run (training $(kill -0 $tp 2> /dev/null && echo alive || echo ended)): phase stopped"
    stop_train $tp; echo ""; return; }
  local base=${fk:-0} pick=""
  for K in $(seq $((base + first)) 500 $((base + maxit - 1))); do
    wait_ck "$rd" $K $tp || { log "$R $name ended before model_$K"; break; }
    local A=$TD/${run}_avg$K/model_$K.pt ok=1 lucky=""
    mkdir -p $(dirname $A)
    $PY jobs/avg_ckpt.py $A $(last8 $rd $K) >> $L 2>&1 || cp $rd/model_$K.pt $A
    for SEED in $seeds; do
      local P=$C/bl_${r}_${run}_avg${K}_s$SEED
      FE_CHECK_FLAT=$([ $mode = flat ] && echo 1 || echo 0) bash jobs/fe_check.sh blind $R $A $SEED $P
      local v=$(cat ${P}_verdict.txt 2> /dev/null)
      log "$R $name avg$K s$SEED ($mode): $v"
      # M (11-06): stairs speed is the polish phase's job (the stairspeed keys are polish keys), so a mixed check whose
      # only miss is 上楼慢 / 下楼慢 passes M; polish is checked on the whole acceptance (3 seeds + tinynav).
      if [ $name = M ] && [[ "$v" == FAIL:* ]]; then
        local rest=$(echo "${v#FAIL: }" | sed "s/；/\n/g" | grep -v -x -E "楼梯：不通过：(上楼慢|下楼慢)(、(上楼慢|下楼慢))?" | grep -c .)
        [ "$rest" = 0 ] && { log "$R $name avg$K s$SEED: only stairs speed misses -> M pass (polish adds the stairspeed keys)"; v=PASS; }
      fi
      if [[ "$v" != PASS ]]; then
        # luck rule (2026-11-10, doc/acceptance.md): one seed of several may fail on a single fall alone
        if [ -z "$lucky" ] && [ $(echo $seeds | wc -w) -gt 1 ] && [[ "$($PY jobs/luck_pass.py $R $P)" == ELIGIBLE* ]]; then
          lucky=$SEED; log "$R $name avg$K s$SEED: a single fall only -> luck rule, the other seeds decide"; continue
        fi
        ok=0; break
      fi
    done
    [ $ok = 1 ] && [ -n "$lucky" ] && log "$R $name avg$K: seeds pass by the luck rule (seed $lucky: one fall)"
    if [ $ok = 1 ] && [ $name = P ]; then
      mkdir -p logs/accept
      local Q=logs/accept/bl_${r}_${run}_$K
      echo "$R $A BLIND $BLIND_PILOT" > $Q.req   # the export needs the recipe (it decides the actor inputs)
      local w=0
      until [ -f $Q.verdict ] || [ $w -ge 5400 ]; do sleep 30; w=$((w + 30)); done
      local tv=$(cat $Q.verdict 2> /dev/null || echo "no verdict")
      log "$R $name avg$K tinynav: $tv"
      [[ "$tv" == PASS* ]] || ok=0
    fi
    if [ $ok = 1 ]; then pick=$A; break; fi
  done
  stop_train $tp
  echo "$pick"
}

# BL_T_PICK=<T average .pt> (with BL_T_RUN, default the newest *_bl2T_<TAG> run): skip T, start at M from it.
# BL_T_MAX (2026-11-10, jobs/recipes RECIPE_T_MAX): T's last iteration (default 5000)
TMAX=${BL_T_MAX:-5000}
if [ -n "${BL_T_PICK:-}" ]; then
  T=$BL_T_PICK
  log "$R blind T PICK (given${BL_T_PICK_WHY:+: $BL_T_PICK_WHY}) $T"
else
  # BL_T_FROM_RUN + BL_T_FROM_K: T resumes from that raw checkpoint (to 5000 in all, checks every 500 from max(K+500, 2000))
  if [ -n "${BL_T_FROM_RUN:-}" ]; then
    TF0=$BL_T_FROM_K; TFIRST=$(( TF0 + 500 >= 2000 ? 500 : 2000 - TF0 ))
    T=$(phase T Unitree-$R-Blind-Teacher bl2T_$TAG $BL_T_FROM_RUN $TF0 $((TMAX - TF0)) $TFIRST flat 0 | tail -n 1)
  else
    T=$(phase T Unitree-$R-Blind-Teacher bl2T_$TAG "" "" $TMAX 2000 flat 0 | tail -n 1)
  fi
  if [ -z "$T" ]; then
    # 2026-11-10 (the user: T at its limit without a pass hands its last check to M, as M does to P; P's whole
    # acceptance -- 3 seeds + tinynav -- decides)
    T=$(ls $TD/bl2T_${TAG}_avg*/model_*.pt 2> /dev/null | sort -V | tail -n 1)
    [ -n "$T" ] || { log "$R blind T: no checkpoint checked by $TMAX"; exit 1; }
    log "$R blind T: no checkpoint passed the flat goals by $TMAX -> M from the last check $T"
  fi
  log "$R blind T PICK $T"
fi
TK=$(basename $T .pt | cut -d_ -f2)
TRUN=${BL_T_RUN:-$(new_run bl2T_$TAG)}
export PAYLOAD_RAMP=0
source jobs/fe_robot.sh $R > /dev/null   # BLIND_PAYLOAD again, now without the ramp
# after fe_robot.sh, which unsets BLIND_PHASE_START_IT (11-05: the first Go2W M ran its stairs ramp from 5000, not 3500)
export BLIND_PHASE_START_IT=$TK BLIND_ANCHOR_CHECKPOINT=$PWD/$T
# No anchor (11-06, go2-home's reply): their mixed and polish ran with BLIND_ANCHOR_COEFF=0 (the delivery README: "also
# BLIND_ANCHOR_COEFF=0"); anchor.py's PhasedAnchoredRunner defaults to 1.0. Anchored to the flat teacher, their Go2W
# mixed parked at the first riser on four wheels (0% up in every band, no termination), as our Go2W g2 / g3 M did, and
# their legged mixed lost its climb (12-15 cm up 62 -> 8%). The runner still needs the checkpoint path.
export BLIND_ANCHOR_COEFF=${BL_ANCHOR_COEFF:-0}
unset BLIND_CONS_INIT; [ -n "$V2_CONS" ] && export BLIND_CONS_INIT=$V2_CONS
log "$R blind M/P anchor coeff $BLIND_ANCHOR_COEFF (checkpoint $T)"
MEND=${BL_M_END:-$((TK + ${BL_M_ITERS:-5000}))}   # BL_M_END: M's last iteration (default T's pick + BL_M_ITERS)
if [ -n "${BL_M_PICK:-}" ]; then
  M=$BL_M_PICK MRUN=$BL_M_RUN
  log "$R blind M PICK (given) $M"
else
  if [ -n "${BL_M_FROM_RUN:-}" ]; then
    # resume M from that raw checkpoint: the stairs ramp still counts from T's pick, M ends at T's pick + 5000 in all
    MF0=$BL_M_FROM_K; MFIRST=$(( MF0 + 500 >= TK + 3000 ? 500 : TK + 3000 - MF0 ))
    M=$(phase M Unitree-$R-Blind-Mixed bl2M_$TAG $BL_M_FROM_RUN $MF0 $((MEND - MF0)) $MFIRST full 0 | tail -n 1)
  else
    M=$(phase M Unitree-$R-Blind-Mixed bl2M_$TAG $TRUN $TK $((MEND - TK)) 3000 full 0 | tail -n 1)
  fi
  if [ -z "$M" ]; then
    # 2026-11-10 (doc/acceptance.md, the user: the mixed phase's handoff as a rule): M at its end without a pass hands
    # its last check to P, which is judged on the whole acceptance (3 seeds + tinynav)
    M=$(ls $TD/bl2M_${TAG}_avg*/model_*.pt 2> /dev/null | sort -V | tail -n 1)
    [ -n "$M" ] || { log "$R blind M: no checkpoint checked by $MEND"; exit 1; }
    log "$R blind M: no checkpoint passed by $MEND -> P from the last check $M"
  fi
  MRUN=$(new_run bl2M_$TAG)
  log "$R blind M PICK $M"
fi
MK=$(basename $M .pt | cut -d_ -f2)
export BLIND_PILOT=$BLIND_PILOT$POLISH_KEYS${BL_P_EXTRA:+,$BL_P_EXTRA}   # BL_P_EXTRA: more polish keys (e.g. sidew6l)
[ -n "${BL_PILOT_P:-}" ] && export BLIND_PILOT=$BL_PILOT_P   # the recipe's whole P pilot (2026-11-10)
export BLIND_PILOT=$(drop_keys "$BLIND_PILOT")
log "$R blind P pilot $BLIND_PILOT"
PIT=${BL_P_ITERS:-2000}
P=$(phase P Unitree-$R-Blind-Polish bl2P_$TAG $MRUN $MK $PIT 500 full "0 1 2" | tail -n 1)
[ -n "$P" ] || { log "$R blind P: no checkpoint passed by +$PIT"; exit 1; }
PK=$(basename $P .pt | cut -d_ -f2)
log "$R blind P PICK $P"
OUT=logs/pkg/${r}_bl2_${TAG}_$PK
mkdir -p $OUT
cp $P $OUT/model_$PK.pt && { cp -r $(new_run bl2P_$TAG)/params $OUT/ 2> /dev/null || cp -r $V2_PKG/params $OUT/ 2> /dev/null; }
echo "$BLIND_PILOT" | sed 's/^,//' > $OUT/blind_pilot.txt
md5sum $OUT/model_$PK.pt | sed "s|$OUT/||" > $OUT/checkpoint.md5
$PY scripts/go2_blind/export_onnx.py --task Unitree-$R-Blind-Flat-Scratch $OUT/model_$PK.pt --out $OUT > logs/jobs/blind_${r}_export.log 2>&1 \
  && $PY scripts/blind/export_deploy_yaml.py Unitree-$R-Blind-Polish --robot $r --out $OUT/deploy.yaml >> logs/jobs/blind_${r}_export.log 2>&1 \
  && log "$R BLIND DELIVERY $OUT (model md5 $(md5sum < $OUT/model_$PK.pt | cut -c1-8), policy.onnx $(md5sum < $OUT/policy.onnx | cut -c1-8))" \
  || log "$R blind export FAILED: $(tail -n 3 logs/jobs/blind_${r}_export.log)"
