#!/bin/bash
# smoke_chain.sh R [TAG] (go2-depth 2026-11-11, the user: "每次正式开跑前，用很少的机器人、每个阶段几步，把整条链走一遍：T→M→P 的
# 交接、扩展模型、打包、导出、tinynav 请求、深度老师和学生，都要走到。为了让流程走完，测试时检查强制判为通过"). One robot's four
# rounds (jobs/full_chain.sh R TAG all) with CHAIN_SMOKE=1:
#   - 64 robots;
#   - T, M and P 60 iterations each, checks every 20 from +20, checkpoints every 10;
#   - the depth teacher 41 iterations (checks at +20 / +40), stage 1 / stage 2 students 40 iterations;
#   - every check jobs/fe_check.sh's smoke subset with the verdict PASS;
#   - the tinynav request answered by the workstation's watcher after the export and the import.
# Before any of it, blind_chain's input check (scripts/blind/obs_dims.py). Logs in logs/jobs/smoke/, nothing in the
# formal logs; the queue's insert file and logs/jobs/repro_tag_<r> are left alone. A tool fault ends it (exit 4).
# Prints what each step reached, then removes this TAG's training runs, packages, checks, picks and requests
# (SMOKE_KEEP=1 keeps them). TAG starts with smoke (default smoke<MMDDHHMM>). Exit = full_chain's (0: walked through).
set -u
R=$1 TAG=${2:-smoke$(date +%m%d%H%M)}
r=${R,,}
[[ $TAG == smoke* ]] || { echo "smoke_chain.sh: TAG must start with smoke" >&2; exit 2; }
cd /root/gpufree-data/go2-depth/depth_fe || exit 2
S=logs/jobs/smoke
mkdir -p $S
t0=$(date +%s)
CHAIN_SMOKE=1 BL_CURVE_EVERY=60 REPRO_LOG=$S/repro_$r.txt BL_LOG=$S/blind_$r.txt FE_LOG=$S/fe_$r.txt bash jobs/full_chain.sh $R $TAG all
rc=$?
echo "== smoke $R $TAG: full_chain exit $rc after $(( ($(date +%s) - t0) / 60 )) min"
step() { local f=$1 what=$2 pat=$3 line; line=$(grep -E "$pat" $f 2> /dev/null | tail -n 1 | cut -c1-200); printf '  %-24s %s\n' "$what" "${line:-MISSING}"; }
step $S/blind_$r.txt "inputs T / M / P" "blind inputs \(actor"
step $S/blind_$r.txt "T pick" "blind T PICK"
step $S/blind_$r.txt "M pick" "blind M PICK"
step $S/blind_$r.txt "widened (if needed)" "checkpoint widened|blind inputs: (M|P) adds|blind inputs \\(actor"
step $S/blind_$r.txt "P tinynav request" "P avg[0-9]+ tinynav"
step $S/blind_$r.txt "blind package" "BLIND DELIVERY"
step $S/fe_$r.txt "teacher pick" "TEACHER PICK"
step $S/fe_$r.txt "stage-1 student" "STAGE-1 STUDENT"
step $S/fe_$r.txt "stage-2 tinynav request" "student avg[0-9]+ tinynav"
step $S/fe_$r.txt "student export" "STUDENT PICK"
# the blind videos run beside the depth round (blind_videos.sh): wait for them (up to 40 min) before cleaning up
for i in $(seq 80); do pf=$(ls logs/pkg/${r}_bl2_${TAG}_*/videos.pid 2> /dev/null | head -n 1); [ -n "$pf" ] && kill -0 $(cat $pf) 2> /dev/null || break; sleep 30; done
step $S/blind_$r.txt "blind videos" "BLIND VIDEOS"
step $S/repro_$r.txt "tool faults" "TOOL FAULT"
if [ "${SMOKE_KEEP:-0}" != 1 ]; then
  for d in logs/rsl_rl/*/*_${TAG} logs/rsl_rl/*/*_${TAG}_* logs/rsl_rl/*/*${TAG}_avg* logs/rsl_rl/*/bl2?_${TAG}_* \
           logs/pkg/${r}_bl2_${TAG}_* logs/picks_v2_$TAG logs/accept/*${TAG}* logs/checks/*${TAG}* logs/exports/*${TAG}* \
           logs/probe/*${TAG}*; do
    [[ $d == *smoke* ]] && [ -e "$d" ] && rm -rf -- "$d"
  done
  echo "  removed this TAG's runs, packages, checks and requests (SMOKE_KEEP=1 keeps them)"
fi
exit $rc
