#!/bin/bash
# export_policy.sh KIND R CK OUTDIR (go2-depth 2026-11-11, the user's unification item 3: one package format and one
# export for blind and depth). Exports checkpoint CK of robot R with the environment it trained with into OUTDIR:
#   policy.onnx      blind: scripts/go2_blind/export_onnx.py on Unitree-<R>-Blind-Flat-Scratch (the deploy contract)
#                    depth: scripts/go2_depth/export_student.py on Unitree-<R>-Depth-Student-Edge (contract metadata)
#   deploy.yaml      scripts/blind/export_deploy_yaml.py on Unitree-<R>-Blind-Polish (robot_ctrl's config)
#   run_env.sh       the environment it was exported with (the run's, its payload ramp stripped)
#   blind_pilot.txt  BLIND_PILOT (tinynav-sim's import_blind_policy.py takes the contact solver from "grip"), and
#   blind_params.txt BLIND_PARAMS, for the readers of the older packages
#   manifest.json    kind, robot, checkpoint (path, md5, iteration), the export's md5s, code commit, time
# The environment: the checkpoint's run_env.sh (scripts/train.py writes it into every run from 2026-11-11; averages and
# packages carry it; jobs/run_env.sh finds it). A checkpoint from before has none: the caller's environment is used as
# it is (the chains and the tinynav gate set it up the old way before calling). Exit 0, 1 when an export failed.
set -u
KIND=$1 R=$2 CK=$3 OUT=$4
r=${R,,}
cd "${ACCEPT_ROOT:-/root/gpufree-data/go2-depth/depth_fe}" || exit 2
if [ -z "${PY:-}" ] && [ -f env.sh ]; then source env.sh > /dev/null 2>&1; fi
PY=${PY:-python3}
[ -f "$CK" ] || { echo "export_policy.sh: no checkpoint $CK" >&2; exit 2; }
if E=$(bash jobs/run_env.sh "$CK"); then
  source "$E"
  ENVSRC=$E
else
  ENVSRC="the caller's environment (no run_env.sh)"
fi
export BLIND_PAYLOAD=$(echo "${BLIND_PAYLOAD:-}" | sed -E 's/,ramp=[0-9]+//')
export BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$([[ $CK = /* ]] && echo $CK || echo $PWD/$CK)
mkdir -p "$OUT"
LOGF=$OUT/export.log
: > $LOGF
case $KIND in
  blind)
    $PY scripts/go2_blind/export_onnx.py --task Unitree-$R-Blind-Flat-Scratch "$CK" --out "$OUT" >> $LOGF 2>&1 ;;
  depth)
    $PY scripts/go2_depth/export_student.py --task Unitree-$R-Depth-Student-Edge --checkpoint "$CK" --out "$OUT/policy.onnx" >> $LOGF 2>&1 ;;
  *) echo "export_policy.sh: KIND blind or depth" >&2; exit 2 ;;
esac
[ -s "$OUT/policy.onnx" ] || { echo "export_policy.sh: no policy.onnx ($(tail -n 2 $LOGF | tr '\n' ' '))" >&2; exit 1; }
$PY scripts/blind/export_deploy_yaml.py Unitree-$R-Blind-Polish --robot $r --out "$OUT/deploy.yaml" >> $LOGF 2>&1 \
  || { echo "export_policy.sh: deploy.yaml failed ($(tail -n 2 $LOGF | tr '\n' ' '))" >&2; exit 1; }
{ echo "# run_env.sh as exported ($ENVSRC; payload ramp stripped)"
  for k in $(compgen -e | grep -E '^(BLIND_|DEPTH_STUDENT_|UNIFIED_TASKS|U_SMOKE|TRAIN_LOG_LAYOUT)' | sort); do printf 'export %s=%q\n' $k "${!k}"; done
  [ -n "${RUN_TASK:-}" ] && printf 'RUN_TASK=%q\nRUN_ARGV=%q\n' "$RUN_TASK" "${RUN_ARGV:-}"; } > "$OUT/run_env.sh.new" \
  && mv "$OUT/run_env.sh.new" "$OUT/run_env.sh"
echo "${BLIND_PILOT:-}" | sed 's/^,//' > "$OUT/blind_pilot.txt"
echo "${BLIND_PARAMS:-}" > "$OUT/blind_params.txt"
"$PY" - "$KIND" "$R" "$CK" "$OUT" "$(cat REPO_COMMIT 2> /dev/null || git rev-parse --short HEAD 2> /dev/null)" <<'EOF'
import hashlib, json, os, sys, time
kind, robot, ck, out, commit = sys.argv[1:6]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
it = None
try:
  import torch
  it = int(torch.load(ck, map_location="cpu", weights_only=False).get("iter"))
except Exception:
  pass
m = {"kind": kind, "robot": robot, "checkpoint": {"path": ck, "md5": md5(ck), "iteration": it},
     "policy_onnx_md5": md5(os.path.join(out, "policy.onnx")), "deploy_yaml_md5": md5(os.path.join(out, "deploy.yaml")),
     "code": commit, "exported": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(m, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(f"exported {kind} {robot} {ck} -> {out} (policy.onnx {m['policy_onnx_md5'][:8]}, deploy.yaml {m['deploy_yaml_md5'][:8]})")
EOF
