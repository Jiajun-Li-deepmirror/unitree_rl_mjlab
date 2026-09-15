#!/bin/bash
# Usage: ./play_dance.sh <dance_name> [extra play.py args...]
# e.g.  ./play_dance.sh jump
#       ./play_dance.sh house --video-length 500

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <dance_name> [extra play.py args...]"
  echo "Available dances:"
  ls logs/rsl_rl/ 2>/dev/null | sed -n 's/^g1_//p'
  exit 1
fi

DANCE="$1"
shift

MOTION_FILE="src/assets/motions/g1/test_g1_${DANCE}.npz"

if [ ! -f "$MOTION_FILE" ]; then
  echo "Motion file not found: $MOTION_FILE"
  exit 1
fi

# Experiment folder naming has varied across runs (g1_<name> vs g1_dance_<name>).
LATEST_RUN=""
for EXP_NAME in "g1_${DANCE}" "g1_dance_${DANCE}"; do
  CANDIDATE=$(ls -td "logs/rsl_rl/${EXP_NAME}"/*/ 2>/dev/null | head -1)
  if [ -n "$CANDIDATE" ]; then
    LATEST_RUN="$CANDIDATE"
    break
  fi
done
if [ -z "$LATEST_RUN" ]; then
  echo "No training runs found for experiment: g1_${DANCE} or g1_dance_${DANCE}"
  exit 1
fi

LATEST_CKPT=$(ls -t "${LATEST_RUN}"model_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
  echo "No checkpoints found in: $LATEST_RUN"
  exit 1
fi

echo "[play_dance] motion=${DANCE}  checkpoint=${LATEST_CKPT}"

python3 scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file="${MOTION_FILE}" \
  --checkpoint_file="${LATEST_CKPT}" \
  --num-envs 1 \
  --video True --video-length 500 \
  "$@"
