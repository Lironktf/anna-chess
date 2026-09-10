#!/usr/bin/env bash
# Runs ON the box inside tmux: the real training run. Usage: vast_train_remote.sh <RUN_NAME> <SUPERBATCHES>
set -euo pipefail
RUN_NAME="${1:?run}"; SB="${2:?superbatches}"
WORK=/workspace; REPO=$WORK/chess
# shellcheck disable=SC1090
source "$WORK/runs/$RUN_NAME/data.env"
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
cd "$REPO/trainer"
echo "=== train start $(date -u +%FT%TZ) run=$RUN_NAME superbatches=$SB data=$DATA" | tee -a "$WORK/runs/$RUN_NAME/train.log"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$WORK/runs/$RUN_NAME/train.log"
DATA="$DATA" NET_ID="$RUN_NAME" SUPERBATCHES="$SB" SAVE_RATE=10 THREADS=4 LOADER_THREADS=8 BUFFER_MB=4096 \
    OUT_DIR="$WORK/checkpoints" ./target/release/trainer 2>&1 | tee -a "$WORK/runs/$RUN_NAME/train.log"
echo "=== train end $(date -u +%FT%TZ) exit=${PIPESTATUS[0]}" | tee -a "$WORK/runs/$RUN_NAME/train.log"
