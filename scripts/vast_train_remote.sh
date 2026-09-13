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
# Checkpoint hygiene for big nets: keep optimiser_state only for the 2 newest checkpoints (resume), in background.
( while true; do ls -dt "$WORK/checkpoints/$RUN_NAME"-*/ 2>/dev/null | tail -n +3 | while read -r d; do [ -d "$d/optimiser_state" ] && rm -rf "$d/optimiser_state" "$d/raw.bin"; done; sleep 120; done ) &
CLEANER=$!
if [[ "${ARCH:-v1}" == "v3" ]]; then
    DATA="$DATA" NET_ID="$RUN_NAME" SB0="${SB0:-40}" SB1="$SB" SB2="${SB2:-60}" SAVE_RATE="${SAVE_RATE:-20}" THREADS="${MAP_THREADS:-12}" \
        L1="${L1:-1024}" LOADER_THREADS=8 BUFFER_MB=4096 OUT_DIR="$WORK/checkpoints" ./target/release/train_v3 2>&1 | tee -a "$WORK/runs/$RUN_NAME/train.log"
else
    DATA="$DATA" NET_ID="$RUN_NAME" SUPERBATCHES="$SB" SAVE_RATE=10 THREADS=4 LOADER_THREADS=8 BUFFER_MB=4096 \
        OUT_DIR="$WORK/checkpoints" ./target/release/trainer 2>&1 | tee -a "$WORK/runs/$RUN_NAME/train.log"
fi
kill $CLEANER 2>/dev/null
echo "=== train end $(date -u +%FT%TZ) exit=${PIPESTATUS[0]}" | tee -a "$WORK/runs/$RUN_NAME/train.log"
