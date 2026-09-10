#!/usr/bin/env bash
# Runs ON THE LAPTOP in a loop: pulls every new checkpoint from the instance to runs/<run>/checkpoints,
# verifies each one loads in the engine, and logs it. Start it BEFORE training starts, keep it running
# until the instance is destroyed. Losing the instance then costs at most one save interval.
#
# Usage: scripts/vast_sync_checkpoints.sh <run-name> <ssh-host> <ssh-port> [interval-seconds]
set -uo pipefail
RUN="${1:?run name}"; HOST="${2:?ssh host}"; PORT="${3:?ssh port}"; INTERVAL="${4:-120}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/runs/$RUN/checkpoints"; LOG="$ROOT/runs/$RUN/sync.log"
mkdir -p "$DEST"
ENGINE="$ROOT/target/release/engine"
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
log "sync loop start run=$RUN host=$HOST port=$PORT interval=$INTERVAL"
while true; do
    # Only pull the small files we need: the quantised net and the training log. Optimiser state is large.
    if nice -n 19 rsync -az --partial --timeout=60 -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=20" \
        --include='*/' --include='quantised.bin' --include='raw.bin' --include='*.log' --include='*.txt' --exclude='*' \
        "root@$HOST:/workspace/checkpoints/" "$DEST/" >> "$LOG" 2>&1; then
        for q in "$DEST"/*/quantised.bin; do
            [[ -f "$q" ]] || continue
            [[ -f "$q.checked" ]] && continue
            if [[ -x "$ENGINE" ]] && "$ENGINE" netcheck "$q" >> "$LOG" 2>&1; then
                touch "$q.checked"; log "checkpoint OK: $q"
            elif [[ -x "$ENGINE" ]]; then
                log "checkpoint FAILED netcheck: $q"
            else
                log "pulled (engine not built, unchecked): $q"
            fi
        done
    else
        log "rsync failed (instance down or unreachable), retrying in $INTERVAL s"
    fi
    sleep "$INTERVAL"
done
