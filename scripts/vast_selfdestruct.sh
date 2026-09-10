#!/usr/bin/env bash
# Runs ON THE BOX, detached, armed BEFORE training starts. Destroys this very instance through the
# vast.ai API when (a) the training log says training ended and a grace period for the laptop's
# checkpoint sync has passed, or (b) a hard wall-clock limit is reached, whichever comes first.
# It needs nothing from the laptop: no ssh, no sync loop, no human. If the laptop dies, this still fires.
#
# Usage: vast_selfdestruct.sh <INSTANCE_ID> <API_KEY> <MAX_HOURS> <GRACE_MINUTES> <TRAIN_LOG>
set -uo pipefail
ID="${1:?instance id}"; KEY="${2:?api key}"; MAX_HOURS="${3:?max hours}"; GRACE_MIN="${4:?grace minutes}"; TRAIN_LOG="${5:?train log}"
LOG=/workspace/selfdestruct.log
START=$(date +%s)
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }
destroy() {
    log "DESTROYING instance $ID: $1"
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        code=$(curl -s -o /workspace/selfdestruct_resp.txt -w '%{http_code}' -X DELETE \
            -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{}' \
            "https://console.vast.ai/api/v0/instances/$ID/")
        log "attempt $attempt http $code $(head -c 200 /workspace/selfdestruct_resp.txt)"
        [[ "$code" == "200" ]] && exit 0
        sleep 30
    done
    log "all destroy attempts failed"
}
log "armed: id=$ID max_hours=$MAX_HOURS grace_min=$GRACE_MIN log=$TRAIN_LOG"
while true; do
    now=$(date +%s)
    elapsed_h=$(( (now - START) / 3600 ))
    if (( now - START >= MAX_HOURS * 3600 )); then
        destroy "hard wall-clock limit ${MAX_HOURS}h reached"
    fi
    if grep -q "=== train end" "$TRAIN_LOG" 2>/dev/null; then
        log "training ended; waiting ${GRACE_MIN} min for the laptop to pull the final checkpoint"
        sleep $(( GRACE_MIN * 60 ))
        destroy "training finished + grace period"
    fi
    sleep 60
done
