#!/usr/bin/env bash
# Cost guard: polls the instance, estimates spend from its actual runtime and hourly rate, and DESTROYS it
# (after a final checkpoint pull) when the cap is reached. Runs on the laptop for the whole run.
# Usage: scripts/vast_cost_guard.sh <RUN> <CAP_USD> [poll-seconds]
set -uo pipefail
RUN="${1:?run}"; CAP="${2:?cap usd}"; POLL="${3:-60}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1090
source "$ROOT/runs/$RUN/instance.env"
LOG="$ROOT/runs/$RUN/cost_guard.log"
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
log "guard start run=$RUN instance=$INSTANCE_ID cap=\$$CAP poll=${POLL}s"
while true; do
    info=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || true)
    if [[ -z "$info" || "$info" == "null" ]]; then
        log "instance $INSTANCE_ID no longer exists; guard exiting"; exit 0
    fi
    read -r rate start status <<< "$(echo "$info" | python3 -c "
import sys,json,time
d=json.load(sys.stdin)
rate=float(d.get('dph_total') or 0)+float(d.get('storage_cost') or 0)*float(d.get('disk_space') or 120)/720.0
print(rate, d.get('start_date') or 0, d.get('actual_status') or '')")"
    now=$(date +%s)
    hours=$(python3 -c "print(max(0.0, ($now - float('$start'))/3600.0))")
    spent=$(python3 -c "print(round(float('$hours')*float('$rate') + 0.10, 3))")  # +$0.10 bandwidth/rounding margin
    log "status=$status rate=\$$rate/hr elapsed=${hours}h est_spent=\$$spent"
    echo "$spent" > "$ROOT/runs/$RUN/est_spent.txt"
    if python3 -c "import sys; sys.exit(0 if float('$spent') >= float('$CAP') else 1)"; then
        log "CAP REACHED (\$$spent >= \$$CAP): pulling checkpoints and destroying instance $INSTANCE_ID"
        rsync -az --timeout=120 -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=20" \
            --include='*/' --include='quantised.bin' --include='raw.bin' --include='*.log' --include='*.txt' --exclude='*' \
            root@"$HOST":/workspace/checkpoints/ "$ROOT/runs/$RUN/checkpoints/" >> "$LOG" 2>&1 || log "final rsync failed"
        vastai destroy instance "$INSTANCE_ID" 2>&1 | tee -a "$LOG"
        echo "$(date -u +%FT%TZ) cost guard destroyed instance $INSTANCE_ID at est \$$spent" >> "$ROOT/runs/$RUN/events.log"
        exit 0
    fi
    sleep "$POLL"
done
