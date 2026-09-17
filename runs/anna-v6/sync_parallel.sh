#!/usr/bin/env bash
# anna-v6 sync loop for the throttled link: every INTERVAL s pull the newest checkpoint's quantised.bin over 12 ssh streams, netcheck it.
cd /home/liron/Desktop/chess; source runs/anna-v6/instance.env; LOG=runs/anna-v6/sync.log
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }
log "parallel sync loop start"
while true; do
  newest=$(timeout 40 ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20 root@"$HOST" 'ls -dt /workspace/checkpoints/anna-v6-*/ 2>/dev/null | head -1' 2>/dev/null | grep -vE "^Welcome|^Have fun" | tr -d '/ ' | xargs -r basename)
  if [[ -n "$newest" && ! -f "runs/anna-v6/checkpoints/$newest/quantised.bin.checked" ]]; then
    mkdir -p "runs/anna-v6/checkpoints/$newest"
    out=$(scripts/vast_pull_parallel.sh "$HOST" "$PORT" "/workspace/checkpoints/$newest/quantised.bin" "runs/anna-v6/checkpoints/$newest/quantised.bin" 12 2>&1 | grep -vE "^Welcome|^Have fun" | tail -1)
    log "$newest: $out"
    if [[ -f "runs/anna-v6/checkpoints/$newest/quantised.bin" ]] && ./target/release/engine netcheck "runs/anna-v6/checkpoints/$newest/quantised.bin" >/dev/null 2>&1; then touch "runs/anna-v6/checkpoints/$newest/quantised.bin.checked"; log "checkpoint OK: $newest"; else log "checkpoint NOT OK: $newest"; fi
  fi
  sleep "${1:-120}"
done
