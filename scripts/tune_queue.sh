#!/usr/bin/env bash
# Unattended SPRT queue runner. Reads sprt/queue.txt line by line:
#     name | new-engine options | base-engine options
# runs each test that has no verdict yet in sprt/verdicts.txt, and appends the verdict
# (PASS / FAIL / STOPPED) with the Elo line. Safe to restart at any time; append lines to the
# queue while it runs. Each test is capped at MAX_GAMES games to keep the queue moving.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE="$ROOT/sprt/queue.txt"; VERDICTS="$ROOT/sprt/verdicts.txt"; BIN="${BIN:-$ROOT/sprt/bin/fable_v1b}"
TC="${TC:-8+0.08}"; MAX_GAMES="${MAX_GAMES:-3000}"
touch "$VERDICTS"
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$ROOT/sprt/queue.log"; }
log "queue runner start (bin=$BIN tc=$TC max_games=$MAX_GAMES)"
while true; do
    next=""
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        name="${line%%|*}"; name="${name// /}"
        grep -q "^$name " "$VERDICTS" && continue
        next="$line"; break
    done < "$QUEUE"
    if [[ -z "$next" ]]; then
        log "queue empty; sleeping 10 min"; sleep 600; continue
    fi
    name="${next%%|*}"; name="${name// /}"
    rest="${next#*|}"; newo="${rest%%|*}"; baseo="${rest#*|}"
    newo="${newo## }"; newo="${newo%% }"; baseo="${baseo## }"; baseo="${baseo%% }"
    log "START $name new=[$newo] base=[$baseo]"
    # Run the SPRT; a watchdog stops it at MAX_GAMES.
    ( "$ROOT/scripts/sprt.sh" "$BIN" "$BIN" -t "$TC" -c 4 -e 0 5 -n "$name" -N "$newo" -B "$baseo" > /dev/null 2>&1 ) &
    spid=$!
    while kill -0 "$spid" 2>/dev/null; do
        sleep 60
        games=$(grep -E "^Games:" "$ROOT/sprt/$name.log" 2>/dev/null | tail -1 | sed -E 's/Games: ([0-9]+).*/\1/')
        if [[ -n "$games" ]] && (( games >= MAX_GAMES )); then
            pkill -x fastchess; sleep 3
        fi
    done
    elo=$(grep -E "^Elo:" "$ROOT/sprt/$name.log" | tail -1)
    llr=$(grep -E "^LLR:" "$ROOT/sprt/$name.log" | tail -1)
    games=$(grep -E "^Games:" "$ROOT/sprt/$name.log" | tail -1 | sed -E 's/Games: ([0-9]+).*/\1/')
    verdict="STOPPED"
    if [[ "$llr" == *"(100.0%)"* || "$llr" == *"(100.1%)"* ]] || echo "$llr" | grep -qE "LLR: (2\.9[4-9]|[3-9])"; then verdict="PASS"; fi
    if echo "$llr" | grep -qE "LLR: -(2\.9[4-9]|[3-9])"; then verdict="FAIL"; fi
    printf '%s %s games=%s | %s | %s | new=[%s] base=[%s]\n' "$name" "$verdict" "${games:-0}" "$elo" "$llr" "$newo" "$baseo" >> "$VERDICTS"
    log "DONE $name $verdict games=${games:-0} $elo"
done
