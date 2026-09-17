#!/usr/bin/env bash
# Runs the search-sync SPRT queue sequentially: each entry "name param" tests option.<param>=1 vs 0 on one binary.
# Usage: scripts/sync_queue.sh <binary> <tc> <concurrency> <hours-per-test> name:param [name:param ...]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$1"; TC="$2"; CONC="$3"; HOURS="$4"; shift 4
for spec in "$@"; do
    name="${spec%%:*}"; param="${spec##*:}"
    echo "$(date '+%F %T') start $name ($param) tc=$TC" >> "$ROOT/sprt/sync_queue.log"
    timeout "${HOURS}h" "$ROOT/scripts/sprt.sh" "$BIN" "$BIN" -N "option.$param=1" -B "option.$param=0" -t "$TC" -c "$CONC" -n "$name" > "$ROOT/sprt/$name.out" 2>&1
    rc=$?
    elo=$(grep -E "^Elo:" "$ROOT/sprt/$name.log" | tail -1 | cut -c1-40)
    llr=$(grep -E "^LLR:" "$ROOT/sprt/$name.log" | tail -1 | cut -c1-60)
    games=$(grep -c "Finished game" "$ROOT/sprt/$name.log")
    verdict="STOPPED"; echo "$llr" | grep -q "2.94)" && { echo "$llr" | grep -qE "^LLR: (2\.9|3\.)" && verdict=PASS; echo "$llr" | grep -qE "^LLR: -(2\.9|3\.)" && verdict=FAIL; }
    echo "$name $verdict games=$games | $elo | $llr | new=[option.$param=1] base=[option.$param=0] tc=$TC rc=$rc" >> "$ROOT/sprt/verdicts.txt"
    echo "$(date '+%F %T') done $name: $verdict $elo" >> "$ROOT/sprt/sync_queue.log"
done
echo "$(date '+%F %T') queue finished" >> "$ROOT/sprt/sync_queue.log"
