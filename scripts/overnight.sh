#!/usr/bin/env bash
# Unattended free-work chain for the laptop (no paid resources):
#   1. wait for the running SPRT (fastchess) to finish
#   2. SPRT each queued candidate (capped by a wall-clock timeout; a stopped test still reports Elo)
#   3. SPSA-tune the search constants
#   4. SPRT the tuned constants against the defaults
# Everything is logged to sprt/overnight.log and sprt/verdicts.txt. Re-runnable: finished stages are skipped.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
LOG=sprt/overnight.log; V=sprt/verdicts.txt; touch "$V"
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
BIN_Q=sprt/bin/anna_q          # candidates (params default off)
BIN_F=sprt/bin/anna_fast       # current engine with the speed work, for tuning
TEST_TIMEOUT="${TEST_TIMEOUT:-4h}"
log "overnight start"
until ! pgrep -x fastchess >/dev/null; do sleep 60; done
log "stage 1: previous SPRT finished: $(grep -a -E '^Elo' sprt/threat_hist.log | tail -n 1)"
grep -q "^threat_hist " "$V" || echo "threat_hist $(grep -a -E '^LLR|^Elo' sprt/threat_hist.log | tail -n 2 | tr '\n' ' ')" >> "$V"

run_test() { # name new-opts base-opts bin
    local name="$1" new="$2" base="$3" bin="$4"
    if grep -q "^$name " "$V"; then log "skip $name (has verdict)"; return; fi
    log "stage 2: SPRT $name: new='$new' base='$base' bin=$bin"
    timeout "$TEST_TIMEOUT" scripts/sprt.sh "$bin" "$bin" -N "$new" -B "$base" -t 8+0.08 -c 4 -e 0 5 -n "$name" > "sprt/$name.out" 2>&1
    local llr elo; llr=$(grep -a '^LLR' "sprt/$name.log" | tail -n 1); elo=$(grep -a '^Elo' "sprt/$name.log" | tail -n 1)
    local verdict="STOPPED"; grep -aq "H1 was accepted" "sprt/$name.log" && verdict=PASS; grep -aq "H0 was accepted" "sprt/$name.log" && verdict=FAIL
    echo "$name $verdict | $elo | $llr" >> "$V"; log "verdict $name: $verdict | $elo | $llr"
}
run_test settm   "option.SeTtmHist=1"   "option.SeTtmHist=0"   "$BIN_Q"
run_test lmp4    "option.LmpBase=4"     "option.LmpBase=3"     "$BIN_Q"
run_test thorder "option.ThreatOrder=1" "option.ThreatOrder=0" "$BIN_Q"

# stage 3: SPSA on the search constants (resumable; skipped if already complete)
if ! grep -q "spsa done" sprt/spsa1/spsa.log 2>/dev/null; then
    log "stage 3: SPSA start"
    nice -n 5 python3 scripts/spsa.py --engine "$BIN_F" --iters 400 --games 8 --tc 8+0.08 --conc 4 --r-end 0.02 --out sprt/spsa1 \
        --param LmrBase:982:0:3000:60 --param LmrScalePct:100:50:200:8 --param LmrCutNode:3000:0:6000:200 --param LmrHistDiv:16384:4096:65536:1500 \
        --param RfpMult:45:20:120:6 --param RfpDepth:14:4:20:1 --param NmpBase:5:2:8:1 --param LmpBase:3:1:8:1 \
        --param FutMargin:119:40:300:12 --param RazorMult:482:100:1000:50 --param SeMargin:59:20:120:6 \
        --param TmOptPct:100:30:300:10 --param TmInstabPct:180:0:400:20 >> sprt/spsa1.out 2>&1
    log "stage 3: SPSA end: $(tail -n 1 sprt/spsa1/spsa.log)"
fi
# stage 4: verify the tuned set against the defaults
if [[ -s sprt/spsa1/best.txt ]]; then
    TUNED=$(tr '\n' ' ' < sprt/spsa1/best.txt)
    run_test spsa1_verify "$TUNED" "" "$BIN_F"
fi
log "overnight done"
