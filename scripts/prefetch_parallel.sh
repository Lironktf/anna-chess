#!/usr/bin/env bash
# Parallel prefetcher that runs NEXT TO scripts/download_data.sh on the box: it fetches manifest files in REVERSE order
# with several curl streams into <name>.part and renames each complete file into place, so the sequential downloader
# (which walks the manifest forward) finds them COMPLETE-UNVERIFIED and only has to sha256 them. Idempotent.
# Usage: scripts/prefetch_parallel.sh <jobs> <group>...
set -uo pipefail
JOBS="${1:?jobs}"; shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DLDIR="$ROOT/data/downloads"; LOG="$DLDIR/prefetch.log"
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }
fetch() { # name size url
    local name="$1" size="$2" url="$3" f="$DLDIR/$1"
    [[ -f "$f.ok" || -f "$f" ]] && { log "SKIP $name (present)"; return 0; }
    local have=0; [[ -f "$f.part" ]] && have=$(stat -c %s "$f.part")
    log "START $name size=$size from=$have"
    for attempt in 1 2 3 4 5 6 7 8; do
        curl -sS -L --fail --retry 5 --retry-delay 10 --retry-all-errors -C - --speed-limit 1024 --speed-time 120 -o "$f.part" "$url" >> "$LOG" 2>&1
        have=0; [[ -f "$f.part" ]] && have=$(stat -c %s "$f.part")
        (( have >= size )) && break
        log "retry $attempt $name have=$have"; sleep 15
    done
    if (( have >= size )) && [[ ! -f "$f" ]]; then mv "$f.part" "$f"; log "DONE $name"; else log "GAVE UP $name have=$have"; fi
}
export -f fetch log; export DLDIR LOG
groups=" $* "
grep -v '^#' "$ROOT/data/MANIFEST.tsv" | awk -F'\t' -v g="$groups" 'index(g, " "$1" ") {print $2"\t"$3"\t"$5}' | tac \
  | xargs -P "$JOBS" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r n s u <<< "{}"; fetch "$n" "$s" "$u"'
log "=== prefetch finished"
