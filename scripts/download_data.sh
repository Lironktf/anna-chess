#!/usr/bin/env bash
# Resumable, verified, low-priority dataset downloader driven by data/MANIFEST.tsv.
#
# Usage:
#   scripts/download_data.sh start [group...]   # run in background (nohup), default groups: validate train
#   scripts/download_data.sh run   [group...]   # run in foreground
#   scripts/download_data.sh status             # show per-file state
#   scripts/download_data.sh stop               # stop the background job
#
# Guarantees:
#   - Every file resumes from where it stopped (curl -C -), so a crash or reboot loses nothing.
#   - Every file is sha256-verified against the manifest before being marked done (<file>.ok).
#   - Runs under nice 19 / ionice idle: it never competes with compiles, tests or searches.
#   - Streams straight to disk; memory use is a few MB regardless of file size.
#   - Refuses to start a file unless free disk >= file size + 8 GB headroom.
#   - Everything is logged with timestamps to data/downloads/log.txt and state to data/downloads/STATE.tsv.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/data/MANIFEST.tsv"
DLDIR="$ROOT/data/downloads"
LOG="$DLDIR/log.txt"
STATE="$DLDIR/STATE.tsv"
PIDFILE="$DLDIR/downloader.pid"
HEADROOM_BYTES=$((8 * 1024 * 1024 * 1024))
# Optional bandwidth cap, e.g. RATE_LIMIT=2M to leave bandwidth for other work.
RATE_LIMIT="${RATE_LIMIT:-}"

mkdir -p "$DLDIR"

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

set_state() { # name state detail
    local name="$1" st="$2" detail="${3:-}"
    touch "$STATE"
    grep -v "^${name}	" "$STATE" > "$STATE.tmp" 2>/dev/null || true
    printf '%s\t%s\t%s\t%s\n' "$name" "$st" "$(date '+%Y-%m-%dT%H:%M:%S')" "$detail" >> "$STATE.tmp"
    mv "$STATE.tmp" "$STATE"
}

verify() { # name size sha
    local name="$1" size="$2" sha="$3" f="$DLDIR/$1"
    local actual_size; actual_size=$(stat -c %s "$f")
    if [[ "$actual_size" != "$size" ]]; then
        log "VERIFY FAIL $name: size $actual_size != $size"; return 1
    fi
    local actual_sha; actual_sha=$(nice -n 19 ionice -c3 sha256sum "$f" | cut -d' ' -f1)
    if [[ "$actual_sha" != "$sha" ]]; then
        log "VERIFY FAIL $name: sha256 $actual_sha != $sha"; return 1
    fi
    printf '%s  %s\n' "$sha" "$name" > "$f.ok"
    log "VERIFIED $name ($size bytes, sha256 ok)"
    set_state "$name" "done" "$size bytes verified"
}

download_one() { # group name size sha url
    local group="$1" name="$2" size="$3" sha="$4" url="$5" f="$DLDIR/$2"
    if [[ -f "$f.ok" ]]; then
        set_state "$name" "done" "already verified"; log "SKIP $name (already verified)"; return 0
    fi
    local have=0; [[ -f "$f" ]] && have=$(stat -c %s "$f")
    if (( have >= size )); then
        log "COMPLETE-UNVERIFIED $name, verifying"; set_state "$name" "verifying" ""
        verify "$name" "$size" "$sha" && return 0
        log "Corrupt file, deleting and re-downloading $name"; rm -f "$f"; have=0
    fi
    local avail; avail=$(df -B1 --output=avail "$DLDIR" | tail -1)
    local need=$(( size - have + HEADROOM_BYTES ))
    if (( avail < need )); then
        log "NO SPACE for $name: need $need, have $avail"; set_state "$name" "blocked-nospace" "need=$need avail=$avail"; return 2
    fi
    log "START $name group=$group size=$size resume_from=$have"
    set_state "$name" "downloading" "from=$have"
    local rate=(); [[ -n "$RATE_LIMIT" ]] && rate=(--limit-rate "$RATE_LIMIT")
    local attempt=0
    while (( attempt < 50 )); do
        attempt=$((attempt + 1))
        nice -n 19 ionice -c3 curl -sS -L --fail --retry 5 --retry-delay 10 --retry-all-errors \
            -C - "${rate[@]}" --speed-limit 1024 --speed-time 120 \
            -o "$f" "$url" >> "$LOG" 2>&1
        local rc=$?
        have=0; [[ -f "$f" ]] && have=$(stat -c %s "$f")
        if (( rc == 0 )) && (( have >= size )); then break; fi
        # rc 33 = range not satisfiable (file already complete); 18 = partial file (resume)
        log "curl rc=$rc after attempt $attempt, have=$have/$size, retrying in 20s"
        set_state "$name" "downloading" "attempt=$attempt have=$have"
        sleep 20
    done
    have=0; [[ -f "$f" ]] && have=$(stat -c %s "$f")
    if (( have < size )); then
        log "GAVE UP $name at $have/$size"; set_state "$name" "failed" "have=$have"; return 1
    fi
    set_state "$name" "verifying" ""
    verify "$name" "$size" "$sha"
}

run_groups() {
    local groups=("$@"); (( ${#groups[@]} == 0 )) && groups=(validate train)
    log "=== downloader run start, groups: ${groups[*]}, pid $$"
    local failures=0
    while IFS=$'\t' read -r group name size sha url; do
        [[ -z "$group" || "$group" == \#* ]] && continue
        local wanted=0; for g in "${groups[@]}"; do [[ "$g" == "$group" ]] && wanted=1; done
        (( wanted )) || continue
        if [[ "$sha" == "UNKNOWN" || "$size" == "0" ]]; then
            log "SKIP $name: manifest has no checksum/size; fill it in first"; set_state "$name" "blocked-nomanifest" ""; continue
        fi
        download_one "$group" "$name" "$size" "$sha" "$url" || failures=$((failures + 1))
    done < "$MANIFEST"
    log "=== downloader run finished, failures=$failures"
    rm -f "$PIDFILE"
    return $failures
}

case "${1:-}" in
    run) shift; run_groups "$@" ;;
    start)
        shift
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "already running, pid $(cat "$PIDFILE")"; exit 0
        fi
        nohup "$0" run "$@" > /dev/null 2>&1 &
        echo $! > "$PIDFILE"
        echo "started pid $! (log: $LOG)"
        ;;
    stop)
        if [[ -f "$PIDFILE" ]]; then pkill -P "$(cat "$PIDFILE")" curl 2>/dev/null; kill "$(cat "$PIDFILE")" 2>/dev/null; rm -f "$PIDFILE"; echo stopped; else echo "not running"; fi
        ;;
    status)
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "RUNNING pid $(cat "$PIDFILE")"; else echo "NOT RUNNING"; fi
        echo "--- state"; [[ -f "$STATE" ]] && column -t -s $'\t' "$STATE"
        echo "--- files"
        while IFS=$'\t' read -r group name size sha url; do
            [[ -z "$group" || "$group" == \#* ]] && continue
            local_f="$DLDIR/$name"; have=0; [[ -f "$local_f" ]] && have=$(stat -c %s "$local_f")
            ok="no"; [[ -f "$local_f.ok" ]] && ok="yes"
            pct=0; (( size > 0 )) && pct=$(( have * 100 / size ))
            printf '%-8s %3d%% verified=%-3s %s\n' "$group" "$pct" "$ok" "$name"
        done < "$MANIFEST"
        echo "--- last log lines"; tail -n 5 "$LOG" 2>/dev/null
        ;;
    *) sed -n '2,12p' "$0"; exit 1 ;;
esac
