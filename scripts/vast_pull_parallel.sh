#!/usr/bin/env bash
# Pull one remote file over N parallel ssh streams (byte ranges), for links that throttle per connection.
# Usage: scripts/vast_pull_parallel.sh <host> <port> <remote-file> <local-file> [streams=8]
# Verifies size and sha256 against the remote before replacing the destination. Idempotent (re-pulls only on mismatch).
set -uo pipefail
HOST="$1"; PORT="$2"; SRC="$3"; DST="$4"; N="${5:-8}"
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o Compression=no root@"$HOST")
read -r SIZE SHA < <("${SSH[@]}" "stat -c %s '$SRC'; sha256sum '$SRC' | cut -d' ' -f1" | paste -sd' ') || { echo "remote stat failed"; exit 1; }
[[ "$SIZE" =~ ^[0-9]+$ ]] || { echo "bad size '$SIZE'"; exit 1; }
if [[ -f "$DST" && "$(stat -c %s "$DST")" == "$SIZE" && "$(sha256sum "$DST" | cut -d' ' -f1)" == "$SHA" ]]; then echo "already have $DST"; exit 0; fi
TMP="$DST.parts.$$"; mkdir -p "$TMP"
CH=$(( (SIZE + N - 1) / N )); t0=$(date +%s)
for i in $(seq 0 $((N-1))); do
    ( off=$((i*CH)); len=$CH; (( off+len > SIZE )) && len=$((SIZE-off)); (( len <= 0 )) && exit 0
      for attempt in 1 2 3 4 5; do
        "${SSH[@]}" "tail -c +$((off+1)) '$SRC' | head -c $len" > "$TMP/$i" 2>/dev/null
        [[ "$(stat -c %s "$TMP/$i")" == "$len" ]] && exit 0; sleep 5
      done; exit 1 ) &
done
wait; cat "$TMP"/$(seq -s' ' 0 $((N-1)) | sed "s# # $TMP/#g") > "$DST.tmp" 2>/dev/null || cat $(for i in $(seq 0 $((N-1))); do [[ -f "$TMP/$i" ]] && echo "$TMP/$i"; done) > "$DST.tmp"
rm -rf "$TMP"
got=$(stat -c %s "$DST.tmp"); gsha=$(sha256sum "$DST.tmp" | cut -d' ' -f1)
if [[ "$got" == "$SIZE" && "$gsha" == "$SHA" ]]; then mv "$DST.tmp" "$DST"; echo "pulled $DST ($SIZE bytes, sha256 ok, $(( $(date +%s)-t0 )) s, $N streams)"; else rm -f "$DST.tmp"; echo "PULL FAILED $SRC: size $got/$SIZE sha match=$([[ "$gsha" == "$SHA" ]] && echo yes || echo no)"; exit 1; fi
