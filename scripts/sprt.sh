#!/usr/bin/env bash
# SPRT match between two engine binaries with fastchess and the UHO book.
#
# Usage: scripts/sprt.sh <new-binary> <base-binary> [options]
#   -t TC        time control (default 8+0.08)
#   -c N         concurrency (default 4)
#   -e ELO0 ELO1 SPRT bounds (default 0 5 for gainers; use -5 0 / -10 0 for non-regression)
#   -g N         fixed number of games instead of SPRT
#   -n NAME      run name for the PGN/log (default timestamp)
#   -o "opts"    extra UCI options for both engines, e.g. "option.Hash=64"
#   -N "opts"    extra UCI options for the new engine only, e.g. "option.Cuckoo=1"
#   -B "opts"    extra UCI options for the base engine only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTCHESS="$ROOT/tools/fastchess/fastchess"
BOOK="$ROOT/books/UHO_Lichess_4852_v1.epd"
NEW="${1:?new binary}"; BASE="${2:?base binary}"; shift 2
TC="8+0.08"; CONC=4; ELO0=0; ELO1=5; GAMES=""; NAME="$(date +%Y%m%d-%H%M%S)"; EXTRA=""; NEWOPTS=""; BASEOPTS=""
while getopts "t:c:e:g:n:o:N:B:" opt; do
    case $opt in
        t) TC="$OPTARG" ;;
        c) CONC="$OPTARG" ;;
        e) ELO0="$OPTARG"; ELO1="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        g) GAMES="$OPTARG" ;;
        n) NAME="$OPTARG" ;;
        o) EXTRA="$OPTARG" ;;
        N) NEWOPTS="$OPTARG" ;;
        B) BASEOPTS="$OPTARG" ;;
        *) exit 1 ;;
    esac
done
mkdir -p "$ROOT/sprt"
LOG="$ROOT/sprt/$NAME.log"; PGN="$ROOT/sprt/$NAME.pgn"
[[ -x "$FASTCHESS" ]] || { echo "fastchess not built: $FASTCHESS"; exit 1; }
[[ -f "$BOOK" ]] || { echo "book missing: $BOOK"; exit 1; }
echo "new=$NEW base=$BASE tc=$TC conc=$CONC bounds=[$ELO0,$ELO1] games=${GAMES:-sprt} log=$LOG"
if [[ -n "$GAMES" ]]; then
    MODE=(-rounds "$((GAMES / 2))" -games 2 -repeat)
else
    MODE=(-sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.05 -rounds 50000 -games 2 -repeat)
fi
# shellcheck disable=SC2086
nice -n 5 "$FASTCHESS" \
    -engine cmd="$NEW" name=new $EXTRA $NEWOPTS \
    -engine cmd="$BASE" name=base $EXTRA $BASEOPTS \
    -each tc="$TC" proto=uci \
    -openings file="$BOOK" format=epd order=random \
    -concurrency "$CONC" -ratinginterval 20 -recover \
    -pgnout file="$PGN" \
    "${MODE[@]}" 2>&1 | tee "$LOG"
