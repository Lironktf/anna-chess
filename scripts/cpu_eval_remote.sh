#!/usr/bin/env bash
# Runs ON A RENTED CPU BOX: build Anna and the reference engines, then play the rating matches.
# Writes everything to /workspace/runs/$RUN_NAME/ (train.log carries the "=== train start/end" markers
# so scripts/vast_selfdestruct.sh can be reused unchanged). Idempotent: re-running skips finished steps.
#
# Env: RUN_NAME (required), NET_V3 (path to v3 quantised.bin), NET_V1 (path to v1 net), CONC (games in
# parallel), GAMES_FAST / GAMES_LONG (per match), TC_FAST / TC_LONG.
set -uo pipefail
RUN_NAME="${RUN_NAME:?}"; WORK=/workspace; SRC=$WORK/chess; OUT=$WORK/runs/$RUN_NAME; mkdir -p "$OUT" $WORK/tools
LOG="$OUT/train.log"
NET_V3="${NET_V3:-$WORK/nets/anna-v3.bin}"; NET_V1="${NET_V1:-$WORK/nets/anna-v1.bin}"; NET_OLD="${NET_OLD:-$WORK/nets/anna-old.bin}"
MATCHES="${MATCHES:-default}"   # default = full anchor set; "validate" = new net vs v1/old/Stormphrax at blitz + long
CONC="${CONC:-28}"; GAMES_FAST="${GAMES_FAST:-400}"; GAMES_LONG="${GAMES_LONG:-100}"; TC_FAST="${TC_FAST:-8+0.08}"; TC_LONG="${TC_LONG:-60+0.6}"
BOOK=$WORK/books/UHO_Lichess_4852_v1.epd
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
log "=== train start (cpu-eval) run=$RUN_NAME conc=$CONC games_fast=$GAMES_FAST games_long=$GAMES_LONG"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|L3" | tee -a "$LOG"
free -g | head -2 | tee -a "$LOG"

export DEBIAN_FRONTEND=noninteractive
command -v g++ >/dev/null && command -v make >/dev/null && command -v git >/dev/null && command -v tmux >/dev/null || {
    apt-get update -qq && apt-get install -y -qq build-essential git curl tmux ca-certificates pkg-config >/dev/null; }
# Ubuntu 22.04 ships GCC 11: too old for Weiss (C23 enums) and its libstdc++ is too old for the Stormphrax 8
# release binary (GLIBCXX_3.4.31). The toolchain PPA provides gcc-13 and a matching libstdc++6.
if ! command -v gcc-13 >/dev/null; then
    apt-get install -y -qq software-properties-common >/dev/null 2>&1; add-apt-repository -y ppa:ubuntu-toolchain-r/test >/dev/null 2>&1
    apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq gcc-13 libstdc++6 >/dev/null 2>&1
fi
if ! command -v cargo >/dev/null; then
    [[ -f "$HOME/.cargo/env" ]] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
log "toolchain: $(rustc --version) / $(g++ --version | head -1)"

# ---- Anna (target-cpu=native from .cargo/config.toml) ----
if [[ ! -x $SRC/target/release/engine ]]; then
    (cd "$SRC" && cargo build --release -p engine 2>&1 | tail -n 3 | tee -a "$LOG")
fi
[[ -x $SRC/target/release/engine ]] || { log "ENGINE BUILD FAILED"; log "=== train end exit=1"; exit 1; }
ENGINE=$SRC/target/release/engine
log "bench: $($ENGINE bench 2>&1 | tail -n 1)"
for n in "$NET_V1" "$NET_V3"; do
    log "netcheck $n: $($ENGINE netcheck "$n" 2>&1 | tail -n 1)"
done
# Single-thread search speed with each net (the number that decides the equal-time result).
for n in "$NET_V1" "$NET_V3"; do
    # keep stdin open for the whole search: the engine quits on EOF
    nps=$( (printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos moves e2e4 e7e5 g1f3\ngo movetime 5000\n' "$n"; sleep 6; echo quit) | $ENGINE 2>/dev/null | grep -a "^info" | grep -o "nps [0-9]*" | tail -n 1)
    log "search speed $n: $nps"
done

# ---- reference engines (exact versions used on the laptop) ----
cd $WORK/tools
if [[ ! -x fastchess/fastchess ]]; then
    git clone -q https://github.com/Disservin/fastchess.git && (cd fastchess && git checkout -q 4e69146 && make -j"$(nproc)" >/dev/null 2>&1)
fi
if [[ ! -x stash-bot/stash ]]; then
    [[ -d stash-bot ]] || git clone -q https://github.com/mhouppin/stash-bot.git
    (cd stash-bot && git checkout -q 13e0a81 && cd src && make -j"$(nproc)" >/dev/null 2>&1 && cp -f stash ../stash)
fi
if [[ ! -x weiss/weiss ]]; then
    [[ -d weiss ]] || git clone -q https://github.com/TerjeKir/weiss.git
    # Weiss' Makefile uses -Werror with a clang-only warning flag; build the same PGO sequence by hand with gcc-13.
    (cd weiss && git checkout -q c735b8f && cd src && F="-std=gnu11 -O3 -flto=auto -march=native -DGIT_HEAD_COMMIT_ID=\"c735b8f\" -DUSE_PEXT -DNDEBUG" \
        && SRCS="*.c pyrrhic/tbprobe.c tuner/*.c query/*.c noobprobe/*.c onlinesyzygy/*.c" && rm -rf pgo \
        && gcc-13 $F $SRCS -pthread -lm -o weiss -fprofile-generate="pgo" >/dev/null 2>&1 && ./weiss bench 12 >/dev/null 2>&1 \
        && gcc-13 $F $SRCS -pthread -lm -o weiss -fprofile-use="pgo" -Wno-missing-profile >/dev/null 2>&1 && cp -f weiss ../weiss)
fi
for e in fastchess/fastchess stash-bot/stash weiss/weiss $WORK/tools/stormphrax $WORK/tools/stockfish; do
    [[ -x $e ]] && log "ok: $e" || log "MISSING: $e"
done
[[ -x fastchess/fastchess ]] || { log "fastchess missing"; log "=== train end exit=2"; exit 2; }
FC=$WORK/tools/fastchess/fastchess
STORM=$WORK/tools/stormphrax; STOCK=$WORK/tools/stockfish; STASH=$WORK/tools/stash-bot/stash; WEISS=$WORK/tools/weiss/weiss
for e in $STORM $STOCK $STASH $WEISS; do
    [[ -x $e ]] && log "$e: $(printf 'uci\nquit\n' | timeout 20 $e 2>/dev/null | grep -a -m1 "id name")"
done

# ---- matches ----
# match NAME GAMES TC "engine A args" "engine B args"
match() {
    local name="$1" games="$2" tc="$3"; shift 3
    local pgn="$OUT/$name.pgn" mlog="$OUT/$name.log"
    if grep -aq "^Games: $games" "$mlog" 2>/dev/null; then log "skip $name (done)"; return; fi
    log "match $name: $games games at $tc"
    $FC "$@" -each tc="$tc" proto=uci option.Hash=64 \
        -openings file="$BOOK" format=epd order=random -concurrency "$CONC" -ratinginterval 50 -recover \
        -rounds "$((games / 2))" -games 2 -repeat -pgnout file="$pgn" > "$mlog" 2>&1
    log "result $name: $(grep -a -E '^Elo' "$mlog" | tail -n 1) | $(grep -a -E '^Games' "$mlog" | tail -n 1)"
}
A_V3="-engine cmd=$ENGINE name=anna_v3 option.EvalFile=$NET_V3"
A_V1="-engine cmd=$ENGINE name=anna_v1 option.EvalFile=$NET_V1"
A_OLD="-engine cmd=$ENGINE name=anna_old option.EvalFile=$NET_OLD"
if [[ "$MATCHES" == "slow4" ]]; then
    # 40/15-list style anchor: 4 threads per engine, 60+0.6, 8 games in parallel on a 32-thread box.
    T4="option.Threads=4"
    CONC=$(( $(nproc) / 4 ))
    N4="-engine cmd=$ENGINE name=anna_new4 option.EvalFile=$NET_V3 $T4"
    match slow4_smp4_vs_1          60  "60+0.6" $N4 -engine cmd=$ENGINE name=anna_new1 option.EvalFile=$NET_V3 option.Threads=1
    match slow4_new_vs_v1          200 "60+0.6" $N4 $A_V1 $T4
    [[ -x $STORM ]] && match slow4_new_vs_stormphrax8 200 "60+0.6" $N4 -engine cmd=$STORM name=stormphrax8 $T4
    [[ -x $STASH ]] && match slow4_new_vs_stash 100 "60+0.6" $N4 -engine cmd=$STASH name=stash $T4
    log "=== train end exit=0"
    exit 0
fi
if [[ "$MATCHES" == "validate" ]]; then
    match new_vs_v1_fast  "$GAMES_FAST" "$TC_FAST" $A_V3 $A_V1
    [[ -f $NET_OLD ]] && match new_vs_old_fast "$GAMES_FAST" "$TC_FAST" $A_V3 $A_OLD
    [[ -x $STORM ]] && match new_vs_stormphrax8 "$GAMES_FAST" "$TC_FAST" $A_V3 -engine cmd=$STORM name=stormphrax8
    match new_vs_v1_long  "$GAMES_LONG" "$TC_LONG" $A_V3 $A_V1
    [[ -f $NET_OLD ]] && match new_vs_old_long "$GAMES_LONG" "$TC_LONG" $A_V3 $A_OLD
    match new_vs_v1_rapid 60 "120+1.2" $A_V3 $A_V1
    log "=== train end exit=0"
    exit 0
fi
match v3_vs_v1_fast      "$GAMES_FAST" "$TC_FAST" $A_V3 $A_V1
[[ -x $STORM ]] && match v3_vs_stormphrax8 "$GAMES_FAST" "$TC_FAST" $A_V3 -engine cmd=$STORM name=stormphrax8
[[ -x $STASH ]] && match v3_vs_stash       "$GAMES_FAST" "$TC_FAST" $A_V3 -engine cmd=$STASH name=stash
[[ -x $STORM ]] && match v1_vs_stormphrax8 "$GAMES_FAST" "$TC_FAST" $A_V1 -engine cmd=$STORM name=stormphrax8
[[ -x $WEISS ]] && match v3_vs_weiss       "$GAMES_FAST" "$TC_FAST" $A_V3 -engine cmd=$WEISS name=weiss
match v3_vs_v1_long      "$GAMES_LONG" "$TC_LONG" $A_V3 $A_V1
[[ -x $STOCK ]] && match v3_vs_stockfish19 200 "$TC_FAST" $A_V3 -engine cmd=$STOCK name=stockfish19
log "=== train end exit=0"
