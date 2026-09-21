#!/usr/bin/env bash
# Self-play data generation with the current engine and net. Writes one file per chunk so the run can be
# stopped at any time and every completed file is usable. Runs at the lowest priority so the match queue wins.
ulimit -t unlimited
cd ~/anna
mkdir -p data/selfplay
N=${1:-16}              # threads
NODES=${2:-5000}        # nodes per position (Stockfish's published sets use 5000)
GAMES=${3:-20000}       # games per chunk
i=$(ls data/selfplay/sp_*.bin 2>/dev/null | wc -l)
while true; do
  i=$((i+1))
  f=$(printf "data/selfplay/sp_%04d.bin" "$i")
  echo "$(date '+%F %T') chunk $i start (threads $N, nodes $NODES, games $GAMES)" >> data/selfplay/gen.log
  nice -n 19 ./target/release/engine datagen --threads "$N" --nodes "$NODES" --games "$GAMES" \
      --seed $((RANDOM * 32768 + RANDOM)) --out "$f" >> data/selfplay/gen.log 2>&1
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  echo "$(date '+%F %T') chunk $i done: $((sz/32)) positions, $((sz/1048576)) MB" >> data/selfplay/gen.log
  [ -f data/selfplay/STOP ] && { echo "$(date '+%F %T') stopped by flag" >> data/selfplay/gen.log; break; }
done
