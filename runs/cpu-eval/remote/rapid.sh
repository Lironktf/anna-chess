#!/usr/bin/env bash
# After the scripted matches end, play a rapid v3 vs v1 match (fewer games, longer time), then mark the end.
OUT=/workspace/runs/cpu-eval; L2=$OUT/train2.log
echo "$(date -u +%FT%TZ) === train start (rapid follow-up)" >> $L2
while ! grep -q '=== train end' $OUT/train.log; do sleep 30; done
E=/workspace/chess/target/release/engine
echo "$(date -u +%FT%TZ) match rapid_v3_vs_v1: 60 games at 120+1.2" >> $L2
/workspace/tools/fastchess/fastchess -engine cmd=$E name=anna_v3 option.EvalFile=/workspace/nets/anna-v3.bin -engine cmd=$E name=anna_v1 option.EvalFile=/workspace/nets/anna-v1.bin \
  -each tc=120+1.2 proto=uci option.Hash=128 -openings file=/workspace/books/UHO_Lichess_4852_v1.epd format=epd order=random \
  -concurrency 28 -ratinginterval 10 -recover -rounds 30 -games 2 -repeat -pgnout file=$OUT/rapid_v3_vs_v1.pgn > $OUT/rapid_v3_vs_v1.log 2>&1
echo "$(date -u +%FT%TZ) result rapid_v3_vs_v1: $(grep -a -E '^Elo' $OUT/rapid_v3_vs_v1.log | tail -n 1) | $(grep -a -E '^Games' $OUT/rapid_v3_vs_v1.log | tail -n 1)" >> $L2
echo "$(date -u +%FT%TZ) === train end exit=0" >> $L2
