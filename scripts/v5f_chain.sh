#!/usr/bin/env bash
# 2026-09-15 evening chain: v5f blitz SPRT vs v3b, then a 4-thread 60+0.6 check (laptop).
cd /home/liron/Desktop/chess
timeout 6h scripts/sprt.sh sprt/bin/anna_v5f sprt/bin/anna_v5f -N "option.EvalFile=/home/liron/Desktop/chess/runs/anna-v5/checkpoints/anna-v5f-s2-30/quantised.bin" -B "option.EvalFile=/home/liron/Desktop/chess/runs/anna-v3b/checkpoints/anna-v3b-s2-30/quantised.bin" -t 8+0.08 -c 4 -e 0 10 -n v5f_blitz > sprt/v5f_blitz.out 2>&1
timeout 6h nice -n 10 tools/fastchess/fastchess -engine cmd=sprt/bin/anna_v5f name=v5f_4t option.EvalFile=/home/liron/Desktop/chess/runs/anna-v5/checkpoints/anna-v5f-s2-30/quantised.bin -engine cmd=sprt/bin/anna_v5f name=v3b_4t option.EvalFile=/home/liron/Desktop/chess/runs/anna-v3b/checkpoints/anna-v3b-s2-30/quantised.bin -each tc=60+0.6 proto=uci option.Hash=256 option.Threads=4 -openings file=books/UHO_Lichess_4852_v1.epd format=epd order=random -concurrency 1 -rounds 40 -games 2 -repeat -ratinginterval 20 -pgnout file=sprt/v5f_4t_slow.pgn > sprt/v5f_4t_slow.log 2>&1
touch sprt/v5f_chain.done
