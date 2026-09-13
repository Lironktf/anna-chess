#!/usr/bin/env bash
# When the overnight chain reaches its SPSA stage (tuned for v1), replace it with an SPSA run for v3b, the engine
# targeted at the CCRL 40/15 list. Kills by exact argv fields, never by substring (see EXPERIMENTS.md lessons).
set -uo pipefail
cd /home/liron/Desktop/chess
until grep -q "stage 3: SPSA start" sprt/overnight.log 2>/dev/null; do sleep 60; done
sleep 20
for pid in $(ps -eo pid,args | awk '($2 ~ /bash$/ && $3=="scripts/overnight.sh") {print $1}'); do kill "$pid"; done
for pid in $(ps -eo pid,args | awk '$2 ~ /python3$/ && $3=="scripts/spsa.py" {print $1}'); do kill "$pid"; done
sleep 3
for pid in $(ps -eo pid,args | awk '$2 ~ /fastchess$/ && $0 ~ /name=plus/ {print $1}'); do kill "$pid"; done
echo "$(date '+%F %T') handoff: v1 SPSA stopped, starting v3b SPSA" >> sprt/overnight.log
V3B=/home/liron/Desktop/chess/runs/anna-v3b/checkpoints/anna-v3b-s2-30/quantised.bin
exec nice -n 5 python3 scripts/spsa.py --engine sprt/bin/anna_fast2 --engine-opts "option.EvalFile=$V3B" --iters 400 --games 8 --tc 8+0.08 --conc 4 --r-end 0.02 --out sprt/spsa_v3b \
    --param LmrBase:982:0:3000:60 --param LmrScalePct:100:50:200:8 --param LmrCutNode:3000:0:6000:200 --param LmrHistDiv:16384:4096:65536:1500 \
    --param RfpMult:45:20:120:6 --param RfpDepth:14:4:20:1 --param NmpBase:5:2:8:1 --param LmpBase:3:1:8:1 \
    --param FutMargin:119:40:300:12 --param RazorMult:482:100:1000:50 --param SeMargin:59:20:120:6 \
    --param TmOptPct:100:30:300:10 --param TmInstabPct:180:0:400:20 >> sprt/spsa_v3b.out 2>&1
