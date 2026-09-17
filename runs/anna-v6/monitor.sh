#!/usr/bin/env bash
# Laptop monitor for anna-v6: polls the box's train.log; logs progress; exits when training ends or the instance is gone.
cd /home/liron/Desktop/chess; source runs/anna-v6/instance.env
while true; do
  out=$(timeout 60 ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20 root@"$HOST" 'sed "s/\x1b\[[0-9;]*m//g" /workspace/runs/anna-v6/train.log | grep -oE "superbatch [0-9]+ \[|=== train end.*|Saved \[[^]]+\]" | tail -2 | tr "\n" " "; ls -dt /workspace/checkpoints/anna-v6-*/ 2>/dev/null | head -1' 2>/dev/null | grep -vE "^Welcome|^Have fun" | tr '\n' ' ')
  echo "$(date '+%H:%M') $out" >> runs/anna-v6/monitor.log
  if echo "$out" | grep -q "=== train end"; then echo "TRAIN END $(date)" >> runs/anna-v6/monitor.log; exit 0; fi
  if [[ "$(vastai show instances --raw 2>/dev/null | python3 -c 'import sys,json; print(len(json.load(sys.stdin)))' 2>/dev/null)" == "0" ]]; then echo "INSTANCE GONE $(date)" >> runs/anna-v6/monitor.log; exit 1; fi
  sleep 300
done
