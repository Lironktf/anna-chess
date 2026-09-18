#!/usr/bin/env bash
# anna-v6 finish: pull the final checkpoint (parallel streams), strict netcheck, destroy, verify, key cleanup, credit.
set -uo pipefail
cd /home/liron/Desktop/chess; source runs/anna-v6/instance.env
CK=anna-v6-s2-30; D=runs/anna-v6/checkpoints/$CK; mkdir -p "$D"
ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$HOST" "grep -a '=== train end' /workspace/runs/anna-v6/train.log; ls -la /workspace/checkpoints/$CK/" 2>&1 | grep -vE "^Welcome|^Have fun"
scripts/vast_pull_parallel.sh "$HOST" "$PORT" /workspace/checkpoints/$CK/quantised.bin "$D/quantised.bin" 12 2>&1 | grep -vE "^Welcome|^Have fun"
./target/release/engine netcheck "$D/quantised.bin" --strict 2>&1 | tail -2
scripts/vast_pull_parallel.sh "$HOST" "$PORT" /workspace/checkpoints/$CK/raw.bin "$D/raw.bin" 12 2>&1 | grep -vE "^Welcome|^Have fun"
scripts/vast_pull_parallel.sh "$HOST" "$PORT" /workspace/runs/anna-v6/train.log runs/anna-v6/remote_train.log 6 2>&1 | grep -vE "^Welcome|^Have fun"
ls -la "$D"
if [[ -s "$D/quantised.bin" && -s "$D/raw.bin" ]]; then
  GO=1 scripts/vast_launch.sh destroy anna-v6 2>&1 | tail -3
  sleep 20; echo "instances: $(vastai show instances --raw | python3 -c 'import sys,json; print(len(json.load(sys.stdin)))')"
  vastai delete api-key "$(cat runs/anna-v6/selfdestruct_key_id.txt)" 2>&1 | tail -1
  echo "credit: $(vastai show user --raw | python3 -c 'import sys,json; print(round(json.load(sys.stdin)["credit"],3))')"
  echo "$(date -u +%FT%TZ) v6 finished: final checkpoint pulled, instance destroyed" >> runs/anna-v6/events.log
else
  echo "FINAL PULL INCOMPLETE - instance NOT destroyed; fix by hand"
fi
