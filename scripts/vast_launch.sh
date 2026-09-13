#!/usr/bin/env bash
# Laptop-side orchestration for a vast.ai training run. Every paid step is gated on GO=1 being set
# explicitly on the command line by the human. Without it the script only prints what it would do.
#
#   scripts/vast_launch.sh search                      # list candidate offers (free)
#   GO=1 scripts/vast_launch.sh create <OFFER_ID> <RUN> # rent (PAID), upload repo, run remote setup + smoke
#   scripts/vast_launch.sh sync <RUN>                   # start checkpoint sync loop (foreground)
#   GO=1 scripts/vast_launch.sh train <RUN> [SUPERBATCHES] # start the real training in tmux on the box
#   scripts/vast_launch.sh status <RUN>                 # tail training log
#   GO=1 scripts/vast_launch.sh destroy <RUN>           # pull final checkpoint, destroy instance
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="nvidia/cuda:12.4.1-devel-ubuntu22.04"
DISK="${DISK:-120}"
cmd="${1:-}"; shift || true

need_go() { [[ "${GO:-0}" == "1" ]] || { echo "REFUSING: this step costs money. Re-run with GO=1 only after the human said go."; exit 3; }; }
run_file() { echo "$ROOT/runs/$1/instance.env"; }
load_run() { # shellcheck disable=SC1090
    source "$(run_file "$1")"; }

do_setup() { # RUN (instance.env must have HOST/PORT)
    local RUN="$1"; load_run "$RUN"
echo "=== prepare box (rsync is not in the CUDA image)"
    ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$HOST" "mkdir -p /workspace/chess && (command -v rsync >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null))"
    echo "=== upload repo"
    rsync -az --exclude target --exclude 'data/downloads' --exclude 'data/*.bin' --exclude 'data/*.binpack' --exclude 'data/*.log' \
        --exclude sprt --exclude runs --exclude tools --exclude books --exclude syzygy --exclude nets --exclude '*.pgn' --exclude checkpoints \
        -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$ROOT/" root@"$HOST":/workspace/chess/
    echo "=== remote setup + smoke (this takes ~10-15 min)"
    ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$HOST" "RUN_NAME=$RUN ARCH='${ARCH:-v1}' L1='${L1:-1024}' EXTRA_GROUPS='${EXTRA_GROUPS:-}' bash /workspace/chess/scripts/vast_setup_remote.sh"
    echo "=== pull smoke checkpoint and netcheck"
    mkdir -p "$ROOT/runs/$RUN/checkpoints"
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" --include='*/' --include='quantised.bin' --include='*.txt' --include='*.log' --exclude='*' root@"$HOST":/workspace/checkpoints/ "$ROOT/runs/$RUN/checkpoints/"
    for q in "$ROOT/runs/$RUN/checkpoints"/smoke*/quantised.bin; do "$ROOT/target/release/engine" netcheck "$q" || { echo "NETCHECK FAILED on $q"; exit 6; }; done
}

case "$cmd" in
    setup)
        # Re-run upload + remote setup + smoke on an existing instance (idempotent).
        RUN="${1:?run}"; do_setup "$RUN"
        echo "SMOKE OK. Next: scripts/vast_launch.sh sync $RUN (in another terminal), then GO=1 scripts/vast_launch.sh train $RUN"
        ;;
    search)
        vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.98 disk_space>=120 inet_down>=500 cuda_vers>=12.4 rentable=true verified=true' -o 'dph+' --raw 2>/dev/null | python3 -c "
import sys,json
for o in json.load(sys.stdin)[:12]:
    print(f\"id={o['id']:<9} \${o['dph_total']:.3f}/hr cpu={o.get('cpu_cores_effective',0):.0f}c ram={o.get('cpu_ram',0)/1024:.0f}G disk={o.get('disk_space',0):.0f}G down={o.get('inet_down',0):.0f}Mb/s rel={o.get('reliability2',0):.3f} {o.get('geolocation','')}\")"
        ;;
    create)
        need_go
        OFFER="${1:?offer id}"; RUN="${2:?run name}"
        mkdir -p "$ROOT/runs/$RUN"
        echo "$(date -u +%FT%TZ) create offer=$OFFER run=$RUN image=$IMAGE disk=$DISK" >> "$ROOT/runs/$RUN/events.log"
        out=$(vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct --raw 2>/dev/null)
        echo "$out" | tee -a "$ROOT/runs/$RUN/events.log"
        ID=$(echo "$out" | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
        echo "INSTANCE_ID=$ID" > "$(run_file "$RUN")"
        echo "instance $ID created; waiting for ssh..."
        for _ in $(seq 1 60); do
            info=$(vastai show instance "$ID" --raw 2>/dev/null || true)
            st=$(echo "$info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('actual_status',''))" 2>/dev/null || true)
            if [[ "$st" == "running" ]]; then
                # Prefer the direct endpoint (public ip + mapped port 22); the proxy can lag by minutes.
                HOST=$(echo "$info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('public_ipaddr') or d.get('ssh_host',''))")
                PORT=$(echo "$info" | python3 -c "import sys,json; d=json.load(sys.stdin); p=(d.get('ports') or {}).get('22/tcp') or []; print(p[0]['HostPort'] if p else d.get('ssh_port',''))")
                { echo "HOST=$HOST"; echo "PORT=$PORT"; } >> "$(run_file "$RUN")"
                break
            fi
            sleep 10
        done
        load_run "$RUN"
        [[ -n "${HOST:-}" ]] || { echo "instance did not come up; check 'vastai show instances' and destroy it"; exit 1; }
        echo "ssh -p $PORT root@$HOST"
        for _ in $(seq 1 60); do ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$HOST" true 2>/dev/null && break; sleep 10; done
        ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$HOST" true 2>/dev/null || { echo "ssh never came up on $HOST:$PORT; destroy the instance and retry"; exit 2; }
        do_setup "$RUN"
        echo "SMOKE OK. Next: scripts/vast_launch.sh sync $RUN (in another terminal), then GO=1 scripts/vast_launch.sh train $RUN"
        ;;
    sync)
        RUN="${1:?run}"; load_run "$RUN"
        exec "$ROOT/scripts/vast_sync_checkpoints.sh" "$RUN" "$HOST" "$PORT" 120
        ;;
    train)
        need_go
        RUN="${1:?run}"; SB="${2:-400}"; load_run "$RUN"
        # Kill layer 1: on-box self-destruct, armed BEFORE training. Needs a restricted API key
        # (instance_read + instance_write only) in runs/<RUN>/selfdestruct_key.txt, created with
        #   vastai create api-key --name <RUN>-selfdestruct --permission_file <json with {"api":{"instance_read":{},"instance_write":{}}}>
        KEYFILE="$ROOT/runs/$RUN/selfdestruct_key.txt"
        [[ -s "$KEYFILE" ]] || { echo "REFUSING: no restricted key at $KEYFILE; create it first (see comment)"; exit 4; }
        MAXH="${MAX_HOURS:-12}"
        scp -P "$PORT" -o StrictHostKeyChecking=no "$ROOT/scripts/vast_selfdestruct.sh" root@"$HOST":/workspace/vast_selfdestruct.sh
        ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$HOST" "chmod +x /workspace/vast_selfdestruct.sh; pgrep -f vast_selfdestruct >/dev/null || setsid nohup /workspace/vast_selfdestruct.sh $INSTANCE_ID $(cat "$KEYFILE") $MAXH 20 /workspace/runs/$RUN/train.log >/dev/null 2>&1 </dev/null & sleep 2; pgrep -f vast_selfdestruct >/dev/null && echo 'self-destruct ARMED' || { echo 'self-destruct NOT running'; exit 5; }"
        echo "$(date -u +%FT%TZ) train start superbatches=$SB self-destruct armed max_hours=$MAXH" >> "$ROOT/runs/$RUN/events.log"
        ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$HOST" "tmux new-session -d -s train 'ARCH=${ARCH:-v1} L1=${L1:-1024} SB0=${SB0:-40} SB2=${SB2:-60} SAVE_RATE=${SAVE_RATE:-20} bash /workspace/chess/scripts/vast_train_remote.sh $RUN $SB'"
        echo "training started in tmux session 'train' on the box; watch with: scripts/vast_launch.sh status $RUN"
        ;;
    status)
        RUN="${1:?run}"; load_run "$RUN"
        ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$HOST" "tail -n 5 /workspace/runs/$RUN/train.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
        ;;
    destroy)
        need_go
        RUN="${1:?run}"; load_run "$RUN"
        mkdir -p "$ROOT/runs/$RUN/checkpoints"
        rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" --include='*/' --include='quantised.bin' --include='raw.bin' --include='*.log' --include='*.txt' --exclude='*' \
            root@"$HOST":/workspace/checkpoints/ "$ROOT/runs/$RUN/checkpoints/" || echo "final rsync failed"
        rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" root@"$HOST":/workspace/runs/"$RUN"/ "$ROOT/runs/$RUN/remote/" || true
        vastai destroy instance "$INSTANCE_ID"
        echo "$(date -u +%FT%TZ) destroyed instance $INSTANCE_ID" >> "$ROOT/runs/$RUN/events.log"
        echo "Destroyed. Verify billing stopped: vastai show instances. Then record the cost in BUDGET.md."
        ;;
    *) sed -n '2,11p' "$0"; exit 1 ;;
esac
