#!/usr/bin/env bash
# Laptop-side orchestration for a rented CPU rating box (no GPU). Paid steps need GO=1.
#
#   GO=1 scripts/cpu_eval.sh create <OFFER_ID> <RUN>   # rent (PAID), upload engine source + nets + books + binaries
#   GO=1 scripts/cpu_eval.sh run <RUN>                 # arm self-destruct, start matches in tmux, start guard + sync
#   scripts/cpu_eval.sh status <RUN>                   # tail the box log
#   GO=1 scripts/cpu_eval.sh destroy <RUN>             # final pull, destroy, verify
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"   # known-good with vast ssh + build tools
DISK="${DISK:-40}"
CAP="${CAP:-1.00}"; MAX_HOURS="${MAX_HOURS:-3}"
NET_V3="${NET_V3:-$ROOT/runs/anna-v3/checkpoints/anna-v3-s2-30/quantised.bin}"
NET_V1="${NET_V1:-$ROOT/nets/default.bin}"
cmd="${1:-}"; shift || true
need_go() { [[ "${GO:-0}" == "1" ]] || { echo "REFUSING: this step costs money. Re-run with GO=1 only after the human said go."; exit 3; }; }
run_file() { echo "$ROOT/runs/$1/instance.env"; }
load_run() { # shellcheck disable=SC1090
    source "$(run_file "$1")"; }
SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=15)

case "$cmd" in
    create)
        need_go
        OFFER="${1:?offer id}"; RUN="${2:?run name}"
        [[ -f "$NET_V3" && -f "$NET_V1" ]] || { echo "nets missing: $NET_V3 / $NET_V1"; exit 1; }
        mkdir -p "$ROOT/runs/$RUN"
        echo "$(date -u +%FT%TZ) create offer=$OFFER run=$RUN image=$IMAGE disk=$DISK cap=$CAP" >> "$ROOT/runs/$RUN/events.log"
        out=$(vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct --raw 2>/dev/null)
        echo "$out" | tee -a "$ROOT/runs/$RUN/events.log"
        ID=$(echo "$out" | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
        echo "INSTANCE_ID=$ID" > "$(run_file "$RUN")"
        echo "instance $ID created; waiting for ssh..."
        for _ in $(seq 1 60); do
            info=$(vastai show instance "$ID" --raw 2>/dev/null || true)
            st=$(echo "$info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('actual_status',''))" 2>/dev/null || true)
            if [[ "$st" == "running" ]]; then
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
        for _ in $(seq 1 60); do ssh -p "$PORT" "${SSH_OPTS[@]}" root@"$HOST" true 2>/dev/null && break; sleep 10; done
        ssh -p "$PORT" "${SSH_OPTS[@]}" root@"$HOST" true 2>/dev/null || { echo "ssh never came up on $HOST:$PORT; destroy the instance and retry"; exit 2; }
        "$0" upload "$RUN"
        ;;
    upload)
        RUN="${1:?run}"; load_run "$RUN"
        echo "=== prepare box"
        ssh -p "$PORT" "${SSH_OPTS[@]}" root@"$HOST" "mkdir -p /workspace/chess /workspace/nets /workspace/books /workspace/tools /workspace/runs/$RUN && (command -v rsync >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null))"
        echo "=== upload engine source"
        rsync -az --exclude target --exclude data --exclude sprt --exclude runs --exclude tools --exclude books --exclude syzygy --exclude nets \
            --exclude '*.pgn' --exclude checkpoints --exclude play/games.sqlite --exclude trainer/target \
            -e "ssh -p $PORT ${SSH_OPTS[*]}" "$ROOT/" root@"$HOST":/workspace/chess/
        echo "=== upload nets, book, prebuilt binaries (stormphrax 8 release, stockfish 19 universal)"
        rsync -az --info=progress2 -e "ssh -p $PORT ${SSH_OPTS[*]}" "$NET_V3" root@"$HOST":/workspace/nets/anna-v3.bin
        rsync -az -e "ssh -p $PORT ${SSH_OPTS[*]}" "$NET_V1" root@"$HOST":/workspace/nets/anna-v1.bin
        rsync -az -e "ssh -p $PORT ${SSH_OPTS[*]}" "$ROOT/books/UHO_Lichess_4852_v1.epd" root@"$HOST":/workspace/books/
        rsync -az --info=progress2 -e "ssh -p $PORT ${SSH_OPTS[*]}" "$ROOT/tools/stormphrax/stormphrax" "$ROOT/tools/stockfish/stockfish" root@"$HOST":/workspace/tools/
        ssh -p "$PORT" "${SSH_OPTS[@]}" root@"$HOST" "chmod +x /workspace/tools/stormphrax /workspace/tools/stockfish; ls -la /workspace/nets /workspace/tools; sha256sum /workspace/nets/anna-v3.bin"
        sha256sum "$NET_V3"
        echo "UPLOAD OK. Next: GO=1 scripts/cpu_eval.sh run $RUN"
        ;;
    run)
        need_go
        RUN="${1:?run}"; load_run "$RUN"
        KEYFILE="$ROOT/runs/$RUN/selfdestruct_key.txt"
        if [[ ! -s "$KEYFILE" ]]; then
            echo "=== creating restricted API key (instance_read+instance_write) for the on-box self-destruct"
            out=$(vastai create api-key --name "$RUN-selfdestruct" --permission_file "$ROOT/runs/anna-v3/selfdestruct_perms.json" 2>&1)
            echo "$out" | python3 -c "
import sys,ast,re
t=sys.stdin.read()
m=re.search(r'\{.*\}', t, re.S); d=ast.literal_eval(m.group(0)) if m else {}
key=d.get('key') or d.get('api_key') or ''
kid=d.get('id') or ''
open('$KEYFILE','w').write(key+'\n'); open('$ROOT/runs/$RUN/selfdestruct_key_id.txt','w').write(str(kid)+'\n')
print('key id', kid, 'len', len(key))"
            [[ -s "$KEYFILE" ]] || { echo "could not create restricted key: $out"; exit 4; }
        fi
        scp -P "$PORT" "${SSH_OPTS[@]}" "$ROOT/scripts/vast_selfdestruct.sh" root@"$HOST":/workspace/vast_selfdestruct.sh
        # Verify through the self-destruct's own log line; `pgrep -f` would match this ssh command line itself.
        ssh -p "$PORT" "${SSH_OPTS[@]}" root@"$HOST" "chmod +x /workspace/vast_selfdestruct.sh; grep -q '^.* armed:' /workspace/selfdestruct.log 2>/dev/null && ps -eo args | grep -q '^bash /workspace/vast_selfdestruct.sh' || setsid nohup /workspace/vast_selfdestruct.sh $INSTANCE_ID $(cat "$KEYFILE") $MAX_HOURS 20 /workspace/runs/$RUN/train.log >/dev/null 2>&1 </dev/null & sleep 3; ps -eo args | grep -q '^bash /workspace/vast_selfdestruct.sh' && grep -q 'armed:' /workspace/selfdestruct.log && { echo 'self-destruct ARMED:'; tail -n 1 /workspace/selfdestruct.log; } || { echo 'self-destruct NOT running'; exit 5; }"
        echo "$(date -u +%FT%TZ) run start self-destruct armed max_hours=$MAX_HOURS cap=$CAP" >> "$ROOT/runs/$RUN/events.log"
        ssh -p "$PORT" "${SSH_OPTS[@]}" root@"$HOST" "command -v tmux >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null); chmod +x /workspace/chess/scripts/cpu_eval_remote.sh; tmux new-session -d -s eval 'RUN_NAME=$RUN CONC=${CONC:-28} GAMES_FAST=${GAMES_FAST:-400} GAMES_LONG=${GAMES_LONG:-100} bash /workspace/chess/scripts/cpu_eval_remote.sh'"
        # Kill layer 2 (laptop cost guard) and the results sync loop, both detached.
        # (no pgrep guards here: `pgrep -f` would match this script's own command line; start, then verify by log)
        (setsid nohup "$ROOT/scripts/vast_cost_guard.sh" "$RUN" "$CAP" 60 >/dev/null 2>&1 </dev/null &)
        (setsid nohup bash -c "exec -a cpu_eval_sync_$RUN bash -c 'while true; do rsync -az -e \"ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=20\" root@$HOST:/workspace/runs/$RUN/ $ROOT/runs/$RUN/remote/ >/dev/null 2>&1; sleep 120; done'" >/dev/null 2>&1 </dev/null &)
        sleep 4; tail -n 1 "$ROOT/runs/$RUN/cost_guard.log" || { echo "cost guard did not start"; exit 6; }
        echo "matches started in tmux 'eval'; guard cap \$$CAP; results sync to runs/$RUN/remote/ every 2 min. Watch: scripts/cpu_eval.sh status $RUN"
        ;;
    status)
        RUN="${1:?run}"; load_run "$RUN"
        ssh -p "$PORT" "${SSH_OPTS[@]}" root@"$HOST" "tail -n 8 /workspace/runs/$RUN/train.log; uptime"
        ;;
    destroy)
        need_go
        RUN="${1:?run}"; load_run "$RUN"
        rsync -az -e "ssh -p $PORT ${SSH_OPTS[*]}" root@"$HOST":/workspace/runs/"$RUN"/ "$ROOT/runs/$RUN/remote/" || echo "final rsync failed"
        vastai destroy instance "$INSTANCE_ID"
        echo "$(date -u +%FT%TZ) destroyed instance $INSTANCE_ID" >> "$ROOT/runs/$RUN/events.log"
        pkill -f "vast_cost_guard.sh $RUN" || true; pkill -f "cpu_eval_sync_$RUN" || true
        sleep 8; echo "instances now:"; vastai show instances
        ;;
    *) sed -n '2,8p' "$0"; exit 1 ;;
esac
