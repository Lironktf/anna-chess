#!/usr/bin/env bash
# Serve the browser UI + WebSocket API for playing against Anna on one port.
#
# Usage: scripts/play.sh [--net nets/default.bin] [--port 8080] [--bind 0.0.0.0] [--threads 1] [--hash 64]
#        [--max-movetime 10000] [--max-depth 30] [--max-games 4] [--syzygy syzygy]
# Then open http://localhost:8080 or expose it with:  cloudflared tunnel --url http://localhost:8080
# (a quick tunnel gives a random *.trycloudflare.com URL; WebSockets are proxied without extra config).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -x "$ROOT/target/release/engine" && -x "$ROOT/target/release/play" ]] || (cd "$ROOT" && cargo build --release)
exec "$ROOT/target/release/play" --engine "$ROOT/target/release/engine" "$@"
