#!/usr/bin/env bash
# Runs ON the rented vast.ai instance (nvidia/cuda devel image). Idempotent.
# The laptop uploads the repo (minus data/target) to /workspace/chess first, then runs:
#   ssh -p PORT root@HOST 'RUN_NAME=fable-v1 bash /workspace/chess/scripts/vast_setup_remote.sh'
# Steps: system deps, Rust, build trainer with cuda, verified data download from the manifest,
# decompress, `inspect stats` on the real data, smoke run (2 superbatches) into /workspace/checkpoints/smoke.
set -euo pipefail
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
WORK=/workspace
REPO=$WORK/chess
mkdir -p "$WORK/runs/$RUN_NAME" "$WORK/checkpoints" "$REPO/data"
LOG="$WORK/runs/$RUN_NAME/setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== setup start $(date -u +%FT%TZ) run=$RUN_NAME host=$(hostname)"

nvidia-smi || { echo "NO GPU VISIBLE, ABORT"; exit 1; }
# Network preflight: the data pull is 15-31 GB; refuse to continue on a box that cannot download.
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl >/dev/null)
spd=$(curl -s -o /dev/null -w '%{speed_download}' -r 0-30000000 --max-time 60 "https://huggingface.co/datasets/official-stockfish/master-smallnet-binpacks/resolve/main/test77-jan2022-2tb7p.high-simple-eval-1k.min-v2.binpack" || echo 0)
echo "network preflight: HuggingFace download ${spd%.*} B/s"
if [[ "${spd%.*}" -lt 3000000 ]]; then echo "NETWORK TOO SLOW (< 3 MB/s), ABORT: destroy this instance and pick another host"; exit 7; fi
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq zstd rsync curl git build-essential pkg-config tmux > /dev/null

if ! command -v cargo > /dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rustc --version; cargo --version
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
[[ -d "$CUDA_PATH" ]] || { echo "CUDA_PATH $CUDA_PATH missing"; exit 1; }
nvcc --version | tail -1

echo "=== build trainer (cuda)"
cd "$REPO/trainer"
cargo build --release --features cuda 2>&1 | tail -3
ls -la target/release/trainer target/release/inspect

echo "=== data"
cd "$REPO"
bash scripts/download_data.sh run train ${EXTRA_GROUPS:-}
for z in data/downloads/*.zst; do
    [[ -f "$z.ok" ]] || { echo "unverified $z, abort"; exit 1; }
    out="data/$(basename "${z%.zst}")"
    [[ -f "$out" ]] || { echo "decompressing $z"; zstd -d -T8 -q "$z" -o "$out"; }
done
ls -la data/*.binpack
DATA_LIST=$(ls -1 "$REPO"/data/*.binpack | paste -sd, -)
echo "DATA=$DATA_LIST" > "$WORK/runs/$RUN_NAME/data.env"

echo "=== inspect real data (format + feature cross-check)"
for f in "$REPO"/data/*.binpack; do
    "$REPO/trainer/target/release/inspect" stats "$f" 3000000
done

echo "=== smoke run"
cd "$REPO/trainer"
if [[ "${ARCH:-v1}" == "v3" ]]; then
    # v3: one superbatch per stage, 50 batches each, save each -> smoke3-s{0,1,2}-1
    DATA="$DATA_LIST" NET_ID=smoke3 SB0=1 SB1=1 SB2=1 BATCHES_PER_SB=50 SAVE_RATE=1 THREADS="${MAP_THREADS:-12}" \
        LOADER_THREADS=8 BUFFER_MB=2048 OUT_DIR="$WORK/checkpoints" ./target/release/train_v3 2>&1 | tail -25
    ls -la "$WORK/checkpoints"/smoke3-*/quantised.bin
else
    DATA="$DATA_LIST" NET_ID=smoke SUPERBATCHES=2 BATCHES_PER_SB=200 BATCH_SIZE=16384 SAVE_RATE=1 THREADS=4 \
        LOADER_THREADS=8 BUFFER_MB=2048 OUT_DIR="$WORK/checkpoints" ./target/release/trainer 2>&1 | tail -15
    ls -la "$WORK/checkpoints"/smoke-*/quantised.bin
fi
echo "=== setup done $(date -u +%FT%TZ)"
