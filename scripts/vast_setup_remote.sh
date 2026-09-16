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
spd=$(curl -sL -o /dev/null -w '%{speed_download}' -r 0-30000000 --max-time 60 "https://huggingface.co/datasets/linrock/test80-2024/resolve/main/test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst" || echo 0)
echo "network preflight: HuggingFace download ${spd%.*} B/s"
if [[ "${spd%.*}" -lt 2000000 ]]; then echo "NETWORK TOO SLOW (< 3 MB/s), ABORT: destroy this instance and pick another host"; exit 7; fi
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

echo "=== early GPU smoke (1 GB validate set): proves the trainer runs on this GPU/driver before the big download"
cd "$REPO"
bash scripts/download_data.sh run validate
VDATA=$(ls -1 "$REPO"/data/downloads/*.high-simple-eval-1k*.binpack 2>/dev/null | paste -sd, -)
[[ -n "$VDATA" ]] || { echo "validate set missing, abort"; exit 1; }
( cd "$REPO/trainer" && DATA="$VDATA" NET_ID=gpusmoke SB0=1 SB1=1 SB2=1 BATCHES_PER_SB=20 SAVE_RATE=1 THREADS=4 L1="${L1:-1024}" \
    LOADER_THREADS=4 BUFFER_MB=512 OUT_DIR="$WORK/checkpoints" timeout 600 "./target/release/train_${ARCH:-v1}" 2>&1 | tail -8 ) \
    || { echo "EARLY GPU SMOKE FAILED: this host cannot run the trainer; destroy it"; exit 1; }
ls "$WORK/checkpoints"/gpusmoke-*/quantised.bin >/dev/null 2>&1 || { echo "EARLY GPU SMOKE produced no checkpoint; destroy the host"; exit 1; }
echo "=== early GPU smoke ok"

echo "=== data"
cd "$REPO"
bash scripts/download_data.sh run train ${EXTRA_GROUPS:-}
for z in data/downloads/*.zst; do
    [[ -f "$z.ok" ]] || { echo "unverified $z, abort"; exit 1; }
    out="data/$(basename "${z%.zst}")"
    [[ -f "$out" ]] || { echo "decompressing $z"; zstd -d -T8 -q "$z" -o "$out" && rm -f "$z"; }   # keep only the decompressed copy (disk)
done
# Plain (non-zstd) binpacks, e.g. Stockfish's published sets: link the verified downloads into data/.
for b in data/downloads/*.binpack; do
    [[ -f "$b" ]] || continue
    [[ -f "$b.ok" ]] || { echo "unverified $b, abort"; exit 1; }
    [[ -e "data/$(basename "$b")" ]] || ln -s "$(realpath "$b")" "data/$(basename "$b")"
done
ls -la data/*.binpack
DATA_LIST=$(ls -1 "$REPO"/data/*.binpack | paste -sd, -)
echo "DATA=$DATA_LIST" > "$WORK/runs/$RUN_NAME/data.env"

echo "=== inspect real data (format + feature cross-check)"
for f in "$REPO"/data/*.binpack; do
    "$REPO/trainer/target/release/inspect" stats "$f" "${INSPECT_ENTRIES:-500000}"
done

echo "=== smoke run"
cd "$REPO/trainer"
if [[ "${ARCH:-v1}" == "v3" || "${ARCH:-v1}" == "v4" ]]; then
    # v3/v4: one superbatch per stage, 50 batches each, save each -> smoke{3,4}-s{0,1,2}-1
    SMOKE="smoke${ARCH#v}"
    DATA="$DATA_LIST" NET_ID="$SMOKE" SB0=1 SB1=1 SB2=1 BATCHES_PER_SB=50 SAVE_RATE=1 THREADS="${MAP_THREADS:-12}" \
        L1="${L1:-1024}" LOADER_THREADS=8 BUFFER_MB=2048 OUT_DIR="$WORK/checkpoints" "./target/release/train_${ARCH}" 2>&1 | tail -25
    ls -la "$WORK/checkpoints"/"$SMOKE"-*/quantised.bin
else
    DATA="$DATA_LIST" NET_ID=smoke SUPERBATCHES=2 BATCHES_PER_SB=200 BATCH_SIZE=16384 SAVE_RATE=1 THREADS=4 \
        LOADER_THREADS=8 BUFFER_MB=2048 OUT_DIR="$WORK/checkpoints" ./target/release/trainer 2>&1 | tail -15
    ls -la "$WORK/checkpoints"/smoke-*/quantised.bin
fi
echo "=== setup done $(date -u +%FT%TZ)"
