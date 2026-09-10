# trainer

Bullet-based trainer for the engine's NNUE. Pinned to bullet rev `629ee50000b2afb7b3337595401c830d3b1e0f42`
(main, 2026-08-29). Rust toolchain used locally: 1.96.1.

Architecture (must match `engine/src/nnue/mod.rs`): `(768x16hm -> 1024)x2 -> 1x8`, SCReLU, QA=255, QB=64, scale 400.

Configuration is by environment variables (see `src/main.rs`): `DATA` (comma-separated SF binpacks),
`SUPERBATCHES`, `BATCHES_PER_SB`, `BATCH_SIZE`, `LR`, `WDL_START`, `WDL_END`, `THREADS`, `LOADER_THREADS`,
`BUFFER_MB`, `SAVE_RATE`, `OUT_DIR`, `NET_ID`, `RESUME`.

Local smoke test (CPU backend, tiny run):
```
DATA=../data/downloads/test77-jan2022-2tb7p.high-simple-eval-1k.min-v2.binpack SUPERBATCHES=1 BATCHES_PER_SB=20 \
  BATCH_SIZE=1024 cargo run --release
```
GPU run: `cargo build --release --features cuda` on the box (needs CUDA_PATH).
