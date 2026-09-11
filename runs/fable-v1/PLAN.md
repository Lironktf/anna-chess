# Run plan: fable-v1 (first NNUE)

Status: **COMPLETE 2026-09-10 18:19 UTC-4 (train end 22:19Z), 800 superbatches, cost $1.87, instance destroyed, instances: 0 verified.**
Final net: runs/fable-v1/checkpoints/fable-v1-800/quantised.bin -> nets/default.bin (also nets/fable-v1-800.bin). Strict netcheck ok (start +61).
Previously: RUNNING since 2026-09-10 14:52 local (owner go at 12:38). Instance 50512726, offer 47635760 (Japan 4090, $0.394/hr + disk).
Measured: smoke run 4.5-5.9M pos/s, ~17 s per superbatch; chosen SUPERBATCHES=800 (~3.8 h, ~$1.6 training, ~$2.1 total).

## What gets trained

- Architecture: `(768 x 16 king buckets, horizontally mirrored -> 1024)x2 -> 1 x 8 output buckets`, SCReLU,
  quantised QA=255 QB=64, eval scale 400. Engine and trainer share the layout (`engine/src/nnue/mod.rs`,
  `trainer/src/lib.rs`); verified identical on 300k real positions (`inspect features`).
- Data: Leela T80 2024-01 and 2024-02 binpacks (linrock/test80-2024, `min-v2.v6`), sha256-verified.
  About 3.0B + ~3.4B positions, ~48.9% kept by the standard filter (ply >= 16, not in check, |score| <= 10000,
  best move quiet). Roughly 3B usable positions.
- Labels: `target = wdl * result + (1 - wdl) * sigmoid(score / 400)`, wdl ramps 0.3 -> 0.5.
- Optimiser: AdamW, lr 0.001 with cosine decay to 0.001 * 0.3^5, weight clipping ±0.99 on the feature transformer.
- Schedule: batch 16384, 6104 batches per superbatch (100M positions), `SUPERBATCHES` chosen after the smoke run
  (see below). Checkpoint every 10 superbatches, synced to the laptop every 2 minutes.

## Chosen GPU (as of 2026-09-10 12:20, offers change hourly; re-run `scripts/vast_launch.sh search` before renting)

Offer **49080479**: RTX 4090, AMD EPYC 7K62 with 24 vCPU (the binpack loader is CPU-bound, so cores matter),
reliability 0.997, US, direct SSH, driver 580 (CUDA 13.0 capable, runs the 12.4 image).
Price: $0.307/hr GPU + $0.044/hr for 120 GB disk = **$0.35/hr**, bandwidth $0.003/GB (15 GB of data = $0.05).
Runner-up: 50372982 (Vietnam, same CPU, $0.32 + $0.056/hr, but $0.039/GB bandwidth = $0.60 for the data).

## Cost control

| Step | Time | Cost at $0.35/hr |
|---|---|---|
| Instance setup, Rust + trainer build, data pull (15 GB at datacenter speed) + decompress + inspect | ~20 min | $0.12 + $0.05 bandwidth |
| Smoke run (2 superbatches x 200 batches, save each) + sync + netcheck | ~5 min | $0.03 |
| Real run, 400 superbatches | estimate 45-90 s per superbatch (measured by the smoke run) = 5-10 h | $1.75-3.50 |
| Pull final checkpoint, destroy | ~2 min | $0.01 |
| **Total** | **6-11 h** | **$2.0-3.7**, hard-capped at ~$4.1 by the superbatch rule below |

Decision rule after the smoke run: `SUPERBATCHES = min(400, floor(11 hours / sb_seconds))`. 11 hours = $3.85,
leaving over half the budget for a second run. If `sb_seconds` implies fewer than 150 superbatches, stop and rethink
(smaller net or fewer buckets) before spending.

## Exact procedure (laptop)

```
scripts/vast_launch.sh search                       # pick an offer: verified, >=120 GB disk, fast link, < $0.36/hr
GO=1 scripts/vast_launch.sh create <OFFER_ID> fable-v1   # PAID from here. Uploads repo, builds, pulls data, smoke run, netcheck
scripts/vast_launch.sh sync fable-v1                 # separate terminal, keeps pulling checkpoints
GO=1 scripts/vast_launch.sh train fable-v1 <SUPERBATCHES>
scripts/vast_launch.sh status fable-v1               # tail the log, GPU utilisation
GO=1 scripts/vast_launch.sh destroy fable-v1         # pulls final checkpoint, destroys, then record cost in BUDGET.md
```

## Acceptance for the produced net

1. `engine netcheck --strict runs/fable-v1/checkpoints/<last>/quantised.bin` passes (start position within ±150 cp,
   missing-rook positions have the right sign).
2. Copy to `nets/default.bin`; `engine bench` runs; `scripts/sprt.sh` NNUE build vs HCE build at 8+0.08 shows a
   large positive Elo (expected several hundred).
3. Log the run: training log, schedule, data list, instance type, wall time, final cost -> `runs/fable-v1/`.

## Risks and mitigations

- Loader too slow to feed the GPU (sfbinpack decoding is CPU-bound): pick an offer with >= 12 cores, LOADER_THREADS=8;
  watch GPU utilisation in `status`. If < 60%, convert the data once to bulletformat on the box and use the direct loader.
- Instance dies mid-run: checkpoints are on the laptop every 10 superbatches; resume with `RESUME=<checkpoint dir>`
  on a new instance (optimiser state is in the checkpoint on the box; if lost, restart from the quantised weights is
  not supported, so the sync also pulls `raw.bin`).
- Quantisation overflow (bullet refuses to write `quantised.bin`): feature-transformer weights are clipped to ±0.99 so
  this cannot happen for l0; l1 weights are unconstrained but tiny in practice. The smoke run proves the file is written.

## SPRT log

| Date | Test | TC | Result | Verdict |
|---|---|---|---|---|
| 2026-09-10 | MatScale (eval * (700+mat/16)/1024) vs off, net ck10 | 4+0.04 | -10.8 +/- 17.3 after 840 games, LLR -0.59 | REJECTED (stopped early, clearly negative); default stays off |
| 2026-09-10 | Cuckoo upcoming-repetition on vs off, net ck10, MatScale off | 4+0.04 | +2.5 +/- 13.5 after 1278 games, LLR +0.05 | INCONCLUSIVE, kept ON (correctness feature, matches Stockfish) |
| 2026-09-10 | Final net measurements 8+0.08, 100 games: vs ck10 +215 +/- 62 (67-12-21); vs SF19 UCI_Elo 3100 +139 +/- 66 (60-22-18). UCI_Elo 3300/3500 results INVALID (option max is 3190; out-of-range breaks the limiter -> 100-0). | | | |
| 2026-09-10 | TmFallingFix (time management falling-eval term fixed) on vs off, final net | 8+0.08 | +86 +/- 17 after 620 games, LLR 2.95 | PASSED; default on |
| 2026-09-11 | Ranking with TM fix (fable_v1b), 8+0.08, 1 thread: vs Stash 37 (3420) +238 +/- 44 (200 g); vs Weiss 2.1-dev (~3355) +137 +/- 35 (200 g); vs Stormphrax 8.0.0 (3744) -139 +/- 45 (100 g: 11-49-40); vs Stockfish 19 full -301 +/- 47 (100 g: 0-70-30). LmrScalePct 115 vs 100: 0 after 276 games (stopped, neutral). | | | |
| 2026-09-11 | RfpMult 60 vs 45 | 8+0.08 | -8.5 +/- 6.2 after 4190 games, LLR -2.96 | FAILED; default stays 45 |
