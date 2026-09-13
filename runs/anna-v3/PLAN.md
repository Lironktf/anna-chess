# Run plan: anna-v3 (threat-input multilayer NNUE)

Status: **DONE 2026-09-12 20:20 local** (instance 50770185, offer 34221023, Korea RTX 4090 + 14-core Xeon, ~$0.45/hr; destroyed 20:49; anna-v3 total $3.50 incl. the dead first host).
Final net: runs/anna-v3/checkpoints/anna-v3-s2-30/quantised.bin (91,247,168 bytes, netcheck --strict ok, trainer float evals match the engine within rounding); raw.bin + optimiser_state kept for a later fine-tune. Full trainer stdout: runs/anna-v3/train.log (1.79M pos/s, 56 s/SB, final running loss 0.0593).
Owner chose option A. Smoke checks passed: engine reads quantised.bin to the trainer's own values (layout proven); 5-SB warmup
shows material learned (missing rook -476/+482). Throughput ~1.6M pos/s (GPU 65%, CPU-bound mapping) = ~65 s/SB.
Schedule: SB0=30 / SB1=310 / SB2=30 (37B samples), ~7 h, ~$3.2 training; cost guard $4.50; self-destruct MAX_HOURS=9.
First attempt on offer 49080480 (instance 50767449) destroyed: dead network (~$0.35 lost).

Previously: engine + trainer BUILT and validated locally.

## What is done (2026-09-12)

- Engine: threat + pawn-pair features ported from bullet's example and proven identical on 200k real
  positions; incremental accumulators verified against from-scratch evaluation on random games (pops, nulls);
  AVX2 kernels bit-identical to scalar; v1/v3 selected automatically by file size; `netcheck` checks the
  incremental path against the reference.
- Trainer: `trainer/src/bin/train_v3.rs` (bullet advanced example adapted), saved layout == engine layout,
  3-stage schedule (warmup / main / pure-WDL fine-tune). Dry run on the mock backend reaches the gradient step.

## The speed problem on this laptop

| | v1 (current) | v3 L1=1024 | v3 L1=768 | v3 L1=512 |
|---|---|---|---|---|
| nps, 1 thread, quiet machine | 915k | 235k | ~265k (est.) | ~330k (est.) |
| net file | 25 MB | 91 MB | 70 MB | 49 MB |

Cause, measured: each move changes ~14 threat rows (both perspectives); every row is a random 1 KB read
from a 66 MB table; this CPU (i5-8250U, ~10 outstanding cache misses) sustains only ~7 GB/s of such reads,
so ~2 us per move is a hardware floor here. Prefetching does not help (verified). Stockfish 19 with its own
168 MB threat net does 530k nps on the same laptop versus >1M on a desktop, so the ratio is expected.
Per-node overhead outside the row traffic is ~2.5 us and can still be cut (~30%), but not below the floor.

Elo arithmetic at blitz on this laptop: ~3.5x fewer nodes costs roughly 100-130 Elo; the architecture is
worth roughly +90-120 (Viridithas: +61 after their ~30% slowdown). Expect about break-even HERE, and a clear
gain on modern hardware (CCRL-class machines, where the rating is defined).

## Options

A. **v3 at L1=1024 (frontier), evaluate on rented CPU** (~$3.5 training + ~$1.5 for a 32-core box to
   play the rating matches at desktop-class memory speed). Total ~$5 of the remaining $8. Best science, best
   world-ranking outcome, weakest on this laptop.
B. **v3 at L1=512**: ~$3.5, faster here (~330k nps), somewhat less accurate; likely still a gain on fast
   hardware and roughly neutral here.
C. **v2.5: multilayer + pairwise without threats** (bullet `4_multi_layer` example, psq inputs only): no
   memory penalty, ~5% slower than v1, expected +20-40 everywhere, ~$3. Safe, not the frontier.

Recommendation: **A**, evaluated on rented hardware, because the goal is world ranking and the laptop is
not representative. If the budget must stay under $5, B.

## Procedure (same as v1)
- `EXTRA_GROUPS=train2 GO=1 scripts/vast_launch.sh create <offer> anna-v3` (4 T80 months, ~31 GB zst).
- Smoke: `train_v3` with SB0=1 SB1=1 SB2=1 BATCHES_PER_SB=50, sync, `engine netcheck` (loads as v3, incremental == reference).
- Real run: stages 40 / 600 / 60 superbatches at 131072 x 763 = 100M positions per superbatch.
  Throughput unknown (feature mapping is CPU-heavy): measure in the smoke run, cap $4.50 via the cost guard.

## Midway sanity checks (free, laptop, fixed nodes so speed does not matter)

| checkpoint | opponent | games | result | Elo |
|---|---|---|---|---|
| anna-v3-s1-60 (stage 1 at 60/310, LR still high) | v1 final (default.bin) | 100 @ 30k nodes/move | +24 -31 =45 | -24 +/- 54 (statistically level) |
| anna-v3-s1-160 (stage 1 at 160/310) | v1 final (default.bin) | 100 @ 30k nodes/move | +34 -15 =51 | +67 +/- 48 |
| **anna-v3-s2-30 (final)** | v1 final (default.bin) | 200 @ 30k nodes/move | +74 -27 =99 | **+83 +/- 32** (sprt/v3final_nodes30k.log) |

A 20%-trained v3 already matches the fully trained v1 at equal nodes (log: sprt/v3mid_s1_60_nodes30k.log).

## Rented CPU box (2026-09-12, instance 50829299, AMD EPYC 7B13 Zen 3, 32 threads, 28 games in parallel, results in runs/cpu-eval/remote/)

Single-thread search speed measured on the box under match load: v1 974k nps, v3 264k nps (ratio 3.7x, same as the laptop).
| match | games | result | Elo |
|---|---|---|---|
| v3 vs v1, 8+0.08 | 400 | +86 -109 =205 | -20 +/- 21 |
