# Run plan: anna-v3 (threat-input multilayer NNUE)

Status: **engine + trainer BUILT and validated locally. NOT launched. Needs the owner's decision on width and a go.**

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
