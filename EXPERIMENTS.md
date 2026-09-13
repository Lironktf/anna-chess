# Experiments ledger

Everything tried on this project, with the outcome, so nothing is retried by accident. Newest at the bottom of each
section. Elo figures are fastchess estimates with 95% error bars; "equal nodes" removes speed from the comparison.
Details live in `runs/*/PLAN.md`, `sprt/verdicts.txt`, `sprt/*.log`.

## Networks (paid runs)

| Net | What | Cost | Result | Status |
|---|---|---|---|---|
| fable-v1 (= v1, shipped `nets/default.bin`) | (768x16hm -> 1024)x2 -> 1x8 SCReLU, 2 months T80 (2024-01/02), 800 SB | $1.87 | ~3577 CCRL blitz by anchors (Stormphrax 8 -167, Stash 37 +238 laptop) | **default net, on the website** |
| anna-v2 (planned, skipped) | v1 arch on 4 months | - | not run: v3 subsumed it | skipped |
| anna-v3 | threats + pawn pairs + psq, L1 1024, multilayer/pairwise, 4 months, 370 SB | $3.50 (incl. $0.35 dead host) | equal nodes +83 vs v1; blitz -20; 60+0.6 +42; 120+1.2 +29 | superseded by v3b |
| anna-v3b | same at L1 512 (half-width rows) | $1.10 | equal nodes +83 vs v1 (same as v3); blitz -29 (box) / -14 (laptop, faster engine); 60+0.6 +53; 120+1.2 +23; vs v3 +32 blitz / level at 60+0.6 | **best net for slow time controls** |
| anna-v4 | v1 inputs + v3 output stack, no threat rows, L1 1024 | running 2026-09-13 | - | in progress |

## Net-side experiments (free)

| Experiment | Result | Verdict |
|---|---|---|
| Material scaling of eval (MatScale) | -11 Elo | rejected |
| 4-bit per-row quantisation of v3b threat rows (post-training, round-to-nearest) | -28 +/- 34 at equal nodes (200 games, 46.0%) | too lossy without quantisation-aware training; parked |
| Threat-row pruning analysis | 15.7% of threat rows are empty (impossible features); the rest are dense (median max weight 96) | no free traffic reduction there |
| Feature-usage histogram (v3b, real search, `--features engine/nnue_profile`) | 23,609 distinct features used; top 512 = 54%, top 2048 = 84%, top 4096 = 93.5%, top 8192 = 98.2% of applied rows | hot set is 2-4 MB: basis for hot/cold layouts and small-table designs |
| Transparent huge pages for the tables (GLIBC_TUNABLES=glibc.malloc.hugetlb=1; THP is already "always" here) | no change (284/299k vs 254/294k) | TLB is not the bottleneck on this laptop |
| Two-tier evaluation (EvalFileFast + TierDepth=4: v3b at interior nodes, v1 in quiescence/shallow nodes; v3b did ~15% of evals, 432k vs 271k nps) | -41 +/- 27 vs v1 at 8+0.08 after 240 games | fails: mixing two evaluators in one tree hurts more than the speed helps; code kept behind the option, idea parked |

## Engine speed (all bench-identical unless noted)

| Change | Effect | Status |
|---|---|---|
| v3 fused row update (`apply_rows`), indexed accumulator stack, cached relative boards | +4% nps (v3) | merged |
| v3 prefetch 0/2/4/16 lines (twice: before and after the split) | no gain, 16 slightly worse | off (2 lines kept, harmless) |
| v1 accumulator stack: no 4 KB zero-fill per push, null-move aliasing, in-place refresh | +10-30% under load | merged |
| Move-picker lists pooled per thread (no 4 KB zero-fill per node) | +5-30% under load | merged |
| No per-node Vec allocations for searched-move lists | small | merged |
| **v3 split accumulator** (psq+pairs with refresh cache; threats incremental except mirror crossings) | full rebuild every 2.5 evals removed; 17-27% fewer CPU-s per identical search | merged (the big one) |
| Both-perspective diff before apply + prefetch all rows | neutral | merged |
| Bitmap set-difference instead of sort+merge | ~+20% (short runs) | merged |
| L2 layer over transposed weights (no horizontal sums) | +10-18% with bitmap diff | merged |
| Fixed-capacity u32 diff lists | neutral | merged |
| Threat-net speed overall | v3b from 0.34x of v1 (box) to ~0.6x (laptop); break-even at blitz needs ~0.75x | ongoing |

## Search features (SPRT at 8+0.08, bounds [0,5] unless noted)

| Feature | Result | Status |
|---|---|---|
| Time management falling-eval fix | +86 +/- 17 | merged |
| TmOptPct 85 / 120 | 85: +4 unresolved; 120: FAIL -16 | default 100 |
| RfpMult 60 | failed | default 45 |
| FutMargin 160 | -16 after 800 games (stopped) | default 119 |
| NmpBase 6, LmpBase 2 | queued earlier, unresolved/neutral | defaults 5 / 3 |
| LmpBase 4 | +7 +/- 8 after 3000 games, LLR +1.1 | unresolved, default 3 |
| Syzygy WDL probing in tree | +9 +/- 8 after 3000 games | on by default (tables needed) |
| ThreatHist (main history split by attacked from/to squares) | +1 +/- 8 after 2880 games | null, off |
| SeTtmHist (Stockfish 2026 singular margins + TT-move history) | +4 +/- 9 after 4 h | unresolved, off |
| ThreatOrder (lesser-piece threat term in quiet ordering) | running (level at 200 games) | pending |
| SPSA of 13 search constants | queued after ThreatOrder | pending |

## Infrastructure lessons

- `pgrep -f` matches the shell that runs it: killed my own shell twice and reported an unarmed self-destruct as armed. Select by exact argv fields.
- vast.ai unverified CPU-only hosts failed 3 of 4 times (ssh never up, never running, container exited); verified hosts with GPUs worked.
- Some hosts drop long rsyncs: `rsync --bwlimit=6000 --partial` per file succeeds where a plain rsync gets "Broken pipe".
- Stormphrax 8 release binary needs GLIBCXX_3.4.31 (toolchain PPA libstdc++6); Weiss needs gcc-13.
- Wall-clock nps is useless while SPRTs run; measure nodes per CPU-second on identical searches.
- Search explosion at depth 16 in one lost position (FEN in runs/anna-v3b/PLAN.md) with v3b: not the accumulator (pre-split
  build identical); NmpBase/RazorMult/LmrBase extremes remove it; none found on 24 book positions. Parked.

## Search-efficiency comparison (16 book positions to depth 13, laptop under load, 2026-09-13)

Anna v1 1.86M nodes / 476k nps; Anna v3b 1.65M / 190k; Stash 37 1.73M / 785k; Stormphrax 8 1.80M / 222k. Same nodes per
depth for all four: the Stormphrax gap (+167) is evaluation quality, not search selectivity; the Stash gap is our net.

## Measurement anchors (CCRL Blitz list, fetched 2026-09-13)

Stockfish 17.1 8CPU 3785 (#1), Stormphrax 8.0.0 3744 (#14), Turbulence 3577 (~#74), ZigQueen 3559 (~#80), Weiss 2.0 8CPU 3425
(#130), Stash 37.0 3421 (#131). Anna v1 ~3577, v3b ~3550 at blitz; +50 over v1 at 60+0.6.
