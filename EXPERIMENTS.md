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
| anna-v3c | v3b resumed (weights + Adam state) for 430 SB on six months (Apr, Jun added) + 30 SB WDL, restart LR 5e-4, L40S on Modal | ~$6 Modal credit | **equal nodes vs v3b: +12 +/- 18** (600 games: +176 -155 =269, 51.75%); midway s1-170 was -78 (mid-anneal) | level-to-slightly-positive; not adopted yet (an SPRT [0,10] decides when the laptop is free). Lesson: v3b was closer to converged than v1@340 (threat inputs learn faster), so "train longer" is worth far less here than on v1; a gentler restart (1e-4) is the only variant worth $4 more |
| anna-v4 | v1 inputs + v3 output stack (pairwise -> 16 -> 32 -> 1), no threat rows, L1 1024, 340 SB on 4 months | $1.62 | **equal nodes -83 +/- 33 vs v1** (200 games, 38%); equal time 8+0.08 on the box -50 +/- 25 (300 games); 40% at s1-80 already; eval scale and inference verified (matches trainer, same |cp| as v1/v3b); loss curve tracked v3b's | **failed**: the +83/node of v3/v3b came from the threat inputs, not the output stack; v1 had 2.4x the positions (800 SB vs 340). **v4 vs v1@340 SB (same training budget) at equal nodes: +12 +/- 33 (120 games)**: the output stack adds nothing measurable at our scale, and v1's extra 460 SB are worth ~+80. Lesson: more superbatches on the same data is worth a lot; v3b (340 SB) has the same headroom |

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
| v1 base speed pass (2026-09-13): fused Finny refresh kernel, attack-table OnceLock fast path, piece-type decode by transmute, unchecked search-stack access | 509k vs 512k nodes/CPU-s at depth 14 (6 runs each): neutral; unchecked stack reverted, the rest kept (bench identical) | profile: v1 NNUE is 40% of time (incremental 19%, Finny refresh 12% at 0.17 refreshes per eval per perspective with 5.5 rows each: inherent to the 2-square king buckets, output 8%); the non-NNUE part alone runs at ~830k nodes/s, i.e. Stash's whole-engine speed, so the base engine is not the gap |
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
| ThreatOrder (lesser-piece threat term in quiet ordering) | +3 +/- 9 after 2000 games (4 h timeout), LLR 0.23 | unresolved, off |
| SPSA of 13 search constants with v3b (sprt/spsa_v3b, 400 iters x 8 games at 8+0.08, 3200 games) | every parameter within 1% of its default (LmrBase 982->986, RfpMult 45->46, TmOptPct 100->97, SeMargin 59->60, LmrCutNode 3000->2960) | no signal at this size: 3200 games is ~1/30 of a real SPSA run; not worth an SPRT. Either the v1-era constants are already near a local optimum for v3b or the run was far too short. A proper tune needs a CPU box (100k games ~ $3.5) |

## Release hardening (2026-09-13 evening, for the rating-list build)

| Check | Before | After |
|---|---|---|
| Repeating control 40 moves / 20 s (CCRL-style moves-to-go), laptop at load 9 | 1 time forfeit in 6 games (1 ms overrun); fastchess "sign mismatch in mate scores" warnings | 0 forfeits, 0 warnings, 0 illegal moves in 20 games |
| 3+0.03 hyper-blitz soak under the same load | (earlier 8+0.08 SPRTs: 9 forfeits in 2037 games, 0.4%) | 0 forfeits, 0 warnings in 40 games |
| Dedicated CPU box, 60+0.6, 4 threads | 0 forfeits in 200 games | - |

Fixes (bench 854588 unchanged, 42 tests): (1) an iteration aborted before its first root move completes no longer prints the
sentinel score, which formatted as "mate 1"; (2) the hash move is put first in the root list so such an abort plays the best
guess instead of the first generated move; (3) the clock is checked every 64 nodes (instead of 1024) within 40 ms of the hard
limit; (4) default Move Overhead 10 -> 20 ms. Moves-to-go allocation checked by hand: 15.4 s per move at 40/15 min from the
start, ~2.5 s of 3 s on the last move of a period.

Embedded default net switched from v1 to v3b for the release build (2026-09-13 22:10): **bench 854588 -> 630353** (deterministic,
checked twice); v1 remains available as `nets/fable-v1-800.bin` via EvalFile.

## Infrastructure lessons

- `pgrep -f` / `pkill -f` match the shell that runs them: killed my own shell three times (last: 2026-09-13, a `pkill -f "ssh ... chmod"`
  inside a command line containing that string). Select by exact argv fields with awk, never by substring.
- The ssh that starts the detached on-box self-destruct can hang after arming (vast_launch.sh and cpu_eval.sh, 2026-09-13): bound it
  with `timeout` and verify arming in a separate ssh.
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

## CCRL 40/15 list (fetched 2026-09-13, 275 engines, entries are 4CPU)

Stockfish 18 3649 (#1), Stormphrax 8.0.0 3634 (#9), Arasan/Renegade 3584 (#40), Motor 3572 (#49), Stash 37.0 3437 (#110), Weiss 2.0 3381 (#131).
The scale is compressed vs blitz (Stormphrax-Stash gap 197 here vs 323 on the blitz list, factor ~0.61). Scaling our blitz
anchors: v1 ~3532 (~#70), v3b ~3565 at slow control (~#55). Top 50 = 3572, top 40 = 3584. Caveat: this list runs 4 threads,
so our Lazy SMP scaling must be verified. Laptop 4-vs-1-thread test (2026-09-13) discarded: 8 cores at load 9, the 4-thread
side never had 4 cores. Moved to the rented-box script (`MATCHES=slow4`: 4 vs 1 threads at 60+0.6, then 4-thread matches vs v1,
Stormphrax 8 and Stash at 60+0.6, the direct 40/15 anchor). **Measured on cpu-eval3 (EPYC 7742, 2026-09-13): v3b 4 threads
vs 1 thread at 60+0.6: +89 +/- 48 (40 games, 62.5%)**: normal Lazy SMP scaling (the usual figure for 4 threads is +80 to +120), so
the 4-CPU list conditions do not penalise us. **v3b vs v1, both 4 threads, 60+0.6: +53 +/- 32 (100 games, 57.5%)**, the same
+53 as single-threaded, so the threat net's slow-control edge carries over to the list's conditions (even at 0.45x of v1's
speed on this EPYC box: 418k vs 926k nps). **v3b vs Stormphrax 8 (3634 on the 40/15 list), both 4 threads, 60+0.6: -108 +/- 30
(100 games, 35%)** -> v3b is about **3525 +/- 35 on the 40/15 scale (~#70)**, ~50 short of the top-50 line (3572). The earlier
"~3565" was a scaled blitz guess and was too optimistic: the list's scale is not as compressed as assumed for this pairing.
Games: runs/cpu-eval3/remote/*.pgn. Box cost $0.71.
