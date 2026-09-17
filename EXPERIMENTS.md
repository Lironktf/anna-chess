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
| anna-v3c | v3b resumed (weights + Adam state) for 430 SB on six months (Apr, Jun added) + 30 SB WDL, restart LR 5e-4, L40S on Modal | ~$6 Modal credit | **equal nodes vs v3b: +12 +/- 18** (600 games: +176 -155 =269, 51.75%); midway s1-170 was -78 (mid-anneal) | SPRT [0,10] at 8+0.08 stopped unresolved at 440 games (+13 +/- 21) to free the cores for the policy net; not adopted. Lesson: v3b was closer to converged than v1@340 (threat inputs learn faster), so "train longer" is worth far less here than on v1; a gentler restart (1e-4) is the only variant worth $4 more |
| anna-v3q | v3b architecture at L1 = 256 (quarter width), same six months and schedule as v3b, from scratch | ~$4.3 Modal | **equal nodes vs v3b: -74 +/- 34** (200 games, 39.5%); midway -53; speed +23% (227k vs 185k nodes/CPU-s) | **failed**: 1024->512 was free per node, 512->256 costs ~74; the speed (~+14 at blitz) cannot pay for it. 512 is the floor for this input set |
| anna-v5 | v3b architecture (L1 512) on Leela Jan-Jun 2024 + Stockfish dfrc_n5000 + nodes5000pv2_UHO (139 GB, sources interleaved in the file order), 800 SB (30/740/30) | ~$9 Modal (owner go 2026-09-14 15:05) | cap hit at s1-670/740 (mixed data streams at 3.6M pos/s, not 5.0M); **s1-670 vs v3b final at equal nodes: +54 +/- 30 (200 games, 57.75%)** before the last 70 SB and the WDL stage; midway +17 | **the first gain since v3b, and the biggest step of the project: new data (Stockfish DFRC + UHO binpacks) is what the nets were missing.** **Blitz SPRT vs v3b at 8+0.08 [0,10]: PASSED, +26 +/- 13 after 976 games, LLR 2.99** (sprt/v5_670_blitz.log; 0 forfeits in 977 games). Adopted as the default net on main (1.1-dev). **But 4 threads, 60+0.6, 60 games on the laptop: -41 +/- 42 (7-14-39)**: then **1 thread 60+0.6 x 200: +5 +/- 28** (51-48-101). Slow control combined (260 games): ~-5 +/- 22, i.e. level with v3b. The gain shrinks with depth (+54/node, +26 blitz, ~0 at 60+0.6): consistent with an unannealed checkpoint (no LR tail, no WDL stage) and with shallow 5000-node Stockfish labels. Kept as main's default (better at blitz, level slow); **not a 40/15 gain yet**; finishing the run (~$1.7) needs owner go |
| anna-v5f | anna-v5 finished (resume from s1-670: 70 SB LR tail to 1e-6 + 30 SB WDL-1.0) | ~$1.8 Modal (owner go 2026-09-15) | **vs v5-s1-670 at equal nodes: +31 +/- 30**; **vs v3b at equal nodes: +78 +/- 30** (70-26-104, 61%); **vs v3b at 60+0.6, 1 thread, 200 games: +42 +/- 25** (62-38-100); **blitz SPRT 8+0.08 [0,10]: PASSED +53 +/- 18 (446 games, LLR 2.95)**; **4 threads 60+0.6, 80 games: +35 +/- 33** (24-16-40); 0 forfeits in all 526 games; **anchor on cpu-eval4 (i9-13900K, 4 threads, 60+0.6, 100 games): -111 +/- 34 vs Stormphrax 8 (9-40-51)**, identical to v3b's -108 | the cut schedule was costing ~30 per node and all of the slow-control gain; v5f is the strongest net at every control; default on main. **The anchor says the +35 over v3b did not transfer against Stormphrax at all: placement stays ~3525 on the 40/15 scale.** Lesson: internal gains vs our previous net overstate gains vs external engines; every net's verdict needs an external anchor |
| anna-v4 | v1 inputs + v3 output stack (pairwise -> 16 -> 32 -> 1), no threat rows, L1 1024, 340 SB on 4 months | $1.62 | **equal nodes -83 +/- 33 vs v1** (200 games, 38%); equal time 8+0.08 on the box -50 +/- 25 (300 games); 40% at s1-80 already; eval scale and inference verified (matches trainer, same |cp| as v1/v3b); loss curve tracked v3b's | **failed**: the +83/node of v3/v3b came from the threat inputs, not the output stack; v1 had 2.4x the positions (800 SB vs 340). **v4 vs v1@340 SB (same training budget) at equal nodes: +12 +/- 33 (120 games)**: the output stack adds nothing measurable at our scale, and v1's extra 460 SB are worth ~+80. Lesson: more superbatches on the same data is worth a lot; v3b (340 SB) has the same headroom |

## Data quality (label checks, 2026-09-16; runs/anna-v6/PLAN.md)

Score agreement with our engine (depth 6, 1500 positions per set), correlation / decisive-result agreement: Leela T80 0.86 /
85%; Stockfish dfrc_n5000 0.96 / 94%; nodes5000pv2_UHO 0.95 / 96%; farseerT75 0.92 / 81%; data_pv-2_diff-100 0.97 / 97%.
The Stockfish-labelled sets agree far better with our search than the Leela set does, which is why they helped so much.

## Transfer test (2026-09-16 14:20, laptop, free): does the internal gain show against an external engine?

Round robin at 60+0.6, 1 thread, UHO: anna-v5f, anna-v3b, Stormphrax 8, 200 games per pairing, same hardware and session
(sprt/transfer.log + transfer2.log). The two 100-game anchors on different boxes gave -111 +/- 34 (v5f) and -108 +/- 30 (v3b).
**20:05 table (578 games, ~193 per pairing):** Stormphrax vs v3b +131 +/- 50; Stormphrax vs v5f +108 +/- 50; v5f vs v3b
+60 +/- 50. v5f is +24 Elo closer to Stormphrax than v3b (+/- ~70): partial transfer (~40% of the head-to-head gain), a lean,
not a verdict. Run 2 continues to ~22:30; Stash pairings (200 games each) run 20:05-23:40 as a second reference.
**22:31 final (both runs, ~700 games, ~232 per pairing):** Stormphrax vs v3b +126 +/- 46; Stormphrax vs v5f +106 +/- 45;
v5f vs v3b +55 +/- 45. Transfer: v5f is +20 Elo closer to Stormphrax than v3b (+/- ~64), i.e. roughly a third of the
head-to-head gain shows up against a stronger, different engine. Verdict stands: data gains are real but shrink externally;
plan every future net verdict on the external anchor, not on the head-to-head number.
**2026-09-17 00:55 Stash reference (build checked):** tools/stash-bot is Stash master 13e0a81 (2026-08-01, id "v37.26"), HCE
evaluation (no net file), built -O3 -flto with native pext, bench 1.1M nps; match settings 60+0.6, 1 thread, Hash 128, UHO,
identical for both sides. v5f vs Stash: +128 -8 =64 (200 games), +241 +/- 34. v3b vs Stash: +112 -9 =79 (200 games), +198 +/- 30. By Stash
v5f is +43 over v3b (head-to-head +55, by Stormphrax +20). CCRL 40/15: Stash 37.0 1CPU 3375, Stormphrax 8 1CPU 3609. The two anchors disagree: by Stash we would be ~3615,
by Stormphrax ~3505. Expected for an HCE reference (NNUE engines beat HCE engines by more than the pooled ratings imply);
Stormphrax stays the conservative anchor, Stash is used only for the v5f-v3b difference (+66 here vs +55 head-to-head).

## Net-side experiments (free)

| Experiment | Result | Verdict |
|---|---|---|
| Material scaling of eval (MatScale) | -11 Elo | rejected |
| 4-bit per-row quantisation of v3b threat rows (post-training, round-to-nearest) | -28 +/- 34 at equal nodes (200 games, 46.0%) | too lossy without quantisation-aware training; parked |
| Threat-row pruning analysis | 15.7% of threat rows are empty (impossible features); the rest are dense (median max weight 96) | no free traffic reduction there |
| Feature-usage histogram (v3b, real search, `--features engine/nnue_profile`) | 23,609 distinct features used; top 512 = 54%, top 2048 = 84%, top 4096 = 93.5%, top 8192 = 98.2% of applied rows | hot set is 2-4 MB: basis for hot/cold layouts and small-table designs |
| Transparent huge pages for the tables (GLIBC_TUNABLES=glibc.malloc.hugetlb=1; THP is already "always" here) | no change (284/299k vs 254/294k) | TLB is not the bottleneck on this laptop |
| Two-tier evaluation (EvalFileFast + TierDepth=4: v3b at interior nodes, v1 in quiescence/shallow nodes; v3b did ~15% of evals, 432k vs 271k nps) | -41 +/- 27 vs v1 at 8+0.08 after 240 games | fails: mixing two evaluators in one tree hurts more than the speed helps; code kept behind the option, idea parked |

## Policy net for move ordering (Leela-search distillation, 2026-09-13; runs/policy/PLAN.md)

| Step | Result |
|---|---|
| Data: 25 Lc0 T80 tars (Aug 2025) through `lc0conv`, every policy/best/played move verified legal | 118.6M positions, 99.995% kept |
| Net policy-v1 (768->256 accumulator + per-move dot, PyTorch on an L4, ~$2) | epoch 0: held-out top-1 31.6%, top-3 57.1% (random ~4%) |
| Engine `policycheck` vs trainer logits | mean diff 0.02 logits, top-1 agreement 97-98% (quantisation only) |
| Nodes to depth 11, 30 UHO positions, v3b: PolicyScale 0 / 1000 / 2000 / 4000 / 8000 | geo-mean node ratio 1.000 / 0.845 / **0.800** / 0.824 / 0.903; CPU time lower too |
| SPRT PolicyScale=2000 vs 0 at 8+0.08 [0,5], eager scalar implementation (sprt/policy2000.log) | **-42 +/- 27 after 200 games (44%)**, stopped: far worse than the measured 8-10% per-node cost explains (~-7); the ordering term leaks into search behaviour beyond ordering (see next rows) |
| Lazy accumulator + AVX2 dot product (2026-09-13 22:30) | per-node CPU cost gone (equal to baseline at depth 12 on 60 positions); nodes to depth 12: 0.986 (scale 2000), 0.906 (scale 3000) |
| policy-v1 epoch 1 (annealed) | held-out top-1 32.5%, top-3 57.5% (epoch 0: 31.6 / 57.1); file runs/policy/policy-v1.bin |
| SPRT PolicyScale=500 vs 0 at 8+0.08 [0,5], lazy binary (sprt/policy500.log) | -4 +/- 20 after 440 games, stopped: neutral |
| Fixed depth 8, PolicyScale=2000 vs 0, 119 games (sprt/policy_depth8.pgn) | +18 +/- 64 (52.5%): the ordering term does not damage the tree at equal depth |
| SPRT PolicyScale=2000 vs 0, annealed net, lazy binary, 8+0.08 [0,5] (sprt/policy2000b.log) | **-9 +/- 8 after 2520 games** (48.65%), LLR -2.02 at the 5 h cap: fail |
| SPRT PolicyLmr=150 vs 0 (policy-guided LMR), 8+0.08 [0,5] (sprt/policylmr150.log) | **-10 +/- 8 after 2520 games** (48.53%), LLR -2.12: fail |
| **Verdict (2026-09-14)** | A static 32%-top-1 prior neither orders nor reduces better than the engine's own histories at blitz; the fewer-nodes-to-depth signal did not convert to Elo. Shelved: code stays behind PolicyFile/PolicyScale/PolicyLmr (default off). Untested: root ordering / time management by policy confidence, slower time controls, a threat-input policy net |

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
| ThreadVote (Lazy SMP: final move by weighted vote over threads, Stockfish scheme; default on) | non-regression SPRT [-5,0] at **4 threads**, 10+0.1: **+5.5 +/- 8 after 2103 games** (LLR 1.10; stopped to free the cores, the 95% interval excludes -5) | kept on: not a regression, likely a small gain |
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

Embedded default net switched from v3b to anna-v5-s1-670 on main (2026-09-14 23:55, version 1.1-dev): bench 630353 -> 821474; then anna-v5f-s2-30 (2026-09-15 10:55): bench -> 650059 (deterministic, checked twice). The v1.0 release assets keep v3b.

## Infrastructure lessons

- `pgrep -f` / `pkill -f` / `awk '$0 ~ /pattern/'` over `ps` output match the shell that runs them (the pattern text is in that
  shell's own argv): killed my own shell four times (last: 2026-09-13 22:45). Rule: first list candidates read-only
  (`ps -eo pid,args | grep ...`), then kill the printed PIDs explicitly, never pattern-kill in one command.
- The ssh that starts the detached on-box self-destruct can hang after arming (vast_launch.sh and cpu_eval.sh, 2026-09-13): bound it
  with `timeout` and verify arming in a separate ssh.
- bullet at rev 629ee50 links `cuFuncLoad` (CUDA 12.4 driver API): a host with NVIDIA driver 535 (CUDA 12.2) fails to link
  (2026-09-16, instance 51236656, ~$0.20 lost). Filter vast offers with `driver_version>=550`.
- vast.ai's `disk_space` in an offer is the host's free disk; the container gets `--disk N` at creation (launcher default 120 GB).
  A 269 GB dataset needs DISK=480 (2026-09-16, ~$0.75 lost on a 120 GB container).
- **vast.ai hosts bill internet transfer per GB** (`inet_down_cost`, $0.001-0.04/GB) and storage per GB-month, on top of
  `dph_total`; the hourly cost guard cannot see it. The Hong Kong 4090 host (2026-09-16) charged ~$2.7 for ~70 GB of
  downloads on top of ~$0.8 of hourly time. Rule: filter offers by `inet_down_cost <= 0.002` and add data_GB * rate to the cap.
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
