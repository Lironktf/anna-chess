# anna-v5: new data, not more of the same (draft plan, 2026-09-14, NOT launched)

Evidence: v3c (2.4x more of the same Leela months) gave +12 over v3b; v1 800 vs 340 SB gave +80 only because 340 was far from
converged. The nets are data-variety limited. The free second source is Stockfish's own published binpacks
(huggingface.co/datasets/official-stockfish/master-binpacks, labels from Stockfish search at 5000 nodes):

| File | Size | What |
|---|---|---|
| dfrc_n5000.binpack | 37.5 GB | DFRC self-play, 5000 nodes (Stormphrax's recipe: DFRC generalises better) |
| nodes5000pv2_UHO.binpack | 40.3 GB | UHO openings, 5000 nodes, pv 2 |
| farseerT75.binpack | 47.0 GB | Leela T75 rescored by Stockfish |
| data_pv-2_diff-100_nodes-5000.binpack | 31.6 GB | filtered (pv-2, |diff| < 100) |

Run: v3b architecture (L1 512, or 256 if anna-v3q holds per node), data = Leela Jan-Jun 2024 (61 GB, on the volume) +
dfrc_n5000 + nodes5000pv2_UHO (~78 GB more; ~$0.5 to fetch), 800 SB instead of 340 (the data supports it), L40S ~4 h,
~$8-9 of the ~$14 Modal credit that remains after anna-v3q. Smoke first as always. Verification: equal nodes vs v3b, then
4 threads 60+0.6 vs v1 and Stormphrax.
Needs the owner's go (Modal credit) and a decision on width after anna-v3q.

## Go and validation (2026-09-14 15:05)

Owner go for the run and for the v1.0 tag. Data validation on the first 200 MB of each Stockfish file (`inspect stats`):
- dfrc_n5000: 2.0M entries read, 73.2% kept by the filter, ~64M entries/47M kept per 200 MB slice (~12B entries in the
  file), score bins wide but symmetric, results 397k/670k/396k (L/D/W), feature cross-checks 1463/1463 passed, sample
  FENs are valid DFRC positions.
- nodes5000pv2_UHO: 73.6% kept, results 377k/717k/377k, feature cross-checks 1471/1471, valid FENs.
Loader note: bullet's concat loader reads files in the given order every epoch, so the run interleaves sources:
`01,02,sfdfrc,03,04,sfuho,05,06`. Schedule SB 30/740/30 (800 SB), L1 512, LR 1e-3 -> 1e-6, L40S, cap 5.5 h (~$9).
Smoke (SB 2/2/1 on the same file list, checkpoint netcheck) before the real run.

## Log
- 16:05 smoke (SB 2/2/1, mixed list): checkpoint round-trips (45,641,280 B), netcheck ok, evals near zero as expected.
- 16:20 real run launched: net anna-v5, months 01,02,sfdfrc,03,04,sfuho,05,06, SB 30/740/30, L1 512, LR1 1e-3, L40S,
  cap 5.5 h. Expected end ~21:00 local. Monitor polls the volume every 15 min.
- 18:05 midway: anna-v5 s1-190 vs anna-v3b s1-200 at equal nodes (30k, 100 games): **+17 +/- 44** (29-24-47). No disaster from
  the DFRC/UHO data; v5's checkpoint is less annealed (LR still ~7.5e-4 at SB 190 of 740 vs v3b's ~3.5e-4 at 200 of 310).
  Final verdict at s2-30 vs anna-v3b-s2-30, 200 games.
- 22:00 **wall-clock cap hit at s1-670 of 740** (trainer killed, exit -9). Throughput on the mixed list was 3.59M pos/s
  (27.9 s/SB) against 5.0M pos/s on the Leela-only list: the loader streams the two large Stockfish files slower, and my
  5.5 h cap assumed 20 s/SB. Cost ~5.5 h x $1.95 = ~$10.7 (estimate was $9). Last checkpoint s1-670 (LR ~9.5e-5 of the
  1e-3 -> 1e-6 decay), no WDL stage. Being evaluated as is (200 games at equal nodes vs anna-v3b-s2-30). Finishing it
  (RESUME from s1-670, LR1 9.5e-5, SB 0/70/30, ~50 min, ~$1.7) needs the owner's go: Modal credit left ~$1.5.
- 23:25 **verdict: anna-v5 s1-670 vs anna-v3b-s2-30 at equal nodes, 200 games: +54 +/- 30** (66-35-99). Same width/speed as
  v3b. Next: blitz SPRT at 8+0.08 [0,10] (sprt/v5_670_blitz.log); owner decision on finishing the last 70 SB + WDL stage.
- 23:52 **blitz SPRT PASSED: v5 s1-670 vs v3b at 8+0.08, +26 +/- 13 after 976 games, LLR 2.99 [0,10].** 0 time losses.
  Adopted as default net on main (nets/default.bin = anna-v5-s1-670, version 1.1-dev). 4-thread 60+0.6 check running overnight.
- 2026-09-15 03:55 4-thread 60+0.6 vs v3b, 60 games (laptop): **-41 +/- 42** (7-14-39). Contradicts blitz; small sample.
  Next: 1-thread 60+0.6 x 200 (sprt/v5_slow1t.log). Hypothesis: the missing annealing tail + WDL stage costs most at slow
  controls; finishing the run is the cheapest test of that.
- 07:15 1-thread 60+0.6 x 200 vs v3b: **+5 +/- 28** (51-48-101). Slow control combined (260 games incl. the 4-thread 60):
  ~-5 +/- 22 -> level. Blitz +26, per node +54. Reading: the advantage decays with depth. Two candidate causes, both
  fixable: (1) the schedule was cut (no LR tail to 1e-6, no WDL-1.0 stage), (2) 5000-node Stockfish labels are shallow
  relative to Leela's; a finished run and, later, a higher-node label source (or rescoring) are the levers.
- 2026-09-15 (owner go "finish v5"): anna-v5f = RESUME from anna-v5-s1-670 (weights + Adam state), SB 0/70/30, LR 9.5e-5 -> 1e-6
  then WDL-1.0 stage at 1e-5 -> 1e-7, same interleaved data, L40S, cap 1.5 h (~$1.7). Verdicts: equal nodes vs v3b and vs
  v5-s1-670, then 60+0.6.
- 09:05 anna-v5f done (resumed from s1-670, 70 SB tail + 30 SB WDL, ~55 min, ~$1.8). Final anna-v5f-s2-30: netcheck ok, engine
  evals match the trainer (28/29, -1449/-1454, 1901/1930, -505/-509, -2055/-2064, 10/16). Verdicts running: equal nodes vs
  v3b and vs v5-s1-670, 200 games each.
- 10:35 anna-v5f vs anna-v5-s1-670 at equal nodes, 200 games: **+31 +/- 30** (73-55-72). The tail + WDL stage were worth ~30/node.
- 10:50 **anna-v5f vs anna-v3b-s2-30 at equal nodes, 200 games: +78 +/- 30** (70-26-104). Embedded as main's default net.
