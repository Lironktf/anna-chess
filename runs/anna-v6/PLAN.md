# anna-v6: more new data (plan, 2026-09-16; NOT launched, needs owner go, vast 4090, cap $5)

Same architecture as anna-v5f (L1 512, threats + pawn pairs). Data = v5's set (Leela T80 Jan-Jun 2024, Stockfish dfrc_n5000,
nodes5000pv2_UHO) plus farseerT75 (47 GB, Leela T75 rescored by Stockfish) and data_pv-2_diff-100_nodes-5000 (32 GB).
Schedule 1200 SB (30/1140/30), sources interleaved in the file order.

## Data validation (first 200 MB of each new file, `inspect stats`, 2026-09-16 12:30)

| Set | entries read | kept by filter | results L/D/W | ply 16-40 / 40-80 / 80-120 / 120-200 / >200 | feature cross-checks |
|---|---|---|---|---|---|
| farseerT75 | 2.0M | 56.3% | 300k / 543k / 284k | 290k / 397k / 248k / 147k / 44k | 1126/1126 |
| data_pv-2_diff-100_nodes-5000 | 2.0M | 74.2% | 380k / 727k / 378k | 292k / 455k / 366k / 305k / 66k | 1484/1484 |

Both parse (BINP), decode to legal positions, and match the engine's feature indices. Next check: label quality =
agreement between each set's scores and anna-v5f's evaluation on a sample of positions (inspect dump -> engine eval).

## Label quality (scripts/label_check.py, 1500 positions per set, engine at depth 6, 2026-09-16 13:00)

Correlation between the file's score and the engine's score (both side-to-move, clamped +/-2000), and the share of decisive
game results whose sign matches the file score:

| Set | corr vs v5f | corr vs v3b (Leela-only net) | sign agree (|score|>=100) | result agree |
|---|---|---|---|---|
| Leela T80 Feb 2024 (used) | 0.855 | 0.819 | 96% | 84.8% |
| dfrc_n5000 (used) | 0.961 | 0.957 | 97% | 93.6% |
| nodes5000pv2_UHO (used) | 0.945 | 0.935 | 98% | 96.1% |
| farseerT75 (new) | 0.916 | 0.914 | 97% | 80.6% |
| data_pv-2_diff-100 (new) | 0.968 | 0.964 | 98% | 96.6% |

Verdict: both new sets are at least as consistent as the sets that produced v5's gain; pv-2 is the cleanest of all five.
farseer's lower result agreement matches its Leela-game origin (Leela's own set is 84.8%). The Leela labels agree least
with our search even for the net trained on them, which explains the size of the Stockfish-data gain.

## Launch plan (owner go 2026-09-16 13:20, cap $5)

vast RTX 4090, offer 39099119 (Japan, EPYC 7542 16 cores, 94 GB RAM, 347 GB disk, 783 Mbps, $0.39/h). Data groups
train + train2 + train3 (Leela Jan-Jun 2024, 52 GB zst -> 61 GB) + sf (4 Stockfish binpacks, 156 GB); 269 GB on disk.
`DATA_ORDER` interleaves: jan, dfrc, feb, uho, mar, farseer, apr, pv2, may, jun. ARCH=v3, L1=512, SB 30/940/30 (1000 SB),
SAVE_RATE 10. Estimate: download ~45 min + smoke + ~5 h training = ~$2.5-3.5; cap $5.00 = guard $5.00, self-destruct
MAX_HOURS 12 (12 h x $0.39 = $4.7). Starts after cpu-eval4 is destroyed (one paid instance at a time).

## Launch log
- 14:05 offer 39099119 -> instance 51235738 never left "loading"; destroyed after 15 min (~$0.01).
- 14:12 offer 51224273 already taken (no_such_ask, nothing created).
- 14:14 offer 44637528 -> instance 51236656 running (Texas, EPYC 7402P 12 cores, 768 GB, 929 Mbps), billed ~$0.56/h with
  storage; guard $5.00 armed. Cap arithmetic: 8.8 h total -> setup ~1 h + train MAX_HOURS 7; schedule trimmed to 900 SB (30/840/30).
