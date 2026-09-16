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
