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
