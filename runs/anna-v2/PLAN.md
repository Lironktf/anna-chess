# Run plan: anna-v2 (second NNUE)

Status: **DRAFT, not launched. Needs the owner's explicit "go".**

## Why a second run

fable-v1 (800 superbatches, 2 T80 months, ~3B usable positions) reached roughly 3600 CCRL blitz. The run was
fast (17 s per superbatch) and cheap ($1.87), so a longer schedule over twice the data is the cheapest remaining
gain: expected +30 to +60 Elo from more data and a longer cosine schedule, same architecture, no engine changes.

## Changes from v1

| | v1 | v2 |
|---|---|---|
| Data | T80 2024-01, 2024-02 (~6.3B raw, ~3.0B filtered) | + 2024-03, 2024-05 (~12.8B raw, ~6.2B filtered) |
| Superbatches | 800 (80B samples, ~26 epochs) | 1600 (160B samples, ~26 epochs over 2x data) |
| WDL | 0.3 -> 0.5 | 0.3 -> 0.6 (final-net games showed slight over-optimism in won endgames) |
| LR | 0.001 cosine -> 0.001 * 0.3^5 | same |
| Everything else | | identical (arch, QA/QB, batch, filter) |

Self-play data (6M+ positions from `engine datagen`, bulletformat) is NOT in v2: it is 0.1% of the T80 volume and
the loader takes one format per run. It is kept for a later fine-tune / RL stage (v3) where it can be weighted.

## Cost

Measured v1 throughput on the same class of box: 17 s per superbatch at 5.9M pos/s.
- Setup (Rust, trainer, 31 GB pull at datacenter speed, decompress, inspect, smoke): ~30 min, $0.20
- 1600 superbatches x 17 s = 7.6 h at $0.39-0.41/hr: **$3.10**
- Disk: 4 zst (33 GB) + 4 raw (36 GB) = 69 GB -> rent 100 GB.
- **Total ~$3.40; hard cap $4.00** (cost guard) leaves ~$4.60 of the $7.99 credit.

## Procedure

Identical to v1 (`scripts/vast_launch.sh`), with:
- `data/MANIFEST.tsv` groups `train` + `train2` (both pulled by `vast_setup_remote.sh`: update it to `run train train2`).
- Restricted API key created fresh for the run, self-destruct armed with `MAX_HOURS=10`, cost guard at $4.00.
- Smoke run first, netcheck, then `GO=1 scripts/vast_launch.sh train anna-v2 1600`.

## Acceptance

- `engine netcheck --strict` passes; bench runs.
- 200-game match v2 vs v1 at 8+0.08 shows a positive result (target: > +20, SPRT bounds [0, 10]).
- Then re-anchor vs Stormphrax and Stash.

## Not in this run (deliberately)

- Architecture change (pairwise-mul FT + L2): a separate, riskier project needing new inference code and its own
  validation; planned as v3 once the search tuning has plateaued.
- Self-play/RL data: see above.
