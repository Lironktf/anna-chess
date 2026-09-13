# Budget ledger

Hard cap: **$10.00** total on vast.ai. Every rental is recorded here before it starts and after it ends.

| Date | Instance | Purpose | Planned $ | Actual $ | Notes |
|------|----------|---------|-----------|----------|-------|
| 2026-09-10 12:55-18:20 | RTX 4090 offer 47635760 Japan, instance 50512726, $0.394/hr + 80 GB disk | fable-v1 first NNUE, 800 superbatches (runs/fable-v1/PLAN.md) | $2.0-3.7, cap $4.1 | **$1.87** | done; instance destroyed 18:22, instances: 0 (verified 18:23); restricted key deleted |
| 2026-09-12 12:27-20:50 | RTX 4090 offer 49080480 -> instance 50767449 DESTROYED after 45 min (dead network, ~$0.35 lost); retry offer 34221023 -> instance 50770185 (Korea, RTX 4090 + 14-core Xeon E5-2680 v4, ~$0.45/hr all-in), 13:50-20:50 | anna-v3 threat-input net (runs/anna-v3/PLAN.md), option A: SB0=30/SB1=310/SB2=30 at ~1.79M pos/s | ~$3.5, cap $4.50 | **$3.50** (both hosts, credit 7.99 -> 4.49) | done; final checkpoint anna-v3-s2-30 (+raw.bin+optimiser_state) synced and netcheck --strict ok; instance destroyed 20:49, instances: 0 (verified 20:49); restricted key 27830155 deleted |

Account credit on 2026-09-10: **$9.86** (vast.ai, liron account). Spent so far: **$5.37** (fable-v1 $1.87, anna-v3 $3.50). Remaining: **$4.49** (credit read from the account 2026-09-12 20:50).

Planned next: CPU rating box for anna-v3 vs v1/Stormphrax/Stash, 32-thread EPYC at ~$0.15/hr, 3-5 h = $0.50-0.90. Needs owner go.
