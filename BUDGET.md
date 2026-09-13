# Budget ledger

Hard cap: **$10.00** total on vast.ai. Every rental is recorded here before it starts and after it ends.

| Date | Instance | Purpose | Planned $ | Actual $ | Notes |
|------|----------|---------|-----------|----------|-------|
| 2026-09-10 12:55-18:20 | RTX 4090 offer 47635760 Japan, instance 50512726, $0.394/hr + 80 GB disk | fable-v1 first NNUE, 800 superbatches (runs/fable-v1/PLAN.md) | $2.0-3.7, cap $4.1 | **$1.87** | done; instance destroyed 18:22, instances: 0 (verified 18:23); restricted key deleted |
| 2026-09-12 12:27-20:50 | RTX 4090 offer 49080480 -> instance 50767449 DESTROYED after 45 min (dead network, ~$0.35 lost); retry offer 34221023 -> instance 50770185 (Korea, RTX 4090 + 14-core Xeon E5-2680 v4, ~$0.45/hr all-in), 13:50-20:50 | anna-v3 threat-input net (runs/anna-v3/PLAN.md), option A: SB0=30/SB1=310/SB2=30 at ~1.79M pos/s | ~$3.5, cap $4.50 | **$3.50** (both hosts, credit 7.99 -> 4.49) | done; final checkpoint anna-v3-s2-30 (+raw.bin+optimiser_state) synced and netcheck --strict ok; instance destroyed 20:49, instances: 0 (verified 20:49); restricted key 27830155 deleted |

| 2026-09-12 21:00-22:29 | CPU box offer 50375418 -> instance 50829299 (Norway, 32 threads of an AMD EPYC 7B13, $0.14/hr) | cpu-eval: rating matches for anna-v3 vs v1 / Stormphrax 8 / Stash / Weiss / Stockfish 19 (runs/anna-v3/PLAN.md) | $0.50-0.90, cap $1.00 | **$0.20** (credit 4.49 -> 4.29) | done; instance destroyed 22:29, instances: 0 (verified 22:29); restricted key 27892172 deleted |

| 2026-09-13 00:20-02:40 | RTX 4090 offer 29761291 -> instance 50845524 (Quebec, EPYC 9554 32 thr, $0.50/hr all-in) | anna-v3b half-width threat net (runs/anna-v3b/PLAN.md), owner go 00:15 | ~$2.0-2.5, cap $3.90 | **$1.10** (credit 4.29 -> 3.19) | done; 370 SB at 6.7M pos/s (~15 s/SB); final anna-v3b-s2-30 (+raw/optimiser) synced, netcheck --strict ok, evals match trainer; destroyed 02:39, instances: 0 (verified 02:39); key 27906658 deleted |

| 2026-09-13 02:50-04:37 | validation CPU box: 3 dead hosts (50855884, 50858124, 50858985, destroyed) then offer 48453746 -> instance 50859195 (Nevada, Xeon E5-2686 v4 36 thr, $0.16/hr) | anna-v3b validation matches (runs/anna-v3b/PLAN.md), the one CPU rental the owner permitted | ~$0.3, cap $0.80 | **$0.27** (credit 3.19 -> 2.92) | done; destroyed 04:37, instances: 0 (verified 04:37); key 27919961 deleted |

| 2026-09-13 11:10 (planned) | RTX 4090 offer 48580533 (Norway, EPYC 7742 32 thr, ~$0.47/hr all-in) | anna-v4 multilayer net without threat inputs (runs/anna-v4/PLAN.md), owner go 10:50 | ~$1.0-1.5, cap $2.50 | - | last paid run in the budget |

Account credit on 2026-09-10: **$9.86** (vast.ai, liron account). Spent so far: **$6.94** (fable-v1 $1.87, anna-v3 $3.50, cpu-eval $0.20, anna-v3b $1.10, cpu-eval2 $0.27). Remaining: **$2.92** (credit read from the account 2026-09-13 04:37).

Candidate next spend: anna-v3b half-width retrain (L1=512, ~$3.5) to cut the threat-net speed penalty. Needs owner go.
