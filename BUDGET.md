# Budget ledger

Hard cap: **$10.00** total on vast.ai. Every rental is recorded here before it starts and after it ends.

| Date | Instance | Purpose | Planned $ | Actual $ | Notes |
|------|----------|---------|-----------|----------|-------|
| 2026-09-10 12:55-18:20 | RTX 4090 offer 47635760 Japan, instance 50512726, $0.394/hr + 80 GB disk | fable-v1 first NNUE, 800 superbatches (runs/fable-v1/PLAN.md) | $2.0-3.7, cap $4.1 | **$1.87** | done; instance destroyed 18:22, instances: 0 (verified 18:23); restricted key deleted |
| 2026-09-12 12:27-20:50 | RTX 4090 offer 49080480 -> instance 50767449 DESTROYED after 45 min (dead network, ~$0.35 lost); retry offer 34221023 -> instance 50770185 (Korea, RTX 4090 + 14-core Xeon E5-2680 v4, ~$0.45/hr all-in), 13:50-20:50 | anna-v3 threat-input net (runs/anna-v3/PLAN.md), option A: SB0=30/SB1=310/SB2=30 at ~1.79M pos/s | ~$3.5, cap $4.50 | **$3.50** (both hosts, credit 7.99 -> 4.49) | done; final checkpoint anna-v3-s2-30 (+raw.bin+optimiser_state) synced and netcheck --strict ok; instance destroyed 20:49, instances: 0 (verified 20:49); restricted key 27830155 deleted |

| 2026-09-12 21:00-22:29 | CPU box offer 50375418 -> instance 50829299 (Norway, 32 threads of an AMD EPYC 7B13, $0.14/hr) | cpu-eval: rating matches for anna-v3 vs v1 / Stormphrax 8 / Stash / Weiss / Stockfish 19 (runs/anna-v3/PLAN.md) | $0.50-0.90, cap $1.00 | **$0.20** (credit 4.49 -> 4.29) | done; instance destroyed 22:29, instances: 0 (verified 22:29); restricted key 27892172 deleted |

| 2026-09-13 00:20-02:40 | RTX 4090 offer 29761291 -> instance 50845524 (Quebec, EPYC 9554 32 thr, $0.50/hr all-in) | anna-v3b half-width threat net (runs/anna-v3b/PLAN.md), owner go 00:15 | ~$2.0-2.5, cap $3.90 | **$1.10** (credit 4.29 -> 3.19) | done; 370 SB at 6.7M pos/s (~15 s/SB); final anna-v3b-s2-30 (+raw/optimiser) synced, netcheck --strict ok, evals match trainer; destroyed 02:39, instances: 0 (verified 02:39); key 27906658 deleted |

| 2026-09-13 02:50-04:37 | validation CPU box: 3 dead hosts (50855884, 50858124, 50858985, destroyed) then offer 48453746 -> instance 50859195 (Nevada, Xeon E5-2686 v4 36 thr, $0.16/hr) | anna-v3b validation matches (runs/anna-v3b/PLAN.md), the one CPU rental the owner permitted | ~$0.3, cap $0.80 | **$0.27** (credit 3.19 -> 2.92) | done; destroyed 04:37, instances: 0 (verified 04:37); key 27919961 deleted |

| 2026-09-13 11:50-13:47 | RTX 4090: offer 48580533 never started (50900782, destroyed after 8 min, ~$0.05), then offer -> instance 50903665 (Texas, RTX 4090, $0.73/hr all-in), 11:54-13:47 | anna-v4 multilayer net without threat inputs (runs/anna-v4/PLAN.md), owner go 10:50 | ~$1.0-1.5, cap $2.50 | **$1.62** (credit 2.92 -> 1.30) | done; 310+30 SB; final anna-v4-s2-30 (+raw.bin, partial optimiser_state) synced, netcheck ok (only the missing-rook magnitude test outside range: -2214 vs [-2000,-100]); destroyed 13:47, instances: 0 (verified 13:47); restricted key 27959390 gone |
| 2026-09-13 13:44-17:57 | CPU box offer 50186808 -> instance 50918711 (Michigan, EPYC 7742 32 effective threads, $0.17/hr billed) | cpu-eval3: 40/15 anchor at 4 threads (SMP scaling, v3b vs v1 / Stormphrax 8 at 60+0.6) + anna-v4 blitz validation, owner go 13:40 | ~$0.5, cap $1.00 then $1.15 | **$0.71** (credit 1.30 -> 0.59) | done; SMP +89, v3b vs v1 +53, v3b vs Stormphrax 8 -108 (all 4 threads, 60+0.6), v4 -50 blitz; destroyed 17:57, instances: 0 (verified 17:58); restricted key deleted |
| 2026-09-16 01:00 (planned) | CPU box offer 51223888 (Japan, i9-13900K 32 threads, $0.134/hr) | cpu-eval4: 4-thread 60+0.6 anchor, anna-v5f vs Stormphrax 8, 100 games, owner go 00:55 | ~$0.35, cap $0.55 (credit $0.58) | - | |

Account credit on 2026-09-10: **$9.86** (vast.ai, liron account). Spent so far: **$9.27** (fable-v1 $1.87, anna-v3 $3.50, cpu-eval $0.20, anna-v3b $1.10, cpu-eval2 $0.27, anna-v4 $1.62, cpu-eval3 $0.71). Remaining: **$0.59** (credit read from the account 2026-09-13 17:58). Modal (separate, owner's $30/month credit): anna-v3c ~$6, policy data + training ~$5, anna-v3q $4.3 (failed), anna-v5 ~$10.7 (hit its 5.5 h cap at SB 670/740) + anna-v5f finish ~$1.8 (2026-09-15, owner go); Modal credit now ~exhausted.

Candidate next spend: anna-v3b half-width retrain (L1=512, ~$3.5) to cut the threat-net speed penalty. Needs owner go.
