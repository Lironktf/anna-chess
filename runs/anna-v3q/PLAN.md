# anna-v3q: quarter-width threat net (L1 = 256), 2026-09-14

Why: v3 (L1 1024) and v3b (L1 512) were equal per node (+83 vs v1 both); width is not the limit, and the threat path's
speed is what costs v3b at fast controls (0.45-0.6x of v1). Halving again should give ~1.5x speed on the threat rows with
little loss per node. Verification: equal-nodes vs v3b (must be ~0), then equal-time at 8+0.08 and at 4 threads 60+0.6.

Run: Modal L40S, owner go 2026-09-14 09:40; same schedule as v3b (SB 30/310/30, LR 1e-3 -> 1e-6, WDL 0.2 -> 0.5, then WDL
1.0 at 1e-5), six T80 months (Jan-Jun 2024), trainer `train_v3` with L1=256, cap 3 h (~$4 expected). Engine: `v3::w256`
(AnyNet::V3Q, 22,838,336 bytes), random net + netcheck round trip ok, 49 tests pass.

## Log
- 10:35 local: s0-30 pulled, netcheck ok (all evals in range), loss 0.01152 at SB 30 (v3b 0.01099), 5.0M pos/s.
- Speed (architecture only, s0-30 net): 227k nodes/CPU-s vs v3b 185k (+23%), v1 458k; depth 11, 30 UHO positions, laptop under load.
- 11:50 midway: v3q s1-90 vs v3b s1-100, equal nodes 30k, 100 games: **-53 +/- 42** (25-40-35). Warning sign; the +23% speed
  is worth ~+14 at blitz, so a per-node loss beyond ~15 makes 256 a net loss. Final: 200 games vs v3b s2-30, then a blitz SPRT.
- 13:20 train end: 370 SB in ~2.2 h on the L40S (~5.0M pos/s; ~$4.3 Modal). Final anna-v3q-s2-30 pulled (22,838,336 B);
  engine evals match the trainer (24/26, -2072/-2059, 1753/1753, -287/-302, -1416/-1414, 59/60); strict netcheck flags only
  the missing-rook magnitude (-2072 vs -2000 bound, as v4 did). Verdict match: 200 games at 30k nodes vs anna-v3b-s2-30.
