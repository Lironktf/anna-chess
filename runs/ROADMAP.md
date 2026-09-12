# Anna roadmap (2026-09-12)

State: ~3600 CCRL blitz (anchored vs Stash, Weiss, Stormphrax, Stockfish). Credit left: $7.99.

## What the frontier looks like (researched 2026-09-12)

Every top engine moved its NNUE to **threat inputs** in the last 12 months: Stockfish SFNNv10 (Nov 2025) through
SFNNv13/16, Stormphrax 8 (threats + pawn-pawn inputs), Viridithas 20 (threats + pawn-pawn, measured **+61 Elo at
60+0.6**), Reckless 0.9, Hobbes. A threat feature is "piece X on square A attacks piece Y on square B"; the net sees
tactics directly instead of inferring them from piece-square inputs. Pawn-pawn inputs (pairs of pawns at most one file
apart) add pawn-structure geometry. Both sit on top of the king-bucketed piece-square inputs we already have.

Reference architecture (Stormphrax 8, `src/eval/arch.h`): psq 16 king buckets mirrored + PawnPawnThreatInputs,
L1 = 1024 with pairwise multiplication and dual activation, L3 = 64, 8 material output buckets, scale 400.
bullet ships a trainer for exactly this family (`examples/advanced/`): inputs.rs defines the threat/pawn-pair feature
indices, main.rs the multilayer model and a 3-stage schedule (warmup at WDL 0.2, main run WDL 0.2->0.5, then a short
pure-WDL fine-tune at tiny LR).

Engine-side cost: incremental update of threat features per move (Viridithas: ~900 lines incl. tests; Stormphrax:
~600 lines of C++). Our incremental-vs-reference test harness is exactly the tool to validate it.

## Decision: skip anna-v2 (same arch, more data), build anna-v3 (threat architecture)

| Option | Expected gain | Cost | Time to result |
|---|---|---|---|
| anna-v2: same net, 4 months, 1600 SB | +30 to 60 | $3.40 | 1 day |
| anna-v3: threats + pawn pairs + multilayer | +60 to 120 | ~$3.5-4.5 | 4-5 days |

v2 and v3 together would nearly exhaust the credit; v3 alone is the better use of the money and is where the field is.
v3 also subsumes v2's data gain (it trains on the same 4 months).

## Plan

### Phase 1 — engine: threat architecture (no cost, 2-3 days)
1. Feature definition ported exactly from bullet `examples/advanced/inputs.rs` (threats + pawn pairs + psq),
   with a cross-check tool like `inspect features` proving engine indices == bullet's on real positions.
2. Incremental threat updates on make/unmake (direct threats of the moved piece, threats onto its squares,
   discovered/blocked slider threats), following Viridithas `threat_updates.rs` / Stormphrax `threats.cpp`.
   Test: after every move in random games, incremental accumulator == from-scratch refresh (same harness as v1).
3. Multilayer inference: pairwise-mul FT (i16 -> i8 via shift), L1 sparse affine, L2/L3 dense, output buckets.
   Test: AVX2 == scalar reference bit-for-bit.
4. Speed check: target >= 700k nps single-thread on this laptop (v1: ~900k). Threat nets trade nps for accuracy.

### Phase 2 — trainer + validation (0.5 day)
- Port bullet's advanced example to our sfbinpack loader and output format; 3-stage schedule.
- Local: feature cross-check on 300k real positions, roundtrip, MockGPU dry run of the loader.
- On-box smoke run (2 superbatches) + netcheck before the real run (same procedure as v1).

### Phase 3 — anna-v3 run (~$3.5-4.5, needs owner go)
- Data: T80 2024-01/02/03/05 (manifest groups train + train2, ~6.2B filtered positions).
- 800 superbatches (throughput will be lower than v1 because feature mapping is CPU-heavier; pick a 24-core box).
- Kill layers as in CLAUDE.md; cap $4.50.

### Phase 4 — search work that only pays with a good net (ongoing, free)
- Threat-aware history indexing (Stockfish: main history indexed by whether from/to squares are threatened).
- TT-move history double extensions (Stockfish, Feb 2026).
- Speed pass (movegen, accumulator refresh, TT prefetch).
- Retest LmpBase=4 (+7 unconfirmed) with the new net.

### Phase 5 — own-data fine-tune (~$0.50, later)
- Stage-3-style pure-WDL fine-tune at tiny LR on Anna's own self-play data (7M+ positions, growing) mixed with
  T80, to align the evaluation with Anna's search. This is the start of the reinforcement loop.

### Measurement
- After v3: 200 games vs Stormphrax 8 and Stash at 8+0.08; one 100-game match at 60+0.6 vs Stormphrax for a
  longer-TC truth check.

## Sources
- Stockfish SFNNv13 commit (threat inputs, L2 16->32): github.com/official-stockfish/Stockfish/commit/a6d055d
- Viridithas 20 release (threats + pawn-pawn, +60.84 +/- 5.92 at 60+0.6): github.com/cosmobobak/viridithas/releases
- Stormphrax 8.0.0 release notes and src/eval/arch.h; Reckless 0.9 release notes
- bullet examples/advanced (inputs.rs, main.rs, filter.rs) at rev 629ee50
