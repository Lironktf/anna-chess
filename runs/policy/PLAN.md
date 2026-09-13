# Policy net for move ordering, distilled from Leela's search (plan, started 2026-09-13 19:00)

Idea: runs/FRONTIER.md section 4. Roadmap item 10.

## Data (done, verified)

- Source: storage.lczero.org/files/training_data/test80/training-run1-test80-YYYYMMDD-HHMM.tar (Lc0 T80 training chunks,
  V6 records, input_format 1 = classical planes, no canonicalisation transforms in this data). One tar (~800 MB) holds
  ~37,700 games = 4.11M positions, each with the full MCTS visit distribution over legal moves (mean 27.9 legal moves),
  best_q / result_q / plies_left.
- Converter: `lc0conv` (workspace crate). `lc0conv convert out.bin in.tar|-` streams gz chunks, decodes the record (planes are
  ReverseBitsInBytes'd, mover's perspective, black mirrored; en passant inferred from the previous history position),
  rebuilds the position in the engine and requires every policy/best/played move to be legal: 4,110,995 / 4,110,995 records
  pass, policy mass on legal moves 1.0000. Output record 96 bytes: occ + nibble piece list, stm, ep file, castling,
  rule50, best/played (engine Move), up to 12 (move, prob) pairs sorted by prob, best_q, result_q, plies_left.
  Throughput ~42k records/s per core (movegen-bound), i.e. ~100 s per tar.
- Scale: 25 tars = ~100M positions = ~20 GB download, 9.6 GB of records; done on a Modal CPU function streaming from
  storage.lczero.org (no tar storage).

## Net (to build)

Inputs: side-to-move relative piece-square features (12 x 64 = 768, ranks flipped for black, no king buckets), sparse.
Accumulator: 768 -> 256, ReLU (or CReLU) ... policy logits per legal move:
    logit(m) = acc . V[piece_type(m)][to(m)]  +  U[from(m)][to(m)]  (+ small promotion / capture terms)
Loss: cross-entropy against Lc0's distribution restricted to the stored top-12 (renormalised); legal-move masking at
training time from the engine's move list is not needed because only listed moves get gradient through the softmax
denominator: use the full legal list from the record? The record stores only 12 moves; the softmax denominator over
those 12 is a biased estimate. Decision: store all legal moves' probabilities (up to 32 entries; mean 27.9) in v2 of the
record so the softmax is exact. (TODO before training: bump MAX_POLICY to 48 and the record to 224 bytes.)
Size: 768x256 i16 = 393 KB, V 6x64x256 i8 = 98 KB, U 64x64 = 8 KB. Fits L2.
Trainer: PyTorch on Modal (L4/L40S), a few epochs over 100M positions: ~1 h.

## Engine integration (to build)

- One extra accumulator per node (256 x i16 = 512 B), incremental like v1's (add/sub rows on make), refreshed on reset.
- Scores computed once per node when the move picker reaches the quiet stage: score(m) = dot(relu(acc), V[pt][to]) + U[from][to].
- Use 1: quiet ordering: history + PolicyScale * logit. Use 2: LMR: reduce less for top-ranked policy moves, more for the
  tail. Use 3: root ordering and "easy move" time management when the top policy probability is very high.
- Each use behind a UCI param defaulting to off; SPRT one at a time at 8+0.08, then 60+0.6.

## Verification ladder

1. Trainer sanity: top-1 accuracy vs Lc0's best move on held-out positions (a 768->256 linear-ish net should reach
   ~40-50%; random is ~4%).
2. Engine reads the net and reproduces the trainer's logits on 1000 positions (like netcheck).
3. Fixed-depth node counts with policy ordering vs history ordering (fewer nodes to depth = better ordering).
4. SPRT.
