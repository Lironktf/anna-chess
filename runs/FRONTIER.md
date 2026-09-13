# Frontier program and the AI-authorship question (2026-09-13)

Sources are linked at the end. Numbers are ours unless cited. Ledger of results: `EXPERIMENTS.md`.

## 1. Will the rating-list community reject an AI-written engine?

Precedent (TalkChess, 2026): "Sable 1.6", a from-scratch C++ engine with a self-trained NNUE, disclosed up front as
written by Claude under the author's direction (author designed the approach, ran datagen/training, tested). Reactions:

- One independent tester (chessica) tested it within days and published a rating (~3356 on her list); another developer
  (smatovic) called it the "Centaur phase of programming".
- Established authors were hostile or dismissive: "what is even the point", "computer chess is doomed" (Ciekce, the author
  of Stormphrax); a suggestion that list maintainers "prefer a real human" engine over "the purely vibe coded engine"
  (Jost Triller); worry about copied code with "no concept of copyright" (chessica).
- No CCRL / CEGT listing is visible in the thread. The author renamed the engine (Onyx) and kept publishing.

Read: an AI-written engine will be tested by some independent testers and may be deprioritised or ignored by others.
The CCRL 40/15 list is run by a handful of volunteers, so a listing there is not guaranteed for any new engine, and
less so for this one. What raises the odds, in order:

1. Disclose it plainly in the README before anyone asks (the repo is public; being outed is far worse than being upfront).
2. Show the parts that are unambiguously ours: nets trained by us from public data with a public trainer, the full
   experiment ledger with negative results, measured anchors, and the design choices that differ from the reference
   engines (split accumulator with mirror-crossing threat rebuilds, the speed work, the failed experiments).
3. Attribution that is precise, not vague: search techniques from Stockfish's published code (list them), threat and
   pawn-pair input design from Stockfish SFNNv10-16 / Pawnocchio / Stormphrax / Viridithas, trainer bullet (jw1912),
   Syzygy probing via Fathom (vendored, MIT), data Leela T80 (linrock).
4. Clean-room check before release: no copied source text. The search is written from technique descriptions, not
   transcribed files; a similarity scan against Stockfish/Stormphrax sources before tagging is cheap and worth doing.
5. Do not lean on CCRL alone. Publish our own measurements against known engines under CCRL 40/15 conditions
   (4 threads, 60+0.6, UHO book) with games and error bars; also announce on TalkChess where independent testers
   (chessica, SPCC-style lists, Ipman) pick engines up. For the resume, "measured within +/-40 of a 3634-rated
   engine under CCRL conditions, games published" is defensible even without an official number.

## 2. What is already at the frontier

- Inputs: threats + pawn pairs restricted to the 3-file band (4560 pairs) is exactly the SFNNv16 input set (Stockfish,
  Sept 2026: pawn pairs "invented by Jonathan for Pawnocchio"; superseded the pawn-pusher threats). We have both since v3.
- Output stack: pairwise -> 16 -> 32 -> 1 x8 is the same shape as SFNNv13+ (L2 doubled to 32 when threat inputs allowed a
  smaller L1). Our v4 test says this stack adds nothing without threat inputs at 340 SB; with them it is standard.
- What the top engines have that we do not: data. Stormphrax trains "entirely self-generated", DFRC, from random init,
  for months; Stockfish trains on tens of billions of positions. Our nets have seen ~0.5B Leela positions for 2 hours.

## 3. Ranked program (expected gain per dollar; "expected" is a guess unless marked measured)

| # | Idea | Why it should work | Cost | Verification |
|---|---|---|---|---|
| 1 | **Continue training v3b** (anna-v3c: +430 SB, two new months) | measured: v1 gained ~+80/node from 340 -> 800 SB; v3b stopped at 340 | ~$5 Modal | equal-nodes vs v3b, then 4-thread 60+0.6 |
| 2 | **Policy net for move ordering, distilled from Leela's search** (new for alpha-beta) | T80 raw chunks (v6 format, 8356 B/record) carry Lc0's MCTS visit distribution over 1858 moves for every position: a free, very strong move-quality label. A tiny net (256-wide accumulator, one dot product per move) gives a static prior that history heuristics lack at fresh nodes. Kociolek 2.2 reports +75..+95 from a root policy net (weak engine; our gain would be smaller). No top alpha-beta engine does this; Monty (MCTS) trains such nets with bullet | 1-2 days of work; ~$2 Modal for training | SPRT at 8+0.08 and 60+0.6; ablate: ordering only, then LMR, then root/time |
| 3 | **Own-data reinforcement loop** (v3c labels, DFRC starts) | every top engine's path; Stormphrax says DFRC data generalises better | CPU box hours: ~$0.15 per 50M positions; a useful set is 1-2B | fine-tune then equal-nodes |
| 4 | **Auxiliary heads on the value net** (WDL 3-way, moves-left) | Lc0 uses both; a moves-left head lets the search tell "winning but long" from "winning now" and feeds time management; cheap in the trainer | ~$2 Modal (retrain head only) | SPRT |
| 5 | **Complexity/uncertainty output** to scale pruning margins (Stockfish uses |psqt - positional| as "complexity"; a learned head is the natural successor) | free once the trainer has a second output | ~$2 Modal | SPRT |
| 6 | **SPSA at scale** on v3b (running) + ablation pass | Triumviratus 4.0 reports +29 from co-tuning 55 params; our constants are v1-era | 0 | SPRT of tuned vs default |
| 7 | **Quantisation-aware 4-bit threat rows** | halves the memory traffic that makes v3b 0.45-0.6x of v1; post-training RTN lost 28 (measured), QAT is how LLMs keep 4-bit lossless | ~$3 Modal | equal-nodes (must be ~0) then equal-time |
| 8 | **Compositional (quotient-remainder) threat embeddings** | recsys trick: 59,808 rows as sums of two small tables that fit in L2; kills the DRAM misses entirely | trainer graph work + ~$3 | same as 7 |
| 9 | 40/15-specific: moves-to-go time management test, 4-thread + 256 MB hash soak | CCRL conditions differ from our SPRTs | 0 | fixed-position soak, no losses on time |

Not worth doing (measured or reasoned): hot/cold row layout, two-tier eval by depth, base-engine micro-optimisations,
another output-stack-only net (v4).

## 4. The policy-net design (item 2), concretely

- Data: `test80-2024-MM-*.tar.zst` from linrock/test80-2024 are Lc0 v6 chunks (`V6TrainingData`, 8356 bytes: 1858 policy
  floats, 104 planes, q/d/m targets, best_idx/played_idx). A Rust converter emits (position, top-8 moves with
  probabilities, best move) in a compact record; ~100M positions per month.
- Net: side-to-move accumulator 768 -> 256 (no king buckets; threats optional later) + per-move score
  `s(m) = relu(acc) . V[piece][to] + B[from][to]`; softmax cross-entropy against Lc0's distribution. ~10 MB. Trained in
  PyTorch on Modal (bullet at our pinned rev has no policy head; Monty extends it, not worth porting).
- Engine: one accumulator per node (incremental like v1, 512 B), scores computed once per move list; used as (a) a
  prior added to history for quiets, (b) LMR reduction offset by policy rank, (c) root ordering and "easy move" time
  management by policy confidence. Each stage SPRT'd separately.
- Cost per node: ~35 moves x 256 MACs = ~9k ops, under 2% of a v3b evaluation.

## Sources

- Sable 1.6 thread (AI-assisted engine, reactions): talkchess.com/viewtopic.php?t=86419 (pages 1-2).
- CCRL submission: kirill-kryukov.com/chess/discussion-board/viewtopic.php?f=7&t=8554; conditions: viewtopic.php?t=1486.
- Stockfish SFNNv16 (pawn pairs supersede pawn-pusher threats; +Elo at LTC/VLTC, 3.5% slowdown):
  github.com/official-stockfish/Stockfish/commit/f4bcd40. SFNNv13 (L2 16 -> 32): commit a6d055d.
- Pawn-pair inputs origin (Pawnocchio, Jonathan Hallstrom); adopted by Stormphrax 8, Viridithas 20 (+61 release).
- Stormphrax questionnaire (self-generated DFRC data, random init): wiki.chessdom.org/Stormphrax_questionnaire_20251029.
- Kociolek 2.2 policy net in alpha-beta: open-chess.org/viewtopic.php?t=4645.
- Lc0 V6 training record: github.com/LeelaChessZero/lc0/blob/master/src/trainingdata/trainingdata_v6.h.
- Monty (MCTS engine, policy nets trained with bullet): github.com/official-monty/Monty.
- Triumviratus 4.0 SPSA co-tuning (+29.5): chessengines.blogspot.com/2026/06/triumviratus-40-new-version-chess-engine.html.
