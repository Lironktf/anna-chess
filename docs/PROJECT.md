# Anna: full project context

Written 2026-09-23. This is the single document to read first. It explains what the engine is, how every
piece works, what has been measured, what the infrastructure is, and what is left. `EXPERIMENTS.md` is the
chronological ledger of every test; this file is the standing picture.

## 1. What this is and where it stands

Anna is a UCI chess engine in Rust with an NNUE evaluation and a Stockfish-family alpha-beta search.
Started 10 September 2026. Released versions: 1.0 (net v3b), 1.1 (net v5f), 1.2 (search sync).

Strength, measured against Stormphrax 8.0.0 (CCRL 40/15 4CPU = 3634) at 4 threads, 60+0.6, on one machine:

| Build | Result vs Stormphrax | Implied CCRL 40/15 |
|---|---|---|
| 1.1 | -116 +/- 54 (161 games) | ~3518 |
| 1.2 | -49 +/- 17 (400 games) | ~3585 |
| 1.2 + NmpBase 6 | -38 +/- 17 (400 games) | **~3596** |

Cross-checks: dead level with Berserk 13 (3587) over 150 games; -84 +/- 17 against Coda 0.9.4 (3630) over
300 games, which is about 50 worse than its list rating predicts. Treat 3596 as the optimistic end of a
3550-3600 range. Rank-50 on the list is 3587.

Lichess bot `anna-bot`: 488 games, 221 wins, 6 losses, 261 draws, ratings ~3007 blitz / 3029 rapid /
3022 bullet. Note the Lichess pool is not the CCRL scale and the numbers do not convert.

## 2. The evaluation (NNUE)

**Inputs, per perspective** (side-to-move relative, board flipped for black):
- piece-square, 768 x 16 king buckets;
- pawn pairs, 4,560 features (pairs of pawns at most one file apart);
- threats, 59,808 features: one switch per (attacker piece type, attacker square, target square, target piece),
  covering attacks on both own and enemy pieces.

**Shape**: each active feature adds a row of 512 into an accumulator, two accumulators (one per side); CReLU and
pairwise multiplication; then 512 -> 16 -> 32 -> 1 with 8 output buckets selected by material count.

**Why threats are handed in rather than learned**: the first layer is a plain sum, so it cannot express a
conjunction like "rook on a1 AND knight on a7 AND the file between them empty". The later layers are nonlinear
but only see the 512-number summary, not the board. Attacks are trivial for the CPU (shifts, table lookups, magic
multiplies), so we compute them exactly and feed them as inputs. Measured worth: +83 Elo at equal nodes versus a
network without them. The idea comes from Pawnocchio; Stockfish adopted the same family in 2024.

**Speed work that mattered** (all bench-identical, all in `engine/src/nnue/`):
- split accumulator: the piece-square part uses a refresh cache keyed by king bucket, the threat part updates
  incrementally. Before this a king move into a new bucket rebuilt everything including threat rows, once every
  2.5 evaluations, 26% of search time. Fix: 17-27% fewer CPU-seconds for identical searches.
- bitmap set-difference instead of sort+merge for changed rows: ~20%.
- second layer over transposed weights, no horizontal sums: 10-18%.
- AVX2 kernels with a scalar reference; a test compares them bit for bit on random positions.
- pooled move lists, no per-node allocation.

**Current net**: anna-v5f, 512 wide, trained on Leela T80 (Jan-Jun 2024) plus Stockfish's `dfrc_n5000` and
`nodes5000pv2_UHO`, 800 superbatches. `nets/default.bin`, embedded in the binary.

## 3. Training pipeline

- Trainer: `bullet` at a pinned revision, built with CUDA on a rented GPU. `trainer/src/bin/train_v3.rs`.
- Data: bulletformat/sfbinpack files from HuggingFace, listed with sizes and sha256 in `data/MANIFEST.tsv`,
  fetched by `scripts/download_data.sh` (resumable, verified) and `scripts/prefetch_parallel.sh` (6 streams,
  113 MB/s measured; a single stream gets ~35).
- Objective: a blend of engine score and game result. Stage 0 warm-up with an LR ramp and WDL 0.2; stage 1 main
  with LR 1e-3 -> 1e-6 and WDL ramping 0.2 -> 0.5; stage 2 short fine-tune at 1e-5 -> 1e-7 on WDL 1.0.
- A superbatch is 763 batches x 16,384 positions, about 12.5M samples.
- Verification: `inspect stats` on every file, feature-index cross-check against the engine, byte-exact round
  trip of our own writer, then `engine netcheck` on every checkpoint, and a strict check before adoption.
- Rental discipline: `scripts/vast_launch.sh` (create/setup/train/destroy), `scripts/vast_selfdestruct.sh` on the
  box, `scripts/vast_cost_guard.sh` on the laptop, checkpoint sync, monitor. See CLAUDE.md item 8 for the
  mandatory pre-rent checklist, written after losing $3.68 on three bad hosts.

**What training taught us**
- Threat inputs: the single biggest evaluation gain (+83 equal nodes).
- Width: 1024 versus 512 on identical data gave no per-node gain and cost 28% of node rate. 512 stays.
- Data volume: v6 used 36 files and 494 GB against v5's 8 files and 222 GB, at a similar sample budget, and
  measured level at blitz and -26 per node. More data at a fixed budget does nothing.
- Untested and next: more superbatches at the same width, with self-play blended in.

## 4. The search

Iterative deepening with aspiration windows around the previous score. At each node: transposition lookup;
static evaluation corrected by correction history; gates that can return before any move is searched (razoring,
reverse futility, null move with verification at high depth, ProbCut); then the move loop with staged ordering,
shallow-depth pruning (move count, futility, history, static exchange), singular extensions, late move reductions
with re-search, and quiescence at depth zero. Lazy SMP with a shared table and a weighted vote for the played move.

**Why alpha-beta is safe and the rest is not.** A cutoff fires when a line is so good for the side to move that
the previous player, who had alternatives, would never have allowed it; the exact value cannot change the decision
above, so we stop. That is exact. Everything else (reductions, futility, null move) is a guess, which is why each
one is SPRT-tested and why they are all disabled near decisive scores so forced mates are not pruned away.

**The search sync campaign, 17-19 September.** Anna's search was re-diffed against Stockfish master 031dfeb and
the differences implemented as eight switchable groups (`runs/SEARCH_SYNC.md`). Results at 5+0.05:

| Group | Verdict |
|---|---|
| SfHist, history bonus/malus, eval-difference ordering, previous-move maluses | **+49 +/- 13** |
| SfCorr, correction-history tables, weights and update rule | **+29 +/- 10** |
| SfPick, no killer/counter stages, master quiet scoring and thresholds | **+11 +/- 6** |
| SfLmr, reduction terms, reductions table, singular rules | **+10 +/- 6** |
| SfQs, quiescence margins and stand-pat blends | **+6 +/- 4** |
| SfPrune, hindsight depth, NMP/RFP/razoring formulas | neutral, off |
| SfTtVerify, TT cutoff verified through the TT move | neutral, off (28,839 games) |
| SfTm, time management and aspiration loop | **-8**, off; both halves fail separately |

Adopted together: **+100 +/- 23 at 60+0.6** against the pre-campaign build, and +50 externally.

**Two later wins**: the null-move reduction base 5 -> 6 (+10.6 +/- 5.7), which also fixed a node explosion in lost
positions (18.0M nodes at depth 14 down to 9.0M), and the cut-node reduction 4026 -> 2800 (+3.2 +/- 2.6).

**What failed**: both Lazy SMP ideas copied from Stockfish (thread-scaled reductions -7.4, spread aspiration windows
-8.3); SPSA over 21 parameters twice, which moved nothing in 21,600 games; and a 14-test parameter batch that
yielded three Elo from about 190,000 games. The search constants are now close to right for this engine.

## 5. Testing infrastructure

- **Laptop** (8 cores): runs the Lichess bot and the play server; used for short SPRTs only when the bot is idle.
- **University server** (`ssh uw`, pool of 64-vCPU EPYC nodes, the work lives on node ubuntu2404-008): all serious
  testing. Match scripts live in `~/anna/sprt/`, driven by tmux sessions. Throughput about 4,800 games/hour at
  5+0.05 with 24 concurrent.
- **Trap to remember**: login shells there cap CPU time at 3600 s per process. At 60+0.6 with 4 threads an engine
  hits that in half an hour and is killed; fastchess records it as a disconnect and awards the game. It silently
  decided 39 of 200 games in each of the first anchors. Every match script must start with `ulimit -t unlimited`.
- **Never point a long match at a path you might rebuild.** Copy the binary first; a mid-match rebuild silently
  mixes two versions.
- Opponents built in `~/anna/tools/opp`: Velvet 8.1.1, Seer 2.8.0, Berserk 13, Clarity 7.2.0, Motor 0.9.0,
  Koivisto 9.0, Coda 0.9.4, plus Stormphrax 8.0.0 in `tools/stormphrax`.
- **Opponents do not transfer.** In a control match on our box Stormphrax beat Velvet by 177 where CCRL has them
  56 apart, so these builds play about 120 below their list ratings here. Only the Stormphrax anchor and the
  Berserk head-to-head are trusted.

## 6. The other software

- `play/`: axum server behind chess.lironkatsif.com via a Cloudflare tunnel, systemd user unit `anna-play`,
  games recorded in `play/games.sqlite`. The page has a board, evaluation bar, move list, a gear that hides the
  engine settings, and a "table talk" panel that is off by default.
- **Table talk**: the page asks `/taunt` on the server, which calls a hosted model (Groq, key in
  `~/.config/anna-play/groq_key`, never in git) for a one-line comment built from the real position and what just
  happened. Clean mode is profanity-free; swearing mode needs the password `swear` and is filtered server-side for
  slurs, sexual content, threats and personal jabs, with a local phrase bank as fallback.
- **Lichess bot** `anna-bot`: bridge in `~/lichess-bot`, systemd user unit `lichess-bot`, token in
  `~/.config/lichess-bot/token`. Two games at once, 4 threads and 512 MB each, standard and Chess960, matchmaking
  on. Update the engine with `cp target/release/engine ~/lichess-bot/engines/anna && systemctl --user restart
  lichess-bot`. Standing rule: never abort a game in progress.

## 7. Money

Total spent: $19.45 of vast.ai credit across all runs; $0.96 left. The expensive lesson was 17 September, when
$3.68 went on three hosts that never trained (one never booted, one had a driver too old to link the trainer, one
had a 120 GB container disk for 269 GB of data and billed bandwidth per gigabyte). CLAUDE.md item 8 is the
checklist written in response. Nothing paid starts without an explicit cap and the owner's go for that run.

**Why a GPU has to be rented**: the trainer needs CUDA. The laptop has Intel integrated graphics and the
university server has no GPU at all, so there is no free path to training here. A resume-from-checkpoint run on
the existing data is about 6.5 hours and under $4.

## 8. What is left

1. Release 1.3 (the two search fixes, +14 over 1.2) and submit to CCRL and CEGT. Listing takes four to eight
   weeks from announcement, going by Coda (announced 8 July, first tested 22 August) and ZeroG.
2. Validate the self-play data now being generated (`scripts/selfplay.sh` on the server, 132M positions so far,
   about 55M/day) with the same checks used on public data.
3. Measure thread scaling against Coda. Ours is +89 for four threads over one; Coda's published 1CPU and 4CPU
   ratings differ by only 21, yet it showed three times our node rate at four threads in an endgame. Something
   there does not add up and it is free to investigate.
4. The training run: resume from v5f, 600-1000 more superbatches on the v5 mix with self-play blended at 10-20%,
   about $4-6 under a cap. This is where the remaining Elo is; the search work has stopped paying.
5. Licensing: still unchosen. GPL-3 is the safe answer given how closely the search follows Stockfish's published
   techniques, and the Coda episode shows the community reacts to licence problems far more strongly than to how
   the code was written.
