# Anna: what it takes to reach the top 40, and past it (deep analysis, 2026-09-13)

Written after two days of measurements; every number below is ours unless a source is cited. The ledger of what was
tried is `EXPERIMENTS.md`.

## 1. Where the 80 Elo actually is

Top 40 on the CCRL blitz list is ~3655. Anna v1 measures ~3577. The nearest strong anchor we play regularly is
Stormphrax 8 (3744, #14), 167 above us. Three things separate two engines: how good the evaluation is per node, how
few nodes the search needs to reach a given quality, and how many nodes per second the code delivers.

Measured on 16 book positions to depth 13 (laptop, under load, same conditions for all):

| engine | nodes to depth 13 | nps | note |
|---|---|---|---|
| Anna v1 | 1.86M | 476k | |
| Anna v3b | 1.65M | 190k | fewer nodes: better eval prunes better |
| Stash 37 (3421) | 1.73M | 785k | lean C, tiny net |
| Stormphrax 8 (3744) | 1.80M | 222k | threat net, same speed class as our v3b (190k) |

Read: all four engines need about the same number of nodes for depth 13, so search selectivity is not what separates
us from either neighbour. Stash is 156 Elo weaker with a weaker net and 1.65x our speed; Stormphrax is 167 Elo stronger
with a threat net that runs no faster than ours. That pins the Stormphrax gap almost entirely on **evaluation quality
per node**: their net is trained on tens of billions of self-generated positions over months, ours on four months of
Leela data for two hours. Our v3b is +83 per node over v1; theirs is roughly +150 further. The lever is data and
training, not search tricks and not raw speed.

## 2. What the threat-net experiment taught

- Threat inputs are the state of the art (Stockfish SFNNv10-13, Stormphrax 8, Viridithas 20) and they work for us too:
  +83 Elo per node, twice, at two widths.
- Their cost is memory traffic: ~14 random weight rows per node. We cut the wasted part (a full rebuild every 2.5
  evaluations, 26% of time) and CPU overheads; what remains is the physics of ~10 outstanding cache misses per core.
  v3b now runs at ~0.6x of v1; break-even at blitz is ~0.75x; at 60+0.6 it already wins by +53.
- Post-training 4-bit rows lose ~24 Elo per node: quantisation has to be trained in, not applied after.
- 16% of threat rows are empty (impossible features); the rest are dense. No free traffic there.

## 3. Ideas from other fields, ranked by expected value per dollar

### A. Treat the threat table as an embedding table (recommender systems)

The 66 MB threat table is exactly the "huge sparse embedding table" problem of recommendation models, and that field
has a decade of answers:

1. **Hot/cold split by usage** (the cheapest, free). Feature usage in chess is extremely skewed: a few thousand
   attacker-victim-square patterns recur constantly, most almost never. If the top ~8k rows by usage account for most
   applied rows, they fit in a 4 MB table that lives in L2/L3, and the DRAM misses shrink to the cold tail. Nothing
   changes in training; the engine reorders rows at load time by a usage table measured once. First step: measure the
   histogram in a real search (instrumented build). If the head is fat, this alone could take v3b from 0.6x to 0.8x.
2. **Quantisation-aware 4-bit rows** (needs a retrain). The LLM literature (any4, SVDQuant) shows 4-bit weights
   are near-lossless when the training knows about it; our RTN test shows the naive version is not. Halves traffic.
3. **Compositional embeddings** (Shi et al., KDD 2020, quotient-remainder trick): represent each of the 59,808 threat
   rows as the sum or product of two rows from small tables (e.g. 256 + 256). Both tables fit in L2, the DRAM traffic
   disappears, and the model keeps a unique vector per feature. This is the most original transfer here; nobody in
   chess does it. Risk: quality loss from the imposed structure; the recsys results say element-wise multiplication
   retains most accuracy. Needs the trainer to build the FT from two sparse lookups, which bullet's graph supports.

### B. Two-tier evaluation (cascade classifiers, early-exit networks)

We have a slow accurate net and a fast decent one. Cascades in vision (Viola-Jones) and early-exit networks spend the
expensive model only where it changes the decision. Stockfish itself already ships two nets and picks by material
imbalance. Our version: use v3b at interior nodes (depth >= some d) where the evaluation shapes the tree, and v1 in
quiescence and at the leaves, which are the majority of evaluations. If v3b evaluates only ~30% of nodes its speed
penalty shrinks to ~0.85x of v1 while most of the +83 per node applies where it matters. Free to test: both nets
loaded, both accumulator stacks maintained (v1's costs almost nothing), one SPRT.

### C. Search: measure, then copy with tests

Our search is a Stockfish-18-shaped transcript. The free candidates tried so far were null (ThreatHist), unresolved
(SeTtmHist, LmpBase 4) or pending (ThreatOrder, SPSA). Two lessons: (1) copying a single feature from a 3800 engine
into a 3580 one often does nothing, because its value depends on the rest of the engine's tuning; (2) constants tuned
for another engine's eval scale are wrong for ours. SPSA at scale is the systematic fix (fishtest runs ~100k games per
tune; the laptop does 10k a day, a 32-thread box 100k for ~$3.5). A more original check: **ablation testing**. Disable
each feature in turn at fixed depth against the full engine; a feature that costs nothing when removed is broken or
mis-tuned, and broken features are worth more than missing ones.

### D. Data: our own games as a signal (the reinforcement loop)

Every strong engine ends up training on its own play. v3b is our strongest evaluator; the laptop can label ~30M
positions a day with fixed-node v3b searches at zero cost. A stage-3 style fine-tune of v4 or v3b on that data costs
under a dollar and starts the loop. The website's games add a trickle of human-opponent positions.

### E. Speed of the base engine

Stash at 785k vs our 476k under the same load says the C-level work is not done. The profile after the NNUE fixes
shows move generation (pext), the move picker, correction-history lookups and small allocations. A 20% base speedup
is +12 Elo for every net at once. Standard work, no research needed.

## 3b. Target locked (2026-09-13 12:50): CCRL 40/15, top 50

The 40/15 list (275 engines, 4 threads each) is where the threat net's evaluation counts. Scaled anchors: v1 ~3532 (~#70),
v3b ~3565 (~#55); top 50 = 3572, top 40 = 3584. Steps: verify Lazy SMP scaling (4 threads), SPSA the search constants with
v3b, own-data fine-tune of v3b, one measured 40/15 anchor run, CCRL submission.

## 4. The program

Ordered by expected Elo per dollar; items 1-4 are free.

| # | Item | Expected | Cost | Status |
|---|---|---|---|---|
| 1 | Feature-usage histogram, hot/cold row layout for v3b | histogram done (top 2048 = 84%); the hot set already fits L2/L3, and reordering rows does not cut the miss count (THP test: TLB is not it) | 0 | dropped |
| 2 | Two-tier eval (v3b interior, v1 leaves), SPRT | -41 +/- 27 at blitz | 0 | failed |
| 3 | Ablation pass over search features | finds broken ones; +5 to +20 | 0 | queued |
| 4 | SPSA of 13 constants with v3b (sprt/spsa_v3b), own-data labelling | +10 to +25 | 0 | running / queued |
| 5 | anna-v4 no-threats multilayer net | +35 to +60 at every control | $1-1.5 | training now |
| 6 | Base engine speed pass | profiled: NNUE is 40% of v1's time, the rest already runs at Stash's whole-engine speed; micro-opts neutral | 0 | done, no gain |
| 7 | CCRL submission of the best build | an official number | 0 | after v3c + SPSA freeze |
| 8 | QAT 4-bit or compositional threat net | the threat net wins at blitz too | ~$2, next budget | research |
| 9 | **anna-v3c: continue training v3b** (+430 SB, six months) | measured on v1: 340 -> 800 SB was worth ~+80/node | ~$5 Modal | running 2026-09-13 15:36 |
| 10 | Policy net for move ordering, distilled from Leela's search (runs/policy/PLAN.md) | built and measured: ordering -9 +/- 8, LMR -10 +/- 8 at blitz (2520 games each) | ~$3 Modal | **failed / shelved 2026-09-14**; root/time uses untested |
| 11 | Auxiliary heads (WDL, moves-left) and a learned complexity output | search knows "long win" vs "win now"; margins scale with uncertainty | ~$2 Modal each | queued |
| 12 | Own-data loop with DFRC starts (Stormphrax's recipe) | the path every top engine walked | CPU-box hours | next budget |
| 13 | L1 = 256 threat net (anna-v3q) | -74 per node vs v3b for +23% speed | $4.3 Modal | **failed 2026-09-14**; 512 is the floor |
| 14 | Thread voting for Lazy SMP (ThreadVote) | non-regression SPRT at 4 threads (sprt/vote4t_b.log) | 0 | running |
| 15 | Mixed data: Leela + Stockfish binpacks, 800 SB (runs/anna-v5/PLAN.md) | **+54 +/- 30 per node over v3b at SB 670 (cap hit before the end)** | ~$10.7 Modal | **the lever that works: v5-670 blitz +26 +/- 13 (976 games); finished v5f +78 +/- 30 per node and +42 +/- 25 at 60+0.6 over v3b**; default net on main (1.1-dev); ~3565 on the 40/15 scale; anchor run next |

If 5 lands in its expected range, top 40 at blitz is a matter of 1 and 3 on top. If it lands low, 1 and 2 are the
paths that make the threat net the blitz engine as well.

The Stormphrax measurement changes the long game: past top 40, the road is the same one every top engine walked, a
threat net trained on our own generated data for a long time. Data generation is CPU work (a 32-thread box makes
~50M positions an hour at 5k nodes each for ~$0.15), the GPU hours are cheap on the right host (6.7M positions/s), and
the engine-side threat path is now within reach of the reference engines' speed. That is a $10-20 programme, not a
$1 one, and it is where the next 100 Elo lives.

## Sources

- Stockfish SFNNv10 threat inputs: github.com/official-stockfish/Stockfish/commit/8e5392d ("Full Threat Input features,
  a subset of Piece(Square)-Piece(Square) pairs"; the accumulator width was cut to pay for them).
- Compositional embeddings: Shi, Mudigere, Naumov, Yang, "Compositional Embeddings Using Complementary Partitions for
  Memory-Efficient Recommendation Systems", KDD 2020 (arXiv 1909.02107).
- 4-bit weight formats: any4 (facebookresearch/any4), SVDQuant (low-rank branch + 4-bit residual).
- Two-network evaluation precedent: Stockfish's big/small NNUE selection by material (2024).
- CCRL Blitz list, 2026-09-13: computerchess.org.uk/ccrl/404.
