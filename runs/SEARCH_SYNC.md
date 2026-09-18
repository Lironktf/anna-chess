# Search sync campaign (free, laptop) — started 2026-09-17 09:30

Goal: close part of the ~100 Elo gap to Stormphrax-class engines on the search side, which transfers fully to external
ratings (net gains transfer at ~1/3). Method: Anna's search is a port of an older Stockfish; Stockfish master
(tools/refs/Stockfish, commit 031dfeb, 2026-09-13) has moved on in many places. Each difference group goes behind a
UCI param (default off, bench unchanged with everything off), is SPRT-tested at 8+0.08 [0,5] on one binary
(scripts/sprt.sh -N "option.X=1" -B "option.X=0", 7 games at a time), and becomes the default when it passes (bench change
noted). Baseline binary: sprt/bin/anna_sync0 (HEAD at start, bench 650059, v5f net). Speed is NOT the gap: on this
laptop Anna does 375-447k nps vs Stormphrax 408-537k with similar nodes-to-depth-16 (5 positions, 2026-09-17 08:50).

Groups (Stockfish master line numbers refer to search.cpp / movepick.cpp at the commit above):
- G1 SfPrune: hindsight depth adjust from (ss-1)->reduction (874-878); in-check static eval = (ss-2) (838); improving |=
  staticEval >= beta after NMP (1061); NMP condition/R with priorNMPFailHigh (1019-1058); RFP depth<19 and blended return
  (1004-1016); razoring returns qsearch directly (997-1000); depth -= 3 on alpha improvement for 3<depth<12 (1550);
  fail-high blend (1574); TT depth+6 when no moves (1641).
- G2 SfHist: eval-diff quiet ordering bonus (986-994); TT-cutoff malus for early quiet moves of the previous ply (894-896);
  prior-capture fail-low capture-history bonus (1618-1624); fail-low bonusScale extra term + scaledBonus formula (1593-1616);
  update_all_stats new bonus/malus formulas incl. non-PV scaling and prevSq single-move malus (2000-2030); continuation
  history weights over plies 1-6 with positiveCount multipliers and +73 (2032-2050); pawn-history weight rule (2060-2070).
- G3 SfLmr: reductions table 2872/128*ln(i) (720); ttPv +929 (1176); all LMR terms (1330-1374); d clamp newDepth+2 (1384);
  doDeeper 53 / doShallower 8 (1396-1399); post-LMR flat 1334 (1405); full-depth reductions (1413-1418); PV re-search TT
  qsearch-dive extension (1432-1435); cutoffCnt rule (1544); singular: is_shuffling guard, multi-cut correction update,
  negative extension only -3 (1261-1318); seekMate handling.
- G4 SfCorr: correction value weights and (ss-2)/(ss-4) continuation correction (85-104); to_corrected v + cv/131072;
  update formula (1644-1653) and per-table weights (109-130); drop major correction.
- G5 SfPick: drop killers/countermove stages; quiet scoring weights (1,1,1,1,1 over plies 1,2,3,4,6), check bonus 16384 with
  see_ge(-75), lesser-piece threat term, low-ply 8/(1+ply); good-capture threshold -value/18; partial sort -3560*depth;
  good/bad quiet split at -14000; evasion scoring.
- G6 SfQs: SEE threshold alpha - futilityBase with bestValue floor (1822-1826); skip non-captures (1829-1831); stand-pat
  blends (1763, 1887); decisive guard on ttValue as eval (1748); stalemate special case (1878-1883).
- G7 SfTm: fallingEval / timeReduction / instability / effort formulas (580-599); single-legal-move 0.5 s cap (601-603);
  mate stop (610); timeman timeAdvantage, mtg for time<1 s, maximum formula (timeman.cpp).
- G8 SfTtVerify: TT cutoff verified against the position after the TT move at depth >= 7 (903-917).

Ledger of results below (also mirrored in EXPERIMENTS.md and sprt/verdicts.txt).

## Implementation notes (2026-09-17 10:40-12:30)
All eight groups are implemented behind params SfPrune, SfHist, SfLmr, SfCorr, SfPick, SfQs, SfTm, SfTtVerify (bench
650059 unchanged with all off; 245454 with all on). Binaries sprt/bin/anna_sync1..8 (each contains all groups implemented up
to its number). Queue: scripts/sync_queue.sh, 5+0.05, 7 concurrent, 7 h cap per test, verdicts in sprt/verdicts.txt and
sprt/sync_queue.log. Throughput on this laptop: ~450 games/h.
Known deviations from Stockfish master: G1 applies the hindsight depth adjustment after the TT cutoff (Stockfish before it,
because it evaluates before the cutoff); G7 uses the previous search's best score where Stockfish uses the best move's
average score, counts best-move changes of the main thread only, and has no ponder handling; G3 skips the multi-cut
correction update (it is in G4) and has no followPV bookkeeping (IIR unchanged); G5 keeps the killer/counter tables updated
but never picks from them.
- 12:46 the owner's university server (64-vCPU node, shared, `ssh uw`) runs G3..G8 at 24 concurrent games in tmux (node
  ubuntu2404-002); the laptop keeps G1 then G2. Server verdicts: sprt/uw/verdicts.txt (rsynced every 10 min).
- **G3 SfLmr PASS** (server, 5+0.05): +9.7 +/- 5.6, 4938 games, LLR 2.96. Will become the default after the queue.
- **G4 SfCorr PASS** (server): +28.9 +/- 9.9, 1544 games, LLR 2.95 (54.2%). The correction-history rework is the biggest single gain so far.
- **G5 SfPick PASS** (server): +11.4 +/- 6.0, 3980 games (51.6%). Killer/counter stages removed, Stockfish-master quiet scoring.
- **G6 SfQs PASS** (server): +6.1 +/- 4.2, 8656 games (50.9%). Small but real.
- **G7 SfTm FAIL** (server): -7.7 +/- 5.7, 4402 games (48.9%), 2 time losses in the log. Stays off. Candidate for a bisect:
  time allocation constants (timeman.rs) vs the iterative-deepening changes (aspiration rules, search-again, effort factor
  that always scales the optimum by <= 0.838, which our optimum constants were not tuned for).
- **G1 SfPrune STOPPED** (laptop, 7 h cap): +0.9 +/- 4.4 after 10080 games (50.1%). Neutral; stays off.
- **G2 SfHist PASS** (laptop): +49 +/- 13 after 1008 games (57.0%), LLR crossed in 42 minutes. The history rework is the
  largest gain of all.
- **Combined verification (laptop, 10+0.1, 2000 games): new defaults (G2+G3+G4+G5+G6) vs pre-campaign anna_sync0:
  +86.0 +/- 9.2** (817-332-851, 62.1%). The five groups stack almost fully (sum of singles ~+105 at 5+0.05).
- **G8 SfTtVerify STOPPED** (server): +0.7 +/- 2.2 after 28,839 games. Neutral; stays off. Queue complete.
- 23:31 server: 60+0.6 confirmation started (new defaults vs anna_sync0, 1000 games, 24 concurrent).
- **60+0.6 confirmation (server, 1000 games, 24 concurrent): new defaults vs anna_sync0 +94.7 +/- 12.0** (403-137-460, 63.3%),
  0 time losses. The gain holds at slow time controls (slightly larger than at 10+0.1).
