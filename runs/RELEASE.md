# Release 1.0 checklist and rating-list submission (draft, 2026-09-13)

## Gates (all free; tick with the result)

| Gate | How | Status |
|---|---|---|
| Perft suite | `cargo test --release -p engine` (perft tests included) | 42 tests pass (2026-09-13) |
| Deterministic bench | `engine bench` twice, identical | 854588 |
| NNUE equivalence | unit tests (incremental == refresh, AVX2 == scalar) | pass |
| Net check of the shipped net | `engine netcheck <net> --strict` | v3b ok |
| Moves-to-go time control (40/15-style) | 40/20 soak under load, hand-checked allocation | 20 games, 0 forfeits (after the 2026-09-13 fixes) |
| Hyper-blitz soak (time losses, illegal moves, crashes) | 3+0.03 x 40 | 0 forfeits |
| 4 threads + 256 MB hash, moves-to-go | 40/60 x 6, Threads=4 | running |
| Long soak (several thousand games, no crash/forfeit) | the SPRTs of the last days on the same code | v3c SPRT (running) counts once the hardening build is used |
| Windows + Linux binaries from CI | tag `v1.0` -> GitHub release | CI green on every push |
| README disclosure + attribution | section "How Anna was made" | done |
| Similarity scan vs Stockfish / Stormphrax sources | (todo: a token-level diff of search.rs against their search files) | todo |
| Version string | `id name Anna 1.0` in the UCI banner | todo |
| Embedded net = the released net | v3b embedded by default (or shipped as a file next to the binary) | decide: v3b is 45 MB; embedding it makes the binary 70 MB, acceptable |

## Which build

Search: main + the hardening commit. Net: v3b unless the v3c SPRT passes [0,10] (then v3c). Policy net: only if its own
SPRT passes; otherwise ship the code with `PolicyScale=0`.

## Announcement (draft for the owner to post; the owner's account, the owner's words)

Board: kirill-kryukov.com/chess/discussion-board, forum "CCRL Public". Also TalkChess "New engine releases 2026 H2".

    Anna 1.0 - Rust UCI engine with a threat-input NNUE

    Anna is a UCI chess engine written in Rust. Evaluation: an NNUE with threat and pawn-pair inputs (the SFNNv16-style
    input set) trained by me with bullet on Leela T80 data; search: alpha-beta in the Stockfish family; Syzygy via Fathom.
    Lazy SMP, Chess960, no own book. Windows and Linux x86-64 (AVX2/BMI2) binaries:
    https://github.com/Lironktf/anna-chess/releases/tag/v1.0

    Strength, measured by me under CCRL 40/15-like conditions (4 threads, 60+0.6, UHO book, 100 games each):
    -108 +/- 30 vs Stormphrax 8.0.0, +53 +/- 32 vs Anna's own earlier net. My estimate is ~3520 on the 40/15 scale;
    games are in the repository (runs/cpu-eval3). I'd be glad to see it tested.

    Disclosure: the code was written by an AI (Claude) under my direction; I chose the design, ran the data pipeline,
    training and all testing. Details, attribution and the full experiment log (including the failures) are in the
    README and EXPERIMENTS.md.

## After posting

- Answer questions on the thread the same day; testers often ask about hash/threads/book handling.
- Keep 1.0 frozen; put improvements into 1.1 with its own SPRTs and a new release, never overwrite a released binary.
