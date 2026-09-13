# Anna — a Rust NNUE chess engine

> ## ⚠️ Never start, rent, resume, or bid on anything on vast.ai without the owner's explicit go-ahead for that specific run.
> This applies to Claude and to any script. Present the plan, the expected cost and the local validation evidence first,
> then wait for a "go". Budget cap: $10 total, tracked in `BUDGET.md`. See `scripts/VAST.md` for the checklist.

Anna is a UCI chess engine written in Rust with an efficiently-updatable neural network (NNUE) evaluation and a
Stockfish-class alpha-beta search. Networks are trained with [bullet](https://github.com/jw1912/bullet) on
Leela/Stockfish data and the engine's own self-play data.

## Build

```
cargo build --release
./target/release/engine            # UCI
./target/release/engine bench      # deterministic node count
./target/release/engine perft 6    # move generator check
```

## Layout

- `engine/` UCI engine, perft, bench, datagen, netcheck (`engine [uci|bench [depth] [threads]|perft <depth> [fen]|datagen ...|netcheck <net>]`)
- `trainer/` bullet trainer (only built on the GPU box)
- `scripts/` data download, vast.ai helpers, SPRT
- `nets/` shipped networks, `data/` datasets (gitignored), `runs/` training run logs

## Play against Anna in a browser (WebSocket)

`play/` is a small server (axum) that spawns one engine process per connection and drives it over UCI.
One port serves the page (`/`), the WebSocket (`/ws`), and `/health`, so a single Cloudflare tunnel exposes it:

```
cargo build --release
scripts/play.sh --port 8080                      # embedded default net (v1)
scripts/play.sh --net nets/anna-v3.bin --threads 2   # any quantised net, more search threads
cloudflared tunnel --url http://localhost:8080   # optional public URL (*.trycloudflare.com)
```

Click a piece, click a target square (promotion asks which piece). Controls: play as White/Black, engine time per
move (100 ms to `--max-movetime`), fixed depth, undo, resign, flip. The panel shows depth, eval, nps and the
principal variation in SAN. Move legality and game-end rules (mate, stalemate, 50 moves, threefold, material)
are decided by the engine crate's own move generator, so the server never accepts a move the engine would not.

Protocol (JSON text frames): client sends `{"t":"new","color":"white|black|random","movetime":1000,"depth":null,"fen":null}`,
`{"t":"move","uci":"e2e4"}`, `{"t":"undo"}`, `{"t":"resign"}`, `{"t":"ping"}`; server sends `hello`, `state`
(fen, turn, legal moves, history with SAN, status, result, thinking), `info` (depth, cp_white, mate_white, nodes,
nps, pv, pv_san), `error`, `pong`. Full description at the top of `play/src/lib.rs`. `--max-games` caps concurrent
engine processes (default 4); idle connections are closed after 30 minutes.

Tests: `cargo test --release -p play` (SAN/rules unit tests + an end-to-end WebSocket game against the real binary).

## Profiling and tuning (free)

- `cargo run --profile release-debug -p prof -- 13 [net]` samples the bench in-process (no perf/ptrace needed) and prints
  self time, inclusive time, callers of the hottest leaves, and writes `prof/flamegraph.svg`.
- `scripts/spsa.py` tunes UCI parameters by SPSA with fastchess (resumable; `best.txt` holds the current point);
  verify any result with an SPRT before making it a default.
- `scripts/overnight.sh` chains queued SPRTs, an SPSA run and its verification unattended; verdicts in `sprt/verdicts.txt`.
- `scripts/cpu_eval.sh` + `scripts/cpu_eval_remote.sh` rent a CPU box and play rating matches against reference engines (paid; needs go).

## Tablebases

Syzygy probing via vendored Fathom (`engine/csrc/fathom`, MIT). Set `SyzygyPath` (UCI) to a directory with
`.rtbw/.rtbz` files; `scripts/` has no downloader yet, the 3-4-5 set (939 MB) came from
https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/ and `3-4-5-dtz/` into `syzygy/`.

## How Anna was made: authorship and attribution

Anna's code was written by an AI (Anthropic's Claude, in Claude Code) working under the direction of the repository owner,
who set the goals, made the design and spending decisions, ran the paid training runs and the rating matches, and
tested every release. Nothing here is copied from another engine's source; the techniques are, and they are credited:

- **Search**: the alpha-beta framework follows the techniques published in [Stockfish](https://github.com/official-stockfish/Stockfish)
  (principal variation search, transposition table with aging, iterative deepening with aspiration windows, null-move
  pruning, ProbCut, late-move reductions and pruning, futility and razoring margins, singular extensions, history and
  continuation-history move ordering, correction history, Lazy SMP). Constants were tuned for this engine with SPSA
  and SPRT; results are in `EXPERIMENTS.md`.
- **Evaluation**: NNUE networks trained by us with [bullet](https://github.com/jw1912/bullet) (jw1912) on public
  [Leela Chess Zero](https://lczero.org) T80 data (linrock's binpack conversions). The input design follows Stockfish's
  threat inputs (SFNNv10 onward) and the pawn-pair inputs invented by Jonathan Hallström for
  [Pawnocchio](https://github.com/JonathanHallstrom/pawnocchio) and used by Stormphrax, Viridithas and Stockfish
  (SFNNv16). The weights are ours; no pretrained network from any other engine was used.
- **Move-ordering policy net** (experimental): trained by us on Leela's published training data (the search's move
  distributions), an idea first tried in alpha-beta engines by Kociolek.
- **Tablebases**: the vendored [Fathom](https://github.com/jdart1/Fathom) Syzygy prober (MIT).
- **Testing**: [fastchess](https://github.com/Disservin/fastchess), the UHO opening books, and the CCRL testing
  conditions as the target.

Every experiment, including the failed ones, is logged in `EXPERIMENTS.md`; every paid run in `BUDGET.md`.

## Rules of the road

See `CLAUDE.md`: perft is law, bench is deterministic, every search change is SPRT-tested, the paid run is
validated end-to-end locally first, and docs are never trusted over source.

## Status (2026-09-13)

Engine complete and tested (perft incl. Chess960, NNUE incremental/SIMD equivalence, deterministic bench, 42 unit tests).
Networks: v1 (default, embedded), v3b (threat + pawn-pair net, +53 Elo over v1 at 60+0.6 and at 4 threads), v3c and v4
(experiments, see `EXPERIMENTS.md`). Measured under CCRL 40/15 conditions (4 threads, 60+0.6): v3b is 108 +/- 30 Elo
below Stormphrax 8 (3634 on that list), i.e. about 3525 on the 40/15 scale. Current work: the move-ordering policy net,
SPSA-tuned constants, and the first release build for rating lists (`runs/ROADMAP2.md`).
