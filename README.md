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

## Tablebases

Syzygy probing via vendored Fathom (`engine/csrc/fathom`, MIT). Set `SyzygyPath` (UCI) to a directory with
`.rtbw/.rtbz` files; `scripts/` has no downloader yet, the 3-4-5 set (939 MB) came from
https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/ and `3-4-5-dtz/` into `syzygy/`.

## Rules of the road

See `CLAUDE.md`: perft is law, bench is deterministic, every search change is SPRT-tested, the paid run is
validated end-to-end locally first, and docs are never trusted over source.

## Status (2026-09-10)

Engine core complete and tested: perft suite (incl. Chess960), NNUE incremental/SIMD equivalence tests, search tests,
deterministic bench, fastchess + UHO book in place, trainer crate pinned to bullet with data cross-checks passing.
Next step is the first paid training run described in `runs/fable-v1/PLAN.md` — waiting for the owner's go.
