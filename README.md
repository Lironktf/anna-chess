# chess — a Rust NNUE chess engine

> ## ⚠️ Never start, rent, resume, or bid on anything on vast.ai without the owner's explicit go-ahead for that specific run.
> This applies to Claude and to any script. Present the plan, the expected cost and the local validation evidence first,
> then wait for a "go". Budget cap: $10 total, tracked in `BUDGET.md`. See `scripts/VAST.md` for the checklist.

A UCI chess engine written in Rust with an efficiently-updatable neural network (NNUE) evaluation and a
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
