# Anna — a Rust NNUE chess engine (UCI)

[![Release](https://img.shields.io/github/v/release/Lironktf/anna-chess?label=release)](https://github.com/Lironktf/anna-chess/releases)
[![Build](https://github.com/Lironktf/anna-chess/actions/workflows/build.yml/badge.svg)](https://github.com/Lironktf/anna-chess/actions)
[![Play online](https://img.shields.io/badge/play-chess.lironkatsif.com-blue)](https://chess.lironkatsif.com)

Anna is an open-source chess engine written in Rust: a UCI engine with an NNUE (efficiently updatable neural network)
evaluation and a Stockfish-class alpha-beta search. It plays standard chess and Chess960, probes Syzygy tablebases, runs
on several threads, and ships as a single binary for Linux and Windows that works in any UCI chess GUI (Arena, CuteChess,
Banksia, ChessBase, Fritz, Scid, En Croissant) or as a Lichess bot through lichess-bot. Its networks are trained by this
project with [bullet](https://github.com/jw1912/bullet) on public Leela Chess Zero and Stockfish data; the code is not a
fork of any other engine. Strength is about 3500 Elo on the CCRL 40/15 scale (own measurement, see below).

| | |
|---|---|
| Language | Rust (no runtime dependencies; one vendored C file for Syzygy) |
| Protocol | UCI |
| Evaluation | threat-input NNUE, 768×16 + 4560 + 59,808 inputs → 512 → 16 → 32 → 1, 8 output buckets |
| Search | PVS, aspiration, null move, ProbCut, LMR/LMP, singular extensions, correction history, Lazy SMP with thread voting |
| Variants | standard, Chess960 (FRC/DFRC) |
| Tablebases | Syzygy WDL + DTZ |
| Platforms | Linux and Windows x86-64 (AVX2/BMI2), builds from source anywhere Rust runs |
| Strength | ~3500 CCRL 40/15 equivalent (1.1); not yet on a rating list |
| Play online | [chess.lironkatsif.com](https://chess.lironkatsif.com), or challenge [anna-bot](https://lichess.org/@/anna-bot) on Lichess |

- Threat-input NNUE (piece-square, pawn-pair and attacker–victim features), trained by this project with
  [bullet](https://github.com/jw1912/bullet) on public Leela Chess Zero and Stockfish data.
- Alpha-beta search in the Stockfish family: PVS, aspiration windows, null move, ProbCut, late-move reductions and pruning,
  singular extensions, correction history, Lazy SMP with thread voting.
- Chess960, Syzygy tablebases (WDL and DTZ), MultiPV, deterministic `bench`.
- Portable Linux and Windows binaries on every release (x86-64 with AVX2 and BMI2).
- A browser board you can play against at [chess.lironkatsif.com](https://chess.lironkatsif.com).

## Strength

All numbers below are this project's own measurements with fastchess and the UHO opening book; error bars are 95%.

| Version | Net | Measured against | Result |
|---|---|---|---|
| 1.2 | anna-v5f | Anna 1.1 (same net, old search), 60+0.6 | +100 ± 23 (952 games) |
| 1.2 | anna-v5f | Anna 1.1 (same net, old search), 10+0.1 | +86 ± 9 (2000 games) |
| 1.2 | anna-v5f | Stormphrax 8.0.0 (3634 on CCRL 40/15), 60+0.6, 4 threads | -66 ± 53 (161 games) |
| 1.1 | anna-v5f | Stormphrax 8.0.0, same machine and session as the 1.2 row above | -116 ± 54 (161 games) |
| 1.1 | anna-v5f | Anna 1.0's net (v3b), 8+0.08 | +53 ± 18 (SPRT, 446 games) |
| 1.1 | anna-v5f | Anna 1.0's net (v3b), 60+0.6, 4 threads | +35 ± 33 (80 games) |
| 1.1 | anna-v5f | Stormphrax 8.0.0 (3634 on CCRL 40/15), 60+0.6, 4 threads | -111 ± 34 (100 games) |
| 1.0 | anna-v3b | Stormphrax 8.0.0 (3634 on CCRL 40/15), 60+0.6, 4 threads | -108 ± 30 (100 games) |
| 1.0 | anna-v3b | Anna's first net (v1), 60+0.6, 4 threads | +53 ± 32 (100 games) |

1.2 changes only the search (histories, correction history, move ordering, reductions and quiescence brought in line with
current Stockfish practice) and keeps 1.1's network. On the CCRL 40/15 scale 1.2 measures at roughly 3570 ± 55 against
Stormphrax, up from about 3520 for 1.1 measured the same way. Anna is not yet listed by any rating list.

## Download and use

Anna is a command-line UCI engine: it has no board of its own, you load it into a chess GUI or a match runner.

Prebuilt binaries are attached to each [release](https://github.com/Lironktf/anna-chess/releases): `anna-linux-x86-64-v3`
and `anna-windows-x86-64-v3.exe`. They need a CPU with AVX2 and BMI2 (Intel Haswell / AMD Zen 3 or newer). The default
network is embedded, so the binary runs on its own.

Typical uses:

- **Chess GUI**: add the binary as a UCI engine in Arena, CuteChess, Banksia GUI, En Croissant, Scid vs. PC, ChessBase or Fritz.
- **Lichess**: Anna plays on Lichess as [anna-bot](https://lichess.org/@/anna-bot) (standard and Chess960, bullet to classical,
  casual or rated); challenge it from any account. To run your own, point [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot)
  at the binary (`protocol: uci`).
- **Engine matches and testing**: `fastchess` or `cutechess-cli` with `-engine cmd=anna`; `anna bench` prints a
  deterministic node count for build verification.
- **Analysis from scripts**: speak UCI over stdin/stdout (`uci`, `position fen ...`, `go depth 20`).

## Build

```
cargo build --release
./target/release/engine              # UCI
./target/release/engine bench        # deterministic node count (prints "Bench: <nodes>")
./target/release/engine perft 6      # move generator check from the start position
cargo test --release -p engine       # perft suite, NNUE equivalence tests, search tests
```

`.cargo/config.toml` sets `target-cpu=native`; the CI workflow builds the portable `x86-64-v3` binaries.

## UCI options

| Option | Default | Notes |
|---|---|---|
| `Hash` | 16 | MB |
| `Threads` | 1 | Lazy SMP |
| `MultiPV` | 1 | |
| `Move Overhead` | 20 | ms per move reserved for the GUI |
| `UCI_Chess960` | false | |
| `EvalFile` | embedded | path to another quantised network |
| `SyzygyPath` | | directory with `.rtbw`/`.rtbz` files |

Search constants are also exposed as options (see `engine/src/params.rs`) so that one binary can be A/B tested; the
defaults are the tested values. Anna has no opening book and no learning.

## Play in a browser

`play/` is a small axum server that spawns one engine process per connection and drives it over UCI. One port serves the
page, the WebSocket and `/health`, so a single tunnel exposes it.

```
cargo build --release
scripts/play.sh --port 8080                     # embedded net
scripts/play.sh --net nets/other.bin --threads 2
```

Every game is recorded in a local SQLite database (`play/games.sqlite`). The board supports play as either colour,
engine time per move or fixed depth, undo, resign and flip, and shows depth, evaluation, speed and the principal
variation. Legality and game-end rules come from the engine's own move generator. Protocol details are at the top of
`play/src/lib.rs`; `cargo test --release -p play` runs the rules tests and an end-to-end WebSocket game.

## Architecture

**Search** (`engine/src/search.rs`, `movepick.rs`, `history.rs`, `tt.rs`): iterative deepening with aspiration windows,
principal variation search, a transposition table with aging, null-move pruning, ProbCut, reverse futility and razoring,
late-move pruning and reductions driven by history, singular and check extensions, main / capture / continuation / pawn /
correction histories, killer and counter moves, SEE, Lazy SMP with per-thread histories, a shared table and a weighted
vote for the final move, and a time manager for increment and repeating time controls.

**Evaluation** (`engine/src/nnue/`): `(psq 768×16 king buckets + pawn pairs 4560 + threats 59,808) → 512` per perspective,
CReLU and pairwise multiplication, then `512 → 16 → 32 → 1` with eight output buckets. The accumulator is updated
incrementally with a Finny refresh cache for the king-bucket part and a mirror-aware threat part; AVX2 kernels with a
scalar reference that the tests compare against bit for bit. Earlier nets (piece-square only) are still loadable.

**Training** (`trainer/`, `scripts/modal_train.py`): bullet at a pinned revision; data validation (`inspect stats`,
feature-index cross-checks against the engine, a byte-exact round trip of the engine's own data writer); runs on a rented
GPU with checkpoints every 10 superbatches synced off the box; every checkpoint passes `engine netcheck` before use. The
1.1 net was trained on Leela T80 (Jan–Jun 2024) plus Stockfish's `dfrc_n5000` and `nodes5000pv2_UHO` binpacks, 800
superbatches.

**Tools**: `prof/` (in-process sampling profiler with flame graph), `lc0conv/` (Leela training-chunk converter),
`scripts/sprt.sh` (fastchess SPRT), `scripts/spsa.py` (SPSA tuner), `scripts/nodes_to_depth.py` (ordering measurements),
`scripts/cpu_eval*.sh` (rating matches against reference engines on a rented CPU box).

## Repository layout

- `engine/` the engine crate (UCI, perft, bench, datagen, netcheck)
- `nets/` the embedded default network and the previous one
- `trainer/` the bullet-based trainer
- `play/` the browser server
- `prof/`, `lc0conv/`, `scripts/` tooling
- `runs/` plans, logs and results of every training run and rating measurement
- `EXPERIMENTS.md` the ledger of everything tried, with the numbers, including what failed

## How the project is run

Perft is law; `bench` is deterministic and any change to it is intentional and noted; every search change is
SPRT-tested; NNUE inference is checked against a scalar reference and incremental updates against a full refresh;
release builds are soaked at repeating time controls and with several threads before tagging. Results, positive and
negative, go into `EXPERIMENTS.md`.

## Attribution

Nothing here is copied from another engine's source; the techniques are, and they are credited:

- **Search**: the alpha-beta framework follows the techniques published in
  [Stockfish](https://github.com/official-stockfish/Stockfish). The code is not derived from Stockfish's source
  (token-shingle overlap 0.3%, the same as between two unrelated engines of this family), but many formulas use
  Stockfish's published tuned constants as starting values (18 of 57 unusual constants in `search.rs`); retuning them
  for Anna is ongoing with SPSA and SPRT.
- **Evaluation**: networks trained by this project with [bullet](https://github.com/jw1912/bullet) (jw1912) on public
  [Leela Chess Zero](https://lczero.org) T80 data (linrock's binpack conversions) and Stockfish's published binpacks.
  The input design follows Stockfish's threat inputs (SFNNv10 onward) and the pawn-pair inputs invented by Jonathan
  Hallström for [Pawnocchio](https://github.com/JonathanHallstrom/pawnocchio) and used by Stormphrax, Viridithas and
  Stockfish (SFNNv16). The weights are this project's; no pretrained network from another engine was used.
- **Tablebases**: the vendored [Fathom](https://github.com/jdart1/Fathom) Syzygy prober (MIT).
- **Testing**: [fastchess](https://github.com/Disservin/fastchess) and the UHO opening books.

## FAQ

**Is Anna a Stockfish fork or derivative?** No. The search follows the techniques Stockfish and its family publish, and
some tuned constants started from Stockfish's values, but the code was written from scratch in Rust (measured
token-shingle overlap with Stockfish 0.3%) and the networks are trained by this project, not taken from another engine.

**How strong is it?** About 3500 on the CCRL 40/15 scale by our own matches against Stormphrax 8; see the Strength table.
Anna is not on any rating list yet. It beats hand-crafted-evaluation engines such as Stash by a wide margin and loses to the
top NNUE engines (Stockfish, Berserk, Stormphrax, Viridithas).

**Does it need a network file?** No, the default network is embedded. `EvalFile` loads another one.

**Which CPUs?** The release binaries need AVX2 and BMI2 (Intel Haswell 2013+, AMD Zen 3 2020+). Older CPUs can build from
source; a scalar fallback exists but is much slower.

**Why "Anna"?** A short name that is easy to say and search for. Earlier artefacts in `runs/` use the previous working name.

## Citing and mentioning

If Anna is useful in a project, article or comparison, cite it as: *Anna chess engine, https://github.com/Lironktf/anna-chess*
(a `CITATION.cff` is included). Stars and links help other people find it.

## License

To be chosen by the owner (a LICENSE file will be added).
