# Chess engine project

Goal: a top-tier UCI chess engine in Rust with an NNUE evaluation and a Stockfish-18-class search,
trained with the `bullet` trainer on a rented vast.ai GPU. Total paid budget for the whole project: **$10**.
Timeline: days, not weeks. Every hour of the paid run and every hour of the human's time matters.

## Layout

- `engine/` — the engine crate (UCI binary, perft, bench, datagen, data conversion). No GPU deps, ever.
- `trainer/` — bullet-based trainer, only built on the GPU box. Pin bullet to an exact git rev.
- `scripts/` — vast.ai setup, data download, SPRT, checkpoint sync. Every script must be idempotent.
- `nets/` — quantised networks that ship with the engine. `data/` — datasets (gitignored).
- `BUDGET.md` — running ledger of every cent spent on vast.ai. Update it before and after each rental.

## The paid run is sacred

A GPU run costs real money from a $10 budget and cannot be redone casually. Treat it like a rocket launch.

1. **Never rent until everything that can be validated locally has been.** bullet without the `cuda` feature
   is a MockGPU that cannot train, so local validation means: the trainer binary compiles, the real data files
   load through the real loader and filter (`trainer/target/release/inspect stats`), bullet's feature indices
   equal the engine's on real positions (`inspect features`), and the engine's bulletformat writer is
   byte-identical to bullet's (`inspect roundtrip`). Then, ON THE BOX, the first thing that runs is a tiny
   smoke run (2 superbatches, SAVE_RATE=1) whose checkpoint is synced down and passes `engine netcheck` before
   the real run starts. No exceptions, no "it should work".
2. **Checkpoints must survive the instance dying.** Spot instances get killed. Save a checkpoint at least
   every 10 superbatches and sync it off the box immediately (rsync/scp to the laptop, or a free remote).
   The sync must be automated in a background loop started before training, not done by hand at the end.
   Confirm the first synced checkpoint arrived and loads in the engine before walking away.
3. **Validate the data before training on it.** Checksums against the source, `bullet-utils validate`,
   position counts, and a spot check that scores and results look plausible. A run on corrupt or
   mis-parsed data is $5 burned.
4. **Pin everything.** Exact bullet git rev, exact CUDA image, exact Rust toolchain. Write them down in
   `trainer/README.md` and in the run's log. A dependency drift discovered on the paid box is unacceptable.
5. **Log everything.** The full trainer stdout, the schedule, the data files used, the instance type, the
   start/stop times, and the final cost go into `runs/<run-name>/`. Future runs are planned from these.
6. **Stop the instance the moment it is not needed.** Destroy, don't just stop. Verify in the vast.ai
   console that billing ended. Record the cost in `BUDGET.md`.
7. **The instance must not be able to outlive the run. Four independent kill layers, all armed before
   training starts, in this order:**
   1. **On-box self-destruct** (`scripts/vast_selfdestruct.sh`, started detached on the instance with `setsid nohup`):
      destroys the instance through the vast API when the train log says training ended plus a grace period
      for the checkpoint sync, or at a hard wall-clock limit. Needs nothing from the laptop. Verify it is running
      (`pgrep -f vast_selfdestruct` over ssh, `/workspace/selfdestruct.log` says "armed") before starting training.
   2. **Laptop cost guard** (`scripts/vast_cost_guard.sh`): polls the instance every minute, destroys at the dollar cap.
   3. **Laptop monitor + Claude**: on "train end", pull the final checkpoint and run `GO=1 scripts/vast_launch.sh destroy`.
   4. **Human**: `vastai show instances` must print zero instances when the run is over. If it does not,
      `vastai destroy instance <ID>` immediately, then investigate.
   After every run, `BUDGET.md` gets the actual cost and the line "instances: 0 (verified <time>)".
   Never rely on a single layer; never start training with fewer than layers 1 and 2 armed.
8. **Pre-rent checklist (mandatory, every rental, written into the run's PLAN.md before `create`).** Added 2026-09-16 after
   $3.68 was lost on three failed hosts in one afternoon (never booted; driver too old to link the trainer; a 120 GB
   container disk for 269 GB of data on a host that also billed internet transfer per GB). Each item is a fact read from
   the offer or the account, not an assumption:
   1. **Total cost, not hourly cost:** `hours * dph_total + data_GB * inet_down_cost + disk_GB * storage_cost`. The hourly
      cost guard sees only `dph_total`; bandwidth and storage charges are invisible to it. Refuse any offer with
      `inet_down_cost > $0.002/GB` (the launcher enforces this; do not bypass it).
   2. **Disk:** the offer's `disk_space` is the host's free space, not the container's. Pass `DISK=` explicitly, sized for
      compressed + decompressed data + 20%. The launcher's default is 120 GB and is wrong for any training run.
   3. **Driver:** `driver_version >= 550` (bullet's pinned rev links CUDA 12.4 driver symbols). The launcher enforces this.
   4. **Host quality:** `verified=true`, `reliability2 >= 0.97`, and `inet_down >= 400` Mbps; measure the real transfer rate
      in the first minutes and abort the rental if it is below 30 MB/s rather than paying hours for a slow link.
   5. **One paid instance at a time**, always; a second rental starts only after `vastai show instances` prints none.
   6. **Reconcile the account after every destroy:** read the credit before and after; if the drop exceeds the guard's
      estimate by more than 20%, stop and find out why before renting anything else.
   7. **Cap first, then go:** state the cap in dollars including bandwidth and disk, get the owner's go for that number,
      and treat a failed host's cost as part of that cap. When the cap is consumed, stop and ask; never "one more try".
9. **Ask before renting. This is absolute.** Never start, resume, or bid on a paid instance without the human explicitly saying go for that
   specific run. Present the plan, the expected cost, and the local validation evidence first.

## Never assume docs

- `bullet` changes its API often. Do not write trainer code from memory or from a blog post. Read the
  actual source at the pinned rev (`examples/`, `crates/bullet_lib/src`) and copy the current idioms.
- The same applies to data formats. Read `bulletformat`, `sfbinpack` and `viriformat` source, not a
  description of them. Verify with a round-trip test: write a position, read it back, compare bitboards.
- HuggingFace file names, sizes and availability change. Check the tree API before downloading.
- Stockfish search constants change weekly. When citing a formula, note which commit it came from.
- If a web search summary and the source disagree, the source wins. If unsure, fetch the raw file.

## Engineering rules

- **Perft is law.** Move generation must pass the standard perft suite (start position, Kiwipete,
  positions 3–6, and a Chess960 set) before anything else is built on top of it. Re-run after any
  change to `position.rs` or `movegen.rs`.
- **Bench must be deterministic.** `engine bench` prints a node count; it must be identical across runs
  with the same binary. Any change that alters it must be intentional and noted.
- **Every search change is SPRT-tested** with fastchess, UHO openings, and stated bounds. No merging on
  intuition, no "it's obviously better". Functional-neutral refactors are verified by identical bench.
- **NNUE inference is verified against a scalar reference.** The AVX2 path must produce bit-identical
  output to the plain-Rust path on a set of random positions. Accumulator incremental updates must match
  a from-scratch refresh after every move in a random-game test.
- **`unsafe` only in SIMD and TT code**, each block with a comment stating the invariant that makes it sound.
- Keep the engine buildable with plain `cargo build --release`; `target-cpu=native` comes from
  `.cargo/config.toml`. Provide a scalar fallback so the binary still runs without AVX2.
- No crate dependencies in `engine/` beyond the standard library unless there is a strong reason. The one C
  dependency is the vendored Fathom Syzygy prober (`engine/csrc/fathom`), compiled by `build.rs` with the system
  C compiler; keep it self-contained.
- Prefer copy-make for position state and a separate accumulator stack; keep the search stack explicit.

## A/B testing with one binary

`engine/src/params.rs` holds runtime-tunable integers exposed as UCI options (e.g. `MatScale`, `Cuckoo`). Add a
new feature behind a param defaulting to the old behaviour, verify `bench` is unchanged with it off, then SPRT:
`scripts/sprt.sh sprt/bin/X sprt/bin/X -N "option.Feature=1" -B "option.Feature=0"`. Once a feature passes, make
the new value the default (and note the bench change). Baseline binaries live in `sprt/bin/`, results in `sprt/`.

## Build and test commands

```
cargo build --release                      # engine
./target/release/engine perft 6            # movegen check from the start position
./target/release/engine bench              # deterministic node count + nps
./target/release/engine                    # UCI mode
scripts/sprt.sh <new-binary> <old-binary>  # SPRT match via fastchess
```

## Data sources (free)

- HuggingFace `linrock/test80-2024` (Leela T80, Stockfish-format binpacks, ~7–12 GB per month, zstd).
- HuggingFace `official-stockfish/master-binpacks` (Stockfish-generated, 5–47 GB each).
- Local `lichess_elite_2021-*.pgn` (~2.1M games) — result labels only unless rescored by the engine.
- Own datagen via `engine datagen` — bulletformat output, runs on the laptop at ~1.5k positions/s.

## Working style

- Say what is verified and what is assumed. "Compiles" is not "works". "Works" means a test ran.
- When something fails, report the actual error output; never paper over it.
- Do not widen scope silently. Extra features go in a list for the human to pick from.
- The human is not watching in real time; finish tasks fully, but stop and ask before any paid action.
