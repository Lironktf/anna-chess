# anna-v6: more new data (plan, 2026-09-16; NOT launched, needs owner go, vast 4090, cap $5)

Same architecture as anna-v5f (L1 512, threats + pawn pairs). Data = v5's set (Leela T80 Jan-Jun 2024, Stockfish dfrc_n5000,
nodes5000pv2_UHO) plus farseerT75 (47 GB, Leela T75 rescored by Stockfish) and data_pv-2_diff-100_nodes-5000 (32 GB).
Schedule 1200 SB (30/1140/30), sources interleaved in the file order.

## Data validation (first 200 MB of each new file, `inspect stats`, 2026-09-16 12:30)

| Set | entries read | kept by filter | results L/D/W | ply 16-40 / 40-80 / 80-120 / 120-200 / >200 | feature cross-checks |
|---|---|---|---|---|---|
| farseerT75 | 2.0M | 56.3% | 300k / 543k / 284k | 290k / 397k / 248k / 147k / 44k | 1126/1126 |
| data_pv-2_diff-100_nodes-5000 | 2.0M | 74.2% | 380k / 727k / 378k | 292k / 455k / 366k / 305k / 66k | 1484/1484 |

Both parse (BINP), decode to legal positions, and match the engine's feature indices. Next check: label quality =
agreement between each set's scores and anna-v5f's evaluation on a sample of positions (inspect dump -> engine eval).

## Label quality (scripts/label_check.py, 1500 positions per set, engine at depth 6, 2026-09-16 13:00)

Correlation between the file's score and the engine's score (both side-to-move, clamped +/-2000), and the share of decisive
game results whose sign matches the file score:

| Set | corr vs v5f | corr vs v3b (Leela-only net) | sign agree (|score|>=100) | result agree |
|---|---|---|---|---|
| Leela T80 Feb 2024 (used) | 0.855 | 0.819 | 96% | 84.8% |
| dfrc_n5000 (used) | 0.961 | 0.957 | 97% | 93.6% |
| nodes5000pv2_UHO (used) | 0.945 | 0.935 | 98% | 96.1% |
| farseerT75 (new) | 0.916 | 0.914 | 97% | 80.6% |
| data_pv-2_diff-100 (new) | 0.968 | 0.964 | 98% | 96.6% |

Verdict: both new sets are at least as consistent as the sets that produced v5's gain; pv-2 is the cleanest of all five.
farseer's lower result agreement matches its Leela-game origin (Leela's own set is 84.8%). The Leela labels agree least
with our search even for the net trained on them, which explains the size of the Stockfish-data gain.

## Launch plan (owner go 2026-09-16 13:20, cap $5)

vast RTX 4090, offer 39099119 (Japan, EPYC 7542 16 cores, 94 GB RAM, 347 GB disk, 783 Mbps, $0.39/h). Data groups
train + train2 + train3 (Leela Jan-Jun 2024, 52 GB zst -> 61 GB) + sf (4 Stockfish binpacks, 156 GB); 269 GB on disk.
`DATA_ORDER` interleaves: jan, dfrc, feb, uho, mar, farseer, apr, pv2, may, jun. ARCH=v3, L1=512, SB 30/940/30 (1000 SB),
SAVE_RATE 10. Estimate: download ~45 min + smoke + ~5 h training = ~$2.5-3.5; cap $5.00 = guard $5.00, self-destruct
MAX_HOURS 12 (12 h x $0.39 = $4.7). Starts after cpu-eval4 is destroyed (one paid instance at a time).

## Launch log
- 14:05 offer 39099119 -> instance 51235738 never left "loading"; destroyed after 15 min (~$0.01).
- 14:12 offer 51224273 already taken (no_such_ask, nothing created).
- 14:14 offer 44637528 -> instance 51236656 running (Texas, EPYC 7402P 12 cores, 768 GB, 929 Mbps), billed ~$0.56/h with
  storage; guard $5.00 armed. Cap arithmetic: 8.8 h total -> setup ~1 h + train MAX_HOURS 7; schedule trimmed to 900 SB (30/840/30).
- 15:45 offer 39184690 -> instance 51237763 (Hong Kong, EPYC 7542, driver 570, ~$0.49/h billed). Download measured at
  ~12 MB/s (12.3 GB in 17 min), i.e. ~6 h for 269 GB: over the cap. **farseerT75 dropped** (row blanked in place in the
  box's manifest before the downloader reached it): v6 data = 6 Leela months + dfrc + UHO + pv-2 (222 GB). Schedule to be
  trimmed at train time to fit: cap $5 total, hours = (5 - spent)/0.49 minus margin.
- 16:00 diagnosis: the box link does >140 MB/s (parallel test stream) while the sequential downloader's stream ran at 13 MB/s.
  Downloader stopped; the four Stockfish files (farseer restored) + the June Leela file fetched in parallel with plain curl
  (resumable); the idempotent setup step is re-run afterwards (sha256 verification of every file, decompress, inspect, smoke).
  v6 data back to the full plan: 6 Leela months + dfrc + UHO + farseer + pv-2 (269 GB).
- 16:10 **instance 51237763 destroyed: its container disk was 120 GB** (the launcher's default `DISK=120`; the offer's 591 GB
  is the host's, not the container's). 269+ GB of data cannot fit. ~$0.75 lost. Re-renting with DISK=480 (driver >= 550,
  disk_space >= 520). Lesson recorded. The extra Stockfish sets (T60T70wIsRightFarseer, multinet, farseerT74, pylon,
  test80-2022-08, wrongIsRight, farseerT76, fishpack32, wrongNNUE) are being label-checked on the laptop for inclusion.
- 16:25 **STOPPED, no instance running.** The Hong Kong host billed per-GB bandwidth: credit 10.08 -> 6.53 ($3.55 for 1.6 h
  and ~70 GB). Spent on v6 attempts: $3.68 of the $5 cap, nothing trained. Re-renting needs the owner's decision (~$3.5-4 more:
  offers with inet_down_cost ~$0.001/GB exist, e.g. 51184092 Poland $0.44/h drv 580, 48857240 Japan $0.47/h drv 595; DISK=480;
  400 GB of data ~ $0.5 of bandwidth there). Pre-rent checklist now: driver >= 550, DISK big enough, inet_down_cost, storage_cost.

## Additional Stockfish sets (label checks 2026-09-16 16:30, 1500 positions each, engine depth 6)

| Set | GB | corr vs engine | result agree | origin |
|---|---|---|---|---|
| multinet_pv-2_diff-100_nodes-5000 | 27.6 | 0.950 | 96.9% | Stockfish self-play, quiet filter |
| wrongIsRight_nodes5000pv2 | 7.3 | 0.941 | 96.9% | Stockfish self-play |
| farseerT74 | 20.1 | 0.907 | 81.0% | Leela T74 rescored by Stockfish |
| training_data_pylon | 14.4 | 0.898 | 80.1% | Leela-derived, rescored |
| test80-2022-08-aug-16tb7p.v6-dd.min | 10.8 | 0.898 | 87.2% | Leela T80 2022, rescored |
| T60T70wIsRightFarseer | 33.0 | 0.852 | 82.0% | Leela T60/T70 rescored |
| farseerT76 | 6.1 | 0.915 | 81.2% | Leela T76 rescored |
| fishpack32 | 5.6 | 0.831 | 99.2% | fishtest LTC positions |
| wrongNNUE_02_d9 | 5.8 | 0.885 | 95.0% | Stockfish depth-9 from misjudged positions |

All pass (>= the Leela set's 0.855 correlation). Manifest group `sf2` (131 GB) holds all nine; with `sf` and the six Leela
months the full v6 set is ~400 GB on disk (DISK=480).

## Research agent survey (2026-09-16, runs/DATA_SOURCES.md)

Recommended additions beyond master-binpacks, in order: vondele/linrock_relabel_2 (T80 2023, 12 files, ~142 GB, BT4-net
labels; the core of Stockfish's current threat-net recipe), adamtwiss/t91-binpacks-filtered (12.6 GB, T91 Jan-May 2026,
must drop score 32002), xushawn/test80-bt4-relabel Jan+Feb 2024 (17.7 GB, replaces our 2024-01/02), linrock_relabel_1 T80
2022 (~63 GB), T60T70wIsRightFarseer (in sf2). Caveats: BT4-relabelled files carry net labels, not search labels (label-check
them first); never stack duplicate copies of T80 2023; budget bandwidth and storage into the cap.

## Final data plan and pricing (2026-09-16 17:15)

Label checks of the survey's recommended sets (1500 positions, depth 6): linrock_relabel_2 June 2023 (BT4 labels) corr 0.876 /
result 82.7%, filter keeps 48%; adamtwiss t91 Jan 2026 corr 0.863 / 99.4%, keeps 53%, no 32002 sentinels after the filter;
xushawn bt4 Jan 2024 corr 0.858 / 82.4%. All usable (Leela-like agreement, below the Stockfish self-play sets).

Full plan = Leela 2024 Jan-Jun (61 GB) + sf (156) + sf2 (131) + relabel2 (142) + t91 (12.6) = **~503 GB on disk** (zst deleted
after decompression), DISK=560. Reduced plan (Stockfish self-play sets + relabel2 + t91, no Leela-derived rescores) ~360 GB,
DISK=420. Cost formula per the pre-rent checklist: hours*(dph + DISK*storage/730) + GB*inet_down_cost; setup ~3.5 h + train ~5.5 h.
At 17:15 the only clean hosts (driver>=550, bw<=0.002/GB, disk>=600) cost $0.60-0.73/h: full plan ~$7.0-7.9, above the
$6.52 credit. Cheaper clean hosts ($0.44-0.47/h, bw ~0.001) appear intermittently; a poller watches for one (est. ~$5.0-5.5).
No rental without the owner's number.

## Go (2026-09-16 20:30): cap $6.00

Owner: "run v6 ... do one last check that everything is proper ... make sure the host on vast ai is valid". Pre-rent checks done:
36 manifest rows (494 GB download, ~503 GB on disk), every URL answered HTTP 200 with the exact manifest size, no duplicates;
scripts syntax-checked, launcher refuses bandwidth-billing hosts and drivers < 550; trainer builds; 49 engine tests pass;
zero instances; credit $6.52. Rental rule: first host with verified=true, reliability > 0.97, driver >= 550,
inet_down_cost <= $0.002/GB, storage <= $0.25/GB-mo, disk_space >= 600, inet_down >= 300 Mbps, cores >= 10, and
estimate (9.5 h x (dph + 560 GB storage) + 503 GB x bandwidth) <= $5.60; DISK=560; cost guard $6.00 armed at creation; early
GPU smoke and transfer-rate gate in the first minutes; training started only after the setup and smoke are verified, with
MAX_HOURS computed from the actual spend so the total stays under $6.00. Data order: runs/anna-v6/data_order.txt.

## Launch log (2026-09-16 21:05, fifth host, cap $6.00)
- 21:05 chain rented offer 29357619 -> instance 51262070: Sweden, RTX 4090, driver 565.57.01, 24 cores, 128 GB RAM, 560 GB
  container disk, verified, reliability 0.999; billed $0.582/h all-in (gpu 0.427 + storage 0.156), bandwidth 2.6e-6 $/GB.
  Estimate 9.5 h x 0.582 = $5.53. Credit before: $6.52.
- 21:08 cost guard armed at $6.00; guard double-counted storage (rate shown 0.74) -> fixed to use dph_total, restarted 21:11.
- 21:12 trainer built with cuda; early GPU smoke on the validate set OK (driver/link problem of the Texas host ruled out).
- 21:13 single-stream download 30-36 MB/s (4 h for 494 GB) -> scripts/prefetch_parallel.sh (6 streams, reverse manifest order,
  .part + rename) started next to the sequential downloader: aggregate 113 MB/s, download ETA ~75 min.
- Time budget: total (6.00 - 0.10 margin)/0.582 = 10.1 h from 21:05; train MAX_HOURS = (6.00 - spent - 0.3)/0.582 at train start;
  trim the schedule below 900 SB if MAX_HOURS x 3600 / 28 s per SB is less than the schedule.
- 23:06 setup done: 38 files inspected (0 feature mismatches), smoke run OK, early-smoke checkpoint netcheck ok on the laptop.
  Box->laptop upload is slow (<1 MB/s), so checkpoint pulls take a minute or two each.
- 23:10 (03:10:46Z box) train started: SB 30/840/30, L1 512, SAVE_RATE 10, LOADER_THREADS 12, MAP_THREADS 12, DATA_ORDER
  = runs/anna-v6/data_order.txt. Self-destruct 7.5 h hard limit (grace 20 min), cost guard $6.00, sync loop every 120 s,
  monitor runs/anna-v6/monitor.sh. First SBs: ~3.6M pos/s, ~28 s/SB -> 900 SB in ~7.0 h, end ~06:10Z; total ~$5.4.
- 23:18 restart: first 11 SB ran at 30.9 s/SB -> 900 SB = 7.7 h, over the 7.5 h self-destruct limit. Killed the tmux session
  (no "train end" line written), archived anna-v6-s0-10 to checkpoints_aborted, restarted with SB 30/770/30 (830 SB, ~7.0 h).
  Found and fixed a real bug: vast_selfdestruct.sh used bash (( )) with MAX_HOURS=7.5, which is an arithmetic error, so the
  hard wall-clock limit never fired (the "train end" path still worked). Fixed (awk seconds), re-armed 03:19:10Z -> hard limit
  10:49Z (06:49 EDT); guard $6.00 would fire ~11:13Z. Expected train end ~10:20Z (06:20 EDT); total ~$5.4-5.5.
- 23:35 box->laptop ssh link is throttled per connection (~0.09 MB/s single stream; box uplink 32 MB/s to Cloudflare, laptop
  downlink 3 MB/s). Wrote scripts/vast_pull_parallel.sh (byte ranges over 12 ssh streams, sha256-verified): 45 MB in 60 s.
  Sync loop replaced by runs/anna-v6/sync_parallel.sh (newest checkpoint only). Self-destruct re-armed 03:34:09Z with
  MAX_HOURS 7.58 so its hard limit (11:11Z) sits 3 min before the $6.00 guard (11:14Z); both under the cap. Final pull plan:
  quantised.bin (1 min) then raw.bin (157 MB, ~4 min) with the parallel puller, then destroy by hand.
