# Run plan: anna-v4 (multilayer/pairwise net WITHOUT threat inputs)

Owner go 2026-09-13 ~10:50 ("launch the thing with no threat inputs"). Purpose: blitz strength with no speed penalty.
Inputs = v1's (768 x 16 king buckets, mirrored) + factoriser; output stack = v3's (pairwise CReLU product -> 16 -> 32 -> 1,
8 material buckets, dual activation). L1 = 1024. Engine module `v3::w1024nt` (HAS_THREATS = false), file size 25,334,336 B.
Data: same 4 months (T80 2024-01/02/03/05). Schedule SB0=30 / SB1=310 / SB2=30 like v3/v3b; expected faster than v3b on the
GPU (no pp block) so the run should be ~1-1.5 h.

Local validation (11:05): engine tests pass for w1024/w512/w1024nt (kernels, incremental == reference, roundtrip); random nt net
loads via netcheck; train_v4 builds and its MockGPU dry run reaches the gradient step; remote scripts accept ARCH=v4.
Box smoke run + netcheck vs "TRAINER EVAL" before the real run, as always.

Budget: credit $2.92. Host offer 48580533 (Norway, RTX 4090 + EPYC 7742 32 thr, ~$0.47/hr all-in). Cap **$2.50**,
self-destruct MAX_HOURS 6. Restricted key: see selfdestruct_key_id.txt (delete after).
Expected: +35 to +60 over v1 at every time control (research estimate; multilayer + more data), ~0.9x of v1's speed.

## Status
- 12:40 local: training at 7.4M pos/s on instance 50903665 (Texas). Midway equal-nodes vs v1 at s1-80 (60 games, 30k nodes): +15 -27 =18, 40% (~-70); v3/v3b were -24/-29 at a similar point, so this run is behind the threat nets' curve. Final verdict at the end.

## Result (2026-09-13 14:10 local)

Final anna-v4-s2-30 vs v1 at equal nodes (30k/move, 200 games): **-83 +/- 33** (+33 -80 =87, 38.25%). Cost $1.62. The +35..+60
prediction was wrong: it assumed a multilayer output stack gains at our data scale, and ignored that v1 was trained on 2.4x the
positions (800 SB vs 340). Checks: engine eval matches trainer eval on the netcheck positions; mean |eval| on 40 book positions
218/224/225 cp for v1/v3b/v4 (no scale problem); running loss 0.0219 vs v3b 0.0207 at SB 310. Follow-up: v4 vs fable-v1-340
(same training budget) at equal nodes, 120 games, to separate "architecture" from "undertrained".

Follow-up result: v4 vs fable-v1-340 at equal nodes, 120 games: **+12 +/- 33** (26-22-72). Same training budget, same strength.
So the deficit against the shipped v1 is training length (800 vs 340 SB), not a broken net; and the multilayer stack itself
bought nothing. The transferable lesson: v3b at 340 SB most likely has +50..+80 left in it from simply training longer
(resume from runs/anna-v3b/checkpoints/anna-v3b-s2-30/raw.bin + optimiser_state). Box confirmation: v4 vs v1 at 8+0.08 -50 +/- 25 (300 games).
