# Run plan: anna-v3b (threat-input net at half width, L1 = 512)

Owner go 2026-09-13 00:15 ("go do the half width stuff. cap it and be very strict.").
Why: anna-v3 (L1 1024) is +83/node over v1 but 3.7x slower (memory-latency bound on ~14 random 1 KB rows per node);
half-width rows halve the traffic (laptop estimate ~330k nps vs 250k) while keeping most of the evaluation gain.

Same data (T80 2024-01/02/03/05, manifest groups train+train2), same 3-stage schedule SB0=30 / SB1=310 / SB2=30,
batch 131072 x 763, trainer `train_v3` with `L1=512` (env). Engine loads both widths (v3::w512), detected by file size.

Local validation (2026-09-13 00:10-00:25): engine tests pass for both widths (kernels scalar==avx2, incremental==reference,
roundtrip); `randomnet-v3 out 512` + `netcheck --strict` ok; trainer builds and its MockGPU dry run with L1=512 reaches the
gradient step; vast scripts pass L1 through. Box smoke run + netcheck vs "TRAINER EVAL" lines before the real run, as for v3.

Budget: credit $4.29. Host offer 29761291 (Quebec, RTX 4090 + 32 threads EPYC 9554, ~$0.48/hr all-in). Expected 4-5 h,
~$2.0-2.5. Cost guard cap **$3.90** (owner 00:40: may use all credit, keep ~$0.30 for the validation box), self-destruct MAX_HOURS 9. Restricted key id 27906658 (delete after).

## Status
- 00:44 local: training running on instance 50845524 (Quebec, EPYC 9554 + 4090). Smoke: engine reads the 512 net as "v3 half width", evals match the trainer (10 vs 10.5, -49 vs -49.0). Throughput 6.7M pos/s (GPU 100%), ~15 s/SB -> ~1.6 h, ~$0.90. Self-destruct armed (max 9 h), guard cap $3.90, sync every 2 min.
- 02:20 local: DONE. 370 superbatches in 1.6 h (6.7M pos/s, GPU-bound). Final net runs/anna-v3b/checkpoints/anna-v3b-s2-30/quantised.bin
  (45,641,280 bytes) + raw.bin/optimiser_state (649 MB). netcheck --strict ok; engine evals vs trainer: 29/29.8, -1592/-1628,
  1706/1746, -396/-399, -1764/-1772, 11/13.4. Instance destroyed 02:39, instances 0, key deleted. Cost $1.10 (credit 3.19 left).
- Midway equal-nodes vs v1 (60 games, 30k nodes, at 60/310): -29 +/- 59.
- Engine speed (same day): split accumulator removed the per-2.5-evals full rebuild; v3b measured ~0.56x of v1's nodes per
  CPU-second under load (v3: 0.43x). Validation: equal-nodes vs v1 on the laptop (running), then the permitted CPU box.
