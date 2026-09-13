# anna-v3c: continue training anna-v3b (plan, 2026-09-13)

## Why

anna-v4 vs the v1 checkpoint with the same training budget (340 SB) at equal nodes: +12 +/- 33, i.e. level; vs the shipped v1
(800 SB): -83. So v1's extra 460 superbatches were worth ~+80 Elo per node on the same data. v3b stopped at 340 SB (+83 over
v1@800 already). The cheapest large gain left is to keep training v3b: resume from anna-v3b-s2-30 (raw.bin + optimiser_state,
both local) for another ~430 SB with fresh months added (T80 2024-04 and 2024-06 on top of 01/02/03/05), then the usual 30-SB
pure-WDL stage. Expected (a guess from the v1 curve, not a measurement): +30 to +70 per node over v3b. Verified only by the
equal-nodes match at the end.

## Where

Modal (owner's $30/month Starter credit, nothing from the $10 vast budget). `scripts/modal_train.py` (pinned like the vast runs:
CUDA 12.4.1 image, rustup 1.98.1, bullet 629ee50). L40S at $1.95/h: ~15 s/SB like the 4090 -> 460 SB ~ 2 h -> ~$4 + CPU ~$1.
H100 at $3.95/h would be ~1.3 h -> ~$5.5. Data fetch (6 months, ~52 GB zst) on a CPU-only function: ~$0.3.

## Owner steps (once)

    pip install modal && modal token new

## Run

    modal run scripts/modal_train.py --action fetch --months 01,02,03,04,05,06
    modal volume put anna-data runs/anna-v3b/checkpoints/anna-v3b-s2-30/optimiser_state /resume/anna-v3b-s2-30/optimiser_state
    ANNA_GPU=L40S ANNA_HOURS=4 modal run --detach scripts/modal_train.py --action train --net-id anna-v3c \
        --months 01,02,03,04,05,06 --resume /data/resume/anna-v3b-s2-30 --sb0 0 --sb1 430 --sb2 30 --lr1 5e-4 --l1 512
    modal run scripts/modal_train.py --action status --net-id anna-v3c
    modal volume get anna-data /checkpoints/anna-v3c/anna-v3c-s2-30/quantised.bin runs/anna-v3c/checkpoints/anna-v3c-s2-30/quantised.bin

Smoke first (same commands with `--net-id anna-v3c-smoke --sb1 2 --sb2 1 ANNA_HOURS=0.5`): pull the checkpoint, `engine netcheck`,
then the real run. Kill: `modal app stop anna-train`. Hard cap: ANNA_HOURS (the function's timeout).

## Local validation done

- trainer builds with RESUME/LR1; `RESUME=<v3b s2-30> SB0=SB1=SB2=0` on the MockGPU loads weights + optimiser state
  ("resumed weights and optimiser state from ..."); the MockGPU cannot evaluate, so the eval comparison happens on the smoke run.
- Schedule: SB0=0 (no warmup), stage 1 LR 5e-4 -> 1e-6 over 430 SB with WDL 0.2 -> 0.5, stage 2 30 SB pure WDL at 1e-5 -> 1e-7.
