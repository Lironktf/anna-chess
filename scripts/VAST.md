# vast.ai procedure

Budget cap is $10 total (see `BUDGET.md`). Nothing here is run without the human saying go.

## One-time laptop setup

1. CLI: `~/.local/bin/vastai` (single-file `vast.py`; works, prints a deprecation warning).
   The official package needs pip: `sudo pacman -S python-pip && pip install --user vastai` if wanted.
2. API key: create at https://cloud.vast.ai/account/ and run in the terminal:
   `! vastai set api-key <KEY>`   (stored in ~/.config/vastai/vast_api_key, never in the repo)
3. SSH key: `~/.ssh/id_ed25519.pub` exists. Register it once: `vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"`
4. Verify: `vastai show user` prints the account and balance.

## Choosing an instance

```
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.98 disk_space>=120 inet_down>=500 cuda_vers>=12.4 rentable=true' -o 'dph+' | head -15
```
Pick on-demand, not interruptible, for the real run: a killed spot instance mid-run wastes the warm-up.
Target price: under $0.35/hr. Disk: 120 GB (two months of T80 data decompress to ~50 GB).

## Run procedure (checklist, in order)

- [ ] Local end-to-end validation passed (trainer builds, tiny CPU run, engine loads the checkpoint).
- [ ] `BUDGET.md` has the planned line item.
- [ ] `vastai create instance <ID> --image nvidia/cuda:12.4.1-devel-ubuntu22.04 --disk 120 --ssh --direct`
- [ ] `vastai show instances` to get host/port; wait for status `running`.
- [ ] Upload trainer + scripts: `rsync -az -e "ssh -p PORT" trainer/ scripts/ data/MANIFEST.tsv root@HOST:/workspace/trainer/`
- [ ] `ssh -p PORT root@HOST 'BULLET_REV=... RUN_NAME=... bash /workspace/trainer/scripts/vast_setup_remote.sh'`
- [ ] Start `scripts/vast_sync_checkpoints.sh RUN HOST PORT` on the laptop in its own terminal.
- [ ] Start training inside tmux on the instance so an SSH drop does not kill it.
- [ ] Confirm the first synced checkpoint passes `engine netcheck` on the laptop.
- [ ] When done: pull the final checkpoint, `vastai destroy instance <ID>`, confirm in the web console.
- [ ] Record actual cost in `BUDGET.md`, copy training log into `runs/<run>/`.
