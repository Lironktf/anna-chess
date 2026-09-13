"""Train (or continue training) an Anna net on Modal instead of vast.ai.

Same pins as the vast runs: nvidia/cuda:12.4.1-devel-ubuntu22.04, rustup 1.98.1, bullet rev 629ee50 (Cargo.toml).
Data and checkpoints live in the Modal Volume "anna-data":
    /data/binpacks/<month>.binpack       T80 2024 months, decompressed once by `fetch`
    /data/resume/<name>/optimiser_state  a local checkpoint uploaded with `modal volume put` (for RESUME)
    /data/checkpoints/<net_id>/...       checkpoints written by the trainer (quantised.bin, raw.bin, optimiser_state)
    /data/runs/<net_id>/train.log        trainer stdout

One-time setup on the laptop (owner):   pip install modal && modal token new
Commands (all from the repo root):
    modal run scripts/modal_train.py --action fetch --months 01,02,03,04,05,06
    modal volume put anna-data runs/anna-v3b/checkpoints/anna-v3b-s2-30/optimiser_state /resume/anna-v3b-s2-30/optimiser_state
    ANNA_GPU=L40S ANNA_HOURS=4 modal run --detach scripts/modal_train.py --action train --net-id anna-v3c \
        --months 01,02,03,04,05,06 --resume /data/resume/anna-v3b-s2-30 --sb1 430 --sb2 30 --lr1 5e-4 --l1 512
    modal run scripts/modal_train.py --action status --net-id anna-v3c
    modal volume get anna-data /checkpoints/anna-v3c/anna-v3c-s2-30/quantised.bin runs/anna-v3c/checkpoints/anna-v3c-s2-30/quantised.bin
Cost control: the train function's `timeout` (from ANNA_HOURS) is the hard wall-clock cap; `modal app stop anna-train` kills it early.
"""

import os
import pathlib
import subprocess
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
HF = "https://huggingface.co/datasets/linrock/test80-2024/resolve/main"
MONTH_FILES = {
    "01": "test80-2024-01-jan-2tb7p.min-v2.v6.binpack",
    "02": "test80-2024-02-feb-2tb7p.min-v2.v6.binpack",
    "03": "test80-2024-03-mar-2tb7p.min-v2.v6.binpack",
    "04": "test80-2024-04-apr-2tb7p.min-v2.v6.binpack",
    "05": "test80-2024-05-may-2tb7p.min-v2.v6.binpack",
    "06": "test80-2024-06-jun-2tb7p.min-v2.v6.binpack",
}

app = modal.App("anna-train")
vol = modal.Volume.from_name("anna-data", create_if_missing=True)
# GPU type and hard wall-clock cap for `train`, chosen per launch: ANNA_GPU=L40S ANNA_HOURS=4 modal run --detach ...
GPU = os.environ.get("ANNA_GPU", "L40S")
HOURS = float(os.environ.get("ANNA_HOURS", "4"))


def _skip(p: pathlib.Path) -> bool:
    return any(part in ("target", "checkpoints", "data") for part in p.parts) or p.suffix in (".log", ".pgn")


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "curl", "git", "zstd", "pkg-config", "ca-certificates")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1")
    .env({"PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "CUDA_PATH": "/usr/local/cuda"})
    .add_local_dir(ROOT / "engine", "/root/chess/engine", copy=True, ignore=_skip)
    .add_local_dir(ROOT / "trainer", "/root/chess/trainer", copy=True, ignore=_skip)
    .add_local_file(ROOT / "nets" / "default.bin", "/root/chess/nets/default.bin", copy=True)
    .add_local_dir(ROOT / "lc0conv", "/root/chess/lc0conv", copy=True, ignore=_skip)
    .run_commands(
        "cd /root/chess/trainer && cargo build --release --features cuda --bin train_v3 --bin inspect 2>&1 | tail -n 5",
        # lc0conv is a member of the repo's root workspace; give it a minimal workspace here (engine + lc0conv only).
        "printf '[workspace]\\nresolver = \"2\"\\nmembers = [\"engine\", \"lc0conv\"]\\n' > /root/chess/Cargo.toml && cd /root/chess && cargo build --release -p lc0conv 2>&1 | tail -n 3",
    )
)

LC0 = "https://storage.lczero.org/files/training_data/test80"


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=16384, timeout=6 * 3600)
def policy_data(tars: list[str]) -> str:
    """Stream Lc0 T80 training tars through lc0conv into /data/policy/<tar>.bin (8 at a time, idempotent)."""
    import concurrent.futures
    os.makedirs("/data/policy", exist_ok=True)

    def one(name: str) -> str:
        dst = f"/data/policy/{name}.bin"
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return f"have {name}"
        cmd = f"curl -sSL --retry 5 '{LC0}/{name}.tar' | /root/chess/target/release/lc0conv convert '{dst}.part' - && mv '{dst}.part' '{dst}'"
        t = time.time()
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        if r.returncode != 0:
            return f"FAILED {name}: {r.stderr[-300:]}"
        return f"{name}: {r.stdout.strip().splitlines()[-1][:160]} ({time.time() - t:.0f} s)"

    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for line in ex.map(one, tars):
            print(line)
            out.append(line)
            vol.commit()
    return "\n".join(out)


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=16384, timeout=4 * 3600)
def fetch(months: list[str]) -> list[str]:
    """Download and decompress the requested months into the volume (idempotent)."""
    out = []
    os.makedirs("/data/binpacks", exist_ok=True)
    for m in months:
        name = MONTH_FILES[m]
        dst = f"/data/binpacks/{name}"
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000_000:
            print(f"have {dst} ({os.path.getsize(dst) / 1e9:.1f} GB)")
            out.append(dst)
            continue
        t = time.time()
        cmd = f"curl -sSL --retry 5 '{HF}/{name}.zst' | zstd -d -T4 -o '{dst}.part' && mv '{dst}.part' '{dst}'"
        print(f"fetch {name} ...")
        subprocess.run(["bash", "-c", cmd], check=True)
        vol.commit()
        print(f"done {name}: {os.path.getsize(dst) / 1e9:.1f} GB in {time.time() - t:.0f} s")
        out.append(dst)
    return out


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=32768, gpu=GPU, timeout=int(HOURS * 3600) + 900)
def train(net_id: str, months: list[str], resume: str, sb0: int, sb1: int, sb2: int, lr1: str, l1: int, hours: float) -> str:
    """Run the pinned trainer; checkpoints and the log go to the volume; hard wall-clock cap = hours."""
    data = ",".join(f"/data/binpacks/{MONTH_FILES[m]}" for m in months)
    for p in data.split(","):
        assert os.path.exists(p), f"missing {p}: run --action fetch first"
    if resume:
        assert os.path.isdir(f"{resume}/optimiser_state"), f"missing {resume}/optimiser_state (modal volume put ...)"
    out_dir = f"/data/checkpoints/{net_id}"
    run_dir = f"/data/runs/{net_id}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ, NET_ID=net_id, DATA=data, OUT_DIR=out_dir, SB0=str(sb0), SB1=str(sb1), SB2=str(sb2), LR1=lr1, L1=str(l1), SAVE_RATE="10")
    if resume:
        env["RESUME"] = resume
    log = open(f"{run_dir}/train.log", "a", buffering=1)
    log.write(f"=== train start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} net={net_id} data={data} resume={resume} SB={sb0}/{sb1}/{sb2} LR1={lr1} L1={l1} cap={hours}h\n")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], stdout=log, stderr=subprocess.STDOUT)
    proc = subprocess.Popen(["/root/chess/trainer/target/release/train_v3"], cwd="/root/chess/trainer", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + hours * 3600
    last_commit = time.time()
    for line in proc.stdout:
        log.write(line)
        if time.time() - last_commit > 60:
            vol.commit()
            last_commit = time.time()
        if time.time() > deadline:
            log.write(f"=== wall-clock cap {hours} h reached, killing the trainer\n")
            proc.kill()
            break
    rc = proc.wait()
    log.write(f"=== train end exit={rc}\n")
    log.close()
    vol.commit()
    return f"exit={rc}; checkpoints in {out_dir}; log {run_dir}/train.log"


@app.function(image=image, volumes={"/data": vol}, timeout=600)
def status(net_id: str) -> str:
    vol.reload()
    parts = []
    log = f"/data/runs/{net_id}/train.log"
    if os.path.exists(log):
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 3000))
            parts.append(f.read().decode(errors="replace"))
    ck = f"/data/checkpoints/{net_id}"
    if os.path.isdir(ck):
        parts.append("checkpoints: " + " ".join(sorted(os.listdir(ck))))
    return "\n".join(parts) or "nothing yet"


@app.local_entrypoint()
def main(action: str = "status", net_id: str = "anna-v3c", months: str = "01,02,03,05", resume: str = "",
         sb0: int = 0, sb1: int = 430, sb2: int = 30, lr1: str = "5e-4", l1: int = 512, tars: str = ""):
    ms = [m.strip() for m in months.split(",") if m.strip()]
    if action == "fetch":
        print(fetch.remote(ms))
    elif action == "policy-data":
        # --tars is a comma-separated list of training-run1-test80-YYYYMMDD-HHMM names (see runs/policy/PLAN.md).
        names = [t.strip() for t in tars.split(",") if t.strip()]
        print(policy_data.remote([f"training-run1-test80-{n}" for n in names]))
    elif action == "train":
        print(f"gpu={GPU} cap={HOURS} h (ANNA_GPU / ANNA_HOURS)")
        print(train.remote(net_id, ms, resume, sb0, sb1, sb2, lr1, l1, HOURS))
    elif action == "status":
        print(status.remote(net_id))
    else:
        raise SystemExit("action must be fetch | train | status")
