"""Train the move-ordering policy net on lc0conv records (PyTorch). Runs locally (CPU, for smoke tests) or on Modal.

Record layout (lc0conv, 304 bytes): see lc0conv/src/main.rs `write_record`.

Model (see runs/policy/PLAN.md):
    features  f = side-to-move relative piece-square indices (12 x 64 = 768; ranks flipped when black to move,
                colours swapped so "our" pieces are 0..5), one per occupied square.
    acc       a = sum(W[f]) + b                      (768 -> H, H = 256)
    hidden    h = clamp(a, 0, 1)                     (CReLU; quantises cleanly)
    logit(m)  = h . V[pt(m), to_rel(m)] + U[from_rel(m), to_rel(m)] + P[promo(m)]
    loss      = cross-entropy(softmax over the position's legal moves, Lc0 visit distribution)

Usage:
    python scripts/policy_train.py --data a.bin,b.bin --epochs 2 --out runs/policy/policy.bin [--limit N] [--eval f.bin]
Exports: <out> (quantised, engine format) and <out>.pt (float state dict). Engine format (little-endian):
    magic "ANPOL1", H u32, then W i16[768*H] (scale 256), b i16[H] (scale 256), V i8[6*64*H] (scale 64),
    U i16[64*64] (scale 64*256/256 = 64), P i16[5] (scale 64).
"""

import argparse
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F

REC = 304
MAXM = 64
H = 256


def load_records(paths, limit=None):
    arrs = []
    for p in paths:
        a = np.fromfile(p, dtype=np.uint8)
        a = a[: (len(a) // REC) * REC].reshape(-1, REC)
        arrs.append(a)
    a = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    if limit:
        a = a[:limit]
    return a


def decode_batch(rec: np.ndarray):
    """rec: (B, 240) uint8 -> feature indices (B, 32) padded with -1, move tuples (B, 48, 3) [pt, from, to] with
    -1 padding, promo (B, 48), target probs (B, 48), n_moves (B,), stm (B,)."""
    B = rec.shape[0]
    occ = rec[:, 0:8].copy().view(np.uint64).reshape(B)
    nib = rec[:, 8:24]
    stm = rec[:, 24].astype(np.int64)  # 1 = black to move
    # Piece list in ascending square order.
    sqs = np.full((B, 32), -1, dtype=np.int64)
    codes = np.full((B, 32), -1, dtype=np.int64)
    bits = occ.copy()
    for i in range(32):
        nz = bits != 0
        sq = np.zeros(B, dtype=np.int64)
        # trailing zeros
        v = bits[nz]
        tz = np.zeros(v.shape, dtype=np.int64)
        vv = v.copy()
        for shift in (32, 16, 8, 4, 2, 1):
            mask = (vv & ((np.uint64(1) << np.uint64(shift)) - np.uint64(1))) == 0
            tz[mask] += shift
            vv[mask] >>= np.uint64(shift)
        sq[nz] = tz
        code = np.where(i % 2 == 0, nib[:, i // 2] & 15, nib[:, i // 2] >> 4).astype(np.int64)
        sqs[nz, i] = sq[nz]
        codes[nz, i] = code[nz]
        bits[nz] = v & (v - np.uint64(1))
    # Side-to-move relative: flip ranks for black, swap colours.
    rel_sq = np.where(stm[:, None] == 1, sqs ^ 56, sqs)
    colour = codes // 6
    ptype = codes % 6
    rel_colour = np.where(stm[:, None] == 1, 1 - colour, colour)
    feat = np.where(codes >= 0, (rel_colour * 6 + ptype) * 64 + rel_sq, -1)
    # Board lookup for the moving piece type.
    board = np.full((B, 64), -1, dtype=np.int64)
    valid = codes >= 0
    bi = np.repeat(np.arange(B), 32).reshape(B, 32)
    board[bi[valid], sqs[valid]] = ptype[valid]
    # Moves.
    ent = rec[:, 36 : 36 + MAXM * 4].reshape(B, MAXM, 2, 2)
    mv = ent[:, :, 0, :].copy().view(np.uint16).reshape(B, MAXM).astype(np.int64)
    prob = ent[:, :, 1, :].copy().view(np.uint16).reshape(B, MAXM).astype(np.float32) / 65535.0
    n_moves = rec[:, 32].astype(np.int64)
    slot = np.arange(MAXM)[None, :]
    present = slot < n_moves[:, None]
    frm = mv & 63
    to = (mv >> 6) & 63
    flag = (mv >> 14) & 3
    promo = np.where(flag == 1, ((mv >> 12) & 3) + 1, 0)  # 1 N 2 B 3 R 4 Q
    pt = board[np.arange(B)[:, None], frm]
    pt = np.where(present, pt, -1)
    rel_from = np.where(stm[:, None] == 1, frm ^ 56, frm)
    rel_to = np.where(stm[:, None] == 1, to ^ 56, to)
    moves = np.stack([np.where(present, pt, -1), np.where(present, rel_from, -1), np.where(present, rel_to, -1)], axis=-1)
    prob = np.where(present, prob, 0.0)
    s = prob.sum(axis=1, keepdims=True)
    prob = np.where(s > 0, prob / np.maximum(s, 1e-9), prob)
    return feat, moves, promo, prob.astype(np.float32), n_moves, stm


class PolicyNet(torch.nn.Module):
    def __init__(self, h=H):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(768, h) * 0.02)
        self.b = torch.nn.Parameter(torch.zeros(h))
        self.V = torch.nn.Parameter(torch.randn(6, 64, h) * 0.02)
        self.U = torch.nn.Parameter(torch.zeros(64, 64))
        self.P = torch.nn.Parameter(torch.zeros(5))

    def forward(self, feat, moves, promo):
        # feat (B, 32) with -1 padding; moves (B, M, 3); promo (B, M)
        fmask = (feat >= 0).float().unsqueeze(-1)
        acc = (self.W[feat.clamp(min=0)] * fmask).sum(1) + self.b  # (B, H)
        h = acc.clamp(0.0, 1.0)
        pt = moves[..., 0].clamp(min=0)
        fr = moves[..., 1].clamp(min=0)
        to = moves[..., 2].clamp(min=0)
        v = self.V[pt, to]  # (B, M, H)
        logit = (v * h.unsqueeze(1)).sum(-1) + self.U[fr, to] + self.P[promo]
        logit = logit.masked_fill(moves[..., 0] < 0, -1e9)
        return logit


def export(model: PolicyNet, path: str):
    W = (model.W.detach().cpu().numpy() * 256).round().clip(-32767, 32767).astype(np.int16)
    b = (model.b.detach().cpu().numpy() * 256).round().clip(-32767, 32767).astype(np.int16)
    V = (model.V.detach().cpu().numpy() * 64).round().clip(-127, 127).astype(np.int8)
    U = (model.U.detach().cpu().numpy() * 64).round().clip(-32767, 32767).astype(np.int16)
    P = (model.P.detach().cpu().numpy() * 64).round().clip(-32767, 32767).astype(np.int16)
    with open(path, "wb") as f:
        f.write(b"ANPOL1")
        f.write(struct.pack("<I", model.W.shape[1]))
        f.write(W.tobytes())
        f.write(b.tobytes())
        f.write(V.tobytes())
        f.write(U.tobytes())
        f.write(P.tobytes())
    sat = float((np.abs(model.V.detach().cpu().numpy() * 64) > 127).mean())
    print(f"exported {path}: W {W.shape} b {b.shape} V {V.shape} U {U.shape} P {P.shape}; V saturation {sat:.4%}")


def evaluate(model, rec, device, batch=4096):
    model.eval()
    tot, top1, top3, ce = 0, 0, 0, 0.0
    with torch.no_grad():
        for i in range(0, len(rec), batch):
            feat, moves, promo, prob, n, _ = decode_batch(rec[i : i + batch])
            feat_t = torch.from_numpy(feat).to(device)
            moves_t = torch.from_numpy(moves).to(device)
            promo_t = torch.from_numpy(promo).to(device)
            prob_t = torch.from_numpy(prob).to(device)
            logit = model(feat_t, moves_t, promo_t)
            logp = F.log_softmax(logit, dim=-1)
            ce += float(-(prob_t * logp).sum())
            best = prob_t.argmax(-1)
            order = logit.argsort(-1, descending=True)
            top1 += int((order[:, 0] == best).sum())
            top3 += int((order[:, :3] == best[:, None]).any(-1).sum())
            tot += len(feat)
    model.train()
    return ce / max(tot, 1), top1 / max(tot, 1), top3 / max(tot, 1)


def _decode_worker(args):
    path, start, idx = args
    a = np.memmap(path, dtype=np.uint8, mode="r")
    a = a[: (len(a) // REC) * REC].reshape(-1, REC)
    return decode_batch(np.ascontiguousarray(a[idx]))


def iterate_batches(paths, batch, workers, seed, hold_out, limit=0):
    """Yield decoded batches over all files (each file loaded, permuted, decoded in worker processes). The last
    `hold_out` records of the last file are never yielded (they are the evaluation set)."""
    import concurrent.futures
    rng = np.random.default_rng(seed)
    order = list(paths)
    rng.shuffle(order)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for path in order:
            n = (np.memmap(path, dtype=np.uint8, mode="r").shape[0]) // REC
            if limit:
                n = min(n, limit)
            if path == paths[-1]:
                n -= hold_out
            perm = rng.permutation(n)
            jobs = [(path, 0, np.sort(perm[i : i + batch])) for i in range(0, n - batch + 1, batch)]
            for out in ex.map(_decode_worker, jobs, chunksize=2):
                yield out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=H)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dump-logits", default="", help="write reference logits for the first --dump-n eval records")
    ap.add_argument("--dump-n", type=int, default=300)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = args.data.split(",")
    # Evaluation set: the tail of the last file (or --eval). Files are streamed and decoded in worker processes.
    last = np.memmap(paths[-1], dtype=np.uint8, mode="r")
    n_last = last.shape[0] // REC
    total = sum(np.memmap(p, dtype=np.uint8, mode="r").shape[0] // REC for p in paths)
    if args.limit:
        # --limit: a smoke run over the first N records of the first file only.
        paths = paths[:1]
        total = min(args.limit, np.memmap(paths[0], dtype=np.uint8, mode="r").shape[0] // REC)
        n_last = total
    hold_out = min(100_000, max(n_last // 10, 1)) if not args.eval else 0
    if args.eval:
        ev = load_records([args.eval], 200_000)
        ev_offset = 0
    else:
        ev_offset = n_last - hold_out
        ev = np.ascontiguousarray(np.memmap(paths[-1], dtype=np.uint8, mode="r")[: n_last * REC].reshape(-1, REC)[ev_offset:n_last])
    n_train = total - hold_out
    print(f"train ~{n_train} records over {len(paths)} files, eval {len(ev)}, device {device}, workers {args.workers}", flush=True)
    model = PolicyNet(args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps = args.epochs * (n_train // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.05)
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        for feat, moves, promo, prob, n, _ in iterate_batches(paths, args.batch, args.workers, seed=1000 + ep, hold_out=hold_out, limit=args.limit):
            if step >= steps:
                break
            feat_t = torch.from_numpy(feat).to(device)
            moves_t = torch.from_numpy(moves).to(device)
            promo_t = torch.from_numpy(promo).to(device)
            prob_t = torch.from_numpy(prob).to(device)
            logit = model(feat_t, moves_t, promo_t)
            loss = -(prob_t * F.log_softmax(logit, dim=-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        ce, t1, t3 = evaluate(model, ev, device)
        print(f"epoch {ep}: eval CE {ce:.4f} top1 {t1:.4f} top3 {t3:.4f}", flush=True)
    export(model, args.out)
    torch.save(model.state_dict(), args.out + ".pt")
    if args.dump_logits:
        dump_logits(model, ev[: args.dump_n], device, args.dump_logits, ev_offset)


def dump_logits(model, rec, device, path, offset=0):
    """Reference logits for `engine policycheck`: one line per (absolute record index, move as engine u16, float logit)."""
    model.eval()
    feat, moves, promo, prob, n, _ = decode_batch(rec)
    with torch.no_grad():
        logit = model(torch.from_numpy(feat).to(device), torch.from_numpy(moves).to(device), torch.from_numpy(promo).to(device)).cpu().numpy()
    ent = rec[:, 36 : 36 + MAXM * 4].reshape(len(rec), MAXM, 2, 2)
    mv = ent[:, :, 0, :].copy().view(np.uint16).reshape(len(rec), MAXM)
    with open(path, "w") as f:
        for i in range(len(rec)):
            for j in range(int(n[i])):
                f.write(f"{offset + i} {int(mv[i, j])} {float(logit[i, j]):.5f}\n")
    print(f"wrote {path}: {len(rec)} records")


if __name__ == "__main__":
    main()
