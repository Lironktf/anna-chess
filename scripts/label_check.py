#!/usr/bin/env python3
"""Label-quality check for a training set: agreement between the file's scores and the engine's evaluation.

    scripts/label_check.py --engine target/release/engine --dump a.txt [--dump b.txt ...] [--depth 1] [--n 2000]

Each dump file (from `inspect dump`) has lines `fen | score | result` with score and result from the side to move's view
(result in -1/0/1). For each set prints: N, Pearson correlation between file score and engine score (side to move,
clamped to +/-2000), mean absolute difference, the sign-agreement rate on positions where |file score| >= 100, and the
result-vs-score agreement (fraction of decisive results whose sign matches the file score). A broken or mislabelled set
shows up as a clearly lower correlation than its siblings.
"""

import argparse
import math
import subprocess


def engine_scores(engine, fens, depth, setoptions=()):
    cmds = "uci\n" + "".join(f"setoption name {o}\n" for o in setoptions) + "isready\n"
    for f in fens:
        cmds += f"position fen {f}\ngo depth {depth}\n"
    cmds += "quit\n"
    out = subprocess.run([engine], input=cmds, capture_output=True, text=True).stdout
    scores = []
    last = None
    for line in out.splitlines():
        if line.startswith("info") and " score " in line:
            parts = line.split()
            i = parts.index("score")
            if parts[i + 1] == "cp":
                last = int(parts[i + 2])
            else:
                last = 3000 if int(parts[i + 2]) > 0 else -3000
        elif line.startswith("bestmove"):
            scores.append(last)
            last = None
    return scores


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--dump", action="append", required=True)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--setoption", action="append", default=[], help='e.g. "EvalFile value nets/x.bin"')
    args = ap.parse_args()
    for path in args.dump:
        fens, scores, results = [], [], []
        with open(path) as f:
            for line in f:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 3:
                    continue
                fens.append(parts[0])
                scores.append(int(parts[1]))
                results.append(int(parts[2]))
                if len(fens) >= args.n:
                    break
        eng = engine_scores(args.engine, fens, args.depth, args.setoption)
        if len(eng) != len(fens):
            print(f"{path}: engine returned {len(eng)} scores for {len(fens)} positions")
            continue
        clamp = lambda v: max(-2000, min(2000, v))
        xs = [clamp(s) for s in scores]
        ys = [clamp(e) for e in eng]
        r = pearson(xs, ys)
        mad = sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)
        big = [(x, y) for x, y in zip(xs, ys) if abs(x) >= 100]
        sign = sum(1 for x, y in big if (x > 0) == (y > 0)) / max(len(big), 1)
        dec = [(s, res) for s, res in zip(scores, results) if res != 0]
        res_agree = sum(1 for s, res in dec if (s > 0) == (res > 0)) / max(len(dec), 1)
        print(f"{path}: n={len(xs)} corr={r:.3f} mean|diff|={mad:.0f}cp sign-agree(|score|>=100)={100*sign:.1f}% (n={len(big)}) result-agree={100*res_agree:.1f}% (decisive n={len(dec)})")


if __name__ == "__main__":
    main()
