#!/usr/bin/env python3
"""Nodes needed to reach a fixed depth on a set of positions: the load-independent measure of move ordering.

    scripts/nodes_to_depth.py --engine target/release/engine --depth 12 --fens books/... --n 40 \
        --opts "option.EvalFile=X" --opts "option.PolicyFile=Y option.PolicyScale=2000" [--label A --label B]

Each --opts is one engine configuration (space-separated `option.Name=value` items); prints total nodes, geometric
mean of the per-position node ratio against the first configuration, and CPU time.
"""

import argparse
import math
import resource
import subprocess


def run(engine, opts, fens, depth):
    cmds = "uci\n"
    for o in opts.split():
        name, value = o.removeprefix("option.").split("=", 1)
        cmds += f"setoption name {name} value {value}\n"
    cmds += "isready\n"
    for f in fens:
        cmds += f"position fen {f}\ngo depth {depth}\n"
    cmds += "quit\n"
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    out = subprocess.run([engine], input=cmds, capture_output=True, text=True).stdout
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    nodes = []
    last = None
    for line in out.splitlines():
        if line.startswith("info") and " nodes " in line and f" depth {depth} " in line:
            last = int(line.split(" nodes ")[1].split()[0])
        elif line.startswith("bestmove"):
            nodes.append(last or 0)
            last = None
    return nodes, cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--fens", required=True, help="EPD/FEN file, one position per line")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--opts", action="append", required=True)
    ap.add_argument("--label", action="append", default=[])
    args = ap.parse_args()
    fens = []
    with open(args.fens) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                fens.append(" ".join(parts[:4]) + " 0 1")
    fens = fens[args.skip : args.skip + args.n]
    base = None
    for i, opts in enumerate(args.opts):
        label = args.label[i] if i < len(args.label) else f"cfg{i}"
        nodes, cpu = run(args.engine, opts, fens, args.depth)
        total = sum(nodes)
        if base is None:
            base = nodes
            ratio = 1.0
        else:
            logs = [math.log(max(a, 1) / max(b, 1)) for a, b in zip(nodes, base)]
            ratio = math.exp(sum(logs) / len(logs))
        print(f"{label:12s} depth {args.depth}: {len(nodes)} positions, total nodes {total / 1e6:.2f}M, geo-mean ratio vs first {ratio:.3f}, cpu {cpu:.1f}s  [{opts}]")


if __name__ == "__main__":
    main()
