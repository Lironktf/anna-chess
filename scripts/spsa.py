#!/usr/bin/env python3
"""SPSA tuner for the engine's UCI parameters, driven by fastchess.

Each iteration perturbs every parameter by +/- c_k (random sign per parameter), plays a short
match theta+ vs theta- (both sides are the same binary with different option values), and moves
theta along the measured score difference. Standard SPSA with the Stockfish/fishtest style
per-parameter step sizes: c_k = c_end * (N/k)^gamma, a_k = a_end * ((N+A)/(k+A))^alpha where
a_end = r_end * c_end^2, so the final update per iteration is r_end * c_end * result.

State (theta, iteration) is written after every iteration to <out>/state.json and the run can be
resumed by re-running the same command. Results in <out>/spsa.log. At the end (or any time),
<out>/best.txt holds the rounded parameters to verify with an SPRT against the defaults.

Usage:
  scripts/spsa.py --engine target/release/engine --iters 1000 --games 4 --tc 8+0.08 --conc 4 \
      --out sprt/spsa1 --param LmrBase:982:0:3000:60 --param RfpMult:45:20:120:6 ...
  (param = name:start:min:max:c_end; c_end ~ 1/20 of the plausible range)
"""
import argparse, json, os, random, subprocess, sys, time, re, math

def parse_param(s):
    name, start, lo, hi, c = s.split(":")
    return {"name": name, "start": float(start), "min": float(lo), "max": float(hi), "c": float(c)}

def play(args, theta_plus, theta_minus, k):
    opts_p = " ".join(f"option.{n}={int(round(v))}" for n, v in theta_plus.items())
    opts_m = " ".join(f"option.{n}={int(round(v))}" for n, v in theta_minus.items())
    rounds = max(1, args.games // 2)
    cmd = [args.fastchess,
           "-engine", f"cmd={args.engine}", "name=plus", *opts_p.split(),
           "-engine", f"cmd={args.engine}", "name=minus", *opts_m.split(),
           "-each", f"tc={args.tc}", "proto=uci", f"option.Hash={args.hash}",
           "-openings", f"file={args.book}", "format=epd", "order=random",
           "-concurrency", str(args.conc), "-rounds", str(rounds), "-games", "2", "-repeat", "-recover",
           "-ratinginterval", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600).stdout
    m = re.findall(r"Games:\s*(\d+),\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)", out)
    if not m:
        # fall back to counting result lines
        w = len(re.findall(r"Finished game \d+ \(plus vs minus\): 1-0|Finished game \d+ \(minus vs plus\): 0-1", out))
        l = len(re.findall(r"Finished game \d+ \(plus vs minus\): 0-1|Finished game \d+ \(minus vs plus\): 1-0", out))
        d = len(re.findall(r"Finished game \d+ .*: 1/2-1/2", out))
        g = w + l + d
    else:
        g, w, l, d = map(int, m[-1])
    if g == 0:
        raise RuntimeError("fastchess produced no games:\n" + out[-2000:])
    return (w - l) / g, g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--fastchess", default="tools/fastchess/fastchess")
    ap.add_argument("--book", default="books/UHO_Lichess_4852_v1.epd")
    ap.add_argument("--tc", default="8+0.08")
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--games", type=int, default=4, help="games per iteration (even)")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--r-end", type=float, default=0.01, help="final learning rate; fishtest uses 0.002 with far more games")
    ap.add_argument("--alpha", type=float, default=0.602)
    ap.add_argument("--gamma", type=float, default=0.101)
    ap.add_argument("--out", required=True)
    ap.add_argument("--param", action="append", required=True, help="name:start:min:max:c_end")
    args = ap.parse_args()
    params = [parse_param(p) for p in args.param]
    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "state.json")
    log_path = os.path.join(args.out, "spsa.log")
    if os.path.exists(state_path):
        st = json.load(open(state_path))
        theta = st["theta"]; k0 = st["k"]
    else:
        theta = {p["name"]: p["start"] for p in params}; k0 = 0
    N = args.iters; A = 0.1 * N
    log = open(log_path, "a")
    def L(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True); log.write(line + "\n"); log.flush()
    L(f"spsa start engine={args.engine} tc={args.tc} games/iter={args.games} iters={N} from k={k0} params={[p['name'] for p in params]}")
    total_games = 0
    for k in range(k0 + 1, N + 1):
        ck = (N / k) ** args.gamma
        ak = ((N + A) / (k + A)) ** args.alpha
        delta = {p["name"]: random.choice((-1.0, 1.0)) for p in params}
        tp, tm = {}, {}
        for p in params:
            c = p["c"] * ck
            tp[p["name"]] = min(p["max"], max(p["min"], theta[p["name"]] + c * delta[p["name"]]))
            tm[p["name"]] = min(p["max"], max(p["min"], theta[p["name"]] - c * delta[p["name"]]))
        try:
            result, g = play(args, tp, tm, k)
        except Exception as e:
            L(f"iter {k}: match failed: {e}"); time.sleep(5); continue
        total_games += g
        for p in params:
            # SPSA: theta_i += a_k * result / (2 c_k delta_i) with a_k = r_end c_end^2 ((N+A)/(k+A))^alpha
            c_k = p["c"] * ck
            a_k = args.r_end * p["c"] * p["c"] * ak
            theta[p["name"]] = min(p["max"], max(p["min"], theta[p["name"]] + a_k * result * delta[p["name"]] / (2.0 * c_k)))
        json.dump({"theta": theta, "k": k}, open(state_path, "w"))
        with open(os.path.join(args.out, "best.txt"), "w") as f:
            for p in params:
                f.write(f"option.{p['name']}={int(round(theta[p['name']]))}\n")
        L(f"iter {k}/{N} result={result:+.3f} games={g} " + " ".join(f"{n}={v:.1f}" for n, v in theta.items()))
    L(f"spsa done, {total_games} games; final: " + " ".join(f"{n}={int(round(v))}" for n, v in theta.items()))

if __name__ == "__main__":
    main()
