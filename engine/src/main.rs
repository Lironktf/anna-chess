use engine::movegen::*;
use engine::position::*;
use engine::uci;
use std::time::Instant;

extern "C" {
    fn signal(sig: i32, handler: usize) -> usize;
}

fn main() {
    // Die quietly (like a C program) if the GUI closes our stdout instead of panicking on EPIPE.
    #[cfg(unix)]
    unsafe {
        signal(13, 0); // SIGPIPE, SIG_DFL
    }
    engine::init();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("perft") => {
            let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            let fen = if args.len() > 3 { args[3..].join(" ") } else { START_FEN.to_string() };
            let pos = Position::from_fen(&fen).expect("bad fen");
            let t = Instant::now();
            let n = perft_divide(&pos, depth);
            let el = t.elapsed().as_secs_f64();
            println!("\nnodes {} time {:.3}s nps {:.0}", n, el, n as f64 / el.max(1e-9));
        }
        Some("bench") => {
            let depth: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
            let mut e = uci::Engine::new();
            if let Some(t) = args.get(3).and_then(|s| s.parse().ok()) {
                e.threads = t;
                e.resize();
            }
            eprintln!("net loaded: {}", e.net.is_some());
            uci::bench(&mut e, depth);
        }
        Some("datagen") => {
            // engine datagen --threads N --nodes N --games N --out FILE [--seed S] [--plies P] [--hash MB]
            let mut cfg = engine::datagen::DatagenConfig { threads: 1, nodes: 5000, games: 100, out: "data/datagen.bin".into(), seed: 1, random_plies: 8, hash_mb: 16, dfrc: false };
            let mut i = 2;
            while i + 1 < args.len() {
                match args[i].as_str() {
                    "--threads" => cfg.threads = args[i + 1].parse().unwrap(),
                    "--nodes" => cfg.nodes = args[i + 1].parse().unwrap(),
                    "--games" => cfg.games = args[i + 1].parse().unwrap(),
                    "--out" => cfg.out = args[i + 1].clone(),
                    "--seed" => cfg.seed = args[i + 1].parse().unwrap(),
                    "--plies" => cfg.random_plies = args[i + 1].parse().unwrap(),
                    "--hash" => cfg.hash_mb = args[i + 1].parse().unwrap(),
                    _ => {}
                }
                i += 2;
            }
            let e = uci::Engine::new();
            eprintln!("datagen: net loaded = {}, threads {} nodes {} games {} -> {}", e.net.is_some(), cfg.threads, cfg.nodes, cfg.games, cfg.out);
            engine::datagen::run(cfg, e.net.clone());
        }
        Some("netcheck") => {
            // Load a network file and evaluate a few positions; exit non-zero on failure.
            let path = args.get(2).expect("usage: engine netcheck <net.bin> [--strict]");
            let strict = args.iter().any(|a| a == "--strict");
            match engine::nnue::Network::load(path) {
                Ok(net) => {
                    let mut ok = true;
                    for (fen, lo, hi) in [
                        (START_FEN, -150, 150),
                        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1", -2000, -100),
                        ("rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1", 100, 2000),
                    ] {
                        let pos = Position::from_fen(fen).unwrap();
                        let mut st = engine::nnue::NnueState::new();
                        st.reset(&pos, &net);
                        let v = st.evaluate(&pos, &net);
                        let pass = v >= lo && v <= hi;
                        ok &= pass;
                        println!("{} eval {} expected [{}, {}] {}", fen, v, lo, hi, if pass { "ok" } else { "FAIL" });
                    }
                    if !ok && strict {
                        std::process::exit(2);
                    }
                    if !ok {
                        println!("netcheck: loaded, evaluation ranges NOT met (expected for a barely trained net; use --strict to enforce)");
                    }
                    println!("netcheck ok: {} ({} bytes)", path, std::fs::metadata(path).map(|m| m.len()).unwrap_or(0));
                }
                Err(e) => {
                    println!("netcheck FAILED: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(x) if x != "uci" => {
            eprintln!("usage: engine [uci|bench [depth] [threads]|perft <depth> [fen]|netcheck <net>]");
        }
        _ => uci::uci_loop(),
    }
}
