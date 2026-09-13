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
        Some("nnuebench") => {
            // Micro-benchmark of the v3 evaluation components on random games.
            use engine::nnue::v3::{NetworkV3, StateV3, L1};
            use engine::nnue::Align64;
            let net = match args.get(2) { Some(p) => NetworkV3::load(p).expect("load"), None => NetworkV3::random(2026) };
            let mut rng = 0x1234_5678u64;
            let mut positions = Vec::new();
            let mut pos = Position::from_fen(START_FEN).unwrap();
            let mut moves_played = Vec::new();
            for _ in 0..4000 {
                let moves = legal_moves(&pos);
                if moves.is_empty() { pos = Position::from_fen(START_FEN).unwrap(); continue; }
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                let m = moves.moves[(rng % moves.len() as u64) as usize].mv;
                let next = pos.make_move(m);
                positions.push((pos, m, next));
                moves_played.push(m);
                pos = next;
                if positions.len() % 120 == 0 { pos = Position::from_fen(START_FEN).unwrap(); }
            }
            let t = Instant::now();
            let mut acc = Align64([0i16; L1]);
            for (p, _, _) in positions.iter() { net.refresh_accumulator(p, engine::types::Color::White, &mut acc.0); }
            let refresh_us = t.elapsed().as_secs_f64() * 1e6 / positions.len() as f64;
            let t = Instant::now();
            let mut sink = 0i64;
            for (p, _, _) in positions.iter() { sink += net.forward(&acc.0, &acc.0, engine::nnue::output_bucket(p)) as i64; }
            let forward_us = t.elapsed().as_secs_f64() * 1e6 / positions.len() as f64;
            let t = Instant::now();
            let mut st = StateV3::new();
            st.reset(&positions[0].0, &net);
            let mut n = 0;
            for (p, m, next) in positions.iter() {
                if p.key() != positions[0].0.key() && n % 120 == 0 { st.reset(p, &net); }
                st.push(p, *m, next);
                sink += st.evaluate(&net) as i64;
                n += 1;
                if n % 120 == 0 { st.reset(next, &net); }
            }
            let incr_us = t.elapsed().as_secs_f64() * 1e6 / positions.len() as f64;
            let rows_per_move = engine::nnue::v3::ROWS_APPLIED.load(std::sync::atomic::Ordering::Relaxed) as f64 / positions.len() as f64;
            println!("threat rows applied per move (both perspectives): {:.1}; prefetch lines {}", rows_per_move, engine::nnue::v3::PREFETCH_LINES);
            let ph: Vec<f64> = engine::nnue::v3::PHASE_NS.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / positions.len() as f64).collect();
            println!("phases us/move: attackers+relboards {:.2} | map_restricted x2 {:.2} | prefetch+psq rows {:.2} | sort+threat rows {:.2}", ph[0], ph[1], ph[2], ph[3]);
            // Component timings.
            use engine::nnue::threats::RelBoard;
            let t = Instant::now();
            let mut cnt = 0usize;
            let mut cntb = 0usize;
            for (p, _, _) in positions.iter() {
                let rel = RelBoard::from_position(p, engine::types::Color::White);
                net.mapper.map_features(&rel, |_| cnt += 1, |_| cntb += 1);
            }
            let mapfull_us = t.elapsed().as_secs_f64() * 1e6 / positions.len() as f64;
            let t = Instant::now();
            let mut cnt2 = 0usize;
            let mut cnt2b = 0usize;
            for (p, m, next) in positions.iter() {
                let changed = engine::types::bb(m.from()) | engine::types::bb(m.to());
                let mut att = changed;
                for s in engine::bitboard::bits(changed) {
                    att |= p.attackers_to_occ(s, p.occupied()) | next.attackers_to_occ(s, next.occupied());
                }
                let rel = RelBoard::from_position(p, engine::types::Color::White);
                net.mapper.map_restricted(&rel, att, changed, |_| cnt2 += 1, |_| cnt2b += 1);
            }
            let maprestr_us = t.elapsed().as_secs_f64() * 1e6 / positions.len() as f64;
            let t = Instant::now();
            let mut v = Align64([0i16; L1]);
            for k in 0..positions.len() * 20 {
                engine::nnue::v3::kernels::add_i8_row(&mut v.0, &net.pp_w[(k * 7919) % engine::nnue::v3::PP_FEATURES].0);
            }
            let row_us = t.elapsed().as_secs_f64() * 1e6 / (positions.len() * 20) as f64;
            let t = Instant::now();
            for k in 0..positions.len() * 20 {
                engine::nnue::v3::kernels::add_i8_row(&mut v.0, &net.pp_w[k % 8].0);
            }
            let row_hot_us = t.elapsed().as_secs_f64() * 1e6 / (positions.len() * 20) as f64;
            println!("i8 row add cache-hot {:.3} us vs random {:.3} us", row_hot_us, row_us);
            // Batched: 14 random rows per "move", with and without prefetching all lines first.
            for pf in [false, true] {
                let t = Instant::now();
                let mut idxs = [0usize; 14];
                let mut seed = 99u64;
                for _ in 0..positions.len() {
                    for k in 0..14 { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; idxs[k] = (seed % engine::nnue::v3::PP_FEATURES as u64) as usize; }
                    if pf {
                        for &f in &idxs { let p = net.pp_w[f].0.as_ptr() as *const i8; for l in 0..16 { unsafe { std::arch::x86_64::_mm_prefetch(p.add(l * 64), std::arch::x86_64::_MM_HINT_T0); } } }
                    }
                    for &f in &idxs { engine::nnue::v3::kernels::add_i8_row(&mut v.0, &net.pp_w[f].0); }
                }
                println!("14 random rows per move, prefetch={}: {:.2} us/move (sink {})", pf, t.elapsed().as_secs_f64() * 1e6 / positions.len() as f64, v.0[5]);
            }
            println!("refresh {:.2} us/pos | forward {:.2} us | incremental push+eval {:.2} us/move | map_features {:.2} us ({} feats/pos) | map_restricted(one board) {:.2} us ({} feats) | i8 row add {:.3} us (sink {} {})",
                refresh_us, forward_us, incr_us, mapfull_us, (cnt + cntb) / positions.len(), maprestr_us, (cnt2 + cnt2b) / positions.len(), row_us, sink % 7, v.0[3]);
        }
        Some("randomnet-v3") => {
            // Write a deterministic random v3 network (for speed tests and pipeline checks).
            let path = args.get(2).expect("usage: engine randomnet-v3 <out.bin> [512|1024|nt]");
            let width = args.get(3).map(|s| s.as_str()).unwrap_or("1024");
            let bytes = match width {
                "512" => engine::nnue::v3::w512::NetworkV3::random(2026).to_bytes(),
                "nt" => engine::nnue::v3::w1024nt::NetworkV3::random(2026).to_bytes(),
                _ => engine::nnue::v3::w1024::NetworkV3::random(2026).to_bytes(),
            };
            let n = bytes.len();
            std::fs::write(path, bytes).expect("write");
            println!("wrote {} ({} bytes, variant {})", path, n, width);
        }
        Some("netcheck") => {
            // Load a network file and evaluate a few positions; exit non-zero on failure.
            let path = args.get(2).expect("usage: engine netcheck <net.bin> [--strict]");
            let strict = args.iter().any(|a| a == "--strict");
            match engine::nnue::AnyNet::load(path) {
                Ok(net) => {
                    println!("architecture: {}", net.arch_name());
                    let mut ok = true;
                    for (fen, lo, hi) in [
                        (START_FEN, -150, 150),
                        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1", -2000, -100),
                        ("rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1", 100, 2000),
                        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", -3000, 3000),
                        ("r1bqkb1r/pp2bppp/2n2n2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 9", -3000, 3000),
                        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", -3000, 3000),
                    ] {
                        let pos = Position::from_fen(fen).unwrap();
                        let mut st = engine::nnue::AnyState::for_net(Some(&net));
                        st.reset(&pos, &net);
                        let v = st.evaluate(&pos, &net);
                        let vr = engine::nnue::AnyState::evaluate_reference(&pos, &net);
                        if v != vr {
                            println!("incremental {} != reference {} for {}", v, vr, fen);
                            ok = false;
                        }
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
