//! In-process sampling profile of the engine's bench: SIGPROF-based (pprof), so it needs neither
//! `perf` nor ptrace. Prints the hottest functions (self and inclusive) and writes a flamegraph.
//!
//!   cargo run --release -p prof -- [depth] [net-file] [out.svg]
//!
//! Symbols come from the `release` profile only if it has debuginfo; the workspace's
//! `release-debug` profile has it: `cargo run --profile release-debug -p prof`.

use std::collections::HashMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let depth: i32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(13);
    let net = args.get(2).cloned();
    let out = args.get(3).cloned().unwrap_or_else(|| "prof/flamegraph.svg".to_string());

    engine::init();
    let mut e = engine::uci::Engine::new();
    if let Some(n) = &net {
        e.eval_file = n.clone();
        e.load_net().expect("load net");
    }
    eprintln!("prof: depth {depth}, net {}", net.as_deref().unwrap_or("embedded default"));

    let guard = pprof::ProfilerGuardBuilder::default().frequency(997).blocklist(&["libc", "libgcc", "pthread", "vdso"]).build().unwrap();
    let t = std::time::Instant::now();
    engine::uci::bench(&mut e, depth);
    let secs = t.elapsed().as_secs_f64();
    {
        use std::sync::atomic::Ordering::Relaxed;
        let (r, i, e) = (engine::nnue::v3::w1024::REFRESHES.load(Relaxed) + engine::nnue::v3::w512::REFRESHES.load(Relaxed),
            engine::nnue::v3::w1024::INCR_APPLIES.load(Relaxed) + engine::nnue::v3::w512::INCR_APPLIES.load(Relaxed),
            engine::nnue::v3::w1024::EVALS.load(Relaxed) + engine::nnue::v3::w512::EVALS.load(Relaxed));
        let tr = engine::nnue::v3::w1024::THR_REFRESHES.load(Relaxed) + engine::nnue::v3::w512::THR_REFRESHES.load(Relaxed);
        if e > 0 {
            println!("v3 stats: evals {e}, incremental applies {i} ({:.2}/eval), psq refreshes {r} ({:.3}/eval), threat rebuilds {tr} ({:.3}/eval, one per {:.0} evals)", i as f64 / e as f64, r as f64 / e as f64, tr as f64 / e as f64, e as f64 / tr.max(1) as f64);
        }
        for (name, hits) in [("w1024", engine::nnue::v3::w1024::FEATURE_HITS.get()), ("w512", engine::nnue::v3::w512::FEATURE_HITS.get())] {
            if let Some(h) = hits {
                let mut v: Vec<u64> = h.iter().map(|a| a.load(Relaxed) as u64).collect();
                let total: u64 = v.iter().sum();
                if total == 0 { continue; }
                v.sort_unstable_by(|a, b| b.cmp(a));
                let used = v.iter().filter(|&&c| c > 0).count();
                let mut line = format!("feature usage ({name}): {total} applied rows over {used} distinct features;");
                let mut acc = 0u64;
                let mut next = [512usize, 1024, 2048, 4096, 8192, 16384, 32768].into_iter().peekable();
                for (i, c) in v.iter().enumerate() {
                    acc += c;
                    while let Some(&k) = next.peek() {
                        if i + 1 == k { line += &format!(" top{k}={:.1}%", 100.0 * acc as f64 / total as f64); next.next(); } else { break; }
                    }
                }
                println!("{line}");
            }
        }
    }
    let report = guard.report().build().unwrap();

    // Aggregate: self time = leaf frame; inclusive = any frame in the stack (deduplicated per sample).
    let mut self_t: HashMap<String, usize> = HashMap::new();
    let mut incl_t: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;
    // leaf -> (caller chain string -> count), for the "who calls the hot leaf" table
    let mut callers: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for (frames, &cnt) in report.data.iter() {
        let count = cnt.max(0) as usize;
        total += count;
        // Expand each physical frame into its full inlined chain (innermost first), so callers of an
        // inlined leaf (e.g. a memcpy inside an inlined update function) are attributed correctly.
        let names: Vec<String> = frames
            .frames
            .iter()
            .flat_map(|f| {
                let v: Vec<String> = f.iter().map(|s| short(&format!("{}:{}", s.name(), s.lineno.unwrap_or(0)))).collect();
                if v.is_empty() { vec!["?".to_string()] } else { v }
            })
            .collect();
        if let Some(leaf) = names.first() {
            *self_t.entry(leaf.clone()).or_default() += count;
            let chain: Vec<&str> = names.iter().skip(1).take(4).map(|s| s.as_str()).collect();
            *callers.entry(leaf.clone()).or_default().entry(chain.join(" <- ")).or_default() += count;
        }
        let mut seen = std::collections::HashSet::new();
        for n in names {
            if seen.insert(n.clone()) {
                *incl_t.entry(n).or_default() += count;
            }
        }
    }
    let mut sv: Vec<_> = self_t.into_iter().collect();
    sv.sort_by(|a, b| b.1.cmp(&a.1));
    let mut iv: Vec<_> = incl_t.into_iter().collect();
    iv.sort_by(|a, b| b.1.cmp(&a.1));
    println!("samples: {total} over {secs:.2} s");
    println!("\n== self time (leaf function) ==");
    for (n, c) in sv.iter().take(30) {
        println!("{:5.1}%  {}", 100.0 * *c as f64 / total as f64, n);
    }
    println!("\n== callers of the hottest leaves ==");
    for (n, c) in sv.iter().take(6) {
        println!("{n} ({:.1}%):", 100.0 * *c as f64 / total as f64);
        let mut cv: Vec<_> = callers.get(n).map(|m| m.iter().collect()).unwrap_or_default();
        cv.sort_by(|a, b| b.1.cmp(a.1));
        for (chain, cc) in cv.iter().take(4) {
            println!("    {:5.1}%  {}", 100.0 * **cc as f64 / total as f64, chain);
        }
    }
    println!("\n== inclusive (function anywhere on the stack) ==");
    for (n, c) in iv.iter().take(30) {
        println!("{:5.1}%  {}", 100.0 * *c as f64 / total as f64, n);
    }
    let file = std::fs::File::create(&out).expect("create svg");
    report.flamegraph(file).expect("flamegraph");
    eprintln!("flamegraph written to {out}");
}

/// Strip generic noise and crate paths for readability.
fn short(n: &str) -> String {
    let n = n.split("::h").next().unwrap_or(n);
    n.replace("engine::", "").replace("core::", "").replace("alloc::", "")
}
