//! `play` binary: serve the browser UI + WebSocket API for playing against Anna.
//!
//! cargo run --release -p play -- [--bind 0.0.0.0] [--port 8080] [--engine target/release/engine]
//!     [--net nets/default.bin] [--syzygy syzygy] [--hash 64] [--threads 1]
//!     [--max-movetime 10000] [--max-depth 30] [--max-games 4] [--name Anna] [--db play/games.sqlite | --no-db]

use std::net::SocketAddr;

use play::Config;

fn usage() -> ! {
    eprintln!("{}", "usage: play [--bind ADDR] [--port N] [--engine PATH] [--net PATH] [--syzygy DIR] [--hash MB] \
        [--threads N] [--max-movetime MS] [--max-depth N] [--max-games N] [--name NAME] [--db FILE | --no-db]");
    std::process::exit(2)
}

#[tokio::main]
async fn main() {
    let mut cfg = Config::default();
    let mut bind = "127.0.0.1".to_string();
    let mut port: u16 = 8080;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let next = |i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| usage())
    };
    while i < args.len() {
        match args[i].as_str() {
            "--bind" => bind = next(&mut i),
            "--port" => port = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--engine" => cfg.engine_bin = next(&mut i).into(),
            "--net" => cfg.eval_file = Some(next(&mut i)),
            "--syzygy" => cfg.syzygy_path = Some(next(&mut i)),
            "--hash" => cfg.hash_mb = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--threads" => cfg.threads = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--max-movetime" => cfg.max_movetime_ms = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--max-depth" => cfg.max_depth = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--max-games" => cfg.max_games = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--name" => cfg.engine_name = next(&mut i),
            "--db" => cfg.db_path = Some(next(&mut i).into()),
            "--no-db" => cfg.db_path = None,
            "-h" | "--help" => usage(),
            _ => usage(),
        }
        i += 1;
    }
    if !cfg.engine_bin.is_file() {
        eprintln!("engine binary not found: {} (build with `cargo build --release`)", cfg.engine_bin.display());
        std::process::exit(1);
    }
    if let Some(n) = &cfg.eval_file {
        if !std::path::Path::new(n).is_file() {
            eprintln!("net file not found: {n}");
            std::process::exit(1);
        }
    }
    let addr: SocketAddr = format!("{bind}:{port}").parse().unwrap_or_else(|_| usage());
    if let Err(e) = play::serve(addr, cfg).await {
        eprintln!("play: {e}");
        std::process::exit(1);
    }
}
