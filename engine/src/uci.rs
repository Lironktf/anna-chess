//! UCI protocol front end, bench and misc commands.

use crate::eval;
use crate::history::History;
use crate::movegen::*;
use crate::nnue::AnyNet;
use crate::position::*;
use crate::search::{self, Limits, Options, Shared};
use crate::timeman::{GoParams, TimeManager};
use crate::types::*;
use std::io::{self, BufRead, Write};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const NAME: &str = "Anna";
pub const VERSION: &str = "0.1";
pub const AUTHOR: &str = "Liron and Claude";
pub const DEFAULT_NET: &str = "nets/default.bin";

pub struct Engine {
    pub pos: Position,
    pub game_keys: Vec<u64>,
    pub shared: Arc<Shared>,
    pub net: Option<Arc<AnyNet>>,
    /// Optional fast second net for two-tier evaluation (UCI option EvalFileFast).
    pub net_fast: Option<Arc<AnyNet>>,
    /// Optional move-ordering policy net (UCI option PolicyFile).
    pub policy: Option<Arc<crate::policy::PolicyNet>>,
    pub hists: Vec<History>,
    pub hash_mb: usize,
    pub threads: usize,
    pub multi_pv: usize,
    pub move_overhead: i64,
    pub chess960: bool,
    pub eval_file: String,
    /// Score of the last completed search (for time management's falling-eval term).
    pub last_score: Value,
}

impl Engine {
    pub fn new() -> Self {
        let mut e = Engine {
            pos: Position::default(),
            game_keys: Vec::new(),
            shared: Shared::new(16, 1),
            net: None,
            hists: vec![History::new()],
            hash_mb: 16,
            threads: 1,
            multi_pv: 1,
            move_overhead: 20,
            chess960: false,
            eval_file: DEFAULT_NET.to_string(),
            net_fast: None,
            policy: None,
            last_score: VALUE_INFINITE,
        };
        e.load_net_quiet();
        e
    }

    fn load_net_quiet(&mut self) {
        if let Some(n) = AnyNet::embedded() {
            self.net = Some(Arc::new(n));
            return;
        }
        let candidates = [self.eval_file.clone(), exe_relative(&self.eval_file)];
        for c in candidates.iter() {
            if let Ok(n) = AnyNet::load(c) {
                self.net = Some(Arc::new(n));
                return;
            }
        }
        self.net = None;
    }

    pub fn load_net(&mut self) -> Result<(), String> {
        let candidates = [self.eval_file.clone(), exe_relative(&self.eval_file)];
        let mut last = String::new();
        for c in candidates.iter() {
            match AnyNet::load(c) {
                Ok(n) => {
                    self.net = Some(Arc::new(n));
                    return Ok(());
                }
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    pub fn set_position(&mut self, fen: &str, moves: &[&str]) -> Result<(), String> {
        let mut pos = Position::from_fen(fen)?;
        if self.chess960 {
            pos.set_chess960(true);
        }
        self.game_keys.clear();
        if moves.is_empty() {
            self.last_score = VALUE_INFINITE;
        }
        for ms in moves {
            let m = pos.parse_uci_move(ms).ok_or(format!("bad move {}", ms))?;
            if !pos.is_pseudo_legal(m) || !pos.is_legal(m) {
                return Err(format!("illegal move {}", ms));
            }
            self.game_keys.push(pos.key());
            pos = pos.make_move(m);
            if pos.rule50() == 0 {
                // Positions before an irreversible move can never repeat; keep them anyway
                // (cheap) since is_draw bounds its walk by rule50.
            }
        }
        self.pos = pos;
        Ok(())
    }

    pub fn new_game(&mut self) {
        let shared = Arc::get_mut(&mut self.shared).expect("search running");
        shared.tt.clear_threaded(self.threads);
        for h in self.hists.iter_mut() {
            h.clear();
        }
        self.last_score = VALUE_INFINITE;
    }

    pub fn resize(&mut self) {
        self.shared = Shared::new(self.hash_mb, self.threads);
    }

    pub fn options(&self) -> Options {
        Options { threads: self.threads, multi_pv: self.multi_pv, move_overhead: self.move_overhead, chess960: self.chess960, silent: false, prev_score: self.last_score }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

fn exe_relative(p: &str) -> String {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            // Look in the exe dir and two levels up (target/release/../../nets).
            for base in [dir.to_path_buf(), dir.join("..").join(".."), dir.join("..").join("..").join("..")] {
                let cand = base.join(p);
                if cand.exists() {
                    return cand.to_string_lossy().to_string();
                }
            }
        }
    }
    p.to_string()
}

pub fn parse_go(tokens: &[&str]) -> GoParams {
    let mut p = GoParams::default();
    let mut i = 0;
    let num = |s: Option<&&str>| -> Option<i64> { s.and_then(|x| x.parse::<i64>().ok()) };
    while i < tokens.len() {
        match tokens[i] {
            "wtime" => p.wtime = num(tokens.get(i + 1)),
            "btime" => p.btime = num(tokens.get(i + 1)),
            "winc" => p.winc = num(tokens.get(i + 1)),
            "binc" => p.binc = num(tokens.get(i + 1)),
            "movestogo" => p.movestogo = num(tokens.get(i + 1)),
            "movetime" => p.movetime = num(tokens.get(i + 1)),
            "depth" => p.depth = num(tokens.get(i + 1)).map(|d| d as i32),
            "nodes" => p.nodes = num(tokens.get(i + 1)).map(|n| n as u64),
            "mate" => p.mate = num(tokens.get(i + 1)).map(|d| d as i32),
            "infinite" => {
                p.infinite = true;
                i += 1;
                continue;
            }
            "ponder" => {
                p.ponder = true;
                i += 1;
                continue;
            }
            "searchmoves" => {
                i += 1;
                while i < tokens.len() && tokens[i].len() >= 4 && tokens[i].as_bytes()[0].is_ascii_lowercase() && tokens[i].as_bytes()[1].is_ascii_digit() {
                    p.searchmoves.push(tokens[i].to_string());
                    i += 1;
                }
                continue;
            }
            _ => {
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    p
}

/// Run a search on the engine state in a background thread; returns after the search finished.
fn run_search(engine: &Arc<Mutex<Engine>>, go: GoParams) {
    let (pos, keys, shared, net, net_fast, policy, opts, limits) = {
        let e = engine.lock().unwrap();
        let tm = TimeManager::new(&go, e.pos.side_to_move() == Color::White, e.pos.game_ply(), e.move_overhead);
        let limits = Limits { go: go.clone(), tm, max_depth: go.depth.unwrap_or(0), max_nodes: go.nodes.unwrap_or(0) };
        (e.pos, e.game_keys.clone(), e.shared.clone(), e.net.clone(), e.net_fast.clone(), e.policy.clone(), e.options(), limits)
    };
    let mut hists = {
        let mut e = engine.lock().unwrap();
        std::mem::take(&mut e.hists)
    };
    let res = search::go(&pos, &keys, &shared, net.as_deref(), net_fast.as_deref(), policy.as_deref(), &limits, &opts, &mut hists);
    {
        let mut e = engine.lock().unwrap();
        e.hists = hists;
        e.last_score = res.score;
    }
    let best = pos.move_to_uci(res.best_move);
    if res.ponder_move.is_none() {
        println!("bestmove {}", best);
    } else {
        let child = pos.make_move(res.best_move);
        println!("bestmove {} ponder {}", best, child.move_to_uci(res.ponder_move));
    }
    let _ = io::stdout().flush();
}

pub fn uci_loop() {
    let engine = Arc::new(Mutex::new(Engine::new()));
    let stdin = io::stdin();
    let mut search_handle: Option<std::thread::JoinHandle<()>> = None;

    let wait_search = |h: &mut Option<std::thread::JoinHandle<()>>| {
        if let Some(j) = h.take() {
            let _ = j.join();
        }
    };

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        match tokens[0] {
            "uci" => {
                println!("id name {} {}", NAME, VERSION);
                println!("id author {}", AUTHOR);
                println!("option name Hash type spin default 16 min 1 max 33554432");
                println!("option name Threads type spin default 1 min 1 max 1024");
                println!("option name MultiPV type spin default 1 min 1 max 256");
                println!("option name Move Overhead type spin default 20 min 0 max 5000");
                println!("option name UCI_Chess960 type check default false");
                println!("option name EvalFile type string default {}", DEFAULT_NET);
                println!("option name EvalFileFast type string default <empty>");
                println!("option name PolicyFile type string default <empty>");
                println!("option name SyzygyPath type string default <empty>");
                crate::params::print_uci_options();
                println!("uciok");
            }
            "isready" => {
                wait_search(&mut search_handle);
                println!("readyok");
            }
            "setoption" => {
                wait_search(&mut search_handle);
                let mut name = String::new();
                let mut value = String::new();
                let mut mode = 0;
                for t in &tokens[1..] {
                    match *t {
                        "name" => mode = 1,
                        "value" => mode = 2,
                        _ => {
                            if mode == 1 {
                                if !name.is_empty() {
                                    name.push(' ');
                                }
                                name.push_str(t);
                            } else if mode == 2 {
                                if !value.is_empty() {
                                    value.push(' ');
                                }
                                value.push_str(t);
                            }
                        }
                    }
                }
                let mut e = engine.lock().unwrap();
                match name.to_lowercase().as_str() {
                    "hash" => {
                        e.hash_mb = value.parse().unwrap_or(16);
                        e.resize();
                    }
                    "threads" => {
                        e.threads = value.parse::<usize>().unwrap_or(1).max(1);
                        e.resize();
                    }
                    "multipv" => e.multi_pv = value.parse::<usize>().unwrap_or(1).max(1),
                    "move overhead" => e.move_overhead = value.parse().unwrap_or(20),
                    "uci_chess960" => {
                        e.chess960 = value == "true";
                        let fen = e.pos.to_fen();
                        let _ = e.set_position(&fen, &[]);
                    }
                    "syzygypath" => {
                        let n = crate::tb::init(&value);
                        println!("info string syzygy: {} ({} men)", if n > 0 { "loaded" } else { "not loaded" }, n);
                    }
                    "policyfile" => {
                        if value.is_empty() || value == "<empty>" {
                            e.policy = None;
                            println!("info string policy net cleared");
                        } else {
                            match crate::policy::PolicyNet::load(&value) {
                                Ok(n) => {
                                    println!("info string loaded policy net {}", value);
                                    e.policy = Some(Arc::new(n));
                                }
                                Err(err) => println!("info string failed to load policy net: {}", err),
                            }
                        }
                    }
                    "evalfilefast" => {
                        if value.is_empty() || value == "<empty>" {
                            e.net_fast = None;
                            println!("info string fast network cleared");
                        } else {
                            match AnyNet::load(&value) {
                                Ok(n) => {
                                    println!("info string loaded fast network {} [{}]", value, n.arch_name());
                                    e.net_fast = Some(Arc::new(n));
                                }
                                Err(err) => println!("info string failed to load fast network: {}", err),
                            }
                        }
                    }
                    "evalfile" => {
                        e.eval_file = value.clone();
                        match e.load_net() {
                            Ok(()) => println!("info string loaded network {} [{}]", value, e.net.as_deref().map(|n| n.arch_name()).unwrap_or("?")),
                            Err(err) => println!("info string failed to load network: {}", err),
                        }
                    }
                    _ => {
                        if !crate::params::set_by_name(&name, &value) {
                            println!("info string unknown option {}", name);
                        }
                    }
                }
            }
            "ucinewgame" => {
                wait_search(&mut search_handle);
                engine.lock().unwrap().new_game();
            }
            "position" => {
                wait_search(&mut search_handle);
                let mut i;
                let fen;
                if tokens.get(1) == Some(&"startpos") {
                    fen = START_FEN.to_string();
                    i = 2;
                } else if tokens.get(1) == Some(&"fen") {
                    let mut parts = Vec::new();
                    i = 2;
                    while i < tokens.len() && tokens[i] != "moves" {
                        parts.push(tokens[i]);
                        i += 1;
                    }
                    fen = parts.join(" ");
                } else {
                    continue;
                }
                let moves: Vec<&str> = if tokens.get(i) == Some(&"moves") { tokens[i + 1..].to_vec() } else { Vec::new() };
                if let Err(err) = engine.lock().unwrap().set_position(&fen, &moves) {
                    println!("info string error: {}", err);
                }
            }
            "go" => {
                wait_search(&mut search_handle);
                let go = parse_go(&tokens[1..]);
                let eng = engine.clone();
                search_handle = Some(std::thread::spawn(move || run_search(&eng, go)));
            }
            "stop" => {
                engine.lock().unwrap().shared.stop.store(true, Ordering::Relaxed);
                wait_search(&mut search_handle);
            }
            "ponderhit" => {}
            "quit" => {
                engine.lock().unwrap().shared.stop.store(true, Ordering::Relaxed);
                wait_search(&mut search_handle);
                break;
            }
            "d" => {
                let e = engine.lock().unwrap();
                print!("{}", e.pos.pretty());
                println!("checkers: {:016x}", e.pos.checkers());
            }
            "eval" => {
                let e = engine.lock().unwrap();
                let mut st = crate::nnue::AnyState::for_net(e.net.as_deref());
                if let Some(n) = e.net.as_deref() {
                    st.reset(&e.pos, n);
                }
                let v = eval::evaluate(&e.pos, e.net.as_deref(), &mut st);
                println!("eval {} cp (stm) net={}", v, e.net.as_deref().map(|n| n.arch_name()).unwrap_or("none"));
            }
            "bench" => {
                wait_search(&mut search_handle);
                let depth = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
                let mut e = engine.lock().unwrap();
                bench(&mut e, depth);
            }
            "perft" => {
                let depth = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
                let e = engine.lock().unwrap();
                let t = Instant::now();
                let n = perft_divide(&e.pos, depth);
                let el = t.elapsed().as_secs_f64();
                println!("nodes {} time {:.3}s nps {:.0}", n, el, n as f64 / el.max(1e-9));
            }
            _ => println!("info string unknown command {}", tokens[0]),
        }
        let _ = io::stdout().flush();
    }
}

pub const BENCH_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 9",
    "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9",
    "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 12",
    "r1bqkb1r/pp3ppp/2n1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6",
    "8/8/1p1k4/p1p2p2/P1P1pP1p/1P2P2P/3K4/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 4 6",
    "3r2k1/pp3pp1/2p1b2p/8/3P4/2N1P1P1/PP3P1P/3R2K1 w - - 0 20",
    "2r3k1/1p3ppp/p2b4/3p4/3P4/1P1B1P2/P4P1P/2R3K1 b - - 0 25",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
];

pub fn bench(e: &mut Engine, depth: i32) {
    let saved_pos = e.pos;
    let mut total_nodes = 0u64;
    let start = Instant::now();
    for fen in BENCH_FENS {
        e.new_game();
        e.set_position(fen, &[]).unwrap();
        let go = GoParams { depth: Some(depth), ..Default::default() };
        let tm = TimeManager::new(&go, e.pos.side_to_move() == Color::White, e.pos.game_ply(), e.move_overhead);
        let limits = Limits { go: go.clone(), tm, max_depth: depth, max_nodes: 0 };
        let mut opts = e.options();
        opts.silent = true;
        let mut hists = std::mem::take(&mut e.hists);
        let res = search::go(&e.pos, &e.game_keys, &e.shared, e.net.as_deref(), e.net_fast.as_deref(), e.policy.as_deref(), &limits, &opts, &mut hists);
        e.hists = hists;
        total_nodes += res.nodes;
        eprintln!("{:<75} depth {:>2} nodes {:>10} bestmove {}", fen, res.depth, res.nodes, e.pos.move_to_uci(res.best_move));
    }
    let el = start.elapsed().as_secs_f64();
    println!("Bench: {} nodes {} nps", total_nodes, (total_nodes as f64 / el.max(1e-9)) as u64);
    e.pos = saved_pos;
}
