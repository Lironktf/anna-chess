//! Anna in the browser. One engine instance per worker; the page talks to it through these calls
//! instead of UCI over a socket. Search is synchronous, so run it in a Web Worker.
use engine::search::{self, Limits};
use engine::timeman::{GoParams, TimeManager};
use engine::types::Color;
use engine::uci::Engine;
use wasm_bindgen::prelude::*;

/// Time source for the engine's clock (see engine/src/timeman.rs).
#[no_mangle]
pub extern "C" fn anna_host_now_ms() -> f64 {
    js_sys::Date::now()
}

#[wasm_bindgen]
pub struct Anna {
    e: Engine,
}

#[wasm_bindgen]
impl Anna {
    #[wasm_bindgen(constructor)]
    pub fn new(hash_mb: usize) -> Anna {
        engine::init();
        let mut e = Engine::new();
        e.threads = 1;
        e.hash_mb = hash_mb.clamp(1, 256);
        e.shared = search::Shared::new(e.hash_mb, 1);
        Anna { e }
    }

    pub fn version(&self) -> String {
        format!("Anna {}", engine::uci::VERSION)
    }

    pub fn has_net(&self) -> bool {
        self.e.net.is_some()
    }

    /// Set the game: a FEN (or "startpos") plus UCI moves separated by spaces.
    pub fn set_position(&mut self, fen: &str, moves: &str) -> bool {
        let start = if fen == "startpos" { engine::position::START_FEN } else { fen };
        let Ok(mut pos) = engine::position::Position::from_fen(start) else { return false };
        let mut keys = Vec::new();
        for m in moves.split_whitespace() {
            let Some(mv) = pos.parse_uci_move(m) else { return false };
            keys.push(pos.key());
            pos = pos.make_move(mv);
        }
        self.e.pos = pos;
        self.e.game_keys = keys;
        true
    }

    /// Legal moves of the current position, UCI, space separated.
    pub fn legal_moves(&self) -> String {
        let pos = &self.e.pos;
        engine::movegen::legal_moves(pos).iter().map(|m| pos.move_to_uci(m)).collect::<Vec<_>>().join(" ")
    }

    pub fn fen(&self) -> String {
        self.e.pos.to_fen()
    }

    /// Search for `movetime_ms` (0 = no clock limit) and/or to `depth` (0 = no depth limit).
    /// Returns JSON: {"bestmove","ponder","cp" or "mate","depth","nodes","pv"}.
    pub fn go(&mut self, movetime_ms: u32, depth: u32) -> String {
        let go = GoParams {
            movetime: if movetime_ms > 0 { Some(movetime_ms as i64) } else { None },
            depth: if depth > 0 { Some(depth as i32) } else { None },
            ..Default::default()
        };
        let e = &mut self.e;
        let tm = TimeManager::new(&go, e.pos.side_to_move() == Color::White, e.pos.game_ply(), e.move_overhead);
        let limits = Limits { go: go.clone(), tm, max_depth: go.depth.unwrap_or(0), max_nodes: 0 };
        let mut opts = e.options();
        opts.threads = 1;
        opts.silent = true;
        let mut hists = std::mem::take(&mut e.hists);
        let res = search::go(&e.pos, &e.game_keys, &e.shared, e.net.as_deref(), None, None, &limits, &opts, &mut hists);
        e.hists = hists;
        e.last_score = res.score;
        let pos = &e.pos;
        let best = pos.move_to_uci(res.best_move);
        let ponder = if res.ponder_move.is_none() { String::new() } else { pos.make_move(res.best_move).move_to_uci(res.ponder_move) };
        let pv: Vec<String> = res.pv.iter().map(|m| pos.move_to_uci(*m)).collect();
        let score = if res.score.abs() >= engine::types::VALUE_MATE_IN_MAX_PLY {
            let plies = if res.score > 0 { engine::types::VALUE_MATE - res.score } else { -engine::types::VALUE_MATE - res.score };
            format!("\"mate\":{}", if plies > 0 { (plies + 1) / 2 } else { (plies - 1) / 2 })
        } else {
            format!("\"cp\":{}", res.score)
        };
        format!("{{\"bestmove\":\"{}\",\"ponder\":\"{}\",{},\"depth\":{},\"nodes\":{},\"pv\":\"{}\"}}", best, ponder, score, res.depth, res.nodes, pv.join(" "))
    }

    pub fn new_game(&mut self) {
        self.e.hists.clear();
        self.e.shared = search::Shared::new(self.e.hash_mb, 1);
    }
}
