//! Self-play data generation in bulletformat (32-byte `ChessBoard` records).
//!
//! Record layout (bulletformat 1.8.0, `ChessBoard::from_raw`): everything is from the side to
//! move's perspective: the board is flipped vertically when black is to move, "our" pieces get
//! nibble `pt`, theirs `8 | pt`; score is stm-relative; result is 0/1/2 for stm loss/draw/win.

use crate::bitboard::*;
use crate::history::History;
use crate::movegen::legal_moves;
use crate::nnue::AnyNet;
use crate::position::{Position, START_FEN};
use crate::search::{self, Limits, Options, Shared};
use crate::timeman::{GoParams, TimeManager};
use crate::types::*;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Encode a position as a bulletformat `ChessBoard` (32 bytes). `score` is white-relative
/// centipawns and `result` is white-relative (1.0 white win, 0.5 draw, 0.0 black win), matching
/// `ChessBoard::from_raw`'s contract; both are converted to stm-relative here.
pub fn to_bulletformat(pos: &Position, score: i16, result: f32) -> [u8; 32] {
    let stm = pos.side_to_move();
    let mut bbs = [0u64; 8];
    bbs[0] = pos.colored(Color::White);
    bbs[1] = pos.colored(Color::Black);
    for (i, pt) in PieceType::ALL.iter().enumerate() {
        bbs[2 + i] = pos.pieces(*pt);
    }
    let mut score = score;
    let mut result = result;
    if stm == Color::Black {
        for b in bbs.iter_mut() {
            *b = b.swap_bytes();
        }
        bbs.swap(0, 1);
        score = -score;
        result = 1.0 - result;
    }
    let occ = bbs[0] | bbs[1];
    let mut pcs = [0u8; 16];
    let mut idx = 0;
    for s in bits(occ) {
        let bit = bb(s);
        let colour = ((bit & bbs[1]) != 0) as u8;
        let mut piece = 0u8;
        for (i, b) in bbs[2..].iter().enumerate() {
            if bit & b != 0 {
                piece = i as u8;
                break;
            }
        }
        let pc = (colour << 3) | piece;
        pcs[idx / 2] |= pc << (4 * (idx & 1));
        idx += 1;
    }
    let ksq = lsb(bbs[0] & bbs[7]);
    let opp_ksq = lsb(bbs[1] & bbs[7]) ^ 56;
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&occ.to_le_bytes());
    out[8..24].copy_from_slice(&pcs);
    out[24..26].copy_from_slice(&score.to_le_bytes());
    out[26] = (2.0 * result) as u8;
    out[27] = ksq;
    out[28] = opp_ksq;
    out
}

#[derive(Clone)]
pub struct DatagenConfig {
    pub threads: usize,
    pub nodes: u64,
    pub games: u64,
    pub out: String,
    pub seed: u64,
    pub random_plies: u32,
    pub hash_mb: usize,
    pub dfrc: bool,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Positions recorded during a game before the result is known.
struct Pending {
    pos: Position,
    score_white: i16,
}

fn play_game(rng: &mut Rng, cfg: &DatagenConfig, shared: &Shared, net: Option<&AnyNet>, hists: &mut Vec<History>) -> (Vec<Pending>, f32) {
    let mut pos = Position::from_fen(START_FEN).unwrap();
    let mut keys: Vec<u64> = Vec::new();
    // Random opening.
    let plies = cfg.random_plies + (rng.next() % 2) as u32;
    for _ in 0..plies {
        let moves = legal_moves(&pos);
        if moves.is_empty() {
            return (Vec::new(), 0.5);
        }
        let m = moves.moves[(rng.next() % moves.len() as u64) as usize].mv;
        keys.push(pos.key());
        pos = pos.make_move(m);
    }
    // Skip openings that are already decided.
    let mut pending = Vec::new();
    let opts = Options { threads: 1, multi_pv: 1, move_overhead: 0, chess960: false, silent: true, prev_score: VALUE_INFINITE };
    let mut win_plies = 0;
    let mut draw_plies = 0;
    let mut ply = plies;
    loop {
        let moves = legal_moves(&pos);
        if moves.is_empty() {
            let res = if pos.in_check() {
                if pos.side_to_move() == Color::White { 0.0 } else { 1.0 }
            } else {
                0.5
            };
            return (pending, res);
        }
        if pos.rule50() >= 100 || pos.is_insufficient_material() {
            return (pending, 0.5);
        }
        // Threefold in game history.
        let reps = keys.iter().rev().take(pos.rule50() as usize).step_by(2).filter(|k| **k == pos.key()).count();
        if reps >= 2 {
            return (pending, 0.5);
        }
        let go = GoParams { nodes: Some(cfg.nodes), ..Default::default() };
        let tm = TimeManager::new(&go, pos.side_to_move() == Color::White, pos.game_ply(), 0);
        let limits = Limits { go, tm, max_depth: 0, max_nodes: cfg.nodes };
        let res = search::go(&pos, &keys, shared, net, None, &limits, &opts, hists);
        let m = res.best_move;
        if m.is_none() {
            return (pending, 0.5);
        }
        let score_stm = res.score;
        let score_white = if pos.side_to_move() == Color::White { score_stm } else { -score_stm };
        // Adjudication.
        if score_stm.abs() >= 2500 {
            win_plies += 1;
        } else {
            win_plies = 0;
        }
        if ply > 80 && score_stm.abs() <= 8 {
            draw_plies += 1;
        } else {
            draw_plies = 0;
        }
        let decisive = is_decisive(score_stm);
        // Record quiet, non-check positions with bounded scores.
        if !pos.in_check() && !pos.is_capture_or_promo(m) && score_stm.abs() < 10000 && !decisive {
            pending.push(Pending { pos, score_white: score_white.clamp(-32000, 32000) as i16 });
        }
        if win_plies >= 4 {
            return (pending, if score_white > 0 { 1.0 } else { 0.0 });
        }
        if draw_plies >= 8 {
            return (pending, 0.5);
        }
        if decisive && (VALUE_MATE - score_stm.abs()) <= 2 {
            // Play it out to the mate quickly; adjudicate as decided.
            return (pending, if score_white > 0 { 1.0 } else { 0.0 });
        }
        keys.push(pos.key());
        pos = pos.make_move(m);
        ply += 1;
        if ply > 600 {
            return (pending, 0.5);
        }
    }
}

pub fn run(cfg: DatagenConfig, net: Option<Arc<AnyNet>>) {
    let start = Instant::now();
    let positions = Arc::new(AtomicU64::new(0));
    let games = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let out = Arc::new(std::sync::Mutex::new(std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(&cfg.out).expect("open output"))));
    std::thread::scope(|s| {
        for t in 0..cfg.threads {
            let cfg = cfg.clone();
            let net = net.clone();
            let positions = positions.clone();
            let games = games.clone();
            let stop = stop.clone();
            let out = out.clone();
            s.spawn(move || {
                let mut rng = Rng(cfg.seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(t as u64 + 1)) | 1);
                let mut shared = Shared::new(cfg.hash_mb, 1);
                let mut hists = vec![History::new()];
                let mut buf: Vec<u8> = Vec::with_capacity(32 * 4096);
                while !stop.load(Ordering::Relaxed) {
                    let g = games.fetch_add(1, Ordering::Relaxed);
                    if g >= cfg.games {
                        break;
                    }
                    // Fresh TT per game keeps games independent and deterministic per seed.
                    Arc::get_mut(&mut shared).unwrap().tt.clear_threaded(1);
                    for h in hists.iter_mut() {
                        h.clear();
                    }
                    let (pending, result) = play_game(&mut rng, &cfg, &*shared, net.as_deref(), &mut hists);
                    for p in &pending {
                        buf.extend_from_slice(&to_bulletformat(&p.pos, p.score_white, result));
                    }
                    positions.fetch_add(pending.len() as u64, Ordering::Relaxed);
                    if buf.len() >= 32 * 2048 {
                        out.lock().unwrap().write_all(&buf).unwrap();
                        buf.clear();
                    }
                    if t == 0 && g % 50 == 0 {
                        let el = start.elapsed().as_secs_f64();
                        let p = positions.load(Ordering::Relaxed);
                        eprintln!("games {} positions {} ({:.0}/s) elapsed {:.0}s", g, p, p as f64 / el.max(1e-9), el);
                    }
                }
                if !buf.is_empty() {
                    out.lock().unwrap().write_all(&buf).unwrap();
                }
            });
        }
    });
    out.lock().unwrap().flush().unwrap();
    let el = start.elapsed().as_secs_f64();
    let p = positions.load(Ordering::Relaxed);
    eprintln!("done: {} games, {} positions in {:.0}s ({:.0} pos/s) -> {}", cfg.games, p, el, p as f64 / el.max(1e-9), cfg.out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bulletformat_layout_matches_reference() {
        crate::init();
        // Start position, white to move: occ, stm-relative pieces, kings.
        let pos = Position::from_fen(START_FEN).unwrap();
        let rec = to_bulletformat(&pos, 17, 1.0);
        assert_eq!(u64::from_le_bytes(rec[0..8].try_into().unwrap()), pos.occupied());
        assert_eq!(rec[27], squares::E1);
        assert_eq!(rec[28], squares::E8 ^ 56);
        assert_eq!(i16::from_le_bytes([rec[24], rec[25]]), 17);
        assert_eq!(rec[26], 2);
        // First piece (a1 rook): our rook = 3; second (b1 knight) = 1 -> nibble packing 0x13.
        assert_eq!(rec[8], 0x13);
        // Black to move: board flipped, score and result negated.
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let rec = to_bulletformat(&pos, 30, 1.0);
        assert_eq!(i16::from_le_bytes([rec[24], rec[25]]), -30);
        assert_eq!(rec[26], 0);
        assert_eq!(rec[27], squares::E1); // black king e8 flipped -> e1
        assert_eq!(rec[28], squares::E1); // white king e1 flipped -> e8, then ^56 -> e1
        // First occupied square after flip is a1 = black's a8 rook -> "our" rook = 3; b1 = knight 1.
        assert_eq!(rec[8], 0x13);
    }
}
