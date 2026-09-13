//! Alpha-beta search (principal variation search) with Stockfish-style pruning, extensions,
//! histories and correction histories, plus Lazy SMP threading.

use crate::eval;
use crate::history::*;
use crate::movegen::*;
use crate::movepick::MovePicker;
use crate::nnue::{AnyNet, AnyState};
use crate::position::Position;
use crate::timeman::{GoParams, TimeManager};
use crate::tt::*;
use crate::types::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Root,
    Pv,
    NonPv,
}

#[derive(Clone, Copy)]
pub struct StackEntry {
    pub current_move: Move,
    pub moved_piece: Piece,
    pub cont_idx: usize,
    pub cont_corr_idx: usize,
    pub static_eval: Value,
    pub stat_score: i32,
    pub move_count: i32,
    pub in_check: bool,
    pub tt_pv: bool,
    pub tt_hit: bool,
    pub cutoff_cnt: u8,
    pub excluded: Move,
    pub is_null: bool,
    pub reduction: i32,
    pub corr_value: i32,
    /// Squares attacked by the opponent at this node (0 unless ThreatHist is on).
    pub threats: u64,
}

impl Default for StackEntry {
    fn default() -> Self {
        StackEntry {
            current_move: Move::NONE,
            moved_piece: Piece::None,
            cont_idx: History::cont_index(false, false, Piece::None, 0),
            cont_corr_idx: Piece::None.idx() * 64,
            static_eval: VALUE_NONE,
            stat_score: 0,
            move_count: 0,
            in_check: false,
            tt_pv: false,
            tt_hit: false,
            cutoff_cnt: 0,
            excluded: Move::NONE,
            is_null: false,
            reduction: 0,
            corr_value: 0,
            threats: 0,
        }
    }
}

#[derive(Clone)]
pub struct RootMove {
    pub mv: Move,
    pub score: Value,
    pub prev_score: Value,
    pub avg_score: Value,
    pub sel_depth: usize,
    pub pv: Vec<Move>,
    pub effort: u64,
}

pub struct Shared {
    pub tt: TranspositionTable,
    pub stop: AtomicBool,
    pub nodes: Vec<AtomicU64>,
    pub tb_hits: AtomicU64,
}

#[derive(Clone)]
pub struct Limits {
    pub go: GoParams,
    pub tm: TimeManager,
    pub max_depth: i32,
    pub max_nodes: u64,
}

pub struct Options {
    pub threads: usize,
    pub multi_pv: usize,
    pub move_overhead: i64,
    pub chess960: bool,
    pub silent: bool,
    /// Best score of the previous search in this game (VALUE_INFINITE if none); used by time management.
    pub prev_score: Value,
}

const SS_OFFSET: usize = 8;

pub struct Thread<'a> {
    pub id: usize,
    shared: &'a Shared,
    net: Option<&'a AnyNet>,
    nnue: AnyState,
    pub hist: History,
    ss: Vec<StackEntry>,
    pv: Vec<[Move; MAX_PLY + 2]>,
    pv_len: Vec<usize>,
    keys: Vec<u64>,
    root_key_idx: usize,
    pub root_moves: Vec<RootMove>,
    root_depth: i32,
    pub completed_depth: i32,
    sel_depth: usize,
    nodes: u64,
    nmp_min_ply: usize,
    pv_idx: usize,
    multi_pv: usize,
    root_delta: Value,
    limits: Limits,
    reductions: [i32; 256],
    best_move_changes: f64,
    silent: bool,
    threads: usize,
    stop_local: bool,
    pub best_prev_score: Value,
    pub iter_value: [Value; 4],
    pub prev_time_reduction: f64,
    last_best_move: Move,
    last_best_depth: i32,
    opt_time: f64,
}

#[inline]
fn value_draw(nodes: u64) -> Value {
    (nodes & 2) as Value - 1
}

impl<'a> Thread<'a> {
    pub fn new(id: usize, shared: &'a Shared, net: Option<&'a AnyNet>, limits: Limits, opts: &Options, hist: History) -> Self {
        let mut reductions = [0i32; 256];
        for (i, r) in reductions.iter_mut().enumerate().skip(1) {
            *r = ((20.37 + (opts.threads as f64).ln() / 2.0) * (i as f64).ln()) as i32;
        }
        Thread {
            id,
            shared,
            net,
            nnue: AnyState::for_net(net),
            hist,
            ss: vec![StackEntry::default(); MAX_PLY + SS_OFFSET + 8],
            pv: vec![[Move::NONE; MAX_PLY + 2]; MAX_PLY + 2],
            pv_len: vec![0; MAX_PLY + 2],
            keys: Vec::new(),
            root_key_idx: 0,
            root_moves: Vec::new(),
            root_depth: 0,
            completed_depth: 0,
            sel_depth: 0,
            nodes: 0,
            nmp_min_ply: 0,
            pv_idx: 0,
            multi_pv: opts.multi_pv.max(1),
            root_delta: 1,
            limits,
            reductions,
            best_move_changes: 0.0,
            silent: opts.silent,
            threads: opts.threads,
            stop_local: false,
            best_prev_score: opts.prev_score,
            iter_value: [VALUE_ZERO_INIT; 4],
            prev_time_reduction: 1.0,
            last_best_move: Move::NONE,
            last_best_depth: 0,
            opt_time: 0.0,
        }
    }

    #[inline(always)]
    fn ss(&mut self, ply: usize) -> &mut StackEntry {
        &mut self.ss[ply + SS_OFFSET]
    }
    #[inline(always)]
    fn ss_at(&self, ply: usize) -> &StackEntry {
        &self.ss[ply + SS_OFFSET]
    }
    #[inline(always)]
    fn ss_prev(&self, ply: usize, back: usize) -> &StackEntry {
        &self.ss[ply + SS_OFFSET - back]
    }

    #[inline(always)]
    fn is_main(&self) -> bool {
        self.id == 0
    }

    #[inline(always)]
    fn should_stop(&self) -> bool {
        self.stop_local || self.shared.stop.load(Ordering::Relaxed)
    }

    fn total_nodes(&self) -> u64 {
        self.shared.nodes.iter().map(|n| n.load(Ordering::Relaxed)).sum()
    }

    #[inline]
    fn check_time(&mut self) {
        if !self.is_main() {
            return;
        }
        let tm = self.limits.tm;
        if tm.use_time && tm.elapsed_ms() >= tm.maximum {
            self.shared.stop.store(true, Ordering::Relaxed);
        }
        if self.limits.max_nodes > 0 && self.total_nodes() >= self.limits.max_nodes {
            self.shared.stop.store(true, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    fn reduction(&self, improving: bool, depth: i32, move_count: i32, delta: Value) -> i32 {
        let d = (depth.max(0) as usize).min(255);
        let mn = (move_count.max(0) as usize).min(255);
        let r = self.reductions[d] * self.reductions[mn];
        (r * crate::params::LMR_SCALE_PCT.get() / 100) - delta * 577 / self.root_delta.max(1) + (!improving as i32) * self.reductions[d] * 197 / 512 + crate::params::LMR_BASE.get()
    }

    #[inline]
    fn evaluate(&mut self, pos: &Position) -> Value {
        eval::evaluate(pos, self.net, &mut self.nnue)
    }

    /// Draw by repetition or 50-move rule (position on top of the key stack).
    fn is_draw(&self, pos: &Position, _ply: usize) -> bool {
        if pos.rule50() >= 100 {
            if !pos.in_check() || !legal_moves(pos).is_empty() {
                return true;
            }
        }
        let n = self.keys.len();
        let rule50 = pos.rule50() as usize;
        let cur = pos.key();
        let mut count = 0;
        let mut k = 2;
        while k <= rule50 && k <= n {
            let idx = n - k;
            if self.keys[idx] == cur {
                // A repetition strictly inside the search tree is a draw; positions from the game
                // history must repeat twice (Stockfish semantics).
                if idx >= self.root_key_idx {
                    return true;
                }
                count += 1;
                if count >= 2 {
                    return true;
                }
            }
            k += 2;
        }
        false
    }

    #[inline(always)]
    fn update_pv(&mut self, ply: usize, m: Move) {
        let child_len = self.pv_len[ply + 1];
        self.pv[ply][0] = m;
        for i in 0..child_len {
            self.pv[ply][i + 1] = self.pv[ply + 1][i];
        }
        self.pv_len[ply] = child_len + 1;
    }

    #[inline(always)]
    fn corr_value(&self, pos: &Position, ply: usize) -> i32 {
        let us = pos.side_to_move();
        let h = &self.hist;
        let pawn = h.pawn_corr[History::corr_idx(us, pos.pawn_key())];
        let minor = h.minor_corr[History::corr_idx(us, pos.minor_key())];
        let major = h.major_corr[History::corr_idx(us, pos.major_key())];
        let npw = h.non_pawn_corr[History::non_pawn_idx(us, Color::White, pos.non_pawn_key(Color::White))];
        let npb = h.non_pawn_corr[History::non_pawn_idx(us, Color::Black, pos.non_pawn_key(Color::Black))];
        let mut cont = 0;
        if ply >= 2 {
            let p2 = self.ss_prev(ply, 2);
            let p1 = self.ss_prev(ply, 1);
            if !p1.is_null && !p2.is_null {
                cont = h.cont_corr[History::cont_corr_idx(p2.cont_corr_idx, p1.moved_piece, p1.current_move.to())];
            }
        }
        (pawn * 2 + minor + major + npw + npb + cont * 2) / 8
    }

    #[inline(always)]
    fn corrected_eval(&self, v: Value, cv: i32) -> Value {
        (v + cv / CORR_GRAIN).clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
    }

    fn update_correction(&mut self, pos: &Position, ply: usize, diff: i32, depth: i32) {
        let us = pos.side_to_move();
        let w = (depth + 1).min(16);
        let diff = diff.clamp(-CORR_MAX / CORR_GRAIN, CORR_MAX / CORR_GRAIN);
        let i = History::corr_idx(us, pos.pawn_key());
        History::corr_update(&mut self.hist.pawn_corr[i], diff, w);
        let i = History::corr_idx(us, pos.minor_key());
        History::corr_update(&mut self.hist.minor_corr[i], diff, w);
        let i = History::corr_idx(us, pos.major_key());
        History::corr_update(&mut self.hist.major_corr[i], diff, w);
        let i = History::non_pawn_idx(us, Color::White, pos.non_pawn_key(Color::White));
        History::corr_update(&mut self.hist.non_pawn_corr[i], diff, w);
        let i = History::non_pawn_idx(us, Color::Black, pos.non_pawn_key(Color::Black));
        History::corr_update(&mut self.hist.non_pawn_corr[i], diff, w);
        if ply >= 2 {
            let p2 = *self.ss_prev(ply, 2);
            let p1 = *self.ss_prev(ply, 1);
            if !p1.is_null && !p2.is_null {
                let i = History::cont_corr_idx(p2.cont_corr_idx, p1.moved_piece, p1.current_move.to());
                History::corr_update(&mut self.hist.cont_corr[i], diff, w);
            }
        }
    }

    /// Continuation history update for the move at `ply` (plies 1,2,3,4,6 back).
    fn update_cont_histories(&mut self, ply: usize, piece: Piece, to: Square, bonus: i32) {
        const WEIGHTS: [(usize, i32); 5] = [(1, 1024), (2, 768), (3, 300), (4, 500), (6, 400)];
        for &(back, w) in WEIGHTS.iter() {
            if ply < back {
                break;
            }
            let prev = self.ss_prev(ply, back);
            if prev.is_null || (back > 2 && self.ss_at(ply).in_check) {
                continue;
            }
            let idx = prev.cont_idx;
            self.hist.cont_update(idx, piece, to, bonus * w / 1024);
        }
    }

    fn update_quiet_histories(&mut self, pos: &Position, ply: usize, m: Move, bonus: i32) {
        let us = pos.side_to_move();
        let pc = pos.moved_piece(m);
        let ti = History::threat_index(m, self.ss_at(ply).threats);
        self.hist.main_update(us, m, ti, bonus);
        self.hist.low_ply_update(ply, m, bonus * 712 / 1024);
        self.update_cont_histories(ply, pc, m.to(), bonus);
        let pb = if bonus > 0 { bonus * 1104 / 1024 } else { bonus * 459 / 1024 };
        self.hist.pawn_update(pos.pawn_key(), pc, m.to(), pb);
    }

    fn update_all_stats(
        &mut self,
        pos: &Position,
        ply: usize,
        best_move: Move,
        quiets: &[Move],
        captures: &[Move],
        depth: i32,
        tt_move: Move,
    ) {
        let bonus = stat_bonus(depth) + 300 * (best_move == tt_move) as i32;
        let malus = stat_malus(depth);
        let moved = pos.moved_piece(best_move);
        if !pos.is_capture_or_promo(best_move) {
            self.update_quiet_histories(pos, ply, best_move, bonus * 899 / 1024);
            self.hist.update_killers(ply, best_move);
            if ply >= 1 {
                let prev = *self.ss_prev(ply, 1);
                if !prev.is_null {
                    self.hist.counter[prev.moved_piece.idx()][prev.current_move.to() as usize] = best_move;
                }
            }
            for (i, &q) in quiets.iter().enumerate() {
                let m = -malus * (921i32.pow(i.min(6) as u32) as i64 / 1024i64.pow(i.min(6) as u32)) as i32;
                let m = if i >= 7 { -malus / 2 } else { m };
                self.update_quiet_histories(pos, ply, q, m);
            }
        } else {
            let captured = pos.captured_type(best_move).unwrap_or(PieceType::Pawn);
            self.hist.capture_update(moved, best_move.to(), captured, bonus * 1427 / 1024);
        }
        for &c in captures {
            let captured = pos.captured_type(c).unwrap_or(PieceType::Pawn);
            let pc = pos.moved_piece(c);
            self.hist.capture_update(pc, c.to(), captured, -malus * 1489 / 1024);
        }
    }

    // -----------------------------------------------------------------------------------------
    // Main search
    // -----------------------------------------------------------------------------------------
    pub fn search(&mut self, pos: &Position, nt: NodeType, mut alpha: Value, mut beta: Value, mut depth: i32, cut_node: bool, ply: usize) -> Value {
        let pv_node = nt != NodeType::NonPv;
        let root = nt == NodeType::Root;
        let all_node = !pv_node && !cut_node;

        if depth <= 0 {
            return self.qsearch(pos, pv_node, alpha, beta, ply);
        }
        depth = depth.min(MAX_PLY as i32 - 1);
        self.pv_len[ply] = 0;

        if self.nodes & 1023 == 0 {
            self.check_time();
        }
        if self.should_stop() {
            return 0;
        }
        if pv_node && self.sel_depth < ply + 1 {
            self.sel_depth = ply + 1;
        }

        let us = pos.side_to_move();
        if !root {
            if self.is_draw(pos, ply) {
                return value_draw(self.nodes);
            }
            // Upcoming repetition: the side to move can force a repetition, so it can claim a draw.
            if alpha < VALUE_DRAW && crate::params::CUCKOO.get() != 0 && crate::cuckoo::has_upcoming_repetition(pos, &self.keys, ply) {
                alpha = value_draw(self.nodes);
                if alpha >= beta {
                    return alpha;
                }
            }
            if ply >= MAX_PLY - 1 {
                return if pos.in_check() { VALUE_DRAW } else { self.evaluate(pos) };
            }
            alpha = alpha.max(mated_in(ply));
            beta = beta.min(mate_in(ply + 1));
            if alpha >= beta {
                return alpha;
            }
        }

        let in_check = pos.in_check();
        let threats = if crate::params::THREAT_HIST.get() != 0 { pos.attacked_squares(!pos.side_to_move()) } else { 0 };
        self.ss(ply).threats = threats;
        let excluded = self.ss_at(ply).excluded;
        {
            let s = self.ss(ply);
            s.in_check = in_check;
            s.move_count = 0;
            s.is_null = false;
        }
        self.ss(ply + 1).excluded = Move::NONE;
        self.ss(ply + 1).cutoff_cnt = 0;
        self.hist.clear_killers(ply + 1);

        // ---- Transposition table ----
        let key = pos.key();
        let (tt_hit, tt, writer) = self.shared.tt.probe(key);
        let tt_hit = tt_hit && excluded.is_none();
        let tt_value = if tt_hit { value_from_tt(tt.value, ply, pos.rule50()) } else { VALUE_NONE };
        let tt_move = if root {
            self.root_moves[self.pv_idx].mv
        } else if tt_hit {
            tt.mv
        } else {
            Move::NONE
        };
        let tt_capture = !tt_move.is_none() && pos.is_capture_or_promo(tt_move);
        self.ss(ply).tt_hit = tt_hit;
        if excluded.is_none() {
            self.ss(ply).tt_pv = pv_node || (tt_hit && tt.is_pv);
        }
        let tt_pv = self.ss_at(ply).tt_pv;
        let mut best_value_floor = -VALUE_INFINITE;

        // TT cutoff
        if !pv_node
            && tt_hit
            && tt.depth > depth - (tt_value <= beta) as i32
            && is_valid(tt_value)
            && (if tt_value >= beta { tt.bound.has_lower() } else { tt.bound.has_upper() })
            && (cut_node == (tt_value >= beta) || depth > 5)
        {
            if !tt_move.is_none() && tt_value >= beta {
                if !tt_capture {
                    let b = stat_bonus(depth);
                    self.update_quiet_histories(pos, ply, tt_move, b);
                }
                // Penalty for the previous quiet move that allowed this early cutoff.
                if ply >= 1 {
                    let prev = *self.ss_prev(ply, 1);
                    if !prev.is_null && !prev.current_move.is_none() && prev.moved_piece != Piece::None {
                        // (simplified: no access to previous position's capture status here)
                    }
                }
            }
            if pos.rule50() < 90 {
                return tt_value;
            }
        }

        // ---- Syzygy tablebase probe ----
        let mut max_value = VALUE_INFINITE;
        if !root && crate::tb::largest() > 0 && excluded.is_none() {
            let cardinality = (crate::params::TB_PROBE_LIMIT.get() as u32).min(crate::tb::largest());
            let pieces = pos.piece_count();
            if pieces <= cardinality
                && (pieces < cardinality || depth >= crate::params::TB_PROBE_DEPTH.get())
                && pos.rule50() == 0
                && pos.castling_rights() == 0
            {
                if let Some(wdl) = crate::tb::probe_wdl(pos) {
                    self.shared.tb_hits.fetch_add(1, Ordering::Relaxed);
                    let draw_score = 1; // treat cursed wins / blessed losses as draws
                    let wdl_i = wdl as i32 - 2; // -2..2
                    let tb_value = VALUE_TB_WIN - pieces as Value;
                    let (value, bound) = if wdl_i < -draw_score {
                        (-tb_value + ply as Value, Bound::Upper)
                    } else if wdl_i > draw_score {
                        (tb_value - ply as Value, Bound::Lower)
                    } else {
                        (VALUE_DRAW + 2 * wdl_i * draw_score, Bound::Exact)
                    };
                    if bound == Bound::Exact || (if bound == Bound::Lower { value >= beta } else { value <= alpha }) {
                        self.shared.tt.save(&writer, key, value_to_tt(value, ply), tt_pv, bound, (depth + 6).min(MAX_PLY as i32 - 1), Move::NONE, VALUE_NONE);
                        return value;
                    }
                    if pv_node {
                        if bound == Bound::Lower {
                            best_value_floor = value;
                            alpha = alpha.max(value);
                        } else {
                            max_value = value;
                        }
                    }
                }
            }
        }

        // ---- Static evaluation ----
        let mut unadjusted_eval = VALUE_NONE;
        let mut eval;
        let cv;
        if in_check {
            self.ss(ply).static_eval = VALUE_NONE;
            eval = VALUE_NONE;
            cv = 0;
        } else {
            cv = self.corr_value(pos, ply);
            if !excluded.is_none() {
                unadjusted_eval = self.ss_at(ply).static_eval;
                eval = unadjusted_eval;
            } else if tt_hit {
                unadjusted_eval = if is_valid(tt.eval) { tt.eval } else { self.evaluate(pos) };
                eval = self.corrected_eval(unadjusted_eval, cv);
                self.ss(ply).static_eval = eval;
                if is_valid(tt_value) && (if tt_value > eval { tt.bound.has_lower() } else { tt.bound.has_upper() }) {
                    eval = tt_value;
                }
            } else {
                unadjusted_eval = self.evaluate(pos);
                eval = self.corrected_eval(unadjusted_eval, cv);
                self.ss(ply).static_eval = eval;
                self.shared.tt.save(&writer, key, VALUE_NONE, tt_pv, Bound::None, DEPTH_UNSEARCHED, Move::NONE, unadjusted_eval);
            }
        }
        self.ss(ply).corr_value = cv;

        let static_eval = self.ss_at(ply).static_eval;
        let improving = !in_check && {
            let p2 = if ply >= 2 { self.ss_prev(ply, 2).static_eval } else { VALUE_NONE };
            let p4 = if ply >= 4 { self.ss_prev(ply, 4).static_eval } else { VALUE_NONE };
            if is_valid(p2) {
                static_eval > p2
            } else if is_valid(p4) {
                static_eval > p4
            } else {
                true
            }
        };
        let opponent_worsening = !in_check && ply >= 1 && {
            let p1 = self.ss_prev(ply, 1).static_eval;
            is_valid(p1) && static_eval + p1 > 2
        };

        let prev_move = if ply >= 1 { self.ss_prev(ply, 1).current_move } else { Move::NONE };
        let prev_null = ply >= 1 && self.ss_prev(ply, 1).is_null;

        if !in_check && !root {
            // ---- Razoring ----
            if !pv_node && eval < alpha - crate::params::RAZOR_MULT.get() * depth && !is_loss(alpha) {
                let v = self.qsearch(pos, false, alpha - 1, alpha, ply);
                if v < alpha && !is_decisive(v) {
                    return v;
                }
            }

            // ---- Reverse futility pruning ----
            if !tt_pv && depth < crate::params::RFP_DEPTH.get() && eval >= beta && (tt_move.is_none() || tt_capture) && !is_loss(beta) && !is_win(eval) {
                let fut_mult = (crate::params::RFP_MULT.get() + depth * 4).min(crate::params::RFP_MULT.get() + 40) - 20 * (!tt_hit) as i32;
                let margin = fut_mult * depth - (2789 * improving as i32 + 335 * opponent_worsening as i32) * fut_mult / 1024 + cv.abs() / 4096;
                if eval - margin >= beta {
                    return if is_decisive(eval) { eval } else { beta + (eval - beta) / 3 };
                }
            }

            // ---- Null move pruning ----
            if cut_node
                && !prev_null
                && excluded.is_none()
                && eval >= beta
                && static_eval >= beta - 13 * depth + 365 - 47 * improving as i32
                && pos.has_non_pawn_material(us)
                && ply >= self.nmp_min_ply
                && !is_loss(beta)
            {
                let r = crate::params::NMP_BASE.get() + depth / 3 + ((eval - beta) / 200).min(4);
                {
                    let s = self.ss(ply);
                    s.current_move = Move::NONE;
                    s.moved_piece = Piece::None;
                    s.is_null = true;
                    s.cont_idx = History::cont_index(false, false, Piece::None, 0);
                    s.cont_corr_idx = Piece::None.idx() * 64;
                }
                let child = pos.make_null_move();
                self.nnue.push_null(&child);
                self.keys.push(key);
                self.nodes += 1;
                let null_value = -self.search(&child, NodeType::NonPv, -beta, -beta + 1, depth - r, false, ply + 1);
                self.keys.pop();
                self.nnue.pop();
                self.ss(ply).is_null = false;

                if null_value >= beta && !is_win(null_value) {
                    if self.nmp_min_ply > 0 || depth < 16 {
                        return null_value;
                    }
                    // Verification search at high depths.
                    self.nmp_min_ply = ply + (3 * (depth - r) / 4).max(0) as usize;
                    let v = self.search(pos, NodeType::NonPv, beta - 1, beta, depth - r, false, ply);
                    self.nmp_min_ply = 0;
                    if v >= beta {
                        return null_value;
                    }
                }
            }

            // ---- Internal iterative reductions ----
            if !all_node && depth >= 6 && tt_move.is_none() {
                depth -= 1;
            }

            // ---- ProbCut ----
            let probcut_beta = beta + 241 - 64 * improving as i32;
            if !pv_node && depth >= 3 && !is_decisive(beta) && !(is_valid(tt_value) && tt_value < probcut_beta) {
                let probcut_depth = (depth - if improving { 5 } else { 3 }).max(0);
                let mut mp = MovePicker::new_probcut(pos, tt_move, probcut_beta - static_eval);
                loop {
                    let m = mp.next(pos, &self.hist);
                    if m.is_none() {
                        break;
                    }
                    if m == excluded || !pos.is_legal(m) {
                        continue;
                    }
                    let capture = pos.is_capture_or_promo(m);
                    let mp_piece = pos.moved_piece(m);
                    {
                        let s = self.ss(ply);
                        s.current_move = m;
                        s.moved_piece = mp_piece;
                        s.cont_idx = History::cont_index(in_check, capture, mp_piece, m.to());
                        s.cont_corr_idx = mp_piece.idx() * 64 + m.to() as usize;
                    }
                    let child = pos.make_move(m);
                    self.nnue.push(pos, m, &child);
                    self.keys.push(key);
                    self.nodes += 1;
                    let mut value = -self.qsearch(&child, false, -probcut_beta, -probcut_beta + 1, ply + 1);
                    if value >= probcut_beta && probcut_depth > 0 {
                        value = -self.search(&child, NodeType::NonPv, -probcut_beta, -probcut_beta + 1, probcut_depth, !cut_node, ply + 1);
                    }
                    self.keys.pop();
                    self.nnue.pop();
                    if value >= probcut_beta {
                        self.shared.tt.save(&writer, key, value_to_tt(value, ply), tt_pv, Bound::Lower, probcut_depth + 1, m, unadjusted_eval);
                        if !is_decisive(value) {
                            return value - (probcut_beta - beta);
                        }
                        return value;
                    }
                }
            }
            // Small ProbCut idea from the TT.
            if tt_hit && tt.bound.has_lower() && tt.depth >= depth - 4 && is_valid(tt_value) && tt_value >= probcut_beta && !is_decisive(tt_value) && !is_decisive(beta) {
                return probcut_beta;
            }
        }

        // ---- Move loop ----
        let cont_idx = [
            if ply >= 1 { self.ss_prev(ply, 1).cont_idx } else { usize::MAX },
            if ply >= 2 { self.ss_prev(ply, 2).cont_idx } else { usize::MAX },
            if ply >= 4 { self.ss_prev(ply, 4).cont_idx } else { usize::MAX },
            if ply >= 6 { self.ss_prev(ply, 6).cont_idx } else { usize::MAX },
        ];
        let counter = if ply >= 1 && !prev_null && !prev_move.is_none() {
            let p = self.ss_prev(ply, 1);
            self.hist.counter[p.moved_piece.idx()][p.current_move.to() as usize]
        } else {
            Move::NONE
        };
        let killers = self.hist.killers[ply];
        let mut mp = MovePicker::new(pos, tt_move, killers, counter, cont_idx, depth, ply);

        let mut best_value = best_value_floor;
        let mut best_move = Move::NONE;
        let mut move_count = 0;
        let mut quiets_searched: Vec<Move> = Vec::with_capacity(32);
        let mut captures_searched: Vec<Move> = Vec::with_capacity(16);
        let mut skip_quiets = false;

        loop {
            let m = mp.next(pos, &self.hist);
            if m.is_none() {
                break;
            }
            if m == excluded {
                continue;
            }
            if root && !self.root_moves[self.pv_idx..].iter().any(|rm| rm.mv == m) {
                continue;
            }
            if !pos.is_legal(m) {
                continue;
            }
            move_count += 1;
            self.ss(ply).move_count = move_count;

            let capture = pos.is_capture_or_promo(m);
            let moved_piece = pos.moved_piece(m);
            let gives_check = pos.gives_check(m);
            let mut new_depth = depth - 1;
            let delta = beta - alpha;
            let mut r = self.reduction(improving, depth, move_count, delta);

            // ---- Pruning at shallow depth ----
            if !root && pos.has_non_pawn_material(us) && !is_loss(best_value) {
                if !skip_quiets && move_count >= (crate::params::LMP_BASE.get() + depth * depth) / (2 - improving as i32) {
                    mp.skip_quiets();
                    skip_quiets = true;
                }
                let mut lmr_depth = new_depth - r / 1024;
                if capture || gives_check {
                    let captured = pos.captured_type(m).unwrap_or(PieceType::Pawn);
                    let capt_hist = self.hist.capture_get(moved_piece, m.to(), captured);
                    if !gives_check && lmr_depth < 7 && !in_check {
                        let fut = static_eval + 232 + 224 * lmr_depth + piece_value(captured) + 131 * capt_hist / 1024;
                        if fut <= alpha {
                            continue;
                        }
                    }
                    let see_margin = 177 * depth + capt_hist * 34 / 1024;
                    if !pos.see_ge(m, -see_margin) {
                        continue;
                    }
                } else {
                    let mut history = self.hist.pawn_get(pos.pawn_key(), moved_piece, m.to());
                    if cont_idx[0] != usize::MAX {
                        history += self.hist.cont_get(cont_idx[0], moved_piece, m.to());
                    }
                    if cont_idx[1] != usize::MAX {
                        history += self.hist.cont_get(cont_idx[1], moved_piece, m.to());
                    }
                    if history < -4136 * depth {
                        continue;
                    }
                    history += 2 * self.hist.main_get(us, m, History::threat_index(m, threats));
                    lmr_depth += history / 3600;
                    if !in_check && lmr_depth < 12 {
                        let fut = static_eval + crate::params::FUT_MARGIN.get() * lmr_depth + 90 * (static_eval > alpha) as i32 + 164;
                        if fut <= alpha {
                            if best_value <= fut && !is_decisive(best_value) && !is_win(fut) {
                                best_value = fut;
                            }
                            continue;
                        }
                    }
                    lmr_depth = lmr_depth.max(0);
                    if !pos.see_ge(m, -23 * lmr_depth * lmr_depth) {
                        continue;
                    }
                }
            }

            // ---- Extensions ----
            let mut extension = 0;
            if !root
                && m == tt_move
                && excluded.is_none()
                && depth >= 6 + tt_pv as i32
                && is_valid(tt_value)
                && !is_decisive(tt_value)
                && tt.bound.has_lower()
                && tt.depth >= depth - 3
            {
                let singular_beta = tt_value - (crate::params::SE_MARGIN.get() + 66 * (tt_pv && !pv_node) as i32) * depth / 63;
                let singular_depth = new_depth / 2;
                self.ss(ply).excluded = m;
                let value = self.search(pos, NodeType::NonPv, singular_beta - 1, singular_beta, singular_depth, cut_node, ply);
                self.ss(ply).excluded = Move::NONE;
                if value < singular_beta {
                    let double_margin = 2 + 204 * pv_node as i32 - 152 * (!tt_capture) as i32 - cv.abs() / 4096;
                    let triple_margin = 70 + 279 * pv_node as i32 - 188 * (!tt_capture) as i32 + 81 * tt_pv as i32;
                    extension = 1 + (value < singular_beta - double_margin) as i32 + (value < singular_beta - triple_margin) as i32;
                    if depth < 16 {
                        depth += 1;
                    }
                } else if value >= beta && !is_decisive(value) {
                    // Multi-cut: the TT move and at least one other move fail high.
                    return value;
                } else if tt_value >= beta {
                    extension = -3;
                } else if cut_node {
                    extension = -2;
                }
            }
            new_depth += extension;

            // ---- Make the move ----
            {
                let s = self.ss(ply);
                s.current_move = m;
                s.moved_piece = moved_piece;
                s.cont_idx = History::cont_index(in_check, capture, moved_piece, m.to());
                s.cont_corr_idx = moved_piece.idx() * 64 + m.to() as usize;
            }
            let child = pos.make_move(m);
            self.shared.tt.prefetch(child.key());
            self.nnue.push(pos, m, &child);
            self.keys.push(key);
            self.nodes += 1;
            self.shared.nodes[self.id].store(self.nodes, Ordering::Relaxed);
            let nodes_before = self.nodes;

            // ---- Late move reductions ----
            if tt_pv {
                r -= 2230 + pv_node as i32 * 1017 + (tt_value > alpha) as i32 * 925 + (tt.depth >= depth) as i32 * (971 + cut_node as i32 * 1002);
            }
            r += 316 - move_count * 32;
            r -= cv.abs() / 128;
            if cut_node {
                r += crate::params::LMR_CUTNODE.get() + 1024 * tt_move.is_none() as i32;
            }
            if tt_capture {
                r += 1350;
            }
            if self.ss_at(ply + 1).cutoff_cnt > 3 {
                r += 981 + all_node as i32 * 833;
            } else if m == tt_move {
                r -= 2000;
            }
            let stat_score = if capture {
                let captured = pos.captured_type(m).unwrap_or(PieceType::Pawn);
                7 * piece_value(captured) + self.hist.capture_get(moved_piece, m.to(), captured) - 5000
            } else {
                let mut s = 2 * self.hist.main_get(us, m, History::threat_index(m, threats));
                if cont_idx[0] != usize::MAX {
                    s += self.hist.cont_get(cont_idx[0], moved_piece, m.to());
                }
                if cont_idx[1] != usize::MAX {
                    s += self.hist.cont_get(cont_idx[1], moved_piece, m.to());
                }
                s - 3996
            };
            self.ss(ply).stat_score = stat_score;
            r -= stat_score * 1287 / crate::params::LMR_HIST_DIV.get();

            let mut value;
            if depth >= 2 && move_count > 1 {
                let d = (new_depth - r / 1024).min(new_depth + !all_node as i32).max(1) + pv_node as i32;
                self.ss(ply).reduction = new_depth - d;
                value = -self.search(&child, NodeType::NonPv, -(alpha + 1), -alpha, d, true, ply + 1);
                self.ss(ply).reduction = 0;
                if value > alpha && d < new_depth {
                    let do_deeper = value > best_value + 43 + 2 * new_depth;
                    let do_shallower = value < best_value + 9;
                    new_depth += do_deeper as i32 - do_shallower as i32;
                    if new_depth > d {
                        value = -self.search(&child, NodeType::NonPv, -(alpha + 1), -alpha, new_depth, !cut_node, ply + 1);
                    }
                    if value >= beta && !capture {
                        self.update_cont_histories(ply, moved_piece, m.to(), 2 * stat_bonus(new_depth));
                    }
                }
            } else if !pv_node || move_count > 1 {
                let d = if tt_move.is_none() && r > 3200 { new_depth - 1 } else { new_depth };
                value = -self.search(&child, NodeType::NonPv, -(alpha + 1), -alpha, d, !cut_node, ply + 1);
            } else {
                value = 0;
            }

            if pv_node && (move_count == 1 || value > alpha) {
                self.pv_len[ply + 1] = 0;
                value = -self.search(&child, NodeType::Pv, -beta, -alpha, new_depth, false, ply + 1);
            }

            self.keys.pop();
            self.nnue.pop();

            if self.should_stop() {
                return 0;
            }

            if root {
                let idx = self.root_moves.iter().position(|rm| rm.mv == m).unwrap();
                self.root_moves[idx].effort += self.nodes - nodes_before;
                if move_count == 1 || value > alpha {
                    self.root_moves[idx].score = value;
                    self.root_moves[idx].sel_depth = self.sel_depth;
                    self.root_moves[idx].pv.clear();
                    self.root_moves[idx].pv.push(m);
                    for i in 0..self.pv_len[ply + 1] {
                        let mv = self.pv[ply + 1][i];
                        self.root_moves[idx].pv.push(mv);
                    }
                    if move_count > 1 && self.pv_idx == 0 {
                        self.best_move_changes += 1.0;
                    }
                } else {
                    self.root_moves[idx].score = -VALUE_INFINITE;
                }
            }

            if value > best_value {
                best_value = value;
                if value > alpha {
                    best_move = m;
                    if pv_node && !root {
                        self.update_pv(ply, m);
                    }
                    if value >= beta {
                        let s = self.ss(ply);
                        s.cutoff_cnt += 1 + (tt_move.is_none()) as u8;
                        break;
                    }
                    if depth > 2 && depth < 14 && !is_decisive(value) {
                        depth -= 2;
                    }
                    alpha = value;
                }
            }
            if m != best_move && move_count <= 32 {
                if capture {
                    captures_searched.push(m);
                } else {
                    quiets_searched.push(m);
                }
            }
        }

        // ---- Checkmate / stalemate ----
        if move_count == 0 {
            best_value = if !excluded.is_none() {
                alpha
            } else if in_check {
                mated_in(ply)
            } else {
                VALUE_DRAW
            };
        } else if !best_move.is_none() {
            self.update_all_stats(pos, ply, best_move, &quiets_searched, &captures_searched, depth, tt_move);
        } else if ply >= 1 && !prev_null && !prev_move.is_none() {
            // Bonus for the previous move that caused this fail-low.
            let p1 = *self.ss_prev(ply, 1);
            let bonus_scale = (-241 - p1.stat_score / 98 + (59 * depth).min(420) + 186 * (p1.move_count > 9) as i32
                + 142 * (!in_check && best_value <= static_eval - 106) as i32)
                .max(0);
            let bonus = stat_bonus(depth) * bonus_scale;
            if p1.moved_piece != Piece::None {
                let piece = p1.moved_piece;
                let to = p1.current_move.to();
                // Continuation histories of the previous position (ply-1).
                self.update_cont_histories_at(ply - 1, piece, to, bonus * 263 / 16384);
                let ti = History::threat_index(p1.current_move, p1.threats);
                self.hist.main_update(!us, p1.current_move, ti, bonus * 215 / 32768);
            }
        }

        if pv_node {
            best_value = best_value.min(max_value);
        }
        if best_value <= alpha {
            let prev_tt_pv = ply >= 1 && self.ss_prev(ply, 1).tt_pv;
            let s = self.ss(ply);
            s.tt_pv = s.tt_pv || (prev_tt_pv && depth > 3);
        }
        let tt_pv = self.ss_at(ply).tt_pv;

        if excluded.is_none() && !(root && self.pv_idx > 0) {
            let bound = if best_value >= beta {
                Bound::Lower
            } else if pv_node && !best_move.is_none() {
                Bound::Exact
            } else {
                Bound::Upper
            };
            self.shared.tt.save(&writer, key, value_to_tt(best_value, ply), tt_pv, bound, depth, best_move, unadjusted_eval);
        }

        // ---- Correction history update ----
        if !in_check
            && !(best_move.is_none() == false && pos.is_capture_or_promo(best_move))
            && !(best_value >= beta && best_value <= static_eval)
            && !(best_move.is_none() && best_value >= static_eval)
            && is_valid(static_eval)
        {
            let diff = best_value - static_eval;
            self.update_correction(pos, ply, diff, depth);
        }

        best_value
    }

    /// Continuation history update for a move made at `ply` (used for the previous ply's move).
    fn update_cont_histories_at(&mut self, ply: usize, piece: Piece, to: Square, bonus: i32) {
        self.update_cont_histories(ply, piece, to, bonus);
    }

    // -----------------------------------------------------------------------------------------
    // Quiescence search
    // -----------------------------------------------------------------------------------------
    pub fn qsearch(&mut self, pos: &Position, pv_node: bool, mut alpha: Value, beta: Value, ply: usize) -> Value {
        self.pv_len[ply] = 0;
        if self.nodes & 1023 == 0 {
            self.check_time();
        }
        if self.should_stop() {
            return 0;
        }
        if pv_node && self.sel_depth < ply + 1 {
            self.sel_depth = ply + 1;
        }
        if self.is_draw(pos, ply) {
            return value_draw(self.nodes);
        }
        if alpha < VALUE_DRAW && crate::params::CUCKOO.get() != 0 && crate::cuckoo::has_upcoming_repetition(pos, &self.keys, ply) {
            alpha = value_draw(self.nodes);
            if alpha >= beta {
                return alpha;
            }
        }
        if ply >= MAX_PLY - 1 {
            return if pos.in_check() { VALUE_DRAW } else { self.evaluate(pos) };
        }
        let in_check = pos.in_check();
        {
            let s = self.ss(ply);
            s.in_check = in_check;
            s.is_null = false;
            s.move_count = 0;
        }

        let key = pos.key();
        let (tt_hit, tt, writer) = self.shared.tt.probe(key);
        let tt_value = if tt_hit { value_from_tt(tt.value, ply, pos.rule50()) } else { VALUE_NONE };
        let tt_move = if tt_hit { tt.mv } else { Move::NONE };
        let pv_hit = tt_hit && tt.is_pv;

        if !pv_node && tt_hit && tt.depth >= DEPTH_QS && is_valid(tt_value) && (if tt_value >= beta { tt.bound.has_lower() } else { tt.bound.has_upper() }) {
            return tt_value;
        }

        let mut best_value;
        let futility_base;
        let mut unadjusted_eval = VALUE_NONE;
        if in_check {
            best_value = -VALUE_INFINITE;
            futility_base = -VALUE_INFINITE;
            self.ss(ply).static_eval = VALUE_NONE;
        } else {
            let cv = self.corr_value(pos, ply);
            if tt_hit {
                unadjusted_eval = if is_valid(tt.eval) { tt.eval } else { self.evaluate(pos) };
                best_value = self.corrected_eval(unadjusted_eval, cv);
                self.ss(ply).static_eval = best_value;
                if is_valid(tt_value) && (if tt_value > best_value { tt.bound.has_lower() } else { tt.bound.has_upper() }) {
                    best_value = tt_value;
                }
            } else {
                unadjusted_eval = self.evaluate(pos);
                best_value = self.corrected_eval(unadjusted_eval, cv);
                self.ss(ply).static_eval = best_value;
            }
            if best_value >= beta {
                if !is_decisive(best_value) {
                    best_value = (best_value + beta) / 2;
                }
                if !tt_hit {
                    self.shared.tt.save(&writer, key, value_to_tt(best_value, ply), false, Bound::Lower, DEPTH_UNSEARCHED, Move::NONE, unadjusted_eval);
                }
                return best_value;
            }
            if best_value > alpha {
                alpha = best_value;
            }
            futility_base = self.ss_at(ply).static_eval + 306;
        }

        let cont_idx = [
            if ply >= 1 { self.ss_prev(ply, 1).cont_idx } else { usize::MAX },
            if ply >= 2 { self.ss_prev(ply, 2).cont_idx } else { usize::MAX },
            usize::MAX,
            usize::MAX,
        ];
        let prev_sq = if ply >= 1 && !self.ss_prev(ply, 1).is_null { self.ss_prev(ply, 1).current_move.to() } else { SQ_NONE };
        let mut mp = MovePicker::new_qsearch(pos, tt_move, cont_idx, ply);
        let mut best_move = Move::NONE;
        let mut move_count = 0;

        loop {
            let m = mp.next(pos, &self.hist);
            if m.is_none() {
                break;
            }
            if !pos.is_legal(m) {
                continue;
            }
            let gives_check = pos.gives_check(m);
            let capture = pos.is_capture_or_promo(m);
            move_count += 1;

            if !is_loss(best_value) {
                if !gives_check && m.to() != prev_sq && !is_loss(futility_base) && !m.is_promo() {
                    if move_count > 2 {
                        continue;
                    }
                    let captured = pos.captured_type(m).map(piece_value).unwrap_or(0);
                    let fut = futility_base + captured;
                    if fut <= alpha {
                        best_value = best_value.max(fut);
                        continue;
                    }
                    if futility_base <= alpha && !pos.see_ge(m, 1) {
                        best_value = best_value.max(futility_base);
                        continue;
                    }
                }
                if !pos.see_ge(m, -74) {
                    continue;
                }
            }

            let moved_piece = pos.moved_piece(m);
            {
                let s = self.ss(ply);
                s.current_move = m;
                s.moved_piece = moved_piece;
                s.cont_idx = History::cont_index(in_check, capture, moved_piece, m.to());
                s.cont_corr_idx = moved_piece.idx() * 64 + m.to() as usize;
            }
            let child = pos.make_move(m);
            self.shared.tt.prefetch(child.key());
            self.nnue.push(pos, m, &child);
            self.keys.push(key);
            self.nodes += 1;
            let value = -self.qsearch(&child, pv_node, -beta, -alpha, ply + 1);
            self.keys.pop();
            self.nnue.pop();

            if self.should_stop() {
                return 0;
            }
            if value > best_value {
                best_value = value;
                if value > alpha {
                    best_move = m;
                    if pv_node {
                        self.update_pv(ply, m);
                    }
                    if value < beta {
                        alpha = value;
                    } else {
                        break;
                    }
                }
            }
        }

        if in_check && best_value == -VALUE_INFINITE {
            return mated_in(ply);
        }
        if !is_decisive(best_value) && best_value > beta {
            best_value = (best_value + beta) / 2;
        }
        let bound = if best_value >= beta { Bound::Lower } else { Bound::Upper };
        self.shared.tt.save(&writer, key, value_to_tt(best_value, ply), pv_hit, bound, DEPTH_QS, best_move, unadjusted_eval);
        best_value
    }

    // -----------------------------------------------------------------------------------------
    // Iterative deepening
    // -----------------------------------------------------------------------------------------
    pub fn iterative_deepening(&mut self, root: &Position, game_keys: &[u64]) {
        self.keys = game_keys.to_vec();
        self.root_key_idx = self.keys.len();
        self.nodes = 0;
        self.nmp_min_ply = 0;
        self.best_move_changes = 0.0;
        self.completed_depth = 0;
        if let Some(net) = self.net {
            self.nnue.reset(root, net);
        }
        for s in self.ss.iter_mut() {
            *s = StackEntry::default();
        }

        // Root moves.
        let legal = legal_moves(root);
        self.root_moves.clear();
        for m in legal.iter() {
            if !self.limits.go.searchmoves.is_empty() {
                let s = root.move_to_uci(m);
                if !self.limits.go.searchmoves.iter().any(|x| *x == s) {
                    continue;
                }
            }
            self.root_moves.push(RootMove { mv: m, score: -VALUE_INFINITE, prev_score: -VALUE_INFINITE, avg_score: -VALUE_INFINITE, sel_depth: 0, pv: vec![m], effort: 0 });
        }
        if self.root_moves.is_empty() {
            return;
        }
        // Syzygy root filtering: keep only moves that preserve the best tablebase result.
        if crate::tb::largest() > 0 && self.is_main() || crate::tb::largest() > 0 {
            if let Some(res) = crate::tb::probe_root(root) {
                if !res.is_empty() {
                    self.shared.tb_hits.fetch_add(1, Ordering::Relaxed);
                    let best_wdl = res.iter().map(|r| r.1).max().unwrap();
                    // Among winning moves prefer the smallest DTZ (fastest progress); among losing
                    // moves the largest.
                    let mut keep: Vec<Move> = Vec::new();
                    if best_wdl >= crate::tb::TB_CURSED_WIN {
                        let min_dtz = res.iter().filter(|r| r.1 == best_wdl).map(|r| r.2).min().unwrap();
                        keep.extend(res.iter().filter(|r| r.1 == best_wdl && r.2 <= min_dtz + 4).map(|r| r.0));
                    } else {
                        keep.extend(res.iter().filter(|r| r.1 == best_wdl).map(|r| r.0));
                    }
                    let filtered: Vec<RootMove> = self.root_moves.iter().filter(|rm| keep.contains(&rm.mv)).cloned().collect();
                    if !filtered.is_empty() {
                        self.root_moves = filtered;
                    }
                }
            }
        }
        let multi_pv = self.multi_pv.min(self.root_moves.len());
        let max_depth = if self.limits.max_depth > 0 { self.limits.max_depth.min(MAX_PLY as i32 - 1) } else { MAX_PLY as i32 - 1 };
        let mut last_info_depth = 0;
        let mut time_reduction = 1.0f64;
        let mut last_best_move_depth = 0;
        let mut last_best = Move::NONE;
        self.opt_time = self.limits.tm.optimum;

        let mut depth = 1;
        while depth <= max_depth {
            self.root_depth = depth;
            if self.should_stop() {
                break;
            }
            for rm in self.root_moves.iter_mut() {
                rm.prev_score = rm.score;
            }
            let mut pv_idx = 0;
            while pv_idx < multi_pv && !self.should_stop() {
                self.pv_idx = pv_idx;
                self.sel_depth = 0;
                let avg = self.root_moves[pv_idx].avg_score;
                let mut delta = 10 + if avg == -VALUE_INFINITE { 0 } else { (avg * avg / 11131).min(400) };
                let (mut alpha, mut beta) = if avg == -VALUE_INFINITE || depth < 4 {
                    (-VALUE_INFINITE, VALUE_INFINITE)
                } else {
                    ((avg - delta).max(-VALUE_INFINITE), (avg + delta).min(VALUE_INFINITE))
                };
                let mut failed_high_cnt = 0;
                loop {
                    let adjusted_depth = (depth - failed_high_cnt).max(1);
                    self.root_delta = beta - alpha;
                    let best_value = self.search(root, NodeType::Root, alpha, beta, adjusted_depth, false, 0);
                    // Stable sort root moves [pv_idx..] by score.
                    self.root_moves[pv_idx..].sort_by(|a, b| b.score.cmp(&a.score));
                    if self.should_stop() {
                        break;
                    }
                    if best_value <= alpha {
                        beta = (alpha + beta) / 2;
                        alpha = (best_value - delta).max(-VALUE_INFINITE);
                        failed_high_cnt = 0;
                    } else if best_value >= beta {
                        beta = (best_value + delta).min(VALUE_INFINITE);
                        failed_high_cnt += 1;
                    } else {
                        break;
                    }
                    delta += delta / 3;
                    if self.is_main() && !self.silent && self.limits.tm.elapsed_ms() > 3000.0 {
                        self.print_info(root, depth, pv_idx, alpha, beta);
                    }
                }
                self.root_moves[..=pv_idx].sort_by(|a, b| b.score.cmp(&a.score));
                // Update running averages.
                for rm in self.root_moves.iter_mut() {
                    if rm.score != -VALUE_INFINITE {
                        rm.avg_score = if rm.avg_score == -VALUE_INFINITE { rm.score } else { (2 * rm.score + rm.avg_score) / 3 };
                    }
                }
                if self.is_main() && !self.silent && (self.should_stop() || pv_idx + 1 == multi_pv || self.limits.tm.elapsed_ms() > 3000.0) {
                    self.print_info(root, depth, pv_idx, -VALUE_INFINITE, VALUE_INFINITE);
                    last_info_depth = depth;
                }
                pv_idx += 1;
            }

            if !self.should_stop() {
                self.completed_depth = depth;
            }
            let best = self.root_moves[0].mv;
            if best != last_best {
                last_best = best;
                last_best_move_depth = depth;
            }

            // Mate limit.
            if let Some(mate) = self.limits.go.mate {
                let sc = self.root_moves[0].score;
                if sc >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - sc <= 2 * mate {
                    break;
                }
            }

            // ---- Time management ----
            if self.is_main() && self.limits.tm.use_time && !self.limits.tm.fixed && !self.should_stop() && !self.limits.go.infinite {
                let best_value = self.root_moves[0].score;
                let prev_iter = self.iter_value[0];
                // Stockfish: fallingEval = (66 + 14*(prevBest - now) + 6*(prevIter - now)) / 616 clamped; with the
                // previous-search score unknown (first search of a game) the first term is zero.
                let prev_best_term = if self.best_prev_score == VALUE_INFINITE {
                    0.0
                } else {
                    14.0 * (self.best_prev_score as f64 - best_value as f64).clamp(-200.0, 200.0)
                };
                let prev_iter_term = if depth <= 1 { 0.0 } else { 6.0 * (prev_iter as f64 - best_value as f64).clamp(-200.0, 200.0) };
                let falling_eval = if crate::params::TM_FALLING_FIX.get() != 0 {
                    ((66.0 + prev_best_term + prev_iter_term) / 616.0).clamp(0.51, 1.51)
                } else {
                    // old (buggy) behaviour kept for A/B: saturates at 1.51
                    1.51
                };
                time_reduction = if last_best_move_depth + 8 <= depth { 1.56 } else { 0.69 };
                let reduction = (1.4 + self.prev_time_reduction) / (2.2 * time_reduction);
                let instability = 1.0 + crate::params::TM_INSTAB_PCT.get() as f64 / 100.0 * self.best_move_changes / self.threads as f64;
                let effort_ratio = if self.nodes > 0 { self.root_moves[0].effort as f64 / self.nodes as f64 } else { 0.0 };
                let effort_scale = if effort_ratio > 0.9 { 0.9 } else if effort_ratio > 0.8 { 0.95 } else { 1.0 };
                let total = self.limits.tm.optimum * crate::params::TM_OPT_PCT.get() as f64 / 100.0 * falling_eval * reduction * instability * effort_scale;
                let elapsed = self.limits.tm.elapsed_ms();
                if elapsed > total {
                    self.shared.stop.store(true, Ordering::Relaxed);
                }
                self.iter_value = [best_value, self.iter_value[0], self.iter_value[1], self.iter_value[2]];
                self.best_move_changes /= 2.0;
                self.prev_time_reduction = time_reduction;
            }
            if self.limits.max_depth > 0 && depth >= self.limits.max_depth {
                break;
            }
            depth += 1;
        }
        let _ = last_info_depth;
        self.last_best_move = self.root_moves[0].mv;
        self.last_best_depth = self.completed_depth;
        self.prev_time_reduction = time_reduction;
    }

    fn print_info(&self, root: &Position, depth: i32, pv_idx: usize, alpha: Value, beta: Value) {
        let elapsed = self.limits.tm.elapsed_ms().max(1.0);
        let nodes = self.total_nodes();
        let hashfull = self.shared.tt.hashfull();
        for i in 0..=pv_idx.min(self.root_moves.len() - 1) {
            let rm = &self.root_moves[i];
            if rm.score == -VALUE_INFINITE && i > 0 {
                continue;
            }
            let v = rm.score;
            let score = if v.abs() >= VALUE_MATE_IN_MAX_PLY {
                let plies = if v > 0 { VALUE_MATE - v } else { -VALUE_MATE - v };
                let moves = if plies > 0 { (plies + 1) / 2 } else { (plies - 1) / 2 };
                format!("mate {}", moves)
            } else {
                format!("cp {}", v)
            };
            let bound = if i == pv_idx {
                if v >= beta {
                    " lowerbound"
                } else if v <= alpha {
                    " upperbound"
                } else {
                    ""
                }
            } else {
                ""
            };
            let pv: Vec<String> = rm.pv.iter().map(|m| root.move_to_uci(*m)).collect();
            println!(
                "info depth {} seldepth {} multipv {} score {}{} nodes {} nps {} hashfull {} tbhits {} time {} pv {}",
                depth,
                rm.sel_depth,
                i + 1,
                score,
                bound,
                nodes,
                (nodes as f64 / elapsed * 1000.0) as u64,
                hashfull,
                self.shared.tb_hits.load(Ordering::Relaxed),
                elapsed as u64,
                pv.join(" ")
            );
        }
    }
}

const VALUE_ZERO_INIT: Value = 0;

/// Result of a search.
pub struct SearchResult {
    pub best_move: Move,
    pub ponder_move: Move,
    pub score: Value,
    pub nodes: u64,
    pub depth: i32,
}

/// Run a search with `opts.threads` threads. Returns the main thread's result. `hists` holds
/// per-thread persistent histories (taken and put back so they survive between searches).
pub fn go(
    root: &Position,
    game_keys: &[u64],
    shared: &Shared,
    net: Option<&AnyNet>,
    limits: &Limits,
    opts: &Options,
    hists: &mut Vec<History>,
) -> SearchResult {
    shared.stop.store(false, Ordering::Relaxed);
    for n in shared.nodes.iter() {
        n.store(0, Ordering::Relaxed);
    }
    shared.tt.new_search();
    shared.tb_hits.store(0, Ordering::Relaxed);
    let threads = opts.threads.max(1);
    while hists.len() < threads {
        hists.push(History::new());
    }
    let taken: Vec<History> = hists.drain(..threads).collect();
    let results: Vec<(SearchResult, History)> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (id, hist) in taken.into_iter().enumerate() {
            let limits = limits.clone();
            let root = *root;
            let keys = game_keys.to_vec();
            handles.push(
                std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn_scoped(s, move || {
                        let mut t = Thread::new(id, shared, net, limits, opts, hist);
                        t.iterative_deepening(&root, &keys);
                        if id == 0 {
                            shared.stop.store(true, Ordering::Relaxed);
                        }
                        let (best, ponder, score) = if t.root_moves.is_empty() {
                            (Move::NONE, Move::NONE, VALUE_DRAW)
                        } else {
                            let rm = &t.root_moves[0];
                            (rm.mv, rm.pv.get(1).copied().unwrap_or(Move::NONE), rm.score)
                        };
                        let res = SearchResult { best_move: best, ponder_move: ponder, score, nodes: t.nodes, depth: t.completed_depth };
                        (res, t.hist)
                    })
                    .unwrap(),
            );
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = None;
    let mut total_nodes = 0;
    for (id, (res, hist)) in results.into_iter().enumerate() {
        total_nodes += res.nodes;
        hists.insert(id, hist);
        if id == 0 {
            out = Some(res);
        }
    }
    let mut res = out.unwrap();
    res.nodes = total_nodes;
    res
}

impl Shared {
    pub fn new(hash_mb: usize, threads: usize) -> Arc<Shared> {
        Arc::new(Shared {
            tt: TranspositionTable::new(hash_mb),
            stop: AtomicBool::new(false),
            nodes: (0..threads.max(1)).map(|_| AtomicU64::new(0)).collect(),
            tb_hits: AtomicU64::new(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::START_FEN;

    fn run(fen: &str, depth: i32) -> SearchResult {
        crate::init();
        let pos = Position::from_fen(fen).unwrap();
        let shared = Shared::new(8, 1);
        let params = GoParams { depth: Some(depth), ..Default::default() };
        let tm = TimeManager::new(&params, true, 0, 0);
        let limits = Limits { go: params, tm, max_depth: depth, max_nodes: 0 };
        let opts = Options { threads: 1, multi_pv: 1, move_overhead: 0, chess960: false, silent: true, prev_score: VALUE_INFINITE };
        let mut hists = Vec::new();
        super::go(&pos, &[], &shared, None, &limits, &opts, &mut hists)
    }

    #[test]
    fn finds_mates() {
        // Mate in 1, 2 and 3 (well-known puzzles).
        let r = run("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 6);
        assert_eq!(r.score, mate_in(1), "mate in 1");
        let r = run("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 10", 8);
        assert_eq!(r.score, mate_in(3), "mate in 2 (3 plies)");
        assert_eq!(r.best_move.to_string(), "d5f6");
        let r = run("r1b1kb1r/pppp1ppp/5q2/4n3/3KP3/2N3PN/PPP4P/R1BQ1B1R b kq - 0 1", 8);
        assert_eq!(r.score, mate_in(5), "mate in 3 (5 plies)");
    }

    #[test]
    fn detects_stalemate_and_mated() {
        let r = run("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 4);
        assert_eq!(r.score, VALUE_DRAW);
        assert!(r.best_move.is_none());
        let r = run("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 4);
        assert!(r.best_move.is_none());
    }

    #[test]
    fn bench_is_deterministic() {
        let a = run(START_FEN, 9).nodes;
        let b = run(START_FEN, 9).nodes;
        assert_eq!(a, b);
        assert!(a > 500);
    }

    #[test]
    fn multithread_search_runs() {
        crate::init();
        let pos = Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3").unwrap();
        let shared = Shared::new(16, 4);
        let go = GoParams { depth: Some(10), ..Default::default() };
        let tm = TimeManager::new(&go, true, 0, 0);
        let limits = Limits { go, tm, max_depth: 10, max_nodes: 0 };
        let opts = Options { threads: 4, multi_pv: 1, move_overhead: 0, chess960: false, silent: true, prev_score: VALUE_INFINITE };
        let mut hists = Vec::new();
        let r = super::go(&pos, &[], &shared, None, &limits, &opts, &mut hists);
        assert!(!r.best_move.is_none());
        assert_eq!(hists.len(), 4);
    }

    #[test]
    fn repetition_is_draw() {
        // Perpetual check position: White to move can only draw by repetition.
        crate::init();
        let pos = Position::from_fen("6k1/5p1p/6p1/8/8/8/5PPP/3q1RK1 b - - 0 1").unwrap();
        // Black queen checks; play out a few repetitions through the game keys path.
        let mut p = pos;
        let mut keys = Vec::new();
        for m in ["d1d4", "g1h1", "d4a1", "h1g1"] {
            let mv = p.parse_uci_move(m).unwrap();
            keys.push(p.key());
            p = p.make_move(mv);
        }
        let shared = Shared::new(8, 1);
        let go = GoParams { depth: Some(8), ..Default::default() };
        let tm = TimeManager::new(&go, true, 0, 0);
        let limits = Limits { go, tm, max_depth: 8, max_nodes: 0 };
        let opts = Options { threads: 1, multi_pv: 1, move_overhead: 0, chess960: false, silent: true, prev_score: VALUE_INFINITE };
        let mut hists = Vec::new();
        let r = super::go(&p, &keys, &shared, None, &limits, &opts, &mut hists);
        // Black is a queen up: it must not see a draw score from repetition unless forced.
        assert!(r.score > 300, "score {}", r.score);
    }
}
