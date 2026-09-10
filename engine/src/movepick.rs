//! Staged move picker.

use crate::history::History;
use crate::movegen::*;
use crate::position::Position;
use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    TT,
    GenCaptures,
    GoodCaptures,
    Killer1,
    Killer2,
    Counter,
    GenQuiets,
    Quiets,
    BadCaptures,
    // evasions
    EvasionTT,
    GenEvasions,
    Evasions,
    // qsearch
    QsTT,
    QsGenCaptures,
    QsCaptures,
    // probcut
    ProbcutTT,
    ProbcutGen,
    Probcut,
    Done,
}

pub struct MovePicker {
    stage: Stage,
    tt_move: Move,
    killers: [Move; 2],
    counter: Move,
    list: MoveList,
    bad_captures: MoveList,
    cur: usize,
    end: usize,
    threshold: Value,
    depth: i32,
    ply: usize,
    skip_quiets: bool,
    /// Continuation history indices for plies 1,2,4,6 back (usize::MAX if unavailable).
    cont_idx: [usize; 4],
}

pub const QUIET_LEFT_MARGIN: i32 = -3560;

impl MovePicker {
    /// Main search picker.
    pub fn new(pos: &Position, tt_move: Move, killers: [Move; 2], counter: Move, cont_idx: [usize; 4], depth: i32, ply: usize) -> Self {
        let tt_ok = !tt_move.is_none() && pos.is_pseudo_legal(tt_move);
        let stage = if pos.in_check() {
            if tt_ok { Stage::EvasionTT } else { Stage::GenEvasions }
        } else if tt_ok {
            Stage::TT
        } else {
            Stage::GenCaptures
        };
        MovePicker {
            stage,
            tt_move: if tt_ok { tt_move } else { Move::NONE },
            killers,
            counter,
            list: MoveList::new(),
            bad_captures: MoveList::new(),
            cur: 0,
            end: 0,
            threshold: 0,
            depth,
            ply,
            skip_quiets: false,
            cont_idx,
        }
    }

    /// Quiescence picker: captures (and queen promotions) only, unless in check.
    pub fn new_qsearch(pos: &Position, tt_move: Move, cont_idx: [usize; 4], ply: usize) -> Self {
        let tt_ok = !tt_move.is_none() && pos.is_pseudo_legal(tt_move) && (pos.in_check() || pos.is_capture_or_promo(tt_move));
        let stage = if pos.in_check() {
            if tt_ok { Stage::EvasionTT } else { Stage::GenEvasions }
        } else if tt_ok {
            Stage::QsTT
        } else {
            Stage::QsGenCaptures
        };
        MovePicker {
            stage,
            tt_move: if tt_ok { tt_move } else { Move::NONE },
            killers: [Move::NONE; 2],
            counter: Move::NONE,
            list: MoveList::new(),
            bad_captures: MoveList::new(),
            cur: 0,
            end: 0,
            threshold: 0,
            depth: 0,
            ply,
            skip_quiets: true,
            cont_idx,
        }
    }

    /// ProbCut picker: captures with SEE >= threshold.
    pub fn new_probcut(pos: &Position, tt_move: Move, threshold: Value) -> Self {
        let tt_ok = !tt_move.is_none() && pos.is_pseudo_legal(tt_move) && pos.is_capture(tt_move) && pos.see_ge(tt_move, threshold);
        MovePicker {
            stage: if tt_ok { Stage::ProbcutTT } else { Stage::ProbcutGen },
            tt_move: if tt_ok { tt_move } else { Move::NONE },
            killers: [Move::NONE; 2],
            counter: Move::NONE,
            list: MoveList::new(),
            bad_captures: MoveList::new(),
            cur: 0,
            end: 0,
            threshold,
            depth: 0,
            ply: 0,
            skip_quiets: true,
            cont_idx: [usize::MAX; 4],
        }
    }

    pub fn skip_quiets(&mut self) {
        self.skip_quiets = true;
    }

    fn score_captures(&mut self, pos: &Position, hist: &History) {
        for i in self.cur..self.end {
            let m = self.list.moves[i].mv;
            let captured = pos.captured_type(m).unwrap_or(PieceType::Pawn);
            let pc = pos.moved_piece(m);
            let mut s = 7 * piece_value(captured) + hist.capture_get(pc, m.to(), captured);
            if m.is_promo() {
                s += piece_value(m.promo_type()) * 4;
            }
            self.list.moves[i].score = s;
        }
    }

    fn score_quiets(&mut self, pos: &Position, hist: &History) {
        let us = pos.side_to_move();
        for i in self.cur..self.end {
            let m = self.list.moves[i].mv;
            let pc = pos.moved_piece(m);
            let to = m.to();
            let mut s = 2 * hist.main_get(us, m);
            s += 2 * hist.pawn_get(pos.pawn_key(), pc, to);
            for (k, &ci) in self.cont_idx.iter().enumerate() {
                if ci != usize::MAX {
                    let w = [2, 2, 1, 1][k];
                    s += w * hist.cont_get(ci, pc, to);
                }
            }
            s += hist.low_ply_get(self.ply, m) * 2;
            if pos.check_squares(pc.piece_type()) & crate::types::bb(to) != 0 {
                s += 4000;
            }
            self.list.moves[i].score = s;
        }
    }

    fn score_evasions(&mut self, pos: &Position, hist: &History) {
        let us = pos.side_to_move();
        for i in self.cur..self.end {
            let m = self.list.moves[i].mv;
            let pc = pos.moved_piece(m);
            self.list.moves[i].score = if pos.is_capture(m) {
                let captured = pos.captured_type(m).unwrap();
                piece_value(captured) + (1 << 28)
            } else {
                let mut s = hist.main_get(us, m) + hist.pawn_get(pos.pawn_key(), pc, m.to());
                if self.cont_idx[0] != usize::MAX {
                    s += hist.cont_get(self.cont_idx[0], pc, m.to());
                }
                s
            };
        }
    }

    /// Selection sort step: pick best in [cur, end).
    #[inline]
    fn pick_best(&mut self) -> Move {
        let mut best = self.cur;
        for i in self.cur + 1..self.end {
            if self.list.moves[i].score > self.list.moves[best].score {
                best = i;
            }
        }
        self.list.moves.swap(self.cur, best);
        let m = self.list.moves[self.cur].mv;
        self.cur += 1;
        m
    }

    /// Partial insertion sort for quiets above a limit (Stockfish style).
    fn sort_quiets(&mut self, limit: i32) {
        let n = self.end;
        let mut sorted_end = self.cur;
        for i in self.cur..n {
            if self.list.moves[i].score >= limit {
                let tmp = self.list.moves[i];
                self.list.moves[i] = self.list.moves[sorted_end];
                let mut j = sorted_end;
                while j > self.cur && self.list.moves[j - 1].score < tmp.score {
                    self.list.moves[j] = self.list.moves[j - 1];
                    j -= 1;
                }
                self.list.moves[j] = tmp;
                sorted_end += 1;
            }
        }
    }

    /// Next pseudo-legal move (caller must check legality), or NONE.
    pub fn next(&mut self, pos: &Position, hist: &History) -> Move {
        loop {
            match self.stage {
                Stage::TT | Stage::EvasionTT | Stage::QsTT | Stage::ProbcutTT => {
                    self.stage = match self.stage {
                        Stage::TT => Stage::GenCaptures,
                        Stage::EvasionTT => Stage::GenEvasions,
                        Stage::QsTT => Stage::QsGenCaptures,
                        _ => Stage::ProbcutGen,
                    };
                    return self.tt_move;
                }
                Stage::GenCaptures | Stage::QsGenCaptures | Stage::ProbcutGen => {
                    self.list.clear();
                    generate(pos, &mut self.list, GenType::Captures);
                    self.cur = 0;
                    self.end = self.list.len();
                    self.score_captures(pos, hist);
                    self.stage = match self.stage {
                        Stage::GenCaptures => Stage::GoodCaptures,
                        Stage::QsGenCaptures => Stage::QsCaptures,
                        _ => Stage::Probcut,
                    };
                }
                Stage::GoodCaptures => {
                    while self.cur < self.end {
                        let m = self.pick_best();
                        if m == self.tt_move {
                            continue;
                        }
                        // Good capture if SEE >= -score/32 (SF: -capture score based threshold).
                        let thr = -self.list.moves[self.cur - 1].score / 32;
                        if pos.see_ge(m, thr.min(0)) {
                            return m;
                        }
                        self.bad_captures.push(m);
                    }
                    self.stage = Stage::Killer1;
                }
                Stage::Killer1 => {
                    self.stage = Stage::Killer2;
                    let m = self.killers[0];
                    if !self.skip_quiets && !m.is_none() && m != self.tt_move && pos.is_pseudo_legal(m) && !pos.is_capture_or_promo(m) {
                        return m;
                    }
                }
                Stage::Killer2 => {
                    self.stage = Stage::Counter;
                    let m = self.killers[1];
                    if !self.skip_quiets && !m.is_none() && m != self.tt_move && m != self.killers[0] && pos.is_pseudo_legal(m) && !pos.is_capture_or_promo(m) {
                        return m;
                    }
                }
                Stage::Counter => {
                    self.stage = Stage::GenQuiets;
                    let m = self.counter;
                    if !self.skip_quiets
                        && !m.is_none()
                        && m != self.tt_move
                        && m != self.killers[0]
                        && m != self.killers[1]
                        && pos.is_pseudo_legal(m)
                        && !pos.is_capture_or_promo(m)
                    {
                        return m;
                    }
                }
                Stage::GenQuiets => {
                    if !self.skip_quiets {
                        self.list.clear();
                        generate(pos, &mut self.list, GenType::Quiets);
                        self.cur = 0;
                        self.end = self.list.len();
                        self.score_quiets(pos, hist);
                        self.sort_quiets(QUIET_LEFT_MARGIN - 3130 * self.depth);
                    } else {
                        self.cur = 0;
                        self.end = 0;
                    }
                    self.stage = Stage::Quiets;
                }
                Stage::Quiets => {
                    if !self.skip_quiets {
                        while self.cur < self.end {
                            let m = self.list.moves[self.cur].mv;
                            self.cur += 1;
                            if m != self.tt_move && m != self.killers[0] && m != self.killers[1] && m != self.counter {
                                return m;
                            }
                        }
                    }
                    self.stage = Stage::BadCaptures;
                    self.cur = 0;
                }
                Stage::BadCaptures => {
                    if self.cur < self.bad_captures.len() {
                        let m = self.bad_captures.moves[self.cur].mv;
                        self.cur += 1;
                        return m;
                    }
                    self.stage = Stage::Done;
                }
                Stage::GenEvasions => {
                    self.list.clear();
                    generate(pos, &mut self.list, GenType::Evasions);
                    self.cur = 0;
                    self.end = self.list.len();
                    self.score_evasions(pos, hist);
                    self.stage = Stage::Evasions;
                }
                Stage::Evasions => {
                    while self.cur < self.end {
                        let m = self.pick_best();
                        if m != self.tt_move {
                            return m;
                        }
                    }
                    self.stage = Stage::Done;
                }
                Stage::QsCaptures => {
                    while self.cur < self.end {
                        let m = self.pick_best();
                        if m != self.tt_move {
                            return m;
                        }
                    }
                    self.stage = Stage::Done;
                }
                Stage::Probcut => {
                    while self.cur < self.end {
                        let m = self.pick_best();
                        if m != self.tt_move && pos.see_ge(m, self.threshold) {
                            return m;
                        }
                    }
                    self.stage = Stage::Done;
                }
                Stage::Done => return Move::NONE,
            }
        }
    }
}
