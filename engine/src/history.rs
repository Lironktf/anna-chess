//! Per-thread move-ordering and evaluation-correction statistics.

use crate::types::*;

pub const HIST_MAX: i32 = 8192; // butterfly / capture history saturation
pub const CONT_MAX: i32 = 8192;
pub const PAWN_HIST_SIZE: usize = 512;
pub const CORR_SIZE: usize = 16384;
pub const CORR_GRAIN: i32 = 256;
pub const CORR_MAX: i32 = 256 * 64; // +-64 cp of correction
pub const LOW_PLY_SIZE: usize = 5;

#[inline(always)]
fn gravity(entry: &mut i16, bonus: i32, max: i32) {
    let b = bonus.clamp(-max, max);
    let v = *entry as i32;
    *entry = (v + b - v * b.abs() / max) as i16;
}

/// [piece 13][to 64] table (13 = includes Piece::None for null moves).
pub type PieceTo = [[i16; 64]; 13];

pub struct History {
    /// Butterfly history [color * 4 + threat_index][from_to]; threat_index is 0 unless ThreatHist is on.
    pub main: Vec<[i16; 4096]>,
    /// Low-ply history [ply][from_to].
    pub low_ply: Vec<[i16; 4096]>,
    /// Capture history [piece][to][captured type].
    pub capture: Vec<[[i16; 6]; 64]>,
    /// Continuation history [in_check][capture][piece][to] -> PieceTo.
    pub cont: Vec<PieceTo>,
    /// Pawn-structure history [pawn_key % SIZE][piece][to].
    pub pawn: Vec<PieceTo>,
    /// Killers [ply][2].
    pub killers: Vec<[Move; 2]>,
    /// Counter moves [piece][to].
    pub counter: Vec<[Move; 64]>,
    // Correction histories: i32 entries scaled by CORR_GRAIN, index [color * CORR_SIZE + key % CORR_SIZE].
    pub pawn_corr: Vec<i32>,
    pub minor_corr: Vec<i32>,
    pub major_corr: Vec<i32>,
    /// [stm][piece color] -> CORR_SIZE entries: index ((stm*2 + color) * CORR_SIZE + key % CORR_SIZE).
    pub non_pawn_corr: Vec<i32>,
    /// Continuation correction [prev piece*64+to][piece*64+to].
    pub cont_corr: Vec<i32>,
    /// TT move history (how often the TT move failed low / high).
    pub tt_move_history: i32,
    // Stockfish-master correction histories (SfCorr): i16 gravity entries with limit SF_CORR_LIMIT, indexed like the
    // tables above (pawn/minor: [stm][key]; non-pawn: [stm][piece colour][key]; continuation: [prev piece*64+to][piece*64+to]).
    pub sf_pawn_corr: Vec<i16>,
    pub sf_minor_corr: Vec<i16>,
    pub sf_non_pawn_corr: Vec<i16>,
    pub sf_cont_corr: Vec<i16>,
}
pub const SF_CORR_LIMIT: i32 = 1024;

impl History {
    pub fn new() -> Self {
        History {
            main: vec![[0; 4096]; 8],
            low_ply: vec![[0; 4096]; LOW_PLY_SIZE],
            capture: vec![[[0; 6]; 64]; 13],
            cont: vec![[[0; 64]; 13]; 2 * 2 * 13 * 64],
            pawn: vec![[[0; 64]; 13]; PAWN_HIST_SIZE],
            killers: vec![[Move::NONE; 2]; MAX_PLY + 8],
            counter: vec![[Move::NONE; 64]; 13],
            pawn_corr: vec![0; 2 * CORR_SIZE],
            minor_corr: vec![0; 2 * CORR_SIZE],
            major_corr: vec![0; 2 * CORR_SIZE],
            non_pawn_corr: vec![0; 4 * CORR_SIZE],
            cont_corr: vec![0; 13 * 64 * 13 * 64],
            tt_move_history: 0,
            sf_pawn_corr: vec![0; 2 * CORR_SIZE],
            sf_minor_corr: vec![0; 2 * CORR_SIZE],
            sf_non_pawn_corr: vec![0; 4 * CORR_SIZE],
            sf_cont_corr: vec![0; 13 * 64 * 13 * 64],
        }
    }

    pub fn clear(&mut self) {
        *self = History::new();
    }

    #[inline(always)]
    pub fn cont_index(in_check: bool, capture: bool, piece: Piece, to: Square) -> usize {
        (((in_check as usize) * 2 + capture as usize) * 13 + piece.idx()) * 64 + to as usize
    }

    /// 0..3: (from square attacked by the opponent) * 2 + (to square attacked). `threats` is 0 when
    /// the ThreatHist parameter is off, which reduces to the plain butterfly table.
    #[inline(always)]
    pub fn threat_index(m: Move, threats: u64) -> usize {
        (((threats >> m.from()) & 1) * 2 + ((threats >> m.to()) & 1)) as usize
    }
    #[inline(always)]
    pub fn main_get(&self, c: Color, m: Move, ti: usize) -> i32 {
        self.main[c.idx() * 4 + ti][m.from_to()] as i32
    }
    #[inline(always)]
    pub fn main_update(&mut self, c: Color, m: Move, ti: usize, bonus: i32) {
        gravity(&mut self.main[c.idx() * 4 + ti][m.from_to()], bonus, HIST_MAX);
    }
    #[inline(always)]
    pub fn low_ply_get(&self, ply: usize, m: Move) -> i32 {
        if ply < LOW_PLY_SIZE {
            self.low_ply[ply][m.from_to()] as i32
        } else {
            0
        }
    }
    #[inline(always)]
    pub fn low_ply_update(&mut self, ply: usize, m: Move, bonus: i32) {
        if ply < LOW_PLY_SIZE {
            gravity(&mut self.low_ply[ply][m.from_to()], bonus, HIST_MAX);
        }
    }
    #[inline(always)]
    pub fn capture_get(&self, piece: Piece, to: Square, captured: PieceType) -> i32 {
        self.capture[piece.idx()][to as usize][captured.idx()] as i32
    }
    #[inline(always)]
    pub fn capture_update(&mut self, piece: Piece, to: Square, captured: PieceType, bonus: i32) {
        gravity(&mut self.capture[piece.idx()][to as usize][captured.idx()], bonus, HIST_MAX);
    }
    #[inline(always)]
    pub fn cont_get(&self, idx: usize, piece: Piece, to: Square) -> i32 {
        self.cont[idx][piece.idx()][to as usize] as i32
    }
    #[inline(always)]
    pub fn cont_update(&mut self, idx: usize, piece: Piece, to: Square, bonus: i32) {
        gravity(&mut self.cont[idx][piece.idx()][to as usize], bonus, CONT_MAX);
    }
    #[inline(always)]
    pub fn pawn_get(&self, pawn_key: u64, piece: Piece, to: Square) -> i32 {
        self.pawn[(pawn_key as usize) % PAWN_HIST_SIZE][piece.idx()][to as usize] as i32
    }
    #[inline(always)]
    pub fn pawn_update(&mut self, pawn_key: u64, piece: Piece, to: Square, bonus: i32) {
        gravity(&mut self.pawn[(pawn_key as usize) % PAWN_HIST_SIZE][piece.idx()][to as usize], bonus, HIST_MAX);
    }
    #[inline(always)]
    /// TT-move history: Stockfish `StatsEntry<i16, 8192>` gravity update (bonus clamped to +-8192).
    pub fn ttm_update(&mut self, bonus: i32) {
        let b = bonus.clamp(-8192, 8192);
        let v = self.tt_move_history;
        self.tt_move_history = v + b - v * b.abs() / 8192;
    }
    pub fn update_killers(&mut self, ply: usize, m: Move) {
        let k = &mut self.killers[ply];
        if k[0] != m {
            k[1] = k[0];
            k[0] = m;
        }
    }
    #[inline(always)]
    pub fn clear_killers(&mut self, ply: usize) {
        self.killers[ply] = [Move::NONE; 2];
    }

    // ---- correction history ----
    /// Exponential moving average toward `diff` (in cp) with weight `w` out of 256.
    #[inline(always)]
    pub fn corr_update(entry: &mut i32, diff: i32, w: i32) {
        let v = *entry;
        *entry = ((v * (256 - w) + diff * CORR_GRAIN * w) / 256).clamp(-CORR_MAX, CORR_MAX);
    }
    /// Stockfish-style correction entry update (gravity toward the bonus, limit SF_CORR_LIMIT).
    #[inline(always)]
    pub fn sf_corr_update(entry: &mut i16, bonus: i32) {
        gravity(entry, bonus, SF_CORR_LIMIT);
    }
    #[inline(always)]
    pub fn corr_idx(c: Color, key: u64) -> usize {
        c.idx() * CORR_SIZE + (key as usize % CORR_SIZE)
    }
    #[inline(always)]
    pub fn non_pawn_idx(stm: Color, c: Color, key: u64) -> usize {
        (stm.idx() * 2 + c.idx()) * CORR_SIZE + (key as usize % CORR_SIZE)
    }
    #[inline(always)]
    pub fn cont_corr_idx(prev: usize, piece: Piece, to: Square) -> usize {
        prev * (13 * 64) + piece.idx() * 64 + to as usize
    }
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}

/// Stat bonus / malus formulas (Stockfish-like).
#[inline(always)]
pub fn stat_bonus(depth: i32) -> i32 {
    (150 * depth - 85).min(1337).max(0)
}
#[inline(always)]
pub fn stat_malus(depth: i32) -> i32 {
    (968 * depth - 235).min(2244).max(0)
}
