//! Threat and pawn-pair input features, ported exactly from bullet `examples/advanced/inputs.rs`
//! (rev 629ee50): `Threats` (Pawnocchio-style piece-attacks-piece features, 59808 of them) and
//! `PawnPawnInputs` (pairs of pawns at most one file apart, 4560). The combined sparse input is
//! `[0, 4560)` pawn pairs followed by `[4560, 4560 + 59808)` threats, per perspective.
//!
//! Everything here works on a *side-to-move-relative* board (`RelBoard`): the perspective's own
//! pieces are "white" and the board is flipped vertically for black, exactly like bulletformat.

use crate::bitboard::*;
use crate::position::Position;
use crate::types::*;

pub const TOTAL_PAIRS: usize = 96 * 95 / 2; // 4560
pub const MAX_PAIRS: usize = 16 * 15 / 2;
pub const MAX_THREATS: usize = 128;

// montyformat piece numbering used by the feature maths.
const PAWN: usize = 2;
const KNIGHT: usize = 3;
const BISHOP: usize = 4;
const ROOK: usize = 5;
const QUEEN: usize = 6;
const KING: usize = 7;

fn make_targets<const N: usize>(valid: [usize; N]) -> [usize; 12] {
    let mut targets = [usize::MAX; 12];
    for i in 0..N {
        targets[valid[i] - 2] = i;
        targets[valid[i] + 4] = i + N;
    }
    targets
}

struct PieceThreatsData {
    piece: usize,
    attacks: [u64; 64],
    indices: [usize; 64],
    targets: [usize; 12],
    count: usize,
}

impl PieceThreatsData {
    fn new<const N: usize>(piece: usize, valid: [usize; N], f: impl Fn(usize) -> u64) -> Self {
        let mut attacks = [0u64; 64];
        let mut indices = [0usize; 64];
        let mut count = 0;
        for sq in 0..64 {
            attacks[sq] = f(sq);
            indices[sq] = count;
            count += attacks[sq].count_ones() as usize;
        }
        Self { piece, attacks, indices, targets: make_targets(valid), count }
    }
    #[inline(always)]
    fn map(&self, src: usize, dest: usize, target: usize, offset: usize) -> Option<usize> {
        if self.targets[target] == usize::MAX || (dest > src && target % 6 == self.piece - 2) {
            return None;
        }
        let idx = self.indices[src] + (self.attacks[src] & ((1u64 << dest) - 1)).count_ones() as usize;
        Some(offset + self.targets[target] * self.count + idx)
    }
}

pub struct Threats {
    pawn_map: [usize; 12],
    non_pk_data: [PieceThreatsData; 4],
    offsets: [usize; 5],
}

const DIAGS: [u64; 15] = [
    0x0100_0000_0000_0000,
    0x0201_0000_0000_0000,
    0x0402_0100_0000_0000,
    0x0804_0201_0000_0000,
    0x1008_0402_0100_0000,
    0x2010_0804_0201_0000,
    0x4020_1008_0402_0100,
    0x8040_2010_0804_0201,
    0x0080_4020_1008_0402,
    0x0000_8040_2010_0804,
    0x0000_0080_4020_1008,
    0x0000_0000_8040_2010,
    0x0000_0000_0080_4020,
    0x0000_0000_0000_8040,
    0x0000_0000_0000_0080,
];

impl Threats {
    pub fn new() -> Self {
        let knight = PieceThreatsData::new(KNIGHT, [PAWN, KNIGHT, BISHOP, ROOK, QUEEN], |sq| {
            let n = 1u64 << sq;
            let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
            let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
            (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
        });
        let bishop = PieceThreatsData::new(BISHOP, [PAWN, KNIGHT, BISHOP, ROOK], |sq| {
            let rank = sq / 8;
            let file = sq % 8;
            DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
        });
        let rook = PieceThreatsData::new(ROOK, [PAWN, KNIGHT, BISHOP, ROOK], |sq| {
            let rank = sq / 8;
            let file = sq % 8;
            (0xFFu64 << (rank * 8)) ^ (0x0101_0101_0101_0101u64 << file)
        });
        let queen = PieceThreatsData::new(QUEEN, [PAWN, KNIGHT, BISHOP, ROOK, QUEEN], |sq| bishop.attacks[sq] | rook.attacks[sq]);
        let mut offsets = [4 * 84; 5];
        for (i, &cnt) in [10 * knight.count, 8 * bishop.count, 8 * rook.count, 10 * queen.count].iter().enumerate() {
            offsets[i + 1] = offsets[i] + cnt;
        }
        Self { pawn_map: make_targets([KNIGHT, ROOK]), non_pk_data: [knight, bishop, rook, queen], offsets }
    }

    pub fn num_inputs(&self) -> usize {
        2 * self.offsets[4]
    }

    /// Feature index (without the perspective offset) for `piece` on `src` attacking `target`
    /// (0..11, perspective-relative colour) on `dest`, both squares already perspective-relative.
    #[inline(always)]
    pub fn map_single(&self, piece: usize, src: usize, dest: usize, target: usize) -> Option<usize> {
        if piece == PAWN {
            if self.pawn_map[target] == usize::MAX {
                return None;
            }
            let id = if dest.abs_diff(src) == [9, 7][(dest > src) as usize] { 0 } else { 1 };
            let attack = 2 * (src % 8) + id - 1;
            Some(self.pawn_map[target] * 84 + (src / 8 - 1) * 14 + attack)
        } else {
            self.non_pk_data[piece - 3].map(src, dest, target, self.offsets[piece - 3])
        }
    }

    #[inline(always)]
    pub fn side_offset(&self, side: usize) -> usize {
        self.offsets[4] * side
    }
}

impl Default for Threats {
    fn default() -> Self {
        Self::new()
    }
}

/// Side-to-move-relative bitboards: [white(=perspective), black, P, N, B, R, Q, K].
#[derive(Clone, Copy)]
pub struct RelBoard(pub [u64; 8]);

impl RelBoard {
    /// Build the board as seen from `p`: `p`'s pieces are white; flipped vertically if `p` is black.
    pub fn from_position(pos: &Position, p: Color) -> RelBoard {
        let mut bbs = [0u64; 8];
        bbs[0] = pos.colored(Color::White);
        bbs[1] = pos.colored(Color::Black);
        for (i, pt) in PieceType::ALL.iter().enumerate() {
            bbs[2 + i] = pos.pieces(*pt);
        }
        if p == Color::Black {
            for b in bbs.iter_mut() {
                *b = b.swap_bytes();
            }
            bbs.swap(0, 1);
        }
        RelBoard(bbs)
    }
    fn flip_view(mut self) -> RelBoard {
        self.0.swap(0, 1);
        for b in self.0.iter_mut() {
            *b = b.swap_bytes();
        }
        self
    }
    fn normalize_hm(mut self) -> RelBoard {
        let ksq = (self.0[0] & self.0[7]).trailing_zeros();
        if ksq % 8 > 3 {
            for b in self.0.iter_mut() {
                *b = b.swap_bytes().reverse_bits();
            }
        }
        self
    }
}

#[inline(always)]
fn attacks_rel(piece: usize, sq: usize, side: usize, occ: u64) -> u64 {
    let s = sq as Square;
    match piece {
        PAWN => pawn_attacks(if side == 0 { Color::White } else { Color::Black }, s),
        KNIGHT => knight_attacks(s),
        BISHOP => bishop_attacks(s, occ),
        ROOK => rook_attacks(s, occ),
        QUEEN => queen_attacks(s, occ),
        _ => unreachable!(),
    }
}

/// Threat features of a relative board for both perspectives: (stm = "white" of the board, ntm).
pub fn map_threats(t: &Threats, bbs: &RelBoard, mut on_stm: impl FnMut(usize), mut on_ntm: impl FnMut(usize)) {
    let b = &bbs.0;
    let stm_king = (b[0] & b[7]).trailing_zeros() as usize;
    let ntm_king = (b[1] & b[7]).trailing_zeros() as usize;
    let stm_mask = if stm_king % 8 > 3 { 7 } else { 0 };
    let ntm_mask = 56 ^ if ntm_king % 8 > 3 { 7 } else { 0 };
    let mut pieces = [13usize; 64];
    for side in 0..2 {
        for piece in PAWN..=KING {
            for sq in bits(b[side] & b[piece]) {
                pieces[sq as usize] = 6 * side + piece - 2;
            }
        }
    }
    let occ = b[0] | b[1];
    for side in 0..2 {
        let stm_offset = t.side_offset(side);
        let ntm_offset = t.side_offset(side ^ 1);
        for piece in PAWN..KING {
            for sq in bits(b[side] & b[piece]) {
                let sq = sq as usize;
                let threats = attacks_rel(piece, sq, side, occ) & occ;
                for dest in bits(threats) {
                    let dest = dest as usize;
                    let target = pieces[dest];
                    if let Some(idx) = t.map_single(piece, sq ^ stm_mask, dest ^ stm_mask, target) {
                        on_stm(stm_offset + idx);
                    }
                    let ntm_target = (target + 6) % 12;
                    if let Some(idx) = t.map_single(piece, sq ^ ntm_mask, dest ^ ntm_mask, ntm_target) {
                        on_ntm(ntm_offset + idx);
                    }
                }
            }
        }
    }
}

pub fn three_file_band_mask() -> [u64; 64] {
    const A: u64 = 0x0101_0101_0101_0101;
    let mut masks = [0u64; 64];
    for (sq, mask) in masks.iter_mut().enumerate().take(56).skip(8) {
        let f = sq & 7;
        let mut m: u64 = A << f;
        if f > 0 {
            m |= A << (f - 1);
        }
        if f < 7 {
            m |= A << (f + 1);
        }
        *mask = m;
    }
    masks
}

#[inline(always)]
fn pawn_id(colour: usize, sq: usize) -> usize {
    colour * 48 + sq - 8
}
#[inline(always)]
pub fn pair_index(id_a: usize, id_b: usize) -> usize {
    let lo = id_a.min(id_b);
    let hi = id_a.max(id_b);
    hi * (hi - 1) / 2 + lo
}

fn emit_same_colour(masks: &[u64; 64], bb: u64, colour: usize, f: &mut impl FnMut(usize)) {
    let mut outer = bb;
    while outer != 0 {
        let sq_a = outer.trailing_zeros() as usize;
        outer &= outer - 1;
        let id_a = pawn_id(colour, sq_a);
        for sq_b in bits(outer & masks[sq_a]) {
            f(pair_index(id_a, pawn_id(colour, sq_b as usize)));
        }
    }
}

fn collect_pairs(masks: &[u64; 64], bbs: &RelBoard, f: &mut impl FnMut(usize)) {
    let friendly = bbs.0[0] & bbs.0[2];
    let enemy = bbs.0[1] & bbs.0[2];
    emit_same_colour(masks, friendly, 0, f);
    for sq_a in bits(friendly) {
        let sq_a = sq_a as usize;
        let id_a = pawn_id(0, sq_a);
        for sq_b in bits(enemy & masks[sq_a]) {
            f(pair_index(id_a, pawn_id(1, sq_b as usize)));
        }
    }
    emit_same_colour(masks, enemy, 1, f);
}

/// Full sparse feature set (pawn pairs + threats) for both perspectives of a relative board.
pub struct FeatureMapper {
    pub threats: Threats,
    pub masks: [u64; 64],
}

impl FeatureMapper {
    pub fn new() -> Self {
        FeatureMapper { threats: Threats::new(), masks: three_file_band_mask() }
    }
    pub fn num_inputs(&self) -> usize {
        TOTAL_PAIRS + self.threats.num_inputs()
    }
    /// Emit features for the perspective that owns the "white" pieces of `bbs` (on_stm) and the other (on_ntm).
    pub fn map_features(&self, bbs: &RelBoard, mut on_stm: impl FnMut(usize), mut on_ntm: impl FnMut(usize)) {
        map_threats(&self.threats, bbs, |s| on_stm(TOTAL_PAIRS + s), |n| on_ntm(TOTAL_PAIRS + n));
        collect_pairs(&self.masks, &bbs.normalize_hm(), &mut on_stm);
        collect_pairs(&self.masks, &bbs.flip_view().normalize_hm(), &mut on_ntm);
    }
    /// Sorted feature lists for (white perspective, black perspective) of an absolute position.
    pub fn features_for(&self, pos: &Position) -> (Vec<usize>, Vec<usize>) {
        let rel = RelBoard::from_position(pos, pos.side_to_move());
        let mut stm = Vec::with_capacity(160);
        let mut ntm = Vec::with_capacity(160);
        self.map_features(&rel, |s| stm.push(s), |n| ntm.push(n));
        stm.sort_unstable();
        ntm.sort_unstable();
        if pos.side_to_move() == Color::White {
            (stm, ntm)
        } else {
            (ntm, stm)
        }
    }
}

impl Default for FeatureMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_match_reference() {
        let t = Threats::new();
        assert_eq!(t.num_inputs(), 59808, "Stockfish FullThreats / bullet Threats dimension");
        let m = FeatureMapper::new();
        assert_eq!(m.num_inputs(), 4560 + 59808);
    }

    #[test]
    fn perspectives_are_symmetric() {
        // A position and its colour-flipped mirror must give swapped feature lists.
        crate::init();
        let m = FeatureMapper::new();
        let a = Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3").unwrap();
        let b = Position::from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3").unwrap();
        let (aw, ab) = m.features_for(&a);
        let (bw, bb) = m.features_for(&b);
        assert_eq!(aw, bb);
        assert_eq!(ab, bw);
        assert!(aw.len() > 20 && aw.len() <= MAX_PAIRS + MAX_THREATS);
    }
}
