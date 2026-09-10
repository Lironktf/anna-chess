//! Bitboard utilities and attack generation (magic / PEXT sliding attacks).

use crate::types::*;
use std::sync::OnceLock;

pub const FILE_A: Bitboard = 0x0101_0101_0101_0101;
pub const FILE_B: Bitboard = FILE_A << 1;
pub const FILE_G: Bitboard = FILE_A << 6;
pub const FILE_H: Bitboard = FILE_A << 7;
pub const RANK_1: Bitboard = 0xFF;
pub const RANK_2: Bitboard = RANK_1 << 8;
pub const RANK_3: Bitboard = RANK_1 << 16;
pub const RANK_4: Bitboard = RANK_1 << 24;
pub const RANK_5: Bitboard = RANK_1 << 32;
pub const RANK_6: Bitboard = RANK_1 << 40;
pub const RANK_7: Bitboard = RANK_1 << 48;
pub const RANK_8: Bitboard = RANK_1 << 56;
pub const DARK_SQUARES: Bitboard = 0xAA55_AA55_AA55_AA55;
pub const LIGHT_SQUARES: Bitboard = !DARK_SQUARES;
pub const QUEEN_SIDE: Bitboard = FILE_A | (FILE_A << 1) | (FILE_A << 2) | (FILE_A << 3);
pub const KING_SIDE: Bitboard = !QUEEN_SIDE;

#[inline(always)]
pub const fn file_bb(f: u8) -> Bitboard {
    FILE_A << f
}
#[inline(always)]
pub const fn rank_bb(r: u8) -> Bitboard {
    RANK_1 << (8 * r)
}
#[inline(always)]
pub const fn shift_north(b: Bitboard) -> Bitboard {
    b << 8
}
#[inline(always)]
pub const fn shift_south(b: Bitboard) -> Bitboard {
    b >> 8
}
#[inline(always)]
pub const fn shift_east(b: Bitboard) -> Bitboard {
    (b & !FILE_H) << 1
}
#[inline(always)]
pub const fn shift_west(b: Bitboard) -> Bitboard {
    (b & !FILE_A) >> 1
}
#[inline(always)]
pub const fn shift_ne(b: Bitboard) -> Bitboard {
    (b & !FILE_H) << 9
}
#[inline(always)]
pub const fn shift_nw(b: Bitboard) -> Bitboard {
    (b & !FILE_A) << 7
}
#[inline(always)]
pub const fn shift_se(b: Bitboard) -> Bitboard {
    (b & !FILE_H) >> 7
}
#[inline(always)]
pub const fn shift_sw(b: Bitboard) -> Bitboard {
    (b & !FILE_A) >> 9
}
#[inline(always)]
pub fn pawn_push(c: Color, b: Bitboard) -> Bitboard {
    match c {
        Color::White => shift_north(b),
        Color::Black => shift_south(b),
    }
}
#[inline(always)]
pub fn pawn_attacks_bb(c: Color, b: Bitboard) -> Bitboard {
    match c {
        Color::White => shift_ne(b) | shift_nw(b),
        Color::Black => shift_se(b) | shift_sw(b),
    }
}

#[inline(always)]
pub fn lsb(b: Bitboard) -> Square {
    debug_assert!(b != 0);
    b.trailing_zeros() as Square
}
#[inline(always)]
pub fn msb(b: Bitboard) -> Square {
    debug_assert!(b != 0);
    (63 - b.leading_zeros()) as Square
}
#[inline(always)]
pub fn pop_lsb(b: &mut Bitboard) -> Square {
    let s = lsb(*b);
    *b &= *b - 1;
    s
}
#[inline(always)]
pub fn popcount(b: Bitboard) -> u32 {
    b.count_ones()
}
#[inline(always)]
pub fn more_than_one(b: Bitboard) -> bool {
    b & b.wrapping_sub(1) != 0
}

/// Iterate set bits.
pub struct BitIter(pub Bitboard);
impl Iterator for BitIter {
    type Item = Square;
    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(pop_lsb(&mut self.0))
        }
    }
}
#[inline(always)]
pub fn bits(b: Bitboard) -> BitIter {
    BitIter(b)
}

#[derive(Clone, Copy)]
struct Magic {
    mask: Bitboard,
    magic: u64,
    shift: u32,
    offset: u32,
}

struct Tables {
    knight: [Bitboard; 64],
    king: [Bitboard; 64],
    pawn: [[Bitboard; 64]; 2],
    between: [[Bitboard; 64]; 64],
    line: [[Bitboard; 64]; 64],
    rook_magics: [Magic; 64],
    bishop_magics: [Magic; 64],
    attacks: Vec<Bitboard>,
    dist: [[u8; 64]; 64],
}

static TABLES: OnceLock<Tables> = OnceLock::new();

#[inline(always)]
fn tables() -> &'static Tables {
    // Initialised once at startup via `init()`; the fast path is a relaxed load.
    TABLES.get_or_init(build_tables)
}

pub fn init() {
    let _ = tables();
}

fn sliding_attack(deltas: &[(i8, i8)], s: Square, occ: Bitboard) -> Bitboard {
    let mut att = 0;
    for &(df, dr) in deltas {
        let mut f = file_of(s) as i8;
        let mut r = rank_of(s) as i8;
        loop {
            f += df;
            r += dr;
            if !(0..8).contains(&f) || !(0..8).contains(&r) {
                break;
            }
            let t = sq(f as u8, r as u8);
            att |= bb(t);
            if occ & bb(t) != 0 {
                break;
            }
        }
    }
    att
}

const ROOK_DELTAS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DELTAS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
const USE_PEXT: bool = true;
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
const USE_PEXT: bool = false;

#[inline(always)]
fn pext(v: u64, mask: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        // SAFETY: bmi2 is a compile-time target feature here.
        unsafe { std::arch::x86_64::_pext_u64(v, mask) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let _ = (v, mask);
        unreachable!()
    }
}

fn init_magics(
    deltas: &[(i8, i8)],
    magics: &mut [Magic; 64],
    attacks: &mut Vec<Bitboard>,
    rng: &mut Rng,
) {
    for s in 0..64u8 {
        let edges = ((RANK_1 | RANK_8) & !rank_bb(rank_of(s))) | ((FILE_A | FILE_H) & !file_bb(file_of(s)));
        let mask = sliding_attack(deltas, s, 0) & !edges;
        let bits = popcount(mask);
        let size = 1usize << bits;
        let offset = attacks.len();
        attacks.resize(offset + size, 0);

        // Enumerate occupancies (carry-rippler) and reference attacks.
        let mut occs = Vec::with_capacity(size);
        let mut refs = Vec::with_capacity(size);
        let mut b: Bitboard = 0;
        loop {
            occs.push(b);
            refs.push(sliding_attack(deltas, s, b));
            b = b.wrapping_sub(mask) & mask;
            if b == 0 {
                break;
            }
        }

        if USE_PEXT {
            for i in 0..size {
                attacks[offset + pext(occs[i], mask) as usize] = refs[i];
            }
            magics[s as usize] = Magic { mask, magic: 0, shift: 0, offset: offset as u32 };
            continue;
        }

        let shift = 64 - bits;
        let mut epoch = vec![0u32; size];
        let mut cnt = 0u32;
        loop {
            let magic = loop {
                let m = rng.sparse();
                if popcount((mask.wrapping_mul(m)) >> 56) >= 6 {
                    break m;
                }
            };
            cnt += 1;
            let mut ok = true;
            for i in 0..size {
                let idx = ((occs[i] & mask).wrapping_mul(magic) >> shift) as usize;
                if epoch[idx] < cnt {
                    epoch[idx] = cnt;
                    attacks[offset + idx] = refs[i];
                } else if attacks[offset + idx] != refs[i] {
                    ok = false;
                    break;
                }
            }
            if ok {
                magics[s as usize] = Magic { mask, magic, shift, offset: offset as u32 };
                break;
            }
        }
    }
}

fn build_tables() -> Tables {
    let mut knight = [0; 64];
    let mut king = [0; 64];
    let mut pawn = [[0; 64]; 2];
    let mut dist = [[0u8; 64]; 64];
    for s in 0..64u8 {
        let b = bb(s);
        knight[s as usize] = shift_north(shift_ne(b))
            | shift_north(shift_nw(b))
            | shift_south(shift_se(b))
            | shift_south(shift_sw(b))
            | shift_east(shift_ne(b))
            | shift_east(shift_se(b))
            | shift_west(shift_nw(b))
            | shift_west(shift_sw(b));
        king[s as usize] = shift_north(b)
            | shift_south(b)
            | shift_east(b)
            | shift_west(b)
            | shift_ne(b)
            | shift_nw(b)
            | shift_se(b)
            | shift_sw(b);
        pawn[0][s as usize] = pawn_attacks_bb(Color::White, b);
        pawn[1][s as usize] = pawn_attacks_bb(Color::Black, b);
        for t in 0..64u8 {
            let df = (file_of(s) as i8 - file_of(t) as i8).unsigned_abs();
            let dr = (rank_of(s) as i8 - rank_of(t) as i8).unsigned_abs();
            dist[s as usize][t as usize] = df.max(dr);
        }
    }

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let dummy = Magic { mask: 0, magic: 0, shift: 0, offset: 0 };
    let mut rook_magics = [dummy; 64];
    let mut bishop_magics = [dummy; 64];
    let mut attacks = Vec::with_capacity(107_648);
    init_magics(&ROOK_DELTAS, &mut rook_magics, &mut attacks, &mut rng);
    init_magics(&BISHOP_DELTAS, &mut bishop_magics, &mut attacks, &mut rng);

    let mut between = [[0; 64]; 64];
    let mut line = [[0; 64]; 64];
    for s in 0..64u8 {
        for t in 0..64u8 {
            if s == t {
                continue;
            }
            for (deltas, is_rook) in [(&ROOK_DELTAS, true), (&BISHOP_DELTAS, false)] {
                let _ = is_rook;
                let a = sliding_attack(deltas, s, 0);
                if a & bb(t) != 0 {
                    line[s as usize][t as usize] = (a & sliding_attack(deltas, t, 0)) | bb(s) | bb(t);
                    between[s as usize][t as usize] =
                        sliding_attack(deltas, s, bb(t)) & sliding_attack(deltas, t, bb(s));
                }
            }
        }
    }

    Tables { knight, king, pawn, between, line, rook_magics, bishop_magics, attacks, dist }
}

#[inline(always)]
pub fn knight_attacks(s: Square) -> Bitboard {
    tables().knight[s as usize]
}
#[inline(always)]
pub fn king_attacks(s: Square) -> Bitboard {
    tables().king[s as usize]
}
#[inline(always)]
pub fn pawn_attacks(c: Color, s: Square) -> Bitboard {
    tables().pawn[c.idx()][s as usize]
}
#[inline(always)]
fn magic_index(m: &Magic, occ: Bitboard) -> usize {
    if USE_PEXT {
        m.offset as usize + pext(occ, m.mask) as usize
    } else {
        m.offset as usize + (((occ & m.mask).wrapping_mul(m.magic)) >> m.shift) as usize
    }
}
#[inline(always)]
pub fn rook_attacks(s: Square, occ: Bitboard) -> Bitboard {
    let t = tables();
    // SAFETY: index is bounded by construction of the magic tables.
    unsafe { *t.attacks.get_unchecked(magic_index(&t.rook_magics[s as usize], occ)) }
}
#[inline(always)]
pub fn bishop_attacks(s: Square, occ: Bitboard) -> Bitboard {
    let t = tables();
    // SAFETY: index is bounded by construction of the magic tables.
    unsafe { *t.attacks.get_unchecked(magic_index(&t.bishop_magics[s as usize], occ)) }
}
#[inline(always)]
pub fn queen_attacks(s: Square, occ: Bitboard) -> Bitboard {
    rook_attacks(s, occ) | bishop_attacks(s, occ)
}
/// Attacks of a piece type from `s` (pawn excluded).
#[inline(always)]
pub fn piece_attacks(pt: PieceType, s: Square, occ: Bitboard) -> Bitboard {
    match pt {
        PieceType::Knight => knight_attacks(s),
        PieceType::Bishop => bishop_attacks(s, occ),
        PieceType::Rook => rook_attacks(s, occ),
        PieceType::Queen => queen_attacks(s, occ),
        PieceType::King => king_attacks(s),
        PieceType::Pawn => 0,
    }
}
/// Squares strictly between two aligned squares (0 if not aligned).
#[inline(always)]
pub fn between(a: Square, b: Square) -> Bitboard {
    tables().between[a as usize][b as usize]
}
/// Full line through two aligned squares including both (0 if not aligned).
#[inline(always)]
pub fn line(a: Square, b: Square) -> Bitboard {
    tables().line[a as usize][b as usize]
}
#[inline(always)]
pub fn aligned(a: Square, b: Square, c: Square) -> bool {
    line(a, b) & bb(c) != 0
}
#[inline(always)]
pub fn distance(a: Square, b: Square) -> u8 {
    tables().dist[a as usize][b as usize]
}

pub fn pretty(b: Bitboard) -> String {
    let mut s = String::new();
    for r in (0..8).rev() {
        for f in 0..8 {
            s.push(if b & bb(sq(f, r)) != 0 { 'X' } else { '.' });
            s.push(' ');
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliders_match_reference() {
        init();
        let mut rng = Rng(12345);
        for _ in 0..20000 {
            let occ = rng.next() & rng.next();
            let s = (rng.next() % 64) as Square;
            assert_eq!(rook_attacks(s, occ), sliding_attack(&ROOK_DELTAS, s, occ));
            assert_eq!(bishop_attacks(s, occ), sliding_attack(&BISHOP_DELTAS, s, occ));
        }
    }

    #[test]
    fn leapers() {
        init();
        assert_eq!(popcount(knight_attacks(squares::A1)), 2);
        assert_eq!(popcount(knight_attacks(27)), 8);
        assert_eq!(popcount(king_attacks(squares::E1)), 5);
        assert_eq!(pawn_attacks(Color::White, squares::A1), bb(9));
        assert_eq!(pawn_attacks(Color::Black, squares::H8), bb(54));
    }

    #[test]
    fn between_and_line() {
        init();
        assert_eq!(between(squares::A1, squares::H8), bb(9) | bb(18) | bb(27) | bb(36) | bb(45) | bb(54));
        assert_eq!(between(squares::A1, 17), 0);
        assert!(aligned(squares::A1, squares::H1, squares::D1));
        assert!(!aligned(squares::A1, squares::H1, 11));
    }
}
