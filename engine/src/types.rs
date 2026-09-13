//! Fundamental types: colors, pieces, squares, moves, scores.

use std::fmt;

pub type Bitboard = u64;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }
    #[inline(always)]
    pub const fn from_idx(i: usize) -> Color {
        if i == 0 {
            Color::White
        } else {
            Color::Black
        }
    }
}

impl std::ops::Not for Color {
    type Output = Color;
    #[inline(always)]
    fn not(self) -> Color {
        self.flip()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }
    #[inline(always)]
    pub const fn from_idx(i: usize) -> PieceType {
        match i {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            _ => PieceType::King,
        }
    }
    pub fn to_char(self) -> char {
        match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        }
    }
    pub fn from_char(c: char) -> Option<PieceType> {
        match c.to_ascii_lowercase() {
            'p' => Some(PieceType::Pawn),
            'n' => Some(PieceType::Knight),
            'b' => Some(PieceType::Bishop),
            'r' => Some(PieceType::Rook),
            'q' => Some(PieceType::Queen),
            'k' => Some(PieceType::King),
            _ => None,
        }
    }
}

/// Piece encoding: 0..=5 white pawn..king, 6..=11 black pawn..king, 12 = none.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Default)]
#[repr(u8)]
pub enum Piece {
    WP = 0,
    WN = 1,
    WB = 2,
    WR = 3,
    WQ = 4,
    WK = 5,
    BP = 6,
    BN = 7,
    BB = 8,
    BR = 9,
    BQ = 10,
    BK = 11,
    #[default]
    None = 12,
}

impl Piece {
    #[inline(always)]
    pub const fn new(c: Color, pt: PieceType) -> Piece {
        Piece::from_idx(c.idx() * 6 + pt.idx())
    }
    #[inline(always)]
    pub const fn from_idx(i: usize) -> Piece {
        // SAFETY: i in 0..=12 by construction at all call sites.
        debug_assert!(i <= 12);
        unsafe { std::mem::transmute(i as u8) }
    }
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }
    #[inline(always)]
    pub const fn is_none(self) -> bool {
        matches!(self, Piece::None)
    }
    #[inline(always)]
    pub const fn color(self) -> Color {
        debug_assert!(!self.is_none());
        if (self as u8) < 6 {
            Color::White
        } else {
            Color::Black
        }
    }
    #[inline(always)]
    pub const fn piece_type(self) -> PieceType {
        debug_assert!(!self.is_none());
        // SAFETY: PieceType is repr(u8) with values 0..=5 in the same order as the pieces of
        // each colour, and (x % 6) is always in 0..=5.
        unsafe { std::mem::transmute((self as u8) % 6) }
    }
    pub fn to_char(self) -> char {
        if self.is_none() {
            return '.';
        }
        let c = self.piece_type().to_char();
        if self.color() == Color::White {
            c.to_ascii_uppercase()
        } else {
            c
        }
    }
    pub fn from_char(c: char) -> Option<Piece> {
        let pt = PieceType::from_char(c)?;
        let color = if c.is_ascii_uppercase() { Color::White } else { Color::Black };
        Some(Piece::new(color, pt))
    }
}

/// Square index: a1 = 0, b1 = 1, ..., h8 = 63.
pub type Square = u8;

pub const SQ_NONE: Square = 64;

#[inline(always)]
pub const fn sq(file: u8, rank: u8) -> Square {
    rank * 8 + file
}
#[inline(always)]
pub const fn file_of(s: Square) -> u8 {
    s & 7
}
#[inline(always)]
pub const fn rank_of(s: Square) -> u8 {
    s >> 3
}
#[inline(always)]
pub const fn flip_rank(s: Square) -> Square {
    s ^ 56
}
#[inline(always)]
pub const fn flip_file(s: Square) -> Square {
    s ^ 7
}
#[inline(always)]
pub const fn relative_rank(c: Color, s: Square) -> u8 {
    rank_of(s) ^ (c as u8 * 7)
}
#[inline(always)]
pub const fn bb(s: Square) -> Bitboard {
    1u64 << s
}

pub fn square_to_string(s: Square) -> String {
    if s >= 64 {
        return "-".to_string();
    }
    let f = (b'a' + file_of(s)) as char;
    let r = (b'1' + rank_of(s)) as char;
    format!("{}{}", f, r)
}

pub fn square_from_str(s: &str) -> Option<Square> {
    let b = s.as_bytes();
    if b.len() < 2 {
        return None;
    }
    let f = b[0].wrapping_sub(b'a');
    let r = b[1].wrapping_sub(b'1');
    if f < 8 && r < 8 {
        Some(sq(f, r))
    } else {
        None
    }
}

pub mod squares {
    pub const A1: u8 = 0;
    pub const B1: u8 = 1;
    pub const C1: u8 = 2;
    pub const D1: u8 = 3;
    pub const E1: u8 = 4;
    pub const F1: u8 = 5;
    pub const G1: u8 = 6;
    pub const H1: u8 = 7;
    pub const A8: u8 = 56;
    pub const B8: u8 = 57;
    pub const C8: u8 = 58;
    pub const D8: u8 = 59;
    pub const E8: u8 = 60;
    pub const F8: u8 = 61;
    pub const G8: u8 = 62;
    pub const H8: u8 = 63;
}

/// Move encoding (16 bits): bits 0-5 from, 6-11 to, 12-13 promotion piece (N=0,B=1,R=2,Q=3),
/// bits 14-15 flag: 0 normal, 1 promotion, 2 en passant, 3 castling.
/// Castling is encoded king-from -> rook-from (Chess960 style) internally.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Default)]
pub struct Move(pub u16);

pub const FLAG_NORMAL: u16 = 0;
pub const FLAG_PROMO: u16 = 1 << 14;
pub const FLAG_EP: u16 = 2 << 14;
pub const FLAG_CASTLE: u16 = 3 << 14;

impl Move {
    pub const NONE: Move = Move(0);
    pub const NULL: Move = Move(65); // from=a1?? use a1->b1 style "null" marker distinct from NONE

    #[inline(always)]
    pub const fn new(from: Square, to: Square) -> Move {
        Move((from as u16) | ((to as u16) << 6))
    }
    #[inline(always)]
    pub const fn new_flag(from: Square, to: Square, flag: u16) -> Move {
        Move((from as u16) | ((to as u16) << 6) | flag)
    }
    #[inline(always)]
    pub const fn new_promo(from: Square, to: Square, promo: PieceType) -> Move {
        Move((from as u16) | ((to as u16) << 6) | (((promo as u16) - 1) << 12) | FLAG_PROMO)
    }
    #[inline(always)]
    pub const fn from(self) -> Square {
        (self.0 & 63) as Square
    }
    #[inline(always)]
    pub const fn to(self) -> Square {
        ((self.0 >> 6) & 63) as Square
    }
    #[inline(always)]
    pub const fn flag(self) -> u16 {
        self.0 & (3 << 14)
    }
    #[inline(always)]
    pub const fn is_promo(self) -> bool {
        self.flag() == FLAG_PROMO
    }
    #[inline(always)]
    pub const fn is_ep(self) -> bool {
        self.flag() == FLAG_EP
    }
    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        self.flag() == FLAG_CASTLE
    }
    #[inline(always)]
    pub const fn promo_type(self) -> PieceType {
        PieceType::from_idx((((self.0 >> 12) & 3) + 1) as usize)
    }
    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
    /// from/to squares packed into 12 bits, used for history indexing.
    #[inline(always)]
    pub const fn from_to(self) -> usize {
        (self.0 & 0xFFF) as usize
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            return write!(f, "0000");
        }
        write!(f, "{}{}", square_to_string(self.from()), square_to_string(self.to()))?;
        if self.is_promo() {
            write!(f, "{}", self.promo_type().to_char())?;
        }
        Ok(())
    }
}

// ---- Scores ----
pub type Value = i32;

pub const VALUE_NONE: Value = 32002;
pub const VALUE_INFINITE: Value = 32001;
pub const VALUE_MATE: Value = 32000;
pub const MAX_PLY: usize = 128;
pub const VALUE_MATE_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY as Value;
pub const VALUE_MATED_IN_MAX_PLY: Value = -VALUE_MATE_IN_MAX_PLY;
pub const VALUE_TB_WIN: Value = VALUE_MATE_IN_MAX_PLY - 1;
pub const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_TB_WIN - MAX_PLY as Value;
pub const VALUE_TB_LOSS_IN_MAX_PLY: Value = -VALUE_TB_WIN_IN_MAX_PLY;
pub const VALUE_DRAW: Value = 0;

#[inline(always)]
pub const fn mate_in(ply: usize) -> Value {
    VALUE_MATE - ply as Value
}
#[inline(always)]
pub const fn mated_in(ply: usize) -> Value {
    -VALUE_MATE + ply as Value
}
#[inline(always)]
pub const fn is_win(v: Value) -> bool {
    v >= VALUE_TB_WIN_IN_MAX_PLY
}
#[inline(always)]
pub const fn is_loss(v: Value) -> bool {
    v <= VALUE_TB_LOSS_IN_MAX_PLY
}
#[inline(always)]
pub const fn is_decisive(v: Value) -> bool {
    is_win(v) || is_loss(v)
}
#[inline(always)]
pub const fn is_valid(v: Value) -> bool {
    v != VALUE_NONE
}

/// Piece values used for SEE, move ordering and pruning margins.
pub const PIECE_VALUES: [Value; 7] = [100, 300, 300, 500, 900, 0, 0];

#[inline(always)]
pub fn piece_value(pt: PieceType) -> Value {
    PIECE_VALUES[pt.idx()]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    Upper = 1,
    Lower = 2,
    Exact = 3,
}

impl Bound {
    #[inline(always)]
    pub const fn from_u8(b: u8) -> Bound {
        match b & 3 {
            0 => Bound::None,
            1 => Bound::Upper,
            2 => Bound::Lower,
            _ => Bound::Exact,
        }
    }
    #[inline(always)]
    pub const fn has_lower(self) -> bool {
        (self as u8) & 2 != 0
    }
    #[inline(always)]
    pub const fn has_upper(self) -> bool {
        (self as u8) & 1 != 0
    }
}
