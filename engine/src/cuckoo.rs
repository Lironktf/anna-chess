//! Upcoming-repetition detection (Marcel van Kervinck's cuckoo hashing of reversible moves), as used
//! by Stockfish: lets the search treat a position where the side to move can force a repetition
//! as a draw-ish node early, and avoids fail-high on positions that can be repeated by the opponent.

use crate::bitboard::*;
use crate::position::Position;
use crate::types::*;
use crate::zobrist::*;
use std::sync::OnceLock;

struct Tables {
    keys: Vec<u64>,
    moves: Vec<Move>,
}

static TABLES: OnceLock<Tables> = OnceLock::new();

#[inline(always)]
fn h1(k: u64) -> usize {
    (k & 0x1fff) as usize
}
#[inline(always)]
fn h2(k: u64) -> usize {
    ((k >> 16) & 0x1fff) as usize
}

fn build() -> Tables {
    crate::bitboard::init();
    let mut keys = vec![0u64; 8192];
    let mut moves = vec![Move::NONE; 8192];
    let mut count = 0;
    for pc_idx in 0..12 {
        let pc = Piece::from_idx(pc_idx);
        let pt = pc.piece_type();
        if pt == PieceType::Pawn {
            continue;
        }
        for s1 in 0..64u8 {
            for s2 in (s1 + 1)..64u8 {
                if piece_attacks(pt, s1, 0) & bb(s2) == 0 {
                    continue;
                }
                let mut mv = Move::new(s1, s2);
                let mut key = psq_key(pc, s1) ^ psq_key(pc, s2) ^ ZOBRIST.side;
                let mut i = h1(key);
                loop {
                    std::mem::swap(&mut keys[i], &mut key);
                    std::mem::swap(&mut moves[i], &mut mv);
                    if mv.is_none() {
                        break;
                    }
                    i = if i == h1(key) { h2(key) } else { h1(key) };
                }
                count += 1;
            }
        }
    }
    debug_assert_eq!(count, 3668);
    Tables { keys, moves }
}

pub fn init() {
    let _ = TABLES.get_or_init(build);
}

/// True if the side to move can reach a repetition of a position in `keys` (the key history,
/// most recent last, not including `pos` itself) within the reversible-move horizon.
/// `ply` is the search ply of `pos`; repetitions of positions before the root require the
/// earlier occurrence to be within the game history (Stockfish semantics).
pub fn has_upcoming_repetition(pos: &Position, keys: &[u64], ply: usize) -> bool {
    let t = TABLES.get_or_init(build);
    let n = keys.len();
    let end = (pos.rule50() as usize).min(n);
    if end < 3 {
        return false;
    }
    let occ = pos.occupied();
    let orig = pos.key();
    // keys[n-1] is the position one ply ago; we need positions 2, 4, ... plies ago (same stm) and
    // the "in-between" ones: key_prev_i = keys[n - i].
    let mut i = 3;
    while i <= end {
        let key_i = keys[n - i];
        // Moves between position i plies ago and now must be reversible; the diff of the keys
        // after the move at ply i-1 identifies a single reversible move.
        let key_im1 = keys[n - i + 1];
        let _ = key_im1;
        let move_key = orig ^ key_i;
        let mut j = h1(move_key);
        if t.keys[j] != move_key {
            j = h2(move_key);
            if t.keys[j] != move_key {
                i += 2;
                continue;
            }
        }
        let mv = t.moves[j];
        let (s1, s2) = (mv.from(), mv.to());
        if (between(s1, s2) & occ) != 0 {
            i += 2;
            continue;
        }
        // The move must be by the side to move: the piece must be on one of the squares.
        let p1 = pos.piece_on(s1);
        let p2 = pos.piece_on(s2);
        let mover = if !p1.is_none() { p1 } else { p2 };
        if mover.is_none() || mover.color() != pos.side_to_move() {
            i += 2;
            continue;
        }
        if ply > i {
            return true;
        }
        // Position i plies ago is before the root: it must itself have been repeated earlier.
        let mut k = i + 2;
        while k <= end {
            if keys[n - k] == key_i {
                return true;
            }
            k += 2;
        }
        i += 2;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_builds() {
        init();
        let t = TABLES.get().unwrap();
        assert_eq!(t.keys.iter().filter(|k| **k != 0).count(), 3668);
    }

    #[test]
    fn detects_repetition_by_knight_shuffle() {
        crate::init();
        init();
        // Play Nf3, Nf6, Ng1, Ng8 -> the start position repeats; before the last move the side
        // to move (black) can repeat with Ng8.
        let mut pos = Position::from_fen(crate::position::START_FEN).unwrap();
        let mut keys = Vec::new();
        for m in ["g1f3", "g8f6", "f3g1"] {
            let mv = pos.parse_uci_move(m).unwrap();
            keys.push(pos.key());
            pos = pos.make_move(mv);
        }
        // Inside the search (ply large) the upcoming repetition of a search-tree position counts.
        assert!(has_upcoming_repetition(&pos, &keys, 10));
        // At the root with only game history, a single earlier occurrence is not enough.
        assert!(!has_upcoming_repetition(&pos, &keys, 0));
        // But if the start position occurred twice in the game history it is.
        let mut pos2 = Position::from_fen(crate::position::START_FEN).unwrap();
        let mut keys2 = Vec::new();
        for m in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"] {
            let mv = pos2.parse_uci_move(m).unwrap();
            keys2.push(pos2.key());
            pos2 = pos2.make_move(mv);
        }
        assert!(has_upcoming_repetition(&pos2, &keys2, 0));
    }
}
