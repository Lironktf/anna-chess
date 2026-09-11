//! Syzygy tablebase probing through the vendored Fathom library (engine/csrc/fathom).

use crate::bitboard::popcount;
use crate::position::Position;
use crate::types::*;
use std::ffi::CString;
use std::sync::atomic::{AtomicU32, Ordering};

extern "C" {
    static TB_LARGEST: u32;
    fn tb_init(path: *const std::os::raw::c_char) -> bool;
    fn tb_free();
    fn tb_probe_wdl_impl(white: u64, black: u64, kings: u64, queens: u64, rooks: u64, bishops: u64, knights: u64, pawns: u64, ep: u32, turn: bool) -> u32;
    fn tb_probe_root_impl(white: u64, black: u64, kings: u64, queens: u64, rooks: u64, bishops: u64, knights: u64, pawns: u64, rule50: u32, ep: u32, turn: bool, results: *mut u32) -> u32;
}

pub const TB_LOSS: u32 = 0;
pub const TB_BLESSED_LOSS: u32 = 1;
pub const TB_DRAW: u32 = 2;
pub const TB_CURSED_WIN: u32 = 3;
pub const TB_WIN: u32 = 4;
const TB_RESULT_FAILED: u32 = 0xFFFF_FFFF;
const TB_MAX_MOVES: usize = 193;

/// Largest piece count available (0 = no tablebases loaded).
static LARGEST: AtomicU32 = AtomicU32::new(0);

#[inline(always)]
pub fn largest() -> u32 {
    LARGEST.load(Ordering::Relaxed)
}

/// Load tablebases from a path list (":"-separated). Not thread safe; call from the UCI thread
/// with no search running.
pub fn init(path: &str) -> u32 {
    let c = CString::new(path).unwrap_or_default();
    // SAFETY: single-threaded call with a valid NUL-terminated string.
    unsafe {
        tb_free();
        if path.is_empty() || path == "<empty>" {
            LARGEST.store(0, Ordering::Relaxed);
            return 0;
        }
        tb_init(c.as_ptr());
        let n = TB_LARGEST;
        LARGEST.store(n, Ordering::Relaxed);
        n
    }
}

#[inline(always)]
fn bbs(pos: &Position) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
    (
        pos.colored(Color::White),
        pos.colored(Color::Black),
        pos.pieces(PieceType::King),
        pos.pieces(PieceType::Queen),
        pos.pieces(PieceType::Rook),
        pos.pieces(PieceType::Bishop),
        pos.pieces(PieceType::Knight),
        pos.pieces(PieceType::Pawn),
    )
}

/// WDL probe from the side to move's view: Some(0..=4) or None on failure. Only valid when the
/// position has no castling rights and the 50-move counter is zero (caller checks).
#[inline]
pub fn probe_wdl(pos: &Position) -> Option<u32> {
    if largest() == 0 || popcount(pos.occupied()) > largest() || pos.castling_rights() != 0 || pos.rule50() != 0 {
        return None;
    }
    let (w, b, k, q, r, bi, n, p) = bbs(pos);
    let ep = if pos.ep_square() == SQ_NONE { 0 } else { pos.ep_square() as u32 };
    // SAFETY: plain integer arguments; Fathom's WDL probe is read-only after init.
    let res = unsafe { tb_probe_wdl_impl(w, b, k, q, r, bi, n, p, ep, pos.side_to_move() == Color::White) };
    if res == TB_RESULT_FAILED {
        None
    } else {
        Some(res & 0xF)
    }
}

/// Root probe: returns, for each legal move, (move, wdl, dtz). Not thread safe; once per search.
pub fn probe_root(pos: &Position) -> Option<Vec<(Move, u32, u32)>> {
    if largest() == 0 || popcount(pos.occupied()) > largest() || pos.castling_rights() != 0 {
        return None;
    }
    let (w, b, k, q, r, bi, n, p) = bbs(pos);
    let ep = if pos.ep_square() == SQ_NONE { 0 } else { pos.ep_square() as u32 };
    let mut results = [TB_RESULT_FAILED; TB_MAX_MOVES];
    // SAFETY: results has TB_MAX_MOVES entries as the API requires.
    let res = unsafe {
        tb_probe_root_impl(w, b, k, q, r, bi, n, p, pos.rule50() as u32, ep, pos.side_to_move() == Color::White, results.as_mut_ptr())
    };
    if res == TB_RESULT_FAILED {
        return None;
    }
    let mut out = Vec::new();
    for &rv in results.iter() {
        if rv == TB_RESULT_FAILED {
            break;
        }
        let wdl = rv & 0xF;
        let from = ((rv >> 4) & 0x3F) as Square;
        let to = ((rv >> 10) & 0x3F) as Square;
        let promo = (rv >> 16) & 0x7; // 0 none, 1 Q, 2 R, 3 B, 4 N
        let is_ep = (rv >> 19) & 1 == 1;
        let dtz = (rv >> 20) & 0xFFF;
        let mv = if promo != 0 {
            let pt = match promo {
                1 => PieceType::Queen,
                2 => PieceType::Rook,
                3 => PieceType::Bishop,
                _ => PieceType::Knight,
            };
            Move::new_promo(from, to, pt)
        } else if is_ep {
            Move::new_flag(from, to, FLAG_EP)
        } else {
            Move::new(from, to)
        };
        out.push((mv, wdl, dtz));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tb_dir() -> Option<String> {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../syzygy");
        if std::path::Path::new(p).join("KQvK.rtbw").exists() && std::path::Path::new(p).join("KQvK.rtbz").exists() {
            Some(p.to_string())
        } else {
            None
        }
    }

    #[test]
    fn wdl_and_root_probes() {
        crate::init();
        let Some(dir) = tb_dir() else {
            eprintln!("syzygy tables not present, skipping");
            return;
        };
        assert!(init(&dir) >= 3);
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1").unwrap();
        assert_eq!(probe_wdl(&pos), Some(TB_WIN));
        let pos_b = Position::from_fen("4k3/8/8/8/8/8/8/4KQ2 b - - 0 1").unwrap();
        assert_eq!(probe_wdl(&pos_b), Some(TB_LOSS));
        let draw = Position::from_fen("4k3/8/8/8/8/8/8/4KB2 w - - 0 1").unwrap();
        assert_eq!(probe_wdl(&draw), Some(TB_DRAW));
        let root = probe_root(&pos).unwrap();
        assert!(!root.is_empty());
        assert!(root.iter().all(|(_, wdl, _)| *wdl == TB_WIN), "every KQvK move keeps the win");
        let legal = crate::movegen::legal_moves(&pos);
        for (m, _, _) in &root {
            assert!(legal.contains(*m), "root probe move {} not legal", m);
        }
    }
}
