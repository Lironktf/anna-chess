//! Pseudo-legal move generation (Stockfish-style; use `Position::is_legal` to filter).

use crate::bitboard::*;
use crate::position::*;
use crate::types::*;

pub const MAX_MOVES: usize = 256;

#[derive(Clone, Copy, Default)]
pub struct ScoredMove {
    pub mv: Move,
    pub score: i32,
}

pub struct MoveList {
    pub moves: [ScoredMove; MAX_MOVES],
    pub len: usize,
}

impl MoveList {
    #[inline(always)]
    pub fn new() -> Self {
        MoveList { moves: [ScoredMove::default(); MAX_MOVES], len: 0 }
    }
    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES);
        self.moves[self.len].mv = m;
        self.len += 1;
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }
    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.moves[..self.len].iter().map(|s| s.mv)
    }
    pub fn contains(&self, m: Move) -> bool {
        self.iter().any(|x| x == m)
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenType {
    /// Captures, en passant and queen promotions.
    Captures,
    /// Non-captures (including underpromotions).
    Quiets,
    /// All check evasions (only valid when in check).
    Evasions,
    /// All pseudo-legal moves (captures + quiets, or evasions if in check).
    All,
}

#[inline(always)]
fn add_promotions(list: &mut MoveList, from: Square, to: Square, gen: GenType) {
    if gen == GenType::Captures || gen == GenType::Evasions || gen == GenType::All {
        list.push(Move::new_promo(from, to, PieceType::Queen));
    }
    if gen == GenType::Quiets || gen == GenType::Evasions || gen == GenType::All {
        list.push(Move::new_promo(from, to, PieceType::Rook));
        list.push(Move::new_promo(from, to, PieceType::Bishop));
        list.push(Move::new_promo(from, to, PieceType::Knight));
    }
}

fn gen_pawn_moves(pos: &Position, list: &mut MoveList, gen: GenType, target: Bitboard) {
    let us = pos.side_to_move();
    let them = !us;
    let (rank7, rank3) = if us == Color::White { (RANK_7, RANK_3) } else { (RANK_2, RANK_6) };
    let up: i8 = if us == Color::White { 8 } else { -8 };
    let (up_left, up_right): (i8, i8) = if us == Color::White { (7, 9) } else { (-9, -7) };
    let pawns = pos.pieces_c(us, PieceType::Pawn);
    let pawns_on7 = pawns & rank7;
    let pawns_not7 = pawns & !rank7;
    let empty = !pos.occupied();
    let enemies = if gen == GenType::Evasions { pos.checkers() } else { pos.colored(them) };
    let enemies = enemies & target;

    // Single and double pushes.
    if gen != GenType::Captures {
        let mut b1 = pawn_push(us, pawns_not7) & empty;
        let mut b2 = pawn_push(us, b1 & rank3) & empty;
        if gen == GenType::Evasions {
            b1 &= target;
            b2 &= target;
        }
        for to in bits(b1) {
            list.push(Move::new((to as i8 - up) as Square, to));
        }
        for to in bits(b2) {
            list.push(Move::new((to as i8 - 2 * up) as Square, to));
        }
    }

    // Promotions.
    if pawns_on7 != 0 {
        let cap_l = shift_promo(us, pawns_on7, true) & enemies;
        let cap_r = shift_promo(us, pawns_on7, false) & enemies;
        let mut push = pawn_push(us, pawns_on7) & empty;
        if gen == GenType::Evasions {
            push &= target;
        }
        for to in bits(cap_l) {
            add_promotions(list, (to as i8 - up_left) as Square, to, gen);
        }
        for to in bits(cap_r) {
            add_promotions(list, (to as i8 - up_right) as Square, to, gen);
        }
        for to in bits(push) {
            add_promotions(list, (to as i8 - up) as Square, to, gen);
        }
    }

    // Captures and en passant.
    if gen == GenType::Captures || gen == GenType::Evasions || gen == GenType::All {
        let cap_l = shift_promo(us, pawns_not7, true) & enemies;
        let cap_r = shift_promo(us, pawns_not7, false) & enemies;
        for to in bits(cap_l) {
            list.push(Move::new((to as i8 - up_left) as Square, to));
        }
        for to in bits(cap_r) {
            list.push(Move::new((to as i8 - up_right) as Square, to));
        }
        let ep = pos.ep_square();
        if ep != SQ_NONE {
            // In evasions, ep is only useful if the captured pawn is the checker
            // (or, rarely, the ep square blocks; SF handles by checking target).
            let cap_sq = (ep as i8 - up) as Square;
            if gen == GenType::Evasions && (target & (bb(cap_sq) | bb(ep))) == 0 {
                return;
            }
            let attackers = pawns_not7 & pawn_attacks(them, ep);
            for from in bits(attackers) {
                list.push(Move::new_flag(from, ep, FLAG_EP));
            }
        }
    }
}

#[inline(always)]
fn shift_promo(us: Color, b: Bitboard, left: bool) -> Bitboard {
    match (us, left) {
        (Color::White, true) => shift_nw(b),
        (Color::White, false) => shift_ne(b),
        (Color::Black, true) => shift_sw(b),
        (Color::Black, false) => shift_se(b),
    }
}

fn gen_piece_moves(pos: &Position, list: &mut MoveList, pt: PieceType, target: Bitboard) {
    let us = pos.side_to_move();
    let occ = pos.occupied();
    for from in bits(pos.pieces_c(us, pt)) {
        let att = piece_attacks(pt, from, occ) & target;
        for to in bits(att) {
            list.push(Move::new(from, to));
        }
    }
}

/// Generate pseudo-legal moves of the given type. When in check, `Captures`/`Quiets` callers
/// should use `Evasions` (or `All`) instead.
pub fn generate(pos: &Position, list: &mut MoveList, gen: GenType) {
    let us = pos.side_to_move();
    let ksq = pos.king_sq(us);
    let in_check = pos.in_check();
    let gen = if in_check && gen == GenType::All { GenType::Evasions } else { gen };

    if gen == GenType::Evasions {
        debug_assert!(in_check);
        // King moves (not along the checking slider's line).
        let mut slider_att = 0;
        let sliders = pos.checkers() & !pos.pieces2(PieceType::Pawn, PieceType::Knight);
        for s in bits(sliders) {
            slider_att |= line(s, ksq) ^ bb(s);
        }
        let kmoves = king_attacks(ksq) & !pos.colored(us) & !slider_att;
        for to in bits(kmoves) {
            list.push(Move::new(ksq, to));
        }
        if more_than_one(pos.checkers()) {
            return;
        }
        let ck = lsb(pos.checkers());
        let target = between(ck, ksq) | bb(ck);
        gen_pawn_moves(pos, list, GenType::Evasions, target);
        for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            gen_piece_moves(pos, list, pt, target);
        }
        return;
    }

    let target = match gen {
        GenType::Captures => pos.colored(!us),
        GenType::Quiets => !pos.occupied(),
        _ => !pos.colored(us),
    };
    gen_pawn_moves(pos, list, gen, target);
    for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        gen_piece_moves(pos, list, pt, target);
    }
    // King moves.
    let kmoves = king_attacks(ksq) & target;
    for to in bits(kmoves) {
        list.push(Move::new(ksq, to));
    }
    // Castling (quiet).
    if gen != GenType::Captures && pos.castling_rights() != 0 {
        let rights = if us == Color::White { [CASTLE_WK, CASTLE_WQ] } else { [CASTLE_BK, CASTLE_BQ] };
        for bit in rights {
            if pos.can_castle(bit) && (pos.castle_path(bit) & pos.occupied()) == 0 {
                list.push(Move::new_flag(ksq, pos.castle_rook_sq(bit), FLAG_CASTLE));
            }
        }
    }
}

/// Fully legal move list.
pub fn legal_moves(pos: &Position) -> MoveList {
    let mut list = MoveList::new();
    generate(pos, &mut list, GenType::All);
    let mut out = MoveList::new();
    for m in list.iter() {
        if pos.is_legal(m) {
            out.push(m);
        }
    }
    out
}

pub fn perft(pos: &Position, depth: u32) -> u64 {
    let moves = legal_moves(pos);
    if depth == 1 {
        return moves.len() as u64;
    }
    if depth == 0 {
        return 1;
    }
    let mut n = 0;
    for m in moves.iter() {
        let p = pos.make_move(m);
        n += perft(&p, depth - 1);
    }
    n
}

pub fn perft_divide(pos: &Position, depth: u32) -> u64 {
    let moves = legal_moves(pos);
    let mut total = 0;
    for m in moves.iter() {
        let p = pos.make_move(m);
        let n = if depth <= 1 { 1 } else { perft(&p, depth - 1) };
        println!("{}: {}", pos.move_to_uci(m), n);
        total += n;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUITE: &[(&str, &[u64])] = &[
        (START_FEN, &[20, 400, 8902, 197281, 4865609, 119060324]),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[48, 2039, 97862, 4085603, 193690690],
        ),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[14, 191, 2812, 43238, 674624, 11030083]),
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            &[6, 264, 9467, 422333, 15833292],
        ),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", &[44, 1486, 62379, 2103487, 89941194]),
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            &[46, 2079, 89890, 3894594, 164075551],
        ),
        // Chess960 positions (from the standard 960 perft set).
        ("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9", &[21, 528, 12189, 326672, 8146062]),
        ("2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9", &[21, 807, 18002, 667366, 16253601]),
        ("b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9", &[20, 479, 10471, 273318, 6417013]),
        // Hand-verified depth-1 counts for two more 960 positions (30 and 28 legal moves).
        ("qbbnrkr1/p1pppppp/1p4n1/8/2P5/6N1/PPNPPPPP/1BRKBRQ1 b FCge - 1 9", &[30]),
        ("1nbbnrkr/p1p1ppp1/3p4/1p3P1p/3Pq2P/8/PPP1P1P1/QNBBNRKR w HFhf - 0 9", &[28]),
    ];

    #[test]
    fn perft_suite_shallow() {
        for (fen, counts) in SUITE {
            let pos = Position::from_fen(fen).unwrap();
            for (i, &c) in counts.iter().enumerate().take(4) {
                assert_eq!(perft(&pos, i as u32 + 1), c, "fen {} depth {}", fen, i + 1);
            }
        }
    }

    #[test]
    #[ignore]
    fn perft_suite_deep() {
        for (fen, counts) in SUITE {
            let pos = Position::from_fen(fen).unwrap();
            for (i, &c) in counts.iter().enumerate() {
                assert_eq!(perft(&pos, i as u32 + 1), c, "fen {} depth {}", fen, i + 1);
            }
        }
    }

    #[test]
    fn fen_roundtrip() {
        for (fen, _) in SUITE.iter().take(6) {
            let pos = Position::from_fen(fen).unwrap();
            assert_eq!(pos.to_fen(), *fen);
        }
    }

    #[test]
    fn keys_are_incremental() {
        // Make random moves and compare the incremental key with a from-scratch one.
        let mut rng = 0x1234_5678u64;
        for (fen, _) in SUITE {
            let mut pos = Position::from_fen(fen).unwrap();
            for _ in 0..200 {
                let moves = legal_moves(&pos);
                if moves.is_empty() {
                    break;
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let m = moves.moves[(rng % moves.len() as u64) as usize].mv;
                pos = pos.make_move(m);
                let fresh = Position::from_fen(&pos.to_fen()).unwrap();
                assert_eq!(pos.key(), fresh.key(), "key mismatch after {} in {}", m, pos.to_fen());
                assert_eq!(pos.pawn_key(), fresh.pawn_key());
                assert_eq!(pos.non_pawn_key(Color::White), fresh.non_pawn_key(Color::White));
                assert_eq!(pos.non_pawn_key(Color::Black), fresh.non_pawn_key(Color::Black));
                assert_eq!(pos.minor_key(), fresh.minor_key());
                assert_eq!(pos.major_key(), fresh.major_key());
            }
        }
    }

    #[test]
    fn pseudo_legal_and_gives_check_agree() {
        let mut rng = 0x9876_5432u64;
        for (fen, _) in SUITE {
            let mut pos = Position::from_fen(fen).unwrap();
            for _ in 0..100 {
                let moves = legal_moves(&pos);
                if moves.is_empty() {
                    break;
                }
                for m in moves.iter() {
                    assert!(pos.is_pseudo_legal(m), "legal move {} not pseudo legal in {}", m, pos.to_fen());
                    let next = pos.make_move(m);
                    assert_eq!(pos.gives_check(m), next.in_check(), "gives_check wrong for {} in {}", m, pos.to_fen());
                }
                // Random junk moves must not be accepted as pseudo-legal if not in the legal list
                // (only check moves that are legal-by-construction: pseudo-legal implies legal or is_legal false).
                let mut all = MoveList::new();
                generate(&pos, &mut all, GenType::All);
                for m in all.iter() {
                    assert!(pos.is_pseudo_legal(m), "generated move {} rejected in {}", m, pos.to_fen());
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let m = moves.moves[(rng % moves.len() as u64) as usize].mv;
                pos = pos.make_move(m);
            }
        }
    }

    #[test]
    fn see_basic() {
        let pos = Position::from_fen("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1").unwrap();
        let m = pos.parse_uci_move("e1e5").unwrap();
        assert!(pos.see_ge(m, 100));
        let pos = Position::from_fen("1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1").unwrap();
        let m = pos.parse_uci_move("d3e5").unwrap();
        assert!(!pos.see_ge(m, 0));
        assert!(pos.see_ge(m, -200));
    }
}
