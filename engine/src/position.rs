//! Board state with copy-make, FEN I/O, attack queries, legality, check detection and SEE.

use crate::bitboard::*;
use crate::types::*;
use crate::zobrist::*;

pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[derive(Clone, Copy)]
pub struct Position {
    by_type: [Bitboard; 6],
    by_color: [Bitboard; 2],
    board: [Piece; 64],
    stm: Color,
    castling: u8,
    /// Rook square for each castling right (index = bit index of the right).
    castle_rook: [Square; 4],
    /// King destination for each right and the squares that must be empty.
    castle_path: [Bitboard; 4],
    /// Castling rights cleared when a move touches this square.
    castle_mask: [u8; 64],
    ep_sq: Square,
    rule50: u8,
    game_ply: u16,
    key: u64,
    pawn_key: u64,
    non_pawn_key: [u64; 2],
    minor_key: u64,
    major_key: u64,
    checkers: Bitboard,
    blockers: [Bitboard; 2],
    pinners: [Bitboard; 2],
    check_squares: [Bitboard; 6],
    captured: Piece,
    chess960: bool,
}

impl Default for Position {
    fn default() -> Self {
        Position::from_fen(START_FEN).unwrap()
    }
}

impl Position {
    pub fn empty() -> Self {
        Position {
            by_type: [0; 6],
            by_color: [0; 2],
            board: [Piece::None; 64],
            stm: Color::White,
            castling: 0,
            castle_rook: [SQ_NONE; 4],
            castle_path: [0; 4],
            castle_mask: [0; 64],
            ep_sq: SQ_NONE,
            rule50: 0,
            game_ply: 0,
            key: 0,
            pawn_key: 0,
            non_pawn_key: [0; 2],
            minor_key: 0,
            major_key: 0,
            checkers: 0,
            blockers: [0; 2],
            pinners: [0; 2],
            check_squares: [0; 6],
            captured: Piece::None,
            chess960: false,
        }
    }

    // ---------- accessors ----------
    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.stm
    }
    #[inline(always)]
    pub fn piece_on(&self, s: Square) -> Piece {
        self.board[s as usize]
    }
    #[inline(always)]
    pub fn pieces(&self, pt: PieceType) -> Bitboard {
        self.by_type[pt.idx()]
    }
    #[inline(always)]
    pub fn pieces2(&self, a: PieceType, b: PieceType) -> Bitboard {
        self.by_type[a.idx()] | self.by_type[b.idx()]
    }
    #[inline(always)]
    pub fn colored(&self, c: Color) -> Bitboard {
        self.by_color[c.idx()]
    }
    #[inline(always)]
    pub fn pieces_c(&self, c: Color, pt: PieceType) -> Bitboard {
        self.by_color[c.idx()] & self.by_type[pt.idx()]
    }
    #[inline(always)]
    pub fn pieces_c2(&self, c: Color, a: PieceType, b: PieceType) -> Bitboard {
        self.by_color[c.idx()] & (self.by_type[a.idx()] | self.by_type[b.idx()])
    }
    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.by_color[0] | self.by_color[1]
    }
    #[inline(always)]
    pub fn king_sq(&self, c: Color) -> Square {
        lsb(self.pieces_c(c, PieceType::King))
    }
    #[inline(always)]
    pub fn ep_square(&self) -> Square {
        self.ep_sq
    }
    #[inline(always)]
    pub fn castling_rights(&self) -> u8 {
        self.castling
    }
    #[inline(always)]
    pub fn can_castle(&self, mask: u8) -> bool {
        self.castling & mask != 0
    }
    #[inline(always)]
    pub fn rule50(&self) -> u8 {
        self.rule50
    }
    #[inline(always)]
    pub fn game_ply(&self) -> u16 {
        self.game_ply
    }
    #[inline(always)]
    pub fn key(&self) -> u64 {
        self.key
    }
    #[inline(always)]
    pub fn pawn_key(&self) -> u64 {
        self.pawn_key
    }
    #[inline(always)]
    pub fn non_pawn_key(&self, c: Color) -> u64 {
        self.non_pawn_key[c.idx()]
    }
    #[inline(always)]
    pub fn minor_key(&self) -> u64 {
        self.minor_key
    }
    #[inline(always)]
    pub fn major_key(&self) -> u64 {
        self.major_key
    }
    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        self.checkers
    }
    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers != 0
    }
    #[inline(always)]
    pub fn blockers_for_king(&self, c: Color) -> Bitboard {
        self.blockers[c.idx()]
    }
    #[inline(always)]
    pub fn pinners(&self, c: Color) -> Bitboard {
        self.pinners[c.idx()]
    }
    #[inline(always)]
    pub fn check_squares(&self, pt: PieceType) -> Bitboard {
        self.check_squares[pt.idx()]
    }
    #[inline(always)]
    pub fn captured_piece(&self) -> Piece {
        self.captured
    }
    #[inline(always)]
    pub fn is_chess960(&self) -> bool {
        self.chess960
    }
    pub fn set_chess960(&mut self, v: bool) {
        self.chess960 = v;
    }
    #[inline(always)]
    pub fn castle_rook_sq(&self, right_bit: u8) -> Square {
        self.castle_rook[right_bit.trailing_zeros() as usize]
    }
    #[inline(always)]
    pub fn castle_path(&self, right_bit: u8) -> Bitboard {
        self.castle_path[right_bit.trailing_zeros() as usize]
    }
    #[inline(always)]
    pub fn non_pawn_material(&self, c: Color) -> Value {
        let mut v = 0;
        for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            v += piece_value(pt) * popcount(self.pieces_c(c, pt)) as Value;
        }
        v
    }
    #[inline(always)]
    pub fn has_non_pawn_material(&self, c: Color) -> bool {
        self.colored(c) & !self.pieces2(PieceType::Pawn, PieceType::King) != 0
    }
    #[inline(always)]
    pub fn piece_count(&self) -> u32 {
        popcount(self.occupied())
    }

    // ---------- piece manipulation ----------
    #[inline(always)]
    fn put_piece(&mut self, p: Piece, s: Square) {
        self.board[s as usize] = p;
        self.by_type[p.piece_type().idx()] |= bb(s);
        self.by_color[p.color().idx()] |= bb(s);
    }
    #[inline(always)]
    fn remove_piece(&mut self, s: Square) {
        let p = self.board[s as usize];
        self.by_type[p.piece_type().idx()] &= !bb(s);
        self.by_color[p.color().idx()] &= !bb(s);
        self.board[s as usize] = Piece::None;
    }
    #[inline(always)]
    fn move_piece(&mut self, from: Square, to: Square) {
        let p = self.board[from as usize];
        let ft = bb(from) | bb(to);
        self.by_type[p.piece_type().idx()] ^= ft;
        self.by_color[p.color().idx()] ^= ft;
        self.board[from as usize] = Piece::None;
        self.board[to as usize] = p;
    }

    // ---------- FEN ----------
    pub fn from_fen(fen: &str) -> Result<Position, String> {
        crate::bitboard::init();
        let mut pos = Position::empty();
        let mut parts = fen.split_whitespace();
        let board = parts.next().ok_or("empty fen")?;
        let mut f = 0u8;
        let mut r = 7u8;
        for ch in board.chars() {
            match ch {
                '/' => {
                    if r == 0 {
                        return Err("too many ranks".into());
                    }
                    r -= 1;
                    f = 0;
                }
                '1'..='8' => f += ch as u8 - b'0',
                _ => {
                    let p = Piece::from_char(ch).ok_or(format!("bad piece {}", ch))?;
                    if f >= 8 {
                        return Err("too many files".into());
                    }
                    pos.put_piece(p, sq(f, r));
                    f += 1;
                }
            }
        }
        if popcount(pos.pieces_c(Color::White, PieceType::King)) != 1
            || popcount(pos.pieces_c(Color::Black, PieceType::King)) != 1
        {
            return Err("need exactly one king per side".into());
        }
        pos.stm = match parts.next().unwrap_or("w") {
            "w" => Color::White,
            "b" => Color::Black,
            x => return Err(format!("bad side {}", x)),
        };
        let castling = parts.next().unwrap_or("-");
        for ch in castling.chars() {
            if ch == '-' {
                continue;
            }
            let c = if ch.is_ascii_uppercase() { Color::White } else { Color::Black };
            let ksq = pos.king_sq(c);
            let rank = relative_rank(Color::White, if c == Color::White { 0 } else { 56 });
            let rank_bb = rank_bb(rank);
            let rooks = pos.pieces_c(c, PieceType::Rook) & rank_bb;
            let rsq = match ch.to_ascii_lowercase() {
                'k' => {
                    // outermost rook on the king side
                    let mut s = SQ_NONE;
                    for x in bits(rooks) {
                        if x > ksq {
                            s = x;
                        }
                    }
                    s
                }
                'q' => {
                    let mut s = SQ_NONE;
                    for x in bits(rooks) {
                        if x < ksq {
                            s = x;
                            break;
                        }
                    }
                    s
                }
                'a'..='h' => {
                    let file = ch.to_ascii_lowercase() as u8 - b'a';
                    sq(file, rank)
                }
                _ => return Err(format!("bad castling char {}", ch)),
            };
            if rsq == SQ_NONE || pos.piece_on(rsq) != Piece::new(c, PieceType::Rook) {
                // Tolerate junk castling rights (some FENs are sloppy).
                continue;
            }
            pos.set_castling_right(c, rsq);
        }
        let ep = parts.next().unwrap_or("-");
        if ep != "-" {
            if let Some(s) = square_from_str(ep) {
                // Only accept if an en passant capture is actually possible.
                let us = pos.stm;
                let them = !us;
                let cap_sq = if us == Color::White { s - 8 } else { s + 8 };
                if pos.piece_on(cap_sq) == Piece::new(them, PieceType::Pawn)
                    && pawn_attacks(them, s) & pos.pieces_c(us, PieceType::Pawn) != 0
                    && pos.piece_on(s).is_none()
                {
                    pos.ep_sq = s;
                }
            }
        }
        pos.rule50 = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
        let fullmove: u16 = parts.next().and_then(|x| x.parse().ok()).unwrap_or(1);
        pos.game_ply = (fullmove.max(1) - 1) * 2 + (pos.stm == Color::Black) as u16;
        pos.chess960 = pos.detect_chess960();
        pos.compute_keys();
        pos.update_state();
        Ok(pos)
    }

    fn detect_chess960(&self) -> bool {
        // Chess960 if castling rooks are not on standard squares or king not on e-file with rights.
        for i in 0..4 {
            if self.castling & (1 << i) == 0 {
                continue;
            }
            let c = if i < 2 { Color::White } else { Color::Black };
            let std_rook = match i {
                0 => squares::H1,
                1 => squares::A1,
                2 => squares::H8,
                _ => squares::A8,
            };
            let std_king = if c == Color::White { squares::E1 } else { squares::E8 };
            if self.castle_rook[i] != std_rook || self.king_sq(c) != std_king {
                return true;
            }
        }
        false
    }

    fn set_castling_right(&mut self, c: Color, rsq: Square) {
        let ksq = self.king_sq(c);
        let kingside = rsq > ksq;
        let bit = match (c, kingside) {
            (Color::White, true) => CASTLE_WK,
            (Color::White, false) => CASTLE_WQ,
            (Color::Black, true) => CASTLE_BK,
            (Color::Black, false) => CASTLE_BQ,
        };
        let i = bit.trailing_zeros() as usize;
        self.castling |= bit;
        self.castle_rook[i] = rsq;
        self.castle_mask[ksq as usize] |= bit;
        self.castle_mask[rsq as usize] |= bit;
        let kto = if kingside { sq(6, rank_of(ksq)) } else { sq(2, rank_of(ksq)) };
        let rto = if kingside { sq(5, rank_of(ksq)) } else { sq(3, rank_of(ksq)) };
        // Squares that must be empty: between king and its target, and rook and its target, excluding king and rook.
        let mut path = 0;
        let (lo, hi) = (ksq.min(kto), ksq.max(kto));
        for s in lo..=hi {
            path |= bb(s);
        }
        let (lo, hi) = (rsq.min(rto), rsq.max(rto));
        for s in lo..=hi {
            path |= bb(s);
        }
        path &= !(bb(ksq) | bb(rsq));
        self.castle_path[i] = path;
    }

    /// King destination square for a castling move encoded king->rook.
    #[inline(always)]
    pub fn castle_king_to(&self, m: Move) -> Square {
        let kingside = m.to() > m.from();
        sq(if kingside { 6 } else { 2 }, rank_of(m.from()))
    }
    #[inline(always)]
    pub fn castle_rook_to(&self, m: Move) -> Square {
        let kingside = m.to() > m.from();
        sq(if kingside { 5 } else { 3 }, rank_of(m.from()))
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for r in (0..8).rev() {
            let mut empty = 0;
            for f in 0..8 {
                let p = self.piece_on(sq(f, r));
                if p.is_none() {
                    empty += 1;
                } else {
                    if empty > 0 {
                        s.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    s.push(p.to_char());
                }
            }
            if empty > 0 {
                s.push((b'0' + empty) as char);
            }
            if r > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.stm == Color::White { 'w' } else { 'b' });
        s.push(' ');
        if self.castling == 0 {
            s.push('-');
        } else {
            for (bit, ch) in [(CASTLE_WK, 'K'), (CASTLE_WQ, 'Q'), (CASTLE_BK, 'k'), (CASTLE_BQ, 'q')] {
                if self.castling & bit != 0 {
                    if self.chess960 {
                        let f = (b'a' + file_of(self.castle_rook_sq(bit))) as char;
                        s.push(if ch.is_ascii_uppercase() { f.to_ascii_uppercase() } else { f });
                    } else {
                        s.push(ch);
                    }
                }
            }
        }
        s.push(' ');
        s.push_str(&square_to_string(self.ep_sq));
        s.push_str(&format!(" {} {}", self.rule50, self.game_ply / 2 + 1));
        s
    }

    fn compute_keys(&mut self) {
        let mut key = 0;
        let mut pawn_key = ZOBRIST.no_pawns;
        let mut npk = [0u64; 2];
        let mut minor = 0;
        let mut major = 0;
        for s in bits(self.occupied()) {
            let p = self.piece_on(s);
            let k = psq_key(p, s);
            key ^= k;
            match p.piece_type() {
                PieceType::Pawn => pawn_key ^= k,
                PieceType::Knight | PieceType::Bishop => {
                    npk[p.color().idx()] ^= k;
                    minor ^= k;
                }
                PieceType::Rook | PieceType::Queen => {
                    npk[p.color().idx()] ^= k;
                    major ^= k;
                }
                PieceType::King => {
                    npk[p.color().idx()] ^= k;
                    minor ^= k;
                    major ^= k;
                }
            }
        }
        if self.ep_sq != SQ_NONE {
            key ^= ZOBRIST.ep[file_of(self.ep_sq) as usize];
        }
        key ^= ZOBRIST.castling[self.castling as usize];
        if self.stm == Color::Black {
            key ^= ZOBRIST.side;
        }
        self.key = key;
        self.pawn_key = pawn_key;
        self.non_pawn_key = npk;
        self.minor_key = minor;
        self.major_key = major;
    }

    // ---------- attacks ----------
    /// All pieces (both colors) attacking `s` given occupancy `occ`.
    #[inline(always)]
    pub fn attackers_to_occ(&self, s: Square, occ: Bitboard) -> Bitboard {
        (pawn_attacks(Color::Black, s) & self.pieces_c(Color::White, PieceType::Pawn))
            | (pawn_attacks(Color::White, s) & self.pieces_c(Color::Black, PieceType::Pawn))
            | (knight_attacks(s) & self.pieces(PieceType::Knight))
            | (rook_attacks(s, occ) & self.pieces2(PieceType::Rook, PieceType::Queen))
            | (bishop_attacks(s, occ) & self.pieces2(PieceType::Bishop, PieceType::Queen))
            | (king_attacks(s) & self.pieces(PieceType::King))
    }
    #[inline(always)]
    pub fn attackers_to(&self, s: Square) -> Bitboard {
        self.attackers_to_occ(s, self.occupied())
    }
    #[inline(always)]
    pub fn attacked_by(&self, c: Color, s: Square) -> bool {
        self.attackers_to(s) & self.colored(c) != 0
    }

    /// Pieces of both colors that block a slider attack from `sliders` towards `s`.
    fn slider_blockers(&self, sliders: Bitboard, s: Square, pinners: &mut Bitboard) -> Bitboard {
        let mut blockers = 0;
        *pinners = 0;
        let snipers = ((rook_attacks(s, 0) & self.pieces2(PieceType::Rook, PieceType::Queen))
            | (bishop_attacks(s, 0) & self.pieces2(PieceType::Bishop, PieceType::Queen)))
            & sliders;
        let occ = self.occupied() ^ snipers;
        for sniper in bits(snipers) {
            let b = between(s, sniper) & occ;
            if b != 0 && !more_than_one(b) {
                blockers |= b;
                if b & self.colored(self.piece_on(s).color()) != 0 {
                    *pinners |= bb(sniper);
                }
            }
        }
        blockers
    }

    fn update_state(&mut self) {
        let us = self.stm;
        let them = !us;
        let ksq = self.king_sq(us);
        self.checkers = self.attackers_to(ksq) & self.colored(them);
        let mut p = 0;
        self.blockers[us.idx()] = self.slider_blockers(self.colored(them), ksq, &mut p);
        self.pinners[them.idx()] = p;
        let eksq = self.king_sq(them);
        self.blockers[them.idx()] = self.slider_blockers(self.colored(us), eksq, &mut p);
        self.pinners[us.idx()] = p;
        let occ = self.occupied();
        self.check_squares[PieceType::Pawn.idx()] = pawn_attacks(them, eksq);
        self.check_squares[PieceType::Knight.idx()] = knight_attacks(eksq);
        self.check_squares[PieceType::Bishop.idx()] = bishop_attacks(eksq, occ);
        self.check_squares[PieceType::Rook.idx()] = rook_attacks(eksq, occ);
        self.check_squares[PieceType::Queen.idx()] =
            self.check_squares[PieceType::Bishop.idx()] | self.check_squares[PieceType::Rook.idx()];
        self.check_squares[PieceType::King.idx()] = 0;
    }

    // ---------- legality ----------
    /// Whether a pseudo-legal move is legal.
    pub fn is_legal(&self, m: Move) -> bool {
        let us = self.stm;
        let from = m.from();
        let to = m.to();
        let ksq = self.king_sq(us);
        debug_assert!(!self.piece_on(from).is_none() && self.piece_on(from).color() == us);

        if m.is_ep() {
            let cap = if us == Color::White { to - 8 } else { to + 8 };
            let occ = (self.occupied() ^ bb(from) ^ bb(cap)) | bb(to);
            return (rook_attacks(ksq, occ) & self.pieces_c2(!us, PieceType::Rook, PieceType::Queen)) == 0
                && (bishop_attacks(ksq, occ) & self.pieces_c2(!us, PieceType::Bishop, PieceType::Queen)) == 0;
        }
        if m.is_castle() {
            // to = rook square. King path must not be attacked; rook must not be pinned (960).
            let kto = self.castle_king_to(m);
            let step: i8 = if kto > from { -1 } else { 1 };
            let mut s = kto as i8;
            while s != from as i8 {
                if self.attackers_to(s as Square) & self.colored(!us) != 0 {
                    return false;
                }
                s += step;
            }
            return !self.chess960 || (self.blockers_for_king(us) & bb(to)) == 0;
        }
        if self.piece_on(from).piece_type() == PieceType::King {
            return self.attackers_to_occ(to, self.occupied() ^ bb(from)) & self.colored(!us) == 0;
        }
        // Non-king piece: legal if not pinned or moving along the pin ray.
        (self.blockers_for_king(us) & bb(from)) == 0 || aligned(from, to, ksq)
    }

    /// Whether a move (e.g. from the TT) is pseudo-legal in this position.
    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        if m.is_none() {
            return false;
        }
        let us = self.stm;
        let from = m.from();
        let to = m.to();
        let pc = self.piece_on(from);
        if pc.is_none() || pc.color() != us {
            return false;
        }
        let pt = pc.piece_type();
        if m.is_castle() {
            if pt != PieceType::King || self.in_check() {
                return false;
            }
            let kingside = to > from;
            let bit = match (us, kingside) {
                (Color::White, true) => CASTLE_WK,
                (Color::White, false) => CASTLE_WQ,
                (Color::Black, true) => CASTLE_BK,
                (Color::Black, false) => CASTLE_BQ,
            };
            return self.can_castle(bit)
                && self.castle_rook_sq(bit) == to
                && (self.castle_path(bit) & self.occupied()) == 0;
        }
        if m.is_ep() {
            return pt == PieceType::Pawn && to == self.ep_sq && (pawn_attacks(us, from) & bb(to)) != 0;
        }
        if m.is_promo() {
            if pt != PieceType::Pawn || relative_rank(us, to) != 7 {
                return false;
            }
        } else if pt == PieceType::Pawn && relative_rank(us, to) == 7 {
            return false;
        }
        let target = self.piece_on(to);
        if !target.is_none() && (target.color() == us || target.piece_type() == PieceType::King) {
            return false;
        }
        if pt == PieceType::Pawn {
            let push = if us == Color::White { 8i8 } else { -8 };
            let f = from as i8;
            if target.is_none() {
                if to as i8 == f + push {
                    // single push
                } else if to as i8 == f + 2 * push
                    && relative_rank(us, from) == 1
                    && self.piece_on((f + push) as Square).is_none()
                {
                    // double push
                } else {
                    return false;
                }
            } else if pawn_attacks(us, from) & bb(to) == 0 {
                return false;
            }
        } else if piece_attacks(pt, from, self.occupied()) & bb(to) == 0 {
            return false;
        }
        // In check: must resolve the check (non-king moves must capture checker or block a single check).
        if self.in_check() && pt != PieceType::King {
            if more_than_one(self.checkers) {
                return false;
            }
            let ck = lsb(self.checkers);
            if (between(ck, self.king_sq(us)) | bb(ck)) & bb(to) == 0 {
                return false;
            }
        }
        true
    }

    /// Does the move give check?
    pub fn gives_check(&self, m: Move) -> bool {
        let us = self.stm;
        let from = m.from();
        let to = m.to();
        let pt = self.piece_on(from).piece_type();
        let eksq = self.king_sq(!us);
        // Direct check.
        if !m.is_castle() && !m.is_promo() && self.check_squares(pt) & bb(to) != 0 {
            return true;
        }
        // Discovered check.
        if self.blockers_for_king(!us) & bb(from) != 0 && !aligned(from, to, eksq) && !m.is_castle() {
            return true;
        }
        match m.flag() {
            FLAG_NORMAL => false,
            FLAG_PROMO => {
                piece_attacks(m.promo_type(), to, self.occupied() ^ bb(from)) & bb(eksq) != 0
                    || (self.blockers_for_king(!us) & bb(from) != 0 && !aligned(from, to, eksq))
            }
            FLAG_EP => {
                let cap = sq(file_of(to), rank_of(from));
                let occ = (self.occupied() ^ bb(from) ^ bb(cap)) | bb(to);
                (rook_attacks(eksq, occ) & self.pieces_c2(us, PieceType::Rook, PieceType::Queen)) != 0
                    || (bishop_attacks(eksq, occ) & self.pieces_c2(us, PieceType::Bishop, PieceType::Queen)) != 0
            }
            _ => {
                // castling: does the rook give check?
                let rto = self.castle_rook_to(m);
                let kto = self.castle_king_to(m);
                let occ = (self.occupied() ^ bb(from) ^ bb(to)) | bb(rto) | bb(kto);
                rook_attacks(rto, occ) & bb(eksq) != 0
            }
        }
    }

    #[inline(always)]
    pub fn is_capture(&self, m: Move) -> bool {
        (!self.piece_on(m.to()).is_none() && !m.is_castle()) || m.is_ep()
    }
    /// Capture or queen promotion ("noisy").
    #[inline(always)]
    pub fn is_capture_or_promo(&self, m: Move) -> bool {
        self.is_capture(m) || m.is_promo()
    }
    #[inline(always)]
    pub fn moved_piece(&self, m: Move) -> Piece {
        self.piece_on(m.from())
    }
    /// Piece type captured by `m` (Pawn for ep), or None.
    #[inline(always)]
    pub fn captured_type(&self, m: Move) -> Option<PieceType> {
        if m.is_ep() {
            Some(PieceType::Pawn)
        } else if m.is_castle() {
            None
        } else {
            let p = self.piece_on(m.to());
            if p.is_none() {
                None
            } else {
                Some(p.piece_type())
            }
        }
    }

    // ---------- make move ----------
    /// Returns a new position with the move applied. Move must be legal.
    pub fn make_move(&self, m: Move) -> Position {
        let mut p = *self;
        p.do_move(m);
        p
    }

    fn do_move(&mut self, m: Move) {
        let us = self.stm;
        let them = !us;
        let from = m.from();
        let mut to = m.to();
        let pc = self.piece_on(from);
        let pt = pc.piece_type();
        let mut key = self.key ^ ZOBRIST.side;

        self.game_ply += 1;
        self.rule50 += 1;
        self.captured = Piece::None;

        // Clear ep.
        if self.ep_sq != SQ_NONE {
            key ^= ZOBRIST.ep[file_of(self.ep_sq) as usize];
            self.ep_sq = SQ_NONE;
        }

        if m.is_castle() {
            // King "captures" its own rook.
            let rfrom = to;
            let kto = self.castle_king_to(m);
            let rto = self.castle_rook_to(m);
            self.remove_piece(from);
            self.remove_piece(rfrom);
            let rook = Piece::new(us, PieceType::Rook);
            self.put_piece(pc, kto);
            self.put_piece(rook, rto);
            key ^= psq_key(pc, from) ^ psq_key(pc, kto) ^ psq_key(rook, rfrom) ^ psq_key(rook, rto);
            let nk = psq_key(pc, from) ^ psq_key(pc, kto) ^ psq_key(rook, rfrom) ^ psq_key(rook, rto);
            self.non_pawn_key[us.idx()] ^= nk;
            self.major_key ^= nk;
            self.minor_key ^= psq_key(pc, from) ^ psq_key(pc, kto);
            to = kto;
        } else {
            let mut captured = self.piece_on(to);
            let mut cap_sq = to;
            if m.is_ep() {
                cap_sq = if us == Color::White { to - 8 } else { to + 8 };
                captured = Piece::new(them, PieceType::Pawn);
            }
            if !captured.is_none() {
                self.remove_piece(cap_sq);
                let ck = psq_key(captured, cap_sq);
                key ^= ck;
                match captured.piece_type() {
                    PieceType::Pawn => self.pawn_key ^= ck,
                    PieceType::Knight | PieceType::Bishop => {
                        self.non_pawn_key[them.idx()] ^= ck;
                        self.minor_key ^= ck;
                    }
                    _ => {
                        self.non_pawn_key[them.idx()] ^= ck;
                        self.major_key ^= ck;
                    }
                }
                self.rule50 = 0;
                self.captured = captured;
            }
            self.move_piece(from, to);
            let mk = psq_key(pc, from) ^ psq_key(pc, to);
            key ^= mk;
            match pt {
                PieceType::Pawn => {
                    self.pawn_key ^= mk;
                    self.rule50 = 0;
                    // Double push: set ep if capturable.
                    if (to as i16 - from as i16).abs() == 16 {
                        let ep = (from + to) / 2;
                        if pawn_attacks(us, ep) & self.pieces_c(them, PieceType::Pawn) != 0 {
                            self.ep_sq = ep;
                            key ^= ZOBRIST.ep[file_of(ep) as usize];
                        }
                    }
                    if m.is_promo() {
                        let promo = Piece::new(us, m.promo_type());
                        self.remove_piece(to);
                        self.put_piece(promo, to);
                        key ^= psq_key(pc, to) ^ psq_key(promo, to);
                        self.pawn_key ^= psq_key(pc, to);
                        let pk = psq_key(promo, to);
                        self.non_pawn_key[us.idx()] ^= pk;
                        match m.promo_type() {
                            PieceType::Knight | PieceType::Bishop => self.minor_key ^= pk,
                            _ => self.major_key ^= pk,
                        }
                    }
                }
                PieceType::Knight | PieceType::Bishop => {
                    self.non_pawn_key[us.idx()] ^= mk;
                    self.minor_key ^= mk;
                }
                PieceType::Rook | PieceType::Queen => {
                    self.non_pawn_key[us.idx()] ^= mk;
                    self.major_key ^= mk;
                }
                PieceType::King => {
                    self.non_pawn_key[us.idx()] ^= mk;
                    self.minor_key ^= mk;
                    self.major_key ^= mk;
                }
            }
        }

        // Castling rights.
        let cm = self.castle_mask[from as usize] | self.castle_mask[m.to() as usize];
        if self.castling & cm != 0 {
            key ^= ZOBRIST.castling[self.castling as usize];
            self.castling &= !cm;
            key ^= ZOBRIST.castling[self.castling as usize];
        }
        let _ = to;

        self.key = key;
        self.stm = them;
        self.update_state();
    }

    pub fn make_null_move(&self) -> Position {
        let mut p = *self;
        p.key ^= ZOBRIST.side;
        if p.ep_sq != SQ_NONE {
            p.key ^= ZOBRIST.ep[file_of(p.ep_sq) as usize];
            p.ep_sq = SQ_NONE;
        }
        p.stm = !p.stm;
        p.rule50 += 1;
        p.game_ply += 1;
        p.captured = Piece::None;
        p.update_state();
        p
    }

    // ---------- SEE ----------
    /// Static exchange evaluation: true if the exchange on `m` is >= threshold.
    pub fn see_ge(&self, m: Move, threshold: Value) -> bool {
        if m.flag() != FLAG_NORMAL {
            return threshold <= 0;
        }
        let from = m.from();
        let to = m.to();
        let mut swap = if self.piece_on(to).is_none() { 0 } else { piece_value(self.piece_on(to).piece_type()) } - threshold;
        if swap < 0 {
            return false;
        }
        swap = piece_value(self.piece_on(from).piece_type()) - swap;
        if swap <= 0 {
            return true;
        }
        let mut occ = self.occupied() ^ bb(from) ^ bb(to);
        let mut stm = self.stm;
        let mut attackers = self.attackers_to_occ(to, occ);
        let mut res = 1;
        loop {
            stm = !stm;
            attackers &= occ;
            let stm_att = attackers & self.colored(stm);
            if stm_att == 0 {
                break;
            }
            // Pinned pieces can't capture unless the pinner is gone.
            let stm_att = if self.pinners(!stm) & occ != 0 {
                stm_att & !self.blockers_for_king(stm)
            } else {
                stm_att
            };
            if stm_att == 0 {
                break;
            }
            res ^= 1;
            let mut b;
            if {
                b = stm_att & self.pieces(PieceType::Pawn);
                b != 0
            } {
                swap = piece_value(PieceType::Pawn) - swap;
                if swap < res {
                    break;
                }
                occ ^= bb(lsb(b));
                attackers |= bishop_attacks(to, occ) & self.pieces2(PieceType::Bishop, PieceType::Queen);
            } else if {
                b = stm_att & self.pieces(PieceType::Knight);
                b != 0
            } {
                swap = piece_value(PieceType::Knight) - swap;
                if swap < res {
                    break;
                }
                occ ^= bb(lsb(b));
            } else if {
                b = stm_att & self.pieces(PieceType::Bishop);
                b != 0
            } {
                swap = piece_value(PieceType::Bishop) - swap;
                if swap < res {
                    break;
                }
                occ ^= bb(lsb(b));
                attackers |= bishop_attacks(to, occ) & self.pieces2(PieceType::Bishop, PieceType::Queen);
            } else if {
                b = stm_att & self.pieces(PieceType::Rook);
                b != 0
            } {
                swap = piece_value(PieceType::Rook) - swap;
                if swap < res {
                    break;
                }
                occ ^= bb(lsb(b));
                attackers |= rook_attacks(to, occ) & self.pieces2(PieceType::Rook, PieceType::Queen);
            } else if {
                b = stm_att & self.pieces(PieceType::Queen);
                b != 0
            } {
                swap = piece_value(PieceType::Queen) - swap;
                if swap < res {
                    break;
                }
                occ ^= bb(lsb(b));
                attackers |= (bishop_attacks(to, occ) & self.pieces2(PieceType::Bishop, PieceType::Queen))
                    | (rook_attacks(to, occ) & self.pieces2(PieceType::Rook, PieceType::Queen));
            } else {
                // King: if the other side still has attackers, our king can't capture.
                return if attackers & !self.colored(stm) != 0 { res == 0 } else { res != 0 };
            }
        }
        res != 0
    }

    // ---------- draw helpers ----------
    /// Insufficient material (bare kings, or a lone minor).
    pub fn is_insufficient_material(&self) -> bool {
        if self.pieces(PieceType::Pawn) | self.pieces2(PieceType::Rook, PieceType::Queen) != 0 {
            return false;
        }
        let minors = self.pieces2(PieceType::Knight, PieceType::Bishop);
        popcount(minors) <= 1
    }

    // ---------- UCI move parsing ----------
    pub fn parse_uci_move(&self, s: &str) -> Option<Move> {
        let from = square_from_str(&s[0..2])?;
        let mut to = square_from_str(&s[2..4])?;
        let pc = self.piece_on(from);
        if pc.is_none() {
            return None;
        }
        if s.len() >= 5 {
            let pt = PieceType::from_char(s.as_bytes()[4] as char)?;
            return Some(Move::new_promo(from, to, pt));
        }
        if pc.piece_type() == PieceType::King {
            let target = self.piece_on(to);
            if !target.is_none() && target.color() == self.stm && target.piece_type() == PieceType::Rook {
                return Some(Move::new_flag(from, to, FLAG_CASTLE));
            }
            if !self.chess960 && distance(from, to) == 2 && rank_of(from) == rank_of(to) {
                let bit = if to > from {
                    if self.stm == Color::White { CASTLE_WK } else { CASTLE_BK }
                } else if self.stm == Color::White {
                    CASTLE_WQ
                } else {
                    CASTLE_BQ
                };
                to = self.castle_rook_sq(bit);
                if to == SQ_NONE {
                    return None;
                }
                return Some(Move::new_flag(from, to, FLAG_CASTLE));
            }
        }
        if pc.piece_type() == PieceType::Pawn && to == self.ep_sq {
            return Some(Move::new_flag(from, to, FLAG_EP));
        }
        Some(Move::new(from, to))
    }

    pub fn move_to_uci(&self, m: Move) -> String {
        if m.is_castle() && !self.chess960 {
            let kto = self.castle_king_to(m);
            return format!("{}{}", square_to_string(m.from()), square_to_string(kto));
        }
        m.to_string()
    }

    pub fn pretty(&self) -> String {
        let mut s = String::new();
        for r in (0..8).rev() {
            for f in 0..8 {
                s.push(self.piece_on(sq(f, r)).to_char());
                s.push(' ');
            }
            s.push('\n');
        }
        s.push_str(&format!("fen: {}\nkey: {:016x}\n", self.to_fen(), self.key));
        s
    }
}
