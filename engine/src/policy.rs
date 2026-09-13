//! Move-ordering policy net (runs/policy/PLAN.md): a side-to-move-relative piece-square accumulator
//! (768 -> H) with one dot product per move.
//!
//!   acc[p]     = b + sum over pieces of W[feature(piece, sq, perspective p)]        (i16, scale 256)
//!   h          = clamp(acc[stm], 0, 256)
//!   logit(m)   = h . V[pt(m)][to_rel] / (256*64) + U[from_rel][to_rel] / 64 + P[promo] / 64
//!
//! File format "ANPOL1" written by scripts/policy_train.py: H u32, W i16[768*H], b i16[H], V i8[6*64*H],
//! U i16[64*64], P i16[5], all little-endian. `logit_q` returns the logit scaled by 1024 as an i32.
//!
//! Both perspectives are kept per ply so no refresh is ever needed except at the root. Features never
//! depend on king squares.

use crate::nnue::{Align64, DirtyPiece};
use crate::position::Position;
use crate::types::MAX_PLY;
use crate::types::{Color, Move, Piece, PieceType, Square};

pub const H: usize = 256;
const FEATURES: usize = 768;

pub struct PolicyNet {
    w: Vec<Align64<[i16; H]>>,
    b: Align64<[i16; H]>,
    v: Vec<Align64<[i8; H]>>, // 6 * 64 rows
    u: Vec<i16>,               // 64 * 64
    p: [i16; 5],
}

/// Feature index of `piece` on `sq` from perspective `p`: our pieces first, ranks flipped for black.
#[inline(always)]
fn feature(p: Color, piece: Piece, sq: Square) -> usize {
    let rel_colour = if piece.color() == p { 0 } else { 6 };
    let rel_sq = if p == Color::Black { sq ^ 56 } else { sq } as usize;
    (rel_colour + piece.piece_type().idx()) * 64 + rel_sq
}

impl PolicyNet {
    pub fn load(path: &str) -> Result<PolicyNet, String> {
        let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if data.len() < 10 || &data[0..6] != b"ANPOL1" {
            return Err(format!("{path}: not an ANPOL1 policy net"));
        }
        let h = u32::from_le_bytes(data[6..10].try_into().unwrap()) as usize;
        if h != H {
            return Err(format!("{path}: H={h}, this build supports H={H}"));
        }
        let expect = 10 + FEATURES * H * 2 + H * 2 + 6 * 64 * H + 64 * 64 * 2 + 5 * 2;
        if data.len() != expect {
            return Err(format!("{path}: size {} != expected {expect}", data.len()));
        }
        let mut off = 10;
        let rd_i16 = |n: usize, off: &mut usize| -> Vec<i16> {
            let v: Vec<i16> = data[*off..*off + n * 2].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
            *off += n * 2;
            v
        };
        let w_flat = rd_i16(FEATURES * H, &mut off);
        let b_flat = rd_i16(H, &mut off);
        let v_bytes = &data[off..off + 6 * 64 * H];
        off += 6 * 64 * H;
        let u = {
            let v: Vec<i16> = data[off..off + 4096 * 2].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
            off += 4096 * 2;
            v
        };
        let mut p = [0i16; 5];
        for (i, c) in data[off..off + 10].chunks_exact(2).enumerate() {
            p[i] = i16::from_le_bytes([c[0], c[1]]);
        }
        let mut w = Vec::with_capacity(FEATURES);
        for f in 0..FEATURES {
            let mut row = Align64([0i16; H]);
            row.0.copy_from_slice(&w_flat[f * H..(f + 1) * H]);
            w.push(row);
        }
        let mut b = Align64([0i16; H]);
        b.0.copy_from_slice(&b_flat);
        let mut v = Vec::with_capacity(6 * 64);
        for r in 0..6 * 64 {
            let mut row = Align64([0i8; H]);
            for i in 0..H {
                row.0[i] = v_bytes[r * H + i] as i8;
            }
            v.push(row);
        }
        Ok(PolicyNet { w, b, v, u, p })
    }

    /// logit(m) * 1024 for the side to move, given its accumulator.
    #[inline]
    pub fn logit_q(&self, acc: &[i16; H], stm: Color, moved: PieceType, m: Move) -> i32 {
        let flip = if stm == Color::Black { 56 } else { 0 };
        let from = (m.from() ^ flip) as usize;
        let to = (m.to() ^ flip) as usize;
        let row = &self.v[moved.idx() * 64 + to].0;
        let mut dot: i32 = 0;
        for i in 0..H {
            let h = acc[i].clamp(0, 256) as i32;
            dot += h * row[i] as i32;
        }
        // dot is scaled 256*64 = 16384; want 1024: >> 4. U and P are scale 64: << 4.
        let promo = if m.is_promo() { m.promo_type().idx() as usize } else { 0 };
        (dot >> 4) + ((self.u[from * 64 + to] as i32) << 4) + ((self.p[promo] as i32) << 4)
    }
}

#[derive(Clone, Copy)]
struct Entry {
    acc: [Align64<[i16; H]>; 2],
}

pub struct PolicyState {
    stack: Vec<Entry>,
    len: usize,
}

impl Default for PolicyState {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyState {
    pub fn new() -> Self {
        PolicyState { stack: vec![Entry { acc: [Align64([0; H]), Align64([0; H])] }; MAX_PLY + 8], len: 0 }
    }

    /// Recompute both perspectives from scratch for the root position.
    pub fn reset(&mut self, pos: &Position, net: &PolicyNet) {
        self.len = 0;
        let e = &mut self.stack[0];
        for p in [Color::White, Color::Black] {
            let acc = &mut e.acc[p.idx()].0;
            acc.copy_from_slice(&net.b.0);
            let mut occ = pos.occupied();
            while occ != 0 {
                let sq = occ.trailing_zeros() as Square;
                occ &= occ - 1;
                let row = &net.w[feature(p, pos.piece_on(sq), sq)].0;
                for i in 0..H {
                    acc[i] = acc[i].wrapping_add(row[i]);
                }
            }
        }
        self.len = 1;
    }

    /// Push the position after `m` (played from `before`), updating both perspectives incrementally.
    #[inline]
    pub fn push(&mut self, before: &Position, m: Move, net: &PolicyNet) {
        let d = DirtyPiece::from_move(before, m);
        debug_assert!(self.len < self.stack.len());
        let (prev, rest) = self.stack.split_at_mut(self.len);
        let src = &prev[self.len - 1];
        let dst = &mut rest[0];
        for p in [Color::White, Color::Black] {
            let a = &src.acc[p.idx()].0;
            let out = &mut dst.acc[p.idx()].0;
            out.copy_from_slice(a);
            for k in 0..d.n_add as usize {
                let (pc, sq) = d.adds[k];
                let row = &net.w[feature(p, pc, sq)].0;
                for i in 0..H {
                    out[i] = out[i].wrapping_add(row[i]);
                }
            }
            for k in 0..d.n_sub as usize {
                let (pc, sq) = d.subs[k];
                let row = &net.w[feature(p, pc, sq)].0;
                for i in 0..H {
                    out[i] = out[i].wrapping_sub(row[i]);
                }
            }
        }
        self.len += 1;
    }

    /// Null move: the board is unchanged.
    #[inline]
    pub fn push_null(&mut self) {
        let (prev, rest) = self.stack.split_at_mut(self.len);
        rest[0] = prev[self.len - 1];
        self.len += 1;
    }

    #[inline]
    pub fn pop(&mut self) {
        debug_assert!(self.len > 1);
        self.len -= 1;
    }

    /// Accumulator of the side to move at the top of the stack.
    #[inline(always)]
    pub fn top(&self, stm: Color) -> &[i16; H] {
        &self.stack[self.len - 1].acc[stm.idx()].0
    }

    /// Logits (x1024) for `moves` in the top position.
    pub fn logits(&self, pos: &Position, net: &PolicyNet, moves: &[Move], out: &mut [i32]) {
        let stm = pos.side_to_move();
        let acc = self.top(stm);
        for (i, &m) in moves.iter().enumerate() {
            out[i] = net.logit_q(acc, stm, pos.piece_on(m.from()).piece_type(), m);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::legal_moves;

    /// A deterministic random net so the incremental update can be checked without a trained file.
    fn random_net(seed: u64) -> PolicyNet {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut w = Vec::new();
        for _ in 0..FEATURES {
            let mut row = Align64([0i16; H]);
            for x in row.0.iter_mut() {
                *x = (next() % 200) as i16 - 100;
            }
            w.push(row);
        }
        let mut b = Align64([0i16; H]);
        for x in b.0.iter_mut() {
            *x = (next() % 200) as i16 - 100;
        }
        let mut v = Vec::new();
        for _ in 0..6 * 64 {
            let mut row = Align64([0i8; H]);
            for x in row.0.iter_mut() {
                *x = (next() % 100) as i8 - 50;
            }
            v.push(row);
        }
        let u: Vec<i16> = (0..4096).map(|_| (next() % 100) as i16 - 50).collect();
        PolicyNet { w, b, v, u, p: [1, 2, 3, 4, 5] }
    }

    #[test]
    fn incremental_matches_refresh_over_random_games() {
        crate::init();
        let net = random_net(0x9E3779B97F4A7C15);
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _game in 0..20 {
            let mut pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
            let mut st = PolicyState::new();
            st.reset(&pos, &net);
            for _ply in 0..120 {
                let list = legal_moves(&pos);
                if list.len == 0 {
                    break;
                }
                let m = list.moves[(rnd() % list.len as u64) as usize].mv;
                let before = pos;
                pos = pos.make_move(m);
                st.push(&before, m, &net);
                let mut fresh = PolicyState::new();
                fresh.reset(&pos, &net);
                for p in [Color::White, Color::Black] {
                    assert_eq!(st.top(p)[..], fresh.top(p)[..], "perspective {:?} after {}", p, m);
                }
                if rnd() % 4 == 0 {
                    st.pop();
                    pos = before;
                }
            }
        }
    }
}
