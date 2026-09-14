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
        let dot = dot_clamped(acc, row);
        // dot is scaled 256*64 = 16384; want 1024: >> 4. U and P are scale 64: << 4.
        let promo = if m.is_promo() { m.promo_type().idx() as usize } else { 0 };
        (dot >> 4) + ((self.u[from * 64 + to] as i32) << 4) + ((self.p[promo] as i32) << 4)
    }
}

#[derive(Clone, Copy)]
struct Entry {
    acc: [Align64<[i16; H]>; 2],
    dirty: DirtyPiece,
    computed: bool,
    is_null: bool,
}

/// Accumulator stack. A push records only the piece changes; accumulators are produced lazily by `ensure`
/// (walking back to the last computed ply), so the many nodes that never score quiet moves cost nothing.
pub struct PolicyState {
    stack: Vec<Entry>,
    len: usize,
}

impl Default for PolicyState {
    fn default() -> Self {
        Self::new()
    }
}

/// sum_i clamp(acc[i], 0, 256) * row[i]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn dot_clamped(acc: &[i16; H], row: &[i8; H]) -> i32 {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 is a compile-time target feature here; H is a multiple of 16 and both arrays have H elements,
    // so every load stays in bounds. madd_epi16 forms 32-bit products, so no intermediate overflow.
    unsafe {
        let zero = _mm256_setzero_si256();
        let cap = _mm256_set1_epi16(256);
        let mut sum = _mm256_setzero_si256();
        for i in 0..H / 16 {
            let a = _mm256_loadu_si256(acc.as_ptr().add(i * 16) as *const __m256i);
            let h = _mm256_min_epi16(_mm256_max_epi16(a, zero), cap);
            let r = _mm256_cvtepi8_epi16(_mm_loadu_si128(row.as_ptr().add(i * 16) as *const __m128i));
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(h, r));
        }
        let lo = _mm256_castsi256_si128(sum);
        let hi = _mm256_extracti128_si256(sum, 1);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        _mm_cvtsi128_si32(s)
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot_clamped(acc: &[i16; H], row: &[i8; H]) -> i32 {
    let mut dot: i32 = 0;
    for i in 0..H {
        dot += acc[i].clamp(0, 256) as i32 * row[i] as i32;
    }
    dot
}

impl PolicyState {
    pub fn new() -> Self {
        let blank = Entry { acc: [Align64([0; H]), Align64([0; H])], dirty: DirtyPiece::default(), computed: false, is_null: false };
        PolicyState { stack: vec![blank; MAX_PLY + 8], len: 0 }
    }

    /// Recompute both perspectives from scratch for the root position.
    pub fn reset(&mut self, pos: &Position, net: &PolicyNet) {
        self.len = 0;
        let e = &mut self.stack[0];
        e.is_null = false;
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
        e.computed = true;
        self.len = 1;
    }

    /// Push the position after `m` (played from `before`): O(1), the accumulator is produced on demand.
    #[inline]
    pub fn push(&mut self, before: &Position, m: Move) {
        debug_assert!(self.len < self.stack.len());
        let e = &mut self.stack[self.len];
        e.dirty = DirtyPiece::from_move(before, m);
        e.computed = false;
        e.is_null = false;
        self.len += 1;
    }

    /// Null move: the board is unchanged.
    #[inline]
    pub fn push_null(&mut self) {
        let e = &mut self.stack[self.len];
        e.computed = false;
        e.is_null = true;
        self.len += 1;
    }

    #[inline]
    pub fn pop(&mut self) {
        debug_assert!(self.len > 1);
        self.len -= 1;
    }

    /// Make the top accumulators valid, applying the recorded piece changes forward from the last computed ply.
    pub fn ensure(&mut self, net: &PolicyNet) {
        let top = self.len - 1;
        if self.stack[top].computed {
            return;
        }
        let mut i = top;
        while i > 0 && !self.stack[i].computed {
            i -= 1;
        }
        debug_assert!(self.stack[i].computed, "the root entry is always computed");
        for k in i + 1..=top {
            let (prev, rest) = self.stack.split_at_mut(k);
            let src = &prev[k - 1];
            let dst = &mut rest[0];
            if dst.is_null {
                dst.acc = src.acc;
            } else {
                let d = dst.dirty;
                for p in [Color::White, Color::Black] {
                    let a = &src.acc[p.idx()].0;
                    let out = &mut dst.acc[p.idx()].0;
                    out.copy_from_slice(a);
                    for j in 0..d.n_add as usize {
                        let (pc, sq) = d.adds[j];
                        let row = &net.w[feature(p, pc, sq)].0;
                        for x in 0..H {
                            out[x] = out[x].wrapping_add(row[x]);
                        }
                    }
                    for j in 0..d.n_sub as usize {
                        let (pc, sq) = d.subs[j];
                        let row = &net.w[feature(p, pc, sq)].0;
                        for x in 0..H {
                            out[x] = out[x].wrapping_sub(row[x]);
                        }
                    }
                }
            }
            dst.computed = true;
        }
    }

    /// Accumulator of perspective `stm` at the top of the stack (call `ensure` first).
    #[inline(always)]
    pub fn acc(&self, stm: Color) -> &[i16; H] {
        debug_assert!(self.stack[self.len - 1].computed);
        &self.stack[self.len - 1].acc[stm.idx()].0
    }

    /// Logits (x1024) for `moves` in the top position.
    pub fn logits(&mut self, pos: &Position, net: &PolicyNet, moves: &[Move], out: &mut [i32]) {
        self.ensure(net);
        let stm = pos.side_to_move();
        let acc = self.acc(stm);
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
    fn dot_matches_scalar_reference() {
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..200 {
            let mut acc = [0i16; H];
            let mut row = [0i8; H];
            for i in 0..H {
                acc[i] = (next() % 1200) as i16 - 400;
                row[i] = (next() % 256) as i8;
            }
            let reference: i32 = (0..H).map(|i| acc[i].clamp(0, 256) as i32 * row[i] as i32).sum();
            assert_eq!(dot_clamped(&acc, &row), reference);
        }
    }

    #[test]
    fn null_move_entries_alias_the_previous_position() {
        crate::init();
        let net = random_net(42);
        let pos = Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3").unwrap();
        let mut st = PolicyState::new();
        st.reset(&pos, &net);
        st.push_null();
        let m = legal_moves(&pos).moves[0].mv;
        let after = pos.make_move(m);
        st.push(&pos, m);
        st.ensure(&net);
        let mut fresh = PolicyState::new();
        fresh.reset(&after, &net);
        assert_eq!(st.acc(Color::White)[..], fresh.acc(Color::White)[..]);
        assert_eq!(st.acc(Color::Black)[..], fresh.acc(Color::Black)[..]);
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
                st.push(&before, m);
                // Sometimes evaluate immediately, sometimes let several plies accumulate before ensuring.
                if rnd() % 3 != 0 {
                    st.ensure(&net);
                    let mut fresh = PolicyState::new();
                    fresh.reset(&pos, &net);
                    for p in [Color::White, Color::Black] {
                        assert_eq!(st.acc(p)[..], fresh.acc(p)[..], "perspective {:?} after {}", p, m);
                    }
                }
                if rnd() % 4 == 0 {
                    st.pop();
                    pos = before;
                }
            }
        }
    }
}
