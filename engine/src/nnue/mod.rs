//! NNUE evaluation: (768 x 16 king buckets, horizontally mirrored -> 1024)x2 -> 1 x 8 output buckets.
//!
//! File layout (bullet `quantised.bin`, little-endian, all i16):
//!   l0w [INPUT_BUCKETS*768][L1]   feature weights (column-major = one contiguous column per feature)
//!   l0b [L1]                      feature bias
//!   l1w [OUTPUT_BUCKETS][2*L1]    output weights (transposed on save)
//!   l1b [OUTPUT_BUCKETS]          output bias
//!   padding with the bytes of "bullet" to a multiple of 64.
//!
//! Feature index for perspective P and piece (c, pt, sq):
//!   ksq' = king_sq(P) ^ (56 if P == Black); flip = 7 if file(ksq') > 3 else 0
//!   idx  = 768 * BUCKET[ksq'] + ((384 * (c != P) + 64 * pt + (sq ^ (56 if P == Black))) ^ flip)
//! This matches bullet's `ChessBucketsMirrored` + `Chess768` exactly (checked at rev 629ee50).

pub mod simd;
pub mod threats;
pub mod v3;

use crate::bitboard::*;
use crate::position::Position;
use crate::types::*;

pub const L1: usize = 1024;
pub const INPUT_BUCKETS: usize = 16;
pub const OUTPUT_BUCKETS: usize = 8;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const SCALE: i32 = 400;
pub const FEATURES: usize = 768 * INPUT_BUCKETS;

#[rustfmt::skip]
pub const BUCKET_LAYOUT: [u8; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

/// Bucket for a king square already relative to the perspective (0..64).
#[inline(always)]
pub fn king_bucket(rel_ksq: Square) -> usize {
    let file = file_of(rel_ksq);
    let mf = if file > 3 { 7 - file } else { file };
    BUCKET_LAYOUT[(rank_of(rel_ksq) * 4 + mf) as usize] as usize
}

#[inline(always)]
pub fn output_bucket(pos: &Position) -> usize {
    ((pos.piece_count() as usize - 2) / ((32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS)).min(OUTPUT_BUCKETS - 1)
}

/// Feature index for perspective `p`.
#[inline(always)]
pub fn feature_index(p: Color, rel_ksq: Square, piece: Piece, s: Square) -> usize {
    let flip = if file_of(rel_ksq) > 3 { 7 } else { 0 };
    let rel_sq = if p == Color::Black { s ^ 56 } else { s } as usize;
    let color_off = if piece.color() == p { 0 } else { 384 };
    768 * king_bucket(rel_ksq) + ((color_off + 64 * piece.piece_type().idx() + rel_sq) ^ flip)
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Align64<T>(pub T);

/// Any supported network architecture, detected from the file size.
pub enum AnyNet {
    V1(Network),
    V3(v3::NetworkV3),
}

impl AnyNet {
    pub fn from_bytes(bytes: &[u8]) -> Result<AnyNet, String> {
        if bytes.len() >= v3::NET_BYTES_UNPADDED && bytes.len() <= v3::NET_BYTES_UNPADDED + 64 {
            v3::NetworkV3::from_bytes(bytes).map(AnyNet::V3)
        } else {
            Network::from_bytes(bytes).map(AnyNet::V1)
        }
    }
    pub fn load(path: &str) -> Result<AnyNet, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
        AnyNet::from_bytes(&bytes)
    }
    pub fn embedded() -> Option<AnyNet> {
        #[cfg(embedded_net)]
        {
            static BYTES: &[u8] = include_bytes!(env!("EMBEDDED_NET_PATH"));
            return AnyNet::from_bytes(BYTES).ok();
        }
        #[cfg(not(embedded_net))]
        {
            None
        }
    }
    pub fn arch_name(&self) -> &'static str {
        match self {
            AnyNet::V1(_) => "v1 (768x16hm->1024)x2->1x8",
            AnyNet::V3(_) => "v3 threats+pawnpairs (768x16hm+64368->1024)x2 pairwise ->16->32->1 x8",
        }
    }
}

/// Per-thread evaluation state for whichever architecture is loaded.
pub enum AnyState {
    V1(NnueState),
    V3(v3::StateV3),
}

impl AnyState {
    pub fn for_net(net: Option<&AnyNet>) -> AnyState {
        match net {
            Some(AnyNet::V3(_)) => AnyState::V3(v3::StateV3::new()),
            _ => AnyState::V1(NnueState::new()),
        }
    }
    #[inline]
    pub fn reset(&mut self, pos: &Position, net: &AnyNet) {
        match (self, net) {
            (AnyState::V1(s), AnyNet::V1(n)) => s.reset(pos, n),
            (AnyState::V3(s), AnyNet::V3(n)) => s.reset(pos, n),
            _ => panic!("nnue state / network architecture mismatch"),
        }
    }
    #[inline]
    pub fn push(&mut self, before: &Position, m: Move, after: &Position) {
        match self {
            AnyState::V1(s) => s.push(before, m, after),
            AnyState::V3(s) => s.push(before, m, after),
        }
    }
    #[inline]
    pub fn push_null(&mut self, after: &Position) {
        match self {
            AnyState::V1(s) => s.push_null(),
            AnyState::V3(s) => s.push_null(after),
        }
    }
    #[inline]
    pub fn pop(&mut self) {
        match self {
            AnyState::V1(s) => s.pop(),
            AnyState::V3(s) => s.pop(),
        }
    }
    #[inline]
    pub fn evaluate(&mut self, pos: &Position, net: &AnyNet) -> Value {
        match (self, net) {
            (AnyState::V1(s), AnyNet::V1(n)) => s.evaluate(pos, n),
            (AnyState::V3(s), AnyNet::V3(n)) => s.evaluate(n),
            _ => panic!("nnue state / network architecture mismatch"),
        }
    }
    /// From-scratch evaluation (tests / netcheck).
    pub fn evaluate_reference(pos: &Position, net: &AnyNet) -> Value {
        match net {
            AnyNet::V1(n) => NnueState::evaluate_reference(pos, n),
            AnyNet::V3(n) => n.evaluate_reference(pos),
        }
    }
}

pub struct Network {
    pub ft_weights: Vec<Align64<[i16; L1]>>, // FEATURES entries
    pub ft_bias: Align64<[i16; L1]>,
    pub out_weights: Vec<Align64<[i16; 2 * L1]>>, // OUTPUT_BUCKETS entries
    pub out_bias: [i16; OUTPUT_BUCKETS],
}

pub const NET_BYTES_UNPADDED: usize = FEATURES * L1 * 2 + L1 * 2 + OUTPUT_BUCKETS * 2 * L1 * 2 + OUTPUT_BUCKETS * 2;

impl Network {
    pub fn from_bytes(bytes: &[u8]) -> Result<Network, String> {
        if bytes.len() < NET_BYTES_UNPADDED {
            return Err(format!("network file too small: {} < {}", bytes.len(), NET_BYTES_UNPADDED));
        }
        if bytes.len() > NET_BYTES_UNPADDED + 64 {
            return Err(format!(
                "network file too large for this architecture: {} (expected {}..{})",
                bytes.len(),
                NET_BYTES_UNPADDED,
                NET_BYTES_UNPADDED + 63
            ));
        }
        let mut off = 0usize;
        let mut rd = |n: usize| -> Vec<i16> {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(i16::from_le_bytes([bytes[off + 2 * i], bytes[off + 2 * i + 1]]));
            }
            off += 2 * n;
            v
        };
        let ftw = rd(FEATURES * L1);
        let ftb = rd(L1);
        let ow = rd(OUTPUT_BUCKETS * 2 * L1);
        let ob = rd(OUTPUT_BUCKETS);
        let mut ft_weights = Vec::with_capacity(FEATURES);
        for f in 0..FEATURES {
            let mut a = Align64([0i16; L1]);
            a.0.copy_from_slice(&ftw[f * L1..(f + 1) * L1]);
            ft_weights.push(a);
        }
        let mut ft_bias = Align64([0i16; L1]);
        ft_bias.0.copy_from_slice(&ftb);
        let mut out_weights = Vec::with_capacity(OUTPUT_BUCKETS);
        for b in 0..OUTPUT_BUCKETS {
            let mut a = Align64([0i16; 2 * L1]);
            a.0.copy_from_slice(&ow[b * 2 * L1..(b + 1) * 2 * L1]);
            out_weights.push(a);
        }
        let mut out_bias = [0i16; OUTPUT_BUCKETS];
        out_bias.copy_from_slice(&ob);
        Ok(Network { ft_weights, ft_bias, out_weights, out_bias })
    }

    /// The network compiled into the binary (nets/default.bin at build time), if any.
    pub fn embedded() -> Option<Network> {
        #[cfg(embedded_net)]
        {
            static BYTES: &[u8] = include_bytes!(env!("EMBEDDED_NET_PATH"));
            return Network::from_bytes(BYTES).ok();
        }
        #[cfg(not(embedded_net))]
        {
            None
        }
    }

    pub fn load(path: &str) -> Result<Network, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
        Network::from_bytes(&bytes)
    }

    /// Deterministic pseudo-random network for tests.
    pub fn random(seed: u64) -> Network {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut ft_weights = Vec::with_capacity(FEATURES);
        for _ in 0..FEATURES {
            let mut a = Align64([0i16; L1]);
            for v in a.0.iter_mut() {
                *v = (next() % 61) as i16 - 30;
            }
            ft_weights.push(a);
        }
        let mut ft_bias = Align64([0i16; L1]);
        for v in ft_bias.0.iter_mut() {
            *v = (next() % 201) as i16 - 100;
        }
        let mut out_weights = Vec::with_capacity(OUTPUT_BUCKETS);
        for _ in 0..OUTPUT_BUCKETS {
            let mut a = Align64([0i16; 2 * L1]);
            for v in a.0.iter_mut() {
                *v = (next() % 255) as i16 - 127;
            }
            out_weights.push(a);
        }
        let mut out_bias = [0i16; OUTPUT_BUCKETS];
        for v in out_bias.iter_mut() {
            *v = (next() % 2001) as i16 - 1000;
        }
        Network { ft_weights, ft_bias, out_weights, out_bias }
    }

    /// Serialize in the bullet quantised.bin layout (used by tests / tools).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(NET_BYTES_UNPADDED + 64);
        for f in &self.ft_weights {
            for v in f.0.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in self.ft_bias.0.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for b in &self.out_weights {
            for v in b.0.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in self.out_bias.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let pad = (64 - out.len() % 64) % 64;
        let chs = b"bullet";
        for i in 0..pad {
            out.push(chs[i % 6]);
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// Accumulators
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Accumulator {
    pub vals: [Align64<[i16; L1]>; 2],
    pub computed: [bool; 2],
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator { vals: [Align64([0; L1]); 2], computed: [false; 2] }
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Piece changes caused by one move, as (piece, square) pairs.
#[derive(Clone, Copy, Default)]
pub struct DirtyPiece {
    pub adds: [(Piece, Square); 2],
    pub subs: [(Piece, Square); 2],
    pub n_add: u8,
    pub n_sub: u8,
}

impl DirtyPiece {
    /// Compute the dirty pieces for `m` played in `pos` (before the move).
    pub fn from_move(pos: &Position, m: Move) -> DirtyPiece {
        let us = pos.side_to_move();
        let from = m.from();
        let to = m.to();
        let pc = pos.piece_on(from);
        let mut d = DirtyPiece::default();
        if m.is_castle() {
            let rook = Piece::new(us, PieceType::Rook);
            d.subs[0] = (pc, from);
            d.subs[1] = (rook, to);
            d.adds[0] = (pc, pos.castle_king_to(m));
            d.adds[1] = (rook, pos.castle_rook_to(m));
            d.n_add = 2;
            d.n_sub = 2;
            return d;
        }
        d.subs[0] = (pc, from);
        d.n_sub = 1;
        if m.is_ep() {
            let cap_sq = if us == Color::White { to - 8 } else { to + 8 };
            d.subs[1] = (Piece::new(!us, PieceType::Pawn), cap_sq);
            d.n_sub = 2;
        } else {
            let cap = pos.piece_on(to);
            if !cap.is_none() {
                d.subs[1] = (cap, to);
                d.n_sub = 2;
            }
        }
        let placed = if m.is_promo() { Piece::new(us, m.promo_type()) } else { pc };
        d.adds[0] = (placed, to);
        d.n_add = 1;
        d
    }
}

#[derive(Clone, Copy)]
struct StackEntry {
    acc: Accumulator,
    dirty: DirtyPiece,
    /// King squares (white, black) after this entry's move.
    kings: [Square; 2],
}

#[derive(Clone, Copy)]
struct FinnyEntry {
    acc: Align64<[i16; L1]>,
    by_color: [Bitboard; 2],
    by_type: [Bitboard; 6],
}

/// Per-thread NNUE state: accumulator stack + refresh cache (Finny tables).
pub struct NnueState {
    stack: Vec<StackEntry>,
    /// [perspective][bucket * 2 + mirror]
    finny: Vec<[FinnyEntry; INPUT_BUCKETS * 2]>,
}

impl NnueState {
    pub fn new() -> Self {
        let empty = FinnyEntry { acc: Align64([0; L1]), by_color: [0; 2], by_type: [0; 6] };
        NnueState {
            stack: Vec::with_capacity(MAX_PLY + 8),
            finny: vec![[empty; INPUT_BUCKETS * 2]; 2],
        }
    }

    /// Reset for a new root position. Accumulators are computed lazily.
    pub fn reset(&mut self, pos: &Position, net: &Network) {
        self.stack.clear();
        // Reset finny tables to "bias only, empty board" so a refresh is a plain add of all pieces.
        for p in 0..2 {
            for e in self.finny[p].iter_mut() {
                e.acc = net.ft_bias;
                e.by_color = [0; 2];
                e.by_type = [0; 6];
            }
        }
        let mut entry = StackEntry {
            acc: Accumulator::new(),
            dirty: DirtyPiece::default(),
            kings: [pos.king_sq(Color::White), pos.king_sq(Color::Black)],
        };
        for p in [Color::White, Color::Black] {
            Self::refresh_perspective(&mut self.finny[p.idx()], &mut entry.acc, p, pos, net);
        }
        self.stack.push(entry);
    }

    /// Push a new ply after `m` was played from `pos_before` giving `pos_after`.
    #[inline]
    pub fn push(&mut self, pos_before: &Position, m: Move, pos_after: &Position) {
        let dirty = DirtyPiece::from_move(pos_before, m);
        self.stack.push(StackEntry {
            acc: Accumulator::new(),
            dirty,
            kings: [pos_after.king_sq(Color::White), pos_after.king_sq(Color::Black)],
        });
    }

    /// Push a null move (no piece changes; accumulators are copied lazily).
    #[inline]
    pub fn push_null(&mut self) {
        let top = *self.stack.last().unwrap();
        self.stack.push(StackEntry { acc: top.acc, dirty: DirtyPiece::default(), kings: top.kings });
    }

    #[inline]
    pub fn pop(&mut self) {
        self.stack.pop();
        debug_assert!(!self.stack.is_empty());
    }

    fn rel_king(p: Color, ksq: Square) -> Square {
        if p == Color::Black {
            ksq ^ 56
        } else {
            ksq
        }
    }

    /// Does moving the king of perspective `p` from `a` to `b` require a refresh?
    #[inline(always)]
    fn needs_refresh(p: Color, a: Square, b: Square) -> bool {
        let ra = Self::rel_king(p, a);
        let rb = Self::rel_king(p, b);
        king_bucket(ra) != king_bucket(rb) || (file_of(ra) > 3) != (file_of(rb) > 3)
    }

    fn refresh_perspective(
        finny: &mut [FinnyEntry; INPUT_BUCKETS * 2],
        acc: &mut Accumulator,
        p: Color,
        pos: &Position,
        net: &Network,
    ) {
        let ksq = pos.king_sq(p);
        let rk = Self::rel_king(p, ksq);
        let idx = king_bucket(rk) * 2 + (file_of(rk) > 3) as usize;
        let e = &mut finny[idx];
        // Diff the cached piece set against the current one.
        for c in [Color::White, Color::Black] {
            for pt in PieceType::ALL {
                let cur = pos.pieces_c(c, pt);
                let old = e.by_color[c.idx()] & e.by_type[pt.idx()];
                let piece = Piece::new(c, pt);
                for s in bits(cur & !old) {
                    let f = feature_index(p, rk, piece, s);
                    simd::add_feature(&mut e.acc.0, &net.ft_weights[f].0);
                }
                for s in bits(old & !cur) {
                    let f = feature_index(p, rk, piece, s);
                    simd::sub_feature(&mut e.acc.0, &net.ft_weights[f].0);
                }
            }
        }
        e.by_color = [pos.colored(Color::White), pos.colored(Color::Black)];
        for pt in PieceType::ALL {
            e.by_type[pt.idx()] = pos.pieces(pt);
        }
        acc.vals[p.idx()] = e.acc;
        acc.computed[p.idx()] = true;
    }

    /// Apply the dirty pieces of stack[i] to produce stack[i].acc[p] from stack[i-1].acc[p].
    #[inline]
    fn apply_incremental(&mut self, i: usize, p: Color, rel_ksq: Square, net: &Network) {
        let d = self.stack[i].dirty;
        let (prev, cur) = self.stack.split_at_mut(i);
        let prev = &prev[i - 1].acc.vals[p.idx()].0;
        let cur = &mut cur[0].acc;
        let out = &mut cur.vals[p.idx()].0;
        let fi = |pc: Piece, s: Square| feature_index(p, rel_ksq, pc, s);
        match (d.n_add, d.n_sub) {
            (0, 0) => out.copy_from_slice(prev),
            (1, 1) => simd::add_sub(out, prev, &net.ft_weights[fi(d.adds[0].0, d.adds[0].1)].0, &net.ft_weights[fi(d.subs[0].0, d.subs[0].1)].0),
            (1, 2) => simd::add_sub_sub(
                out,
                prev,
                &net.ft_weights[fi(d.adds[0].0, d.adds[0].1)].0,
                &net.ft_weights[fi(d.subs[0].0, d.subs[0].1)].0,
                &net.ft_weights[fi(d.subs[1].0, d.subs[1].1)].0,
            ),
            (2, 2) => simd::add_add_sub_sub(
                out,
                prev,
                &net.ft_weights[fi(d.adds[0].0, d.adds[0].1)].0,
                &net.ft_weights[fi(d.adds[1].0, d.adds[1].1)].0,
                &net.ft_weights[fi(d.subs[0].0, d.subs[0].1)].0,
                &net.ft_weights[fi(d.subs[1].0, d.subs[1].1)].0,
            ),
            _ => unreachable!(),
        }
        cur.computed[p.idx()] = true;
    }

    /// Make sure the top-of-stack accumulator for perspective `p` is computed.
    fn ensure(&mut self, p: Color, pos: &Position, net: &Network) {
        let top = self.stack.len() - 1;
        if self.stack[top].acc.computed[p.idx()] {
            return;
        }
        // Walk back to the nearest computed entry, or to a king-bucket change (refresh point).
        let mut i = top;
        loop {
            if self.stack[i].acc.computed[p.idx()] {
                break;
            }
            if i == 0 {
                break;
            }
            let ka = self.stack[i - 1].kings[p.idx()];
            let kb = self.stack[i].kings[p.idx()];
            if ka != kb && Self::needs_refresh(p, ka, kb) {
                // Refresh at i from the position (only valid for i == top since we only have `pos`
                // for the top). For deeper entries we still refresh from `pos` only when i == top;
                // otherwise fall through to a full refresh at top.
                break;
            }
            i -= 1;
        }
        if !self.stack[i].acc.computed[p.idx()] {
            // Need a refresh. We only hold the top position, so refresh the top directly.
            let mut acc = self.stack[top].acc;
            Self::refresh_perspective(&mut self.finny[p.idx()], &mut acc, p, pos, net);
            self.stack[top].acc = acc;
            return;
        }
        let rel_ksq = Self::rel_king(p, self.stack[top].kings[p.idx()]);
        for j in i + 1..=top {
            self.apply_incremental(j, p, rel_ksq, net);
        }
    }

    /// Evaluate the top-of-stack position from the side to move's perspective.
    pub fn evaluate(&mut self, pos: &Position, net: &Network) -> Value {
        self.ensure(Color::White, pos, net);
        self.ensure(Color::Black, pos, net);
        let top = self.stack.len() - 1;
        let acc = &self.stack[top].acc;
        let us = pos.side_to_move();
        let bucket = output_bucket(pos);
        let sum = simd::output(&acc.vals[us.idx()].0, &acc.vals[(!us).idx()].0, &net.out_weights[bucket].0);
        let v = (sum / QA + net.out_bias[bucket] as i32) * SCALE / (QA * QB);
        v
    }

    /// Slow reference evaluation from scratch (tests).
    pub fn evaluate_reference(pos: &Position, net: &Network) -> Value {
        let mut acc = [[0i32; L1]; 2];
        for p in [Color::White, Color::Black] {
            let rk = Self::rel_king(p, pos.king_sq(p));
            for i in 0..L1 {
                acc[p.idx()][i] = net.ft_bias.0[i] as i32;
            }
            for s in bits(pos.occupied()) {
                let f = feature_index(p, rk, pos.piece_on(s), s);
                for i in 0..L1 {
                    acc[p.idx()][i] += net.ft_weights[f].0[i] as i32;
                }
            }
        }
        let us = pos.side_to_move();
        let bucket = output_bucket(pos);
        let w = &net.out_weights[bucket].0;
        let mut sum: i32 = 0;
        for i in 0..L1 {
            let v = acc[us.idx()][i].clamp(0, QA);
            sum = sum.wrapping_add(v * w[i] as i32 * v);
        }
        for i in 0..L1 {
            let v = acc[(!us).idx()][i].clamp(0, QA);
            sum = sum.wrapping_add(v * w[L1 + i] as i32 * v);
        }
        (sum / QA + net.out_bias[bucket] as i32) * SCALE / (QA * QB)
    }
}

impl Default for NnueState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::legal_moves;

    #[test]
    fn roundtrip_bytes() {
        let net = Network::random(7);
        let bytes = net.to_bytes();
        assert_eq!(bytes.len() % 64, 0);
        let net2 = Network::from_bytes(&bytes).unwrap();
        assert_eq!(net.ft_weights[123].0[..], net2.ft_weights[123].0[..]);
        assert_eq!(net.out_bias, net2.out_bias);
    }

    #[test]
    fn incremental_matches_reference_random_games() {
        crate::init();
        let net = Network::random(42);
        let mut rng = 0xDEAD_BEEFu64;
        let fens = [
            crate::position::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
        ];
        for fen in fens {
            for game in 0..6 {
                let mut pos = Position::from_fen(fen).unwrap();
                let mut st = NnueState::new();
                st.reset(&pos, &net);
                assert_eq!(st.evaluate(&pos, &net), NnueState::evaluate_reference(&pos, &net));
                let mut stack: Vec<Position> = vec![pos];
                for ply in 0..120 {
                    let moves = legal_moves(&pos);
                    if moves.is_empty() {
                        break;
                    }
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    // Occasionally pop back to test lazy evaluation across unwinds.
                    if ply > 4 && rng % 7 == 0 && stack.len() > 2 {
                        st.pop();
                        stack.pop();
                        pos = *stack.last().unwrap();
                        assert_eq!(st.evaluate(&pos, &net), NnueState::evaluate_reference(&pos, &net), "after pop");
                        continue;
                    }
                    let m = moves.moves[(rng % moves.len() as u64) as usize].mv;
                    let next = pos.make_move(m);
                    st.push(&pos, m, &next);
                    pos = next;
                    stack.push(pos);
                    // Evaluate only sometimes so that multi-ply lazy updates get exercised.
                    if rng % 3 != 0 {
                        assert_eq!(
                            st.evaluate(&pos, &net),
                            NnueState::evaluate_reference(&pos, &net),
                            "fen {} game {} ply {} move {}",
                            fen,
                            game,
                            ply,
                            m
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn null_move_keeps_accumulators() {
        crate::init();
        let net = Network::random(9);
        let pos = Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
        let mut st = NnueState::new();
        st.reset(&pos, &net);
        let null = pos.make_null_move();
        st.push_null();
        assert_eq!(st.evaluate(&null, &net), NnueState::evaluate_reference(&null, &net));
        st.pop();
        assert_eq!(st.evaluate(&pos, &net), NnueState::evaluate_reference(&pos, &net));
    }
}
