//! Anna v3 network: threat-input, multilayer NNUE.
//!
//!   inputs (per perspective):  psq 768 x 16 king buckets (mirrored)  -> FT (i16 weights, x255)
//!                              pawn pairs 4560 + threats 59808       -> FT (i8 weights, x255)
//!   FT:      L1 = 1024 accumulators per perspective, CReLU to [0,255], pairwise product of the two
//!            halves >> 9 -> 512 u8 (0..127) per perspective, concatenated to 1024.
//!   L1:      1024 -> 16 per output bucket, i8 weights (x128), i32 sums, converted to f32 with the
//!            exact scale 512 / (128 * 255 * 255); f32 biases.
//!   act:     dual activation [crelu(v), clamp(v^2, 0, 1)] -> 32
//!   L2:      32 -> 32 (f32), crelu.   L3: 32 -> 1 (f32).   eval = out * 400.
//!
//! File layout (all little-endian, in this order, padded to 64 with "bullet"):
//!   psq_w  [12288][L1] i16 | pp_w [64368][L1] i8 | ft_b [L1] i16 |
//!   l1_w [8*16][L1] i8 | l1_b [8*16] f32 | l2_w [8*32][32] f32 | l2_b [8*32] f32 | l3_w [8][32] f32 | l3_b [8] f32
//! matching trainer/src/main_v3.rs `SavedFormat` order.

use super::threats::{FeatureMapper, RelBoard};
use super::{feature_index, king_bucket, output_bucket, Align64, INPUT_BUCKETS, OUTPUT_BUCKETS};
use crate::bitboard::*;
use crate::position::Position;
use crate::types::*;

pub const L1: usize = 1024;
pub const HALF: usize = L1 / 2;
pub const L2: usize = 16;
pub const L2_DUAL: usize = 2 * L2;
pub const L3: usize = 32;
pub const PSQ_FEATURES: usize = 768 * INPUT_BUCKETS;
pub const PP_FEATURES: usize = 4560 + 59808;
pub const QA: i32 = 255;
pub const QB: i32 = 128;
pub const SCALE: f32 = 400.0;
/// Real value of one L1 input unit: (a*b)>>9 with a,b in [0,255] representing [0,1] (max 127, so the
/// u8 x i8 multiply-add cannot saturate).
pub const L1_INPUT_SCALE: f32 = 255.0 * 255.0 / 512.0;
pub const PAIR_SHIFT: u32 = 9;

pub const NET_BYTES_UNPADDED: usize = PSQ_FEATURES * L1 * 2
    + PP_FEATURES * L1
    + L1 * 2
    + OUTPUT_BUCKETS * L2 * L1
    + OUTPUT_BUCKETS * L2 * 4
    + OUTPUT_BUCKETS * L3 * L2_DUAL * 4
    + OUTPUT_BUCKETS * L3 * 4
    + OUTPUT_BUCKETS * L3 * 4
    + OUTPUT_BUCKETS * 4;

pub struct NetworkV3 {
    pub psq_w: Vec<Align64<[i16; L1]>>,
    pub pp_w: Vec<Align64<[i8; L1]>>,
    pub ft_b: Align64<[i16; L1]>,
    pub l1_w: Vec<Align64<[i8; L1]>>, // [bucket*L2 + out]
    pub l1_b: Vec<f32>,               // [bucket*L2 + out]
    pub l2_w: Vec<[f32; L2_DUAL]>,    // [bucket*L3 + out]
    pub l2_b: Vec<f32>,               // [bucket*L3 + out]
    pub l3_w: Vec<[f32; L3]>,         // [bucket]
    pub l3_b: Vec<f32>,               // [bucket]
    pub mapper: FeatureMapper,
}

struct Reader<'a> {
    b: &'a [u8],
    off: usize,
}
impl<'a> Reader<'a> {
    fn i16s(&mut self, n: usize) -> Vec<i16> {
        let v = (0..n).map(|i| i16::from_le_bytes([self.b[self.off + 2 * i], self.b[self.off + 2 * i + 1]])).collect();
        self.off += 2 * n;
        v
    }
    fn i8s(&mut self, n: usize) -> Vec<i8> {
        let v = self.b[self.off..self.off + n].iter().map(|x| *x as i8).collect();
        self.off += n;
        v
    }
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        let v = (0..n)
            .map(|i| f32::from_le_bytes([self.b[self.off + 4 * i], self.b[self.off + 4 * i + 1], self.b[self.off + 4 * i + 2], self.b[self.off + 4 * i + 3]]))
            .collect();
        self.off += 4 * n;
        v
    }
}

impl NetworkV3 {
    pub fn from_bytes(bytes: &[u8]) -> Result<NetworkV3, String> {
        if bytes.len() < NET_BYTES_UNPADDED || bytes.len() > NET_BYTES_UNPADDED + 64 {
            return Err(format!("v3 network size {} not in [{}, {}]", bytes.len(), NET_BYTES_UNPADDED, NET_BYTES_UNPADDED + 63));
        }
        let mut r = Reader { b: bytes, off: 0 };
        let psq = r.i16s(PSQ_FEATURES * L1);
        let pp = r.i8s(PP_FEATURES * L1);
        let ftb = r.i16s(L1);
        let l1w = r.i8s(OUTPUT_BUCKETS * L2 * L1);
        let l1b = r.f32s(OUTPUT_BUCKETS * L2);
        let l2w = r.f32s(OUTPUT_BUCKETS * L3 * L2_DUAL);
        let l2b = r.f32s(OUTPUT_BUCKETS * L3);
        let l3w = r.f32s(OUTPUT_BUCKETS * L3);
        let l3b = r.f32s(OUTPUT_BUCKETS);
        let mut psq_w = Vec::with_capacity(PSQ_FEATURES);
        for f in 0..PSQ_FEATURES {
            let mut a = Align64([0i16; L1]);
            a.0.copy_from_slice(&psq[f * L1..(f + 1) * L1]);
            psq_w.push(a);
        }
        let mut pp_w = Vec::with_capacity(PP_FEATURES);
        for f in 0..PP_FEATURES {
            let mut a = Align64([0i8; L1]);
            a.0.copy_from_slice(&pp[f * L1..(f + 1) * L1]);
            pp_w.push(a);
        }
        let mut ft_b = Align64([0i16; L1]);
        ft_b.0.copy_from_slice(&ftb);
        let mut l1_w = Vec::with_capacity(OUTPUT_BUCKETS * L2);
        for o in 0..OUTPUT_BUCKETS * L2 {
            let mut a = Align64([0i8; L1]);
            a.0.copy_from_slice(&l1w[o * L1..(o + 1) * L1]);
            l1_w.push(a);
        }
        let mut l2_w = Vec::with_capacity(OUTPUT_BUCKETS * L3);
        for o in 0..OUTPUT_BUCKETS * L3 {
            let mut a = [0f32; L2_DUAL];
            a.copy_from_slice(&l2w[o * L2_DUAL..(o + 1) * L2_DUAL]);
            l2_w.push(a);
        }
        let mut l3_w = Vec::with_capacity(OUTPUT_BUCKETS);
        for b in 0..OUTPUT_BUCKETS {
            let mut a = [0f32; L3];
            a.copy_from_slice(&l3w[b * L3..(b + 1) * L3]);
            l3_w.push(a);
        }
        for v in l1b.iter().chain(l2w.iter()).chain(l2b.iter()).chain(l3w.iter()).chain(l3b.iter()) {
            if !v.is_finite() || v.abs() > 1000.0 {
                return Err(format!("v3 network has an implausible float weight {}", v));
            }
        }
        Ok(NetworkV3 { psq_w, pp_w, ft_b, l1_w, l1_b: l1b, l2_w, l2_b: l2b, l3_w, l3_b: l3b, mapper: FeatureMapper::new() })
    }

    pub fn load(path: &str) -> Result<NetworkV3, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
        NetworkV3::from_bytes(&bytes)
    }

    /// Deterministic pseudo-random net for tests (small magnitudes so activations are in range).
    pub fn random(seed: u64) -> NetworkV3 {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut psq_w = Vec::with_capacity(PSQ_FEATURES);
        for _ in 0..PSQ_FEATURES {
            let mut a = Align64([0i16; L1]);
            for v in a.0.iter_mut() {
                *v = (next() % 41) as i16 - 20;
            }
            psq_w.push(a);
        }
        let mut pp_w = Vec::with_capacity(PP_FEATURES);
        for _ in 0..PP_FEATURES {
            let mut a = Align64([0i8; L1]);
            for v in a.0.iter_mut() {
                *v = (next() % 21) as i8 - 10;
            }
            pp_w.push(a);
        }
        let mut ft_b = Align64([0i16; L1]);
        for v in ft_b.0.iter_mut() {
            *v = (next() % 301) as i16 - 100;
        }
        let mut l1_w = Vec::new();
        for _ in 0..OUTPUT_BUCKETS * L2 {
            let mut a = Align64([0i8; L1]);
            for v in a.0.iter_mut() {
                *v = (next() % 61) as i8 - 30;
            }
            l1_w.push(a);
        }
        let fr = |x: u64| (x % 2001) as f32 / 1000.0 - 1.0;
        let l1_b: Vec<f32> = (0..OUTPUT_BUCKETS * L2).map(|_| fr(next()) * 0.5).collect();
        let l2_w: Vec<[f32; L2_DUAL]> = (0..OUTPUT_BUCKETS * L3)
            .map(|_| {
                let mut a = [0f32; L2_DUAL];
                for v in a.iter_mut() {
                    *v = fr(next()) * 0.3;
                }
                a
            })
            .collect();
        let l2_b: Vec<f32> = (0..OUTPUT_BUCKETS * L3).map(|_| fr(next()) * 0.2).collect();
        let l3_w: Vec<[f32; L3]> = (0..OUTPUT_BUCKETS)
            .map(|_| {
                let mut a = [0f32; L3];
                for v in a.iter_mut() {
                    *v = fr(next()) * 0.3;
                }
                a
            })
            .collect();
        let l3_b: Vec<f32> = (0..OUTPUT_BUCKETS).map(|_| fr(next()) * 0.2).collect();
        NetworkV3 { psq_w, pp_w, ft_b, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b, mapper: FeatureMapper::new() }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(NET_BYTES_UNPADDED + 64);
        for f in &self.psq_w {
            for v in f.0.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for f in &self.pp_w {
            out.extend(f.0.iter().map(|v| *v as u8));
        }
        for v in self.ft_b.0.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for f in &self.l1_w {
            out.extend(f.0.iter().map(|v| *v as u8));
        }
        for v in self.l1_b.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for f in &self.l2_w {
            for v in f.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in self.l2_b.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for f in &self.l3_w {
            for v in f.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in self.l3_b.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let pad = (64 - out.len() % 64) % 64;
        for i in 0..pad {
            out.push(b"bullet"[i % 6]);
        }
        out
    }

    // ---------------------------------------------------------------------------------------
    // Reference (scalar, from scratch) forward pass. The incremental/SIMD paths must match it.
    // ---------------------------------------------------------------------------------------

    /// Accumulate the perspective `p` feature transformer from scratch into `acc`.
    pub fn refresh_accumulator(&self, pos: &Position, p: Color, acc: &mut [i16; L1]) {
        acc.copy_from_slice(&self.ft_b.0);
        let ksq = pos.king_sq(p);
        let rk = if p == Color::Black { ksq ^ 56 } else { ksq };
        let mut psq = [0usize; 32];
        let mut np = 0;
        for s in bits(pos.occupied()) {
            let f = feature_index(p, rk, pos.piece_on(s), s);
            prefetch_row(&self.psq_w[f]);
            psq[np] = f;
            np += 1;
        }
        let rel = RelBoard::from_position(pos, p);
        let mut feats = [0usize; 320];
        let mut n = 0;
        self.mapper.map_features(&rel, |s| {
            feats[n] = s;
            n += 1;
        }, |_| {});
        for &f in &feats[..n] {
            prefetch_row(&self.pp_w[f]);
        }
        for &f in &psq[..np] {
            add_i16_row(acc, &self.psq_w[f].0);
        }
        for &f in &feats[..n] {
            add_i8_row(acc, &self.pp_w[f].0);
        }
    }

    /// Forward pass from two computed accumulators (stm first). Scalar reference.
    pub fn forward_scalar(&self, us: &[i16; L1], them: &[i16; L1], bucket: usize) -> Value {
        let mut x = [0u8; L1];
        for (k, acc) in [us, them].iter().enumerate() {
            for i in 0..HALF {
                let a = (acc[i] as i32).clamp(0, QA);
                let b = (acc[i + HALF] as i32).clamp(0, QA);
                x[k * HALF + i] = ((a * b) >> PAIR_SHIFT) as u8;
            }
        }
        let mut sums = [0i32; L2];
        for o in 0..L2 {
            let w = &self.l1_w[bucket * L2 + o].0;
            let mut sum: i32 = 0;
            for i in 0..L1 {
                sum += x[i] as i32 * w[i] as i32;
            }
            sums[o] = sum;
        }
        self.tail(&sums, bucket)
    }

    /// Forward pass using the SIMD kernels (bit-identical to `forward_scalar`).
    pub fn forward(&self, us: &[i16; L1], them: &[i16; L1], bucket: usize) -> Value {
        use super::simd::v3k::{l1_dots, pairwise};
        let mut x = Align64([0u8; L1]);
        pairwise(us, &mut x.0, 0);
        pairwise(them, &mut x.0, HALF);
        let mut sums = [0i32; L2];
        // SAFETY-free reinterpretation: l1_w rows are Align64<[i8; L1]>, contiguous; build a slice of rows.
        let rows: &[[i8; L1]] = unsafe { std::slice::from_raw_parts(self.l1_w[bucket * L2..].as_ptr() as *const [i8; L1], L2) };
        // (Align64<[i8;L1]> has the same size as [i8;L1] rounded to 64 = 1024, so the stride is exact.)
        l1_dots(&x.0, rows, &mut sums);
        self.tail(&sums, bucket)
    }

    #[inline]
    fn tail(&self, sums: &[i32; L2], bucket: usize) -> Value {
        let mut h1 = [0f32; L2_DUAL];
        for o in 0..L2 {
            let v = sums[o] as f32 / (QB as f32 * L1_INPUT_SCALE) + self.l1_b[bucket * L2 + o];
            h1[o] = v.clamp(0.0, 1.0);
            h1[L2 + o] = (v * v).clamp(0.0, 1.0);
        }
        let mut h2 = [0f32; L3];
        super::simd::v3k::l2_forward(&h1, &self.l2_w[bucket * L3..bucket * L3 + L3], &self.l2_b[bucket * L3..bucket * L3 + L3], &mut h2);
        let w = &self.l3_w[bucket];
        let mut out = self.l3_b[bucket];
        for i in 0..L3 {
            out += w[i] * h2[i];
        }
        (out * SCALE) as Value
    }

    /// Full from-scratch evaluation (tests / netcheck).
    pub fn evaluate_reference(&self, pos: &Position) -> Value {
        let mut w = [0i16; L1];
        let mut b = [0i16; L1];
        self.refresh_accumulator(pos, Color::White, &mut w);
        self.refresh_accumulator(pos, Color::Black, &mut b);
        let (us, them) = if pos.side_to_move() == Color::White { (&w, &b) } else { (&b, &w) };
        self.forward_scalar(us, them, output_bucket(pos))
    }
}


// ---------------------------------------------------------------------------------------------
// Incremental state
// ---------------------------------------------------------------------------------------------

use super::DirtyPiece;

#[derive(Clone, Copy)]
struct Entry3 {
    acc: [Align64<[i16; L1]>; 2],
    computed: [bool; 2],
    /// Position after this entry's move.
    pos: Position,
    dirty: DirtyPiece,
    /// Squares whose contents changed (absolute), 0 for null moves / root.
    changed: u64,
    is_null: bool,
    /// White-relative board of `pos`, filled in lazily and reused as the next entry's "old" board.
    rel: RelBoard,
    rel_ok: bool,
}

impl Entry3 {
    fn blank() -> Entry3 {
        Entry3 {
            acc: [Align64([0; L1]); 2],
            computed: [false; 2],
            pos: Position::empty(),
            dirty: DirtyPiece::default(),
            changed: 0,
            is_null: false,
            rel: RelBoard { bb: [0; 8], pieces: [13; 64] },
            rel_ok: false,
        }
    }
}

/// Per-thread incremental state. The entry stack is allocated once and indexed by `len`: a push
/// only writes the small metadata fields, never the 4 KB of accumulators (those are produced lazily,
/// straight from the previous entry, when an evaluation is actually needed).
pub struct StateV3 {
    stack: Vec<Entry3>,
    len: usize,
    scratch_old: Vec<usize>,
    scratch_new: Vec<usize>,
    scratch_old_b: Vec<usize>,
    scratch_new_b: Vec<usize>,
    apply_add: Vec<usize>,
    apply_sub: Vec<usize>,
}

use super::simd::v3k::{add_i16_row, add_i8_row, apply_rows};

/// Number of 64-byte lines to prefetch per applied weight row (0 disables). Two lines are enough to
/// start the hardware prefetcher on the sequential 1 KB stream.
pub const PREFETCH_LINES: usize = 2;
/// Applied-row counter for benchmarking (relaxed, only read by nnuebench).
pub static ROWS_APPLIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Phase timers (ns) for nnuebench: [attackers+relboards, map_restricted, diff+prefetch, row apply].
/// Only active with `--features nnue_profile`; otherwise compiled out.
pub static PHASE_NS: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
pub const PROFILE: bool = cfg!(feature = "nnue_profile");
#[inline(always)]
fn phase(i: usize, t: &mut std::time::Instant) {
    if PROFILE {
        let now = std::time::Instant::now();
        PHASE_NS[i].fetch_add((now - *t).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        *t = now;
    }
}
#[inline(always)]
fn now() -> std::time::Instant {
    if PROFILE {
        std::time::Instant::now()
    } else {
        // Never read when PROFILE is false; avoids a clock read per node.
        unsafe { std::mem::zeroed() }
    }
}

#[inline(always)]
fn prefetch_row<T>(row: &T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let p = row as *const T as *const i8;
        let bytes = std::mem::size_of::<T>();
        let mut off = 0;
        let mut n = 0;
        while off < bytes && n < PREFETCH_LINES {
            std::arch::x86_64::_mm_prefetch(p.add(off), std::arch::x86_64::_MM_HINT_T0);
            off += 64;
            n += 1;
        }
    }
}

impl StateV3 {
    pub fn new() -> Self {
        StateV3 {
            stack: vec![Entry3::blank(); MAX_PLY + 8],
            len: 0,
            scratch_old: Vec::with_capacity(512),
            scratch_new: Vec::with_capacity(512),
            scratch_old_b: Vec::with_capacity(512),
            scratch_new_b: Vec::with_capacity(512),
            apply_add: Vec::with_capacity(512),
            apply_sub: Vec::with_capacity(512),
        }
    }

    pub fn reset(&mut self, pos: &Position, net: &NetworkV3) {
        let e = &mut self.stack[0];
        e.pos = *pos;
        e.dirty = DirtyPiece::default();
        e.changed = 0;
        e.is_null = false;
        e.rel_ok = false;
        net.refresh_accumulator(pos, Color::White, &mut e.acc[0].0);
        net.refresh_accumulator(pos, Color::Black, &mut e.acc[1].0);
        e.computed = [true; 2];
        self.len = 1;
    }

    #[inline]
    pub fn push(&mut self, pos_before: &Position, m: Move, pos_after: &Position) {
        let dirty = DirtyPiece::from_move(pos_before, m);
        let mut changed = bb(m.from()) | bb(m.to());
        if m.is_castle() {
            changed |= bb(pos_before.castle_king_to(m)) | bb(pos_before.castle_rook_to(m));
        } else if m.is_ep() {
            let cap = if pos_before.side_to_move() == Color::White { m.to() - 8 } else { m.to() + 8 };
            changed |= bb(cap);
        }
        debug_assert!(self.len < self.stack.len());
        let e = &mut self.stack[self.len];
        e.computed = [false; 2];
        e.pos = *pos_after;
        e.dirty = dirty;
        e.changed = changed;
        e.is_null = false;
        e.rel_ok = false;
        self.len += 1;
    }

    #[inline]
    pub fn push_null(&mut self, pos_after: &Position) {
        debug_assert!(self.len < self.stack.len());
        let e = &mut self.stack[self.len];
        e.computed = [false; 2];
        e.pos = *pos_after;
        e.dirty = DirtyPiece::default();
        e.changed = 0;
        e.is_null = true;
        e.rel_ok = false;
        self.len += 1;
    }

    #[inline]
    pub fn pop(&mut self) {
        self.len -= 1;
        debug_assert!(self.len > 0);
    }

    #[inline(always)]
    fn rel_king(p: Color, ksq: Square) -> Square {
        if p == Color::Black {
            ksq ^ 56
        } else {
            ksq
        }
    }
    #[inline(always)]
    fn needs_refresh(p: Color, a: Square, b: Square) -> bool {
        let ra = Self::rel_king(p, a);
        let rb = Self::rel_king(p, b);
        king_bucket(ra) != king_bucket(rb) || (file_of(ra) > 3) != (file_of(rb) > 3)
    }

    /// Compute entry j's accumulators from entry j-1's for the requested perspectives. The threat
    /// diff is computed once (both perspectives come out of one pass over the affected attackers);
    /// each accumulator is then produced in a single fused pass over all changed rows.
    fn apply_incremental(&mut self, j: usize, need: [bool; 2], net: &NetworkV3) {
        let StateV3 { stack, scratch_old: ow, scratch_new: nw, scratch_old_b: ob, scratch_new_b: nb, apply_add, apply_sub, .. } = self;
        let (before, after) = stack.split_at_mut(j);
        let prev = &mut before[j - 1];
        let cur = &mut after[0];
        if cur.is_null {
            for p in 0..2 {
                if need[p] {
                    cur.acc[p] = prev.acc[p];
                    cur.computed[p] = true;
                }
            }
            // The board did not change: the cached relative board carries over.
            if prev.rel_ok {
                cur.rel = prev.rel;
                cur.rel_ok = true;
            }
            return;
        }
        let (old_pos, new_pos, dirty, changed) = (prev.pos, cur.pos, cur.dirty, cur.changed);
        let mut tm = now();
        // Piece-square feature indices per perspective (<= 2 adds, <= 2 subs).
        let mut psq_add = [[0usize; 2]; 2];
        let mut psq_sub = [[0usize; 2]; 2];
        for p in [Color::White, Color::Black] {
            if !need[p.idx()] {
                continue;
            }
            let rk = Self::rel_king(p, new_pos.king_sq(p));
            for k in 0..dirty.n_add as usize {
                let (pc, s) = dirty.adds[k];
                psq_add[p.idx()][k] = feature_index(p, rk, pc, s);
            }
            for k in 0..dirty.n_sub as usize {
                let (pc, s) = dirty.subs[k];
                psq_sub[p.idx()][k] = feature_index(p, rk, pc, s);
            }
        }
        // Threat + pawn-pair part: features emitted by affected attackers, old vs new, both
        // perspectives from the white-relative board (on_stm = white, on_ntm = black).
        let mut att_old = changed;
        let mut att_new = changed;
        for s in bits(changed) {
            att_old |= old_pos.attackers_to_occ(s, old_pos.occupied());
            att_new |= new_pos.attackers_to_occ(s, new_pos.occupied());
        }
        if !prev.rel_ok {
            prev.rel = RelBoard::from_position(&old_pos, Color::White);
            prev.rel_ok = true;
        }
        cur.rel = RelBoard::from_position(&new_pos, Color::White);
        cur.rel_ok = true;
        phase(0, &mut tm);
        ow.clear();
        ob.clear();
        nw.clear();
        nb.clear();
        net.mapper.map_restricted(&prev.rel, att_old, changed, |f| ow.push(f), |f| ob.push(f));
        net.mapper.map_restricted(&cur.rel, att_new, changed, |f| nw.push(f), |f| nb.push(f));
        phase(1, &mut tm);
        for p in 0..2 {
            if !need[p] {
                continue;
            }
            let (so, sn): (&mut Vec<usize>, &mut Vec<usize>) = if p == 0 { (&mut *ow, &mut *nw) } else { (&mut *ob, &mut *nb) };
            so.sort_unstable();
            sn.sort_unstable();
            // Set difference both ways: rows in old only are subtracted, rows in new only are added.
            apply_add.clear();
            apply_sub.clear();
            let (mut a, mut b) = (0, 0);
            while a < so.len() || b < sn.len() {
                if b >= sn.len() || (a < so.len() && so[a] < sn[b]) {
                    apply_sub.push(so[a]);
                    a += 1;
                } else if a >= so.len() || sn[b] < so[a] {
                    apply_add.push(sn[b]);
                    b += 1;
                } else {
                    a += 1;
                    b += 1;
                }
            }
            if PREFETCH_LINES > 0 {
                for &f in apply_add.iter().chain(apply_sub.iter()) {
                    prefetch_row(&net.pp_w[f]);
                }
            }
            phase(2, &mut tm);
            let na = dirty.n_add as usize;
            let ns = dirty.n_sub as usize;
            apply_rows(&prev.acc[p].0, &mut cur.acc[p].0, &net.psq_w, &psq_add[p][..na], &psq_sub[p][..ns], &net.pp_w, apply_add, apply_sub);
            cur.computed[p] = true;
            ROWS_APPLIED.fetch_add((apply_add.len() + apply_sub.len()) as u64, std::sync::atomic::Ordering::Relaxed);
            phase(3, &mut tm);
        }
    }

    /// Make both top-of-stack accumulators computed, sharing work between perspectives.
    fn ensure_both(&mut self, net: &NetworkV3) {
        let top = self.len - 1;
        let mut start = [usize::MAX; 2]; // index of the last computed entry to start from
        for p in [Color::White, Color::Black] {
            let pi = p.idx();
            if self.stack[top].computed[pi] {
                continue;
            }
            let mut i = top;
            loop {
                if self.stack[i].computed[pi] || i == 0 {
                    break;
                }
                let ka = self.stack[i - 1].pos.king_sq(p);
                let kb = self.stack[i].pos.king_sq(p);
                if ka != kb && Self::needs_refresh(p, ka, kb) {
                    break;
                }
                i -= 1;
            }
            if !self.stack[i].computed[pi] {
                let pos = self.stack[top].pos;
                let mut acc = Align64([0i16; L1]);
                net.refresh_accumulator(&pos, p, &mut acc.0);
                self.stack[top].acc[pi] = acc;
                self.stack[top].computed[pi] = true;
            } else {
                start[pi] = i;
            }
        }
        let lo = start[0].min(start[1]);
        if lo == usize::MAX {
            return;
        }
        for j in lo + 1..=top {
            let need = [start[0] != usize::MAX && start[0] < j, start[1] != usize::MAX && start[1] < j];
            self.apply_incremental(j, need, net);
        }
    }

    pub fn evaluate(&mut self, net: &NetworkV3) -> Value {
        self.ensure_both(net);
        let top = &self.stack[self.len - 1];
        let us = top.pos.side_to_move();
        let bucket = output_bucket(&top.pos);
        net.forward(&top.acc[us.idx()].0, &top.acc[(!us).idx()].0, bucket)
    }
}

const _: () = assert!(std::mem::size_of::<Align64<[i8; L1]>>() == L1);

impl Default for StateV3 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::legal_moves;

    #[test]
    fn incremental_matches_reference_random_games() {
        crate::init();
        let net = NetworkV3::random(11);
        let mut rng = 0xC0FFEEu64;
        let fens = [
            crate::position::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        ];
        for fen in fens {
            for game in 0..5 {
                let mut pos = Position::from_fen(fen).unwrap();
                let mut st = StateV3::new();
                st.reset(&pos, &net);
                assert_eq!(st.evaluate(&net), net.evaluate_reference(&pos));
                let mut stack = vec![pos];
                for ply in 0..100 {
                    let moves = legal_moves(&pos);
                    if moves.is_empty() {
                        break;
                    }
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    if ply > 4 && rng % 9 == 0 && stack.len() > 2 {
                        st.pop();
                        stack.pop();
                        pos = *stack.last().unwrap();
                        assert_eq!(st.evaluate(&net), net.evaluate_reference(&pos), "after pop");
                        continue;
                    }
                    if ply > 2 && rng % 11 == 0 && !pos.in_check() {
                        let null = pos.make_null_move();
                        st.push_null(&null);
                        assert_eq!(st.evaluate(&net), net.evaluate_reference(&null), "null move");
                        st.pop();
                    }
                    let m = moves.moves[(rng % moves.len() as u64) as usize].mv;
                    let next = pos.make_move(m);
                    st.push(&pos, m, &next);
                    pos = next;
                    stack.push(pos);
                    if rng % 3 != 0 {
                        assert_eq!(st.evaluate(&net), net.evaluate_reference(&pos), "fen {} game {} ply {} move {}", fen, game, ply, m);
                    }
                }
            }
        }
    }

    #[test]
    fn roundtrip_and_reference_eval() {
        crate::init();
        let net = NetworkV3::random(3);
        let bytes = net.to_bytes();
        assert_eq!(bytes.len() % 64, 0);
        let net2 = NetworkV3::from_bytes(&bytes).unwrap();
        let pos = Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3").unwrap();
        let v1 = net.evaluate_reference(&pos);
        let v2 = net2.evaluate_reference(&pos);
        assert_eq!(v1, v2);
        // Colour-flipped mirror position must evaluate identically from the side to move's view.
        let mirror = Position::from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3").unwrap();
        assert_eq!(net.evaluate_reference(&mirror), v1);
    }
}
