//! Shared transposition table. Entries are 10 bytes packed in 32-byte clusters of 3.
//! Reads and writes are intentionally racy across threads (Lazy SMP); a torn entry can only
//! produce a wrong move/score that the search validates (`is_pseudo_legal`) or tolerates.

use crate::types::*;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, Ordering};

pub const DEPTH_QS: i32 = 0;
pub const DEPTH_UNSEARCHED: i32 = -2;
pub const DEPTH_ENTRY_OFFSET: i32 = -3;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Entry {
    key16: u16,
    depth8: u8,
    gen_bound8: u8, // generation (5 bits) | pv (1 bit) | bound (2 bits)
    move16: u16,
    value16: i16,
    eval16: i16,
}

const CLUSTER_SIZE: usize = 3;

#[repr(C, align(32))]
#[derive(Clone, Copy, Default)]
struct Cluster {
    entries: [Entry; CLUSTER_SIZE],
    _pad: [u8; 2],
}

const GEN_BITS: u8 = 3;
const GEN_DELTA: u8 = 1 << GEN_BITS;
const GEN_CYCLE: i32 = 255 + (1 << GEN_BITS);
const GEN_MASK: i32 = (0xFF << GEN_BITS) & 0xFF;

#[derive(Clone, Copy)]
pub struct TTData {
    pub mv: Move,
    pub value: Value,
    pub eval: Value,
    pub depth: i32,
    pub bound: Bound,
    pub is_pv: bool,
}

pub struct TTWriter {
    cluster: usize,
    slot: usize,
}

pub struct TranspositionTable {
    clusters: UnsafeCell<Vec<Cluster>>,
    count: usize,
    generation: AtomicU8,
}

// SAFETY: concurrent racy access is by design (see module docs); every read result is validated
// or tolerated by the search and no memory unsafety can result from torn 10-byte entries since
// all fields are plain integers.
unsafe impl Sync for TranspositionTable {}
unsafe impl Send for TranspositionTable {}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let mut tt = TranspositionTable { clusters: UnsafeCell::new(Vec::new()), count: 0, generation: AtomicU8::new(0) };
        tt.resize(mb);
        tt
    }

    pub fn resize(&mut self, mb: usize) {
        let bytes = mb.max(1) * 1024 * 1024;
        let count = bytes / std::mem::size_of::<Cluster>();
        let v = vec![Cluster::default(); count];
        *self.clusters.get_mut() = v;
        self.count = count;
    }

    pub fn clear(&mut self) {
        for c in self.clusters.get_mut().iter_mut() {
            *c = Cluster::default();
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    /// Clear using multiple threads (large tables).
    pub fn clear_threaded(&mut self, threads: usize) {
        let v = self.clusters.get_mut();
        let n = v.len();
        let threads = threads.max(1).min(n.max(1));
        let chunk = n.div_ceil(threads);
        std::thread::scope(|s| {
            for part in v.chunks_mut(chunk.max(1)) {
                s.spawn(move || {
                    for c in part.iter_mut() {
                        *c = Cluster::default();
                    }
                });
            }
        });
        self.generation.store(0, Ordering::Relaxed);
    }

    pub fn new_search(&self) {
        self.generation.fetch_add(GEN_DELTA, Ordering::Relaxed);
    }

    #[inline(always)]
    fn generation(&self) -> u8 {
        self.generation.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn index(&self, key: u64) -> usize {
        ((key as u128 * self.count as u128) >> 64) as usize
    }

    #[inline(always)]
    fn cluster(&self, i: usize) -> &mut Cluster {
        // SAFETY: index is in bounds (see `index`); racy access is intended (module docs).
        unsafe { &mut *(*self.clusters.get()).as_mut_ptr().add(i) }
    }

    #[inline(always)]
    fn relative_age(&self, gen_bound8: u8) -> i32 {
        (GEN_CYCLE + self.generation() as i32 - gen_bound8 as i32) & GEN_MASK
    }

    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        {
            let i = self.index(key);
            // SAFETY: prefetch is a hint; the pointer is in bounds.
            unsafe {
                let p = (*self.clusters.get()).as_ptr().add(i) as *const i8;
                std::arch::x86_64::_mm_prefetch(p, std::arch::x86_64::_MM_HINT_T0);
            }
        }
    }

    /// Probe: returns (hit, data, writer).
    pub fn probe(&self, key: u64) -> (bool, TTData, TTWriter) {
        let ci = self.index(key);
        let cl = self.cluster(ci);
        let key16 = key as u16;
        for (i, e) in cl.entries.iter_mut().enumerate() {
            if e.key16 == key16 && e.depth8 != 0 {
                let data = TTData {
                    mv: Move(e.move16),
                    value: e.value16 as Value,
                    eval: e.eval16 as Value,
                    depth: e.depth8 as i32 + DEPTH_ENTRY_OFFSET,
                    bound: Bound::from_u8(e.gen_bound8 & 3),
                    is_pv: e.gen_bound8 & 4 != 0,
                };
                // Refresh generation on hit.
                e.gen_bound8 = self.generation() | (e.gen_bound8 & 7);
                return (true, data, TTWriter { cluster: ci, slot: i });
            }
        }
        // Find replacement slot.
        let mut best = 0;
        let mut best_score = i32::MAX;
        for (i, e) in cl.entries.iter().enumerate() {
            let s = e.depth8 as i32 - self.relative_age(e.gen_bound8) * 2;
            if e.depth8 == 0 {
                best = i;
                break;
            }
            if s < best_score {
                best_score = s;
                best = i;
            }
        }
        let empty = TTData { mv: Move::NONE, value: VALUE_NONE, eval: VALUE_NONE, depth: DEPTH_UNSEARCHED, bound: Bound::None, is_pv: false };
        (false, empty, TTWriter { cluster: ci, slot: best })
    }

    pub fn save(&self, w: &TTWriter, key: u64, value: Value, is_pv: bool, bound: Bound, depth: i32, mv: Move, eval: Value) {
        let cl = self.cluster(w.cluster);
        let e = &mut cl.entries[w.slot];
        let key16 = key as u16;
        if !mv.is_none() || key16 != e.key16 {
            e.move16 = mv.0;
        }
        let depth8 = (depth - DEPTH_ENTRY_OFFSET).clamp(1, 255) as u8;
        if bound == Bound::Exact || key16 != e.key16 || depth8 as i32 + 2 * is_pv as i32 > e.depth8 as i32 - 4 || self.relative_age(e.gen_bound8) != 0 {
            e.key16 = key16;
            e.depth8 = depth8;
            e.gen_bound8 = self.generation() | ((is_pv as u8) << 2) | (bound as u8);
            e.value16 = value.clamp(-32000, 32000) as i16;
            e.eval16 = eval.clamp(-32000, 32000) as i16;
        }
    }

    pub fn hashfull(&self) -> usize {
        let mut cnt = 0;
        let n = self.count.min(1000);
        for i in 0..n {
            let cl = self.cluster(i);
            for e in cl.entries.iter() {
                if e.depth8 != 0 && (e.gen_bound8 & GEN_MASK as u8) == self.generation() {
                    cnt += 1;
                }
            }
        }
        cnt / CLUSTER_SIZE
    }
}

/// Adjust a mate/TB score for storing (relative to root) from a ply-relative one.
#[inline(always)]
pub fn value_to_tt(v: Value, ply: usize) -> Value {
    if v >= VALUE_TB_WIN_IN_MAX_PLY {
        v + ply as Value
    } else if v <= VALUE_TB_LOSS_IN_MAX_PLY {
        v - ply as Value
    } else {
        v
    }
}

/// Inverse of `value_to_tt`; returns VALUE_NONE for VALUE_NONE.
#[inline(always)]
pub fn value_from_tt(v: Value, ply: usize, rule50: u8) -> Value {
    if v == VALUE_NONE {
        return VALUE_NONE;
    }
    if v >= VALUE_TB_WIN_IN_MAX_PLY {
        // Downgrade mates that would exceed the 50-move rule horizon.
        if v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 100 - rule50 as Value {
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
        }
        if VALUE_TB_WIN - v > 100 - rule50 as Value {
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
        }
        return v - ply as Value;
    }
    if v <= VALUE_TB_LOSS_IN_MAX_PLY {
        if v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 100 - rule50 as Value {
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        }
        if VALUE_TB_WIN + v > 100 - rule50 as Value {
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        }
        return v + ply as Value;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_probe() {
        let tt = TranspositionTable::new(1);
        let key = 0x1234_5678_9ABC_DEF0u64;
        let (hit, _, w) = tt.probe(key);
        assert!(!hit);
        tt.save(&w, key, 123, true, Bound::Exact, 7, Move::new(12, 28), -5);
        let (hit, d, _) = tt.probe(key);
        assert!(hit);
        assert_eq!(d.value, 123);
        assert_eq!(d.eval, -5);
        assert_eq!(d.depth, 7);
        assert_eq!(d.bound, Bound::Exact);
        assert!(d.is_pv);
        assert_eq!(d.mv, Move::new(12, 28));
    }

    #[test]
    fn mate_score_roundtrip() {
        for ply in [0, 3, 10] {
            let v = mate_in(5);
            assert_eq!(value_from_tt(value_to_tt(v, ply), ply, 0), v);
            let v = mated_in(7);
            assert_eq!(value_from_tt(value_to_tt(v, ply), ply, 0), v);
        }
    }
}
