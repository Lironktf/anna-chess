//! SIMD kernels for the NNUE with a scalar reference. The AVX2 path is selected at compile time
//! (`target-cpu=native`); the scalar path is always compiled and used in tests as the reference.

use super::{Align64, L1, QA};

// ---------------------------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------------------------
pub mod scalar {
    use super::*;

    #[inline]
    pub fn add_feature(acc: &mut [i16; L1], w: &[i16; L1]) {
        for i in 0..L1 {
            acc[i] = acc[i].wrapping_add(w[i]);
        }
    }
    /// acc += sum(rows[adds]) - sum(rows[subs]) in one pass over the accumulator.
    #[inline]
    pub fn apply_rows(acc: &mut [i16; L1], rows: &[Align64<[i16; L1]>], adds: &[usize], subs: &[usize]) {
        for i in 0..L1 {
            let mut v = acc[i];
            for &r in adds { v = v.wrapping_add(rows[r].0[i]); }
            for &r in subs { v = v.wrapping_sub(rows[r].0[i]); }
            acc[i] = v;
        }
    }
    #[inline]
    pub fn sub_feature(acc: &mut [i16; L1], w: &[i16; L1]) {
        for i in 0..L1 {
            acc[i] = acc[i].wrapping_sub(w[i]);
        }
    }
    #[inline]
    pub fn add_sub(out: &mut [i16; L1], prev: &[i16; L1], a: &[i16; L1], s: &[i16; L1]) {
        for i in 0..L1 {
            out[i] = prev[i].wrapping_add(a[i]).wrapping_sub(s[i]);
        }
    }
    #[inline]
    pub fn add_sub_sub(out: &mut [i16; L1], prev: &[i16; L1], a: &[i16; L1], s1: &[i16; L1], s2: &[i16; L1]) {
        for i in 0..L1 {
            out[i] = prev[i].wrapping_add(a[i]).wrapping_sub(s1[i]).wrapping_sub(s2[i]);
        }
    }
    #[inline]
    pub fn add_add_sub_sub(
        out: &mut [i16; L1],
        prev: &[i16; L1],
        a1: &[i16; L1],
        a2: &[i16; L1],
        s1: &[i16; L1],
        s2: &[i16; L1],
    ) {
        for i in 0..L1 {
            out[i] = prev[i].wrapping_add(a1[i]).wrapping_add(a2[i]).wrapping_sub(s1[i]).wrapping_sub(s2[i]);
        }
    }
    /// SCReLU dot product: sum(clamp(x)^2 * w) over both perspectives. Uses wrapping i32 math
    /// so that it is bit-identical to the SIMD kernel.
    #[inline]
    pub fn output(us: &[i16; L1], them: &[i16; L1], w: &[i16; 2 * L1]) -> i32 {
        let mut sum: i32 = 0;
        for i in 0..L1 {
            let v = (us[i] as i32).clamp(0, QA);
            sum = sum.wrapping_add(v * w[i] as i32 * v);
        }
        for i in 0..L1 {
            let v = (them[i] as i32).clamp(0, QA);
            sum = sum.wrapping_add(v * w[L1 + i] as i32 * v);
        }
        sum
    }
}

// ---------------------------------------------------------------------------------------------
// AVX2
// ---------------------------------------------------------------------------------------------
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    const CHUNK: usize = 16; // i16 lanes per 256-bit register
    const _: () = assert!(L1 % (CHUNK * 4) == 0);

    // SAFETY (all functions): pointers come from `[i16; L1]` arrays (any alignment: unaligned
    // loads/stores are used); every access is within bounds because L1 is a multiple of CHUNK and
    // we iterate exactly L1/CHUNK times. avx2 is a compile-time target feature.

    #[inline]
    pub fn add_feature(acc: &mut [i16; L1], w: &[i16; L1]) {
        unsafe {
            let a = acc.as_mut_ptr() as *mut __m256i;
            let b = w.as_ptr() as *const __m256i;
            for i in 0..L1 / CHUNK {
                _mm256_storeu_si256(a.add(i), _mm256_add_epi16(_mm256_loadu_si256(a.add(i)), _mm256_loadu_si256(b.add(i))));
            }
        }
    }
    /// acc += sum(rows[adds]) - sum(rows[subs]), one pass over the accumulator: each 32-byte chunk
    /// is loaded once, every row is applied to it, and it is stored once.
    #[inline]
    pub fn apply_rows(acc: &mut [i16; L1], rows: &[Align64<[i16; L1]>], adds: &[usize], subs: &[usize]) {
        // SAFETY: L1 is a multiple of CHUNK; every index in adds/subs is a feature index below
        // rows.len() (checked here once instead of per chunk).
        for &r in adds.iter().chain(subs) { assert!(r < rows.len()); }
        unsafe {
            let a = acc.as_mut_ptr() as *mut __m256i;
            for i in 0..L1 / CHUNK {
                let mut v = _mm256_loadu_si256(a.add(i));
                for &r in adds { v = _mm256_add_epi16(v, _mm256_loadu_si256((rows.get_unchecked(r).0.as_ptr() as *const __m256i).add(i))); }
                for &r in subs { v = _mm256_sub_epi16(v, _mm256_loadu_si256((rows.get_unchecked(r).0.as_ptr() as *const __m256i).add(i))); }
                _mm256_storeu_si256(a.add(i), v);
            }
        }
    }
    #[inline]
    pub fn sub_feature(acc: &mut [i16; L1], w: &[i16; L1]) {
        unsafe {
            let a = acc.as_mut_ptr() as *mut __m256i;
            let b = w.as_ptr() as *const __m256i;
            for i in 0..L1 / CHUNK {
                _mm256_storeu_si256(a.add(i), _mm256_sub_epi16(_mm256_loadu_si256(a.add(i)), _mm256_loadu_si256(b.add(i))));
            }
        }
    }
    #[inline]
    pub fn add_sub(out: &mut [i16; L1], prev: &[i16; L1], a: &[i16; L1], s: &[i16; L1]) {
        unsafe {
            let o = out.as_mut_ptr() as *mut __m256i;
            let p = prev.as_ptr() as *const __m256i;
            let a = a.as_ptr() as *const __m256i;
            let s = s.as_ptr() as *const __m256i;
            for i in 0..L1 / CHUNK {
                let v = _mm256_sub_epi16(_mm256_add_epi16(_mm256_loadu_si256(p.add(i)), _mm256_loadu_si256(a.add(i))), _mm256_loadu_si256(s.add(i)));
                _mm256_storeu_si256(o.add(i), v);
            }
        }
    }
    #[inline]
    pub fn add_sub_sub(out: &mut [i16; L1], prev: &[i16; L1], a: &[i16; L1], s1: &[i16; L1], s2: &[i16; L1]) {
        unsafe {
            let o = out.as_mut_ptr() as *mut __m256i;
            let p = prev.as_ptr() as *const __m256i;
            let a = a.as_ptr() as *const __m256i;
            let s1 = s1.as_ptr() as *const __m256i;
            let s2 = s2.as_ptr() as *const __m256i;
            for i in 0..L1 / CHUNK {
                let v = _mm256_add_epi16(_mm256_loadu_si256(p.add(i)), _mm256_loadu_si256(a.add(i)));
                let v = _mm256_sub_epi16(v, _mm256_loadu_si256(s1.add(i)));
                let v = _mm256_sub_epi16(v, _mm256_loadu_si256(s2.add(i)));
                _mm256_storeu_si256(o.add(i), v);
            }
        }
    }
    #[inline]
    pub fn add_add_sub_sub(
        out: &mut [i16; L1],
        prev: &[i16; L1],
        a1: &[i16; L1],
        a2: &[i16; L1],
        s1: &[i16; L1],
        s2: &[i16; L1],
    ) {
        unsafe {
            let o = out.as_mut_ptr() as *mut __m256i;
            let p = prev.as_ptr() as *const __m256i;
            let a1 = a1.as_ptr() as *const __m256i;
            let a2 = a2.as_ptr() as *const __m256i;
            let s1 = s1.as_ptr() as *const __m256i;
            let s2 = s2.as_ptr() as *const __m256i;
            for i in 0..L1 / CHUNK {
                let v = _mm256_add_epi16(_mm256_loadu_si256(p.add(i)), _mm256_loadu_si256(a1.add(i)));
                let v = _mm256_add_epi16(v, _mm256_loadu_si256(a2.add(i)));
                let v = _mm256_sub_epi16(v, _mm256_loadu_si256(s1.add(i)));
                let v = _mm256_sub_epi16(v, _mm256_loadu_si256(s2.add(i)));
                _mm256_storeu_si256(o.add(i), v);
            }
        }
    }

    #[inline]
    unsafe fn dot_half(acc: *const __m256i, w: *const __m256i, sum: __m256i) -> __m256i {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA as i16);
        let mut sum = sum;
        for i in 0..L1 / CHUNK {
            let v = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(acc.add(i)), zero), qa);
            // v*w fits in i16 (255 * 127 < 32768); madd with v gives i32 lanes of v*w*v pairs.
            let p = _mm256_mullo_epi16(v, _mm256_loadu_si256(w.add(i)));
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(p, v));
        }
        sum
    }

    #[inline]
    pub fn output(us: &[i16; L1], them: &[i16; L1], w: &[i16; 2 * L1]) -> i32 {
        unsafe {
            let sum = _mm256_setzero_si256();
            let sum = dot_half(us.as_ptr() as *const __m256i, w.as_ptr() as *const __m256i, sum);
            let sum = dot_half(them.as_ptr() as *const __m256i, (w.as_ptr() as *const __m256i).add(L1 / CHUNK), sum);
            // Horizontal sum of 8 i32 lanes.
            let hi = _mm256_extracti128_si256(sum, 1);
            let lo = _mm256_castsi256_si128(sum);
            let s = _mm_add_epi32(hi, lo);
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
            _mm_cvtsi128_si32(s)
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use avx2::*;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub use scalar::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn arr(seed: &mut u64, range: i32) -> super::super::Align64<[i16; L1]> {
        let mut a = super::super::Align64([0i16; L1]);
        for v in a.0.iter_mut() {
            *v = (rnd(seed) % (2 * range as u64 + 1)) as i16 - range as i16;
        }
        a
    }

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 1u64;
        for _ in 0..200 {
            let prev = arr(&mut seed, 400);
            let a1 = arr(&mut seed, 200);
            let a2 = arr(&mut seed, 200);
            let s1 = arr(&mut seed, 200);
            let s2 = arr(&mut seed, 200);
            let mut o1 = super::super::Align64([0i16; L1]);
            let mut o2 = super::super::Align64([0i16; L1]);
            add_sub(&mut o1.0, &prev.0, &a1.0, &s1.0);
            scalar::add_sub(&mut o2.0, &prev.0, &a1.0, &s1.0);
            assert_eq!(o1.0[..], o2.0[..]);
            add_sub_sub(&mut o1.0, &prev.0, &a1.0, &s1.0, &s2.0);
            scalar::add_sub_sub(&mut o2.0, &prev.0, &a1.0, &s1.0, &s2.0);
            assert_eq!(o1.0[..], o2.0[..]);
            add_add_sub_sub(&mut o1.0, &prev.0, &a1.0, &a2.0, &s1.0, &s2.0);
            scalar::add_add_sub_sub(&mut o2.0, &prev.0, &a1.0, &a2.0, &s1.0, &s2.0);
            assert_eq!(o1.0[..], o2.0[..]);
            let mut c1 = prev;
            let mut c2 = prev;
            add_feature(&mut c1.0, &a1.0);
            scalar::add_feature(&mut c2.0, &a1.0);
            sub_feature(&mut c1.0, &s1.0);
            scalar::sub_feature(&mut c2.0, &s1.0);
            assert_eq!(c1.0[..], c2.0[..]);

            let mut w = super::super::Align64([0i16; 2 * L1]);
            for v in w.0.iter_mut() {
                *v = (rnd(&mut seed) % 255) as i16 - 127;
            }
            let us = arr(&mut seed, 600);
            let them = arr(&mut seed, 600);
            assert_eq!(output(&us.0, &them.0, &w.0), scalar::output(&us.0, &them.0, &w.0));
        }
    }
}
