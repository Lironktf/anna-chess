//! SIMD kernels for the NNUE with a scalar reference. The AVX2 path is selected at compile time
//! (`target-cpu=native`); the scalar path is always compiled and used in tests as the reference.

use super::{L1, QA};

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

// ---------------------------------------------------------------------------------------------
// v3 kernels: i8 feature rows into i16 accumulators, pairwise activation, dense u8 x i8 L1.
// ---------------------------------------------------------------------------------------------
pub mod v3k {
    use super::super::v3::{HALF, L1 as N, L2, L2_DUAL, L3, PAIR_SHIFT, QA};

    pub mod scalar {
        use super::*;
        #[inline]
        pub fn add_i16_row(acc: &mut [i16; N], w: &[i16; N]) {
            for i in 0..N {
                acc[i] = acc[i].wrapping_add(w[i]);
            }
        }
        #[inline]
        pub fn sub_i16_row(acc: &mut [i16; N], w: &[i16; N]) {
            for i in 0..N {
                acc[i] = acc[i].wrapping_sub(w[i]);
            }
        }
        #[inline]
        pub fn add_i8_row(acc: &mut [i16; N], w: &[i8; N]) {
            for i in 0..N {
                acc[i] = acc[i].wrapping_add(w[i] as i16);
            }
        }
        #[inline]
        pub fn sub_i8_row(acc: &mut [i16; N], w: &[i8; N]) {
            for i in 0..N {
                acc[i] = acc[i].wrapping_sub(w[i] as i16);
            }
        }
        /// Pairwise CReLU product of the two accumulator halves into `out[off..off+HALF]`.
        #[inline]
        pub fn pairwise(acc: &[i16; N], out: &mut [u8; N], off: usize) {
            for i in 0..HALF {
                let a = (acc[i] as i32).clamp(0, QA);
                let b = (acc[i + HALF] as i32).clamp(0, QA);
                out[off + i] = ((a * b) >> PAIR_SHIFT) as u8;
            }
        }
        /// L2 (f32): h2[o] = crelu(b[o] + sum_i w[o][i] * h1[i]).
        #[inline]
        pub fn l2_forward(h1: &[f32; L2_DUAL], w: &[[f32; L2_DUAL]], b: &[f32], h2: &mut [f32; L3]) {
            for o in 0..L3 {
                let mut v = b[o];
                for i in 0..L2_DUAL {
                    v += w[o][i] * h1[i];
                }
                h2[o] = v.clamp(0.0, 1.0);
            }
        }
        /// Dot products of the u8 input with L2 rows of i8 weights (rows contiguous, N each).
        #[inline]
        pub fn l1_dots(x: &[u8; N], w_rows: &[[i8; N]], out: &mut [i32; L2]) {
            for (o, w) in w_rows.iter().enumerate().take(L2) {
                let mut sum: i32 = 0;
                for i in 0..N {
                    sum += x[i] as i32 * w[i] as i32;
                }
                out[o] = sum;
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    pub mod avx2 {
        use super::*;
        use std::arch::x86_64::*;
        // SAFETY (all): arrays are 64-byte aligned (Align64) or plain arrays read with unaligned
        // loads; every index stays inside N/HALF which are multiples of 32; avx2 is a compile-time feature.
        #[inline]
        pub fn add_i16_row(acc: &mut [i16; N], w: &[i16; N]) {
            unsafe {
                let a = acc.as_mut_ptr() as *mut __m256i;
                let b = w.as_ptr() as *const __m256i;
                for i in 0..N / 16 {
                    _mm256_storeu_si256(a.add(i), _mm256_add_epi16(_mm256_loadu_si256(a.add(i)), _mm256_loadu_si256(b.add(i))));
                }
            }
        }
        #[inline]
        pub fn sub_i16_row(acc: &mut [i16; N], w: &[i16; N]) {
            unsafe {
                let a = acc.as_mut_ptr() as *mut __m256i;
                let b = w.as_ptr() as *const __m256i;
                for i in 0..N / 16 {
                    _mm256_storeu_si256(a.add(i), _mm256_sub_epi16(_mm256_loadu_si256(a.add(i)), _mm256_loadu_si256(b.add(i))));
                }
            }
        }
        #[inline]
        pub fn add_i8_row(acc: &mut [i16; N], w: &[i8; N]) {
            unsafe {
                let a = acc.as_mut_ptr() as *mut __m256i;
                for i in 0..N / 16 {
                    let wv = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(i * 16) as *const __m128i));
                    _mm256_storeu_si256(a.add(i), _mm256_add_epi16(_mm256_loadu_si256(a.add(i)), wv));
                }
            }
        }
        #[inline]
        pub fn sub_i8_row(acc: &mut [i16; N], w: &[i8; N]) {
            unsafe {
                let a = acc.as_mut_ptr() as *mut __m256i;
                for i in 0..N / 16 {
                    let wv = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(i * 16) as *const __m128i));
                    _mm256_storeu_si256(a.add(i), _mm256_sub_epi16(_mm256_loadu_si256(a.add(i)), wv));
                }
            }
        }
        #[inline]
        pub fn pairwise(acc: &[i16; N], out: &mut [u8; N], off: usize) {
            unsafe {
                let zero = _mm256_setzero_si256();
                let qa = _mm256_set1_epi16(QA as i16);
                let p = acc.as_ptr() as *const __m256i;
                // Process 32 outputs per iteration (two i16 vectors -> one packed u8 vector).
                for i in 0..HALF / 32 {
                    let a0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p.add(2 * i)), zero), qa);
                    let a1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p.add(2 * i + 1)), zero), qa);
                    let b0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p.add(HALF / 16 + 2 * i)), zero), qa);
                    let b1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p.add(HALF / 16 + 2 * i + 1)), zero), qa);
                    // (a*b) >> 9 == ((a << 7) * b) >> 16 (a*b < 65536).
                    let p0 = _mm256_mulhi_epu16(_mm256_slli_epi16(a0, 16 - PAIR_SHIFT as i32), b0);
                    let p1 = _mm256_mulhi_epu16(_mm256_slli_epi16(a1, 16 - PAIR_SHIFT as i32), b1);
                    // packus interleaves 128-bit lanes: fix the order with a permute.
                    let packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0b11_01_10_00);
                    _mm256_storeu_si256(out.as_mut_ptr().add(off + i * 32) as *mut __m256i, packed);
                }
            }
        }
        #[inline]
        pub fn l1_dots(x: &[u8; N], w_rows: &[[i8; N]], out: &mut [i32; L2]) {
            unsafe {
                let ones = _mm256_set1_epi16(1);
                let xp = x.as_ptr() as *const __m256i;
                for (o, w) in w_rows.iter().enumerate().take(L2) {
                    let wp = w.as_ptr() as *const __m256i;
                    let mut acc0 = _mm256_setzero_si256();
                    let mut acc1 = _mm256_setzero_si256();
                    for i in (0..N / 32).step_by(2) {
                        // x <= 127 and |w| <= 127: pair sums <= 2*127*127 < 32767, no saturation.
                        let p0 = _mm256_maddubs_epi16(_mm256_loadu_si256(xp.add(i)), _mm256_loadu_si256(wp.add(i)));
                        let p1 = _mm256_maddubs_epi16(_mm256_loadu_si256(xp.add(i + 1)), _mm256_loadu_si256(wp.add(i + 1)));
                        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones));
                        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p1, ones));
                    }
                    let acc = _mm256_add_epi32(acc0, acc1);
                    let hi = _mm256_extracti128_si256(acc, 1);
                    let lo = _mm256_castsi256_si128(acc);
                    let s = _mm_add_epi32(hi, lo);
                    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
                    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
                    out[o] = _mm_cvtsi128_si32(s);
                }
            }
        }
        /// L2 forward with 8-wide FMA over the 32 inputs.
        #[inline]
        pub fn l2_forward(h1: &[f32; L2_DUAL], w: &[[f32; L2_DUAL]], b: &[f32], h2: &mut [f32; L3]) {
            unsafe {
                let zero = _mm256_setzero_ps();
                let one = _mm256_set1_ps(1.0);
                let x0 = _mm256_loadu_ps(h1.as_ptr());
                let x1 = _mm256_loadu_ps(h1.as_ptr().add(8));
                let x2 = _mm256_loadu_ps(h1.as_ptr().add(16));
                let x3 = _mm256_loadu_ps(h1.as_ptr().add(24));
                for o in 0..L3 {
                    let wp = w[o].as_ptr();
                    let mut acc = _mm256_mul_ps(x0, _mm256_loadu_ps(wp));
                    acc = _mm256_fmadd_ps(x1, _mm256_loadu_ps(wp.add(8)), acc);
                    acc = _mm256_fmadd_ps(x2, _mm256_loadu_ps(wp.add(16)), acc);
                    acc = _mm256_fmadd_ps(x3, _mm256_loadu_ps(wp.add(24)), acc);
                    // horizontal sum
                    let hi = _mm256_extractf128_ps(acc, 1);
                    let lo = _mm256_castps256_ps128(acc);
                    let s = _mm_add_ps(hi, lo);
                    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
                    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0b01));
                    let v = _mm_cvtss_f32(s) + b[o];
                    h2[o] = v.clamp(0.0, 1.0);
                }
                let _ = (zero, one);
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
        fn rnd(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        #[test]
        fn v3_kernels_match_scalar() {
            let mut s = 5u64;
            for _ in 0..100 {
                let mut acc = [0i16; N];
                for v in acc.iter_mut() {
                    *v = (rnd(&mut s) % 1201) as i16 - 400;
                }
                let mut w8 = [0i8; N];
                for v in w8.iter_mut() {
                    *v = (rnd(&mut s) % 255) as i8;
                }
                let mut a1 = acc;
                let mut a2 = acc;
                add_i8_row(&mut a1, &w8);
                scalar::add_i8_row(&mut a2, &w8);
                assert_eq!(a1[..], a2[..]);
                sub_i8_row(&mut a1, &w8);
                scalar::sub_i8_row(&mut a2, &w8);
                assert_eq!(a1[..], a2[..]);
                let mut o1 = [0u8; N];
                let mut o2 = [0u8; N];
                pairwise(&acc, &mut o1, 0);
                scalar::pairwise(&acc, &mut o2, 0);
                pairwise(&acc, &mut o1, HALF);
                scalar::pairwise(&acc, &mut o2, HALF);
                assert_eq!(o1[..], o2[..]);
                let mut rows = vec![[0i8; N]; L2];
                for r in rows.iter_mut() {
                    for v in r.iter_mut() {
                        *v = (rnd(&mut s) % 255) as i8;
                    }
                }
                let mut d1 = [0i32; L2];
                let mut d2 = [0i32; L2];
                l1_dots(&o1, &rows, &mut d1);
                scalar::l1_dots(&o1, &rows, &mut d2);
                assert_eq!(d1, d2);
                let mut h1 = [0f32; L2_DUAL];
                for v in h1.iter_mut() {
                    *v = (rnd(&mut s) % 1000) as f32 / 1000.0;
                }
                let mut w2 = vec![[0f32; L2_DUAL]; L3];
                for r in w2.iter_mut() {
                    for v in r.iter_mut() {
                        *v = (rnd(&mut s) % 2001) as f32 / 1000.0 - 1.0;
                    }
                }
                let b2: Vec<f32> = (0..L3).map(|_| (rnd(&mut s) % 2001) as f32 / 1000.0 - 1.0).collect();
                let mut r1 = [0f32; L3];
                let mut r2 = [0f32; L3];
                l2_forward(&h1, &w2, &b2, &mut r1);
                scalar::l2_forward(&h1, &w2, &b2, &mut r2);
                for i in 0..L3 {
                    assert!((r1[i] - r2[i]).abs() < 1e-4, "l2 {} vs {}", r1[i], r2[i]);
                }
            }
        }
    }
}
