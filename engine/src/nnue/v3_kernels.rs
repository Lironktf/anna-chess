// v3 kernels: i8 feature rows into i16 accumulators, pairwise activation, dense u8 x i8 L1.
// Included by each width module of v3.rs (`kernels`), so N/HALF come from that module.

use super::{HALF, L1 as N, L2, L2_DUAL, L3, PAIR_SHIFT, QA};

pub mod scalar {
    use super::*;

    /// One pass: `out = base + sum(psq[add16]) - sum(psq[sub16]) + sum(pp[add8]) - sum(pp[sub8])`.
    /// Every weight row is read exactly once and the accumulator is written once (i16 wrapping).
    #[inline]
    pub fn apply_rows(
        base: &[i16; N], out: &mut [i16; N],
        psq: &[super::super::super::super::Align64<[i16; N]>], add16: &[usize], sub16: &[usize],
        pp: &[super::super::super::super::Align64<[i8; N]>], add8: &[usize], sub8: &[usize],
    ) {
        for i in 0..N {
            let mut v = base[i];
            for &r in add16 { v = v.wrapping_add(psq[r].0[i]); }
            for &r in sub16 { v = v.wrapping_sub(psq[r].0[i]); }
            for &r in add8 { v = v.wrapping_add(pp[r].0[i] as i16); }
            for &r in sub8 { v = v.wrapping_sub(pp[r].0[i] as i16); }
            out[i] = v;
        }
    }
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
    /// `pairwise` over the element-wise (wrapping i16) sum of two accumulators (piece-square part +
    /// threat part), without materialising the sum.
    #[inline]
    pub fn pairwise2(x: &[i16; N], y: &[i16; N], out: &mut [u8; N], off: usize) {
        for i in 0..HALF {
            let a = (x[i].wrapping_add(y[i]) as i32).clamp(0, QA);
            let b = (x[i + HALF].wrapping_add(y[i + HALF]) as i32).clamp(0, QA);
            out[off + i] = ((a * b) >> PAIR_SHIFT) as u8;
        }
    }
    /// L2 over transposed weights `wt[i][o]` (32 inputs x 32 outputs), no horizontal sums.
    /// Per output: acc = wt[0][o]*h1[0], then acc = fma(wt[i][o], h1[i], acc) for i = 1..32; crelu(acc + b[o]).
    #[inline]
    pub fn l2_forward_t(h1: &[f32; L2_DUAL], wt: &[[f32; L3]], b: &[f32], h2: &mut [f32; L3]) {
        for o in 0..L3 {
            let mut acc = wt[0][o] * h1[0];
            for i in 1..L2_DUAL {
                acc = wt[i][o].mul_add(h1[i], acc);
            }
            h2[o] = (acc + b[o]).clamp(0.0, 1.0);
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

    /// See scalar::apply_rows. Per 16-lane chunk all rows are summed in registers, so the
    /// accumulator is read and written once instead of once per row.
    #[inline]
    pub fn apply_rows(
        base: &[i16; N], out: &mut [i16; N],
        psq: &[super::super::super::super::Align64<[i16; N]>], add16: &[usize], sub16: &[usize],
        pp: &[super::super::super::super::Align64<[i8; N]>], add8: &[usize], sub8: &[usize],
    ) {
        unsafe {
            let b = base.as_ptr() as *const __m256i;
            let o = out.as_mut_ptr() as *mut __m256i;
            for i in 0..N / 16 {
                let mut v = _mm256_loadu_si256(b.add(i));
                for &r in add16 { v = _mm256_add_epi16(v, _mm256_loadu_si256((psq.get_unchecked(r).0.as_ptr() as *const __m256i).add(i))); }
                for &r in sub16 { v = _mm256_sub_epi16(v, _mm256_loadu_si256((psq.get_unchecked(r).0.as_ptr() as *const __m256i).add(i))); }
                for &r in add8 { v = _mm256_add_epi16(v, _mm256_cvtepi8_epi16(_mm_loadu_si128(pp.get_unchecked(r).0.as_ptr().add(i * 16) as *const __m128i))); }
                for &r in sub8 { v = _mm256_sub_epi16(v, _mm256_cvtepi8_epi16(_mm_loadu_si128(pp.get_unchecked(r).0.as_ptr().add(i * 16) as *const __m128i))); }
                _mm256_storeu_si256(o.add(i), v);
            }
        }
    }
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

    /// See scalar::pairwise2.
    #[inline]
    pub fn pairwise2(x: &[i16; N], y: &[i16; N], out: &mut [u8; N], off: usize) {
        unsafe {
            let zero = _mm256_setzero_si256();
            let qa = _mm256_set1_epi16(QA as i16);
            let p = x.as_ptr() as *const __m256i;
            let q = y.as_ptr() as *const __m256i;
            // Process 32 outputs per iteration (two i16 vectors -> one packed u8 vector).
            for i in 0..HALF / 32 {
                let a0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(_mm256_loadu_si256(p.add(2 * i)), _mm256_loadu_si256(q.add(2 * i))), zero), qa);
                let a1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(_mm256_loadu_si256(p.add(2 * i + 1)), _mm256_loadu_si256(q.add(2 * i + 1))), zero), qa);
                let b0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(_mm256_loadu_si256(p.add(HALF / 16 + 2 * i)), _mm256_loadu_si256(q.add(HALF / 16 + 2 * i))), zero), qa);
                let b1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(_mm256_loadu_si256(p.add(HALF / 16 + 2 * i + 1)), _mm256_loadu_si256(q.add(HALF / 16 + 2 * i + 1))), zero), qa);
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
    /// L2 over transposed weights: 4 accumulators of 8 outputs, one broadcast per input, no horizontal sums.
    /// Bit-identical to scalar::l2_forward_t (same mul-then-fma order per output).
    #[inline]
    pub fn l2_forward_t(h1: &[f32; L2_DUAL], wt: &[[f32; L3]], b: &[f32], h2: &mut [f32; L3]) {
        unsafe {
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);
            let w0 = wt[0].as_ptr();
            let x = _mm256_set1_ps(h1[0]);
            let mut a0 = _mm256_mul_ps(_mm256_loadu_ps(w0), x);
            let mut a1 = _mm256_mul_ps(_mm256_loadu_ps(w0.add(8)), x);
            let mut a2 = _mm256_mul_ps(_mm256_loadu_ps(w0.add(16)), x);
            let mut a3 = _mm256_mul_ps(_mm256_loadu_ps(w0.add(24)), x);
            for i in 1..L2_DUAL {
                let wp = wt[i].as_ptr();
                let x = _mm256_set1_ps(h1[i]);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(wp), x, a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(8)), x, a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(16)), x, a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(24)), x, a3);
            }
            let bp = b.as_ptr();
            let o = h2.as_mut_ptr();
            for (k, a) in [a0, a1, a2, a3].into_iter().enumerate() {
                let v = _mm256_add_ps(a, _mm256_loadu_ps(bp.add(8 * k)));
                _mm256_storeu_ps(o.add(8 * k), _mm256_min_ps(_mm256_max_ps(v, zero), one));
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

#[cfg(test)]
mod tests_pairwise2 {
    use super::*;
    #[test]
    fn pairwise2_matches_scalar() {
        let mut x = [0i16; N];
        let mut y = [0i16; N];
        let mut seed = 12345u64;
        for i in 0..N {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            x[i] = (seed % 700) as i16 - 200;
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            y[i] = (seed % 700) as i16 - 200;
        }
        let mut a = [0u8; N];
        let mut b = [0u8; N];
        pairwise2(&x, &y, &mut a, 0);
        scalar::pairwise2(&x, &y, &mut b, 0);
        assert_eq!(a[..HALF], b[..HALF]);
        // and equals pairwise over the materialised sum
        let mut sum = [0i16; N];
        for i in 0..N { sum[i] = x[i].wrapping_add(y[i]); }
        let mut c = [0u8; N];
        scalar::pairwise(&sum, &mut c, 0);
        assert_eq!(a[..HALF], c[..HALF]);
    }
}

#[cfg(test)]
mod tests_l2t {
    use super::*;
    #[test]
    fn l2_forward_t_matches_scalar_and_old() {
        let mut seed = 99u64;
        let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed % 2000) as f32 - 1000.0) / 1000.0 };
        let mut h1 = [0f32; L2_DUAL];
        for v in h1.iter_mut() { *v = rnd().abs(); }
        let mut w = [[0f32; L2_DUAL]; L3];
        for o in 0..L3 { for i in 0..L2_DUAL { w[o][i] = rnd(); } }
        let mut wt = [[0f32; L3]; L2_DUAL];
        for o in 0..L3 { for i in 0..L2_DUAL { wt[i][o] = w[o][i]; } }
        let mut b = [0f32; L3];
        for v in b.iter_mut() { *v = rnd(); }
        let (mut a, mut c, mut d) = ([0f32; L3], [0f32; L3], [0f32; L3]);
        l2_forward_t(&h1, &wt, &b, &mut a);
        scalar::l2_forward_t(&h1, &wt, &b, &mut c);
        assert_eq!(a, c, "avx2 vs scalar transposed");
        l2_forward(&h1, &w, &b, &mut d);
        for o in 0..L3 { assert!((a[o] - d[o]).abs() < 1e-5, "vs old kernel at {o}: {} {}", a[o], d[o]); }
    }
}
