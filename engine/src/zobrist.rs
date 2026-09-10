//! Zobrist hashing keys (deterministic PRNG so hashes are stable across builds).

use crate::types::*;

pub struct Zobrist {
    pub psq: [[u64; 64]; 13],
    pub ep: [u64; 8],
    pub castling: [u64; 16],
    pub side: u64,
    pub no_pawns: u64,
}

const fn splitmix(mut x: u64) -> (u64, u64) {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31), x)
}

const fn build() -> Zobrist {
    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let mut psq = [[0u64; 64]; 13];
    let mut p = 0;
    while p < 12 {
        let mut s = 0;
        while s < 64 {
            let (v, ns) = splitmix(seed);
            seed = ns;
            psq[p][s] = v;
            s += 1;
        }
        p += 1;
    }
    let mut ep = [0u64; 8];
    let mut f = 0;
    while f < 8 {
        let (v, ns) = splitmix(seed);
        seed = ns;
        ep[f] = v;
        f += 1;
    }
    let mut castling = [0u64; 16];
    // Key for a castling-rights mask is the xor of keys of its set bits, so that
    // changing one right changes the hash consistently.
    let mut bit_keys = [0u64; 4];
    let mut b = 0;
    while b < 4 {
        let (v, ns) = splitmix(seed);
        seed = ns;
        bit_keys[b] = v;
        b += 1;
    }
    let mut m = 0;
    while m < 16 {
        let mut k = 0u64;
        let mut i = 0;
        while i < 4 {
            if m & (1 << i) != 0 {
                k ^= bit_keys[i];
            }
            i += 1;
        }
        castling[m] = k;
        m += 1;
    }
    let (side, ns) = splitmix(seed);
    let (no_pawns, _) = splitmix(ns);
    Zobrist { psq, ep, castling, side, no_pawns }
}

pub static ZOBRIST: Zobrist = build();

#[inline(always)]
pub fn psq_key(p: Piece, s: Square) -> u64 {
    ZOBRIST.psq[p.idx()][s as usize]
}
