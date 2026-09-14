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

/// Threat-input net at the trained width 1024 (anna-v3).
pub mod w1024 {
    pub const L1: usize = 1024;
    pub const HAS_THREATS: bool = true;
    include!("v3_body.rs");
}
/// Same architecture at half width (anna-v3b): 512 B threat rows instead of 1 KB.
pub mod w512 {
    pub const L1: usize = 512;
    pub const HAS_THREATS: bool = true;
    include!("v3_body.rs");
}
/// Quarter width (anna-v5 experiment): 256 B threat rows; v3 (1024) and v3b (512) were equal per node.
pub mod w256 {
    pub const L1: usize = 256;
    pub const HAS_THREATS: bool = true;
    include!("v3_body.rs");
}
/// anna-v4: the same multilayer/pairwise net WITHOUT threat or pawn-pair inputs (piece-square only),
/// i.e. v1's inputs with v3's output stack. No threat rows to fetch, so it runs at ~v1 speed.
pub mod w1024nt {
    pub const L1: usize = 1024;
    pub const HAS_THREATS: bool = false;
    include!("v3_body.rs");
}
pub use w1024::*;
