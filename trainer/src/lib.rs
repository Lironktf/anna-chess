//! Shared pieces between the trainer binary and the inspection tools.
use sfbinpack::chess::{piecetype::PieceType, r#move::MoveType};
use sfbinpack::TrainingDataEntry;

pub const L1: usize = 1024;
pub const OUTPUT_BUCKETS: usize = 8;
pub const QA: i16 = 255;
pub const QB: i16 = 64;
pub const SCALE: f32 = 400.0;

#[rustfmt::skip]
pub const BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

/// The standard Leela-T80 training filter: skip the opening, positions in check, huge scores,
/// and positions whose best move is a capture or special move (noisy for a static evaluation).
pub fn filter(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}
