pub mod bitboard;
pub mod cuckoo;
pub mod params;
pub mod datagen;
pub mod history;
pub mod movepick;
pub mod search;
pub mod tb;
pub mod timeman;
pub mod tt;
pub mod uci;
pub mod eval;
pub mod nnue;
pub mod movegen;
pub mod position;
pub mod types;
pub mod zobrist;

pub fn init() {
    bitboard::init();
    cuckoo::init();
}
