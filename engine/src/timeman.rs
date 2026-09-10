//! Time management.

use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct GoParams {
    pub wtime: Option<i64>,
    pub btime: Option<i64>,
    pub winc: Option<i64>,
    pub binc: Option<i64>,
    pub movestogo: Option<i64>,
    pub movetime: Option<i64>,
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub mate: Option<i32>,
    pub infinite: bool,
    pub ponder: bool,
    pub searchmoves: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct TimeManager {
    pub start: Instant,
    /// Soft limit in ms (may be exceeded when the search is unstable).
    pub optimum: f64,
    /// Hard limit in ms.
    pub maximum: f64,
    pub use_time: bool,
}

impl TimeManager {
    pub fn new(p: &GoParams, us_white: bool, ply: u16, overhead: i64) -> Self {
        let start = Instant::now();
        let (time, inc) = if us_white { (p.wtime, p.winc.unwrap_or(0)) } else { (p.btime, p.binc.unwrap_or(0)) };
        if let Some(mt) = p.movetime {
            let t = (mt - overhead).max(1) as f64;
            return TimeManager { start, optimum: t, maximum: t, use_time: true };
        }
        let Some(time) = time else {
            return TimeManager { start, optimum: f64::MAX, maximum: f64::MAX, use_time: false };
        };
        let time = time.max(1) as f64;
        let inc = inc as f64;
        let overhead = overhead as f64;
        let mtg = p.movestogo.map(|m| (m as f64).min(50.0)).unwrap_or(50.0);
        // Stockfish-derived formulas (search.cpp / timeman.cpp, 2025).
        let time_left = (time + inc * (mtg - 1.0) - overhead * (2.0 + mtg)).max(1.0);
        let ply = ply as f64;
        let (opt_scale, max_scale);
        if p.movestogo.is_none() {
            let opt_constant = (0.0032 + 0.000321 * (time / 1000.0).min(120.0).ln().max(0.0)).min(0.0060);
            let max_constant = (3.39 + 3.01 * (time / 1000.0).min(120.0).ln().max(0.0)).max(2.93);
            opt_scale = (0.0122 + (ply + 3.0).powf(0.45) * opt_constant.max(0.0032)).min(0.213 * time / time_left);
            max_scale = (max_constant + ply / 12.0).min(6.64);
        } else {
            opt_scale = ((0.88 + ply / 116.4) / mtg).min(0.88 * time / time_left);
            max_scale = (1.3 + 0.11 * mtg).min(8.45);
        }
        let optimum = opt_scale * time_left;
        let maximum = (max_scale * optimum).min(0.825 * time - overhead).max(1.0);
        TimeManager { start, optimum: optimum.max(1.0), maximum, use_time: true }
    }

    #[inline]
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}
