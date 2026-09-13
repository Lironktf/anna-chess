//! Runtime-tunable integer parameters (set through UCI `setoption`), so one binary can A/B a
//! feature or constant in SPRT matches: `-engine ... option.MatScale=0`.

use std::sync::atomic::{AtomicI32, Ordering};

pub struct Param {
    pub name: &'static str,
    pub value: AtomicI32,
    pub default: i32,
    pub min: i32,
    pub max: i32,
}

macro_rules! params {
    ($($id:ident, $name:literal, $default:expr, $min:expr, $max:expr;)*) => {
        $(pub static $id: Param = Param { name: $name, value: AtomicI32::new($default), default: $default, min: $min, max: $max };)*
        pub static ALL: &[&Param] = &[$(&$id),*];
    };
}

params! {
    MAT_SCALE, "MatScale", 0, 0, 1;
    CUCKOO, "Cuckoo", 1, 0, 1;
    // Main history indexed by whether the from/to squares are attacked by the opponent (0 = plain butterfly).
    THREAT_HIST, "ThreatHist", 0, 0, 1;
    // Stockfish (2026) singular-extension margins: TT-move history term, corr adjustment in the triple margin,
    // and the ply>rootDepth terms; also maintains the TT-move history counter. 0 = previous margins.
    SE_TTM, "SeTtmHist", 0, 0, 1;
    MAT_SCALE_BASE, "MatScaleBase", 700, 400, 1024;
    MAT_SCALE_DIV, "MatScaleDiv", 16, 4, 64;
    // Time management (percent scalers)
    TM_OPT_PCT, "TmOptPct", 100, 30, 300;
    TM_FALLING_FIX, "TmFallingFix", 1, 0, 1;
    TM_INSTAB_PCT, "TmInstabPct", 180, 0, 400;
    // Search constants
    LMR_BASE, "LmrBase", 982, 0, 3000;
    LMR_SCALE_PCT, "LmrScalePct", 100, 50, 200;
    LMR_CUTNODE, "LmrCutNode", 3000, 0, 6000;
    LMR_HIST_DIV, "LmrHistDiv", 16384, 4096, 65536;
    RFP_MULT, "RfpMult", 45, 20, 120;
    RFP_DEPTH, "RfpDepth", 14, 4, 20;
    NMP_BASE, "NmpBase", 5, 2, 8;
    LMP_BASE, "LmpBase", 3, 1, 8;
    FUT_MARGIN, "FutMargin", 119, 40, 300;
    RAZOR_MULT, "RazorMult", 482, 100, 1000;
    SE_MARGIN, "SeMargin", 59, 20, 120;
    // Syzygy
    TB_PROBE_DEPTH, "SyzygyProbeDepth", 1, 1, 100;
    TB_PROBE_LIMIT, "SyzygyProbeLimit", 7, 0, 7;
}

impl Param {
    #[inline(always)]
    pub fn get(&self) -> i32 {
        self.value.load(Ordering::Relaxed)
    }
    pub fn set(&self, v: i32) {
        self.value.store(v.clamp(self.min, self.max), Ordering::Relaxed);
    }
}

/// Set a parameter by (case-insensitive) name; returns false if unknown.
pub fn set_by_name(name: &str, value: &str) -> bool {
    for p in ALL {
        if p.name.eq_ignore_ascii_case(name) {
            if let Ok(v) = value.trim().parse::<i32>() {
                p.set(v);
                return true;
            }
        }
    }
    false
}

pub fn print_uci_options() {
    for p in ALL {
        println!("option name {} type spin default {} min {} max {}", p.name, p.default, p.min, p.max);
    }
}
