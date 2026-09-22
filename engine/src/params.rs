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
    // Stockfish move ordering: +/- 20*piece_value for a quiet move escaping / entering a square attacked by a lesser piece.
    THREAT_ORDER, "ThreatOrder", 0, 0, 1;
    // Search-sync groups (runs/SEARCH_SYNC.md): Stockfish master (2026-09) differences, each SPRT-tested separately.
    // G1: hindsight depth adjust, in-check static eval, NMP/RFP/razoring/depth-reduction/fail-high formulas.
    SF_PRUNE, "SfPrune", 0, 0, 1;
    // G2: history bonus/malus formulas, eval-difference ordering bonus, previous-move maluses, continuation weights.
    SF_HIST, "SfHist", 1, 0, 1;
    // G3: LMR terms and constants, reductions table, LMR-depth divisor table, singular/negative-extension rules.
    SF_LMR, "SfLmr", 1, 0, 1;
    // G4: Stockfish master correction history (tables, weights, update rule, multi-cut update; no major-piece table).
    SF_CORR, "SfCorr", 1, 0, 1;
    // G5: Stockfish master move picker: no killer/counter stages, quiet scoring weights, check/threat terms,
    // capture threshold, sort limit, good/bad quiet split.
    SF_PICK, "SfPick", 1, 0, 1;
    // G6: Stockfish master quiescence: SEE threshold, stand-pat blends, decisive TT guard, stalemate case.
    SF_QS, "SfQs", 1, 0, 1;
    // G7: Stockfish master time management and root aspiration loop (falling eval, stability, effort, search-again,
    // fail-high recovery, single-move cap, mate stop; timeman constants, time advantage).
    SF_TM, "SfTm", 0, 0, 1;
    // G8: TT cutoff at depth >= 7 verified against the TT entry of the position after the TT move.
    SF_TT_VERIFY, "SfTtVerify", 0, 0, 1;
    // G7 bisect: the time-allocation constants (timeman.rs) and the iterative-deepening changes (aspiration, search-again,
    // effort/stability factors) separately. SfTm=1 still enables both.
    SF_TM_ALLOC, "SfTmAlloc", 0, 0, 1;
    SF_TM_LOOP, "SfTmLoop", 0, 0, 1;
    // Lazy SMP: give each helper thread a slightly different tree. 0 = every thread searches identically.
    // SmpThreadLmr scales the reduction table with the thread count (Stockfish's ln(threads)/2 term, x100).
    // SmpThreadDelta spreads the aspiration window across threads.
    SMP_THREAD_LMR, "SmpThreadLmr", 0, 0, 1;
    SMP_THREAD_DELTA, "SmpThreadDelta", 0, 0, 1;
    // --- constants of the synced search, exposed so SPSA can retune them for this engine ---
    // Late move reductions (the SfLmr path).
    LMR_BASE_OFF, "LmrBaseOff", 697, 200, 1400;
    LMR_STAT, "LmrStat", 439, 150, 900;
    LMR_CUT, "LmrCut", 2800, 2000, 6000;
    LMR_TTPV, "LmrTtPv", 3023, 1500, 4500;
    LMR_MOVECNT, "LmrMoveCnt", 65, 20, 140;
    LMR_ALLNODE, "LmrAllNode", 276, 80, 500;
    LMR_DEEPER, "LmrDeeper", 53, 10, 120;
    LMR_SHALLOWER, "LmrShallower", 8, 0, 40;
    // History bonus and malus.
    STAT_BONUS_MUL, "StatBonusMul", 133, 60, 260;
    STAT_BONUS_MAX, "StatBonusMax", 1487, 700, 2600;
    STAT_MALUS_MUL, "StatMalusMul", 968, 400, 1700;
    STAT_MALUS_MAX, "StatMalusMax", 2244, 1100, 3600;
    // Correction history: the divisor that turns the weighted sum into centipawns (lower = stronger).
    CORR_SCALE, "CorrScale", 512, 220, 1200;
    // Quiescence margins.
    QS_FUT_BASE, "QsFutBase", 306, 150, 520;
    QS_SEE, "QsSee", 74, 20, 200;
    // Move ordering.
    PICK_CHECK_BONUS, "PickCheckBonus", 16384, 6000, 28000;
    PICK_GOOD_QUIET, "PickGoodQuiet", 14000, 6000, 26000;
    // Time management (our own formulas; Stockfish's did not port).
    TM_FALL_BASE, "TmFallBase", 66, 0, 200;
    TM_FALL_PREV, "TmFallPrev", 14, 0, 40;
    TM_FALL_ITER, "TmFallIter", 6, 0, 24;
    TM_FALL_DIV, "TmFallDiv", 616, 300, 1000;
    TM_STABLE_HI, "TmStableHi", 156, 100, 260;
    TM_STABLE_LO, "TmStableLo", 69, 30, 130;
    TM_RED_DEN, "TmRedDen", 220, 120, 340;
    // Policy net (UCI option PolicyFile): quiet-move ordering adds PolicyScale * logit (0 = off).
    POLICY_SCALE, "PolicyScale", 0, 0, 20000;
    // Policy-guided LMR: reduce quiet moves by PolicyLmr * logit / 1024 fewer units (0 = off).
    POLICY_LMR, "PolicyLmr", 0, 0, 2000;
    // Lazy SMP: pick the final move by weighted vote over all threads (Stockfish scheme) instead of thread 0's.
    THREAD_VOTE, "ThreadVote", 1, 0, 1;
    // Two-tier evaluation: when a second net is loaded (EvalFileFast), nodes with depth < TierDepth and all quiescence
    // nodes use the fast net; interior nodes use the main net. 0 = fast net only in quiescence (if loaded at all).
    TIER_DEPTH, "TierDepth", 0, 0, 30;
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
    NMP_BASE, "NmpBase", 6, 2, 8;
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
