//! Data inspection and cross-checks that can run without a GPU:
//!   inspect stats <binpack> [max_entries]  - filter statistics, score/result/ply distributions, sample FENs
//!   inspect features <binpack> [n]         - verify bullet's feature indices == engine's for n positions
//!   inspect roundtrip                      - engine bulletformat writer == bulletformat crate for test FENs
use bullet::game::inputs::{ChessBucketsMirrored, SparseInputType};
use bullet::game::outputs::{MaterialCount, OutputBuckets};
use bulletformat::ChessBoard;
use engine::nnue::{feature_index, output_bucket};
use engine::position::Position;
use engine::types::*;
use sfbinpack::chess::color::Color as SfColor;
use sfbinpack::chess::piecetype::PieceType as SfPt;
use sfbinpack::{CompressedTrainingDataEntryReader, TrainingDataEntry};
use std::fs::File;
use trainer::{filter, BUCKET_LAYOUT, OUTPUT_BUCKETS};

fn to_bullet(entry: &TrainingDataEntry) -> ChessBoard {
    // Exactly what bullet's SfBinpackLoader does (value/loader/sfbinpack.rs at rev 629ee50).
    let mut bbs = [0u64; 8];
    let stm = usize::from(entry.pos.side_to_move().ordinal());
    let pc_bb = |pt| entry.pos.pieces_bb_color(SfColor::Black, pt).bits() | entry.pos.pieces_bb_color(SfColor::White, pt).bits();
    bbs[0] = entry.pos.pieces_bb(SfColor::White).bits();
    bbs[1] = entry.pos.pieces_bb(SfColor::Black).bits();
    bbs[2] = pc_bb(SfPt::Pawn);
    bbs[3] = pc_bb(SfPt::Knight);
    bbs[4] = pc_bb(SfPt::Bishop);
    bbs[5] = pc_bb(SfPt::Rook);
    bbs[6] = pc_bb(SfPt::Queen);
    bbs[7] = pc_bb(SfPt::King);
    let mut score = entry.score;
    let mut result = f32::from(1 + entry.result) / 2.0;
    if stm > 0 {
        score = -score;
        result = 1.0 - result;
    }
    ChessBoard::from_raw(bbs, stm, score, result).expect("malformed")
}

/// Engine-side feature lists for the side to move (stm) and the other side (ntm), sorted.
fn engine_features(pos: &Position) -> (Vec<usize>, Vec<usize>) {
    let us = pos.side_to_move();
    let mut stm = Vec::new();
    let mut ntm = Vec::new();
    for s in 0..64u8 {
        let p = pos.piece_on(s);
        if p.is_none() {
            continue;
        }
        let rk_us = if us == Color::Black { pos.king_sq(us) ^ 56 } else { pos.king_sq(us) };
        let rk_them = if !us == Color::Black { pos.king_sq(!us) ^ 56 } else { pos.king_sq(!us) };
        stm.push(feature_index(us, rk_us, p, s));
        ntm.push(feature_index(!us, rk_them, p, s));
    }
    stm.sort_unstable();
    ntm.sort_unstable();
    (stm, ntm)
}

fn bullet_features(board: &ChessBoard) -> (Vec<usize>, Vec<usize>) {
    let inputs = ChessBucketsMirrored::new(BUCKET_LAYOUT);
    let mut stm = Vec::new();
    let mut ntm = Vec::new();
    inputs.map_features(board, |a, b| {
        stm.push(a);
        ntm.push(b);
    });
    stm.sort_unstable();
    ntm.sort_unstable();
    (stm, ntm)
}

fn check_position(fen: &str, board: &ChessBoard) -> Result<(), String> {
    let pos = Position::from_fen(fen).map_err(|e| format!("{}: {}", fen, e))?;
    let (es, en) = engine_features(&pos);
    let (bs, bn) = bullet_features(board);
    if es != bs || en != bn {
        return Err(format!("feature mismatch for {}\n engine stm {:?}\n bullet stm {:?}\n engine ntm {:?}\n bullet ntm {:?}", fen, es, bs, en, bn));
    }
    let eb = output_bucket(&pos);
    let bb = MaterialCount::<OUTPUT_BUCKETS>.bucket(board) as usize;
    if eb != bb {
        return Err(format!("output bucket mismatch for {}: engine {} bullet {}", fen, eb, bb));
    }
    Ok(())
}

fn main() {
    engine::init();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("stats") => {
            let path = &args[2];
            let max: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2_000_000);
            let mut reader = CompressedTrainingDataEntryReader::new(File::open(path).unwrap()).unwrap();
            let (mut total, mut kept) = (0u64, 0u64);
            let mut score_hist = [0u64; 9]; // <-1000,-1000..-300,-300..-100,-100..0,0,0..100,100..300,300..1000,>1000
            let mut results = [0u64; 3];
            let mut ply_hist = [0u64; 6]; // <16, 16-40, 40-80, 80-120, 120-200, >200
            let mut samples = Vec::new();
            let mut checks = 0;
            while reader.has_next() && total < max {
                let e = reader.next();
                total += 1;
                if !filter(&e) {
                    continue;
                }
                kept += 1;
                let s = e.score as i32;
                let bin = if s < -1000 { 0 } else if s < -300 { 1 } else if s < -100 { 2 } else if s < 0 { 3 } else if s == 0 { 4 } else if s <= 100 { 5 } else if s <= 300 { 6 } else if s <= 1000 { 7 } else { 8 };
                score_hist[bin] += 1;
                results[(e.result + 1) as usize] += 1;
                let p = e.ply;
                let pb = if p < 16 { 0 } else if p < 40 { 1 } else if p < 80 { 2 } else if p < 120 { 3 } else if p < 200 { 4 } else { 5 };
                ply_hist[pb] += 1;
                if samples.len() < 5 && kept % 9973 == 1 {
                    samples.push(format!("{} | score {} | result {} | ply {}", e.pos.fen().unwrap(), e.score, e.result, e.ply));
                }
                if kept % 1000 == 0 {
                    let fen = e.pos.fen().unwrap();
                    if let Err(msg) = check_position(&fen, &to_bullet(&e)) {
                        eprintln!("{}", msg);
                        std::process::exit(2);
                    }
                    checks += 1;
                }
            }
            let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let read = reader.read_bytes();
            println!("file: {} ({} bytes)", path, file_size);
            println!("entries read: {}  kept by filter: {} ({:.1}%)", total, kept, 100.0 * kept as f64 / total.max(1) as f64);
            if read > 0 && file_size > 0 {
                let est_total = total as f64 * file_size as f64 / read as f64;
                println!("bytes consumed: {}  -> estimated entries in file: {:.0}M, kept after filter: {:.0}M", read, est_total / 1e6, est_total * kept as f64 / total.max(1) as f64 / 1e6);
            }
            println!("score bins [<-1000, -1000..-300, -300..-100, -100..0, 0, 0..100, 100..300, 300..1000, >1000]: {:?}", score_hist);
            println!("results (stm loss/draw/win): {:?}", results);
            println!("ply bins [<16, 16-40, 40-80, 80-120, 120-200, >200]: {:?}", ply_hist);
            println!("feature cross-checks passed: {}", checks);
            for s in samples {
                println!("sample: {}", s);
            }
        }
        Some("features") => {
            let path = &args[2];
            let n: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200_000);
            let mut reader = CompressedTrainingDataEntryReader::new(File::open(path).unwrap()).unwrap();
            let mut checked = 0;
            while reader.has_next() && checked < n {
                let e = reader.next();
                let fen = e.pos.fen().unwrap();
                if let Err(msg) = check_position(&fen, &to_bullet(&e)) {
                    eprintln!("{}", msg);
                    std::process::exit(2);
                }
                checked += 1;
            }
            println!("features ok: {} positions cross-checked (engine feature_index == bullet ChessBucketsMirrored, output buckets equal)", checked);
        }
        Some("features2") => {
            // Threat + pawn-pair features: engine (nnue::threats) vs bullet's example implementation.
            use engine::nnue::threats::FeatureMapper;
            use trainer::bullet_inputs::{three_file_band_mask, PawnPawnInputs};
            let path = &args[2];
            let n: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200_000);
            let mine = FeatureMapper::new();
            let theirs = PawnPawnInputs::new(three_file_band_mask());
            assert_eq!(mine.num_inputs(), theirs.num_inputs());
            let mut reader = CompressedTrainingDataEntryReader::new(File::open(path).unwrap()).unwrap();
            let mut checked = 0u64;
            let mut max_active = 0usize;
            while reader.has_next() && checked < n {
                let e = reader.next();
                let fen = e.pos.fen().unwrap();
                let pos = Position::from_fen(&fen).unwrap();
                let board = to_bullet(&e);
                let mut bs = Vec::new();
                let mut bn = Vec::new();
                theirs.map_features(&board, |s| bs.push(s), |t| bn.push(t));
                bs.sort_unstable();
                bn.sort_unstable();
                let (ew, eb) = mine.features_for(&pos);
                let (es, en) = if pos.side_to_move() == Color::White { (ew, eb) } else { (eb, ew) };
                if es != bs || en != bn {
                    eprintln!("threat/pair feature mismatch for {}\n engine stm {:?}\n bullet stm {:?}\n engine ntm {:?}\n bullet ntm {:?}", fen, es, bs, en, bn);
                    std::process::exit(2);
                }
                max_active = max_active.max(es.len()).max(en.len());
                checked += 1;
            }
            println!("features2 ok: {} positions, engine threat+pawn-pair features == bullet's (max active {})", checked, max_active);
        }
        Some("roundtrip") => {
            // Engine's bulletformat writer must produce byte-identical records to the bulletformat crate.
            let fens = [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
                "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
                "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9",
            ];
            for fen in fens {
                for (score, result) in [(123i16, 1.0f32), (-45, 0.5), (0, 0.0)] {
                    let pos = Position::from_fen(fen).unwrap();
                    let mine = engine::datagen::to_bulletformat(&pos, score, result);
                    let mut bbs = [0u64; 8];
                    bbs[0] = pos.colored(Color::White);
                    bbs[1] = pos.colored(Color::Black);
                    for (i, pt) in PieceType::ALL.iter().enumerate() {
                        bbs[2 + i] = pos.pieces(*pt);
                    }
                    let theirs = ChessBoard::from_raw(bbs, pos.side_to_move().idx(), score, result).unwrap();
                    let theirs_bytes: [u8; 32] = unsafe { std::mem::transmute(theirs) };
                    if mine != theirs_bytes {
                        eprintln!("roundtrip mismatch for {} score {} result {}\n mine   {:?}\n bullet {:?}", fen, score, result, mine, theirs_bytes);
                        std::process::exit(2);
                    }
                    // And the features of the bullet-parsed board match the engine's.
                    check_position(fen, &theirs).unwrap_or_else(|m| {
                        eprintln!("{}", m);
                        std::process::exit(2)
                    });
                }
            }
            println!("roundtrip ok: engine bulletformat writer is byte-identical to the bulletformat crate on {} fens", fens.len());
        }
        _ => {
            eprintln!("usage: inspect stats <binpack> [max] | features <binpack> [n] | roundtrip");
            std::process::exit(1);
        }
    }
}
