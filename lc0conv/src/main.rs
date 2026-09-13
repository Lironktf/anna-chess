//! lc0conv: Leela Chess Zero V6 training chunks -> compact policy records.
//!
//!   lc0conv convert <out.bin> <in.tar>...      # tars of .gz chunks (storage.lczero.org training-run1-test80-*.tar)
//!   lc0conv convert <out.bin> -                 # one tar streamed on stdin (curl url | lc0conv convert out.bin -)
//!   lc0conv verify <in.tar>...                  # decode + legality checks only, print statistics
//!   lc0conv dump <out.bin> [n]                  # print the first n records as FEN + policy
//!
//! Decoding follows lc0's src/neural/encoder.cc, src/trainingdata/trainingdata.cc and src/utils/bititer.h at
//! master (2026-09): planes are stored ReverseBitsInBytes'd, in the side-to-move perspective (board mirrored for
//! black), with the canonicalisation transform (flip=1, mirror=2, transpose=4) in invariance_info bits 0-2 and
//! "black to move" in bit 7 for the canonical formats (input_format >= 3). Policy indices refer to moves in that
//! same transformed frame and are undone exactly like lc0's MoveFromNNIndex.
//!
//! Every decoded position is rebuilt in the engine from a FEN and every policy move with probability >= 0 is
//! checked against the engine's legal move list; records with any mismatch are dropped and counted, so a
//! decoding error shows up as a non-zero drop rate rather than silently corrupt data.

mod policy_index;

use std::collections::HashMap;
use std::io::{BufWriter, Read, Write};

use engine::movegen::legal_moves;
use engine::position::Position;
use engine::types::{Move, PieceType, Square, FLAG_CASTLE};

const RECORD: usize = 8356;
const MAX_POLICY: usize = 12;
/// Output record size in bytes (see `write_record`).
pub const OUT_RECORD: usize = 96;

fn reverse_bits_in_bytes(mut v: u64) -> u64 {
    v = ((v >> 1) & 0x5555_5555_5555_5555) | ((v & 0x5555_5555_5555_5555) << 1);
    v = ((v >> 2) & 0x3333_3333_3333_3333) | ((v & 0x3333_3333_3333_3333) << 2);
    v = ((v >> 4) & 0x0F0F_0F0F_0F0F_0F0F) | ((v & 0x0F0F_0F0F_0F0F_0F0F) << 4);
    v
}
fn reverse_bytes_in_bytes(v: u64) -> u64 {
    v.swap_bytes()
}
/// Transpose across the diagonal connecting bit 7 to bit 56 (lc0's TransposeBitsInBytes).
fn transpose_bits_in_bytes(mut v: u64) -> u64 {
    v = (v & 0xAA00_AA00_AA00_AA00) >> 9 | (v & 0x0055_0055_0055_0055) << 9 | (v & 0x55AA_55AA_55AA_55AA);
    v = (v & 0xCCCC_0000_CCCC_0000) >> 18 | (v & 0x0000_3333_0000_3333) << 18 | (v & 0x3333_CCCC_3333_CCCC);
    v = (v & 0xF0F0_F0F0_0000_0000) >> 36 | (v & 0x0000_0000_0F0F_0F0F) << 36 | (v & 0x0F0F_0F0F_F0F0_F0F0);
    v
}
const FLIP: u8 = 1;
const MIRROR: u8 = 2;
const TRANSPOSE: u8 = 4;

/// Undo the canonicalisation transform on a bitboard (inverse order of lc0's forward application).
fn untransform_bb(mut v: u64, t: u8) -> u64 {
    if t & TRANSPOSE != 0 {
        v = transpose_bits_in_bytes(v);
    }
    if t & MIRROR != 0 {
        v = reverse_bytes_in_bytes(v);
    }
    if t & FLIP != 0 {
        v = reverse_bits_in_bytes(v);
    }
    v
}

/// lc0's `Transform(Square, transform)`: mirror|transpose flip the rank, flip|transpose flop the file.
fn transform_sq(sq: u8, t: u8) -> u8 {
    let (mut file, mut rank) = (sq & 7, sq >> 3);
    if t & (MIRROR | TRANSPOSE) != 0 {
        rank ^= 7;
    }
    if t & (FLIP | TRANSPOSE) != 0 {
        file ^= 7;
    }
    rank * 8 + file
}

/// lc0's `MoveFromNNIndex`: policy index -> (from, to, promotion char) in the side-to-move frame.
fn move_from_nn_index(idx: usize, t: u8) -> (u8, u8, u8) {
    let s = policy_index::POLICY_INDEX[idx].as_bytes();
    let parse = |b: &[u8]| (b[1] - b'1') * 8 + (b[0] - b'a');
    let (mut from, mut to) = (parse(&s[0..2]), parse(&s[2..4]));
    if t != 0 {
        let inv = if t & TRANSPOSE != 0 {
            let mut i = TRANSPOSE;
            if t & FLIP != 0 {
                i |= MIRROR;
            }
            if t & MIRROR != 0 {
                i |= FLIP;
            }
            i
        } else {
            t
        };
        to = transform_sq(to, inv);
        from = transform_sq(from, inv);
    }
    (from, to, if s.len() == 5 { s[4] } else { 0 })
}

#[derive(Debug)]
pub struct Decoded {
    pub input_format: u32,
    pub transform: u8,
    pub fen: String,
    pub stm_black: bool,
    /// (from, to, promo char or 0) in absolute (white-perspective) coordinates, with the probability.
    pub policy: Vec<(u8, u8, u8, f32)>,
    pub best: (u8, u8, u8),
    pub played: (u8, u8, u8),
    pub best_q: f32,
    pub result_q: f32,
    pub plies_left: f32,
}

fn sq_str(sq: u8) -> String {
    format!("{}{}", (b'a' + (sq & 7)) as char, (b'1' + (sq >> 3)) as char)
}

fn f32_at(r: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(r[off..off + 4].try_into().unwrap())
}

pub fn decode(r: &[u8]) -> Result<Decoded, &'static str> {
    let version = u32::from_le_bytes(r[0..4].try_into().unwrap());
    if version != 6 {
        return Err("version");
    }
    let input_format = u32::from_le_bytes(r[4..8].try_into().unwrap());
    let canonical = input_format >= 3;
    let planes_off = 8 + 1858 * 4;
    let plane = |i: usize| u64::from_le_bytes(r[planes_off + i * 8..planes_off + i * 8 + 8].try_into().unwrap());
    let b = planes_off + 104 * 8;
    let (us_ooo, us_oo, them_ooo, them_oo) = (r[b], r[b + 1], r[b + 2], r[b + 3]);
    let stm_or_ep = r[b + 4];
    let rule50 = r[b + 5];
    let inv = r[b + 6];
    let t = if canonical { inv & 7 } else { 0 };
    let stm_black = if canonical { inv & 0x80 != 0 } else { stm_or_ep != 0 };

    // 12 piece bitboards in the mover's frame, then to absolute coordinates.
    let mut bbs = [0u64; 12];
    for (i, bb) in bbs.iter_mut().enumerate() {
        let mut v = reverse_bits_in_bytes(plane(i));
        v = untransform_bb(v, t);
        if stm_black {
            v = reverse_bytes_in_bytes(v);
        }
        *bb = v;
    }
    // bbs[0..6] = mover's P N B R Q K, bbs[6..12] = opponent's.
    let (white, black): ([u64; 6], [u64; 6]) = if stm_black {
        (bbs[6..12].try_into().unwrap(), bbs[0..6].try_into().unwrap())
    } else {
        (bbs[0..6].try_into().unwrap(), bbs[6..12].try_into().unwrap())
    };
    if white[5].count_ones() != 1 || black[5].count_ones() != 1 {
        return Err("kings");
    }
    let occ_w = white.iter().fold(0, |a, b| a | b);
    let occ_b = black.iter().fold(0, |a, b| a | b);
    if occ_w & occ_b != 0 {
        return Err("overlap");
    }
    // FEN board.
    let mut fen = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            let sq = rank * 8 + file;
            let bit = 1u64 << sq;
            let mut c = None;
            for (i, ch) in "PNBRQK".bytes().enumerate() {
                if white[i] & bit != 0 {
                    c = Some(ch);
                }
                if black[i] & bit != 0 {
                    c = Some(ch.to_ascii_lowercase());
                }
            }
            match c {
                Some(ch) => {
                    if empty > 0 {
                        fen.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    fen.push(ch as char);
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            fen.push((b'0' + empty) as char);
        }
        if rank > 0 {
            fen.push('/');
        }
    }
    fen.push(' ');
    fen.push(if stm_black { 'b' } else { 'w' });
    fen.push(' ');
    // Castling: the record's "us" is the side to move. Canonical transforms are never applied when any castling
    // right exists, so no untransforming is needed here.
    let (w_oo, w_ooo, b_oo, b_ooo) = if stm_black { (them_oo, them_ooo, us_oo, us_ooo) } else { (us_oo, us_ooo, them_oo, them_ooo) };
    let mut cs = String::new();
    if w_oo != 0 {
        cs.push('K');
    }
    if w_ooo != 0 {
        cs.push('Q');
    }
    if b_oo != 0 {
        cs.push('k');
    }
    if b_ooo != 0 {
        cs.push('q');
    }
    if cs.is_empty() {
        cs.push('-');
    }
    fen.push_str(&cs);
    fen.push(' ');
    // En passant: canonical formats store the file mask (bit-reversed if the flip transform was applied; with pawns
    // on the board only the flip transform can be active).
    let mut ep_file: Option<u8> = None;
    if canonical {
        if stm_or_ep != 0 {
            let mut m = stm_or_ep;
            if t & FLIP != 0 {
                m = m.reverse_bits();
            }
            ep_file = Some(m.trailing_zeros() as u8);
        }
    } else {
        // Classical format: no en passant field. Infer it from the previous position (history set 1, whose
        // boards the encoder already put in the current mover's perspective): the opponent pawn that stood two
        // ranks further back one ply ago and now stands on its 4th rank just double-pushed.
        let mut prev_their_pawns = reverse_bits_in_bytes(plane(13 + 6));
        if stm_black {
            prev_their_pawns = reverse_bytes_in_bytes(prev_their_pawns);
        }
        let (opp_now, rank_now, rank_prev) = if stm_black { (white[0], 0xFFu64 << 24, 0xFFu64 << 8) } else { (black[0], 0xFFu64 << 32, 0xFFu64 << 48) };
        let mut cand = opp_now & rank_now;
        while cand != 0 {
            let sq = cand.trailing_zeros() as u8;
            cand &= cand - 1;
            let file = sq & 7;
            let prev_sq = if stm_black { 8 + file } else { 48 + file };
            if prev_their_pawns & rank_prev & (1u64 << prev_sq) != 0 && opp_now & (1u64 << prev_sq) == 0 {
                ep_file = Some(file);
            }
        }
    }
    match ep_file {
        Some(f) => {
            fen.push((b'a' + f) as char);
            fen.push(if stm_black { '3' } else { '6' });
        }
        None => fen.push('-'),
    }
    fen.push_str(&format!(" {rule50} 1"));

    // Moves: undo the transform in the mover's frame, then flip ranks for black.
    let abs = |(from, to, promo): (u8, u8, u8)| -> (u8, u8, u8) {
        if stm_black {
            (from ^ 56, to ^ 56, promo)
        } else {
            (from, to, promo)
        }
    };
    let mut policy = Vec::with_capacity(48);
    for i in 0..1858 {
        let p = f32_at(r, 8 + i * 4);
        if p >= 0.0 {
            let (f, to, pr) = abs(move_from_nn_index(i, t));
            policy.push((f, to, pr, p));
        }
    }
    let played_idx = u16::from_le_bytes(r[8344..8346].try_into().unwrap()) as usize;
    let best_idx = u16::from_le_bytes(r[8346..8348].try_into().unwrap()) as usize;
    if played_idx >= 1858 || best_idx >= 1858 {
        return Err("idx");
    }
    Ok(Decoded {
        input_format,
        transform: t,
        fen,
        stm_black,
        policy,
        best: abs(move_from_nn_index(best_idx, t)),
        played: abs(move_from_nn_index(played_idx, t)),
        best_q: f32_at(r, 8284),
        result_q: f32_at(r, 8308),
        plies_left: f32_at(r, 8304),
    })
}

/// Map a decoded (from, to, promo) onto one of the engine's legal moves. Lc0 writes castling as king-to-rook
/// (e1h1) in the canonical formats and as king-two-squares (e1g1) in the legacy ones; knight promotions have no
/// suffix; en passant is a plain from-to.
fn match_move(pos: &Position, legal: &HashMap<(u8, u8, u8), Move>, ksq: Square, m: (u8, u8, u8)) -> Option<Move> {
    let (from, to, promo) = m;
    let promo_pt = match promo {
        b'q' => 4u8,
        b'r' => 3,
        b'b' => 2,
        0 => {
            // A pawn reaching the last rank without a suffix is a knight promotion.
            let is_pawn = pos.pieces(PieceType::Pawn) & (1u64 << from) != 0;
            if is_pawn && (to >> 3 == 0 || to >> 3 == 7) {
                1
            } else {
                0
            }
        }
        _ => return None,
    };
    if from == ksq {
        let us = pos.side_to_move();
        let own_rooks = pos.pieces_c(us, PieceType::Rook);
        let mut rook_to = None;
        if own_rooks & (1u64 << to) != 0 {
            rook_to = Some(to);
        } else if (to as i8 - from as i8).abs() == 2 && to >> 3 == from >> 3 {
            // e1g1 / e1c1 form: the rook square is the corner on that side.
            let corner = if to > from { (from & 56) + 7 } else { from & 56 };
            rook_to = Some(corner);
        }
        if let Some(rt) = rook_to {
            let cm = Move::new_flag(from, rt, FLAG_CASTLE);
            if legal.values().any(|&lm| lm == cm) {
                return Some(cm);
            }
        }
    }
    legal.get(&(from, to, promo_pt)).copied()
}

fn legal_map(pos: &Position) -> HashMap<(u8, u8, u8), Move> {
    let list = legal_moves(pos);
    let mut map = HashMap::with_capacity(list.len);
    for sm in &list.moves[..list.len] {
        let m = sm.mv;
        let promo = if m.is_promo() { m.promo_type() as u8 } else { 0 };
        map.insert((m.from(), m.to(), promo), m);
    }
    map
}

#[derive(Default, Debug)]
struct Stats {
    records: u64,
    decode_err: u64,
    fen_err: u64,
    illegal_policy: u64,
    illegal_best: u64,
    illegal_played: u64,
    written: u64,
    mass: f64,
    n_policy: u64,
    by_t: [u64; 8],
    formats: HashMap<u32, u64>,
    fail_by_t: [u64; 8],
}

/// Output record (96 bytes):
///   0  occ u64 | 8 pieces [u8;16] (nibble per occupied square in ascending order: colour*6 + type)
///   24 stm u8 | 25 ep_file u8 (255 = none) | 26 castling u8 (1 K, 2 Q, 4 k, 8 q) | 27 rule50 u8
///   28 best u16 (engine Move) | 30 played u16 | 32 n u8 | 33 pad
///   34 entries [(u16 move, u16 prob*65535)] x 12 = 48 bytes -> 82
///   82 pad u16 | 84 best_q f32 | 88 result_q f32 | 92 plies_left f32 -> 96
fn write_record(out: &mut impl Write, pos: &Position, d: &Decoded, best: Move, played: Move, entries: &[(Move, f32)]) -> std::io::Result<()> {
    let mut buf = [0u8; OUT_RECORD];
    let occ = pos.occupied();
    buf[0..8].copy_from_slice(&occ.to_le_bytes());
    let mut n = 0;
    let mut bits = occ;
    while bits != 0 {
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        let pc = pos.piece_on(sq);
        let code = pc.color() as u8 * 6 + pc.piece_type() as u8;
        if n % 2 == 0 {
            buf[8 + n / 2] |= code;
        } else {
            buf[8 + n / 2] |= code << 4;
        }
        n += 1;
    }
    buf[24] = d.stm_black as u8;
    buf[25] = match d.fen.split(' ').nth(3) {
        Some("-") | None => 255,
        Some(s) => s.as_bytes()[0] - b'a',
    };
    let cs = d.fen.split(' ').nth(2).unwrap_or("-");
    buf[26] = (cs.contains('K') as u8) | (cs.contains('Q') as u8) << 1 | (cs.contains('k') as u8) << 2 | (cs.contains('q') as u8) << 3;
    buf[27] = d.fen.split(' ').nth(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    buf[28..30].copy_from_slice(&best.0.to_le_bytes());
    buf[30..32].copy_from_slice(&played.0.to_le_bytes());
    buf[32] = entries.len() as u8;
    for (i, (m, p)) in entries.iter().enumerate() {
        let o = 34 + i * 4;
        buf[o..o + 2].copy_from_slice(&m.0.to_le_bytes());
        buf[o + 2..o + 4].copy_from_slice(&((p.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes());
    }
    buf[84..88].copy_from_slice(&d.best_q.to_le_bytes());
    buf[88..92].copy_from_slice(&d.result_q.to_le_bytes());
    buf[92..96].copy_from_slice(&d.plies_left.to_le_bytes());
    out.write_all(&buf)
}

fn process_record(r: &[u8], st: &mut Stats, out: &mut Option<BufWriter<std::fs::File>>) {
    st.records += 1;
    let d = match decode(r) {
        Ok(d) => d,
        Err(_) => {
            st.decode_err += 1;
            return;
        }
    };
    let pos = match Position::from_fen(&d.fen) {
        Ok(p) => p,
        Err(_) => {
            st.fen_err += 1;
            return;
        }
    };
    let legal = legal_map(&pos);
    let ksq = pos.king_sq(pos.side_to_move());
    let mut entries: Vec<(Move, f32)> = Vec::with_capacity(d.policy.len());
    let mut ok = true;
    let mut mass = 0.0f64;
    for &(f, t, pr, p) in &d.policy {
        match match_move(&pos, &legal, ksq, (f, t, pr)) {
            Some(m) => {
                entries.push((m, p));
                mass += p as f64;
            }
            None => {
                if st.illegal_policy < 6 {
                    eprintln!("illegal policy move {}{}{} (p={p:.3}) in {}", sq_str(f), sq_str(t), if pr != 0 { (pr as char).to_string() } else { String::new() }, d.fen);
                }
                ok = false;
                break;
            }
        }
    }
    st.by_t[d.transform as usize] += 1;
    *st.formats.entry(d.input_format).or_default() += 1;
    if !ok {
        st.illegal_policy += 1;
        st.fail_by_t[d.transform as usize] += 1;
        return;
    }
    let best = match match_move(&pos, &legal, ksq, d.best) {
        Some(m) => m,
        None => {
            st.illegal_best += 1;
            return;
        }
    };
    let played = match match_move(&pos, &legal, ksq, d.played) {
        Some(m) => m,
        None => {
            st.illegal_played += 1;
            return;
        }
    };
    st.mass += mass;
    st.n_policy += entries.len() as u64;
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    entries.truncate(MAX_POLICY);
    if let Some(o) = out {
        write_record(o, &pos, &d, best, played, &entries).expect("write");
    }
    st.written += 1;
}

fn for_each_record_in_tar(reader: impl Read, mut f: impl FnMut(&[u8])) {
    let mut archive = tar::Archive::new(reader);
    let mut buf = vec![0u8; RECORD];
    for entry in archive.entries().expect("tar entries") {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
        if !path.ends_with(".gz") {
            continue;
        }
        let mut gz = flate2::read::MultiGzDecoder::new(entry);
        loop {
            match gz.read_exact(&mut buf) {
                Ok(()) => f(&buf),
                Err(_) => break,
            }
        }
    }
}

fn main() {
    engine::init();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("convert") | Some("verify") => {
            let convert = args[1] == "convert";
            let mut out = if convert {
                Some(BufWriter::new(std::fs::File::create(&args[2]).expect("create out")))
            } else {
                None
            };
            let inputs = if convert { &args[3..] } else { &args[2..] };
            let mut st = Stats::default();
            for path in inputs {
                let t = std::time::Instant::now();
                let before = st.records;
                if path == "-" {
                    for_each_record_in_tar(std::io::stdin().lock(), |r| process_record(r, &mut st, &mut out));
                } else {
                    let file = std::fs::File::open(path).expect("open tar");
                    for_each_record_in_tar(file, |r| process_record(r, &mut st, &mut out));
                }
                eprintln!("{path}: {} records in {:.1} s", st.records - before, t.elapsed().as_secs_f64());
            }
            if let Some(o) = out.as_mut() {
                o.flush().expect("flush");
            }
            let pm = if st.written > 0 { st.mass / st.written as f64 } else { 0.0 };
            let np = if st.written > 0 { st.n_policy as f64 / st.written as f64 } else { 0.0 };
            println!("input formats {:?}; records by transform {:?}, illegal-policy failures by transform {:?}", st.formats, st.by_t, st.fail_by_t);
            println!(
                "records {} written {} ({:.3}%) | decode_err {} fen_err {} illegal_policy {} illegal_best {} illegal_played {} | mean policy mass on legal moves {:.4}, mean legal policy entries {:.1}",
                st.records, st.written, 100.0 * st.written as f64 / st.records.max(1) as f64, st.decode_err, st.fen_err, st.illegal_policy, st.illegal_best, st.illegal_played, pm, np
            );
        }
        Some("dump") => {
            let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
            let data = std::fs::read(&args[2]).expect("read");
            for rec in data.chunks_exact(OUT_RECORD).take(n) {
                let occ = u64::from_le_bytes(rec[0..8].try_into().unwrap());
                let mut board = ['.'; 64];
                let mut i = 0;
                let mut bits = occ;
                while bits != 0 {
                    let sq = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let code = if i % 2 == 0 { rec[8 + i / 2] & 15 } else { rec[8 + i / 2] >> 4 };
                    let ch = b"PNBRQKpnbrqk"[code as usize] as char;
                    board[sq] = ch;
                    i += 1;
                }
                let mut s = String::new();
                for rank in (0..8).rev() {
                    for file in 0..8 {
                        s.push(board[rank * 8 + file]);
                    }
                    s.push('/');
                }
                let best = Move(u16::from_le_bytes(rec[28..30].try_into().unwrap()));
                let cnt = rec[32] as usize;
                let mut pol = String::new();
                for k in 0..cnt {
                    let o = 34 + k * 4;
                    let m = Move(u16::from_le_bytes(rec[o..o + 2].try_into().unwrap()));
                    let p = u16::from_le_bytes(rec[o + 2..o + 4].try_into().unwrap()) as f32 / 65535.0;
                    pol.push_str(&format!(" {m}:{p:.3}"));
                }
                println!("{s} stm={} ep={} castle={:04b} r50={} best={best} q={:.3} policy:{pol}", if rec[24] == 1 { 'b' } else { 'w' }, rec[25], rec[26], rec[27], f32_at(rec, 84));
            }
        }
        _ => {
            eprintln!("usage: lc0conv convert <out.bin> <in.tar>|- ... | verify <in.tar>... | dump <out.bin> [n]");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_are_involutions() {
        let v = 0x0123_4567_89AB_CDEFu64;
        assert_eq!(reverse_bits_in_bytes(reverse_bits_in_bytes(v)), v);
        assert_eq!(reverse_bytes_in_bytes(reverse_bytes_in_bytes(v)), v);
        assert_eq!(transpose_bits_in_bytes(transpose_bits_in_bytes(v)), v);
    }

    #[test]
    fn square_transform_matches_bitboard_transform() {
        // The square-level Transform must agree with the bitboard-level transforms for flip and mirror
        // (transpose combines both in lc0's square version, which is what MoveFromNNIndex inverts).
        for sq in 0..64u8 {
            let bb = 1u64 << sq;
            assert_eq!(reverse_bits_in_bytes(bb), 1u64 << transform_sq(sq, FLIP));
            assert_eq!(reverse_bytes_in_bytes(bb), 1u64 << transform_sq(sq, MIRROR));
        }
    }

    #[test]
    fn policy_index_roundtrip_identity() {
        let (f, t, p) = move_from_nn_index(0, 0);
        assert_eq!((f, t, p), (0, 1, 0)); // a1b1
        let (f, t, p) = move_from_nn_index(1857, 0);
        assert_eq!((f, t, p), (55, 63, b'b')); // h7h8b
    }
}
