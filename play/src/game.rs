//! Game state for one human-vs-engine session: rules, history, SAN, and the UCI position command.
//! Legality and game-end detection come from the engine crate's own move generator, so the server
//! can never accept a move the engine would consider illegal.

use engine::movegen::legal_moves;
use engine::position::Position;
use engine::types::{file_of, rank_of, square_to_string, Color, Move, PieceType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Playing,
    Checkmate { winner: Color },
    Stalemate,
    FiftyMoveRule,
    Repetition,
    InsufficientMaterial,
    Resigned { loser: Color },
}

impl Status {
    pub fn is_over(self) -> bool {
        self != Status::Playing
    }
    pub fn name(self) -> &'static str {
        match self {
            Status::Playing => "playing",
            Status::Checkmate { .. } => "checkmate",
            Status::Stalemate => "stalemate",
            Status::FiftyMoveRule => "fifty_move_rule",
            Status::Repetition => "threefold_repetition",
            Status::InsufficientMaterial => "insufficient_material",
            Status::Resigned { .. } => "resigned",
        }
    }
    /// PGN-style result string, `None` while the game is still on.
    pub fn result(self) -> Option<&'static str> {
        match self {
            Status::Playing => None,
            Status::Checkmate { winner: Color::White } => Some("1-0"),
            Status::Checkmate { winner: Color::Black } => Some("0-1"),
            Status::Resigned { loser: Color::White } => Some("0-1"),
            Status::Resigned { loser: Color::Black } => Some("1-0"),
            _ => Some("1/2-1/2"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlayedMove {
    pub mv: Move,
    pub uci: String,
    pub san: String,
}

pub struct Game {
    /// `None` means the standard start position (`position startpos`).
    start_fen: Option<String>,
    /// positions[0] is the start position; positions[i] is after moves[i-1].
    positions: Vec<Position>,
    moves: Vec<PlayedMove>,
    pub human: Color,
    resigned: Option<Color>,
}

impl Game {
    pub fn new(start_fen: Option<&str>, human: Color) -> Result<Game, String> {
        let (pos, start_fen) = match start_fen {
            None => (Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")?, None),
            Some(f) => {
                let p = Position::from_fen(f)?;
                if p.is_chess960() {
                    return Err("Chess960 positions are not supported in the web UI".into());
                }
                (p, Some(f.to_string()))
            }
        };
        Ok(Game { start_fen, positions: vec![pos], moves: Vec::new(), human, resigned: None })
    }

    pub fn start_fen(&self) -> Option<&str> {
        self.start_fen.as_deref()
    }
    pub fn current(&self) -> &Position {
        self.positions.last().expect("positions is never empty")
    }
    pub fn turn(&self) -> Color {
        self.current().side_to_move()
    }
    pub fn engine_color(&self) -> Color {
        self.human.flip()
    }
    pub fn moves(&self) -> &[PlayedMove] {
        &self.moves
    }
    pub fn legal(&self) -> Vec<Move> {
        legal_moves(self.current()).iter().collect()
    }
    pub fn legal_uci(&self) -> Vec<String> {
        let pos = self.current();
        self.legal().into_iter().map(|m| pos.move_to_uci(m)).collect()
    }

    pub fn status(&self) -> Status {
        if let Some(loser) = self.resigned {
            return Status::Resigned { loser };
        }
        let pos = self.current();
        if legal_moves(pos).is_empty() {
            return if pos.in_check() {
                Status::Checkmate { winner: pos.side_to_move().flip() }
            } else {
                Status::Stalemate
            };
        }
        if pos.rule50() >= 100 {
            return Status::FiftyMoveRule;
        }
        let key = pos.key();
        if self.positions.iter().filter(|p| p.key() == key).count() >= 3 {
            return Status::Repetition;
        }
        if pos.is_insufficient_material() {
            return Status::InsufficientMaterial;
        }
        Status::Playing
    }

    /// Parse a UCI move string against the current position; only fully legal moves are accepted.
    pub fn parse_move(&self, uci: &str) -> Option<Move> {
        if uci.len() < 4 || uci.len() > 5 || !uci.is_ascii() {
            return None;
        }
        let pos = self.current();
        let m = pos.parse_uci_move(uci)?;
        if legal_moves(pos).contains(m) {
            Some(m)
        } else {
            None
        }
    }

    pub fn push(&mut self, m: Move) {
        let pos = self.current();
        let uci = pos.move_to_uci(m);
        let san = san(pos, m);
        let next = pos.make_move(m);
        self.positions.push(next);
        self.moves.push(PlayedMove { mv: m, uci, san });
    }

    /// Take back `n` plies (at most the number played).
    pub fn undo(&mut self, n: usize) {
        for _ in 0..n.min(self.moves.len()) {
            self.moves.pop();
            self.positions.pop();
        }
        self.resigned = None;
    }

    pub fn resign(&mut self, c: Color) {
        self.resigned = Some(c);
    }

    /// The UCI `position` command that reproduces the current position in the engine.
    pub fn position_cmd(&self) -> String {
        let mut s = match &self.start_fen {
            None => "position startpos".to_string(),
            Some(f) => format!("position fen {f}"),
        };
        if !self.moves.is_empty() {
            s.push_str(" moves");
            for m in &self.moves {
                s.push(' ');
                s.push_str(&m.uci);
            }
        }
        s
    }

    /// Convert a principal variation (UCI strings from the current position) to SAN, stopping at
    /// the first move that is not legal.
    pub fn pv_to_san(&self, pv: &[String]) -> Vec<String> {
        let mut pos = self.current().clone();
        let mut out = Vec::new();
        for u in pv {
            let Some(m) = pos.parse_uci_move(u) else { break };
            if !legal_moves(&pos).contains(m) {
                break;
            }
            out.push(san(&pos, m));
            pos = pos.make_move(m);
        }
        out
    }
}

/// Standard algebraic notation for a legal move in `pos`.
pub fn san(pos: &Position, m: Move) -> String {
    let mut s = String::new();
    let from = m.from();
    let to = m.to();
    let pt = pos.moved_piece(m).piece_type();
    if m.is_castle() {
        s.push_str(if pos.castle_king_to(m) > from { "O-O" } else { "O-O-O" });
    } else if pt == PieceType::Pawn {
        if pos.is_capture(m) || m.is_ep() {
            s.push((b'a' + file_of(from)) as char);
            s.push('x');
        }
        s.push_str(&square_to_string(to));
        if m.is_promo() {
            s.push('=');
            s.push(m.promo_type().to_char().to_ascii_uppercase());
        }
    } else {
        s.push(pt.to_char().to_ascii_uppercase());
        // Disambiguate against other legal moves of the same piece type to the same square.
        let others: Vec<Move> = legal_moves(pos)
            .iter()
            .filter(|&o| o != m && o.to() == to && pos.moved_piece(o).piece_type() == pt)
            .collect();
        if !others.is_empty() {
            let same_file = others.iter().any(|o| file_of(o.from()) == file_of(from));
            let same_rank = others.iter().any(|o| rank_of(o.from()) == rank_of(from));
            if !same_file {
                s.push((b'a' + file_of(from)) as char);
            } else if !same_rank {
                s.push((b'1' + rank_of(from)) as char);
            } else {
                s.push_str(&square_to_string(from));
            }
        }
        if pos.is_capture(m) {
            s.push('x');
        }
        s.push_str(&square_to_string(to));
    }
    let next = pos.make_move(m);
    if next.in_check() {
        s.push(if legal_moves(&next).is_empty() { '#' } else { '+' });
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(fen: Option<&str>) -> Game {
        engine::init();
        Game::new(fen, Color::White).unwrap()
    }

    #[test]
    fn san_basics() {
        let mut game = g(None);
        for (u, expect) in [("e2e4", "e4"), ("e7e5", "e5"), ("g1f3", "Nf3"), ("b8c6", "Nc6"), ("f1c4", "Bc4"), ("g8f6", "Nf6"), ("e1g1", "O-O")] {
            let m = game.parse_move(u).expect(u);
            game.push(m);
            assert_eq!(game.moves().last().unwrap().san, expect);
        }
        assert_eq!(game.position_cmd(), "position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1");
        assert_eq!(game.status(), Status::Playing);
    }

    #[test]
    fn san_disambiguation_capture_promo_mate() {
        // Two knights can reach d2: Nbd2 vs Nfd2.
        let game = g(Some("r3k2r/8/8/8/8/1N3N2/8/R3K2R w KQkq - 0 1"));
        let m = game.parse_move("b3d2").unwrap();
        assert_eq!(san(game.current(), m), "Nbd2");
        // Promotion with capture and check.
        let game = g(Some("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1"));
        let m = game.parse_move("a7b8q").unwrap();
        assert_eq!(san(game.current(), m), "axb8=Q+");
        // Scholar's mate.
        let game = g(Some("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"));
        let m = game.parse_move("h5f7").unwrap();
        assert_eq!(san(game.current(), m), "Qxf7#");
        let mut game = game;
        game.push(m);
        assert_eq!(game.status(), Status::Checkmate { winner: Color::White });
        assert_eq!(game.status().result(), Some("1-0"));
    }

    #[test]
    fn illegal_moves_rejected_and_undo() {
        let mut game = g(None);
        assert!(game.parse_move("e2e5").is_none());
        assert!(game.parse_move("e7e5").is_none());
        assert!(game.parse_move("zz").is_none());
        assert!(game.parse_move("").is_none());
        let m = game.parse_move("e2e4").unwrap();
        game.push(m);
        assert_eq!(game.turn(), Color::Black);
        game.undo(5);
        assert_eq!(game.moves().len(), 0);
        assert_eq!(game.turn(), Color::White);
        assert_eq!(game.legal_uci().len(), 20);
    }

    #[test]
    fn repetition_and_stalemate() {
        let mut game = g(None);
        for u in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"] {
            let m = game.parse_move(u).unwrap();
            game.push(m);
        }
        assert_eq!(game.status(), Status::Repetition);
        let game = g(Some("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"));
        assert_eq!(game.status(), Status::Stalemate);
        let game = g(Some("8/8/8/8/8/4k3/8/4K2N w - - 0 1"));
        assert_eq!(game.status(), Status::InsufficientMaterial);
    }
}
