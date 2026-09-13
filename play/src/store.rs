//! Persistent record of every game played through the server (SQLite, one file).
//! Every move is written the moment it is made, so a game abandoned mid-way is still on disk.
//! Engine moves carry the search output that produced them (depth, eval, nodes, pv).

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};

use crate::uci::Info;

pub struct Store {
    conn: Mutex<Connection>,
}

pub struct NewGame<'a> {
    pub human: &'a str,        // "w" | "b"
    pub start_fen: Option<&'a str>,
    pub movetime_ms: u64,
    pub depth: Option<u32>,
    pub engine_name: &'a str,
    pub net: &'a str,
    pub threads: u32,
    pub client: &'a str,       // remote address as seen by the server (may be a proxy)
}

impl Store {
    pub fn open(path: &Path) -> rusqlite::Result<Store> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS games (
                 id INTEGER PRIMARY KEY,
                 started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                 ended_at TEXT,
                 human TEXT NOT NULL,
                 start_fen TEXT,
                 movetime_ms INTEGER NOT NULL,
                 depth INTEGER,
                 engine TEXT NOT NULL,
                 net TEXT NOT NULL,
                 threads INTEGER NOT NULL,
                 client TEXT,
                 status TEXT NOT NULL DEFAULT 'playing',
                 result TEXT,
                 plies INTEGER NOT NULL DEFAULT 0,
                 undos INTEGER NOT NULL DEFAULT 0,
                 pgn TEXT
             );
             CREATE TABLE IF NOT EXISTS moves (
                 game_id INTEGER NOT NULL REFERENCES games(id),
                 ply INTEGER NOT NULL,
                 uci TEXT NOT NULL,
                 san TEXT NOT NULL,
                 fen_after TEXT NOT NULL,
                 by_engine INTEGER NOT NULL,
                 played_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                 depth INTEGER, seldepth INTEGER, cp_white INTEGER, mate_white INTEGER,
                 nodes INTEGER, nps INTEGER, time_ms INTEGER, pv TEXT,
                 PRIMARY KEY (game_id, ply)
             );
             CREATE INDEX IF NOT EXISTS games_started ON games(started_at);",
        )?;
        Ok(Store { conn: Mutex::new(conn) })
    }

    pub fn new_game(&self, g: &NewGame) -> rusqlite::Result<i64> {
        let c = self.conn.lock().unwrap();
        c.execute(
            "INSERT INTO games (human, start_fen, movetime_ms, depth, engine, net, threads, client) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![g.human, g.start_fen, g.movetime_ms as i64, g.depth.map(|d| d as i64), g.engine_name, g.net, g.threads as i64, g.client],
        )?;
        Ok(c.last_insert_rowid())
    }

    /// `info` is the search output that produced an engine move (white's point of view), `None` for human moves.
    pub fn add_move(&self, game_id: i64, ply: usize, uci: &str, san: &str, fen_after: &str, by_engine: bool, info: Option<&Info>, sign: i32) -> rusqlite::Result<()> {
        let c = self.conn.lock().unwrap();
        let (depth, seldepth, cp, mate, nodes, nps, time_ms, pv) = match info {
            Some(i) => (
                i.depth.map(|v| v as i64), i.seldepth.map(|v| v as i64), i.cp.map(|v| (v * sign) as i64), i.mate.map(|v| (v * sign) as i64),
                i.nodes.map(|v| v as i64), i.nps.map(|v| v as i64), i.time_ms.map(|v| v as i64), Some(i.pv.join(" ")),
            ),
            None => (None, None, None, None, None, None, None, None),
        };
        c.execute(
            "INSERT OR REPLACE INTO moves (game_id, ply, uci, san, fen_after, by_engine, depth, seldepth, cp_white, mate_white, nodes, nps, time_ms, pv)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
            params![game_id, ply as i64, uci, san, fen_after, by_engine as i64, depth, seldepth, cp, mate, nodes, nps, time_ms, pv],
        )?;
        c.execute("UPDATE games SET plies = ?2 WHERE id = ?1", params![game_id, ply as i64 + 1])?;
        Ok(())
    }

    /// Remove a game row that never got a move (nothing to keep).
    pub fn delete_if_empty(&self, game_id: i64) -> rusqlite::Result<()> {
        let c = self.conn.lock().unwrap();
        c.execute("DELETE FROM games WHERE id = ?1 AND NOT EXISTS (SELECT 1 FROM moves WHERE game_id = ?1)", params![game_id])?;
        Ok(())
    }

    /// Undo: drop moves from `keep` onwards and count it.
    pub fn truncate(&self, game_id: i64, keep: usize) -> rusqlite::Result<()> {
        let c = self.conn.lock().unwrap();
        c.execute("DELETE FROM moves WHERE game_id = ?1 AND ply >= ?2", params![game_id, keep as i64])?;
        c.execute("UPDATE games SET plies = ?2, undos = undos + 1, status = 'playing', result = NULL, ended_at = NULL WHERE id = ?1", params![game_id, keep as i64])?;
        Ok(())
    }

    pub fn set_level(&self, game_id: i64, movetime_ms: u64, depth: Option<u32>) -> rusqlite::Result<()> {
        let c = self.conn.lock().unwrap();
        c.execute("UPDATE games SET movetime_ms = ?2, depth = ?3 WHERE id = ?1", params![game_id, movetime_ms as i64, depth.map(|d| d as i64)])?;
        Ok(())
    }

    /// Final status ("checkmate", "resigned", "abandoned", ...) and PGN result ("1-0", "*", ...).
    /// The PGN is built from the stored moves and kept with the game.
    pub fn finish(&self, game_id: i64, status: &str, result: &str) -> rusqlite::Result<()> {
        let c = self.conn.lock().unwrap();
        let pgn = Self::build_from_rows(&c, game_id, Some(result))?.unwrap_or_default();
        c.execute(
            "UPDATE games SET status = ?2, result = ?3, pgn = ?4, ended_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?1",
            params![game_id, status, result, pgn],
        )?;
        Ok(())
    }

    fn build_from_rows(c: &Connection, game_id: i64, result_override: Option<&str>) -> rusqlite::Result<Option<String>> {
        let row: Option<(String, String, Option<String>, Option<String>, String, String)> = c
            .query_row(
                "SELECT started_at, human, start_fen, result, net, engine FROM games WHERE id = ?1",
                params![game_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
            )
            .ok();
        let Some((started, human, start_fen, result, net, engine)) = row else { return Ok(None) };
        let mut st = c.prepare("SELECT san FROM moves WHERE game_id = ?1 ORDER BY ply")?;
        let sans: Vec<String> = st.query_map(params![game_id], |r| r.get(0))?.collect::<Result<_, _>>()?;
        let result = result_override.unwrap_or(result.as_deref().unwrap_or("*"));
        Ok(Some(build_pgn(&started, &human, start_fen.as_deref(), result, &format!("{engine} ({net})"), &sans)))
    }

    /// Recent games, newest first, as JSON-ready rows.
    pub fn recent(&self, limit: usize) -> rusqlite::Result<Vec<serde_json::Value>> {
        let c = self.conn.lock().unwrap();
        let mut st = c.prepare(
            "SELECT id, started_at, ended_at, human, movetime_ms, depth, net, status, result, plies, undos FROM games ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = st.query_map(params![limit as i64], |r| {
            Ok(serde_json::json!({
                "id": r.get::<_, i64>(0)?, "started_at": r.get::<_, String>(1)?, "ended_at": r.get::<_, Option<String>>(2)?,
                "human": r.get::<_, String>(3)?, "movetime_ms": r.get::<_, i64>(4)?, "depth": r.get::<_, Option<i64>>(5)?,
                "net": r.get::<_, String>(6)?, "status": r.get::<_, String>(7)?, "result": r.get::<_, Option<String>>(8)?,
                "plies": r.get::<_, i64>(9)?, "undos": r.get::<_, i64>(10)?,
            }))
        })?;
        rows.collect()
    }

    /// PGN of a game (stored at the end; rebuilt from the moves table for games still in progress).
    pub fn pgn(&self, game_id: i64) -> rusqlite::Result<Option<String>> {
        let c = self.conn.lock().unwrap();
        let stored: Option<Option<String>> = c.query_row("SELECT pgn FROM games WHERE id = ?1", params![game_id], |r| r.get(0)).ok();
        match stored {
            None => Ok(None),
            Some(Some(p)) => Ok(Some(p)),
            Some(None) => Self::build_from_rows(&c, game_id, None),
        }
    }
}

pub fn build_pgn(started: &str, human: &str, start_fen: Option<&str>, result: &str, engine: &str, sans: &[String]) -> String {
    let (white, black) = if human == "w" { ("Human", engine) } else { (engine, "Human") };
    let date = started.get(0..10).unwrap_or("????.??.??").replace('-', ".");
    let mut s = format!(
        "[Event \"Anna web game\"]\n[Site \"chess.lironkatsif.com\"]\n[Date \"{date}\"]\n[Round \"-\"]\n[White \"{white}\"]\n[Black \"{black}\"]\n[Result \"{result}\"]\n"
    );
    if let Some(f) = start_fen {
        s.push_str(&format!("[SetUp \"1\"]\n[FEN \"{f}\"]\n"));
    }
    s.push('\n');
    let mut line = String::new();
    for (i, san) in sans.iter().enumerate() {
        let tok = if i % 2 == 0 { format!("{}. {san}", i / 2 + 1) } else { san.clone() };
        if line.len() + tok.len() + 1 > 80 {
            s.push_str(line.trim_end());
            s.push('\n');
            line.clear();
        }
        line.push_str(&tok);
        line.push(' ');
    }
    s.push_str(&line);
    s.push_str(result);
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_rebuilds() {
        let dir = std::env::temp_dir().join(format!("anna-play-test-{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        let st = Store::open(&dir).unwrap();
        let id = st
            .new_game(&NewGame { human: "w", start_fen: None, movetime_ms: 500, depth: None, engine_name: "Anna", net: "v1", threads: 1, client: "test" })
            .unwrap();
        st.add_move(id, 0, "e2e4", "e4", "fen1", false, None, 1).unwrap();
        let info = Info { depth: Some(10), cp: Some(-30), pv: vec!["e7e5".into()], ..Default::default() };
        st.add_move(id, 1, "e7e5", "e5", "fen2", true, Some(&info), -1).unwrap();
        st.add_move(id, 2, "g1f3", "Nf3", "fen3", false, None, 1).unwrap();
        st.truncate(id, 2).unwrap();
        let r = st.recent(10).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0]["plies"], 2);
        assert_eq!(r[0]["undos"], 1);
        assert_eq!(r[0]["status"], "playing");
        let pgn = st.pgn(id).unwrap().unwrap();
        assert!(pgn.contains("1. e4 e5 *"), "{pgn}");
        assert!(pgn.contains("[White \"Human\"]"));
        st.finish(id, "resigned", "0-1").unwrap();
        assert!(st.pgn(id).unwrap().unwrap().contains("1. e4 e5 0-1"));
        assert_eq!(st.recent(10).unwrap()[0]["result"], "0-1");
        // engine move eval stored from white's point of view: cp -30 for black to move * sign(-1) = +30
        let c = st.conn.lock().unwrap();
        let cp: i64 = c.query_row("SELECT cp_white FROM moves WHERE game_id=?1 AND ply=1", params![id], |r| r.get(0)).unwrap();
        assert_eq!(cp, 30);
        drop(c);
        std::fs::remove_file(&dir).unwrap();
    }

    #[test]
    fn pgn_wraps_and_sets_fen() {
        let sans: Vec<String> = (0..60).map(|i| if i % 2 == 0 { "Nf3".to_string() } else { "Nf6".to_string() }).collect();
        let p = build_pgn("2026-09-12T20:00:00Z", "b", Some("8/8/8/8/8/8/8/K6k w - - 0 1"), "1/2-1/2", "Anna (v3)", &sans);
        assert!(p.contains("[Date \"2026.09.12\"]"));
        assert!(p.contains("[Black \"Human\"]"));
        assert!(p.contains("[FEN \"8/8/8/8/8/8/8/K6k w - - 0 1\"]"));
        assert!(p.lines().all(|l| l.len() <= 81), "{p}");
        assert!(p.trim_end().ends_with("1/2-1/2"));
    }
}
