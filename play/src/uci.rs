//! One engine subprocess per session, driven over UCI. The engine binary is the same one used for
//! matches; nothing about the search runs in this process.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use serde::Serialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::mpsc;

pub struct EngineProc {
    child: Child,
    stdin: ChildStdin,
    pub lines: mpsc::Receiver<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Info {
    pub depth: Option<u32>,
    pub seldepth: Option<u32>,
    /// centipawns from the side to move's point of view
    pub cp: Option<i32>,
    /// mate in N (negative: getting mated)
    pub mate: Option<i32>,
    pub nodes: Option<u64>,
    pub nps: Option<u64>,
    pub time_ms: Option<u64>,
    pub hashfull: Option<u32>,
    pub tbhits: Option<u64>,
    pub multipv: Option<u32>,
    pub pv: Vec<String>,
}

pub struct EngineOptions<'a> {
    pub hash_mb: u32,
    pub threads: u32,
    pub eval_file: Option<&'a str>,
    pub syzygy_path: Option<&'a str>,
}

impl EngineProc {
    pub async fn spawn(binary: &Path, opts: EngineOptions<'_>) -> Result<EngineProc, String> {
        let mut child = Command::new(binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("cannot start engine {}: {e}", binary.display()))?;
        let stdin = child.stdin.take().ok_or("engine stdin")?;
        let stdout = child.stdout.take().ok_or("engine stdout")?;
        let (tx, rx) = mpsc::channel::<String>(256);
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if tx.send(line).await.is_err() {
                    break;
                }
            }
        });
        let mut e = EngineProc { child, stdin, lines: rx };
        e.send("uci").await?;
        e.wait_for("uciok", Duration::from_secs(20)).await?;
        e.send(&format!("setoption name Hash value {}", opts.hash_mb)).await?;
        e.send(&format!("setoption name Threads value {}", opts.threads)).await?;
        if let Some(f) = opts.eval_file {
            e.send(&format!("setoption name EvalFile value {f}")).await?;
        }
        if let Some(p) = opts.syzygy_path {
            e.send(&format!("setoption name SyzygyPath value {p}")).await?;
        }
        e.send("isready").await?;
        // Loading a 91 MB net and tablebases can take a while on a slow disk.
        e.wait_for("readyok", Duration::from_secs(120)).await?;
        e.send("ucinewgame").await?;
        e.send("isready").await?;
        e.wait_for("readyok", Duration::from_secs(60)).await?;
        Ok(e)
    }

    pub async fn send(&mut self, cmd: &str) -> Result<(), String> {
        self.stdin
            .write_all(format!("{cmd}\n").as_bytes())
            .await
            .map_err(|e| format!("engine stdin closed: {e}"))?;
        self.stdin.flush().await.map_err(|e| format!("engine stdin flush: {e}"))
    }

    /// Discard output until a line starting with `token` arrives.
    pub async fn wait_for(&mut self, token: &str, timeout: Duration) -> Result<(), String> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let line = tokio::time::timeout_at(deadline, self.lines.recv())
                .await
                .map_err(|_| format!("engine did not answer '{token}' within {timeout:?}"))?
                .ok_or_else(|| "engine exited".to_string())?;
            if line.starts_with(token) {
                return Ok(());
            }
        }
    }

    pub async fn quit(&mut self) {
        let _ = self.send("quit").await;
        let _ = tokio::time::timeout(Duration::from_secs(2), self.child.wait()).await;
        let _ = self.child.start_kill();
    }
}

/// Parse a UCI `info` line. Lines without a score or pv (e.g. `info string ...`) return `None`.
pub fn parse_info(line: &str) -> Option<Info> {
    let mut info = Info::default();
    let toks: Vec<&str> = line.split_whitespace().collect();
    if toks.first() != Some(&"info") || toks.contains(&"string") {
        return None;
    }
    let mut i = 1;
    let mut seen_score_or_pv = false;
    while i < toks.len() {
        let num = |j: usize| toks.get(j).and_then(|t| t.parse::<i64>().ok());
        match toks[i] {
            "depth" => info.depth = num(i + 1).map(|v| v as u32),
            "seldepth" => info.seldepth = num(i + 1).map(|v| v as u32),
            "multipv" => info.multipv = num(i + 1).map(|v| v as u32),
            "nodes" => info.nodes = num(i + 1).map(|v| v as u64),
            "nps" => info.nps = num(i + 1).map(|v| v as u64),
            "time" => info.time_ms = num(i + 1).map(|v| v as u64),
            "hashfull" => info.hashfull = num(i + 1).map(|v| v as u32),
            "tbhits" => info.tbhits = num(i + 1).map(|v| v as u64),
            "score" => {
                seen_score_or_pv = true;
                match toks.get(i + 1) {
                    Some(&"cp") => info.cp = num(i + 2).map(|v| v as i32),
                    Some(&"mate") => info.mate = num(i + 2).map(|v| v as i32),
                    _ => {}
                }
                i += 2;
            }
            "pv" => {
                seen_score_or_pv = true;
                info.pv = toks[i + 1..].iter().map(|s| s.to_string()).collect();
                break;
            }
            _ => {}
        }
        i += 1;
    }
    if seen_score_or_pv {
        Some(info)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_info_lines() {
        let i = parse_info("info depth 12 seldepth 20 multipv 1 score cp 35 nodes 123456 nps 900000 hashfull 12 tbhits 0 time 137 pv e2e4 e7e5 g1f3").unwrap();
        assert_eq!(i.depth, Some(12));
        assert_eq!(i.cp, Some(35));
        assert_eq!(i.mate, None);
        assert_eq!(i.nps, Some(900000));
        assert_eq!(i.pv, vec!["e2e4", "e7e5", "g1f3"]);
        let m = parse_info("info depth 5 score mate -3 lowerbound nodes 10 pv a1a2").unwrap();
        assert_eq!(m.mate, Some(-3));
        assert_eq!(m.cp, None);
        assert!(parse_info("info string hello").is_none());
        assert!(parse_info("info depth 3 currmove e2e4 currmovenumber 1").is_none());
        assert!(parse_info("bestmove e2e4").is_none());
    }
}
