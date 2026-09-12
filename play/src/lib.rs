//! Play against Anna over a WebSocket. One HTTP port serves the browser page (`/`), the WebSocket
//! (`/ws`) and a health check (`/health`), so a single Cloudflare tunnel exposes everything.
//!
//! Protocol (JSON text frames, one object per frame):
//!
//! client -> server
//!   {"t":"new", "color":"white"|"black"|"random", "movetime":1000, "depth":null, "fen":null}
//!   {"t":"move", "uci":"e2e4"}          human move (promotion: "e7e8q")
//!   {"t":"undo"}                         take back to the human's previous turn
//!   {"t":"resign"}
//!   {"t":"ping"}
//!
//! server -> client
//!   {"t":"hello", "engine":..., "net":..., "max_movetime":..., "max_depth":..., "threads":...}
//!   {"t":"state", "fen", "turn":"w"|"b", "human":"w"|"b", "legal":[uci...],
//!      "history":[{"uci","san"}...], "status":"playing"|..., "result":"1-0"|null,
//!      "thinking":bool, "last":uci|null, "movetime":ms, "depth":n|null}
//!   {"t":"info", "depth", "seldepth", "cp_white", "mate_white", "nodes", "nps", "time_ms", "pv":[uci], "pv_san":[san]}
//!   {"t":"error", "msg":"..."}
//!   {"t":"pong"}

pub mod game;
pub mod uci;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::Router;
use engine::types::Color;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Semaphore;

use game::Game;
use uci::{EngineOptions, EngineProc};

pub const INDEX_HTML: &str = include_str!("../static/index.html");

#[derive(Clone, Debug)]
pub struct Config {
    pub engine_bin: PathBuf,
    pub eval_file: Option<String>,
    pub syzygy_path: Option<String>,
    pub hash_mb: u32,
    pub threads: u32,
    pub max_movetime_ms: u64,
    pub max_depth: u32,
    pub max_games: usize,
    pub engine_name: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            engine_bin: PathBuf::from("target/release/engine"),
            eval_file: None,
            syzygy_path: None,
            hash_mb: 64,
            threads: 1,
            max_movetime_ms: 10_000,
            max_depth: 30,
            max_games: 4,
            engine_name: "Anna".to_string(),
        }
    }
}

struct AppState {
    cfg: Config,
    slots: Semaphore,
}

pub fn router(cfg: Config) -> Router {
    let state = Arc::new(AppState { slots: Semaphore::new(cfg.max_games), cfg });
    Router::new()
        .route("/", get(index))
        .route("/health", get(|| async { "ok" }))
        .route("/ws", get(ws_upgrade))
        .with_state(state)
}

/// Bind and serve until the future is dropped or the process exits.
pub async fn serve(addr: SocketAddr, cfg: Config) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("play: listening on http://{} (engine {}, net {})", listener.local_addr()?, cfg.engine_bin.display(),
        cfg.eval_file.as_deref().unwrap_or("embedded default"));
    axum::serve(listener, router(cfg)).await
}

async fn index() -> impl IntoResponse {
    Html(INDEX_HTML)
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(st): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| session(socket, st))
}

#[derive(Deserialize)]
#[serde(tag = "t")]
enum ClientMsg {
    #[serde(rename = "new")]
    New { color: Option<String>, movetime: Option<u64>, depth: Option<u32>, fen: Option<String> },
    #[serde(rename = "move")]
    Move { uci: String },
    #[serde(rename = "undo")]
    Undo,
    #[serde(rename = "resign")]
    Resign,
    #[serde(rename = "ping")]
    Ping,
}

struct Limits {
    movetime_ms: u64,
    depth: Option<u32>,
}

async fn send_json(ws: &mut WebSocket, v: serde_json::Value) -> bool {
    ws.send(Message::Text(v.to_string().into())).await.is_ok()
}

fn color_char(c: Color) -> &'static str {
    if c == Color::White { "w" } else { "b" }
}

fn state_json(game: &Game, thinking: bool, limits: &Limits) -> serde_json::Value {
    let status = game.status();
    json!({
        "t": "state",
        "fen": game.current().to_fen(),
        "turn": color_char(game.turn()),
        "human": color_char(game.human),
        "legal": if thinking || status.is_over() { Vec::new() } else { game.legal_uci() },
        "history": game.moves().iter().map(|m| json!({"uci": m.uci, "san": m.san})).collect::<Vec<_>>(),
        "status": status.name(),
        "result": status.result(),
        "thinking": thinking,
        "last": game.moves().last().map(|m| m.uci.clone()),
        "movetime": limits.movetime_ms,
        "depth": limits.depth,
    })
}

async fn session(mut ws: WebSocket, st: Arc<AppState>) {
    let Ok(_permit) = st.slots.try_acquire() else {
        send_json(&mut ws, json!({"t":"error","msg":"server is full, try again later"})).await;
        return;
    };
    let cfg = &st.cfg;
    let mut eng = match EngineProc::spawn(
        &cfg.engine_bin,
        EngineOptions {
            hash_mb: cfg.hash_mb,
            threads: cfg.threads,
            eval_file: cfg.eval_file.as_deref(),
            syzygy_path: cfg.syzygy_path.as_deref(),
        },
    )
    .await
    {
        Ok(e) => e,
        Err(e) => {
            eprintln!("play: {e}");
            send_json(&mut ws, json!({"t":"error","msg": format!("engine failed to start: {e}")})).await;
            return;
        }
    };
    let net_name = cfg
        .eval_file
        .as_deref()
        .map(|p| p.rsplit('/').next().unwrap_or(p).to_string())
        .unwrap_or_else(|| "embedded default".to_string());
    if !send_json(
        &mut ws,
        json!({"t":"hello","engine":cfg.engine_name,"net":net_name,"max_movetime":cfg.max_movetime_ms,
               "max_depth":cfg.max_depth,"threads":cfg.threads}),
    )
    .await
    {
        eng.quit().await;
        return;
    }

    let mut game = Game::new(None, Color::White).expect("start position");
    let mut limits = Limits { movetime_ms: 1000, depth: None };
    let mut thinking = false;
    // A `stop` was sent and the next bestmove must be ignored (undo / new game / resign mid-search).
    let mut discard_bestmove = false;
    let mut client_alive = true;

    // Start a search if it is the engine's turn. Requires no search in flight.
    macro_rules! maybe_go {
        () => {
            if !thinking && !game.status().is_over() && game.turn() == game.engine_color() {
                let go = match limits.depth {
                    Some(d) => format!("go depth {d}"),
                    None => format!("go movetime {}", limits.movetime_ms),
                };
                if eng.send(&game.position_cmd()).await.is_err() || eng.send(&go).await.is_err() {
                    send_json(&mut ws, json!({"t":"error","msg":"engine died"})).await;
                    break;
                }
                thinking = true;
            }
        };
    }
    macro_rules! push_state {
        () => {
            if !send_json(&mut ws, state_json(&game, thinking, &limits)).await {
                client_alive = false;
                break;
            }
        };
    }
    // Interrupt a running search; its bestmove will be discarded.
    macro_rules! interrupt {
        () => {
            if thinking {
                let _ = eng.send("stop").await;
                discard_bestmove = true;
                thinking = false;
            }
        };
    }

    if !send_json(&mut ws, state_json(&game, thinking, &limits)).await {
        eng.quit().await;
        return;
    }
    let idle_timeout = Duration::from_secs(30 * 60);
    loop {
        tokio::select! {
            msg = ws.recv() => {
                let Some(Ok(msg)) = msg else { client_alive = false; break };
                let text = match msg {
                    Message::Text(t) => t.to_string(),
                    Message::Close(_) => { client_alive = false; break }
                    _ => continue,
                };
                let parsed: Result<ClientMsg, _> = serde_json::from_str(&text);
                let Ok(cm) = parsed else {
                    if !send_json(&mut ws, json!({"t":"error","msg":"bad message"})).await { break }
                    continue;
                };
                match cm {
                    ClientMsg::Ping => { if !send_json(&mut ws, json!({"t":"pong"})).await { break } }
                    ClientMsg::New { color, movetime, depth, fen } => {
                        let human = match color.as_deref() {
                            Some("black") => Color::Black,
                            Some("random") => if rand::random::<bool>() { Color::White } else { Color::Black },
                            _ => Color::White,
                        };
                        match Game::new(fen.as_deref().filter(|f| !f.trim().is_empty()), human) {
                            Ok(g) => {
                                interrupt!();
                                game = g;
                                limits.movetime_ms = movetime.unwrap_or(1000).clamp(50, cfg.max_movetime_ms);
                                limits.depth = depth.map(|d| d.clamp(1, cfg.max_depth));
                                let _ = eng.send("ucinewgame").await;
                                if !discard_bestmove { maybe_go!(); }
                                push_state!();
                            }
                            Err(e) => { if !send_json(&mut ws, json!({"t":"error","msg":e})).await { break } }
                        }
                    }
                    ClientMsg::Move { uci } => {
                        if thinking || game.status().is_over() || game.turn() != game.human {
                            if !send_json(&mut ws, json!({"t":"error","msg":"not your turn"})).await { break }
                            continue;
                        }
                        match game.parse_move(&uci) {
                            Some(m) => {
                                game.push(m);
                                if !discard_bestmove { maybe_go!(); }
                                push_state!();
                            }
                            None => { if !send_json(&mut ws, json!({"t":"error","msg":format!("illegal move {uci}")})).await { break } }
                        }
                    }
                    ClientMsg::Undo => {
                        // Back to the human's previous turn: one ply if the human just moved, else two.
                        let n = if game.turn() == game.engine_color() { 1 } else { 2 };
                        interrupt!();
                        game.undo(n);
                        push_state!();
                    }
                    ClientMsg::Resign => {
                        if !game.status().is_over() {
                            interrupt!();
                            game.resign(game.human);
                            push_state!();
                        }
                    }
                }
            }
            line = eng.lines.recv() => {
                let Some(line) = line else {
                    send_json(&mut ws, json!({"t":"error","msg":"engine exited"})).await;
                    break;
                };
                if let Some(rest) = line.strip_prefix("bestmove") {
                    if discard_bestmove {
                        discard_bestmove = false;
                        maybe_go!();
                        if thinking { push_state!(); }
                        continue;
                    }
                    if !thinking { continue; }
                    thinking = false;
                    let mv = rest.split_whitespace().next().unwrap_or("");
                    match game.parse_move(mv) {
                        Some(m) => game.push(m),
                        None => {
                            // "(none)" when the engine has no legal move (should be caught by status) or a bug.
                            if !send_json(&mut ws, json!({"t":"error","msg":format!("engine returned unusable move '{mv}'")})).await { break }
                        }
                    }
                    push_state!();
                } else if thinking && !discard_bestmove {
                    if let Some(info) = uci::parse_info(&line) {
                        let sign = if game.turn() == Color::White { 1 } else { -1 };
                        let pv_san = game.pv_to_san(&info.pv);
                        let v = json!({
                            "t":"info", "depth":info.depth, "seldepth":info.seldepth,
                            "cp_white": info.cp.map(|c| c * sign), "mate_white": info.mate.map(|m| m * sign),
                            "nodes":info.nodes, "nps":info.nps, "time_ms":info.time_ms,
                            "pv":info.pv, "pv_san":pv_san,
                        });
                        if !send_json(&mut ws, v).await { client_alive = false; break }
                    }
                }
            }
            _ = tokio::time::sleep(idle_timeout) => {
                send_json(&mut ws, json!({"t":"error","msg":"idle timeout, reconnect to play"})).await;
                break;
            }
        }
    }
    let _ = client_alive;
    eng.quit().await;
}
