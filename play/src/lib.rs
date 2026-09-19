//! Play against Anna over a WebSocket. One HTTP port serves the browser page (`/`), the WebSocket
//! (`/ws`) and a health check (`/health`), so a single Cloudflare tunnel exposes everything.
//!
//! Protocol (JSON text frames, one object per frame):
//!
//! client -> server
//!   {"t":"new", "color":"white"|"black"|"random", "movetime":1000, "depth":null, "fen":null}
//!   {"t":"move", "uci":"e2e4"}          human move (promotion: "e7e8q")
//!   {"t":"level", "movetime":1000, "depth":null}   change the engine limits for the following moves
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
pub mod store;
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
use store::{NewGame, Store};
use uci::{EngineOptions, EngineProc, Info};

pub const INDEX_HTML: &str = include_str!("../static/index.html");

/// cburnett piece set (Wikimedia Commons, CC BY-SA 3.0), served at /pieces/<piece><l|d>.svg
const PIECES: [(&str, &[u8]); 12] = [
    ("kl.svg", include_bytes!("../static/pieces/kl.svg")), ("kd.svg", include_bytes!("../static/pieces/kd.svg")),
    ("ql.svg", include_bytes!("../static/pieces/ql.svg")), ("qd.svg", include_bytes!("../static/pieces/qd.svg")),
    ("rl.svg", include_bytes!("../static/pieces/rl.svg")), ("rd.svg", include_bytes!("../static/pieces/rd.svg")),
    ("bl.svg", include_bytes!("../static/pieces/bl.svg")), ("bd.svg", include_bytes!("../static/pieces/bd.svg")),
    ("nl.svg", include_bytes!("../static/pieces/nl.svg")), ("nd.svg", include_bytes!("../static/pieces/nd.svg")),
    ("pl.svg", include_bytes!("../static/pieces/pl.svg")), ("pd.svg", include_bytes!("../static/pieces/pd.svg")),
];

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
    /// SQLite file recording every game and move; `None` disables recording.
    pub db_path: Option<PathBuf>,
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
            db_path: Some(PathBuf::from("play/games.sqlite")),
        }
    }
}

struct AppState {
    cfg: Config,
    slots: Semaphore,
    store: Option<Store>,
}

pub fn router(cfg: Config) -> Router {
    let store = match &cfg.db_path {
        Some(p) => match Store::open(p) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("play: cannot open game database {}: {e} (games will NOT be recorded)", p.display());
                None
            }
        },
        None => None,
    };
    let state = Arc::new(AppState { slots: Semaphore::new(cfg.max_games), store, cfg });
    Router::new()
        .route("/", get(index))
        .route("/health", get(|| async { "ok" }))
        .route("/games", get(games_list))
        .route("/games/{id}/pgn", get(game_pgn))
        .route("/pieces/{name}", get(piece_svg))
        .route("/ws", get(ws_upgrade))
        .route("/taunt", axum::routing::post(taunt))
        .with_state(state)
}


// ---------------------------------------------------------------------------------------------
// Table talk: the page asks for a line, we ask a hosted model for one. The API key is read once
// from ~/.config/anna-play/groq_key (or GROQ_API_KEY) and never leaves the server. If anything
// fails, or the line trips the content filter, we answer 204 and the page uses its own phrase bank.
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct TauntReq {
    /// "clean" or "spicy"
    mode: String,
    /// Short description of the position and what just happened, built by the page.
    context: String,
    /// The last few lines already shown, so the model does not repeat itself.
    #[serde(default)]
    recent: Vec<String>,
}

fn groq_key() -> Option<String> {
    if let Ok(k) = std::env::var("GROQ_API_KEY") {
        if !k.trim().is_empty() {
            return Some(k.trim().to_string());
        }
    }
    let p = std::env::var("HOME").ok()? + "/.config/anna-play/groq_key";
    let k = std::fs::read_to_string(p).ok()?;
    let k = k.trim().to_string();
    if k.is_empty() { None } else { Some(k) }
}

/// Words that must never appear on a page anyone can open: slurs, sexual content, violence.
/// Ordinary profanity (the point of the feature) is deliberately not on this list.
const BANNED: &[&str] = &[
    "nigg", "fag", "retard", "tranny", "kike", "spic", "chink", "wetback", "coon", "dyke", "paki",
    "rape", "molest", "pedo", "cunt", "whore", "slut", "jerk off", "jerking off", "fap", "cum", "dick",
    "kill yourself", "kys", "hang yourself", "suicide", "die in a", "i'll find you", "your mother",
    "ur mom", "your mom", "your family",
];

/// Short words that are only a problem as whole words (appearance jabs; "fat" must not match "fatal").
const BANNED_WORDS: &[&str] = &["fat", "ugly", "bald", "smelly", "virgin", "incel"];

fn clean_line(s: &str) -> Option<String> {
    let mut line = s.trim().replace(['\n', '\r'], " ");
    line = line.trim_matches(|c| c == '"' || c == '\'' || c == '`').trim().to_string();
    if line.is_empty() || line.chars().count() > 140 {
        return None;
    }
    let low = line.to_lowercase();
    if BANNED.iter().any(|b| low.contains(b)) {
        return None;
    }
    if low
        .split(|c: char| !c.is_ascii_alphabetic())
        .any(|w| BANNED_WORDS.contains(&w))
    {
        return None;
    }
    Some(line)
}

async fn taunt(axum::extract::Json(req): axum::extract::Json<TauntReq>) -> axum::response::Response {
    let Some(key) = groq_key() else {
        return axum::http::StatusCode::NO_CONTENT.into_response();
    };
    let spicy = req.mode == "spicy";
    let sys = if spicy {
        "You are Anna, a chess engine. You shit-talk the human playing you, like a toxic 13-year-old in a \
         Call of Duty lobby. Keep it STUPID SIMPLE and blunt. At most 8 words. Lowercase. It must read like real chat, \
         something a person would actually type; never garbled or half a sentence. \
         Never open with 'wow' and never use 'what a' or a limp rhetorical question; go straight at them. \
         No clever jokes, no metaphors, no wordplay, no full sentences with commas, no quotes, no emoji. \
         Swear in almost every line: fuck, fucking, shit, ass, pussy, trash, garbage, dogshit, bot, L. \
         This is exactly the register wanted: 'hey pussy' / 'holy garbage move' / 'so fuckn ass' / \
         'lmao you're trash' / 'quit bro' / 'that's ass' / 'you're dogshit at this' / 'uninstall'. \
         Write one line like those, fitted to what just happened. Do not copy an example word for word. \
         Only mock their chess and their skill. No slurs, no sexual content, no threats, nothing about \
         race, sex, religion, family or appearance."
    } else {
        "You are Anna, a chess engine playing a human on your own website. You are cocky and dry, and you \
         tease the human about the position. At most 10 words, plain and simple. Keep it completely clean: \
         no profanity at all, no quotes, no emoji, no stage directions. Be specific about what just \
         happened and vary your wording every time."
    };
    let mut ctx: String = req.context.chars().take(400).collect();
    if !req.recent.is_empty() {
        let recent: Vec<String> = req.recent.iter().rev().take(6).map(|l| l.chars().take(60).collect()).collect();
        ctx.push_str(" Do not repeat or rephrase any of these lines you already used: ");
        ctx.push_str(&recent.join(" / "));
    }
    let body = serde_json::json!({
        "model": "qwen/qwen3.8-27b",
        "temperature": 1.2,
        "max_tokens": 32,
        "messages": if spicy {
            serde_json::json!([
                {"role": "system", "content": sys},
                {"role": "user", "content": "A new game just started. Taunt them."},
                {"role": "assistant", "content": "hey pussy"},
                {"role": "user", "content": "The human blundered a piece. Taunt them."},
                {"role": "assistant", "content": "holy garbage move"},
                {"role": "user", "content": "Anna is winning by 6 pawns. Taunt them."},
                {"role": "assistant", "content": "you're so fuckn ass"},
                {"role": "user", "content": "Anna has mate in two. Taunt them."},
                {"role": "assistant", "content": "ur cooked kid"},
                {"role": "user", "content": "Anna took their queen. Taunt them."},
                {"role": "assistant", "content": "queen gone lmao trash"},
                {"role": "user", "content": ctx}
            ])
        } else {
            serde_json::json!([
                {"role": "system", "content": sys},
                {"role": "user", "content": ctx}
            ])
        }
    });
    let client = match reqwest::Client::builder().timeout(Duration::from_secs(6)).build() {
        Ok(c) => c,
        Err(_) => return axum::http::StatusCode::NO_CONTENT.into_response(),
    };
    let mut best: Option<String> = None;
    for attempt in 0..2 {
        let _ = attempt;
    let resp = client
        .post("https://api.groq.com/openai/v1/chat/completions")
        .bearer_auth(key.clone())
        .json(&body)
        .send()
        .await;
    let Ok(resp) = resp else { break };
    let Ok(v) = resp.json::<serde_json::Value>().await else { break };
    let text = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    if let Some(line) = clean_line(text) {
        if !spicy || has_swear(&line) {
            return axum::Json(serde_json::json!({ "line": line })).into_response();
        }
        best = Some(line);
    }
    }
    match best {
        Some(line) => axum::Json(serde_json::json!({ "line": line })).into_response(),
        None => axum::http::StatusCode::NO_CONTENT.into_response(),
    }
}

/// Swearing mode has to actually swear; a polite line is a failed generation.
fn has_swear(s: &str) -> bool {
    const SWEARS: &[&str] = &["fuck", "shit", "ass", "pussy", "bitch", "damn", "crap", "dogwater", "garbage", "trash", "bot", "suck"];
    let low = s.to_lowercase();
    SWEARS.iter().any(|w| low.contains(w))
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

async fn piece_svg(axum::extract::Path(name): axum::extract::Path<String>) -> axum::response::Response {
    match PIECES.iter().find(|(n, _)| *n == name) {
        Some((_, bytes)) => (
            [(axum::http::header::CONTENT_TYPE, "image/svg+xml"), (axum::http::header::CACHE_CONTROL, "public, max-age=86400")],
            *bytes,
        )
            .into_response(),
        None => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
}

async fn games_list(State(st): State<Arc<AppState>>) -> axum::response::Response {
    match &st.store {
        Some(s) => match s.recent(200) {
            Ok(rows) => axum::Json(rows).into_response(),
            Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        },
        None => (axum::http::StatusCode::NOT_FOUND, "game recording disabled").into_response(),
    }
}

async fn game_pgn(State(st): State<Arc<AppState>>, axum::extract::Path(id): axum::extract::Path<i64>) -> axum::response::Response {
    match st.store.as_ref().and_then(|s| s.pgn(id).ok().flatten()) {
        Some(p) => ([(axum::http::header::CONTENT_TYPE, "application/x-chess-pgn; charset=utf-8")], p).into_response(),
        None => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(st): State<Arc<AppState>>, headers: axum::http::HeaderMap) -> impl IntoResponse {
    // Behind Cloudflare the real client is in CF-Connecting-IP; otherwise X-Forwarded-For or nothing.
    let client = ["cf-connecting-ip", "x-forwarded-for"]
        .iter()
        .find_map(|h| headers.get(*h).and_then(|v| v.to_str().ok()).map(|v| v.to_string()))
        .unwrap_or_else(|| "direct".to_string());
    ws.on_upgrade(move |socket| session(socket, st, client))
}

#[derive(Deserialize)]
#[serde(tag = "t")]
enum ClientMsg {
    #[serde(rename = "new")]
    New { color: Option<String>, movetime: Option<u64>, depth: Option<u32>, fen: Option<String> },
    #[serde(rename = "move")]
    Move { uci: String },
    #[serde(rename = "level")]
    Level { movetime: Option<u64>, depth: Option<u32> },
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

async fn session(mut ws: WebSocket, st: Arc<AppState>, client: String) {
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
    // Recording: the database row is created on the first move of a game, so empty games leave no trace.
    let mut game_id: Option<i64> = None;
    let mut last_info: Option<Info> = None;
    let store = st.store.as_ref();
    let record_err = |e: rusqlite::Error| eprintln!("play: game record failed: {e}");

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
    // Record a move that was just pushed onto `game` (info = engine search output for engine moves).
    macro_rules! record_move {
        ($by_engine:expr, $info:expr) => {
            if let Some(store) = store {
                if game_id.is_none() {
                    match store.new_game(&NewGame {
                        human: color_char(game.human), start_fen: game.start_fen(), movetime_ms: limits.movetime_ms, depth: limits.depth,
                        engine_name: &cfg.engine_name, net: &net_name, threads: cfg.threads, client: &client,
                    }) {
                        Ok(id) => game_id = Some(id),
                        Err(e) => record_err(e),
                    }
                }
                if let Some(id) = game_id {
                    let ply = game.moves().len() - 1;
                    let m = &game.moves()[ply];
                    // eval sign: the search ran with the mover to move; convert to white's point of view
                    let sign = if game.turn() == Color::White { -1 } else { 1 };
                    if let Err(e) = store.add_move(id, ply, &m.uci, &m.san, &game.current().to_fen(), $by_engine, $info, sign) { record_err(e); }
                    let status = game.status();
                    if status.is_over() {
                        if let Err(e) = store.finish(id, status.name(), status.result().unwrap_or("*")) { record_err(e); }
                    }
                }
            }
        };
    }
    // Close the record of the current game if it has moves and is not finished (new game / disconnect).
    macro_rules! record_abandon {
        () => {
            if let (Some(store), Some(id)) = (store, game_id) {
                if game.moves().is_empty() {
                    if let Err(e) = store.delete_if_empty(id) { record_err(e); }
                } else if !game.status().is_over() {
                    if let Err(e) = store.finish(id, "abandoned", "*") { record_err(e); }
                }
            }
            game_id = None;
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
                                record_abandon!();
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
                                record_move!(false, None);
                                if !discard_bestmove { maybe_go!(); }
                                push_state!();
                            }
                            None => { if !send_json(&mut ws, json!({"t":"error","msg":format!("illegal move {uci}")})).await { break } }
                        }
                    }
                    ClientMsg::Level { movetime, depth } => {
                        if let Some(m) = movetime { limits.movetime_ms = m.clamp(50, cfg.max_movetime_ms); }
                        limits.depth = depth.map(|d| d.clamp(1, cfg.max_depth));
                        if let (Some(store), Some(id)) = (store, game_id) {
                            if let Err(e) = store.set_level(id, limits.movetime_ms, limits.depth) { record_err(e); }
                        }
                        push_state!();
                    }
                    ClientMsg::Undo => {
                        // Back to the human's previous turn: one ply if the human just moved, else two.
                        let n = if game.turn() == game.engine_color() { 1 } else { 2 };
                        interrupt!();
                        game.undo(n);
                        if let (Some(store), Some(id)) = (store, game_id) {
                            if let Err(e) = store.truncate(id, game.moves().len()) { record_err(e); }
                        }
                        push_state!();
                    }
                    ClientMsg::Resign => {
                        if !game.status().is_over() {
                            interrupt!();
                            game.resign(game.human);
                            if let (Some(store), Some(id)) = (store, game_id) {
                                if let Err(e) = store.finish(id, "resigned", game.status().result().unwrap_or("*")) { record_err(e); }
                            }
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
                        Some(m) => {
                            game.push(m);
                            let info = last_info.take();
                            record_move!(true, info.as_ref());
                        }
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
                        if info.multipv.unwrap_or(1) == 1 { last_info = Some(info.clone()); }
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
    record_abandon!();
    let _ = game_id;
    eng.quit().await;
}
