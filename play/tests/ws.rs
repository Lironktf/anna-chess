//! End-to-end: start the server in-process with the real engine binary, play a few moves over the
//! WebSocket, undo, resign, and check the protocol invariants.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

fn engine_bin() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let p = root.join("target/release/engine");
    assert!(p.is_file(), "build the engine first: cargo build --release (missing {})", p.display());
    p.canonicalize().unwrap()
}

async fn start() -> SocketAddr {
    let db = std::env::temp_dir().join(format!("anna-play-ws-test-{}.sqlite", std::process::id()));
    let _ = std::fs::remove_file(&db);
    let cfg = play::Config { engine_bin: engine_bin(), max_games: 2, hash_mb: 16, db_path: Some(db), ..Default::default() };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, play::router(cfg)).await.unwrap() });
    addr
}

async fn connect(addr: SocketAddr) -> Ws {
    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws")).await.unwrap();
    ws
}

async fn send(ws: &mut Ws, v: Value) {
    ws.send(Message::Text(v.to_string().into())).await.unwrap();
}

async fn recv(ws: &mut Ws) -> Value {
    loop {
        let m = tokio::time::timeout(Duration::from_secs(30), ws.next()).await.expect("timeout").unwrap().unwrap();
        if let Message::Text(t) = m {
            return serde_json::from_str(&t).unwrap();
        }
    }
}

/// Read states (skipping info frames) until `cond` holds; fail after 8 messages.
async fn wait_state(ws: &mut Ws, cond: impl Fn(&Value) -> bool) -> Value {
    for _ in 0..8 {
        let v = recv_non_info(ws).await;
        assert_eq!(v["t"], "state", "unexpected message {v}");
        if cond(&v) {
            return v;
        }
    }
    panic!("condition not reached");
}

/// Skip `info` frames and return the next non-info message.
async fn recv_non_info(ws: &mut Ws) -> Value {
    loop {
        let v = recv(ws).await;
        if v["t"] != "info" {
            return v;
        }
    }
}

#[tokio::test]
async fn full_game_flow() {
    let addr = start().await;
    let ok = reqwest_free_get(addr, "/health").await;
    assert!(ok.contains("ok"));
    let page = reqwest_free_get(addr, "/").await;
    assert!(page.contains("<title>Play Anna</title>"));

    let mut ws = connect(addr).await;
    let hello = recv(&mut ws).await;
    assert_eq!(hello["t"], "hello");
    assert_eq!(hello["engine"], "Anna");
    let st = recv(&mut ws).await;
    assert_eq!(st["t"], "state");
    assert_eq!(st["status"], "playing");
    assert_eq!(st["legal"].as_array().unwrap().len(), 20);

    // Ping/pong.
    send(&mut ws, serde_json::json!({"t":"ping"})).await;
    assert_eq!(recv(&mut ws).await["t"], "pong");

    // New game as white with a fast engine, play 1.e4.
    send(&mut ws, serde_json::json!({"t":"new","color":"white","movetime":100})).await;
    let st = recv(&mut ws).await;
    assert_eq!(st["human"], "w");
    assert_eq!(st["movetime"], 100);
    send(&mut ws, serde_json::json!({"t":"move","uci":"e2e4"})).await;
    let st = recv(&mut ws).await;
    assert_eq!(st["t"], "state");
    assert_eq!(st["thinking"], true);
    assert_eq!(st["history"][0]["san"], "e4");
    assert!(st["legal"].as_array().unwrap().is_empty(), "no legal moves offered while engine thinks");
    // The engine answers: some info frames then a state with a black move played.
    let mut saw_info = false;
    let st = loop {
        let v = recv(&mut ws).await;
        if v["t"] == "info" {
            saw_info = true;
            assert!(v["depth"].is_number());
            assert!(v["pv"].is_array());
            continue;
        }
        break v;
    };
    assert!(saw_info, "expected info frames while thinking");
    assert_eq!(st["t"], "state");
    assert_eq!(st["thinking"], false);
    assert_eq!(st["turn"], "w");
    assert_eq!(st["history"].as_array().unwrap().len(), 2);
    let fen = st["fen"].as_str().unwrap().to_string();

    // Illegal move rejected, state unchanged.
    send(&mut ws, serde_json::json!({"t":"move","uci":"e4e6"})).await;
    let err = recv(&mut ws).await;
    assert_eq!(err["t"], "error");
    assert!(err["msg"].as_str().unwrap().contains("illegal"));

    // Undo takes back both plies.
    send(&mut ws, serde_json::json!({"t":"undo"})).await;
    let st = recv_non_info(&mut ws).await;
    assert_eq!(st["history"].as_array().unwrap().len(), 0);
    assert_ne!(st["fen"], fen);
    assert_eq!(st["turn"], "w");

    // Play as black: the engine moves first.
    send(&mut ws, serde_json::json!({"t":"new","color":"black","movetime":100})).await;
    let st = wait_state(&mut ws, |s| s["thinking"] == true).await;
    assert_eq!(st["human"], "b");
    let st = wait_state(&mut ws, |s| s["thinking"] == false).await;
    assert_eq!(st["turn"], "b");
    assert_eq!(st["history"].as_array().unwrap().len(), 1);

    // Move during the engine's think is refused: play a move, then immediately try another.
    let legal = st["legal"].as_array().unwrap()[0].as_str().unwrap().to_string();
    send(&mut ws, serde_json::json!({"t":"move","uci":legal})).await;
    let st = recv_non_info(&mut ws).await;
    assert_eq!(st["thinking"], true);
    send(&mut ws, serde_json::json!({"t":"move","uci":"a7a6"})).await;
    let v = recv_non_info(&mut ws).await;
    assert_eq!(v["t"], "error");
    assert_eq!(v["msg"], "not your turn");
    // Undo mid-search: server stops the engine, discards its move, removes our ply.
    send(&mut ws, serde_json::json!({"t":"undo"})).await;
    let st = recv_non_info(&mut ws).await;
    assert_eq!(st["t"], "state");
    assert_eq!(st["thinking"], false);
    assert_eq!(st["history"].as_array().unwrap().len(), 1);
    assert_eq!(st["turn"], "b");
    // Resign.
    send(&mut ws, serde_json::json!({"t":"resign"})).await;
    let st = recv_non_info(&mut ws).await;
    assert_eq!(st["status"], "resigned");
    assert_eq!(st["result"], "1-0");

    // Fixed depth mode and a custom FEN: mate in one for the engine as white.
    send(&mut ws, serde_json::json!({"t":"new","color":"black","depth":4,
        "fen":"r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"})).await;
    let st = wait_state(&mut ws, |s| s["thinking"] == true).await;
    assert_eq!(st["depth"], 4);
    let st = wait_state(&mut ws, |s| s["thinking"] == false).await;
    assert_eq!(st["status"], "checkmate");
    assert_eq!(st["result"], "1-0");
    assert_eq!(st["history"][0]["san"], "Qxf7#");

    // Bad FEN -> error, no state change.
    send(&mut ws, serde_json::json!({"t":"new","color":"white","fen":"garbage"})).await;
    assert_eq!(recv_non_info(&mut ws).await["t"], "error");
    // Malformed JSON -> error.
    ws.send(Message::Text("{not json".into())).await.unwrap();
    assert_eq!(recv_non_info(&mut ws).await["t"], "error");

    // Recorded games: the mate game and the resigned game. The first game was undone to zero moves
    // before the next "new", so its empty row was deleted rather than kept as "abandoned".
    let games = reqwest_free_get(addr, "/games").await;
    let body = games.split("\r\n\r\n").nth(1).unwrap();
    let rows: Vec<Value> = serde_json::from_str(body).unwrap();
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert_eq!(rows[0]["status"], "checkmate");
    assert_eq!(rows[0]["result"], "1-0");
    assert_eq!(rows[0]["plies"], 1);
    assert_eq!(rows[1]["status"], "resigned");
    assert_eq!(rows[1]["undos"], 1);
    assert_eq!(rows[1]["plies"], 1);
    let id = rows[0]["id"].as_i64().unwrap();
    let pgn = reqwest_free_get(addr, &format!("/games/{id}/pgn")).await;
    assert!(pgn.contains("1. Qxf7# 1-0"), "{pgn}");
    assert!(pgn.contains("[SetUp \"1\"]"));

    // Server capacity: max_games=2, so a third connection is refused.
    let mut ws2 = connect(addr).await;
    assert_eq!(recv(&mut ws2).await["t"], "hello");
    let mut ws3 = connect(addr).await;
    let v = recv(&mut ws3).await;
    assert_eq!(v["t"], "error");
    assert!(v["msg"].as_str().unwrap().contains("full"));
    drop(ws2);
    ws.close(None).await.unwrap();
}

/// Minimal HTTP GET without pulling in an HTTP client crate.
async fn reqwest_free_get(addr: SocketAddr, path: &str) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(format!("GET {path} HTTP/1.0\r\nHost: {addr}\r\n\r\n").as_bytes()).await.unwrap();
    let mut buf = String::new();
    s.read_to_string(&mut buf).await.unwrap();
    assert!(buf.starts_with("HTTP/1.0 200") || buf.starts_with("HTTP/1.1 200"), "{buf}");
    buf
}
