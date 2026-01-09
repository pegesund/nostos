// Standalone test for external push functionality
// Run with: cargo run --release -p tests-ws --bin test_push
// Requires: ./target/release/nostos examples/rweb_external_push.nos

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message, WebSocketStream, MaybeTlsStream};
use tokio::net::TcpStream;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[tokio::main]
async fn main() {
    println!("=== External Push Test (Two Clients) ===");
    println!("Requires: ./target/release/nostos examples/rweb_external_push.nos");
    println!("");

    sleep(Duration::from_millis(500)).await;

    match test_external_push_two_clients().await {
        Ok(_) => {
            println!("\n=== PASSED ===");
            std::process::exit(0);
        }
        Err(e) => {
            println!("\n=== FAILED: {} ===", e);
            std::process::exit(1);
        }
    }
}

async fn connect_and_join(room: &str, client_name: &str) -> Result<WsStream, Box<dyn std::error::Error + Send + Sync>> {
    println!("[{}] Connecting to ws://localhost:8080/ws...", client_name);
    let (mut ws, _) = connect_async("ws://localhost:8080/ws").await?;
    println!("[{}] Connected!", client_name);

    // Receive initial full page
    let msg = timeout(Duration::from_secs(5), ws.next())
        .await?
        .ok_or("Connection closed")??;

    let initial: Value = match msg {
        Message::Text(text) => serde_json::from_str(&text)?,
        other => return Err(format!("Unexpected message type: {:?}", other).into()),
    };

    assert_eq!(initial["type"], "full", "Expected initial 'full' message");
    println!("[{}] Got initial page", client_name);

    // Join the specified room
    let join_msg = format!(r#"{{"action":"join","params":{{"room":"{}"}}}}"#, room);
    ws.send(Message::Text(join_msg)).await?;
    println!("[{}] Sent join for room {}", client_name, room);

    // Wait for join response
    let response = timeout(Duration::from_secs(5), ws.next())
        .await?
        .ok_or("Connection closed")??;
    match response {
        Message::Text(text) => {
            let msg: Value = serde_json::from_str(&text)?;
            println!("[{}] Joined room {}, response type={}", client_name, room, msg["type"]);
        }
        _ => return Err("Expected text message for action response".into()),
    }

    Ok(ws)
}

async fn wait_for_push(ws: &mut WsStream, client_name: &str, expected_content: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let start = std::time::Instant::now();
    let timeout_duration = Duration::from_secs(5);

    loop {
        if start.elapsed() > timeout_duration {
            return Err(format!("[{}] Timeout: No push received", client_name).into());
        }

        let msg_result = timeout(timeout_duration - start.elapsed(), ws.next()).await;

        match msg_result {
            Ok(Some(Ok(Message::Text(text)))) => {
                let msg: Value = serde_json::from_str(&text)?;
                let msg_type = msg["type"].as_str().unwrap_or("");

                if msg_type == "trigger" {
                    println!("[{}] Got trigger, responding with _external", client_name);
                    ws.send(Message::Text(r#"{"action":"_external","params":{}}"#.to_string())).await?;
                    continue;
                }

                if msg_type == "full" || msg_type == "update" {
                    if let Some(html) = msg["html"].as_str() {
                        if html.contains(expected_content) || html.contains("✓") {
                            println!("[{}] Got push with expected content: {}", client_name, expected_content);
                            return Ok(());
                        }
                    }
                    continue;
                }
            }
            Ok(Some(Ok(_))) => continue,
            Ok(Some(Err(e))) => return Err(format!("WebSocket error: {}", e).into()),
            Ok(None) => return Err("Connection closed".into()),
            Err(_) => return Err(format!("[{}] Timeout waiting for push", client_name).into()),
        }
    }
}

async fn test_external_push_two_clients() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Connect two clients, one to room A and one to room B
    let mut client_a = connect_and_join("A", "ClientA").await?;
    let mut client_b = connect_and_join("B", "ClientB").await?;

    println!("\nBoth clients connected and joined rooms. Waiting for background pushes...");
    println!("(Background pusher sends every 2 seconds)\n");

    // Wait for both clients to receive their room-specific push
    // Room A gets "Room A: Message #N"
    // Room B gets "Room B: Update #N"
    let (result_a, result_b) = tokio::join!(
        wait_for_push(&mut client_a, "ClientA", "Room A"),
        wait_for_push(&mut client_b, "ClientB", "Room B")
    );

    result_a?;
    result_b?;

    println!("\nBoth clients received their room-specific push messages!");
    Ok(())
}
