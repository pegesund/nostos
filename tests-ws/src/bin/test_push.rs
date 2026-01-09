// Standalone test for external push functionality
// Run with: cargo run --release -p tests-ws --bin test_push
// Requires: ./target/release/nostos examples/rweb_external_push.nos

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tokio::main]
async fn main() {
    println!("=== External Push Test ===");
    println!("Requires: ./target/release/nostos examples/rweb_external_push.nos");
    println!("");

    sleep(Duration::from_millis(500)).await;

    match test_external_push().await {
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

async fn test_external_push() -> Result<(), Box<dyn std::error::Error>> {
    println!("Connecting to ws://localhost:8080/ws...");

    let (mut ws, _) = connect_async("ws://localhost:8080/ws").await?;
    println!("Connected!");

    // Receive initial full page
    println!("Waiting for initial page...");
    let msg = timeout(Duration::from_secs(5), ws.next())
        .await?
        .ok_or("Connection closed")??;

    let initial: Value = match msg {
        Message::Text(text) => serde_json::from_str(&text)?,
        other => return Err(format!("Unexpected message type: {:?}", other).into()),
    };

    assert_eq!(initial["type"], "full", "Expected initial 'full' message");
    println!("Got initial page (type=full)");

    // Send a "join" action to join room A (required to receive push updates)
    println!("Sending 'join' action to join room A...");
    ws.send(Message::Text(r#"{"action":"join","params":{"room":"A"}}"#.to_string())).await?;

    // Wait for the join action response
    println!("Waiting for join response...");
    let response = timeout(Duration::from_secs(5), ws.next())
        .await?
        .ok_or("Connection closed")??;
    match response {
        Message::Text(text) => {
            let msg: Value = serde_json::from_str(&text)?;
            println!("Got action response: type={}", msg["type"]);
        }
        _ => return Err("Expected text message for action response".into()),
    }

    // Now wait for the background pusher's trigger message
    // When we receive "trigger", we need to respond with "_external" action
    // just like the browser JavaScript does
    println!("Waiting for background push trigger (up to 5 seconds)...");

    let start = std::time::Instant::now();
    let timeout_duration = Duration::from_secs(5);

    loop {
        if start.elapsed() > timeout_duration {
            return Err("Timeout: No background push received within 5 seconds".into());
        }

        let msg_result = timeout(timeout_duration - start.elapsed(), ws.next()).await;

        match msg_result {
            Ok(Some(Ok(Message::Text(text)))) => {
                let msg: Value = serde_json::from_str(&text)?;
                let msg_type = msg["type"].as_str().unwrap_or("");
                println!("Received message: type={}", msg_type);

                if msg_type == "trigger" {
                    // Respond like browser JavaScript would
                    println!("Responding with _external action...");
                    ws.send(Message::Text(r#"{"action":"_external","params":{}}"#.to_string())).await?;
                    // Continue waiting for the actual update
                    continue;
                }

                if msg_type == "full" || msg_type == "update" {
                    if let Some(html) = msg["html"].as_str() {
                        // Check if this is the push update (contains push message)
                        if html.contains("Push #") || html.contains("✓") {
                            println!("Push contains expected content!");
                            return Ok(());
                        }
                    }
                    // This might be a response to our _external, continue waiting
                    continue;
                }

                return Err(format!("Unexpected message type: '{}'", msg_type).into());
            }
            Ok(Some(Ok(other))) => {
                return Err(format!("Unexpected message type: {:?}", other).into());
            }
            Ok(Some(Err(e))) => {
                return Err(format!("WebSocket error: {}", e).into());
            }
            Ok(None) => {
                return Err("Connection closed before receiving push".into());
            }
            Err(_) => {
                return Err("Timeout: No background push received within 5 seconds".into());
            }
        }
    }
}
