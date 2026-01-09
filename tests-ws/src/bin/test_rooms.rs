// Test for room-based push functionality
// Run with: cargo run --release -p tests-ws --bin test_rooms
// Requires: ./target/release/nostos examples/rweb_external_push.nos

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tokio::main]
async fn main() {
    println!("=== Room-Based Push Test ===");
    println!("Requires: ./target/release/nostos examples/rweb_external_push.nos");
    println!("");

    sleep(Duration::from_millis(500)).await;

    match test_rooms().await {
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

async fn test_rooms() -> Result<(), Box<dyn std::error::Error>> {
    // Connect client 1
    println!("Connecting client 1...");
    let (mut ws1, _) = connect_async("ws://localhost:8080/ws").await?;
    println!("Client 1 connected!");

    // Get initial page for client 1
    let msg = timeout(Duration::from_secs(5), ws1.next())
        .await?
        .ok_or("Connection closed")??;
    let initial: Value = match msg {
        Message::Text(text) => serde_json::from_str(&text)?,
        other => return Err(format!("Unexpected: {:?}", other).into()),
    };
    println!("Client 1 got initial page");

    // Check initial HTML contains writerId
    if let Some(html) = initial["html"].as_str() {
        if html.contains("writerId: 0") {
            println!("WARNING: writerId is 0 (placeholder)");
        } else if html.contains("writerId:") {
            println!("Client 1 has writerId assigned");
        }
        if html.contains("Current Room: (none)") {
            println!("Client 1 initially in no room (correct)");
        }
    }

    // Client 1 joins Room A
    println!("\nClient 1 joining Room A...");
    let join_msg = r#"{"action":"join","params":{"room":"A"}}"#;
    println!("Sending: {}", join_msg);
    ws1.send(Message::Text(join_msg.to_string())).await?;

    // Wait for response
    let response = timeout(Duration::from_secs(5), ws1.next())
        .await?
        .ok_or("Connection closed")??;

    match response {
        Message::Text(text) => {
            println!("Got response: {}", &text[..text.len().min(200)]);
            let msg: Value = serde_json::from_str(&text)?;
            if let Some(html) = msg["html"].as_str() {
                if html.contains("Current Room: A") {
                    println!("SUCCESS: Client 1 joined Room A!");
                } else if html.contains("Current Room: (none)") {
                    println!("FAIL: Room is still (none) after join");
                    println!("HTML snippet: {}", &html[..html.len().min(500)]);
                    return Err("Join did not work - room still (none)".into());
                } else {
                    println!("HTML: {}", &html[..html.len().min(500)]);
                }
            }
        }
        other => println!("Unexpected response: {:?}", other),
    }

    // Connect client 2
    println!("\nConnecting client 2...");
    let (mut ws2, _) = connect_async("ws://localhost:8080/ws").await?;
    println!("Client 2 connected!");

    // Get initial page for client 2
    let msg = timeout(Duration::from_secs(5), ws2.next())
        .await?
        .ok_or("Connection closed")??;
    let _: Value = match msg {
        Message::Text(text) => serde_json::from_str(&text)?,
        other => return Err(format!("Unexpected: {:?}", other).into()),
    };
    println!("Client 2 got initial page");

    // Client 2 joins Room B
    println!("\nClient 2 joining Room B...");
    ws2.send(Message::Text(r#"{"action":"join","params":{"room":"B"}}"#.to_string())).await?;

    let response = timeout(Duration::from_secs(5), ws2.next())
        .await?
        .ok_or("Connection closed")??;

    match response {
        Message::Text(text) => {
            let msg: Value = serde_json::from_str(&text)?;
            if let Some(html) = msg["html"].as_str() {
                if html.contains("Current Room: B") {
                    println!("SUCCESS: Client 2 joined Room B!");
                } else {
                    println!("FAIL: Client 2 room join failed");
                    return Err("Client 2 join failed".into());
                }
            }
        }
        other => println!("Unexpected response: {:?}", other),
    }

    // Wait for background pusher to send room-specific messages
    println!("\nWaiting for room-specific push messages...");

    let start = std::time::Instant::now();
    let timeout_duration = Duration::from_secs(5);
    let mut client1_got_room_a_msg = false;
    let mut client2_got_room_b_msg = false;

    while start.elapsed() < timeout_duration && (!client1_got_room_a_msg || !client2_got_room_b_msg) {
        tokio::select! {
            msg = ws1.next() => {
                if let Some(Ok(Message::Text(text))) = msg {
                    let parsed: Value = serde_json::from_str(&text)?;
                    let msg_type = parsed["type"].as_str().unwrap_or("");

                    if msg_type == "trigger" {
                        ws1.send(Message::Text(r#"{"action":"_external","params":{}}"#.to_string())).await?;
                    } else if let Some(html) = parsed["html"].as_str() {
                        if html.contains("Room A: Message") {
                            println!("Client 1 received Room A message!");
                            client1_got_room_a_msg = true;
                        }
                    }
                }
            }
            msg = ws2.next() => {
                if let Some(Ok(Message::Text(text))) = msg {
                    let parsed: Value = serde_json::from_str(&text)?;
                    let msg_type = parsed["type"].as_str().unwrap_or("");

                    if msg_type == "trigger" {
                        ws2.send(Message::Text(r#"{"action":"_external","params":{}}"#.to_string())).await?;
                    } else if let Some(html) = parsed["html"].as_str() {
                        if html.contains("Room B: Update") {
                            println!("Client 2 received Room B message!");
                            client2_got_room_b_msg = true;
                        }
                    }
                }
            }
        }
    }

    if client1_got_room_a_msg && client2_got_room_b_msg {
        println!("\nBoth clients received their room-specific messages!");
        Ok(())
    } else {
        Err(format!(
            "Timeout: client1_room_a={}, client2_room_b={}",
            client1_got_room_a_msg, client2_got_room_b_msg
        ).into())
    }
}
