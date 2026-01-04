// WebSocket test client for RWeb
// Run with: cargo run --release -p tests-ws
// Requires server running: ./target/release/nostos /var/tmp/ptest.nos

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};

struct TestClient {
    ws: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    id: usize,
}

impl TestClient {
    async fn connect(url: &str, id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let (ws, _) = connect_async(url).await?;
        Ok(Self { ws, id })
    }

    async fn recv_message(&mut self) -> Result<Value, Box<dyn std::error::Error>> {
        let msg = timeout(Duration::from_secs(5), self.ws.next())
            .await?
            .ok_or("Connection closed")??;

        match msg {
            Message::Text(text) => Ok(serde_json::from_str(&text)?),
            other => Err(format!("Unexpected message type: {:?}", other).into()),
        }
    }

    async fn send_action(&mut self, action: &str) -> Result<(), Box<dyn std::error::Error>> {
        let msg = json!({ "action": action, "params": {} });
        self.ws.send(Message::Text(msg.to_string())).await?;
        Ok(())
    }

    fn extract_count(html: &str) -> Option<i64> {
        // Look for "Count: N" pattern
        if let Some(pos) = html.find("Count: ") {
            let rest = &html[pos + 7..];
            let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
            rest[..end].parse().ok()
        } else {
            None
        }
    }

    async fn close(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.ws.close(None).await?;
        Ok(())
    }
}

// Test 1: Single client basic flow
async fn test_single_client() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 1: Single Client Basic Flow ===");

    let mut client = TestClient::connect("ws://localhost:8080/ws", 1).await?;
    println!("  Connected");

    // Should receive initial full page
    let initial = client.recv_message().await?;
    assert_eq!(initial["type"], "full", "Expected 'full' message type");

    let html = initial["html"].as_str().unwrap();
    let count = TestClient::extract_count(html).expect("Should find count in HTML");
    assert_eq!(count, 0, "Initial count should be 0");
    println!("  Initial count: {}", count);

    // Click increment
    client.send_action("inc").await?;
    println!("  Sent 'inc' action");

    // Should receive update
    let update = client.recv_message().await?;
    let html = update["html"].as_str().unwrap();
    let count = TestClient::extract_count(html).expect("Should find count in update");
    assert_eq!(count, 1, "Count should be 1 after increment");
    println!("  Count after click: {}", count);

    // Click a few more times
    for expected in 2..=5 {
        client.send_action("inc").await?;
        let update = client.recv_message().await?;
        let html = update["html"].as_str().unwrap();
        let count = TestClient::extract_count(html).unwrap();
        assert_eq!(count, expected, "Count should be {}", expected);
    }
    println!("  Count after 5 clicks: 5");

    client.close().await?;
    println!("  PASSED ✓");
    Ok(())
}

// Test 2: Multiple clients have independent state
async fn test_multiple_clients_independent() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 2: Multiple Clients Independent State ===");

    let mut client1 = TestClient::connect("ws://localhost:8080/ws", 1).await?;
    let mut client2 = TestClient::connect("ws://localhost:8080/ws", 2).await?;
    println!("  Two clients connected");

    // Both should receive initial page with count 0
    let init1 = client1.recv_message().await?;
    let init2 = client2.recv_message().await?;

    let count1 = TestClient::extract_count(init1["html"].as_str().unwrap()).unwrap();
    let count2 = TestClient::extract_count(init2["html"].as_str().unwrap()).unwrap();
    assert_eq!(count1, 0);
    assert_eq!(count2, 0);
    println!("  Both clients start at 0");

    // Client 1 clicks 3 times
    for _ in 0..3 {
        client1.send_action("inc").await?;
        client1.recv_message().await?;
    }
    println!("  Client 1 clicked 3 times");

    // Client 2 clicks 1 time
    client2.send_action("inc").await?;
    let update2 = client2.recv_message().await?;
    let count2 = TestClient::extract_count(update2["html"].as_str().unwrap()).unwrap();

    // Client 2 should be at 1, not 4
    assert_eq!(count2, 1, "Client 2 should have independent state");
    println!("  Client 2 count: {} (independent from client 1)", count2);

    // Verify client 1 is at 3
    client1.send_action("inc").await?;
    let update1 = client1.recv_message().await?;
    let count1 = TestClient::extract_count(update1["html"].as_str().unwrap()).unwrap();
    assert_eq!(count1, 4, "Client 1 should be at 4");
    println!("  Client 1 count: {}", count1);

    client1.close().await?;
    client2.close().await?;
    println!("  PASSED ✓");
    Ok(())
}

// Test 3: Reconnect gets fresh state
async fn test_reconnect_fresh_state() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 3: Reconnect Gets Fresh State ===");

    let mut client = TestClient::connect("ws://localhost:8080/ws", 1).await?;
    client.recv_message().await?; // initial

    // Click 5 times
    for _ in 0..5 {
        client.send_action("inc").await?;
        client.recv_message().await?;
    }
    println!("  Clicked 5 times, count is 5");

    // Disconnect
    client.close().await?;
    println!("  Disconnected");

    // Reconnect
    sleep(Duration::from_millis(100)).await;
    let mut client = TestClient::connect("ws://localhost:8080/ws", 1).await?;
    let initial = client.recv_message().await?;
    let count = TestClient::extract_count(initial["html"].as_str().unwrap()).unwrap();

    assert_eq!(count, 0, "Reconnected client should start fresh at 0");
    println!("  Reconnected, count is: {} (fresh state)", count);

    client.close().await?;
    println!("  PASSED ✓");
    Ok(())
}

// Test 4: Rapid actions
async fn test_rapid_actions() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 4: Rapid Actions ===");

    let mut client = TestClient::connect("ws://localhost:8080/ws", 1).await?;
    client.recv_message().await?; // initial
    println!("  Connected");

    // Send 20 rapid clicks
    let clicks = 20;
    for _ in 0..clicks {
        client.send_action("inc").await?;
    }
    println!("  Sent {} rapid clicks", clicks);

    // Receive all updates
    let mut last_count = 0;
    for _ in 0..clicks {
        let update = client.recv_message().await?;
        last_count = TestClient::extract_count(update["html"].as_str().unwrap()).unwrap();
    }

    assert_eq!(last_count, clicks, "Should process all {} clicks", clicks);
    println!("  Final count: {}", last_count);

    client.close().await?;
    println!("  PASSED ✓");
    Ok(())
}

// Test 5: Many concurrent clients with timing
async fn test_many_concurrent_clients() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 5: Many Concurrent Clients (1000 clients) ===");

    let num_clients = 1000;
    let clicks_per_client = 5;

    let start_time = std::time::Instant::now();

    // Spawn all clients concurrently
    let mut handles = vec![];
    for id in 0..num_clients {
        let client_start = std::time::Instant::now();
        handles.push(tokio::spawn(async move {
            let result: Result<(usize, u128), String> = async {
                let connect_start = std::time::Instant::now();
                let mut client = TestClient::connect("ws://localhost:8080/ws", id)
                    .await
                    .map_err(|e| e.to_string())?;
                let connect_time = connect_start.elapsed().as_millis();

                // Receive initial
                let initial = client.recv_message().await.map_err(|e| e.to_string())?;
                let count = TestClient::extract_count(initial["html"].as_str().unwrap()).unwrap();
                if count != 0 {
                    return Err(format!("Client {} should start at 0, got {}", id, count));
                }

                // Click N times
                for _ in 0..clicks_per_client {
                    client.send_action("inc").await.map_err(|e| e.to_string())?;
                    let _update = client.recv_message().await.map_err(|e| e.to_string())?;
                }

                // Verify final count
                client.send_action("inc").await.map_err(|e| e.to_string())?;
                let update = client.recv_message().await.map_err(|e| e.to_string())?;
                let final_count = TestClient::extract_count(update["html"].as_str().unwrap()).unwrap();
                if final_count != (clicks_per_client + 1) as i64 {
                    return Err(format!("Client {} should have {} clicks, got {}", id, clicks_per_client + 1, final_count));
                }

                let total_time = client_start.elapsed().as_millis();
                client.close().await.map_err(|e| e.to_string())?;
                Ok((id, total_time))
            }.await;
            result
        }));
    }

    println!("  Spawned {} concurrent clients", num_clients);

    // Wait for all and collect timing
    let mut success = 0;
    let mut times: Vec<(usize, u128)> = vec![];
    for handle in handles {
        match handle.await {
            Ok(Ok((id, time))) => {
                success += 1;
                times.push((id, time));
            }
            Ok(Err(e)) => println!("  Client error: {}", e),
            Err(e) => println!("  Task error: {}", e),
        }
    }

    let total_time = start_time.elapsed().as_millis();

    // Sort by completion time and show stats
    times.sort_by_key(|(_, t)| *t);

    if !times.is_empty() {
        let min_time = times.first().unwrap().1;
        let max_time = times.last().unwrap().1;
        let avg_time: u128 = times.iter().map(|(_, t)| t).sum::<u128>() / times.len() as u128;

        println!("  Timing stats:");
        println!("    Fastest client: {}ms", min_time);
        println!("    Slowest client: {}ms", max_time);
        println!("    Average: {}ms", avg_time);
        println!("    Total wall time: {}ms", total_time);

        // Show first 5 and last 5 to see if there's slowdown
        println!("  First 5 to complete:");
        for (id, time) in times.iter().take(5) {
            println!("    Client {}: {}ms", id, time);
        }
        println!("  Last 5 to complete:");
        for (id, time) in times.iter().rev().take(5).collect::<Vec<_>>().iter().rev() {
            println!("    Client {}: {}ms", id, time);
        }
    }

    assert_eq!(success, num_clients, "All clients should succeed");
    println!("  All {} clients completed successfully", success);
    println!("  PASSED ✓");
    Ok(())
}

#[tokio::main]
async fn main() {
    println!("WebSocket Test Suite for RWeb");
    println!("==============================");
    println!("Make sure server is running: ./target/release/nostos /var/tmp/ptest.nos");

    sleep(Duration::from_millis(500)).await;

    let mut passed = 0;
    let mut failed = 0;

    match test_single_client().await {
        Ok(_) => passed += 1,
        Err(e) => { println!("  FAILED: {}", e); failed += 1; }
    }

    match test_multiple_clients_independent().await {
        Ok(_) => passed += 1,
        Err(e) => { println!("  FAILED: {}", e); failed += 1; }
    }

    match test_reconnect_fresh_state().await {
        Ok(_) => passed += 1,
        Err(e) => { println!("  FAILED: {}", e); failed += 1; }
    }

    match test_rapid_actions().await {
        Ok(_) => passed += 1,
        Err(e) => { println!("  FAILED: {}", e); failed += 1; }
    }

    match test_many_concurrent_clients().await {
        Ok(_) => passed += 1,
        Err(e) => { println!("  FAILED: {}", e); failed += 1; }
    }

    println!("\n==============================");
    println!("Results: {} passed, {} failed", passed, failed);

    if failed > 0 {
        std::process::exit(1);
    }
}
