// WebSocket test client for RWeb
// Run with: cargo run --release -p tests-ws
// Requires server running: ./target/release/nostos /var/tmp/ptest.nos

use futures_util::{SinkExt, StreamExt};
use rand::Rng;
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

// Test 5: Realistic load - 10k users, measure per-click response time
async fn test_realistic_load() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 5: Realistic Load (10000 users, staggered clicks) ===");

    let num_clients = 10000;
    let stagger_window_ms = 2000; // Spread clicks over 2 seconds

    // Spawn all clients concurrently
    let mut handles = vec![];
    for id in 0..num_clients {
        handles.push(tokio::spawn(async move {
            let result: Result<u128, String> = async {
                let mut client = TestClient::connect("ws://localhost:8080/ws", id)
                    .await
                    .map_err(|e| e.to_string())?;

                // Receive initial page
                client.recv_message().await.map_err(|e| e.to_string())?;

                // Wait random time (0 to stagger_window) to spread out clicks
                let delay = rand::thread_rng().gen_range(0..stagger_window_ms);
                sleep(Duration::from_millis(delay)).await;

                // Measure ONE click response time
                let click_start = std::time::Instant::now();
                client.send_action("inc").await.map_err(|e| e.to_string())?;
                client.recv_message().await.map_err(|e| e.to_string())?;
                let response_time = click_start.elapsed().as_millis();

                client.close().await.map_err(|e| e.to_string())?;
                Ok(response_time)
            }.await;
            result
        }));
    }

    println!("  Spawned {} clients, clicks staggered over {}ms", num_clients, stagger_window_ms);

    // Collect response times
    let mut response_times: Vec<u128> = vec![];
    let mut failures = 0;
    for handle in handles {
        match handle.await {
            Ok(Ok(time)) => response_times.push(time),
            Ok(Err(e)) => { println!("  Error: {}", e); failures += 1; }
            Err(e) => { println!("  Task error: {}", e); failures += 1; }
        }
    }

    response_times.sort();

    if !response_times.is_empty() {
        let min = response_times.first().unwrap();
        let max = response_times.last().unwrap();
        let avg: u128 = response_times.iter().sum::<u128>() / response_times.len() as u128;
        let p50 = response_times[response_times.len() / 2];
        let p95 = response_times[response_times.len() * 95 / 100];
        let p99 = response_times[response_times.len() * 99 / 100];

        println!("  Per-click response times:");
        println!("    Min:  {}ms", min);
        println!("    Avg:  {}ms", avg);
        println!("    P50:  {}ms (median)", p50);
        println!("    P95:  {}ms", p95);
        println!("    P99:  {}ms", p99);
        println!("    Max:  {}ms", max);
    }

    println!("  Success: {}, Failures: {}", response_times.len(), failures);
    assert_eq!(failures, 0, "No failures expected");
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

    match test_realistic_load().await {
        Ok(_) => passed += 1,
        Err(e) => { println!("  FAILED: {}", e); failed += 1; }
    }

    println!("\n==============================");
    println!("Results: {} passed, {} failed", passed, failed);

    if failed > 0 {
        std::process::exit(1);
    }
}
