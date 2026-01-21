use tower_lsp::{LspService, Server};
use std::io::Write;

mod server;

fn log(msg: &str) {
    eprintln!("{}", msg);
    std::io::stderr().flush().ok();
    // Also write to file for debugging
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/nostos_lsp.log")
    {
        let _ = writeln!(f, "{}", msg);
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    // Clear previous log
    let _ = std::fs::remove_file("/tmp/nostos_lsp.log");

    // Set up panic handler
    std::panic::set_hook(Box::new(|panic_info| {
        log(&format!("LSP PANIC: {}", panic_info));
    }));

    log(&format!("Starting Nostos LSP server... PID={}", std::process::id()));

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    log("Setting up LSP service...");

    let (service, socket) = LspService::new(server::NostosLanguageServer::new);

    log("Calling serve()...");

    Server::new(stdin, stdout, socket).serve(service).await;

    log("!!! serve() RETURNED - exiting !!!");
}
