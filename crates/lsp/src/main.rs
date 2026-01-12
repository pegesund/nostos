use tower_lsp::{LspService, Server};

mod server;

#[tokio::main]
async fn main() {
    // Log to stderr so it doesn't interfere with LSP communication on stdout
    eprintln!("Starting Nostos LSP server...");

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(server::NostosLanguageServer::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
