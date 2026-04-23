#![allow(
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::needless_borrow,
    clippy::redundant_closure,
    clippy::unnecessary_map_or,
    clippy::type_complexity,
    clippy::ptr_arg,
    clippy::needless_lifetimes,
    clippy::needless_borrows_for_generic_args,
    clippy::clone_on_copy,
    clippy::collapsible_match,
    clippy::redundant_pattern_matching,
    clippy::manual_strip,
    clippy::match_result_ok,
    clippy::manual_pattern_char_comparison,
    clippy::writeln_empty_string,
    dead_code
)]

use tower_lsp::{LspService, Server};

mod server;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    )
    .target(env_logger::Target::Stderr)
    .init();

    // Set up panic handler
    std::panic::set_hook(Box::new(|panic_info| {
        log::error!("LSP PANIC: {}", panic_info);
    }));

    log::info!("Starting Nostos LSP server... PID={}", std::process::id());

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(server::NostosLanguageServer::new);

    Server::new(stdin, stdout, socket).serve(service).await;
}
