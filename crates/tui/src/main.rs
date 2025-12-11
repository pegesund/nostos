use cursive::Cursive;
use cursive::traits::*;
use cursive::views::{Dialog, EditView, LinearLayout, ScrollView, TextView};
use cursive::theme::{Color, PaletteColor, Theme, BorderStyle};
use cursive::view::Resizable;
use nostos_repl::{ReplEngine, ReplConfig};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let mut siv = cursive::default();

    // Custom theme for "matrix" look or similar
    let mut theme = Theme::default();
    theme.borders = BorderStyle::Simple;
    theme.palette[PaletteColor::Background] = Color::TerminalDefault;
    theme.palette[PaletteColor::View] = Color::TerminalDefault;
    theme.palette[PaletteColor::Primary] = Color::Rgb(255, 255, 255); // White
    theme.palette[PaletteColor::TitlePrimary] = Color::Rgb(0, 255, 0); // Green
    theme.palette[PaletteColor::Highlight] = Color::Rgb(0, 255, 0); // Green
    siv.set_theme(theme);

    // Initialize REPL engine
    let config = ReplConfig::default();
    let mut engine = ReplEngine::new(config);
    if let Err(e) = engine.load_stdlib() {
        eprintln!("Failed to load stdlib: {}", e);
    }
    let engine = Rc::new(RefCell::new(engine));

    // Output view
    let output_view = TextView::new(format!("Nostos TUI v{}\nType :help for commands\n\n", env!("CARGO_PKG_VERSION")))
        .scrollable()
        .with_name("output_scroll");

    // Input view
    let engine_clone = engine.clone();
    let input_view = EditView::new()
        .on_submit(move |s, text| {
            let text = text.trim();
            if text.is_empty() { return; }

            let input_text = text.to_string();

            // Clear input
            s.call_on_name("input", |view: &mut EditView| {
                view.set_content("");
            });

            // Check for exit
            if input_text == ":quit" || input_text == ":q" {
                s.quit();
                return;
            }

            // Run eval
            let result = match engine_clone.borrow_mut().eval(&input_text) {
                Ok(output) => output,
                Err(e) => format!("Error: {}", e),
            };

            // Update output
            s.call_on_name("output_scroll", |view: &mut ScrollView<TextView>| {
                let text_view = view.get_inner_mut();
                text_view.append(format!("nos> {}\n", input_text));
                
                if !result.is_empty() {
                    text_view.append(format!("{}\n", result));
                }
                text_view.append("\n");
                
                view.scroll_to_bottom();
            });
        })
        .with_name("input");

    // Layout
    let layout = LinearLayout::vertical()
        .child(output_view.full_height()) // Expand to fill
        .child(Dialog::around(input_view).title("Input"));

    siv.add_layer(Dialog::around(layout).title("Nostos REPL"));

    siv.run();
}
