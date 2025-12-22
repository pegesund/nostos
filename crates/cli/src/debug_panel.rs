//! Debug panel for the TUI - shows call stack, locals, and debug controls.

use cursive::event::{Event, EventResult, Key};
use cursive::theme::{Color, ColorStyle};
use cursive::view::{View, CannotFocus};
use cursive::direction::Direction;
use cursive::{Printer, Vec2};
use nostos_vm::shared_types::{DebugEvent, StackFrame, Breakpoint};
use nostos_vm::DebugSession;
use std::collections::HashSet;

/// Debug panel state
#[derive(Clone, Debug, PartialEq)]
pub enum DebugState {
    /// No active debug session
    Idle,
    /// Paused at a breakpoint or step
    Paused {
        function: String,
        file: Option<String>,
        line: usize,
    },
    /// Running (not paused)
    Running,
    /// Execution finished
    Finished {
        result: Option<String>,
    },
}

/// The debug panel
pub struct DebugPanel {
    /// Current debug state
    pub state: DebugState,
    /// Call stack frames
    pub stack: Vec<StackFrame>,
    /// Local variables: (name, value, type)
    pub locals: Vec<(String, String, String)>,
    /// Selected stack frame index
    selected_frame: usize,
    /// Breakpoints set by user
    pub breakpoints: HashSet<String>,
    /// Scroll offset for locals
    locals_scroll: usize,
    /// Height for calculations
    visible_rows: usize,
}

impl DebugPanel {
    pub fn new() -> Self {
        Self {
            state: DebugState::Idle,
            stack: Vec::new(),
            locals: Vec::new(),
            selected_frame: 0,
            breakpoints: HashSet::new(),
            locals_scroll: 0,
            visible_rows: 10,
        }
    }

    /// Add a breakpoint for a function
    pub fn add_breakpoint(&mut self, function: String) {
        self.breakpoints.insert(function);
    }

    /// Remove a breakpoint
    pub fn remove_breakpoint(&mut self, function: &str) {
        self.breakpoints.remove(function);
    }

    /// Check if we have any breakpoints set
    pub fn has_breakpoints(&self) -> bool {
        !self.breakpoints.is_empty()
    }

    /// Clear all debug state
    pub fn clear(&mut self) {
        self.state = DebugState::Idle;
        self.stack.clear();
        self.locals.clear();
        self.selected_frame = 0;
        self.locals_scroll = 0;
    }

    /// Update state when paused
    pub fn on_paused(&mut self, function: String, file: Option<String>, line: usize) {
        self.state = DebugState::Paused { function, file, line };
    }

    /// Update state when running
    pub fn on_running(&mut self) {
        self.state = DebugState::Running;
    }

    /// Update state when finished
    pub fn on_finished(&mut self, result: Option<String>) {
        self.state = DebugState::Finished { result };
    }

    /// Update call stack
    pub fn set_stack(&mut self, frames: Vec<StackFrame>) {
        self.stack = frames;
        self.selected_frame = 0;
    }

    /// Update locals
    pub fn set_locals(&mut self, locals: Vec<(String, String, String)>) {
        self.locals = locals;
        self.locals_scroll = 0;
    }

    /// Select previous stack frame
    fn select_prev_frame(&mut self) {
        if self.selected_frame > 0 {
            self.selected_frame -= 1;
        }
    }

    /// Select next stack frame
    fn select_next_frame(&mut self) {
        if self.selected_frame < self.stack.len().saturating_sub(1) {
            self.selected_frame += 1;
        }
    }

    /// Scroll locals up
    fn scroll_locals_up(&mut self) {
        if self.locals_scroll > 0 {
            self.locals_scroll -= 1;
        }
    }

    /// Scroll locals down
    fn scroll_locals_down(&mut self) {
        if self.locals_scroll < self.locals.len().saturating_sub(self.visible_rows) {
            self.locals_scroll += 1;
        }
    }

    /// Get status text for display
    fn status_text(&self) -> String {
        match &self.state {
            DebugState::Idle => {
                if self.breakpoints.is_empty() {
                    "No breakpoints set. Use :debug <function> to add.".to_string()
                } else {
                    format!("{} breakpoint(s) set. Run code to debug.", self.breakpoints.len())
                }
            }
            DebugState::Paused { function, line, .. } => {
                format!("▶ Paused in {} at line {}", function, line)
            }
            DebugState::Running => "Running...".to_string(),
            DebugState::Finished { result } => {
                if let Some(val) = result {
                    format!("✓ Finished: {}", val)
                } else {
                    "✓ Finished".to_string()
                }
            }
        }
    }
}

impl View for DebugPanel {
    fn draw(&self, printer: &Printer) {
        let width = printer.size.x;
        let height = printer.size.y;

        // Colors
        let color_title = ColorStyle::new(Color::Rgb(255, 200, 100), Color::TerminalDefault);
        let color_status = ColorStyle::new(Color::Rgb(100, 255, 150), Color::TerminalDefault);
        let color_label = ColorStyle::new(Color::Rgb(150, 150, 150), Color::TerminalDefault);
        let color_value = ColorStyle::new(Color::Rgb(220, 220, 220), Color::TerminalDefault);
        let color_selected = ColorStyle::new(Color::Rgb(0, 0, 0), Color::Rgb(255, 255, 0));
        let color_key = ColorStyle::new(Color::Rgb(255, 200, 100), Color::TerminalDefault);
        let color_dim = ColorStyle::new(Color::Rgb(100, 100, 100), Color::TerminalDefault);

        let mut y = 0;

        // Status line
        printer.with_color(color_status, |p| {
            p.print((1, y), &self.status_text());
        });
        y += 2;

        // If idle, show breakpoints
        if matches!(self.state, DebugState::Idle) {
            if !self.breakpoints.is_empty() {
                printer.with_color(color_title, |p| {
                    p.print((1, y), "Breakpoints:");
                });
                y += 1;
                for bp in &self.breakpoints {
                    printer.with_color(color_value, |p| {
                        p.print((3, y), &format!("• {}", bp));
                    });
                    y += 1;
                }
            }

            // Help text
            y = height.saturating_sub(2);
            printer.with_color(color_dim, |p| {
                p.print((1, y), ":debug <func> to add breakpoint");
            });
            return;
        }

        // If finished, just show result
        if matches!(self.state, DebugState::Finished { .. }) {
            y = height.saturating_sub(2);
            printer.with_color(color_dim, |p| {
                p.print((1, y), "Press any key to dismiss");
            });
            return;
        }

        // If running, just show status
        if matches!(self.state, DebugState::Running) {
            return;
        }

        // === Paused state - show call stack and locals ===

        // Call Stack section
        printer.with_color(color_title, |p| {
            p.print((1, y), "Call Stack:");
        });
        y += 1;

        if self.stack.is_empty() {
            printer.with_color(color_dim, |p| {
                p.print((3, y), "(no stack available)");
            });
            y += 1;
        } else {
            let stack_height = 5.min(self.stack.len());
            for (i, frame) in self.stack.iter().take(stack_height).enumerate() {
                let is_selected = i == self.selected_frame && printer.focused;
                let style = if is_selected { color_selected } else { color_value };
                let prefix = if is_selected { "→ " } else { "  " };

                let line_info = if frame.line > 0 {
                    format!(" (line {})", frame.line)
                } else {
                    String::new()
                };

                printer.with_color(style, |p| {
                    let text = format!("{}{}(){}", prefix, frame.function, line_info);
                    let display = if text.len() > width - 2 {
                        format!("{}...", &text[..width.saturating_sub(5)])
                    } else {
                        text
                    };
                    p.print((1, y), &display);
                });
                y += 1;
            }
            if self.stack.len() > stack_height {
                printer.with_color(color_dim, |p| {
                    p.print((3, y), &format!("... {} more", self.stack.len() - stack_height));
                });
                y += 1;
            }
        }

        y += 1;

        // Locals section
        printer.with_color(color_title, |p| {
            p.print((1, y), "Locals:");
        });
        y += 1;

        if self.locals.is_empty() {
            printer.with_color(color_dim, |p| {
                p.print((3, y), "(no locals)");
            });
            y += 1;
        } else {
            let locals_height = (height - y - 3).min(self.locals.len());
            for (name, value, _type_name) in self.locals.iter()
                .skip(self.locals_scroll)
                .take(locals_height)
            {
                let name_display = if name.len() > 15 {
                    format!("{}...", &name[..12])
                } else {
                    name.clone()
                };

                printer.with_color(color_label, |p| {
                    p.print((3, y), &format!("{} = ", name_display));
                });

                let value_x = 3 + name_display.len() + 3;
                let max_value_len = width.saturating_sub(value_x + 1);
                let value_display = if value.len() > max_value_len {
                    format!("{}...", &value[..max_value_len.saturating_sub(3)])
                } else {
                    value.clone()
                };

                printer.with_color(color_value, |p| {
                    p.print((value_x, y), &value_display);
                });
                y += 1;
            }
        }

        // Key bindings at bottom
        y = height.saturating_sub(2);
        for i in 0..width {
            printer.print((i, y), "─");
        }
        y += 1;

        printer.with_color(color_key, |p| {
            let keys = "[F5/c] Continue  [F10/n] Over  [F11/s] In  [Shift+F11/o] Out  [q] Stop";
            let display = if keys.len() > width - 2 {
                &keys[..width.saturating_sub(2)]
            } else {
                keys
            };
            p.print((1, y), display);
        });
    }

    fn required_size(&mut self, constraint: Vec2) -> Vec2 {
        // Use full available size
        constraint
    }

    fn take_focus(&mut self, _: Direction) -> Result<EventResult, CannotFocus> {
        Ok(EventResult::Consumed(None))
    }

    fn on_event(&mut self, event: Event) -> EventResult {
        match event {
            // Let Tab propagate for window cycling
            Event::Key(Key::Tab) | Event::Shift(Key::Tab) => {
                EventResult::Ignored
            }
            // Stack navigation
            Event::Key(Key::Up) => {
                self.select_prev_frame();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Down) => {
                self.select_next_frame();
                EventResult::Consumed(None)
            }
            // Locals scrolling
            Event::Key(Key::PageUp) => {
                self.scroll_locals_up();
                EventResult::Consumed(None)
            }
            Event::Key(Key::PageDown) => {
                self.scroll_locals_down();
                EventResult::Consumed(None)
            }
            // Debug commands are handled by the TUI layer
            // since they need access to the debug session
            Event::Key(Key::F5) | Event::Char('c') | Event::Char('C') |
            Event::Key(Key::F10) | Event::Char('n') | Event::Char('N') |
            Event::Key(Key::F11) | Event::Char('s') | Event::Char('S') |
            Event::Char('o') | Event::Char('O') |
            Event::Char('q') | Event::Char('Q') => {
                // These will be handled at the TUI level
                EventResult::Ignored
            }
            _ => EventResult::Ignored,
        }
    }
}
