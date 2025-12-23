//! Debug panel for the TUI - shows call stack, locals, and debug controls.

use cursive::event::{Event, EventResult, Key};
use cursive::theme::{Color, ColorStyle};
use cursive::view::{View, CannotFocus};
use cursive::direction::Direction;
use cursive::{Printer, Vec2};
use nostos_vm::shared_types::StackFrame;
use std::collections::{HashMap, HashSet};

/// Debug logging disabled. Uncomment to enable.
#[allow(unused)]
fn debug_log(_msg: &str) {
    // use std::io::Write;
    // if let Ok(mut f) = std::fs::OpenOptions::new()
    //     .create(true)
    //     .append(true)
    //     .open("/tmp/nostos_debug_panel.log")
    // {
    //     let _ = writeln!(f, "{}", _msg);
    // }
}

/// Commands from debug panel to TUI (set via user_data)
#[derive(Clone, Debug, PartialEq)]
pub enum DebugPanelCommand {
    Continue,
    StepOver,
    StepIn,
    StepOut,
    Stop,
}

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
    /// Local variables per frame: frame_index -> (name, value, type)
    frame_locals: HashMap<usize, Vec<(String, String, String)>>,
    /// Selected stack frame index
    selected_frame: usize,
    /// Breakpoints set by user
    pub breakpoints: HashSet<String>,
    /// Scroll offset for locals
    locals_scroll: usize,
    /// Height for calculations
    visible_rows: usize,
    /// Pending command from key press (polled by TUI)
    pub pending_command: Option<DebugPanelCommand>,
    /// Frame index for which we need to request locals (polled by TUI)
    pending_locals_request: Option<usize>,
    /// Whether we need to request the stack (polled by TUI)
    pending_stack_request: bool,
    /// Source code of the current function
    source_code: Option<String>,
    /// Starting line number of the source in the file
    source_start_line: usize,
    /// Scroll offset for source view
    source_scroll: usize,
}

impl DebugPanel {
    pub fn new() -> Self {
        Self {
            state: DebugState::Idle,
            stack: Vec::new(),
            frame_locals: HashMap::new(),
            selected_frame: 0,
            breakpoints: HashSet::new(),
            locals_scroll: 0,
            visible_rows: 10,
            pending_command: None,
            pending_locals_request: None,
            pending_stack_request: false,
            source_code: None,
            source_start_line: 1,
            source_scroll: 0,
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
        self.frame_locals.clear();
        self.selected_frame = 0;
        self.locals_scroll = 0;
        self.pending_locals_request = None;
        self.pending_stack_request = false;
        self.source_code = None;
        self.source_start_line = 1;
        self.source_scroll = 0;
    }

    /// Update state when paused
    pub fn on_paused(&mut self, function: String, file: Option<String>, line: usize, source: Option<String>, source_start_line: usize) {
        debug_log(&format!("on_paused: function={}, file={:?}, line={}, source={:?}, source_start_line={}", function, file, line, source, source_start_line));
        self.state = DebugState::Paused { function, file, line };
        self.source_code = source;
        self.source_start_line = source_start_line;
        self.source_scroll = 0;
        // Clear cached locals (values may have changed after stepping)
        self.frame_locals.clear();
        self.selected_frame = 0;
        self.locals_scroll = 0;
        // Request stack and locals for frame 0
        self.pending_stack_request = true;
        self.pending_locals_request = Some(0);
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
        self.frame_locals.clear();
        self.locals_scroll = 0;
        // Update source from frame 0
        self.update_source_from_selected_frame();
        // Request locals for the current frame
        self.pending_locals_request = Some(0);
    }

    /// Update locals for the current frame (legacy, uses frame 0)
    pub fn set_locals(&mut self, locals: Vec<(String, String, String)>) {
        self.frame_locals.insert(0, locals);
        self.locals_scroll = 0;
    }

    /// Update locals for a specific frame
    pub fn set_locals_for_frame(&mut self, frame_index: usize, locals: Vec<(String, String, String)>) {
        self.frame_locals.insert(frame_index, locals);
        // Clear the pending request if it was for this frame
        if self.pending_locals_request == Some(frame_index) {
            self.pending_locals_request = None;
        }
        // Reset scroll if this is the currently selected frame
        if self.selected_frame == frame_index {
            self.locals_scroll = 0;
        }
    }

    /// Get pending locals request (polled by TUI)
    pub fn take_pending_locals_request(&mut self) -> Option<usize> {
        self.pending_locals_request.take()
    }

    /// Get pending stack request (polled by TUI)
    pub fn take_pending_stack_request(&mut self) -> bool {
        let result = self.pending_stack_request;
        self.pending_stack_request = false;
        result
    }

    /// Get locals for the currently selected frame
    fn current_locals(&self) -> &[(String, String, String)] {
        self.frame_locals.get(&self.selected_frame)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Take pending command (if any) - used by TUI to poll for commands
    pub fn take_pending_command(&mut self) -> Option<DebugPanelCommand> {
        self.pending_command.take()
    }

    /// Take debug panel state for preservation across rebuilds
    pub fn take_state(&mut self) -> (DebugState, Vec<StackFrame>, HashMap<usize, Vec<(String, String, String)>>) {
        let state = std::mem::replace(&mut self.state, DebugState::Idle);
        let stack = std::mem::take(&mut self.stack);
        let frame_locals = std::mem::take(&mut self.frame_locals);
        (state, stack, frame_locals)
    }

    /// Restore debug panel state after rebuild
    pub fn restore_state(&mut self, state: DebugState, stack: Vec<StackFrame>, frame_locals: HashMap<usize, Vec<(String, String, String)>>) {
        self.state = state;
        self.stack = stack;
        self.frame_locals = frame_locals;
    }

    /// Update source display from currently selected frame
    fn update_source_from_selected_frame(&mut self) {
        if let Some(frame) = self.stack.get(self.selected_frame) {
            self.source_code = frame.source.clone();
            self.source_start_line = frame.source_start_line;
            self.source_scroll = 0;
        }
    }

    /// Select previous stack frame (move up the call stack)
    fn select_prev_frame(&mut self) {
        if self.selected_frame > 0 {
            self.selected_frame -= 1;
            self.locals_scroll = 0;
            self.update_source_from_selected_frame();
            // Request locals if not cached
            if !self.frame_locals.contains_key(&self.selected_frame) {
                self.pending_locals_request = Some(self.selected_frame);
            }
        }
    }

    /// Select next stack frame (move down the call stack)
    fn select_next_frame(&mut self) {
        if self.selected_frame < self.stack.len().saturating_sub(1) {
            self.selected_frame += 1;
            self.locals_scroll = 0;
            self.update_source_from_selected_frame();
            // Request locals if not cached
            if !self.frame_locals.contains_key(&self.selected_frame) {
                self.pending_locals_request = Some(self.selected_frame);
            }
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
        let locals_len = self.current_locals().len();
        if self.locals_scroll < locals_len.saturating_sub(self.visible_rows) {
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

        // === Paused state - show source, call stack and locals ===

        // Get current line from selected frame (or fallback to state for frame 0)
        let current_line = if let Some(frame) = self.stack.get(self.selected_frame) {
            frame.line
        } else {
            match &self.state {
                DebugState::Paused { line, .. } => *line,
                _ => 0,
            }
        };

        // Source code section (takes priority, shown at top)
        debug_log(&format!("draw: source_code={:?}, current_line={}", self.source_code.as_ref().map(|s| s.len()), current_line));
        if let Some(ref source) = self.source_code {
            let source_lines: Vec<&str> = source.lines().collect();
            let source_height = 8.min(source_lines.len()); // Show up to 8 lines

            printer.with_color(color_title, |p| {
                p.print((1, y), "Source:");
            });
            y += 1;

            // Convert current_line (file line) to relative line within source
            let relative_line = current_line.saturating_sub(self.source_start_line);

            // Calculate scroll to keep current line visible (using relative line)
            let scroll = if relative_line > 4 {
                (relative_line - 4).min(source_lines.len().saturating_sub(source_height))
            } else {
                0
            };

            let color_arrow = ColorStyle::new(Color::Rgb(100, 255, 100), Color::TerminalDefault);
            let color_line_num = ColorStyle::new(Color::Rgb(100, 100, 100), Color::TerminalDefault);
            let color_current_line = ColorStyle::new(Color::Rgb(255, 255, 200), Color::TerminalDefault);

            for (i, line_content) in source_lines.iter().skip(scroll).take(source_height).enumerate() {
                // Line number in the file = source_start_line + offset within source
                let line_num = self.source_start_line + scroll + i;
                let is_current = line_num == current_line;

                // Arrow for current line
                if is_current {
                    printer.with_color(color_arrow, |p| {
                        p.print((1, y), "▶");
                    });
                } else {
                    printer.print((1, y), " ");
                }

                // Line number
                printer.with_color(color_line_num, |p| {
                    p.print((3, y), &format!("{:3} ", line_num));
                });

                // Source line (truncated if needed)
                let max_line_len = width.saturating_sub(9);
                let display_line = if line_content.len() > max_line_len {
                    format!("{}…", &line_content[..max_line_len.saturating_sub(1)])
                } else {
                    line_content.to_string()
                };

                let line_style = if is_current { color_current_line } else { color_value };
                printer.with_color(line_style, |p| {
                    p.print((8, y), &display_line);
                });
                y += 1;
            }
            y += 1;
        }

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

                // Strip arity suffix (e.g., "module.func/_" -> "module.func")
                let func_name = frame.function.split('/').next().unwrap_or(&frame.function);

                printer.with_color(style, |p| {
                    let text = format!("{}{}{}", prefix, func_name, line_info);
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

        // Locals section (for selected frame)
        let locals = self.current_locals();
        let locals_title = if self.selected_frame == 0 {
            "Locals:".to_string()
        } else {
            format!("Locals (frame {}):", self.selected_frame)
        };
        printer.with_color(color_title, |p| {
            p.print((1, y), &locals_title);
        });
        y += 1;

        if locals.is_empty() {
            let msg = if self.pending_locals_request.is_some() {
                "(loading...)"
            } else {
                "(no locals)"
            };
            printer.with_color(color_dim, |p| {
                p.print((3, y), msg);
            });
            y += 1;
        } else {
            let locals_height = (height - y - 3).min(locals.len());
            let color_mvar = ColorStyle::new(Color::Rgb(255, 180, 100), Color::TerminalDefault);
            let color_return = ColorStyle::new(Color::Rgb(100, 255, 100), Color::TerminalDefault); // Green for return value
            for (name, value, type_name) in locals.iter()
                .skip(self.locals_scroll)
                .take(locals_height)
            {
                let is_mvar = type_name == "mvar";
                let is_return = type_name == "return";
                let name_display = if name.len() > 15 {
                    format!("{}...", &name[..12])
                } else {
                    name.clone()
                };

                // Use orange color for mvars, green for return value
                let name_color = if is_return { color_return } else if is_mvar { color_mvar } else { color_label };
                printer.with_color(name_color, |p| {
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
        debug_log(&format!("on_event: {:?}, state: {:?}", event, self.state));
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
            // Debug commands - set pending command for TUI to poll
            Event::Key(Key::F5) | Event::Char('c') | Event::Char('C') => {
                debug_log("Setting pending_command = Continue");
                self.pending_command = Some(DebugPanelCommand::Continue);
                EventResult::Consumed(None)
            }
            Event::Key(Key::F10) | Event::Char('n') | Event::Char('N') => {
                debug_log("Setting pending_command = StepOver");
                self.pending_command = Some(DebugPanelCommand::StepOver);
                EventResult::Consumed(None)
            }
            Event::Key(Key::F11) | Event::Char('s') | Event::Char('S') => {
                debug_log("Setting pending_command = StepIn");
                self.pending_command = Some(DebugPanelCommand::StepIn);
                EventResult::Consumed(None)
            }
            Event::Char('o') | Event::Char('O') => {
                debug_log("Setting pending_command = StepOut");
                self.pending_command = Some(DebugPanelCommand::StepOut);
                EventResult::Consumed(None)
            }
            Event::Char('q') | Event::Char('Q') => {
                debug_log("Setting pending_command = Stop");
                self.pending_command = Some(DebugPanelCommand::Stop);
                EventResult::Consumed(None)
            }
            _ => EventResult::Ignored,
        }
    }
}
