//! Git history panel for the TUI - shows commit history and diffs for definitions/modules.

use cursive::event::{Event, EventResult, Key};
use cursive::theme::{Color, ColorStyle};
use cursive::view::{View, CannotFocus};
use cursive::direction::Direction;
use cursive::{Printer, Vec2};
use nostos_source::CommitInfo;
use std::collections::HashMap;

/// What we're viewing history for
#[derive(Clone, Debug)]
pub enum HistoryTarget {
    /// History for a single definition
    Definition(String),
    /// History for an entire module
    Module(String),
}

impl HistoryTarget {
    pub fn display_name(&self) -> &str {
        match self {
            HistoryTarget::Definition(name) => name,
            HistoryTarget::Module(name) => name,
        }
    }
}

/// View mode within the panel
#[derive(Clone, Debug, PartialEq)]
pub enum HistoryViewMode {
    /// Showing commit list with inline diff preview
    CommitList,
    /// Showing full content at a specific commit
    FullContent,
}

/// Commands from git panel to TUI
#[derive(Clone, Debug)]
pub enum GitPanelCommand {
    /// Close the panel
    Close,
    /// Restore the selected commit version
    Restore { commit: String, content: String },
}

/// The git history panel
pub struct GitHistoryPanel {
    /// What we're viewing history for
    pub target: HistoryTarget,
    /// List of commits (newest first)
    commits: Vec<CommitInfo>,
    /// Currently selected commit index
    selected_index: usize,
    /// Cached diffs keyed by commit hash
    diff_cache: HashMap<String, String>,
    /// Cached content at commit keyed by commit hash
    content_cache: HashMap<String, String>,
    /// Scroll offset for commit list
    list_scroll: usize,
    /// Scroll offset for diff/content view
    diff_scroll: usize,
    /// Current view mode
    view_mode: HistoryViewMode,
    /// Visible height (updated during draw)
    visible_height: usize,
    /// Pending command for TUI to poll
    pub pending_command: Option<GitPanelCommand>,
    /// Error message to display (if any)
    error_message: Option<String>,
}

impl GitHistoryPanel {
    /// Create a new git history panel
    pub fn new(target: HistoryTarget) -> Self {
        Self {
            target,
            commits: Vec::new(),
            selected_index: 0,
            diff_cache: HashMap::new(),
            content_cache: HashMap::new(),
            list_scroll: 0,
            diff_scroll: 0,
            view_mode: HistoryViewMode::CommitList,
            visible_height: 20,
            pending_command: None,
            error_message: None,
        }
    }

    /// Set the commits to display
    pub fn set_commits(&mut self, commits: Vec<CommitInfo>) {
        self.commits = commits;
        self.selected_index = 0;
        self.list_scroll = 0;
        self.diff_scroll = 0;
    }

    /// Set an error message
    pub fn set_error(&mut self, msg: String) {
        self.error_message = Some(msg);
    }

    /// Get the currently selected commit (if any)
    pub fn selected_commit(&self) -> Option<&CommitInfo> {
        self.commits.get(self.selected_index)
    }

    /// Check if we need a diff for the selected commit
    pub fn needs_diff(&self) -> Option<&str> {
        if let Some(commit) = self.selected_commit() {
            if !self.diff_cache.contains_key(&commit.hash) {
                return Some(&commit.hash);
            }
        }
        None
    }

    /// Check if we need content for the selected commit (in full view mode)
    pub fn needs_content(&self) -> Option<&str> {
        if self.view_mode == HistoryViewMode::FullContent {
            if let Some(commit) = self.selected_commit() {
                if !self.content_cache.contains_key(&commit.hash) {
                    return Some(&commit.hash);
                }
            }
        }
        None
    }

    /// Set diff for a commit
    pub fn set_diff(&mut self, commit_hash: &str, diff: String) {
        self.diff_cache.insert(commit_hash.to_string(), diff);
    }

    /// Set content for a commit
    pub fn set_content(&mut self, commit_hash: &str, content: String) {
        self.content_cache.insert(commit_hash.to_string(), content);
    }

    /// Take pending command (polled by TUI)
    pub fn take_pending_command(&mut self) -> Option<GitPanelCommand> {
        self.pending_command.take()
    }

    /// Navigate up in commit list
    fn select_prev(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
            self.diff_scroll = 0;
            // Adjust scroll if needed
            if self.selected_index < self.list_scroll {
                self.list_scroll = self.selected_index;
            }
        }
    }

    /// Navigate down in commit list
    fn select_next(&mut self) {
        if self.selected_index < self.commits.len().saturating_sub(1) {
            self.selected_index += 1;
            self.diff_scroll = 0;
            // Adjust scroll if needed (assuming ~8 visible commits)
            let visible_commits = 8;
            if self.selected_index >= self.list_scroll + visible_commits {
                self.list_scroll = self.selected_index - visible_commits + 1;
            }
        }
    }

    /// Scroll diff view up
    fn scroll_diff_up(&mut self) {
        if self.diff_scroll > 0 {
            self.diff_scroll -= 1;
        }
    }

    /// Scroll diff view down
    fn scroll_diff_down(&mut self) {
        self.diff_scroll += 1;
    }

    /// Toggle between list and full content view
    fn toggle_view_mode(&mut self) {
        self.view_mode = match self.view_mode {
            HistoryViewMode::CommitList => HistoryViewMode::FullContent,
            HistoryViewMode::FullContent => HistoryViewMode::CommitList,
        };
        self.diff_scroll = 0;
    }

    /// Format a date string (truncate to date only)
    fn format_date(date: &str) -> String {
        // Input format: "2024-12-25 14:30:00 +0100"
        // Output: "2024-12-25"
        date.split_whitespace().next().unwrap_or(date).to_string()
    }

    /// Get diff lines with coloring info
    fn get_diff_lines(&self) -> Vec<(String, DiffLineType)> {
        let Some(commit) = self.selected_commit() else {
            return vec![];
        };

        let Some(diff) = self.diff_cache.get(&commit.hash) else {
            return vec![("Loading...".to_string(), DiffLineType::Context)];
        };

        if diff.is_empty() {
            return vec![("(no changes in this commit)".to_string(), DiffLineType::Context)];
        }

        diff.lines()
            .filter(|line| {
                // Skip diff headers
                !line.starts_with("diff --git") &&
                !line.starts_with("index ") &&
                !line.starts_with("--- ") &&
                !line.starts_with("+++ ") &&
                !line.starts_with("@@ ")
            })
            .map(|line| {
                let line_type = if line.starts_with('+') {
                    DiffLineType::Added
                } else if line.starts_with('-') {
                    DiffLineType::Removed
                } else {
                    DiffLineType::Context
                };
                (line.to_string(), line_type)
            })
            .collect()
    }

    /// Get content lines for full view
    fn get_content_lines(&self) -> Vec<String> {
        let Some(commit) = self.selected_commit() else {
            return vec![];
        };

        let Some(content) = self.content_cache.get(&commit.hash) else {
            return vec!["Loading...".to_string()];
        };

        content.lines().map(String::from).collect()
    }
}

#[derive(Clone, Copy)]
enum DiffLineType {
    Added,
    Removed,
    Context,
}

impl View for GitHistoryPanel {
    fn draw(&self, printer: &Printer) {
        let width = printer.size.x;
        let height = printer.size.y;

        // Colors
        let color_title = ColorStyle::new(Color::Rgb(255, 200, 100), Color::TerminalDefault);
        let color_hash = ColorStyle::new(Color::Rgb(255, 180, 100), Color::TerminalDefault);
        let color_date = ColorStyle::new(Color::Rgb(150, 150, 150), Color::TerminalDefault);
        let color_message = ColorStyle::new(Color::Rgb(220, 220, 220), Color::TerminalDefault);
        let color_selected = ColorStyle::new(Color::Rgb(0, 0, 0), Color::Rgb(255, 255, 100));
        let color_added = ColorStyle::new(Color::Rgb(100, 255, 100), Color::TerminalDefault);
        let color_removed = ColorStyle::new(Color::Rgb(255, 100, 100), Color::TerminalDefault);
        let color_context = ColorStyle::new(Color::Rgb(180, 180, 180), Color::TerminalDefault);
        let color_dim = ColorStyle::new(Color::Rgb(100, 100, 100), Color::TerminalDefault);
        let color_key = ColorStyle::new(Color::Rgb(255, 200, 100), Color::TerminalDefault);
        let color_error = ColorStyle::new(Color::Rgb(255, 100, 100), Color::TerminalDefault);

        let mut y = 0;

        // Title
        let mode_str = match self.view_mode {
            HistoryViewMode::CommitList => "History",
            HistoryViewMode::FullContent => "Content",
        };
        let title = format!("{}: {} ({})", mode_str, self.target.display_name(),
            if self.commits.is_empty() { "no commits".to_string() }
            else { format!("{} commits", self.commits.len()) });
        printer.with_color(color_title, |p| {
            p.print((1, y), &title);
        });
        y += 2;

        // Error message if any
        if let Some(ref error) = self.error_message {
            printer.with_color(color_error, |p| {
                p.print((1, y), &format!("Error: {}", error));
            });
            y += 2;
        }

        if self.commits.is_empty() {
            printer.with_color(color_dim, |p| {
                p.print((1, y), "No history available for this definition.");
            });
            y = height.saturating_sub(2);
            printer.with_color(color_key, |p| {
                p.print((1, y), "[Esc] Close");
            });
            return;
        }

        // Calculate layout: commit list on top, diff/content below
        let commit_list_height = 10.min(self.commits.len() + 1);
        let diff_height = height.saturating_sub(commit_list_height + 5);

        // Draw commit list
        printer.with_color(color_dim, |p| {
            p.print((1, y), "Commits:");
        });
        y += 1;

        let visible_commits = (commit_list_height - 1).min(self.commits.len());
        for (i, commit) in self.commits.iter()
            .skip(self.list_scroll)
            .take(visible_commits)
            .enumerate()
        {
            let actual_index = self.list_scroll + i;
            let is_selected = actual_index == self.selected_index;

            let date_str = Self::format_date(&commit.date);
            let msg_max_len = width.saturating_sub(22);
            let message = if commit.message.len() > msg_max_len {
                format!("{}...", &commit.message[..msg_max_len.saturating_sub(3)])
            } else {
                commit.message.clone()
            };

            if is_selected {
                // Draw selected row with highlight
                printer.with_color(color_selected, |p| {
                    // Clear the line first
                    let line = format!(" {} {} {} ",
                        commit.short_hash, date_str, message);
                    let padded = format!("{:width$}", line, width = width.saturating_sub(2));
                    p.print((1, y), &padded);
                });
            } else {
                // Draw normal row
                printer.with_color(color_hash, |p| {
                    p.print((1, y), &commit.short_hash);
                });
                printer.with_color(color_date, |p| {
                    p.print((10, y), &date_str);
                });
                printer.with_color(color_message, |p| {
                    p.print((22, y), &message);
                });
            }
            y += 1;
        }

        // Show scroll indicator if needed
        if self.commits.len() > visible_commits {
            printer.with_color(color_dim, |p| {
                let indicator = format!("... {} more (scroll with arrows)",
                    self.commits.len() - visible_commits);
                p.print((1, y), &indicator);
            });
        }
        y += 2;

        // Draw separator
        for i in 0..width {
            printer.print((i, y), "─");
        }
        y += 1;

        // Draw diff or content section
        match self.view_mode {
            HistoryViewMode::CommitList => {
                printer.with_color(color_dim, |p| {
                    p.print((1, y), "Diff:");
                });
                y += 1;

                let diff_lines = self.get_diff_lines();
                let visible_diff_lines = diff_height.saturating_sub(1);

                for line_info in diff_lines.iter()
                    .skip(self.diff_scroll)
                    .take(visible_diff_lines)
                {
                    let (line, line_type) = line_info;
                    let color = match line_type {
                        DiffLineType::Added => color_added,
                        DiffLineType::Removed => color_removed,
                        DiffLineType::Context => color_context,
                    };

                    let display_line = if line.len() > width.saturating_sub(2) {
                        format!("{}...", &line[..width.saturating_sub(5)])
                    } else {
                        line.clone()
                    };

                    printer.with_color(color, |p| {
                        p.print((1, y), &display_line);
                    });
                    y += 1;
                }
            }
            HistoryViewMode::FullContent => {
                if let Some(commit) = self.selected_commit() {
                    printer.with_color(color_dim, |p| {
                        p.print((1, y), &format!("Content at {}:", commit.short_hash));
                    });
                }
                y += 1;

                let content_lines = self.get_content_lines();
                let visible_content_lines = diff_height.saturating_sub(1);

                for (i, line) in content_lines.iter()
                    .skip(self.diff_scroll)
                    .take(visible_content_lines)
                    .enumerate()
                {
                    let line_num = self.diff_scroll + i + 1;
                    printer.with_color(color_dim, |p| {
                        p.print((1, y), &format!("{:3} ", line_num));
                    });

                    let display_line = if line.len() > width.saturating_sub(6) {
                        format!("{}...", &line[..width.saturating_sub(9)])
                    } else {
                        line.clone()
                    };

                    printer.with_color(color_message, |p| {
                        p.print((5, y), &display_line);
                    });
                    y += 1;
                }
            }
        }

        // Key bindings at bottom
        y = height.saturating_sub(2);
        for i in 0..width {
            printer.print((i, y), "─");
        }
        y += 1;

        let keys = match self.view_mode {
            HistoryViewMode::CommitList =>
                "[↑↓] Select  [PgUp/PgDn] Scroll diff  [Enter] View content  [r] Restore  [Esc] Close",
            HistoryViewMode::FullContent =>
                "[↑↓] Select  [PgUp/PgDn] Scroll  [Enter] Back to diff  [r] Restore  [Esc] Close",
        };
        printer.with_color(color_key, |p| {
            let display = if keys.len() > width - 2 {
                &keys[..width.saturating_sub(2)]
            } else {
                keys
            };
            p.print((1, y), display);
        });
    }

    fn required_size(&mut self, constraint: Vec2) -> Vec2 {
        self.visible_height = constraint.y;
        constraint
    }

    fn take_focus(&mut self, _: Direction) -> Result<EventResult, CannotFocus> {
        Ok(EventResult::Consumed(None))
    }

    fn on_event(&mut self, event: Event) -> EventResult {
        match event {
            // Close panel
            Event::Key(Key::Esc) | Event::Char('q') => {
                self.pending_command = Some(GitPanelCommand::Close);
                EventResult::Consumed(None)
            }
            // Navigate commits
            Event::Key(Key::Up) => {
                self.select_prev();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Down) => {
                self.select_next();
                EventResult::Consumed(None)
            }
            // Scroll diff/content
            Event::Key(Key::PageUp) => {
                self.scroll_diff_up();
                EventResult::Consumed(None)
            }
            Event::Key(Key::PageDown) => {
                self.scroll_diff_down();
                EventResult::Consumed(None)
            }
            // Toggle view mode
            Event::Key(Key::Enter) => {
                self.toggle_view_mode();
                EventResult::Consumed(None)
            }
            // Restore (future feature)
            Event::Char('r') | Event::Char('R') => {
                if let Some(commit) = self.selected_commit() {
                    if let Some(content) = self.content_cache.get(&commit.hash) {
                        self.pending_command = Some(GitPanelCommand::Restore {
                            commit: commit.hash.clone(),
                            content: content.clone(),
                        });
                    }
                }
                EventResult::Consumed(None)
            }
            // Let Tab propagate for window cycling
            Event::Key(Key::Tab) | Event::Shift(Key::Tab) => {
                EventResult::Ignored
            }
            _ => EventResult::Ignored,
        }
    }
}
