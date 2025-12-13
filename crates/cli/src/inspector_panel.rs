//! Inspector panel with tabbed value browsing for the TUI.

use cursive::event::{Event, EventResult, Key};
use cursive::theme::{Color, ColorStyle};
use cursive::view::{View, CannotFocus};
use cursive::direction::Direction;
use cursive::{Printer, Vec2};
use nostos_repl::{InspectEntry, ThreadSafeValue, ThreadSafeMapKey};
use std::collections::VecDeque;

/// Maximum number of tabs in the inspector
const MAX_TABS: usize = 10;

/// An entry in the inspector (a tab)
#[derive(Clone)]
pub struct InspectorTab {
    /// Name/label of the tab
    pub name: String,
    /// The value being inspected (thread-safe copy)
    pub value: ThreadSafeValue,
    /// Current navigation path (breadcrumb)
    pub path: Vec<String>,
    /// Selected index in current view
    pub selected: usize,
    /// Scroll offset for pagination
    pub scroll_offset: usize,
}

impl InspectorTab {
    fn new(name: String, value: ThreadSafeValue) -> Self {
        Self {
            name,
            value,
            path: Vec::new(),
            selected: 0,
            scroll_offset: 0,
        }
    }

    /// Get the current value being viewed (following the path)
    fn current_value(&self) -> Option<ThreadSafeValue> {
        let mut current = self.value.clone();
        for segment in &self.path {
            current = self.navigate_into(&current, segment)?;
        }
        Some(current)
    }

    /// Navigate into a child by path segment
    fn navigate_into(&self, value: &ThreadSafeValue, segment: &str) -> Option<ThreadSafeValue> {
        match value {
            ThreadSafeValue::List(items) => {
                // Parse index from "[n]"
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                items.get(idx).cloned()
            }
            ThreadSafeValue::Tuple(items) => {
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                items.get(idx).cloned()
            }
            ThreadSafeValue::Record { fields, field_names, .. } => {
                // Parse field name from ".name"
                let field_name = segment.trim_start_matches('.');
                let idx = field_names.iter().position(|n| n == field_name)?;
                fields.get(idx).cloned()
            }
            ThreadSafeValue::Map(entries) => {
                // For maps, we navigate by index for simplicity
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                entries.get(idx).map(|(_, v)| v.clone())
            }
            ThreadSafeValue::Variant { fields, .. } => {
                let idx: usize = segment.trim_start_matches('.').parse().ok()?;
                fields.get(idx).cloned()
            }
            ThreadSafeValue::Closure { captures, .. } => {
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                captures.get(idx).cloned()
            }
            ThreadSafeValue::Int64Array(items) => {
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                items.get(idx).map(|v| ThreadSafeValue::Int64(*v))
            }
            ThreadSafeValue::Float64Array(items) => {
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                items.get(idx).map(|v| ThreadSafeValue::Float64(*v))
            }
            _ => None,
        }
    }

    /// Get slots (children) of a value
    fn get_slots(&self, value: &ThreadSafeValue) -> Vec<(String, String, String, bool)> {
        // Returns: (path_segment, type_name, preview, is_leaf)
        match value {
            ThreadSafeValue::List(items) => {
                items.iter().enumerate().map(|(i, v)| {
                    (format!("[{}]", i), self.type_name(v), self.preview(v), self.is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Tuple(items) => {
                items.iter().enumerate().map(|(i, v)| {
                    (format!("[{}]", i), self.type_name(v), self.preview(v), self.is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Record { fields, field_names, .. } => {
                field_names.iter().zip(fields.iter()).map(|(name, v)| {
                    (format!(".{}", name), self.type_name(v), self.preview(v), self.is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Map(entries) => {
                entries.iter().enumerate().map(|(i, (k, v))| {
                    (format!("[{}]", i), format!("{} -> {}", self.map_key_preview(k), self.type_name(v)), self.preview(v), self.is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Variant { fields, .. } => {
                fields.iter().enumerate().map(|(i, v)| {
                    (format!(".{}", i), self.type_name(v), self.preview(v), self.is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Closure { captures, capture_names, .. } => {
                capture_names.iter().zip(captures.iter()).enumerate().map(|(i, (name, v))| {
                    (format!("[{}]", i), format!("{}: {}", name, self.type_name(v)), self.preview(v), self.is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Int64Array(items) => {
                items.iter().enumerate().map(|(i, v)| {
                    (format!("[{}]", i), "Int64".to_string(), v.to_string(), true)
                }).collect()
            }
            ThreadSafeValue::Float64Array(items) => {
                items.iter().enumerate().map(|(i, v)| {
                    (format!("[{}]", i), "Float64".to_string(), v.to_string(), true)
                }).collect()
            }
            ThreadSafeValue::Set(items) => {
                items.iter().enumerate().map(|(i, k)| {
                    (format!("[{}]", i), "Key".to_string(), self.map_key_preview(k), true)
                }).collect()
            }
            _ => Vec::new(),
        }
    }

    fn type_name(&self, value: &ThreadSafeValue) -> String {
        match value {
            ThreadSafeValue::Unit => "Unit".to_string(),
            ThreadSafeValue::Bool(_) => "Bool".to_string(),
            ThreadSafeValue::Int64(_) => "Int64".to_string(),
            ThreadSafeValue::Float64(_) => "Float64".to_string(),
            ThreadSafeValue::Pid(_) => "Pid".to_string(),
            ThreadSafeValue::String(_) => "String".to_string(),
            ThreadSafeValue::Char(_) => "Char".to_string(),
            ThreadSafeValue::List(items) => format!("List({})", items.len()),
            ThreadSafeValue::Tuple(items) => format!("Tuple({})", items.len()),
            ThreadSafeValue::Record { type_name, fields, .. } => format!("{}({})", type_name, fields.len()),
            ThreadSafeValue::Closure { .. } => "Closure".to_string(),
            ThreadSafeValue::Variant { type_name, constructor, .. } => format!("{}::{}", type_name, constructor),
            ThreadSafeValue::Function(f) => format!("Fn({})", f.name),
            ThreadSafeValue::NativeFunction(f) => format!("Native({})", f.name),
            ThreadSafeValue::Map(entries) => format!("Map({})", entries.len()),
            ThreadSafeValue::Set(items) => format!("Set({})", items.len()),
            ThreadSafeValue::Int64Array(items) => format!("Int64Array({})", items.len()),
            ThreadSafeValue::Float64Array(items) => format!("Float64Array({})", items.len()),
        }
    }

    fn preview(&self, value: &ThreadSafeValue) -> String {
        match value {
            ThreadSafeValue::Unit => "()".to_string(),
            ThreadSafeValue::Bool(b) => b.to_string(),
            ThreadSafeValue::Int64(n) => n.to_string(),
            ThreadSafeValue::Float64(f) => f.to_string(),
            ThreadSafeValue::Pid(p) => format!("<{}>", p),
            ThreadSafeValue::String(s) => {
                if s.len() > 30 {
                    format!("\"{}...\"", &s[..27])
                } else {
                    format!("\"{}\"", s)
                }
            }
            ThreadSafeValue::Char(c) => format!("'{}'", c),
            ThreadSafeValue::List(items) if items.is_empty() => "[]".to_string(),
            ThreadSafeValue::List(items) if items.len() <= 3 => {
                let previews: Vec<String> = items.iter().take(3).map(|v| self.preview(v)).collect();
                format!("[{}]", previews.join(", "))
            }
            ThreadSafeValue::List(items) => format!("[...] ({} items)", items.len()),
            ThreadSafeValue::Tuple(items) if items.is_empty() => "()".to_string(),
            ThreadSafeValue::Tuple(items) => format!("(...) ({} items)", items.len()),
            ThreadSafeValue::Record { type_name, .. } => format!("{}{{...}}", type_name),
            ThreadSafeValue::Closure { function, .. } => format!("<closure {}>", function.name),
            ThreadSafeValue::Variant { constructor, fields, .. } if fields.is_empty() => constructor.to_string(),
            ThreadSafeValue::Variant { constructor, .. } => format!("{}(...)", constructor),
            ThreadSafeValue::Function(f) => format!("<fn {}>", f.name),
            ThreadSafeValue::NativeFunction(f) => format!("<native {}>", f.name),
            ThreadSafeValue::Map(entries) if entries.is_empty() => "{}".to_string(),
            ThreadSafeValue::Map(entries) => format!("{{...}} ({} entries)", entries.len()),
            ThreadSafeValue::Set(items) if items.is_empty() => "Set{}".to_string(),
            ThreadSafeValue::Set(items) => format!("Set{{...}} ({} items)", items.len()),
            ThreadSafeValue::Int64Array(items) => format!("Int64Array({} items)", items.len()),
            ThreadSafeValue::Float64Array(items) => format!("Float64Array({} items)", items.len()),
        }
    }

    fn map_key_preview(&self, key: &ThreadSafeMapKey) -> String {
        match key {
            ThreadSafeMapKey::Unit => "()".to_string(),
            ThreadSafeMapKey::Bool(b) => b.to_string(),
            ThreadSafeMapKey::Char(c) => format!("'{}'", c),
            ThreadSafeMapKey::Int8(n) => n.to_string(),
            ThreadSafeMapKey::Int16(n) => n.to_string(),
            ThreadSafeMapKey::Int32(n) => n.to_string(),
            ThreadSafeMapKey::Int64(n) => n.to_string(),
            ThreadSafeMapKey::UInt8(n) => n.to_string(),
            ThreadSafeMapKey::UInt16(n) => n.to_string(),
            ThreadSafeMapKey::UInt32(n) => n.to_string(),
            ThreadSafeMapKey::UInt64(n) => n.to_string(),
            ThreadSafeMapKey::String(s) => {
                if s.len() > 20 {
                    format!("\"{}...\"", &s[..17])
                } else {
                    format!("\"{}\"", s)
                }
            }
        }
    }

    fn is_leaf(&self, value: &ThreadSafeValue) -> bool {
        match value {
            ThreadSafeValue::Unit | ThreadSafeValue::Bool(_) | ThreadSafeValue::Int64(_) |
            ThreadSafeValue::Float64(_) | ThreadSafeValue::Pid(_) | ThreadSafeValue::Char(_) |
            ThreadSafeValue::Function(_) | ThreadSafeValue::NativeFunction(_) => true,
            ThreadSafeValue::String(s) => s.len() <= 50,
            ThreadSafeValue::List(items) => items.is_empty(),
            ThreadSafeValue::Tuple(items) => items.is_empty(),
            ThreadSafeValue::Record { fields, .. } => fields.is_empty(),
            ThreadSafeValue::Closure { captures, .. } => captures.is_empty(),
            ThreadSafeValue::Variant { fields, .. } => fields.is_empty(),
            ThreadSafeValue::Map(entries) => entries.is_empty(),
            ThreadSafeValue::Set(items) => items.is_empty(),
            ThreadSafeValue::Int64Array(items) => items.is_empty(),
            ThreadSafeValue::Float64Array(items) => items.is_empty(),
        }
    }
}

/// The inspector panel with tabs
pub struct InspectorPanel {
    /// Tabs in order (front = most recent)
    tabs: VecDeque<InspectorTab>,
    /// Currently active tab index
    active_tab: usize,
    /// Visible rows in the value browser
    visible_rows: usize,
}

impl InspectorPanel {
    pub fn new() -> Self {
        Self {
            tabs: VecDeque::new(),
            active_tab: 0,
            visible_rows: 10,
        }
    }

    /// Add or update a tab with the given name and value.
    /// If a tab with this name exists, move it to front and update.
    /// If max tabs reached, remove oldest.
    pub fn add_or_update(&mut self, name: String, value: ThreadSafeValue) {
        // Check if tab exists
        if let Some(pos) = self.tabs.iter().position(|t| t.name == name) {
            // Remove and re-add at front
            self.tabs.remove(pos);
        }

        // Remove oldest if at max
        if self.tabs.len() >= MAX_TABS {
            self.tabs.pop_back();
        }

        // Add at front
        self.tabs.push_front(InspectorTab::new(name, value));
        self.active_tab = 0;
    }

    /// Process incoming inspect entries
    pub fn process_entries(&mut self, entries: Vec<InspectEntry>) {
        for entry in entries {
            self.add_or_update(entry.name, entry.value);
        }
    }

    /// Close the current tab
    pub fn close_current_tab(&mut self) {
        if !self.tabs.is_empty() {
            self.tabs.remove(self.active_tab);
            if self.active_tab >= self.tabs.len() && self.active_tab > 0 {
                self.active_tab -= 1;
            }
        }
    }

    /// Select next tab
    pub fn next_tab(&mut self) {
        if !self.tabs.is_empty() {
            self.active_tab = (self.active_tab + 1) % self.tabs.len();
        }
    }

    /// Select previous tab
    pub fn prev_tab(&mut self) {
        if !self.tabs.is_empty() {
            if self.active_tab == 0 {
                self.active_tab = self.tabs.len() - 1;
            } else {
                self.active_tab -= 1;
            }
        }
    }

    /// Get the number of tabs
    pub fn tab_count(&self) -> usize {
        self.tabs.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tabs.is_empty()
    }

    fn current_tab(&self) -> Option<&InspectorTab> {
        self.tabs.get(self.active_tab)
    }

    fn current_tab_mut(&mut self) -> Option<&mut InspectorTab> {
        self.tabs.get_mut(self.active_tab)
    }

    /// Navigate into selected slot
    fn navigate_into(&mut self) {
        if let Some(tab) = self.current_tab_mut() {
            if let Some(current) = tab.current_value() {
                let slots = tab.get_slots(&current);
                if let Some((path_seg, _, _, is_leaf)) = slots.get(tab.selected) {
                    if !is_leaf {
                        tab.path.push(path_seg.clone());
                        tab.selected = 0;
                        tab.scroll_offset = 0;
                    }
                }
            }
        }
    }

    /// Navigate up (back)
    fn navigate_up(&mut self) {
        if let Some(tab) = self.current_tab_mut() {
            if !tab.path.is_empty() {
                tab.path.pop();
                tab.selected = 0;
                tab.scroll_offset = 0;
            }
        }
    }

    /// Select previous item
    fn select_prev(&mut self) {
        if let Some(tab) = self.current_tab_mut() {
            if let Some(current) = tab.current_value() {
                let slots = tab.get_slots(&current);
                if !slots.is_empty() {
                    if tab.selected == 0 {
                        tab.selected = slots.len() - 1;
                    } else {
                        tab.selected -= 1;
                    }
                    // Update scroll
                    if tab.selected < tab.scroll_offset {
                        tab.scroll_offset = tab.selected;
                    }
                }
            }
        }
    }

    /// Select next item
    fn select_next(&mut self) {
        let visible_rows = self.visible_rows;
        if let Some(tab) = self.current_tab_mut() {
            if let Some(current) = tab.current_value() {
                let slots = tab.get_slots(&current);
                if !slots.is_empty() {
                    tab.selected = (tab.selected + 1) % slots.len();
                    // Update scroll
                    if tab.selected >= tab.scroll_offset + visible_rows {
                        tab.scroll_offset = tab.selected - visible_rows + 1;
                    }
                    if tab.selected < tab.scroll_offset {
                        tab.scroll_offset = tab.selected;
                    }
                }
            }
        }
    }
}

impl View for InspectorPanel {
    fn draw(&self, printer: &Printer) {
        if self.tabs.is_empty() {
            printer.print((1, 1), "No values to inspect");
            printer.print((1, 2), "Use inspect(value, \"name\") in your code");
            return;
        }

        let width = printer.size.x;
        let height = printer.size.y;

        // Draw tab bar at top
        let mut x = 0;
        for (i, tab) in self.tabs.iter().enumerate() {
            let label = if tab.name.len() > 12 {
                format!("{}...", &tab.name[..9])
            } else {
                tab.name.clone()
            };

            let style = if i == self.active_tab {
                ColorStyle::new(Color::Rgb(0, 0, 0), Color::Rgb(255, 255, 0))
            } else {
                ColorStyle::new(Color::Rgb(200, 200, 200), Color::TerminalDefault)
            };

            printer.with_color(style, |p| {
                p.print((x, 0), &format!(" {} ", label));
            });
            x += label.len() + 3;
            if x >= width - 3 {
                break;
            }
        }

        // Draw close hint
        if width > 20 {
            printer.print((width.saturating_sub(8), 0), "[x:close]");
        }

        // Draw separator
        for i in 0..width {
            printer.print((i, 1), "─");
        }

        // Draw current tab content
        if let Some(tab) = self.current_tab() {
            // Draw breadcrumb path
            let path_str = if tab.path.is_empty() {
                tab.name.clone()
            } else {
                format!("{}{}", tab.name, tab.path.join(""))
            };
            let path_display = if path_str.len() > width - 2 {
                format!("...{}", &path_str[path_str.len() - width + 5..])
            } else {
                path_str
            };
            printer.with_color(ColorStyle::new(Color::Rgb(100, 200, 255), Color::TerminalDefault), |p| {
                p.print((0, 2), &path_display);
            });

            // Draw type and preview
            if let Some(current) = tab.current_value() {
                let type_str = tab.type_name(&current);
                let preview = tab.preview(&current);
                printer.print((0, 3), &format!("{}: {}", type_str, preview));

                // Draw slots
                let slots = tab.get_slots(&current);
                if slots.is_empty() {
                    printer.print((0, 5), "(no children to browse)");
                } else {
                    let start_y = 5;
                    let visible = (height - start_y).min(slots.len() - tab.scroll_offset);

                    for (i, (path_seg, type_name, preview, is_leaf)) in slots.iter()
                        .skip(tab.scroll_offset)
                        .take(visible)
                        .enumerate()
                    {
                        let y = start_y + i;
                        let actual_idx = tab.scroll_offset + i;
                        let selected = actual_idx == tab.selected;

                        let prefix = if selected { ">" } else { " " };
                        let arrow = if *is_leaf { " " } else { "→" };

                        let line = format!("{} {} {} {} {}", prefix, path_seg, type_name, arrow, preview);
                        let line = if line.len() > width {
                            format!("{}...", &line[..width.saturating_sub(3)])
                        } else {
                            line
                        };

                        if selected && printer.focused {
                            printer.with_color(ColorStyle::new(Color::Rgb(0, 0, 0), Color::Rgb(255, 255, 0)), |p| {
                                p.print((0, y), &line);
                                // Pad to full width
                                for x in line.len()..width {
                                    p.print((x, y), " ");
                                }
                            });
                        } else if selected {
                            printer.with_color(ColorStyle::new(Color::Rgb(255, 255, 0), Color::TerminalDefault), |p| {
                                p.print((0, y), &line);
                            });
                        } else {
                            printer.print((0, y), &line);
                        }
                    }

                    // Scroll indicator
                    if slots.len() > visible + tab.scroll_offset {
                        printer.print((width.saturating_sub(5), height - 1), "more↓");
                    }
                    if tab.scroll_offset > 0 {
                        printer.print((width.saturating_sub(5), start_y), "more↑");
                    }
                }
            }
        }
    }

    fn required_size(&mut self, constraint: Vec2) -> Vec2 {
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
            // Close tab
            Event::Char('x') | Event::Char('X') => {
                self.close_current_tab();
                EventResult::Consumed(None)
            }
            // Navigation within value browser
            Event::Key(Key::Up) => {
                self.select_prev();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Down) => {
                self.select_next();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Enter) => {
                self.navigate_into();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Right) => {
                // If at root level (no path), switch to next tab; otherwise navigate into
                if self.current_tab().map(|t| t.path.is_empty()).unwrap_or(true) {
                    self.next_tab();
                } else {
                    self.navigate_into();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Left) => {
                // If at root level (no path), switch to prev tab; otherwise navigate up
                if self.current_tab().map(|t| t.path.is_empty()).unwrap_or(true) {
                    self.prev_tab();
                } else {
                    self.navigate_up();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Backspace) => {
                self.navigate_up();
                EventResult::Consumed(None)
            }
            // Page navigation
            Event::Key(Key::PageUp) => {
                for _ in 0..self.visible_rows {
                    self.select_prev();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::PageDown) => {
                for _ in 0..self.visible_rows {
                    self.select_next();
                }
                EventResult::Consumed(None)
            }
            _ => EventResult::Ignored,
        }
    }

    fn layout(&mut self, size: Vec2) {
        self.visible_rows = size.y.saturating_sub(6);
    }
}
