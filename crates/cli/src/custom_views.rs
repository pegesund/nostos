use cursive::event::{Event, EventResult, Key};
use cursive::view::{View, ViewWrapper};
use cursive::{Printer, Vec2, Rect};
use cursive::theme::{Color, ColorStyle};

/// A wrapper that draws a colored border when focused.
pub struct ActiveWindow<V> {
    view: V,
    title: String,
}

impl<V: View> ActiveWindow<V> {
    pub fn new(view: V, title: &str) -> Self {
        Self {
            view,
            title: title.to_string(),
        }
    }
}

impl<V: View> ViewWrapper for ActiveWindow<V> {
    type V = V;

    fn with_view<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&Self::V) -> R,
    {
        Some(f(&self.view))
    }

    fn with_view_mut<F, R>(&mut self, f: F) -> Option<R>
    where
        F: FnOnce(&mut Self::V) -> R,
    {
        Some(f(&mut self.view))
    }

    fn wrap_draw(&self, printer: &Printer) {
        // Try to detect focus. Note: printer.focused might be false if a child is focused
        // but we assume this wrapper is used directly on focusable items or we accept the limitation.
        let style = if printer.focused {
            // Active color (Orange/Yellow)
            ColorStyle::new(Color::Rgb(255, 255, 0), Color::TerminalDefault)
        } else {
            // Inactive color (Grey)
            ColorStyle::new(Color::Rgb(100, 100, 100), Color::TerminalDefault)
        };

        printer.with_color(style, |p| {
            p.print_box((0, 0), p.size, true);
            p.print((2, 0), &format!(" {} ", self.title));
        });

        // Draw content
        let inner_size = printer.size.saturating_sub((2, 2));
        if inner_size.x > 0 && inner_size.y > 0 {
            self.view.draw(&printer.windowed(Rect::from_size((1, 1), inner_size)));
        }
    }

    fn wrap_required_size(&mut self, req: Vec2) -> Vec2 {
        let child_req = self.view.required_size(req.saturating_sub((2, 2)));
        child_req + (2, 2)
    }

    fn wrap_layout(&mut self, size: Vec2) {
        self.view.layout(size.saturating_sub((2, 2)));
    }

    fn wrap_on_event(&mut self, event: Event) -> EventResult {
        // Ignore Shift+Tab to allow global cycling
        if let Event::Shift(Key::Tab) = event {
            return EventResult::Ignored;
        }
        self.view.on_event(event)
    }
}