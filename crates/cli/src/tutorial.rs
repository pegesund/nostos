//! Tutorial content for the Nostos TUI
//!
//! This module provides embedded tutorial content that can be displayed
//! in a scrollable panel within the TUI.

/// A tutorial chapter with title and content
pub struct Chapter {
    pub number: usize,
    pub title: &'static str,
    pub content: &'static str,
}

/// All tutorial chapters
pub static CHAPTERS: &[Chapter] = &[
    Chapter {
        number: 0,
        title: "Table of Contents",
        content: include_str!("../../../docs/tutorial/00_index.md"),
    },
    Chapter {
        number: 1,
        title: "Language Basics",
        content: include_str!("../../../docs/tutorial/01_basics.md"),
    },
    Chapter {
        number: 2,
        title: "Functions",
        content: include_str!("../../../docs/tutorial/02_functions.md"),
    },
    Chapter {
        number: 3,
        title: "Pattern Matching",
        content: include_str!("../../../docs/tutorial/03_pattern_matching.md"),
    },
    Chapter {
        number: 4,
        title: "Lists & Tuples",
        content: include_str!("../../../docs/tutorial/04_lists_tuples.md"),
    },
    Chapter {
        number: 5,
        title: "Maps & Sets",
        content: include_str!("../../../docs/tutorial/05_maps_sets.md"),
    },
    Chapter {
        number: 6,
        title: "Typed Arrays",
        content: include_str!("../../../docs/tutorial/06_typed_arrays.md"),
    },
    Chapter {
        number: 7,
        title: "Type System",
        content: include_str!("../../../docs/tutorial/07_type_system.md"),
    },
    Chapter {
        number: 8,
        title: "Traits",
        content: include_str!("../../../docs/tutorial/08_traits.md"),
    },
    Chapter {
        number: 9,
        title: "Builtin Traits",
        content: include_str!("../../../docs/tutorial/09_builtin_traits.md"),
    },
    Chapter {
        number: 10,
        title: "Trait Bounds",
        content: include_str!("../../../docs/tutorial/10_trait_bounds.md"),
    },
    Chapter {
        number: 11,
        title: "Error Handling",
        content: include_str!("../../../docs/tutorial/11_error_handling.md"),
    },
    Chapter {
        number: 12,
        title: "Modules & Imports",
        content: include_str!("../../../docs/tutorial/12_modules.md"),
    },
    Chapter {
        number: 13,
        title: "Standard Library",
        content: include_str!("../../../docs/tutorial/13_stdlib.md"),
    },
    Chapter {
        number: 14,
        title: "JSON",
        content: include_str!("../../../docs/tutorial/14_json.md"),
    },
    Chapter {
        number: 15,
        title: "Concurrency",
        content: include_str!("../../../docs/tutorial/15_concurrency.md"),
    },
    Chapter {
        number: 16,
        title: "Async Runtime",
        content: include_str!("../../../docs/tutorial/16_async_runtime.md"),
    },
    Chapter {
        number: 17,
        title: "Async I/O & HTTP",
        content: include_str!("../../../docs/tutorial/17_async_io_http.md"),
    },
    Chapter {
        number: 18,
        title: "Reflection & Eval",
        content: include_str!("../../../docs/tutorial/18_reflection.md"),
    },
    Chapter {
        number: 19,
        title: "Debugging & Profiling",
        content: include_str!("../../../docs/tutorial/19_debugging_profiling.md"),
    },
    Chapter {
        number: 20,
        title: "HTML Templating",
        content: include_str!("../../../docs/tutorial/20_html_templating.md"),
    },
    Chapter {
        number: 21,
        title: "Mutability",
        content: include_str!("../../../docs/tutorial/21_mutability.md"),
    },
    Chapter {
        number: 22,
        title: "Templates & Metaprogramming",
        content: include_str!("../../../docs/tutorial/22_templates.md"),
    },
];

/// Get the total number of chapters
pub fn chapter_count() -> usize {
    CHAPTERS.len()
}

/// Get a chapter by index (0-based)
pub fn get_chapter(index: usize) -> Option<&'static Chapter> {
    CHAPTERS.get(index)
}

/// Get the title for display in the panel header
pub fn format_chapter_title(index: usize) -> String {
    if let Some(chapter) = get_chapter(index) {
        if chapter.number == 0 {
            format!("Tutorial: {}", chapter.title)
        } else {
            format!("Tutorial: {}/{} - {}", chapter.number, chapter_count() - 1, chapter.title)
        }
    } else {
        "Tutorial".to_string()
    }
}
