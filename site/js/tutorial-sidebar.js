// Shared tutorial sidebar component
// This file generates the sidebar for all tutorial pages

const tutorialChapters = [
    { num: 1, file: "01_basics.html", title: "Basics" },
    { num: 2, file: "02_functions.html", title: "Functions" },
    { num: 3, file: "03_pattern_matching.html", title: "Pattern Matching" },
    { num: 4, file: "04_lists_tuples.html", title: "Lists & Tuples" },
    { num: 5, file: "05_maps_sets.html", title: "Maps & Sets" },
    { num: 6, file: "06_typed_arrays.html", title: "Typed Arrays" },
    { num: 7, file: "07_type_system.html", title: "Type System" },
    { num: 8, file: "08_traits.html", title: "Traits" },
    { num: 9, file: "09_builtin_traits.html", title: "Builtin Traits" },
    { num: 10, file: "10_trait_bounds.html", title: "Trait Bounds" },
    { num: 11, file: "11_error_handling.html", title: "Error Handling" },
    { num: 12, file: "12_modules.html", title: "Modules & Imports" },
    { num: 13, file: "13_stdlib.html", title: "Standard Library" },
    { num: 14, file: "14_json.html", title: "JSON" },
    { num: 15, file: "15_concurrency.html", title: "Concurrency" },
    { num: 16, file: "16_async_runtime.html", title: "Async Runtime" },
    { num: 17, file: "17_async_io_http.html", title: "Async I/O & HTTP" },
    { num: 18, file: "18_reflection.html", title: "Reflection & Eval" },
    { num: 19, file: "19_debugging_profiling.html", title: "Debugging & Profiling" },
    { num: 20, file: "20_html_templating.html", title: "HTML Templating" },
    { num: 21, file: "21_mutability.html", title: "Mutability" },
    { num: 22, file: "22_complete_web_app.html", title: "Complete Web App" },
    { num: 23, file: "23_ffi.html", title: "FFI & Extensions" },
    { num: 24, file: "24_command_line.html", title: "Command Line" },
    { num: 25, file: "25_reactive_records.html", title: "Reactive Records" },
];

function generateTutorialSidebar() {
    const currentPage = window.location.pathname.split('/').pop();

    const sidebar = document.getElementById('tutorial-sidebar');
    if (!sidebar) return;

    let html = `
        <div class="p-6">
            <h3 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">Contents</h3>
            <ul class="space-y-2">
    `;

    tutorialChapters.forEach(chapter => {
        const isActive = currentPage === chapter.file;
        const activeClass = isActive
            ? 'text-blue-400 font-medium'
            : 'text-slate-400 hover:text-white transition';

        html += `<li><a href="${chapter.file}" class="block ${activeClass}">${chapter.num}. ${chapter.title}</a></li>\n`;
    });

    html += `
            </ul>
        </div>
    `;

    sidebar.innerHTML = html;
}

// Run when DOM is ready
document.addEventListener('DOMContentLoaded', generateTutorialSidebar);
