// Shared tutorial sidebar component
const tutorialChapters = [
    { num: 0, file: "00_installation.html", title: "Installation" },
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
    { num: 22, file: "22_templates.html", title: "Templates & Metaprogramming" },
    { num: 23, file: "23_complete_web_app.html", title: "Complete Web App" },
    { num: 24, file: "24_ffi.html", title: "FFI & Extensions" },
    { num: 25, file: "25_command_line.html", title: "Command Line" },
    { num: 26, file: "26_reactive_records.html", title: "Reactive Records" },
    { num: 27, file: "27_reactive_web.html", title: "Reactive Web (RWeb)" },
    { num: 28, file: "28_logging.html", title: "Logging" },
    { num: 29, file: "29_selenium.html", title: "Selenium WebDriver" },
    { num: 30, file: "30_tcp_sockets.html", title: "TCP Sockets" },
    { num: 31, file: "31_vscode.html", title: "VS Code Integration" },
];

function generateTutorialSidebar() {
    const currentPage = window.location.pathname.split('/').pop();
    const sidebar = document.getElementById('tutorial-sidebar');
    if (!sidebar) return;

    let html = '<div class="p-6"><h3 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">Contents</h3><ul class="space-y-2">';
    tutorialChapters.forEach(chapter => {
        const isActive = currentPage === chapter.file;
        const cls = isActive ? 'text-blue-400 font-medium' : 'text-slate-400 hover:text-white transition';
        html += `<li><a href="${chapter.file}" class="block ${cls}">${chapter.num}. ${chapter.title}</a></li>`;
    });
    html += '</ul></div>';
    sidebar.innerHTML = html;
}

function generateMobileChapters() {
    const currentPage = window.location.pathname.split('/').pop();
    const container = document.getElementById('mobile-chapters');
    if (!container) return;
    if (container.children.length > 0) return; // Already done

    tutorialChapters.forEach(chapter => {
        const isActive = currentPage === chapter.file;
        const a = document.createElement('a');
        a.href = chapter.file;
        a.className = isActive ? 'block text-blue-400 font-medium py-1 text-sm' : 'block text-slate-300 hover:text-white transition py-1 text-sm';
        a.textContent = chapter.num + '. ' + chapter.title;
        container.appendChild(a);
    });
}

function init() {
    generateTutorialSidebar();
    generateMobileChapters();
}

// Run on multiple events to ensure it works
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
window.addEventListener('load', init);
