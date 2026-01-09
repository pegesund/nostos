# Nostos TUI

The Nostos TUI (Terminal User Interface) provides an interactive development environment within your terminal. It combines a REPL console, code browser, multiple editors, and inspectors in a tile-based workspace.

## Starting the TUI

```bash
# Start with no files
nostos tui

# Load a directory of source files
nostos tui ./src

# Load specific files
nostos tui main.nos lib.nos
```

## Layout

The TUI uses a tile-based workspace:
- **Console** - Shows REPL output and accepts commands
- **Editors** - Code editors for functions/types (opened via browser or `:edit`)
- **REPL Panels** - Independent REPL instances for experimentation

Layout automatically adjusts:
- 1-3 windows: Single row, equal width
- 4-6 windows: Two rows (3 on top, rest on bottom)

## Keyboard Shortcuts

### Global

| Key | Action |
|-----|--------|
| `Shift+Tab` | Cycle between windows |
| `Ctrl+B` | Open code browser |
| `Ctrl+R` | Open new REPL panel |
| `Ctrl+Y` | Copy focused window content to clipboard |

### Editor Windows

| Key | Action |
|-----|--------|
| `Ctrl+S` | Save changes |
| `Ctrl+W` | Close editor |
| `Ctrl+Y` | Copy editor content |
| `Esc` | Close editor |

### Browser Dialog

| Key | Action |
|-----|--------|
| `Enter` | Open selected item / navigate into module |
| `Backspace` | Go up one level |
| `Esc` | Close browser |
| `/` | Focus search/filter |

## Commands

Commands are entered in the input field at the bottom of the Console. All commands start with `:`.

### Navigation & Exploration

| Command | Description |
|---------|-------------|
| `:browse` or `:b` | Open the code browser (or press `Ctrl+B`) |
| `:edit <name>` or `:e <name>` | Open editor for function/type |
| `:info <name>` | Show info about a function/type |
| `:view <name>` | Show source code |
| `:type <expr>` | Show the type of an expression |

### Dependencies

| Command | Description |
|---------|-------------|
| `:deps <name>` | Show what a function depends on |
| `:rdeps <name>` | Show what depends on a function (reverse dependencies) |

### File Operations

| Command | Description |
|---------|-------------|
| `:load <file>` | Load a source file |
| `:reload` | Reload previously loaded files |

### Clipboard

| Command | Description |
|---------|-------------|
| `:copy` or `:cp` | Copy focused window content to clipboard |

### Session

| Command | Description |
|---------|-------------|
| `:quit`, `:q`, or `:exit` | Exit the TUI |
| `:help` | Show available commands |
| `:debug` | Show debug info |

## Code Browser

The browser (`Ctrl+B` or `:browse`) provides hierarchical navigation of your codebase:

- **Modules** - Navigate into submodules
- **Functions** - View and edit function definitions
- **Types** - View type definitions
- **Traits** - View trait definitions

### Status Indicators

The browser shows compile status for each item:
- No indicator = compiles successfully
- Red indicators = compile errors
- Items with errors can still be edited

### Filtering

Type in the search field to filter items. The filter matches:
- Item names
- Partial matches

## Editors

Editors provide syntax-highlighted editing with:
- Real-time syntax highlighting
- Auto-indentation
- Error display on save
- Integration with the VM (saves compile to the running environment)

### Saving

When you save (`Ctrl+S`):
1. Code is compiled
2. Errors are displayed if any
3. On success, the function/type is updated in the running VM
4. Other windows reflect the changes immediately

## REPL Panels

Open additional REPL panels (`Ctrl+R`) for:
- Isolated experimentation
- Side-by-side testing
- Running code while editing

Each REPL panel has its own input field but shares the same VM state.

## Workflow Tips

### Iterative Development

1. Start TUI with your project: `nostos tui ./src`
2. Open browser (`Ctrl+B`) to explore code
3. Select a function to edit
4. Make changes and save (`Ctrl+S`)
5. Test in Console or a REPL panel
6. Repeat

### Multi-Window Setup

1. Open several functions with `Ctrl+B` → select item
2. Use `Shift+Tab` to cycle between them
3. Edit and save each as needed
4. Use Console to test interactions

### Exploring Dependencies

1. Use `:deps fib` to see what `fib` calls
2. Use `:rdeps helper` to find all callers of `helper`
3. Open callers in editors to understand usage

## Text Selection & System Clipboard

The TUI supports normal terminal text selection. You can:
- Select text with your mouse
- Copy using your terminal's copy function
- Use `Ctrl+Y` to copy entire window contents to system clipboard

Note: Mouse capture is disabled to allow normal terminal selection.
