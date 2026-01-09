# Source Management Architecture

## Overview

This document describes the source code management system for Nostos projects. The system provides:

- **Memory as truth** during a session
- **Per-definition git history** via `.nostos/defs/`
- **Generated module files** for distribution/sharing
- **Incremental compilation** based on content hashing

## Directory Structure

```
/myproject/
  ├── nostos.toml              # Project marker, config
  ├── main.nos                 # Generated module files
  ├── utils/
  │   └── math.nos
  └── .nostos/
      ├── .git/                # Git repo for history
      └── defs/                # Individual definition files
          ├── main/
          │   ├── _imports.nos
          │   ├── main.nos
          │   └── helper.nos
          └── utils/
              └── math/
                  ├── _imports.nos
                  ├── add.nos      # All add overloads together
                  └── Point.nos    # Type definition
```

## File Naming Convention

In `.nostos/defs/<module>/`:

| Content | Filename | Example |
|---------|----------|---------|
| Function (all overloads) | `name.nos` | `add.nos` |
| Type | `TypeName.nos` | `Point.nos` |
| Trait | `TraitName.nos` | `Show.nos` |
| Variable | `var_name.nos` | `counter.nos` |
| Module imports | `_imports.nos` | `_imports.nos` |

Naming follows Nostos convention: lowercase = function/variable, uppercase = type/trait.

## nostos.toml

Minimal project configuration:

```toml
[project]
name = "myproject"
version = "0.1.0"

# Future: dependencies, build settings, etc.
```

## Lifecycle

### Load Project (`nostos myproject/`)

```
1. Check for nostos.toml
   - If missing: create with defaults
2. Check for .nostos/
   - If missing: create and git init
3. Load definitions from .nostos/defs/
   - If empty: bootstrap from *.nos files in project
4. Compile all definitions
5. Start TUI
```

### Bootstrap (First Load of Existing Project)

When `.nostos/` doesn't exist but `*.nos` files do:

```
1. Create nostos.toml
2. Create .nostos/ and git init
3. Scan for *.nos files recursively
4. Parse each file, extract definitions
5. Write each definition to .nostos/defs/<module>/<name>.nos
6. Extract imports, write to .nostos/defs/<module>/_imports.nos
7. Git commit: "Import from source files"
```

### Edit + Save (Ctrl+S in TUI Editor)

```
1. Parse editor content
2. For each definition in editor:
   a. Update in-memory representation
   b. Compute content hash
   c. If changed:
      - Write to .nostos/defs/<module>/<name>.nos
      - Mark dependents for recompilation
3. Git add + commit changed files
4. Recompile dirty definitions
```

Main `*.nos` files are NOT touched on Ctrl+S.

### Write (`:w` command)

```
1. For each module with definitions:
   a. Collect all definitions for that module
   b. Generate module file content:
      - Imports first (from _imports.nos)
      - Types
      - Traits (with implementations)
      - Functions (grouped by base name, overloads together)
      - Variables
   c. Write to <module_path>.nos
   d. Create directories as needed
2. Clear "needs write" flags
```

### Delete (`:delete name`)

```
1. Find definition by name
   - :delete add       → delete all add overloads
   - :delete add/Int,Int → delete specific overload
2. Remove from memory
3. Delete .nostos/defs/<module>/<name>.nos
   - If last overload, delete file
   - If specific overload, rewrite file without it
4. Git commit deletion
5. Mark dependents for recompilation (will error if unresolved)
```

### Move (`:move name path`)

```
1. Find definition in memory
2. Update module path
3. Move file in .nostos/defs/
   - Git tracks as rename, preserves history
4. Git commit
```

## Data Structures

### SourceManager

```rust
pub struct SourceManager {
    /// Project root directory
    project_root: PathBuf,

    /// All modules
    modules: HashMap<ModulePath, Module>,

    /// Quick lookup: definition name → module
    def_index: HashMap<QualifiedName, ModulePath>,

    /// Definitions created in REPL (no module yet)
    repl_defs: HashMap<String, Definition>,
}

pub type ModulePath = Vec<String>;  // e.g., ["utils", "math"]
```

### Module

```rust
pub struct Module {
    /// Path like ["utils", "math"]
    path: ModulePath,

    /// Import statements for this module
    imports: Vec<String>,

    /// All definitions in this module
    definitions: HashMap<String, DefinitionGroup>,

    /// Needs to be written to main file
    dirty: bool,
}
```

### DefinitionGroup

Groups overloaded functions together:

```rust
pub struct DefinitionGroup {
    /// Base name (e.g., "add")
    name: String,

    /// Kind (Function, Type, Trait, Variable)
    kind: DefKind,

    /// All overloads (for functions) or single definition
    overloads: Vec<Definition>,

    /// Any overload changed since last git commit
    git_dirty: bool,
}
```

### Definition

```rust
pub struct Definition {
    /// Full source text
    source: String,

    /// Content hash for change detection
    content_hash: u64,

    /// Signature (for functions with overloads)
    signature: Option<String>,
}

pub enum DefKind {
    Function,
    Type,
    Trait,
    Variable,
}
```

## Definition File Format

### Functions (with overloads)

`.nostos/defs/utils/math/add.nos`:
```
add(x: Int, y: Int) = x + y

add(x: Float, y: Float) = x + y

add(x: String, y: String) = x ++ y
```

Blank line separates overloads.

### Types

`.nostos/defs/utils/math/Point.nos`:
```
type Point = { x: Int, y: Int }
```

### Types with Trait Implementations

`.nostos/defs/utils/math/Point.nos`:
```
type Point = { x: Int, y: Int } deriving Eq, Hash

# Show implementation
Point.show(self) = "Point(" ++ self.x.show() ++ ", " ++ self.y.show() ++ ")"
```

### Traits

`.nostos/defs/core/Show.nos`:
```
trait Show
    show(self) -> String
end
```

### Variables

`.nostos/defs/main/counter.nos`:
```
var counter = 0
```

### Module Imports

`.nostos/defs/utils/math/_imports.nos`:
```
use core.Show
use utils.helpers
```

## Generated Module Files

When `:w` is executed, module files are generated:

`utils/math.nos`:
```
use core.Show
use utils.helpers

type Point = { x: Int, y: Int } deriving Eq, Hash

Point.show(self) = "Point(" ++ self.x.show() ++ ", " ++ self.y.show() ++ ")"

trait Numeric
    add(self, other: Self) -> Self
end

add(x: Int, y: Int) = x + y

add(x: Float, y: Float) = x + y

add(x: String, y: String) = x ++ y

var defaultPoint = Point(0, 0)
```

Order: imports, types, traits, functions, variables.

## Conflict Resolution

On load, if both `.nostos/defs/` and main `*.nos` files exist:

- **`.nostos/defs/` wins** (it has git history, is the source of truth)
- Main files are regenerated on `:w`

If `.nostos/` is missing but `*.nos` files exist:
- Bootstrap from `*.nos` files
- This is the "import existing project" flow

## Git Integration

### Auto-commit on Save

Every Ctrl+S in editor:
```bash
cd .nostos
git add defs/<module>/<name>.nos
git commit -m "Update <module>.<name>"
```

### Commit Messages

- Save: `"Update utils.math.add"`
- Delete: `"Delete utils.math.add"`
- Move: `"Move add from utils.math to utils.arithmetic"`
- Bootstrap: `"Import from source files"`

### History Benefits

- `git log .nostos/defs/utils/math/add.nos` shows function history
- `git blame` works per-definition
- Easy to revert individual definitions
- Branch/merge at definition level

## REPL Definitions

Definitions created in REPL (not from a file):

```
nos> double(x: Int) = x * 2
```

1. Stored in `repl_defs` HashMap
2. Not written to disk until explicitly placed
3. Commands to place:
   - `:move double utils.math` → moves to module
   - `:w` → prompts for location if repl_defs not empty

## Implementation Plan

### Phase 1: SourceManager Core
- [ ] Create `crates/source/` crate
- [ ] Implement `SourceManager` struct
- [ ] Implement loading from `.nostos/defs/`
- [ ] Implement bootstrap from `*.nos` files

### Phase 2: Integration with Compiler
- [ ] Modify compiler to accept definitions from SourceManager
- [ ] Implement content hashing
- [ ] Implement incremental compilation (only dirty defs)

### Phase 3: TUI Integration
- [ ] Modify `:edit` to pull from SourceManager
- [ ] Implement save flow (Ctrl+S → .nostos/defs/)
- [ ] Implement `:w` to generate module files

### Phase 4: Git Integration
- [ ] Auto-init git in .nostos/
- [ ] Auto-commit on definition changes
- [ ] Implement `:delete` and `:move` commands

### Phase 5: Polish
- [ ] Handle REPL definitions
- [ ] Error recovery (partial compilation)
- [ ] nostos.toml configuration options
