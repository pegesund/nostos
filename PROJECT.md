# Nostos Project Structure

Nostos adopts a "directory-as-module" approach, similar to Java or Python, making project structure intuitive and implicit. There are no manifest files required to define modules; the file system structure *is* the module structure.

## Project Layout

A Nostos project is simply a directory containing `.nos` source files.

### Example Structure

```
my_project/
├── main.nos            # Defines module "main" (top-level)
├── config.nos          # Defines module "config"
├── utils/
│   └── math.nos        # Defines module "utils.math"
└── models/
    ├── user.nos        # Defines module "models.user"
    └── product.nos     # Defines module "models.product"
```

## Module Resolution

Modules are automatically named based on their path relative to the project root (the directory you run `nostos` on).

- `main.nos` -> Module `main`
- `utils/math.nos` -> Module `utils.math`
- `models/user.nos` -> Module `models.user`

## Importing and Using Modules

You use the `use` keyword to import functions or types from other modules.

### Importing Specific Items

```nostos
use utils.math.{add, subtract}
use models.user.{User, create_user}

main() = {
    u = create_user("Alice")
    res = add(10, 20)
}
```

### Implicit Qualified Access (Future)

Currently, explicit imports via `use` are required to bring names into scope. (Future versions may support fully qualified access like `utils.math.add(1, 2)` without explicit import).

## Running a Project

To run a project, pass the directory path to the Nostos CLI:

```bash
nostos my_project/
```

### Entry Point

Nostos looks for an entry point in the following order:
1.  `main.main` (Function `main` inside `main.nos`)
2.  `main` (Global function `main` in any top-level file, though `main.nos` is preferred)

If `main.nos` exists, it is treated as the root of the application.

## Circular Dependencies

Nostos supports circular dependencies between modules out of the box.

**How it works:**
Nostos uses a **Two-Pass Compilation** strategy to avoid the "partially initialized module" problems common in interpreted languages like Python.

1.  **Registration Pass:** The compiler scans *all* files in the project and registers every type, trait, and function signature into a global symbol table. No code is executed.
2.  **Compilation Pass:** Once all symbols are known, the compiler compiles the function bodies.

This means Module A can call a function in Module B, even if Module B calls a function in Module A. Since all function names are known before compilation starts, the order of files does not matter.

## Single File vs Project

Running a single file is treated as a "Project of one file".

```bash
nostos script.nos
```

This compiles `script.nos` as the root module (empty path `[]` or `main`) and runs it using the exact same multi-pass pipeline as a full project, ensuring consistent behavior.
