# Modules & Imports

Nostos uses a simple module system to organize code. Modules let you split your code into separate files and import functionality from the standard library or other modules.

## Importing from the Standard Library

The `use` keyword imports functions and types from modules. The standard library lives under the `stdlib` namespace.

```nostos
# Import specific items from stdlib.json
use stdlib.json.{jsonParse, jsonStringify}

main() = {
    data = %{"name": "Alice", "age": 30}
    json_str = jsonStringify(data)
    println(json_str)  # {"name":"Alice","age":30}

    parsed = jsonParse(json_str)
    println(parsed)
}
```

### Import Syntax Variations

```nostos
# Import specific items (recommended)
use stdlib.json.{jsonParse, jsonStringify}

# Import types alongside functions
use stdlib.json.{jsonParse, Null, Bool, Number, String, Array, Object}

# The Json type has these variants:
#   Null, Bool(Bool), Number(Float), String(String),
#   Array(List[Json]), Object(List[(String, Json)])
```

## Built-in Modules

Some modules are built into the runtime and don't require explicit imports:

| Module | Description |
|--------|-------------|
| Pg | PostgreSQL database access |
| Http | HTTP client for making requests |
| File | File system operations |
| String | String manipulation functions |
| List | List operations (map, filter, fold, etc.) |
| Map | Map operations (get, insert, keys, etc.) |

These built-in modules are accessed directly without import:

```nostos
# No import needed for built-in modules
main() = {
    # PostgreSQL
    conn = Pg.connect("host=localhost user=postgres")
    rows = Pg.query(conn, "SELECT * FROM users", [])
    Pg.close(conn)

    # HTTP requests
    response = Http.get("https://api.example.com/data")

    # File I/O
    content = File.read("config.txt")
    File.write("output.txt", "Hello!")

    # String operations
    upper = String.toUpper("hello")  # "HELLO"
    parts = String.split("a,b,c", ",")  # ["a", "b", "c"]
}
```

## Standard Library Modules

The `stdlib` namespace contains modules written in Nostos itself. These require explicit imports:

| Module | Provides |
|--------|----------|
| stdlib.json | JSON parsing and stringification |
| stdlib.list | Extended list operations |
| stdlib.io | I/O utilities |

## Example: Working with JSON

```nostos
# Import JSON parsing functions and type variants
use stdlib.json.{jsonParse, jsonStringify}

# Define a type that maps to JSON structure
type User = { name: String, age: Int, active: Bool }

main() = {
    # Parse JSON string
    json = jsonParse("{\"name\": \"Alice\", \"age\": 30, \"active\": true}")

    # Convert to typed value using jsonToType
    user = jsonToType[User](json)

    println("Name: " ++ user.name)    # "Name: Alice"
    println("Age: " ++ show(user.age)) # "Age: 30"

    # Modify and convert back to JSON
    updated = User("Alice", 31, true)
    json_out = jsonStringify(updated)
    println(json_out)  # {"name":"Alice","age":31,"active":true}
}
```

## Project Structure

For multi-file projects, Nostos uses a `nostos.toml` file to define project metadata:

```toml
# nostos.toml
[project]
name = "my_app"
version = "0.1.0"
```

Files in the same directory can reference each other. The runtime automatically discovers and compiles all `.nos` files in the project.

### Multiple Entry Points

Projects can define multiple executables using `[[bin]]` sections. Each entry specifies a name and an entry point in `module.function` format:

```toml
# nostos.toml
[project]
name = "my_app"
version = "0.1.0"

[[bin]]
name = "server"
entry = "server.main"

[[bin]]
name = "cli"
entry = "cli.main"
default = true
```

With this configuration:
- `nostos myproject/` runs the default entry point (`cli.main`)
- `nostos myproject/ --bin server` runs `server.main`
- `nostos myproject/ -b cli` runs `cli.main` (short form)

Each `[[bin]]` entry has these fields:
- `name` - The name used with `--bin` flag
- `entry` - Entry point as `module.function` (e.g., `server.main`)
- `default` - Optional, marks this as the default entry point

When a project has `[[bin]]` entries, the `main.nos` file is not required - the runtime uses the specified entry points instead.

### Best Practices

- Import only what you need with selective imports: `use module.{a, b}`
- Group related functions and types in the same file
- Use built-in modules (Pg, Http, File) directly without imports
- Check the stdlib source for available functions and types
