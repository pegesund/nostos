# Nostos Vision: A Language You Can See Through

## Philosophy

Nostos combines:
- **Haskell's introspection** - Rich type info, signatures, browsing definitions
- **Forth/Lisp's liveness** - Everything redefinable, the system is always running
- **Smalltalk's transparency** - View any code, edit it live, inspect any value

The REPL is not a testing tool - it IS the development environment. You sculpt a running system.

## Core Principles

1. **Everything is inspectable** - Any value, function, type can be examined
2. **Everything is modifiable** - See something? Change it. It's live immediately.
3. **The system is always live** - No compile-run-debug cycle. Just a running image.
4. **Code is data you can browse** - View source, edit inline, see dependencies

## The Development Flow

Instead of: write -> compile -> run -> debug -> repeat

It's: **sculpt the running system**

```
λ> :load myapp.nos          # Merge definitions into live image
λ> main()                   # Run it, see issue
λ> :view handle_request     # Look at the code
λ> handle_request(req) = {  # Redefine it - live!
|    ...fixed version...
| }
λ> main()                   # Test again, already using new version
λ> :save myapp.nos          # Persist when happy
```

## REPL Commands

### Loading & Saving
| Command | Description |
|---------|-------------|
| `:load <file>` | Merge file's definitions into image |
| `:load <dir>` | Load all .nos files from directory |
| `:reload` | Re-read all loaded source files |
| `:save <file>` | Save definitions back to file |
| `:reset` | Clear everything, reload stdlib |

### Introspection
| Command | Description |
|---------|-------------|
| `:browse` | List all definitions in image |
| `:browse <pattern>` | Filter definitions by pattern |
| `:info <name>` | Show detailed info (signature, source location, doc) |
| `:type <expr>` | Show type of expression |
| `:view <name>` | Show source code of definition |
| `:deps <name>` | Show what this function calls |
| `:rdeps <name>` | Show what calls this function (reverse deps) |

### Editing
| Command | Description |
|---------|-------------|
| `:edit <name>` | Open definition in $EDITOR, auto-reload on save |
| `:undef <name>` | Remove a definition |

### Debugging
| Command | Description |
|---------|-------------|
| `:trace <name>` | Watch all calls to function |
| `:untrace <name>` | Stop watching |
| `:time <expr>` | Time expression evaluation |
| `:inspect <value>` | Deep inspect a value |

### Process Inspection (for concurrent programs)
| Command | Description |
|---------|-------------|
| `:processes` | List running processes |
| `:inspect <pid>` | Show process state, mailbox |
| `:send <pid> <msg>` | Inject message into process |
| `:kill <pid>` | Terminate process |

### Utility
| Command | Description |
|---------|-------------|
| `:help` | Show help |
| `:quit` | Exit |
| `:cd <dir>` | Change directory |
| `:pwd` | Show current directory |
| `:! <cmd>` | Run shell command |
| `:set <opt>` | Set option (+t show types, +s stats) |

## Example Sessions

### Exploring a Module
```
λ> :load stdlib/list.nos
Loaded 24 definitions from stdlib/list.nos

λ> :browse
-- Functions (24) --
  map       : (a -> b) -> [a] -> [b]
  filter    : (a -> Bool) -> [a] -> [a]
  fold      : (b -> a -> b) -> b -> [a] -> b
  head      : [a] -> a
  tail      : [a] -> [a]
  ...

λ> :info map
map : (a -> b) -> [a] -> [b]
  Defined in: stdlib/list.nos:15
  Calls: map (recursive)
  Called by: (nothing loaded)

λ> :view map
map(f, []) = []
map(f, [x | xs]) = [f(x) | map(f, xs)]
```

### Live Development
```
λ> double(x) = x * 2
double : Int -> Int

λ> triple(x) = double(x) + x
triple : Int -> Int

λ> triple(10)
30

λ> double(x) = x * 3   # Oops, redefine
double : Int -> Int (redefined)

λ> triple(10)          # Uses new double!
40

λ> :deps triple
triple calls: double

λ> :rdeps double
double is called by: triple
```

### Inspecting Values
```
λ> p = Point(3, 4)
λ> p
Point(3, 4)

λ> :inspect p
Point(3, 4) : Point
  .0 = 3 : Int
  .1 = 4 : Int
  hash = 5400308442836416362

λ> users = %{"alice" => 1, "bob" => 2}
λ> :inspect users
Map[String, Int] (2 entries)
  "alice" => 1
  "bob" => 2
```

### Process Debugging
```
λ> :load server.nos
λ> pid = spawn(server_loop, [8080])
<0.5>

λ> :processes
PID     STATUS    MAILBOX  FUNCTION
<0.1>   running   0        main
<0.5>   waiting   0        server_loop

λ> :send <0.5> {request, "GET", "/"}
Sent.

λ> :inspect <0.5>
Process <0.5>
  status: waiting
  mailbox: [{request, "GET", "/"}]
  function: server_loop
```

---

## Infrastructure Requirements

### What We Need to Store During Compilation

1. **Source code preservation**
   - Keep original source text for each definition
   - Map definitions back to source file and line number

2. **Type signatures**
   - Infer and store function signatures
   - Parameter types, return type
   - Type parameter bounds

3. **Module registry**
   - Track which file each definition came from
   - Support "virtual" REPL module for interactive definitions

4. **Documentation**
   - Extract doc comments (to be designed)
   - Associate with definitions

5. **Dependency graph**
   - Already have CallGraph in repl crate
   - Need reverse dependency lookup

### What We Need for Eval

1. **Incremental compilation**
   - Compile single function without full module recompile
   - Generate synthetic source location for REPL input

2. **Hot-swap in VM**
   - Replace function in running VM
   - All future calls use new version

3. **Expression evaluation**
   - Compile expression to temporary function
   - Run and display result
   - Handle values that need pretty-printing

### Current Infrastructure Audit

**Already Have:**
- `FunctionValue` has: name, arity, param_names, module, source_span, debug_symbols
- `TypeValue` has: name, kind, fields, constructors, traits
- `Compiler` has: types, trait_defs, trait_impls, fn_asts (!)
- `ReplSession` has: definitions with source, call graph

**Missing:**
- Type signatures as displayable strings
- Source code stored with compiled functions
- Doc comment extraction
- Reverse dependency lookup (easy to add)
- Type inference results accessible
- Trait instance listing per type

---

## Implementation Phases

### Phase 1: Basic REPL Loop
- [ ] `nostos repl` CLI command
- [ ] Command parser (`:cmd args`)
- [ ] Expression evaluation (wrap in temp fn, run, display)
- [ ] Function definition (parse, compile, register)
- [ ] `:help`, `:quit`
- [ ] `:load <file>` - merge definitions
- [ ] `:browse` - list all definitions
- [ ] Simple prompt

### Phase 2: Introspection Foundation
- [ ] Store source code with FunctionValue
- [ ] Generate displayable type signatures
- [ ] `:info <name>` - show signature, location
- [ ] `:view <name>` - show source
- [ ] `:type <expr>` - show expression type
- [ ] `:deps` and `:rdeps`

### Phase 3: Live Editing
- [ ] `:edit <name>` - open in $EDITOR
- [ ] Watch for file save, auto-reload
- [ ] `:reload` - re-read source files
- [ ] `:save <file>` - persist definitions
- [ ] `:undef <name>` - remove definition

### Phase 4: Deep Inspection
- [ ] `:inspect` for values
- [ ] Pretty-print complex types (lists, maps, records)
- [ ] Show type, hash, structure
- [ ] `:trace` / `:untrace`
- [ ] `:time` expression

### Phase 5: Process Debugging
- [ ] `:processes` list
- [ ] `:inspect <pid>`
- [ ] `:send <pid> <msg>`
- [ ] `:kill <pid>`

### Phase 6: Polish
- [ ] Readline integration (history, completion)
- [ ] Colored output
- [ ] Multiline input (`:{ ... :}` or auto-detect)
- [ ] Tab completion for commands and names
- [ ] Persistent history (~/.nostos_history)

### Future: GUI
- Smalltalk-style browser
- Click to edit
- Visual process inspector
- But REPL-first - GUI builds on REPL infrastructure
