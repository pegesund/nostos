# REPL Call Graph Design

This document describes how the Nostos REPL maintains type safety across function redefinitions using a call graph.

## The Problem

In a REPL, users can redefine functions at any time:

```
nostos> foo() = 42
nostos> bar() = foo() + 1
nostos> foo() = "hello"    // Type changed! What happens to bar?
```

In dynamically-typed languages, this "just works" until runtime when `"hello" + 1` fails. We want to catch this at definition time.

## Core Principle

**The REPL is always in a known state.** Every definition is either:
- **Valid**: Fully type-checked, compiled, callable
- **Broken**: Has type errors, cannot be called, errors are reported

There is no "maybe valid" or "check later" state.

## The Call Graph

The call graph tracks dependencies between definitions:

```
           dependents
    foo  ←───────────  bar  ←───────────  baz
     │                  │                  │
     │   dependencies   │   dependencies   │
     └────────────→    foo ←───────────────┘
                        ↑
                        └── baz also depends on foo transitively
```

### Data Structure

```rust
struct CallGraph {
    /// For each function, which functions call it?
    /// foo -> {bar, baz} means bar and baz depend on foo
    dependents: HashMap<String, HashSet<String>>,

    /// For each function, which functions does it call?
    /// bar -> {foo} means bar calls foo
    dependencies: HashMap<String, HashSet<String>>,
}
```

### Building the Graph

When a function is defined, we extract its dependencies from the AST:

```rust
fn extract_dependencies(ast: &Ast) -> HashSet<String> {
    // Walk the AST, collect all function references
    let mut deps = HashSet::new();
    visit_expr(ast, |expr| {
        if let Expr::Call(name, _) = expr {
            deps.insert(name.clone());
        }
        if let Expr::Var(name) = expr {
            // Could be a function reference
            deps.insert(name.clone());
        }
    });
    deps
}
```

### Updating the Graph

When `bar` is defined/redefined:

```rust
fn update_graph(name: &str, new_deps: HashSet<String>) {
    // Remove old dependency edges
    if let Some(old_deps) = graph.dependencies.get(name) {
        for dep in old_deps {
            graph.dependents.get_mut(dep).remove(name);
        }
    }

    // Add new dependency edges
    for dep in &new_deps {
        graph.dependents.entry(dep).or_default().insert(name);
    }
    graph.dependencies.insert(name, new_deps);
}
```

## Redefinition Algorithm

When a function is redefined:

```rust
fn redefine(name: &str, new_source: &str) -> Result<(), ReplError> {
    // 1. Parse and typecheck the new definition
    let ast = parse(new_source)?;
    let new_type = typecheck(&ast, &current_type_env)?;

    // 2. Check if type changed
    let old_type = types.get(name);
    let type_changed = old_type != Some(&new_type);

    // 3. Update this definition
    definitions.insert(name, Definition {
        source: new_source.to_string(),
        ast: ast.clone(),
        typ: new_type.clone(),
        compiled: Some(compile(&ast)?),
        errors: vec![],
    });

    // 4. Update call graph
    let deps = extract_dependencies(&ast);
    update_graph(name, deps);

    // 5. If type changed, re-typecheck all dependents
    if type_changed {
        recheck_dependents(name);
    }

    Ok(())
}
```

## Re-checking Dependents

When a function's type changes, all dependents must be re-typechecked:

```rust
fn recheck_dependents(changed: &str) {
    // Get all functions that depend on the changed function (transitively)
    let affected = transitive_dependents(changed);

    // Process in topological order (leaves first, or functions with no dependents)
    // This ensures we process bar before baz if baz depends on bar
    for name in topological_sort(&affected) {
        recheck_single(name);
    }

    // Report results
    let broken: Vec<_> = affected.iter()
        .filter(|n| !definitions[n].errors.is_empty())
        .collect();

    if !broken.is_empty() {
        report_broken_functions(&broken);
    }
}

fn recheck_single(name: &str) {
    let def = &definitions[name];

    // Try to typecheck with current type environment
    match typecheck(&def.ast, &current_type_env) {
        Ok(typ) => {
            // Still valid! Update type and recompile
            def.typ = typ;
            def.compiled = Some(compile(&def.ast));
            def.errors = vec![];
        }
        Err(errors) => {
            // Now broken
            def.compiled = None;
            def.errors = errors;
        }
    }
}
```

## Transitive Dependents

Finding all affected functions requires following the graph transitively:

```rust
fn transitive_dependents(name: &str) -> HashSet<String> {
    let mut result = HashSet::new();
    let mut queue = VecDeque::new();

    // Start with direct dependents
    if let Some(deps) = graph.dependents.get(name) {
        for dep in deps {
            queue.push_back(dep.clone());
        }
    }

    // BFS to find all transitive dependents
    while let Some(current) = queue.pop_front() {
        if result.insert(current.clone()) {
            if let Some(deps) = graph.dependents.get(&current) {
                for dep in deps {
                    queue.push_back(dep.clone());
                }
            }
        }
    }

    result
}
```

## Topological Sort for Re-checking

We must re-check in the right order. If `baz` depends on `bar`, we need to re-check `bar` first so `baz` sees `bar`'s new type (or knows `bar` is broken).

```rust
fn topological_sort(names: &HashSet<String>) -> Vec<String> {
    // Kahn's algorithm
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    let mut result = Vec::new();
    let mut queue = VecDeque::new();

    // Calculate in-degrees (within the affected set)
    for name in names {
        let deps = graph.dependencies.get(name).unwrap_or(&empty);
        let count = deps.iter().filter(|d| names.contains(*d)).count();
        in_degree.insert(name.clone(), count);
        if count == 0 {
            queue.push_back(name.clone());
        }
    }

    // Process
    while let Some(name) = queue.pop_front() {
        result.push(name.clone());
        if let Some(dependents) = graph.dependents.get(&name) {
            for dep in dependents {
                if names.contains(dep) {
                    let count = in_degree.get_mut(dep).unwrap();
                    *count -= 1;
                    if *count == 0 {
                        queue.push_back(dep.clone());
                    }
                }
            }
        }
    }

    result
}
```

## Tricky Parts

### 1. Cyclic Dependencies (Mutual Recursion)

```
isEven(n) = if n == 0 then true else isOdd(n - 1)
isOdd(n) = if n == 0 then false else isEven(n - 1)
```

These form a cycle in the call graph. Solutions:

**Option A: Require explicit mutual recursion blocks**
```
let rec
    isEven(n) = if n == 0 then true else isOdd(n - 1)
    isOdd(n) = if n == 0 then false else isEven(n - 1)
```
These are defined atomically - both or neither.

**Option B: Allow forward references**
The first definition of `isEven` sees `isOdd` as undefined. We could:
- Error: "isOdd is not defined"
- Allow it with an "incomplete" marker, resolved when `isOdd` is defined

**Option C: Two-phase definition**
```
nostos> isOdd : Int -> Bool              // Declare type first
nostos> isEven(n) = ... isOdd(n - 1) ... // Now this works
nostos> isOdd(n) = ... isEven(n - 1) ... // And this too
```

**Recommendation:** Start with Option A (explicit `let rec`), simplest to implement correctly.

### 2. Transitive Breakage Reporting

When `foo` changes and breaks `bar`, and `baz` depends on `bar`:

```
nostos> foo() = "hello"

2 functions have type errors:

  bar (depends on foo):
    foo() + 1
    ^^^^^^^^^ Cannot apply (+) to String and Int

  baz (depends on bar):
    bar() * 2
    ^^^^^ Cannot call 'bar': bar has type errors
```

The error for `baz` should be clear: it's not broken because of `foo` directly, but because `bar` is broken.

### 3. Self-Recursion

```
fact(n) = if n <= 1 then 1 else n * fact(n - 1)
```

When `fact` is being defined, it references itself. This is handled by:
1. First pass: infer type assuming `fact` has some type variable `?a`
2. Unify the recursive call with the inferred type
3. Result: `fact : Int -> Int`

When `fact` is redefined, it's its own dependent! We must:
- Update `fact` first
- Then re-check other dependents (not `fact` again)

### 4. Partial Breakage in a Chain

```
foo() = 42
bar() = foo()        // Just returns foo's result, no arithmetic
baz() = bar() + 1    // Adds 1

foo() = "hello"      // Type changes to String
```

Now:
- `bar` is still valid! It just returns whatever `foo` returns. `bar : () -> String`
- `baz` is broken: `String + Int`

The topological re-check handles this:
1. Re-check `bar` → succeeds with new type `() -> String`
2. Re-check `baz` → fails because `String + Int`

### 5. Type Inference Propagation

With Hindley-Milner type inference, types flow through the graph:

```
foo() = 42
bar(x) = foo() + x   // x : Int inferred from usage
baz() = bar(3.14)    // Error! bar expects Int

foo() = 3.14         // Now foo : () -> Float
// bar's inferred type changes to (Float -> Float)
// baz is now valid!
```

This means re-typechecking can **fix** previously broken code, not just break valid code.

### 6. Deleting a Function

When a function is deleted:

```rust
fn delete(name: &str) {
    // Find all dependents
    let affected = transitive_dependents(name);

    // Remove from definitions and graph
    definitions.remove(name);
    graph.dependencies.remove(name);
    graph.dependents.remove(name);
    // Also remove from others' dependent lists
    for (_, deps) in &mut graph.dependents {
        deps.remove(name);
    }

    // Re-check affected (they'll all fail since `name` is gone)
    for dep in affected {
        recheck_single(&dep);
    }

    report_broken_functions(&affected);
}
```

### 7. Shadowing vs Redefinition

Are these the same?

```
nostos> x = 42
nostos> x = "hello"   // Redefinition? Or new binding that shadows?
```

**Recommendation:** In REPL, treat as redefinition (updates the binding). Shadowing is a lexical scope concept that doesn't apply at top level.

### 8. Performance with Large Graphs

If the codebase grows large, re-typechecking all dependents could be slow.

**Optimizations:**
- Cache type information that didn't change
- Incremental type checking (only re-check what's affected)
- Lazy re-checking (mark as "needs recheck", do it on demand)

For a REPL, eager re-checking is probably fine - the graph won't be huge.

## REPL Session State

Complete state model:

```rust
struct ReplSession {
    /// All definitions (functions, values)
    definitions: HashMap<String, Definition>,

    /// The call graph
    call_graph: CallGraph,

    /// Global type environment (for type checking)
    type_env: TypeEnv,

    /// History for undo support
    history: Vec<Snapshot>,
}

struct Definition {
    /// Original source code
    source: String,

    /// Parsed AST
    ast: Ast,

    /// Inferred/declared type
    typ: Type,

    /// Compiled bytecode (None if broken)
    compiled: Option<CompiledCode>,

    /// Type errors (empty if valid)
    errors: Vec<TypeError>,
}

impl Definition {
    fn is_valid(&self) -> bool {
        self.errors.is_empty() && self.compiled.is_some()
    }
}
```

## User Commands

```
:graph              Show the call graph
:graph foo          Show what depends on foo
:deps foo           Show what foo depends on
:errors             Show all current type errors
:broken             List all broken functions
:valid              List all valid functions
:recheck            Force re-typecheck everything
:undo               Undo last definition/redefinition
```

## Example Session

```
nostos> add(x, y) = x + y
add : Int -> Int -> Int

nostos> double(x) = add(x, x)
double : Int -> Int

nostos> quad(x) = double(double(x))
quad : Int -> Int

nostos> :graph add
add is called by:
  - double
  - (transitively) quad

nostos> add(x, y) = x ++ y   // String concatenation

Type of add changed: (Int, Int) -> Int  →  (String, String) -> String

Re-checking dependents...
  ✗ double: add(x, x) - cannot unify Int with String
  ✗ quad: depends on broken 'double'

2 functions are now broken. Use :errors for details.

nostos> :errors
double (line 2):
  add(x, x)
  ^^^^^^^^^ Expected (String, String), got (Int, Int)

quad (line 3):
  double(double(x))
  ^^^^^^ Cannot call 'double': function has type errors

nostos> double(s) = add(s, s)   // Fix: now takes String
double : String -> String

Re-checking dependents...
  ✗ quad: double(double(x)) - cannot unify Int with String

1 function is still broken.

nostos> quad(s) = double(double(s))   // Fix: now takes String
quad : String -> String

All functions are valid.
```

## Summary

1. **Call graph** tracks who depends on whom
2. **On redefinition**, find all transitive dependents
3. **Re-typecheck** in topological order
4. **Report** which functions are now broken
5. **No late binding** - errors caught at definition time, not runtime
6. **User fixes** broken functions by redefining them
