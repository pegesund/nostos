# REPL Infrastructure Requirements

## Current State Audit

### AST Already Has (syntax/ast.rs)

**FnDef:**
- `doc: Option<String>` - Doc comments
- `type_params: Vec<TypeParam>` - Generic parameters with bounds
- `clauses[].return_type: Option<TypeExpr>` - Return type annotation
- `clauses[].params[].ty: Option<TypeExpr>` - Parameter type annotations

**TypeDef:**
- `doc: Option<String>` - Doc comments
- `type_params: Vec<TypeParam>` - Generic parameters
- `deriving: Vec<Ident>` - Derived traits

**TypeParam:**
- `name: Ident`
- `constraints: Vec<Ident>` - Trait bounds

### FunctionValue (vm/value.rs)

```rust
pub struct FunctionValue {
    pub name: String,
    pub arity: usize,
    pub param_names: Vec<String>,      // Have param names
    pub code: Arc<Chunk>,
    pub module: Option<String>,         // Have module
    pub source_span: Option<(usize, usize)>,  // Have span
    pub jit_code: Option<JitFunction>,
    pub call_count: AtomicU32,
    pub debug_symbols: Vec<LocalVarSymbol>,
}
```

**Missing for REPL:**
- `source_code: Option<Arc<String>>` - Original source code
- `source_file: Option<PathBuf>` - Path to source file
- `signature: Option<String>` - Pretty-printed type signature "Int -> Int -> Int"
- `doc: Option<String>` - Doc comment from AST
- `param_types: Vec<Option<String>>` - Parameter type names
- `return_type: Option<String>` - Return type name

### TypeValue (vm/value.rs)

```rust
pub struct TypeValue {
    pub name: String,
    pub kind: TypeKind,
    pub fields: Vec<FieldInfo>,
    pub constructors: Vec<ConstructorInfo>,
    pub traits: Vec<String>,  // Derived traits
}
```

**Missing for REPL:**
- `source_code: Option<String>` - Original type definition source
- `source_file: Option<PathBuf>` - Path to source file
- `doc: Option<String>` - Doc comment
- `type_params: Vec<String>` - Generic parameters like [T, E]

### Compiler Already Has (compiler/compile.rs)

```rust
trait_defs: HashMap<String, TraitInfo>,
trait_impls: HashMap<(String, String), TraitImplInfo>,
type_traits: HashMap<String, Vec<String>>,
fn_asts: HashMap<String, FnDef>,  // Have ASTs! Can reconstruct source
fn_type_params: HashMap<String, Vec<TypeParam>>,  // Have type params!
```

### ReplSession Already Has (repl/session.rs)

```rust
pub struct Definition {
    pub source: String,        // Original source!
    pub fn_def: FnDef,         // Full AST!
    pub dependencies: HashSet<String>,
}
```

---

## Gap Analysis

### What's Good
1. AST has all the info (doc, types, params)
2. Compiler stores ASTs in `fn_asts`
3. ReplSession stores source with definitions
4. FunctionValue has basic metadata (name, arity, module, span)

### What's Missing

1. **Source code doesn't reach FunctionValue** - Compiler has AST but doesn't store source in the compiled function

2. **Type annotations aren't propagated** - AST has types, but FunctionValue doesn't get them

3. **Doc comments aren't propagated** - AST has doc, but FunctionValue doesn't get it

4. **No unified "image" view** - Functions from files vs REPL are tracked differently

5. **Reverse dependencies** - CallGraph has forward deps, need reverse lookup

6. **No type signature stringification** - Can't turn `TypeExpr` into displayable "Int -> Bool"

---

## Implementation Plan

### Step 1: Extend FunctionValue

Add new fields to `FunctionValue`:

```rust
pub struct FunctionValue {
    // ... existing fields ...

    // NEW: For REPL introspection
    pub source_code: Option<Arc<String>>,  // The actual source text
    pub source_file: Option<String>,       // File path or "<repl>"
    pub doc: Option<String>,               // Doc comment
    pub param_types: Vec<String>,          // ["Int", "String", ""]
    pub return_type: Option<String>,       // "Bool"
}
```

### Step 2: Extend TypeValue

```rust
pub struct TypeValue {
    // ... existing fields ...

    pub source_code: Option<String>,
    pub source_file: Option<String>,
    pub doc: Option<String>,
    pub type_params: Vec<String>,  // ["T", "E"]
}
```

### Step 3: TypeExpr to String

Add helper to stringify type expressions:

```rust
// In syntax or compiler crate
pub fn type_expr_to_string(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Name(ident) => ident.node.clone(),
        TypeExpr::Generic(name, args) => {
            let args_str = args.iter()
                .map(type_expr_to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}[{}]", name.node, args_str)
        }
        TypeExpr::Function(params, ret) => {
            let params_str = params.iter()
                .map(type_expr_to_string)
                .collect::<Vec<_>>()
                .join(" -> ");
            format!("{} -> {}", params_str, type_expr_to_string(ret))
        }
        TypeExpr::Tuple(elems) => {
            let elems_str = elems.iter()
                .map(type_expr_to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({})", elems_str)
        }
        TypeExpr::List(elem) => format!("[{}]", type_expr_to_string(elem)),
        TypeExpr::Map(k, v) => format!("Map[{}, {}]",
            type_expr_to_string(k), type_expr_to_string(v)),
        TypeExpr::Set(elem) => format!("Set[{}]", type_expr_to_string(elem)),
    }
}
```

### Step 4: Generate Signature String

```rust
pub fn function_signature(fn_def: &FnDef) -> String {
    let clause = &fn_def.clauses[0]; // Use first clause
    let param_types: Vec<String> = clause.params.iter()
        .map(|p| p.ty.as_ref()
            .map(type_expr_to_string)
            .unwrap_or_else(|| "?".to_string()))
        .collect();

    let ret_type = clause.return_type.as_ref()
        .map(type_expr_to_string)
        .unwrap_or_else(|| "?".to_string());

    if param_types.is_empty() {
        ret_type
    } else {
        format!("{} -> {}", param_types.join(" -> "), ret_type)
    }
}
```

### Step 5: Propagate Through Compilation

In compiler, when creating FunctionValue:

```rust
fn compile_function(&mut self, fn_def: &FnDef, source: &str) -> FunctionValue {
    // ... existing compilation ...

    FunctionValue {
        name: fn_def.name.node.clone(),
        arity,
        param_names,
        code: Arc::new(chunk),
        module: self.current_module(),
        source_span: Some((fn_def.span.start, fn_def.span.end)),

        // NEW: Propagate from AST
        source_code: Some(Arc::new(extract_source(source, fn_def.span))),
        source_file: self.current_source_name.clone(),
        doc: fn_def.doc.clone(),
        param_types: extract_param_types(fn_def),
        return_type: extract_return_type(fn_def),

        // ... rest ...
    }
}
```

### Step 6: Reverse Dependencies

Add to CallGraph:

```rust
impl CallGraph {
    /// Get functions that call the given function
    pub fn callers(&self, name: &str) -> HashSet<String> {
        self.edges.iter()
            .filter(|(_, deps)| deps.contains(name))
            .map(|(caller, _)| caller.clone())
            .collect()
    }
}
```

### Step 7: Add Compiler Query Methods

```rust
impl Compiler {
    /// Get displayable signature for a function
    pub fn get_signature(&self, name: &str) -> Option<String> {
        self.fn_asts.get(name).map(function_signature)
    }

    /// Get source code for a function
    pub fn get_source(&self, name: &str) -> Option<String> {
        self.functions.get(name)
            .and_then(|f| f.source_code.as_ref())
            .map(|s| s.to_string())
    }

    /// Get doc comment for a function
    pub fn get_doc(&self, name: &str) -> Option<String> {
        self.fn_asts.get(name).and_then(|f| f.doc.clone())
    }

    /// Get all traits for a type
    pub fn get_type_traits(&self, type_name: &str) -> Vec<String> {
        self.type_traits.get(type_name).cloned().unwrap_or_default()
    }

    /// Get all types implementing a trait
    pub fn get_trait_implementors(&self, trait_name: &str) -> Vec<String> {
        self.trait_impls.iter()
            .filter(|((_, t), _)| t == trait_name)
            .map(|((ty, _), _)| ty.clone())
            .collect()
    }
}
```

---

## Testing Strategy

### Unit Tests for Infrastructure

```rust
#[test]
fn test_function_has_source() {
    let source = "double(x) = x * 2";
    let compiler = compile_source(source);
    let f = compiler.get_function("double").unwrap();
    assert!(f.source_code.is_some());
    assert!(f.source_code.as_ref().unwrap().contains("x * 2"));
}

#[test]
fn test_function_has_types() {
    let source = "add(x: Int, y: Int) -> Int = x + y";
    let compiler = compile_source(source);
    let f = compiler.get_function("add").unwrap();
    assert_eq!(f.param_types, vec!["Int", "Int"]);
    assert_eq!(f.return_type, Some("Int".to_string()));
}

#[test]
fn test_signature_generation() {
    let source = "add(x: Int, y: Int) -> Int = x + y";
    let compiler = compile_source(source);
    assert_eq!(compiler.get_signature("add"), Some("Int -> Int -> Int".to_string()));
}

#[test]
fn test_doc_propagation() {
    let source = r#"
    /// Doubles a number
    double(x: Int) -> Int = x * 2
    "#;
    let compiler = compile_source(source);
    let f = compiler.get_function("double").unwrap();
    assert_eq!(f.doc, Some("Doubles a number".to_string()));
}

#[test]
fn test_reverse_deps() {
    let source = r#"
    double(x) = x * 2
    quadruple(x) = double(double(x))
    "#;
    let compiler = compile_source(source);
    let callers = compiler.get_callers("double");
    assert!(callers.contains("quadruple"));
}
```

---

## Additional Work: Doc Comments

**Current state:** Comments starting with `#` are skipped in lexer.

**Needed:**
1. Add `##` or `///` as doc comment syntax
2. Lexer emits `Token::DocComment(String)` instead of skipping
3. Parser attaches doc to following definition
4. Propagate through AST and compilation

**Syntax options:**
```
## This is a doc comment for the following definition
double(x) = x * 2

# This is a regular comment (ignored)
```

Or use `///` like Rust:
```
/// This is a doc comment
double(x) = x * 2
```

**Decision:** Use `##` for consistency with `#` comments.

---

## Summary

**Effort estimate:**
- Step 1-2 (extend structs): Small
- Step 3-4 (type stringification): Medium
- Step 5 (propagation): Medium
- Step 6-7 (query methods): Small
- Doc comment parsing: Medium (separate task)

**Total: ~1-2 sessions of work**

Once this is done, we have all the infrastructure for:
- `:info` - signature, doc, location
- `:view` - source code
- `:type` - expression types (via temp compilation)
- `:deps` / `:rdeps` - dependency tracking
- `:browse` - list with signatures
