# Type System Design: Inference + Monomorphization

## 1. Type Representation

```rust
// crates/types/src/lib.rs

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    // Concrete types
    Int,
    Float,
    Bool,
    String,
    Unit,

    // Compound types
    List(Box<Type>),
    Tuple(Vec<Type>),
    Function(Vec<Type>, Box<Type>),  // args -> return

    // User-defined
    Named(String, Vec<Type>),  // e.g., Option[Int], Result[String, Error]

    // For inference
    Var(TypeVarId),  // Unknown type, to be solved

    // Constraints
    Constrained(TypeVarId, Vec<Constraint>),  // T where T: Num
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Constraint {
    Num,      // Int | Float
    Eq,       // Types that support ==
    Show,     // Types that support show()
    Any,      // No constraint (fully polymorphic)
}

pub type TypeVarId = u32;
```

## 2. Syntax Extensions

```
// Type annotations (optional - inferred if omitted)
double(x: Num) -> Num = x + x
identity(x: a) -> a = x
map(f: a -> b, xs: List[a]) -> List[b] = ...

// Explicit type application (rarely needed)
identity@Int(42)

// Type aliases
type Point = (Float, Float)
type Result[T] = Ok(T) | Err(String)
```

## 3. Type Inference Algorithm

### Phase 1: Constraint Generation

```rust
struct TypeChecker {
    next_var: TypeVarId,
    constraints: Vec<(Type, Type)>,  // equality constraints
    var_types: HashMap<String, Type>,
    func_types: HashMap<String, FunctionType>,
}

struct FunctionType {
    type_params: Vec<(TypeVarId, Vec<Constraint>)>,  // generic params with bounds
    param_types: Vec<Type>,
    return_type: Type,
}

impl TypeChecker {
    fn fresh_var(&mut self) -> Type {
        let id = self.next_var;
        self.next_var += 1;
        Type::Var(id)
    }

    fn infer_expr(&mut self, expr: &Expr) -> Type {
        match expr {
            // Literals have known types
            Expr::Int(_) => Type::Int,
            Expr::Float(_) => Type::Float,
            Expr::Bool(_) => Type::Bool,
            Expr::String(_) => Type::String,

            // Variables: lookup or create fresh
            Expr::Var(name) => {
                self.var_types.get(name).cloned()
                    .unwrap_or_else(|| {
                        let t = self.fresh_var();
                        self.var_types.insert(name.clone(), t.clone());
                        t
                    })
            }

            // Binary ops: constrain operands
            Expr::BinOp(op, left, right) => {
                let t_left = self.infer_expr(left);
                let t_right = self.infer_expr(right);

                match op {
                    // Arithmetic: both must be Num, result same type
                    Op::Add | Op::Sub | Op::Mul | Op::Div => {
                        self.constraints.push((t_left.clone(), t_right.clone()));
                        self.add_constraint(&t_left, Constraint::Num);
                        t_left
                    }
                    // Comparison: both same type, result Bool
                    Op::Lt | Op::Gt | Op::Eq => {
                        self.constraints.push((t_left.clone(), t_right.clone()));
                        Type::Bool
                    }
                    // String concat
                    Op::Concat => {
                        self.constraints.push((t_left, Type::String));
                        self.constraints.push((t_right, Type::String));
                        Type::String
                    }
                }
            }

            // Function call: instantiate function type
            Expr::Call(func_name, args) => {
                let func_type = self.func_types.get(func_name)
                    .expect("undefined function");

                // Create fresh type vars for each generic param
                let substitution: HashMap<TypeVarId, Type> =
                    func_type.type_params.iter()
                        .map(|(id, constraints)| {
                            let fresh = self.fresh_var();
                            for c in constraints {
                                self.add_constraint(&fresh, c.clone());
                            }
                            (*id, fresh)
                        })
                        .collect();

                // Apply substitution to param types and constrain args
                for (arg, param_type) in args.iter().zip(&func_type.param_types) {
                    let arg_type = self.infer_expr(arg);
                    let expected = self.substitute(param_type, &substitution);
                    self.constraints.push((arg_type, expected));
                }

                // Return substituted return type
                self.substitute(&func_type.return_type, &substitution)
            }

            // Let binding
            Expr::Let(name, value, body) => {
                let value_type = self.infer_expr(value);
                self.var_types.insert(name.clone(), value_type);
                self.infer_expr(body)
            }

            // If expression
            Expr::If(cond, then_branch, else_branch) => {
                let cond_type = self.infer_expr(cond);
                self.constraints.push((cond_type, Type::Bool));

                let then_type = self.infer_expr(then_branch);
                let else_type = self.infer_expr(else_branch);
                self.constraints.push((then_type.clone(), else_type));
                then_type
            }

            // ... other cases
        }
    }
}
```

### Phase 2: Unification (Solving Constraints)

```rust
struct Substitution {
    mappings: HashMap<TypeVarId, Type>,
}

impl Substitution {
    fn unify(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        let t1 = self.apply(t1);
        let t2 = self.apply(t2);

        match (&t1, &t2) {
            // Same concrete types: OK
            (Type::Int, Type::Int) => Ok(()),
            (Type::Float, Type::Float) => Ok(()),
            (Type::Bool, Type::Bool) => Ok(()),
            (Type::String, Type::String) => Ok(()),

            // Type var: bind it
            (Type::Var(id), other) | (other, Type::Var(id)) => {
                if self.occurs_check(*id, other) {
                    Err(TypeError::InfiniteType)
                } else {
                    self.mappings.insert(*id, other.clone());
                    Ok(())
                }
            }

            // Compound types: unify components
            (Type::List(a), Type::List(b)) => self.unify(a, b),

            (Type::Tuple(as_), Type::Tuple(bs)) if as_.len() == bs.len() => {
                for (a, b) in as_.iter().zip(bs) {
                    self.unify(a, b)?;
                }
                Ok(())
            }

            (Type::Function(args1, ret1), Type::Function(args2, ret2))
                if args1.len() == args2.len() => {
                for (a, b) in args1.iter().zip(args2) {
                    self.unify(a, b)?;
                }
                self.unify(ret1, ret2)
            }

            // Mismatch
            _ => Err(TypeError::Mismatch(t1, t2)),
        }
    }

    fn apply(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(id) => {
                self.mappings.get(id)
                    .map(|t| self.apply(t))
                    .unwrap_or_else(|| ty.clone())
            }
            Type::List(inner) => Type::List(Box::new(self.apply(inner))),
            Type::Tuple(elems) => Type::Tuple(elems.iter().map(|t| self.apply(t)).collect()),
            Type::Function(args, ret) => Type::Function(
                args.iter().map(|t| self.apply(t)).collect(),
                Box::new(self.apply(ret)),
            ),
            _ => ty.clone(),
        }
    }
}
```

## 4. Monomorphization

After type inference, we know the concrete types at each call site.

```rust
struct Monomorphizer {
    // Maps (function_name, concrete_types) -> monomorphized_name
    instances: HashMap<(String, Vec<Type>), String>,
    // Generated function bodies
    generated: Vec<MonoFunction>,
}

struct MonoFunction {
    name: String,           // e.g., "double$Int" or "double$Float"
    param_types: Vec<Type>, // Concrete types
    return_type: Type,
    body: TypedExpr,        // AST annotated with types
}

impl Monomorphizer {
    fn monomorphize_call(&mut self,
                         func_name: &str,
                         arg_types: &[Type],
                         func_def: &FunctionDef) -> String {
        let key = (func_name.to_string(), arg_types.to_vec());

        if let Some(mono_name) = self.instances.get(&key) {
            return mono_name.clone();
        }

        // Generate new monomorphized name
        let mono_name = format!("{}${}", func_name,
            arg_types.iter().map(type_suffix).collect::<Vec<_>>().join("_"));

        // Specialize the function body with concrete types
        let mono_func = self.specialize_function(func_def, arg_types);

        self.instances.insert(key, mono_name.clone());
        self.generated.push(mono_func);

        mono_name
    }
}

fn type_suffix(ty: &Type) -> String {
    match ty {
        Type::Int => "I".to_string(),
        Type::Float => "F".to_string(),
        Type::Bool => "B".to_string(),
        Type::String => "S".to_string(),
        Type::List(inner) => format!("L{}", type_suffix(inner)),
        Type::Tuple(elems) => format!("T{}", elems.iter().map(type_suffix).collect::<String>()),
        _ => "X".to_string(),
    }
}
```

## 5. Code Generation with Types

Now the compiler knows exact types and can emit correct instructions:

```rust
impl Compiler {
    fn compile_binop(&mut self, op: &Op, left: &TypedExpr, right: &TypedExpr) -> Reg {
        let left_reg = self.compile_typed_expr(left);
        let right_reg = self.compile_typed_expr(right);
        let dst = self.alloc_reg();

        match (op, &left.ty) {
            (Op::Add, Type::Int) => self.emit(Instruction::Add(dst, left_reg, right_reg)),
            (Op::Add, Type::Float) => self.emit(Instruction::AddFloat(dst, left_reg, right_reg)),
            (Op::Sub, Type::Int) => self.emit(Instruction::Sub(dst, left_reg, right_reg)),
            (Op::Sub, Type::Float) => self.emit(Instruction::SubFloat(dst, left_reg, right_reg)),
            // ... etc
        }

        dst
    }
}
```

## 6. Example Walkthrough

```
// Source code
double(x: Num) -> Num = x + x

main() = {
    a = double(42)
    b = double(3.14)
    println(show(a))
    println(show(b))
}
```

### Step 1: Parse & Create Function Signatures

```
double: forall T. (T: Num) => T -> T
main: () -> ()
```

### Step 2: Type Inference in main()

```
a = double(42)
  - 42 : Int
  - double call: T = Int (from argument)
  - a : Int

b = double(3.14)
  - 3.14 : Float
  - double call: T = Float (from argument)
  - b : Float
```

### Step 3: Monomorphization

Collect all instantiations of `double`:
- `double@Int` (from `double(42)`)
- `double@Float` (from `double(3.14)`)

Generate:
```
double$I(x: Int) -> Int = x + x     // uses Add instruction
double$F(x: Float) -> Float = x + x // uses AddFloat instruction
```

### Step 4: Code Generation

```
main:
  ; a = double$I(42)
  LoadConst r0, 42
  Call r1, double$I, [r0]

  ; b = double$F(3.14)
  LoadConst r2, 3.14
  Call r3, double$F, [r2]

  ; println(show(a))
  Call r4, show$I, [r1]
  Call _, println, [r4]

  ; println(show(b))
  Call r5, show$F, [r3]
  Call _, println, [r5]
```

## 7. Implementation Phases

### Phase 1: Basic Type Annotations (simple, immediate benefit)
- Parse type annotations on function parameters
- Use annotations in `is_float_expr` instead of heuristics
- No inference yet - just trust the annotations

### Phase 2: Local Type Inference
- Infer types within function bodies
- Track types through let bindings
- Still no cross-function inference

### Phase 3: Full Hindley-Milner
- Constraint generation
- Unification
- Polymorphic functions

### Phase 4: Monomorphization
- Track call sites with concrete types
- Generate specialized function variants
- Update call instructions to use specialized versions

## 8. Syntax Summary

```
// Explicit types (optional)
add(x: Int, y: Int) -> Int = x + y

// Constrained polymorphism
double(x: Num) -> Num = x + x
printAny(x: Show) = println(show(x))

// Full polymorphism
identity(x: a) -> a = x
first(pair: (a, b)) -> a = ...

// Generic types
map(f: a -> b, xs: List[a]) -> List[b] = ...

// Type aliases
type Point = (Float, Float)
type Maybe[T] = Some(T) | None
```

## 9. Current Implementation Status (December 2024)

### Trait-Bounded Polymorphism with Monomorphization

The compiler now supports **trait-bounded polymorphism** where:
1. Types can implement traits with methods
2. Generic functions can call trait methods on type parameters
3. At each call site, a specialized (monomorphized) variant is generated

#### Example: Comparable Trait

```nos
# Define a trait with methods
trait Comparable
    compare(self, other: Self) -> Int
end

# Type with Float field
type Temperature = { celsius: Float }

# Implement trait for the type
Temperature: Comparable
    compare(self, other) = {
        if self.celsius < other.celsius then -1
        else if self.celsius > other.celsius then 1
        else 0
    }
end

# Use it
main() = {
    cold = Temperature(0.0)
    warm = Temperature(25.0)
    result = cold.compare(warm)  # Returns -1
}
```

#### Key Implementation Details

1. **Record Constructor Type Inference**: `Temperature(0.0)` is now correctly
   identified as type `Temperature` via `expr_type_name()` checking record types

2. **Self-Typed Parameter Tracking**: When compiling trait implementations,
   `self` and `Self`-typed parameters are registered in `param_types` so that
   field access like `self.celsius` correctly identifies the field type

3. **Float Field Comparison**: The compiler correctly infers that `self.celsius`
   is a `Float` type and generates `LessThanFloat` instruction instead of
   `LessThan`

#### How Monomorphization Works

For a polymorphic function like:
```nos
describe(x: T where T: Describable) = x.describe()
```

At each call site:
- `describe(cat)` generates `describe$Cat` specialized for Cat type
- `describe(dog)` generates `describe$Dog` specialized for Dog type

The mangled name includes the concrete type, and the specialized variant has
the trait method call resolved to the concrete implementation.
