# Types in Nostos

## Built-in Types

```nostos
Int         # 64-bit integer
Float       # 64-bit floating point
Bool        # true or false
String      # UTF-8 string
Char        # Single character
()          # Unit type (empty value)
List[T]     # List of elements
(A, B)      # Tuple
Map[K, V]   # Map/dictionary
Set[T]      # Set
```

## Type Annotations

```nostos
# Variables (usually inferred)
x: Int = 42
name: String = "Alice"
items: List[Int] = [1, 2, 3]

# Function parameters and return
add(a: Int, b: Int) -> Int = a + b

# Complex types
data: Map[String, List[Int]] = %{"a": [1, 2], "b": [3, 4]}
```

## Record Types

```nostos
# Define a record type
type Person = { name: String, age: Int }

# Create instance
alice = Person("Alice", 30)
bob = Person(name: "Bob", age: 25)

# Access fields
alice.name      # "Alice"
alice.age       # 30

# Records are immutable - create new with changes
older = Person(alice.name, alice.age + 1)
```

## Variant Types (Sum Types)

```nostos
# Define variants
type Color = Red | Green | Blue

# Variants with data
type Shape = Circle(Float) | Rectangle(Float, Float) | Point

# Use pattern matching
describe(c: Color) = match c {
    Red -> "red",
    Green -> "green",
    Blue -> "blue"
}

area(s: Shape) = match s {
    Circle(r) -> 3.14159 * r * r,
    Rectangle(w, h) -> w * h,
    Point -> 0.0
}

# Common pattern: Option
type Option[T] = Some(T) | None

# Common pattern: Result
type Result[T, E] = Ok(T) | Err(E)
```

## Variant with Named Fields

```nostos
# Variants can have named fields
type Event =
    | Click { x: Int, y: Int }
    | KeyPress { key: Char, modifiers: List[String] }
    | Scroll { delta: Int }

# Pattern match with field names
handle(e: Event) = match e {
    Click { x, y } -> "Clicked at " ++ show(x) ++ "," ++ show(y),
    KeyPress { key, _ } -> "Pressed " ++ show(key),
    Scroll { delta } -> "Scrolled " ++ show(delta)
}
```

## Generic Types

```nostos
# Generic record
type Box[T] = { value: T }

intBox: Box[Int] = Box(42)
strBox: Box[String] = Box("hello")

# Generic variant
type Tree[T] = Leaf(T) | Node(Tree[T], Tree[T])

# Multiple type parameters
type Pair[A, B] = { first: A, second: B }
type Either[L, R] = Left(L) | Right(R)
```

## Type Aliases

```nostos
# Simple alias
type UserId = Int
type Email = String

# Generic alias
type StringMap[V] = Map[String, V]
type Callback[T] = (T) -> ()

# Using aliases
users: StringMap[Person] = %{"alice": Person("Alice", 30)}
```

## Tuples

```nostos
# Tuple types
point: (Int, Int) = (10, 20)
triple: (String, Int, Bool) = ("hello", 42, true)

# Access by index (pattern matching)
(x, y) = point
(name, age, active) = triple

# Tuple in function return
divmod(a: Int, b: Int) -> (Int, Int) = (a / b, a % b)
(quotient, remainder) = divmod(17, 5)
```

## Nested Types

```nostos
type Address = { street: String, city: String, zip: String }
type Company = { name: String, address: Address }
type Employee = { name: String, company: Company }

# Accessing nested fields
emp = Employee("Alice", Company("Acme", Address("123 Main", "NYC", "10001")))
emp.company.address.city    # "NYC"
```

## Recursive Types

```nostos
# Linked list
type LinkedList[T] = Nil | Cons(T, LinkedList[T])

# Binary tree
type BinaryTree[T] = Empty | Node(T, BinaryTree[T], BinaryTree[T])

# JSON-like structure
type Json =
    | JsonNull
    | JsonBool(Bool)
    | JsonNumber(Float)
    | JsonString(String)
    | JsonArray(List[Json])
    | JsonObject(Map[String, Json])
```

## Type Inference

```nostos
# Types are inferred when possible
x = 42                  # x: Int
y = "hello"             # y: String
z = [1, 2, 3]           # z: List[Int]

# Inference through functions
double(x) = x * 2       # Inferred: Int -> Int
greet(name) = "Hi " ++ name  # Inferred: String -> String

# Sometimes annotation needed
identity(x: T) -> T = x  # Generic needs explicit parameter
```

## Working with Option

```nostos
type Option[T] = Some(T) | None

# Creating
found: Option[Int] = Some(42)
missing: Option[Int] = None

# Pattern match
getValue(opt: Option[Int]) -> Int = match opt {
    Some(x) -> x,
    None -> 0
}

# Common methods
Some(42).map(x => x * 2)        # Some(84)
None.map(x => x * 2)            # None
Some(42).getOrElse(0)           # 42
None.getOrElse(0)               # 0
```

## Working with Result

```nostos
type Result[T, E] = Ok(T) | Err(E)

# Creating
success: Result[Int, String] = Ok(42)
failure: Result[Int, String] = Err("not found")

# Pattern match
handle(r: Result[Int, String]) = match r {
    Ok(value) -> "Got: " ++ show(value),
    Err(msg) -> "Error: " ++ msg
}

# Common methods
Ok(42).map(x => x * 2)          # Ok(84)
Err("fail").map(x => x * 2)     # Err("fail")
Ok(42).mapErr(e => "Error: " ++ e)  # Ok(42)
```

## Constructing Records/Variants

```nostos
type Person = { name: String, age: Int }

# Positional
p1 = Person("Alice", 30)

# Named (order doesn't matter)
p2 = Person(age: 25, name: "Bob")

# Variants
type Status = Active | Inactive(String)

s1 = Active
s2 = Inactive("vacation")
```
