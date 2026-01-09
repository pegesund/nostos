# Functions

Functions are first-class citizens in Nostos. They are defined using a concise syntax and can be passed around like any other value.

## Defining Functions

Simple functions are defined using the `name(args) = body` syntax.

```nostos
add(a, b) = a + b
double(x) = x * 2

main() = {
    result_add = add(5, 3)    # result_add is 8
    result_double = double(4) # result_double is 8
}
```

## Lambda Expressions

Anonymous functions, or lambdas, are defined using the `args => body` syntax.

```nostos
multiply = (a, b) => a * b
increment = x => x + 1

main() = {
    product = multiply(6, 7) # product is 42
    next_val = increment(10) # next_val is 11
}
```

## Higher-Order Functions

Higher-order functions (HOFs) are functions that can take other functions as arguments, or return functions as their results. This is a powerful feature for abstracting common patterns and writing more concise, reusable code.

### Functions as Arguments

You can pass functions (including lambdas) as arguments to other functions.

```nostos
# apply a function 'f' to a value 'x'
apply(f, x) = f(x)

# apply a function 'f' twice to a value 'x'
twice(f, x) = f(f(x))

main() = {
    increment = x => x + 1
    double = x => x * 2

    # apply increment to 5
    result1 = apply(increment, 5) # result1 is 6

    # apply double twice to 3
    result2 = twice(double, 3)    # result2 is 12 (double(double(3)))
}
```

### Functions as Return Values (Closures)

Functions can also return new functions, often "capturing" values from their environment. These are known as closures.

```nostos
# Returns a function that adds 'n' to its input
make_adder(n) = x => x + n

main() = {
    add5 = make_adder(5)
    add10 = make_adder(10)

    result1 = add5(2)  # result1 is 7
    result2 = add10(2) # result2 is 12
}
```

### Common HOF Patterns (Map & Filter)

Higher-order functions are frequently used to implement powerful data transformation patterns like `map` and `filter`.

```nostos
# Applies a function 'f' to each element of a list
map(_, []) = []
map(f, [x | xs]) = [f(x) | map(f, xs)]

# Filters elements from a list based on a predicate 'p'
filter(_, []) = []
filter(p, [x | xs]) = if p(x) then [x | filter(p, xs)] else filter(p, xs)

main() = {
    numbers = [1, 2, 3, 4]

    squared_numbers = map(x => x * x, numbers)
    # squared_numbers is [1, 4, 9, 16]

    even_numbers = filter(x => x % 2 == 0, numbers)
    # even_numbers is [2, 4]
}
```

## Method-Style Chaining

Nostos supports calling functions using dot notation, which enables fluent method-style chaining. When you write `x.f(y)`, it's equivalent to `f(x, y)`. This makes data transformation pipelines much more readable.

```nostos
# These are equivalent:
result1 = double(5)
result2 = 5.double()

# Chaining multiple operations
numbers = [1, 2, 3, 4, 5]

# Without chaining (nested calls, read inside-out)
result = sum(filter(x => x > 2, map(x => x * 2, numbers)))

# With chaining (read left-to-right)
result = numbers
    .map(x => x * 2)
    .filter(x => x > 2)
    .sum()
# result is 24 (4 + 6 + 8 + 10)
```

This works with any function - the value before the dot becomes the first argument:

```nostos
# String operations
message = "hello world"
    .split(" ")
    .map(s => s.capitalize())
    .join(" ")
# message is "Hello World"

# Numeric operations
calculate(n) = n
    .abs()
    .sqrt()
    .floor()

main() = {
    println(calculate(-16))  # 4
}
```

### Chaining Best Practices

- Use chaining for data transformation pipelines
- Break long chains across multiple lines for readability
- Prefer chaining when operations flow naturally left-to-right
- Use traditional function calls when order doesn't matter or for single operations
