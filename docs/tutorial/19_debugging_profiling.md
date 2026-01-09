# Debugging & Profiling

Nostos provides built-in tools for debugging your code and measuring performance. These tools are available in the REPL and help you understand what your code is doing and where time is being spent.

## The REPL

The Nostos REPL (Read-Eval-Print Loop) is your primary environment for interactive development. Start it by running `nostos` without arguments:

```
$ nostos
Nostos REPL v0.1.0
Type :help for available commands

nostos> 1 + 2
3

nostos> fib(n) = if n < 2 then n else fib(n-1) + fib(n-2)
nostos> fib(10)
55
```

## REPL Commands

The REPL provides several special commands, all prefixed with `:`.

| Command | Description |
|---------|-------------|
| `:help` | Show available commands |
| `:load <file>` | Load and execute a .nos file |
| `:profile <expr>` | Run expression with timing |
| `:debug <fn>` | Set a breakpoint on a function |
| `:undebug <fn>` | Remove a breakpoint |
| `:type <expr>` | Show the type of an expression |
| `:demo` | Load demo folder with examples |

## Profiling with :profile

The `:profile` command measures how long an expression takes to execute. This is essential for identifying performance bottlenecks.

```nostos
# Profile a simple expression
nostos> :profile fib(30)
Result: 832040
Time: 45.2ms

# Profile a more complex computation
nostos> :profile [1..1000].map(x => x * x).filter(x => x % 2 == 0).sum()
Result: 166666500
Time: 0.8ms

# Compare different implementations
nostos> :profile naive_sort(big_list)
Time: 1250ms

nostos> :profile quick_sort(big_list)
Time: 12ms
```

### Profiling Tips

- Run the same expression multiple times to account for JIT warmup
- Profile with realistic data sizes
- Compare different approaches with the same input
- Look for unexpected slowdowns in hot paths

## Debugging with Breakpoints

The `:debug` command sets breakpoints on functions. When a breakpointed function is called, execution pauses and you can inspect the arguments and state.

```nostos
# Define a function
nostos> factorial(0) = 1
nostos> factorial(n) = n * factorial(n - 1)

# Set a breakpoint
nostos> :debug factorial

# Now run it - execution will pause at each call
nostos> factorial(5)
[BREAK] factorial(5)
  Press Enter to continue, 'c' to continue without stopping...

[BREAK] factorial(4)
  Press Enter to continue...

# Continue until done
c
Result: 120
```

### Managing Breakpoints

```nostos
# List all active breakpoints
nostos> :debug
Active breakpoints:
  - factorial
  - process_data

# Remove a specific breakpoint
nostos> :undebug factorial
Breakpoint removed: factorial

# Clear all breakpoints
nostos> :undebug
All breakpoints cleared
```

## Print Debugging

Sometimes the simplest approach is the best. Use `println` and `show` to trace values through your code.

```nostos
process_order(order) = {
    println("Processing order: " ++ show(order))

    validated = validate(order)
    println("After validation: " ++ show(validated))

    priced = calculate_price(validated)
    println("After pricing: " ++ show(priced))

    finalize(priced)
}

# The dbg function is a shorthand that prints and returns the value
# useful for inserting into expressions without changing them
calculate(x) = {
    intermediate = dbg(x * 2)  # Prints: [dbg] 20
    intermediate + 10
}

main() = calculate(10)  # Returns 30, prints "[dbg] 20"
```

## Type Inspection with :type

Use `:type` to see the inferred type of any expression without evaluating it.

```nostos
nostos> :type 42
Int

nostos> :type [1, 2, 3]
List[Int]

nostos> :type x => x * 2
Int -> Int

nostos> :type %{"name": "Alice", "age": 30}
Map[String, Int | String]

nostos> :type Some
T -> Option[T]
```

## Common Debugging Patterns

### Debugging Pattern Matching

When a match isn't working as expected, add a catch-all pattern that prints the unexpected value:

```nostos
process(data) = match data {
    Some(x) -> handle(x)
    None -> default_value()
    other -> {
        println("Unexpected value: " ++ show(other))
        panic("Unhandled case in process")
    }
}
```

### Debugging Concurrent Code

For concurrent programs, include the PID in your debug output:

```nostos
worker(id) = {
    me = self()
    log(msg) = println("[Worker " ++ show(id) ++ " PID:" ++ show(me) ++ "] " ++ msg)

    log("Starting")
    result = do_work()
    log("Finished with: " ++ show(result))
    result
}
```

### Debugging Recursive Functions

Track recursion depth to understand the call structure:

```nostos
tree_sum(Empty, depth) = {
    println("  ".repeat(depth) ++ "Empty -> 0")
    0
}
tree_sum(Node(val, left, right), depth) = {
    indent = "  ".repeat(depth)
    println(indent ++ "Node(" ++ show(val) ++ ")")
    l = tree_sum(left, depth + 1)
    r = tree_sum(right, depth + 1)
    result = val + l + r
    println(indent ++ "  = " ++ show(result))
    result
}
```

**Remember:** Remove debug output before committing code. Consider using a `DEBUG` flag that can be toggled:

```nostos
DEBUG = false
debug_log(msg) = if DEBUG then println("[DEBUG] " ++ msg) else ()
```
