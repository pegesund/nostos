# Mutability & Persistent Data Structures

Nostos uses **persistent data structures** - Lists, Maps, and Sets are immutable. Operations like `insert` or `append` return a *new* structure, leaving the original unchanged. Combined with mutable bindings, this gives you the best of both worlds: the safety of immutability with the convenience of imperative-style updates.

## Immutable vs Mutable Bindings

Quick recap: bindings are immutable by default. Use `var` for mutable bindings.

```nostos
# Immutable binding
x = 10
# x = 20  # ERROR: Can't rebind to different value

# Mutable binding
var count = 0
count = count + 1  # OK
count = count + 1  # OK
println(count)     # 2
```

## Persistent Data Structures

Lists, Maps, and Sets in Nostos are **persistent** (immutable). When you "modify" them, you get a new structure - the original remains unchanged.

```nostos
original = [1, 2, 3]
modified = original ++ [4]  # Append returns a NEW list

println(original)  # [1, 2, 3] - unchanged!
println(modified)  # [1, 2, 3, 4]

# Same for Maps
map1 = %{"a": 1}
map2 = map1.insert("b", 2)  # Returns a NEW map

println(map1)  # %{"a": 1} - unchanged!
println(map2)  # %{"a": 1, "b": 2}
```

**Why persistent?** Persistent structures are inherently thread-safe - no locks needed when sharing between processes. They also make debugging easier since values never change unexpectedly.

## The Pattern: Mutable Var + Persistent Structure

The idiomatic way to build up a collection is to use a mutable binding that holds successive versions of a persistent structure:

```nostos
# Building a list with a mutable binding
main() = {
    var items = []

    items = items ++ [1]
    items = items ++ [2]
    items = items ++ [3]

    println(items)  # [1, 2, 3]
}
```

The same pattern works for Maps and Sets:

```nostos
# Building a map incrementally
main() = {
    var config = %{}

    config = config.insert("debug", true)
    config = config.insert("port", 8080)
    config = config.insert("host", "localhost")

    println(config)
    # %{"debug": true, "port": 8080, "host": "localhost"}
}

# Building a set
main() = {
    var seen = #{}

    seen = seen.insert("apple")
    seen = seen.insert("banana")
    seen = seen.insert("apple")  # Duplicate - no effect

    println(seen)  # #{"apple", "banana"}
}
```

## Building Collections in Loops

This pattern shines in loops where you accumulate results:

```nostos
# Collect squares of numbers
main() = {
    var squares = []

    for i = 1 to 5 {
        squares = squares ++ [i * i]
    }

    println(squares)  # [1, 4, 9, 16, 25]
}

# Count word frequencies
count_words(words) = {
    var counts = %{}

    words.each(word => {
        current = counts.get(word).unwrapOr(0)
        counts = counts.insert(word, current + 1)
    })

    counts
}

main() = {
    words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
    println(count_words(words))
    # %{"apple": 3, "banana": 2, "cherry": 1}
}
```

## Comparison: Imperative vs Functional

Both styles are valid. Use whichever fits your situation:

**Mutable var + loop:**
```nostos
var result = []
for i = 1 to 5 {
    result = result ++ [i * i]
}
result  # [1, 4, 9, 16, 25]
```

**Functional with map:**
```nostos
[1, 2, 3, 4, 5].map(i => i * i)
# [1, 4, 9, 16, 25]
```

**Mutable var accumulation:**
```nostos
var total = 0
for x in [1, 2, 3, 4, 5] {
    total = total + x
}
total  # 15
```

**Functional with fold:**
```nostos
[1, 2, 3, 4, 5].fold(0, (acc, x) => acc + x)
# 15
```

## When to Use Each Approach

### Guidelines

- **Use mutable vars** when building collections incrementally over complex logic, or when translating imperative algorithms
- **Use functional style** (map, filter, fold) for simple transformations - it's often more concise
- **Mix both** freely - Nostos doesn't force you into one paradigm
- **For shared state** between processes, use `mvar` instead (see Concurrency chapter)

## Note on Thread Safety

Mutable vars (`var`) are local to a single process and are **not** shared between processes. For shared mutable state across processes, use `mvar` (module-level mutable variables) which have automatic locking.

```nostos
# Local mutable var - only visible to this process
main() = {
    var local_count = 0
    local_count = local_count + 1  # OK, single-process access
}

# Shared mvar - safe for concurrent access
mvar shared_count: Int = 0

increment() = {
    shared_count = shared_count + 1  # Automatically locked
}

main() = {
    # Spawn 100 processes all incrementing
    for i = 1 to 100 {
        spawn { increment() }
    }
    sleep(100)
    println(shared_count)  # 100 - no race conditions!
}
```
