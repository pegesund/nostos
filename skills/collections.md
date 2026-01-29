# Collections in Nostos

## Maps (Dictionaries)

```nostos
# Create a map with %{ }
ages = %{"alice": 30, "bob": 25, "carol": 35}

# Type annotation
scores: Map[String, Int] = %{"math": 95, "english": 87}

# Empty map
empty: Map[String, Int] = %{}
```

## Map Index Syntax

Maps support convenient bracket syntax for get and set:

```nostos
m = %{"name": "Alice", "age": 30}

# Get value (returns value or Unit if not found)
name = m["name"]        # "Alice"
age = m["age"]          # 30

# Set value (updates variable with new map)
m["city"] = "Oslo"      # m now has 3 keys
m["age"] = 31           # Update existing key

# Works with any key type
ids = %{1: "one", 2: "two"}
ids[3] = "three"        # Add new entry
val = ids[1]            # "one"
```

Since maps are immutable, `m["key"] = value` is equivalent to `m = Map.insert(m, "key", value)`.

## Map Operations

```nostos
m = %{"a": 1, "b": 2, "c": 3}

# Get value (returns Option)
m.get("a")              # Some(1)
m.get("z")              # None

# Get with default
m.getOrDefault("z", 0)  # 0

# Insert (returns new map)
m2 = m.insert("d", 4)   # %{"a": 1, "b": 2, "c": 3, "d": 4}

# Remove (returns new map)
m3 = m.remove("a")      # %{"b": 2, "c": 3}

# Update value
m4 = m.update("a", x => x + 10)  # %{"a": 11, "b": 2, "c": 3}

# Check existence
m.contains("a")         # true
m.contains("z")         # false

# Size
m.size()                # 3
m.isEmpty()             # false
```

## Iterating Maps

```nostos
m = %{"a": 1, "b": 2, "c": 3}

# Get keys as list
m.keys()                # ["a", "b", "c"]

# Get values as list
m.values()              # [1, 2, 3]

# Get entries as list of tuples
m.entries()             # [("a", 1), ("b", 2), ("c", 3)]

# Map over values
m.mapValues(v => v * 2) # %{"a": 2, "b": 4, "c": 6}

# Filter entries
m.filter((k, v) => v > 1)  # %{"b": 2, "c": 3}

# Fold
m.fold(0, (acc, k, v) => acc + v)  # 6
```

## Merging Maps

```nostos
m1 = %{"a": 1, "b": 2}
m2 = %{"b": 20, "c": 3}

# Merge (right side wins on conflict)
m1.merge(m2)            # %{"a": 1, "b": 20, "c": 3}

# Merge with custom conflict resolution
m1.mergeWith(m2, (v1, v2) => v1 + v2)  # %{"a": 1, "b": 22, "c": 3}
```

## Sets

```nostos
# Create a set with #{ }
colors = #{"red", "green", "blue"}

# Type annotation
numbers: Set[Int] = #{1, 2, 3, 4, 5}

# Empty set
empty: Set[String] = #{}
```

## Set Operations

```nostos
s = #{1, 2, 3, 4, 5}

# Add element (returns new set)
s2 = s.insert(6)        # #{1, 2, 3, 4, 5, 6}

# Remove element (returns new set)
s3 = s.remove(1)        # #{2, 3, 4, 5}

# Check membership
s.contains(3)           # true
s.contains(10)          # false

# Size
s.size()                # 5
s.isEmpty()             # false

# Convert to list
s.toList()              # [1, 2, 3, 4, 5]
```

## Set Index Syntax

Sets support bracket syntax for membership checking:

```nostos
s = #{1, 2, 3, 4, 5}

# Check membership (returns Bool)
s[3]                    # true
s[10]                   # false

# Use in conditions
if s[3] then {
    println("3 is in the set")
}

# Combine checks
s[1] && !s[100]         # true

# With variables
elem = 3
s[elem]                 # true
```

This is equivalent to `s.contains(elem)` but more concise.

## Set Math Operations

```nostos
a = #{1, 2, 3, 4}
b = #{3, 4, 5, 6}

# Union
a.union(b)              # #{1, 2, 3, 4, 5, 6}

# Intersection
a.intersection(b)       # #{3, 4}

# Difference
a.difference(b)         # #{1, 2}

# Subset check
#{1, 2}.isSubset(a)     # true
a.isSuperset(#{1, 2})   # true
```

## Typed Arrays

```nostos
# Efficient numeric arrays

# Float64Array
floats = Float64Array.new(100)          # 100 zeros
floats = Float64Array.from([1.0, 2.0, 3.0])

# Int64Array
ints = Int64Array.new(100)
ints = Int64Array.from([1, 2, 3, 4, 5])

# Float32Array (for GPU compatibility)
f32 = Float32Array.new(100)
```

## Typed Array Operations

```nostos
arr = Float64Array.from([1.0, 2.0, 3.0, 4.0, 5.0])

# Index access
arr[0]                  # 1.0
arr[2]                  # 3.0

# Index assignment (returns new array in immutable mode)
arr[0] = 10.0           # [10.0, 2.0, 3.0, 4.0, 5.0]

# Length
arr.length()            # 5

# Map (returns new typed array)
arr.map(x => x * 2)     # [2.0, 4.0, 6.0, 8.0, 10.0]

# Fold/reduce
arr.fold(0.0, (acc, x) => acc + x)  # 15.0

# Sum (optimized)
arr.sum()               # 15.0

# Slice
arr.slice(1, 4)         # [2.0, 3.0, 4.0]
```

## Buffer Type

```nostos
# Growable byte buffer

buf = Buffer.new()
buf = buf.append("Hello")
buf = buf.append(" World")
buf.toString()          # "Hello World"

# Useful for building strings or binary data
buildCsv(rows: List[List[String]]) -> String = {
    buf = Buffer.new()
    rows.forEach(row => {
        buf = buf.append(row.join(","))
        buf = buf.append("\n")
    })
    buf.toString()
}
```

## Tuples

```nostos
# Fixed-size, mixed-type collection
point = (10, 20)
named = ("Alice", 30, true)

# Destructuring
(x, y) = point
(name, age, active) = named

# Access by pattern matching
getFirst((a, _, _)) = a
getSecond((_, b, _)) = b

# Nested tuples
nested = ((1, 2), (3, 4))
((a, b), (c, d)) = nested
```

## Converting Between Collections

```nostos
# List to Set (removes duplicates)
[1, 2, 2, 3, 3, 3].toSet()      # #{1, 2, 3}

# Set to List
#{1, 2, 3}.toList()             # [1, 2, 3]

# List of pairs to Map
[("a", 1), ("b", 2)].toMap()    # %{"a": 1, "b": 2}

# Map to list of pairs
%{"a": 1, "b": 2}.entries()     # [("a", 1), ("b", 2)]

# List to typed array
[1.0, 2.0, 3.0].toFloat64Array()

# Typed array to list
Float64Array.from([1.0, 2.0]).toList()
```

## Common Patterns

```nostos
# Count occurrences
countWords(words: List[String]) -> Map[String, Int] =
    words.fold(%{}, (acc, word) => {
        count = acc.getOrDefault(word, 0)
        acc.insert(word, count + 1)
    })

# Group by key
groupBy(items: List[T], keyFn: T -> K) -> Map[K, List[T]] =
    items.fold(%{}, (acc, item) => {
        key = keyFn(item)
        existing = acc.getOrDefault(key, [])
        acc.insert(key, existing ++ [item])
    })

# Index lookup table
createIndex(items: List[T], keyFn: T -> K) -> Map[K, T] =
    items.fold(%{}, (acc, item) => acc.insert(keyFn(item), item))

# Deduplicate while preserving order
dedupe(items: List[T]) -> List[T] = {
    (result, _) = items.fold(([], #{}), ((acc, seen), item) => {
        if seen.contains(item) then (acc, seen)
        else (acc ++ [item], seen.insert(item))
    })
    result
}
```

## Performance Considerations

```nostos
# Maps and Sets: O(log n) for most operations
# - Use for lookup-heavy workloads
# - Immutable, returns new collection on modification

# Lists: O(n) for index access, O(1) for head/cons
# - Good for sequential processing
# - Use fold/map instead of index loops

# Typed Arrays: O(1) index access
# - Use for numeric computation
# - More memory efficient than List[Float]
# - Good for interop with FFI

# Choose based on:
# - Access pattern (random vs sequential)
# - Data type (numeric vs mixed)
# - Mutability needs
```
