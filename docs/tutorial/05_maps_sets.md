# Maps & Sets

Nostos provides `Map` for efficient key-value storage and `Set` for managing unique collections. Both are immutable by default, promoting predictable data flow.

## Maps

Maps are unordered collections of key-value pairs, where keys must be hashable. They offer efficient O(1) average-case lookup, insertion, and deletion. Maps are created using the `%{key: value}` syntax.

```nostos
# Create an empty map
empty_map = %{}

# Map with various key/value types
user_profile = %{
    "name": "Alice",
    "age": 30,
    "is_active": true,
    1: "one",
    true: "boolean_key"
}

main() = {
    # Accessing values (returns Option[Value])
    name = user_profile.get("name")      # Some("Alice")
    status = user_profile.get("is_active") # Some(true)
    non_existent = user_profile.get("email") # None

    # Inserting/Updating values (returns a new map)
    updated_profile = user_profile.put("age", 31)
    new_entry_profile = updated_profile.put("city", "New York")

    # Removing values (returns a new map)
    profile_without_age = user_profile.delete("age")

    println(user_profile)      # original map is unchanged
    println(updated_profile)
}
```

## Index Syntax for Maps

Maps support convenient bracket syntax for get and set operations:

```nostos
main() = {
    # Create a map
    config = %{"host": "localhost", "port": 8080}

    # Get value using brackets (returns value or Unit if not found)
    host = config["host"]       # "localhost"
    port = config["port"]       # 8080

    # Set value using brackets (updates the variable with new map)
    config["debug"] = true      # config now has 3 entries
    config["port"] = 9000       # Update existing key

    # Works with any hashable key type
    scores = %{1: 100, 2: 85, 3: 92}
    first_score = scores[1]     # 100
    scores[4] = 78              # Add new entry

    println(config)
}
```

Since maps are immutable, `map["key"] = value` is equivalent to:
```nostos
map = Map.insert(map, "key", value)
```

The variable is updated to hold the new map; the previous map value remains unchanged if referenced elsewhere.

## Sets

Sets are unordered collections of unique elements. Like Maps, elements must be hashable. They are useful for membership testing, eliminating duplicates, and performing set operations (union, intersection, difference). Sets are created using the `#{element1, element2}` syntax.

```nostos
# Create an empty set
empty_set = #{}

# Set with duplicate elements (duplicates are automatically removed)
numbers_set = #{1, 2, 3, 2, 1, 4} # numbers_set is #{1, 2, 3, 4}

main() = {
    # Membership testing
    has_two = numbers_set.contains?(2) # true
    has_five = numbers_set.contains?(5) # false

    # Adding elements (returns a new set)
    set_with_five = numbers_set.add(5)
    set_with_five_again = set_with_five.add(5) # No change, still #{1, 2, 3, 4, 5}

    # Removing elements (returns a new set)
    set_without_two = numbers_set.delete(2)

    # Set operations
    set_a = #{1, 2, 3}
    set_b = #{3, 4, 5}

    # union_set = set_a.union(set_b)           # #{1, 2, 3, 4, 5}
    # intersection_set = set_a.intersection(set_b) # #{3}
    # difference_set = set_a.difference(set_b)   # #{1, 2}

    println(numbers_set)
    println(set_with_five)
}
```

## Index Syntax for Sets

Sets support bracket syntax for membership checking, providing a concise alternative to `.contains()`:

```nostos
main() = {
    numbers = #{1, 2, 3, 4, 5}

    # Check membership using brackets (returns Bool)
    has_three = numbers[3]      # true
    has_ten = numbers[10]       # false

    # Use in conditions
    if numbers[3] then {
        println("3 is in the set")
    }

    # Combine membership checks
    all_present = numbers[1] && numbers[2]    # true
    any_missing = !numbers[99]                 # true

    # Works with any hashable element type
    names = #{"alice", "bob", "charlie"}
    found = names["alice"]      # true
    missing = names["dave"]     # false

    0
}
```

This is equivalent to `set.contains(elem)` but more readable in expressions:

```nostos
# These are equivalent:
if set[key] then ...
if set.contains(key) then ...
```
