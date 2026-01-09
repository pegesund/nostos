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
