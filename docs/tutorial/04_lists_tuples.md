# Lists & Tuples

Nostos provides immutable Lists and fixed-size Tuples as fundamental data structures. They are often used together with pattern matching to process structured data.

## Lists

Lists are immutable, singly linked-list based collections, similar to those found in functional programming languages. They are best suited for recursive processing and scenarios where prepending elements is efficient.

```nostos
empty_list = []
numbers = [1, 2, 3, 4, 5] # List creation

# Using list pattern matching (head | tail)
get_head([h | _]) = h
get_tail([_ | t]) = t

# Recursive function to sum list elements
sum_list([]) = 0
sum_list([x | xs]) = x + sum_list(xs)

main() = {
    head_val = get_head(numbers) # head_val is 1
    tail_list = get_tail(numbers) # tail_list is [2, 3, 4, 5]
    total_sum = sum_list(numbers) # total_sum is 15

    # Prepending an element (creates a new list)
    new_list = [0 | numbers] # new_list is [0, 1, 2, 3, 4, 5]
}
```

## Tuples

Tuples are fixed-size, ordered collections that can hold values of different types. They are commonly used for grouping related data or for functions that need to return multiple values.

```nostos
point_2d = (10, 20)
user_info = ("Alice", 30, true)

# Destructuring tuples with pattern matching
get_x((x, _)) = x
get_name((name, _, _)) = name

# Function returning multiple values
divide_with_remainder(a, b) = (a / b, a % b)

main() = {
    x_coord = get_x(point_2d) # x_coord is 10
    name = get_name(user_info) # name is "Alice"

    (quotient, remainder) = divide_with_remainder(10, 3)
    println("10 / 3 = " ++ quotient.show() ++ " with remainder " ++ remainder.show())
}
```

### Tuple Field Access

Access tuple elements by index using `.0`, `.1`, etc. or bracket notation.

```nostos
main() = {
    point = (10, 20, 30)

    # Access by index using dot notation
    x = point.0    # 10
    y = point.1    # 20
    z = point.2    # 30

    # Or using bracket notation
    first = point[0]   # 10

    # Get tuple length
    len = point.length()  # 3

    # Useful with query results from databases
    rows = Pg.query(conn, "SELECT id, name FROM users", [])
    rows.each(row => {
        id = row.0
        name = row.1
        println(id.show() ++ ": " ++ name)
    })
}
```

## Method Chaining

Nostos supports method call syntax that allows chaining operations fluently. When you call `x.f(y)`, it's equivalent to `f(x, y)`. This enables powerful, readable data transformations.

```nostos
main() = {
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Chain operations fluently
    result = numbers
        .filter(x => x % 2 == 0)    # Keep even numbers [2,4,6,8,10]
        .map(x => x * x)            # Square them [4,16,36,64,100]
        .fold(0, (acc, x) => acc + x)  # Sum them: 220

    # Each step returns a new list, enabling the chain
    evens = numbers.filter(x => x % 2 == 0)  # [2, 4, 6, 8, 10]
    doubled = evens.map(x => x * 2)          # [4, 8, 12, 16, 20]

    # Works with any function that takes the value as first argument
    words = ["hello", "world"]
    lengths = words.map(s => String.length(s))  # [5, 5]

    # Sorting and transforming
    sorted_desc = numbers.sort().reverse()  # [10, 9, 8, ..., 1]
}
```

### Common List Methods

- `.map(fn)` - Transform each element
- `.filter(pred)` - Keep matching elements
- `.fold(init, fn)` - Reduce to single value
- `.find(pred)` - Find first match
- `.sort()` - Sort elements
- `.reverse()` - Reverse order
- `.take(n)` - First n elements
- `.drop(n)` - Skip first n
- `.any(pred)` - True if any match
- `.all(pred)` - True if all match
- `.each(fn)` - Execute for side effects
- `.unique()` - Remove duplicates
