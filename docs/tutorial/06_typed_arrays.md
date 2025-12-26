# Typed Arrays

For high-performance numerical computing, Nostos offers specialized typed arrays: `Int64Array`, `Float64Array`, and `Float32Array`. These provide contiguous memory storage for unboxed numeric values, enabling efficient operations, especially when combined with Nostos's JIT compiler.

## Int64Array

`Int64Array` stores 64-bit signed integers. It's ideal for tasks like image processing, simulations, or any computation involving large sets of integer data where memory layout and access speed are critical.

```nostos
# Create an array of 5 64-bit integers, initialized to 0
my_int_array = newInt64Array(5)

main() = {
    println(my_int_array.length()) # Prints 5

    # Access and mutate elements using 0-based indexing
    my_int_array[0] = 10
    my_int_array[1] = 20
    my_int_array[2] = my_int_array[0] + my_int_array[1] # 30

    println(my_int_array[2]) # Prints 30

    # Iterate and sum elements
    var total = 0
    for i = 0 to my_int_array.length() {
        total = total + my_int_array[i]
    }
    println("Sum: " ++ total.show()) # Prints "Sum: 60" (10+20+30+0+0)
}
```

## Float64Array

`Float64Array` stores 64-bit floating-point numbers. It's perfect for scientific computing, machine learning, and graphics, where precision and speed for real numbers are paramount.

```nostos
# Create an array of 3 64-bit floats, initialized to 0.0
my_float_array = newFloat64Array(3)

main() = {
    my_float_array[0] = 3.14
    my_float_array[1] = 2.71
    my_float_array[2] = my_float_array[0] * my_float_array[1]

    println(my_float_array[2]) # Prints ~8.5094

    # Example: Scale all elements
    scale_factor = 2.0
    for i = 0 to my_float_array.length() {
        my_float_array[i] = my_float_array[i] * scale_factor
    }
    println(my_float_array[0]) # Prints 6.28
}
```

## Float32Array

`Float32Array` stores 32-bit floating-point numbers. It's the native format for pgvector and ideal for AI/ML embeddings where memory efficiency matters more than precision.

```nostos
main() = {
    # Create from a list of floats
    embedding = Float32Array.fromList([0.1, 0.2, 0.3, 0.4])

    # Get length and access elements
    len = embedding.length()  # 4
    first = embedding.get(0)  # 0.1

    # Set returns a new array (functional style)
    updated = embedding.set(0, 0.5)

    # Convert back to list
    asList = embedding.toList()  # [0.1, 0.2, 0.3, 0.4]

    # Create with size and initial value
    zeros = Float32Array.make(128, 0.0)  # 128-dim zero vector

    println("Vector length: " ++ len.show())
}
```

## Typed Arrays vs. Lists

While lists are flexible and immutable, typed arrays offer significant performance advantages for numerical work due to their contiguous memory, fixed element size, and direct indexing.

- **Lists**: Immutable, linked-list structure, good for functional patterns and variable-sized elements.
- **Typed Arrays**: Mutable, contiguous memory, fixed element types (`Int64`, `Float64`, `Float32`), 0-indexed random access, excellent for performance-critical numerical computations.
