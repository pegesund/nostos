# Standard Library

Nostos comes with a comprehensive standard library providing utilities for strings, collections, time, file system, cryptography, and more. All modules are available by default without imports.

## Available Modules

String, Map, Set, Time, Random, Path, File, Dir, Env, Regex, Uuid, Crypto, Http, Json, Pg, Float32Array, Float64Array, Int64Array

List functions (map, filter, fold, etc.) are available as methods on lists directly.

## String Module

String manipulation functions for common text operations.

```nostos
main() = {
    # Length and case
    println("hello".length())          # 5
    println("hello".toUpper())         # "HELLO"
    println("WORLD".toLower())         # "world"

    # Searching
    println("hello".contains("ell"))   # true
    println("hello".startsWith("he"))  # true
    println("hello".endsWith("lo"))    # true

    # Extraction
    println("hello".substring(1, 4))   # "ell"

    # Transformation
    println("  hi  ".trim())           # "hi"
    println("hello".replace("l", "L")) # "heLlo"
    println("ab".repeat(3))            # "ababab"
    println("hello".reverse())         # "olleh"

    # Split (use Regex)
    parts = Regex.split(",", "a,b,c")  # ["a", "b", "c"]
}
```

## Map Module

Immutable key-value maps with efficient lookups.

```nostos
main() = {
    m = %{"a": 1, "b": 2, "c": 3}

    println(m.get("a"))        # 1
    println(m.contains("b"))   # true
    println(m.size())          # 3

    # Modification (returns new map)
    m2 = m.insert("d", 4)
    m3 = m2.remove("a")

    println(m.keys())          # ["a", "b", "c"]
    println(m.values())        # [1, 2, 3]
}
```

## Set Module

Immutable sets with unique elements.

```nostos
main() = {
    s = #{1, 2, 3, 4, 5}

    println(s.contains(3))     # true
    println(s.size())          # 5

    s2 = s.insert(6)
    s3 = s2.remove(1)

    # Set operations
    a = #{1, 2, 3}
    b = #{2, 3, 4}
    println(a.union(b))        # #{1, 2, 3, 4}
    println(a.intersection(b)) # #{2, 3}
    println(a.difference(b))   # #{1}
}
```

## Time Module

Date and time operations with Unix timestamps.

```nostos
main() = {
    now = Time.now()
    println(now.format("%Y-%m-%d %H:%M:%S"))

    println("Year: " ++ now.year().show())
    println("Month: " ++ now.month().show())

    # Parse from string
    ts = Time.parse("2024-06-15 10:30:00", "%Y-%m-%d %H:%M:%S")

    # Arithmetic (seconds)
    tomorrow = now.add(86400)
}
```

## Random Module

Random number generation and selection.

```nostos
main() = {
    n = Random.int(1, 100)
    f = Random.float()
    b = Random.bool()

    colors = ["red", "green", "blue"]
    color = Random.choice(colors)

    nums = [1, 2, 3, 4, 5]
    shuffled = Random.shuffle(nums)
}
```

## Path Module

File path manipulation utilities.

```nostos
main() = {
    println(Path.join("/home", "user"))           # "/home/user"
    println(Path.dirname("/home/user/file.txt"))  # "/home/user"
    println(Path.basename("/home/user/file.txt")) # "file.txt"
    println(Path.extension("document.pdf"))       # "pdf"
    println(Path.normalize("/home/../etc"))       # "/etc"
}
```

## Env Module

Access environment variables.

```nostos
main() = {
    home = Env.get("HOME")
    match home {
        Some(path) -> println("Home: " ++ path)
        None -> println("HOME not set")
    }

    cwd = Env.cwd()
    args = Env.args()
}
```

## Regex Module

Regular expression pattern matching.

```nostos
main() = {
    text = "The quick brown fox"

    println(Regex.matches(text, "quick"))    # true
    matches = Regex.findAll(text, "[a-zA-Z]+")
    result = Regex.replace(text, "fox", "dog")
    parts = Regex.split(text, " ")
}
```

## UUID Module

Generate and validate UUIDs.

```nostos
main() = {
    id = Uuid.v4()
    println(Uuid.isValid(id))  # true
}
```

## Crypto Module

Cryptographic hashing and password functions.

```nostos
main() = {
    println(Crypto.sha256("hello"))
    println(Crypto.sha512("hello"))

    # Password hashing with bcrypt
    hash = Crypto.bcryptHash("secret123", 10)
    isValid = Crypto.bcryptVerify("secret123", hash)

    randomHex = Crypto.randomBytes(16)
}
```

## File Module

File system operations. All operations throw exceptions on error.

```nostos
main() = {
    File.writeAll("/tmp/test.txt", "Hello!")
    content = File.readAll("/tmp/test.txt")
    println(File.exists("/tmp/test.txt"))
    File.remove("/tmp/test.txt")

    try {
        data = File.readAll("/nonexistent.txt")
    } catch { e ->
        println("File error: " ++ e)
    }
}
```

## Dir Module

Directory operations.

```nostos
main() = {
    Dir.create("/tmp/mydir")
    Dir.createAll("/tmp/nested/path/here")
    files = Dir.list("/tmp")
    Dir.remove("/tmp/mydir")
    Dir.removeAll("/tmp/nested")
}
```

## List Functions

List operations with method chaining.

```nostos
main() = {
    xs = [1, 2, 3, 4, 5]

    println(head(xs))             # 1
    println(tail(xs))             # [2, 3, 4, 5]
    println(last(xs))             # 5

    doubled = xs.map(x => x * 2)
    evens = xs.filter(x => x % 2 == 0)
    sum = xs.fold(0, (acc, x) => acc + x)

    # Chained operations
    result = [1, 2, 3, 4, 5]
        .filter(x => x > 2)
        .map(x => x * 10)
        .fold(0, (a, b) => a + b)
    println(result)               # 120
}
```

## Pg Module (PostgreSQL)

Connect to PostgreSQL databases.

```nostos
conn = Pg.connect("host=localhost user=postgres password=secret dbname=mydb")

# Query returns list of tuples
rows = Pg.query(conn, "SELECT id, name FROM users WHERE age > $1", [18])

# Iterate over results
rows.each(row => {
    println("User " ++ show(row.0) ++ ": " ++ row.1)
})

# Execute with parameters
Pg.execute(conn, "INSERT INTO users (name) VALUES ($1)", ["Alice"])

# Transactions
Pg.begin(conn)
Pg.execute(conn, "UPDATE accounts SET balance = balance - 100 WHERE id = $1", [1])
Pg.commit(conn)

# Vector support (pgvector)
embedding = Float32Array.fromList([0.1, 0.2, 0.3])
Pg.execute(conn, "INSERT INTO items (embedding) VALUES ($1)", [embedding])

Pg.close(conn)
```

## Float32Array / Float64Array

Typed arrays for numerical computing.

```nostos
# Create from list
arr = Float32Array.fromList([1.0, 2.0, 3.0])

# Create with size and initial value
zeros = Float32Array.make(128, 0.0)

# Access elements
len = Float32Array.length(arr)
val = Float32Array.get(arr, 0)

# Convert back to list
list = Float32Array.toList(arr)
```
