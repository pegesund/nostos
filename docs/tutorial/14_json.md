# JSON

Nostos provides comprehensive JSON support for parsing, serialization, and typed deserialization. The `fromJson[T]` builtin makes it easy to convert JSON directly to your custom types with full type safety.

## Parsing JSON

Use `jsonParse` to parse a JSON string into a `Json` value.

**Tip:** Use single-quoted strings for JSON to avoid escaping double quotes.

```nostos
# Import from stdlib
use stdlib.json.{jsonParse, jsonStringify, Json}

main() = {
    # Parse a JSON string (single quotes make it cleaner!)
    json: Json = jsonParse('{"name": "Alice", "age": 30}')

    # The Json type is a variant:
    # Null | Bool(Bool) | Number(Float) | String(String)
    # | Array(List[Json]) | Object(List[(String, Json)])

    println(json)
}
```

**Note:** Core stdlib modules (list, string, map, set, option, result, traits, io) are auto-imported. For JSON functions, you need to explicitly import them with `use stdlib.json.*`.

## Typed Deserialization with fromJson

The `fromJson[T](json)` builtin converts JSON to typed Nostos values.

```nostos
use stdlib.json.{jsonParse}

# Define your types
type Person = { name: String, age: Int }

main() = {
    json: Json = jsonParse('{"name": "Alice", "age": 30}')

    # Convert to typed value (fromJson is a builtin)
    person: Person = fromJson[Person](json)

    println(person.name)  # "Alice"
    println(person.age)   # 30
}
```

## Supported Types

`fromJson` supports all Nostos types including numeric types, records, variants, tuples, and nested structures.

```nostos
# Records with any field types
type Config = { name: String, enabled: Bool, value: Float }

# Nested types
type Address = { city: String, zip: Int }
type Person = { name: String, address: Address }

# Tuples (represented as JSON arrays)
type Pair = { data: (Int, String) }

main() = {
    # Nested record example
    json: Json = jsonParse('{"name": "Bob", "address": {"city": "Oslo", "zip": 1234}}')
    person: Person = fromJson[Person](json)
    println(person.address.city)  # "Oslo"

    # Tuple example
    json2: Json = jsonParse('{"data": [42, "hello"]}')
    pair: Pair = fromJson[Pair](json2)
    println(pair.data)  # (42, hello)
}
```

## Variants (Sum Types)

Variants use a JSON object with the constructor name as the key. Fields use `_0`, `_1`, etc. for positional values.

```nostos
type Result = Ok(Int) | Err(String)

getValue(Ok(n)) = n
getValue(Err(_)) = 0

main() = {
    # Ok(42) is represented as: {"Ok": {"_0": 42}}
    json: Json = jsonParse('{"Ok": {"_0": 42}}')
    result: Result = fromJson[Result](json)

    println(getValue(result))  # 42

    # Err("fail") is: {"Err": {"_0": "something went wrong"}}
    json2: Json = jsonParse('{"Err": {"_0": "something went wrong"}}')
    result2: Result = fromJson[Result](json2)
}

# Multi-field variants use _0, _1, _2...:
type Coord = Point(Int, Int)
# Point(10, 20) is: {"Point": {"_0": 10, "_1": 20}}

# Unit variants use null or empty object:
type Status = Active | Pending | Done
# Active is: {"Active": null} or {"Active": {}}
```

## Error Handling

`fromJson` throws catchable exceptions when the JSON doesn't match the expected type.

```nostos
type Person = { name: String, age: Int }

main() = {
    # Missing required field
    json: Json = jsonParse('{"name": "Alice"}')

    try {
        person: Person = fromJson[Person](json)
        "success: " ++ person.name
    } catch { e -> "Error: " ++ e }
    # Returns: "Error: Missing field: age"
}
```

### Common Errors

- `Missing field: <name>` - Required field not in JSON
- `Unknown constructor: <name>` - Variant constructor doesn't exist
- `Unknown type: <name>` - Type not defined
- `Expected Json Object, found <type>` - Wrong JSON structure

## Round-Trip: Value to JSON and Back

Use `reflect()` to convert a typed value to JSON, and `fromJson` to parse it back.

```nostos
type User = { id: Int, name: String, active: Bool }

main() = {
    user: User = User { id: 1, name: "Bob", active: true }

    # Convert to JSON using reflect
    json: Json = reflect(user)
    jsonStr: String = jsonStringify(json)

    # Parse back to typed value
    parsed: Json = jsonParse(jsonStr)
    user2: User = fromJson[User](parsed)

    println("Equal: " ++ show(user == user2))  # true
}
```

## Practical Example: Parsing API Responses

```nostos
type IpResponse = { origin: String }

main() = {
    (status, resp) = Http.get("https://httpbin.org/ip")

    if status == "ok" then {
        json: Json = jsonParse(resp.body)
        ipResp: IpResponse = fromJson[IpResponse](json)
        println("Your IP: " ++ ipResp.origin)
    } else {
        println("Failed to fetch IP")
    }
}
```
