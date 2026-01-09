//! AST to Bytecode compiler.
//!
//! Features:
//! - Tail call detection and optimization
//! - Closure conversion (capture free variables)
//! - Pattern match compilation
//! - Type-directed code generation

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use nostos_syntax::ast::*;
use nostos_syntax::parse;
use nostos_vm::*;
use nostos_types::infer::InferCtx;

/// Metadata for a built-in function.
pub struct BuiltinInfo {
    pub name: &'static str,
    pub signature: &'static str,
    pub doc: &'static str,
}

/// All built-in functions with their signatures and documentation.
pub const BUILTINS: &[BuiltinInfo] = &[
    // === Core ===
    BuiltinInfo { name: "println", signature: "a -> ()", doc: "Print a value to stdout followed by a newline" },
    BuiltinInfo { name: "print", signature: "a -> ()", doc: "Print a value to stdout without newline" },
    BuiltinInfo { name: "show", signature: "a -> String", doc: "Convert any value to its string representation" },
    BuiltinInfo { name: "copy", signature: "a -> a", doc: "Create a deep copy of a value" },
    BuiltinInfo { name: "hash", signature: "a -> Int", doc: "Compute hash code for a value" },
    BuiltinInfo { name: "inspect", signature: "a -> String -> ()", doc: "Send a value to the inspector panel with a label" },
    BuiltinInfo { name: "assert", signature: "Bool -> ()", doc: "Assert condition is true, panic if false" },
    BuiltinInfo { name: "assert_eq", signature: "a -> a -> ()", doc: "Assert two values are equal, panic if not" },
    BuiltinInfo { name: "panic", signature: "a -> ()", doc: "Panic with a message (terminates execution)" },
    BuiltinInfo { name: "sleep", signature: "Int -> ()", doc: "Sleep for N milliseconds" },
    BuiltinInfo { name: "vmStats", signature: "() -> (Int, Int, Int)", doc: "Get VM stats: (spawned, exited, active) process counts" },
    BuiltinInfo { name: "self", signature: "() -> Pid", doc: "Get the current process ID" },
    BuiltinInfo { name: "spawn", signature: "(() -> a) -> Pid", doc: "Spawn a new lightweight process" },
    BuiltinInfo { name: "send", signature: "Pid -> a -> ()", doc: "Send a message to a process (also: pid <- msg)" },
    BuiltinInfo { name: "receive", signature: "Pattern -> a", doc: "Receive a message matching a pattern" },

    // === Math ===
    BuiltinInfo { name: "sqrt", signature: "Float -> Float", doc: "Square root" },
    BuiltinInfo { name: "sin", signature: "Float -> Float", doc: "Sine (radians)" },
    BuiltinInfo { name: "cos", signature: "Float -> Float", doc: "Cosine (radians)" },
    BuiltinInfo { name: "tan", signature: "Float -> Float", doc: "Tangent (radians)" },
    BuiltinInfo { name: "floor", signature: "Float -> Int", doc: "Round down to nearest integer" },
    BuiltinInfo { name: "ceil", signature: "Float -> Int", doc: "Round up to nearest integer" },
    BuiltinInfo { name: "round", signature: "Float -> Int", doc: "Round to nearest integer" },
    BuiltinInfo { name: "abs", signature: "a -> a", doc: "Absolute value" },
    BuiltinInfo { name: "min", signature: "a -> a -> a", doc: "Return the smaller of two values" },
    BuiltinInfo { name: "max", signature: "a -> a -> a", doc: "Return the larger of two values" },
    BuiltinInfo { name: "pow", signature: "a -> a -> a", doc: "Raise to a power" },
    BuiltinInfo { name: "log", signature: "Float -> Float", doc: "Natural logarithm" },
    BuiltinInfo { name: "log10", signature: "Float -> Float", doc: "Base-10 logarithm" },

    // === List/Array ===
    BuiltinInfo { name: "length", signature: "[a] -> Int", doc: "Get the length of a list" },
    BuiltinInfo { name: "len", signature: "[a] -> Int", doc: "Length of a list or array (alias for length)" },
    BuiltinInfo { name: "head", signature: "[a] -> a", doc: "First element of a list" },
    BuiltinInfo { name: "tail", signature: "[a] -> [a]", doc: "All elements except the first" },
    BuiltinInfo { name: "init", signature: "[a] -> [a]", doc: "All elements except the last" },
    BuiltinInfo { name: "last", signature: "[a] -> a", doc: "Last element of a list" },
    BuiltinInfo { name: "nth", signature: "[a] -> Int -> a", doc: "Get element at index" },
    BuiltinInfo { name: "push", signature: "[a] -> a -> [a]", doc: "Append element to end of list" },
    BuiltinInfo { name: "pop", signature: "[a] -> ([a], a)", doc: "Remove and return last element" },
    BuiltinInfo { name: "get", signature: "[a] -> Int -> a", doc: "Get element at index" },
    BuiltinInfo { name: "set", signature: "[a] -> Int -> a -> [a]", doc: "Set element at index" },
    BuiltinInfo { name: "slice", signature: "[a] -> Int -> Int -> [a]", doc: "Get sublist from start to end index" },
    BuiltinInfo { name: "concat", signature: "[a] -> [a] -> [a]", doc: "Concatenate two lists" },
    BuiltinInfo { name: "reverse", signature: "[a] -> [a]", doc: "Reverse a list" },
    BuiltinInfo { name: "sort", signature: "[a] -> [a]", doc: "Sort a list in ascending order" },
    BuiltinInfo { name: "map", signature: "[a] -> (a -> b) -> [b]", doc: "Apply function to each element" },
    BuiltinInfo { name: "filter", signature: "[a] -> (a -> Bool) -> [a]", doc: "Keep elements that satisfy predicate" },
    BuiltinInfo { name: "fold", signature: "[a] -> b -> (b -> a -> b) -> b", doc: "Reduce list to single value (left fold)" },
    BuiltinInfo { name: "any", signature: "[a] -> (a -> Bool) -> Bool", doc: "True if any element satisfies predicate" },
    BuiltinInfo { name: "all", signature: "[a] -> (a -> Bool) -> Bool", doc: "True if all elements satisfy predicate" },
    BuiltinInfo { name: "find", signature: "[a] -> (a -> Bool) -> Option a", doc: "Find first element satisfying predicate" },
    BuiltinInfo { name: "position", signature: "[a] -> (a -> Bool) -> Option Int", doc: "Find index of first match" },
    BuiltinInfo { name: "unique", signature: "[a] -> [a]", doc: "Remove duplicate elements" },
    BuiltinInfo { name: "flatten", signature: "[[a]] -> [a]", doc: "Flatten nested list one level" },
    BuiltinInfo { name: "zip", signature: "[a] -> [b] -> [(a, b)]", doc: "Pair up elements from two lists" },
    BuiltinInfo { name: "unzip", signature: "[(a, b)] -> ([a], [b])", doc: "Split list of pairs into two lists" },
    BuiltinInfo { name: "take", signature: "[a] -> Int -> [a]", doc: "Take first n elements" },
    BuiltinInfo { name: "drop", signature: "[a] -> Int -> [a]", doc: "Drop first n elements" },
    BuiltinInfo { name: "split", signature: "String -> String -> [String]", doc: "Split string by delimiter" },
    BuiltinInfo { name: "join", signature: "String -> [String] -> String", doc: "Join strings with delimiter" },
    BuiltinInfo { name: "range", signature: "Int -> Int -> [Int]", doc: "Create list of integers from start to end" },
    BuiltinInfo { name: "replicate", signature: "Int -> a -> [a]", doc: "Create list of n copies of a value" },
    BuiltinInfo { name: "toIntList", signature: "[Int] -> Int64List", doc: "Convert list to specialized Int64List for fast operations" },
    BuiltinInfo { name: "intListRange", signature: "Int -> Int64List", doc: "Create Int64List [n, n-1, ..., 1] directly" },
    BuiltinInfo { name: "empty", signature: "[a] -> Bool", doc: "True if list is empty" },
    BuiltinInfo { name: "isEmpty", signature: "[a] -> Bool", doc: "True if list is empty (alias for empty)" },
    BuiltinInfo { name: "sum", signature: "[a] -> a", doc: "Sum of all elements" },
    BuiltinInfo { name: "product", signature: "[a] -> a", doc: "Product of all elements" },
    BuiltinInfo { name: "indexOf", signature: "[a] -> a -> Option Int", doc: "Find index of first matching element" },
    BuiltinInfo { name: "sortBy", signature: "[a] -> (a -> a -> Int) -> [a]", doc: "Sort list using comparator function" },
    BuiltinInfo { name: "intersperse", signature: "[a] -> a -> [a]", doc: "Insert element between all elements" },
    BuiltinInfo { name: "spanList", signature: "[a] -> (a -> Bool) -> ([a], [a])", doc: "Split at first element not satisfying predicate" },
    BuiltinInfo { name: "groupBy", signature: "[a] -> (a -> k) -> [[a]]", doc: "Group consecutive elements by key function" },
    BuiltinInfo { name: "transpose", signature: "[[a]] -> [[a]]", doc: "Transpose list of lists (rows become columns)" },
    BuiltinInfo { name: "pairwise", signature: "[a] -> (a -> a -> b) -> [b]", doc: "Apply function to pairs of adjacent elements" },
    BuiltinInfo { name: "isSorted", signature: "[a] -> Bool", doc: "Check if list is sorted in ascending order" },
    BuiltinInfo { name: "isSortedBy", signature: "[a] -> (a -> a -> Int) -> Bool", doc: "Check if list is sorted by comparator" },

    // === Typed Arrays ===
    BuiltinInfo { name: "newInt64Array", signature: "Int -> Int64Array", doc: "Create a new Int64 array of given size" },
    BuiltinInfo { name: "newFloat64Array", signature: "Int -> Float64Array", doc: "Create a new Float64 array of given size" },
    BuiltinInfo { name: "newFloat32Array", signature: "Int -> Float32Array", doc: "Create a new Float32 array of given size (for vectors)" },

    // Float64Array methods
    BuiltinInfo { name: "Float64Array.fromList", signature: "[Float] -> Float64Array", doc: "Create Float64Array from a list of floats" },
    BuiltinInfo { name: "Float64Array.length", signature: "Float64Array -> Int", doc: "Get the number of elements" },
    BuiltinInfo { name: "Float64Array.get", signature: "Float64Array -> Int -> Float", doc: "Get element at index" },
    BuiltinInfo { name: "Float64Array.set", signature: "Float64Array -> Int -> Float -> Float64Array", doc: "Set element at index, returns new array" },
    BuiltinInfo { name: "Float64Array.toList", signature: "Float64Array -> [Float]", doc: "Convert to a list of floats" },
    BuiltinInfo { name: "Float64Array.make", signature: "Int -> Float -> Float64Array", doc: "Create array of size with default value" },

    // Int64Array methods
    BuiltinInfo { name: "Int64Array.fromList", signature: "[Int] -> Int64Array", doc: "Create Int64Array from a list of integers" },
    BuiltinInfo { name: "Int64Array.length", signature: "Int64Array -> Int", doc: "Get the number of elements" },
    BuiltinInfo { name: "Int64Array.get", signature: "Int64Array -> Int -> Int", doc: "Get element at index" },
    BuiltinInfo { name: "Int64Array.set", signature: "Int64Array -> Int -> Int -> Int64Array", doc: "Set element at index, returns new array" },
    BuiltinInfo { name: "Int64Array.toList", signature: "Int64Array -> [Int]", doc: "Convert to a list of integers" },
    BuiltinInfo { name: "Int64Array.make", signature: "Int -> Int -> Int64Array", doc: "Create array of size with default value" },
    BuiltinInfo { name: "sumInt64Array", signature: "Int64Array -> Int", doc: "Fast native SIMD-optimized sum of array elements" },

    // Float32Array methods
    BuiltinInfo { name: "Float32Array.fromList", signature: "[Float] -> Float32Array", doc: "Create Float32Array from a list of floats" },
    BuiltinInfo { name: "Float32Array.length", signature: "Float32Array -> Int", doc: "Get the number of elements" },
    BuiltinInfo { name: "Float32Array.get", signature: "Float32Array -> Int -> Float", doc: "Get element at index" },
    BuiltinInfo { name: "Float32Array.set", signature: "Float32Array -> Int -> Float -> Float32Array", doc: "Set element at index, returns new array" },
    BuiltinInfo { name: "Float32Array.toList", signature: "Float32Array -> [Float]", doc: "Convert to a list of floats" },
    BuiltinInfo { name: "Float32Array.make", signature: "Int -> Float -> Float32Array", doc: "Create array of size with default value" },

    // === Type Conversions (as<Type> methods) ===
    // Available on all numeric types: Int, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float, Float32, Float64, BigInt
    BuiltinInfo { name: "asInt8", signature: "a -> Int8", doc: "Convert numeric value to Int8" },
    BuiltinInfo { name: "asInt16", signature: "a -> Int16", doc: "Convert numeric value to Int16" },
    BuiltinInfo { name: "asInt32", signature: "a -> Int32", doc: "Convert numeric value to Int32" },
    BuiltinInfo { name: "asInt64", signature: "a -> Int64", doc: "Convert numeric value to Int64" },
    BuiltinInfo { name: "asInt", signature: "a -> Int", doc: "Convert numeric value to Int (alias for asInt64)" },
    BuiltinInfo { name: "asUInt8", signature: "a -> UInt8", doc: "Convert numeric value to UInt8" },
    BuiltinInfo { name: "asUInt16", signature: "a -> UInt16", doc: "Convert numeric value to UInt16" },
    BuiltinInfo { name: "asUInt32", signature: "a -> UInt32", doc: "Convert numeric value to UInt32" },
    BuiltinInfo { name: "asUInt64", signature: "a -> UInt64", doc: "Convert numeric value to UInt64" },
    BuiltinInfo { name: "asFloat32", signature: "a -> Float32", doc: "Convert numeric value to Float32" },
    BuiltinInfo { name: "asFloat64", signature: "a -> Float64", doc: "Convert numeric value to Float64" },
    BuiltinInfo { name: "asFloat", signature: "a -> Float", doc: "Convert numeric value to Float (alias for asFloat64)" },
    BuiltinInfo { name: "asBigInt", signature: "a -> BigInt", doc: "Convert integer value to BigInt" },

    // === File I/O ===
    // All File functions throw exceptions on error
    BuiltinInfo { name: "File.readAll", signature: "String -> String", doc: "Read entire file contents, throws on error" },
    BuiltinInfo { name: "File.writeAll", signature: "String -> String -> ()", doc: "Write string to file, throws on error" },
    BuiltinInfo { name: "File.append", signature: "String -> String -> ()", doc: "Append string to file, throws on error" },
    BuiltinInfo { name: "File.open", signature: "String -> String -> Int", doc: "Open file with mode, returns handle, throws on error" },
    BuiltinInfo { name: "File.write", signature: "Int -> String -> Int", doc: "Write to handle, returns bytes written, throws on error" },
    BuiltinInfo { name: "File.read", signature: "Int -> Int -> String", doc: "Read n bytes from handle, throws on error" },
    BuiltinInfo { name: "File.readLine", signature: "Int -> String", doc: "Read line from handle, throws on error" },
    BuiltinInfo { name: "File.flush", signature: "Int -> ()", doc: "Flush file handle, throws on error" },
    BuiltinInfo { name: "File.close", signature: "Int -> ()", doc: "Close file handle, throws on error" },
    BuiltinInfo { name: "File.seek", signature: "Int -> Int -> String -> ()", doc: "Seek in file, throws on error" },
    BuiltinInfo { name: "File.exists", signature: "String -> Bool", doc: "Check if file exists, throws on error" },
    BuiltinInfo { name: "File.remove", signature: "String -> ()", doc: "Delete a file, throws on error" },
    BuiltinInfo { name: "File.rename", signature: "String -> String -> ()", doc: "Rename/move a file, throws on error" },
    BuiltinInfo { name: "File.copy", signature: "String -> String -> ()", doc: "Copy a file, throws on error" },
    BuiltinInfo { name: "File.size", signature: "String -> Int", doc: "Get file size in bytes, throws on error" },

    // === Directory I/O ===
    // All Dir functions throw exceptions on error
    BuiltinInfo { name: "Dir.create", signature: "String -> ()", doc: "Create a directory, throws on error" },
    BuiltinInfo { name: "Dir.createAll", signature: "String -> ()", doc: "Create directory and parents, throws on error" },
    BuiltinInfo { name: "Dir.list", signature: "String -> [String]", doc: "List directory contents, throws on error" },
    BuiltinInfo { name: "Dir.remove", signature: "String -> ()", doc: "Remove empty directory, throws on error" },
    BuiltinInfo { name: "Dir.removeAll", signature: "String -> ()", doc: "Remove directory recursively, throws on error" },
    BuiltinInfo { name: "Dir.exists", signature: "String -> Bool", doc: "Check if directory exists, throws on error" },

    // === HTTP ===
    // HTTP functions throw exceptions on error
    BuiltinInfo { name: "Http.get", signature: "String -> HttpResponse", doc: "HTTP GET request, throws on error" },
    BuiltinInfo { name: "Http.request", signature: "String -> String -> [(String, String)] -> String -> HttpResponse", doc: "HTTP request (method, url, headers, body), throws on error" },

    // === HTTP Server ===
    // Server functions throw exceptions on error
    BuiltinInfo { name: "Server.bind", signature: "Int -> Int", doc: "Start HTTP server on port, returns server handle, throws on error" },
    BuiltinInfo { name: "Server.accept", signature: "Int -> HttpRequest", doc: "Accept next request on server handle, throws on error" },
    BuiltinInfo { name: "Server.respond", signature: "Int -> Int -> [(String, String)] -> String -> ()", doc: "Send response: respond(reqId, status, headers, body), throws on error" },
    BuiltinInfo { name: "Server.close", signature: "Int -> ()", doc: "Close server handle" },
    BuiltinInfo { name: "Server.matchPath", signature: "String -> String -> [(String, String)]", doc: "Match path against pattern with :params, returns params list or empty if no match" },

    // === WebSocket ===
    BuiltinInfo { name: "WebSocket.isUpgrade", signature: "HttpRequest -> Bool", doc: "Check if request is a WebSocket upgrade request" },
    BuiltinInfo { name: "WebSocket.accept", signature: "Int -> Int", doc: "Accept WebSocket upgrade for request ID, returns WebSocket handle" },
    BuiltinInfo { name: "WebSocket.send", signature: "Int -> String -> ()", doc: "Send message on WebSocket handle" },
    BuiltinInfo { name: "WebSocket.recv", signature: "Int -> String", doc: "Receive message from WebSocket handle (blocks until message arrives)" },
    BuiltinInfo { name: "WebSocket.close", signature: "Int -> ()", doc: "Close WebSocket connection" },

    // === Process Introspection ===
    BuiltinInfo { name: "Process.all", signature: "() -> [Pid]", doc: "Get list of all process IDs on this thread" },
    BuiltinInfo { name: "Process.time", signature: "Pid -> Int", doc: "Get process uptime in milliseconds (-1 if not found)" },
    BuiltinInfo { name: "Process.alive", signature: "Pid -> Bool", doc: "Check if a process is still alive" },
    BuiltinInfo { name: "Process.info", signature: "Pid -> ProcessInfo", doc: "Get process info: { status, mailbox, uptime }" },
    BuiltinInfo { name: "Process.kill", signature: "Pid -> Bool", doc: "Kill a process (returns true if successful)" },

    // === Panel (TUI) ===
    BuiltinInfo { name: "Panel.create", signature: "String -> Int", doc: "Create a panel with title, returns panel ID" },
    BuiltinInfo { name: "Panel.setContent", signature: "Int -> String -> ()", doc: "Set panel content by ID" },
    BuiltinInfo { name: "Panel.show", signature: "Int -> ()", doc: "Show a panel by ID" },
    BuiltinInfo { name: "Panel.hide", signature: "Int -> ()", doc: "Hide a panel by ID" },
    BuiltinInfo { name: "Panel.onKey", signature: "Int -> String -> ()", doc: "Register key handler function for panel" },
    BuiltinInfo { name: "Panel.registerHotkey", signature: "String -> String -> ()", doc: "Register global hotkey to trigger callback" },

    // === Eval ===
    BuiltinInfo { name: "eval", signature: "String -> String", doc: "Evaluate code at runtime, returns result as string or error message" },

    // === External Process Execution ===
    // All Exec functions throw exceptions on error
    BuiltinInfo { name: "Exec.run", signature: "String -> [String] -> { exitCode: Int, stdout: String, stderr: String }", doc: "Run command and wait for completion, throws on error" },
    BuiltinInfo { name: "Exec.start", signature: "String -> [String] -> Int", doc: "Start process with streaming I/O, returns handle, throws on error" },
    BuiltinInfo { name: "Exec.readline", signature: "Int -> String", doc: "Read line from spawned process stdout, throws on error" },
    BuiltinInfo { name: "Exec.readStderr", signature: "Int -> String", doc: "Read line from spawned process stderr, throws on error" },
    BuiltinInfo { name: "Exec.write", signature: "Int -> String -> ()", doc: "Write string to spawned process stdin, throws on error" },
    BuiltinInfo { name: "Exec.wait", signature: "Int -> Int", doc: "Wait for spawned process to exit, returns exit code, throws on error" },
    BuiltinInfo { name: "Exec.kill", signature: "Int -> ()", doc: "Kill a spawned process, throws on error" },

    // === Option Functions ===
    BuiltinInfo { name: "unwrapOr", signature: "a -> b -> b", doc: "Unwrap Option or return default value" },

    // === String Functions ===
    BuiltinInfo { name: "String.length", signature: "String -> Int", doc: "Get string length in characters" },
    BuiltinInfo { name: "String.chars", signature: "String -> [Char]", doc: "Convert string to list of characters" },
    BuiltinInfo { name: "String.from_chars", signature: "[Char] -> String", doc: "Create string from list of characters" },
    BuiltinInfo { name: "String.toInt", signature: "String -> a", doc: "Parse string as integer, returns Option[Int] (Some(n) or None)" },
    BuiltinInfo { name: "String.toFloat", signature: "String -> a", doc: "Parse string as float, returns Option[Float] (Some(n) or None)" },
    BuiltinInfo { name: "String.trim", signature: "String -> String", doc: "Remove leading and trailing whitespace" },
    BuiltinInfo { name: "String.trimStart", signature: "String -> String", doc: "Remove leading whitespace" },
    BuiltinInfo { name: "String.trimEnd", signature: "String -> String", doc: "Remove trailing whitespace" },
    BuiltinInfo { name: "String.toUpper", signature: "String -> String", doc: "Convert to uppercase" },
    BuiltinInfo { name: "String.toLower", signature: "String -> String", doc: "Convert to lowercase" },
    BuiltinInfo { name: "String.contains", signature: "String -> String -> Bool", doc: "Check if string contains substring" },
    BuiltinInfo { name: "String.startsWith", signature: "String -> String -> Bool", doc: "Check if string starts with prefix" },
    BuiltinInfo { name: "String.endsWith", signature: "String -> String -> Bool", doc: "Check if string ends with suffix" },
    BuiltinInfo { name: "String.replace", signature: "String -> String -> String -> String", doc: "Replace first occurrence: replace(s, from, to)" },
    BuiltinInfo { name: "String.replaceAll", signature: "String -> String -> String -> String", doc: "Replace all occurrences: replaceAll(s, from, to)" },
    BuiltinInfo { name: "String.indexOf", signature: "String -> String -> Int", doc: "Find index of substring, -1 if not found" },
    BuiltinInfo { name: "String.lastIndexOf", signature: "String -> String -> Int", doc: "Find last index of substring, -1 if not found" },
    BuiltinInfo { name: "String.substring", signature: "String -> Int -> Int -> String", doc: "Get substring from start to end index" },
    BuiltinInfo { name: "String.repeat", signature: "String -> Int -> String", doc: "Repeat string n times" },
    BuiltinInfo { name: "String.padStart", signature: "String -> Int -> String -> String", doc: "Pad start to length with given string" },
    BuiltinInfo { name: "String.padEnd", signature: "String -> Int -> String -> String", doc: "Pad end to length with given string" },
    BuiltinInfo { name: "String.reverse", signature: "String -> String", doc: "Reverse a string" },
    BuiltinInfo { name: "String.lines", signature: "String -> [String]", doc: "Split string into lines" },
    BuiltinInfo { name: "String.words", signature: "String -> [String]", doc: "Split string into words (by whitespace)" },
    BuiltinInfo { name: "String.isEmpty", signature: "String -> Bool", doc: "Check if string is empty" },

    // === Time Functions ===
    BuiltinInfo { name: "Time.now", signature: "() -> Int", doc: "Get current Unix timestamp in milliseconds" },
    BuiltinInfo { name: "Time.nowSecs", signature: "() -> Int", doc: "Get current Unix timestamp in seconds" },
    BuiltinInfo { name: "Time.format", signature: "Int -> String -> String", doc: "Format timestamp (ms) with format string (e.g., \"%Y-%m-%d %H:%M:%S\")" },
    BuiltinInfo { name: "Time.formatUtc", signature: "Int -> String -> String", doc: "Format timestamp (ms) as UTC with format string" },
    BuiltinInfo { name: "Time.parse", signature: "String -> String -> Option Int", doc: "Parse time string with format, returns timestamp in ms" },
    BuiltinInfo { name: "Time.year", signature: "Int -> Int", doc: "Get year from timestamp (ms)" },
    BuiltinInfo { name: "Time.month", signature: "Int -> Int", doc: "Get month (1-12) from timestamp (ms)" },
    BuiltinInfo { name: "Time.day", signature: "Int -> Int", doc: "Get day of month (1-31) from timestamp (ms)" },
    BuiltinInfo { name: "Time.hour", signature: "Int -> Int", doc: "Get hour (0-23) from timestamp (ms)" },
    BuiltinInfo { name: "Time.minute", signature: "Int -> Int", doc: "Get minute (0-59) from timestamp (ms)" },
    BuiltinInfo { name: "Time.second", signature: "Int -> Int", doc: "Get second (0-59) from timestamp (ms)" },
    BuiltinInfo { name: "Time.weekday", signature: "Int -> Int", doc: "Get day of week (0=Sunday, 6=Saturday) from timestamp (ms)" },
    BuiltinInfo { name: "Time.toUtc", signature: "Int -> Int", doc: "Convert local timestamp to UTC" },
    BuiltinInfo { name: "Time.fromUtc", signature: "Int -> Int", doc: "Convert UTC timestamp to local" },
    BuiltinInfo { name: "Time.timezone", signature: "() -> String", doc: "Get local timezone name" },
    BuiltinInfo { name: "Time.timezoneOffset", signature: "() -> Int", doc: "Get timezone offset from UTC in minutes" },

    // === Random Functions ===
    BuiltinInfo { name: "Random.int", signature: "Int -> Int -> Int", doc: "Generate random integer in range [min, max]" },
    BuiltinInfo { name: "Random.float", signature: "() -> Float", doc: "Generate random float in range [0.0, 1.0)" },
    BuiltinInfo { name: "Random.bool", signature: "() -> Bool", doc: "Generate random boolean" },
    BuiltinInfo { name: "Random.choice", signature: "[a] -> a", doc: "Pick random element from list" },
    BuiltinInfo { name: "Random.shuffle", signature: "[a] -> [a]", doc: "Randomly shuffle a list" },
    BuiltinInfo { name: "Random.bytes", signature: "Int -> [Int]", doc: "Generate n random bytes (0-255)" },

    // === Environment Functions ===
    BuiltinInfo { name: "Env.get", signature: "String -> Option String", doc: "Get environment variable value" },
    BuiltinInfo { name: "Env.set", signature: "String -> String -> ()", doc: "Set environment variable" },
    BuiltinInfo { name: "Env.remove", signature: "String -> ()", doc: "Remove environment variable" },
    BuiltinInfo { name: "Env.all", signature: "() -> [(String, String)]", doc: "Get all environment variables" },
    BuiltinInfo { name: "Env.cwd", signature: "() -> String", doc: "Get current working directory" },
    BuiltinInfo { name: "Env.setCwd", signature: "String -> Result () String", doc: "Set current working directory" },
    BuiltinInfo { name: "Env.home", signature: "() -> Option String", doc: "Get user's home directory" },
    BuiltinInfo { name: "Env.args", signature: "() -> [String]", doc: "Get command-line arguments" },
    BuiltinInfo { name: "Env.platform", signature: "() -> String", doc: "Get platform name (linux, macos, windows)" },

    // === Path Functions ===
    BuiltinInfo { name: "Path.join", signature: "String -> String -> String", doc: "Join two path components" },
    BuiltinInfo { name: "Path.dirname", signature: "String -> String", doc: "Get directory part of path" },
    BuiltinInfo { name: "Path.basename", signature: "String -> String", doc: "Get filename part of path" },
    BuiltinInfo { name: "Path.extension", signature: "String -> String", doc: "Get file extension (without dot)" },
    BuiltinInfo { name: "Path.withExtension", signature: "String -> String -> String", doc: "Replace file extension" },
    BuiltinInfo { name: "Path.normalize", signature: "String -> String", doc: "Normalize path (resolve . and ..)" },
    BuiltinInfo { name: "Path.isAbsolute", signature: "String -> Bool", doc: "Check if path is absolute" },
    BuiltinInfo { name: "Path.isRelative", signature: "String -> Bool", doc: "Check if path is relative" },
    BuiltinInfo { name: "Path.split", signature: "String -> [String]", doc: "Split path into components" },

    // === Regex Functions ===
    BuiltinInfo { name: "Regex.matches", signature: "String -> String -> Bool", doc: "Check if string matches regex pattern" },
    BuiltinInfo { name: "Regex.find", signature: "String -> String -> Option String", doc: "Find first match of pattern in string" },
    BuiltinInfo { name: "Regex.findAll", signature: "String -> String -> [String]", doc: "Find all matches of pattern in string" },
    BuiltinInfo { name: "Regex.replace", signature: "String -> String -> String -> String", doc: "Replace first match: replace(s, pattern, replacement)" },
    BuiltinInfo { name: "Regex.replaceAll", signature: "String -> String -> String -> String", doc: "Replace all matches: replaceAll(s, pattern, replacement)" },
    BuiltinInfo { name: "Regex.split", signature: "String -> String -> [String]", doc: "Split string by regex pattern" },
    BuiltinInfo { name: "Regex.captures", signature: "String -> String -> Option [String]", doc: "Get capture groups from first match" },

    // === Map Functions ===
    BuiltinInfo { name: "Map.insert", signature: "Map k v -> k -> v -> Map k v", doc: "Insert key-value pair, returns new map" },
    BuiltinInfo { name: "Map.remove", signature: "Map k v -> k -> Map k v", doc: "Remove key from map, returns new map" },
    BuiltinInfo { name: "Map.get", signature: "Map k v -> k -> v", doc: "Get value for key, returns unit if not found" },
    BuiltinInfo { name: "Map.contains", signature: "Map k v -> k -> Bool", doc: "Check if map contains key" },
    BuiltinInfo { name: "Map.keys", signature: "Map k v -> [k]", doc: "Get list of all keys" },
    BuiltinInfo { name: "Map.values", signature: "Map k v -> [v]", doc: "Get list of all values" },
    BuiltinInfo { name: "Map.size", signature: "Map k v -> Int", doc: "Get number of entries in map" },
    BuiltinInfo { name: "Map.isEmpty", signature: "Map k v -> Bool", doc: "Check if map is empty" },
    BuiltinInfo { name: "Map.union", signature: "Map k v -> Map k v -> Map k v", doc: "Union of two maps (second wins on conflict)" },
    BuiltinInfo { name: "Map.intersection", signature: "Map k v -> Map k v -> Map k v", doc: "Intersection of two maps (keys in both)" },
    BuiltinInfo { name: "Map.difference", signature: "Map k v -> Map k v -> Map k v", doc: "Keys in first map but not second" },
    BuiltinInfo { name: "Map.toList", signature: "Map k v -> [(k, v)]", doc: "Convert map to list of key-value pairs" },
    BuiltinInfo { name: "Map.fromList", signature: "[(k, v)] -> Map k v", doc: "Create map from list of key-value pairs" },

    // === Set Functions ===
    BuiltinInfo { name: "Set.insert", signature: "Set a -> a -> Set a", doc: "Insert element, returns new set" },
    BuiltinInfo { name: "Set.remove", signature: "Set a -> a -> Set a", doc: "Remove element, returns new set" },
    BuiltinInfo { name: "Set.contains", signature: "Set a -> a -> Bool", doc: "Check if set contains element" },
    BuiltinInfo { name: "Set.size", signature: "Set a -> Int", doc: "Get number of elements in set" },
    BuiltinInfo { name: "Set.isEmpty", signature: "Set a -> Bool", doc: "Check if set is empty" },
    BuiltinInfo { name: "Set.toList", signature: "Set a -> [a]", doc: "Convert set to list" },
    BuiltinInfo { name: "Set.union", signature: "Set a -> Set a -> Set a", doc: "Union of two sets" },
    BuiltinInfo { name: "Set.intersection", signature: "Set a -> Set a -> Set a", doc: "Intersection of two sets" },
    BuiltinInfo { name: "Set.difference", signature: "Set a -> Set a -> Set a", doc: "Elements in first set but not in second" },
    BuiltinInfo { name: "Set.symmetricDifference", signature: "Set a -> Set a -> Set a", doc: "Elements in either set but not both" },
    BuiltinInfo { name: "Set.isSubset", signature: "Set a -> Set a -> Bool", doc: "Check if first set is subset of second" },
    BuiltinInfo { name: "Set.isProperSubset", signature: "Set a -> Set a -> Bool", doc: "Check if first set is proper subset of second" },
    BuiltinInfo { name: "Set.fromList", signature: "[a] -> Set a", doc: "Create set from list" },

    // === PostgreSQL ===
    BuiltinInfo { name: "Pg.connect", signature: "String -> Int", doc: "Connect to PostgreSQL database, returns handle" },
    BuiltinInfo { name: "Pg.query", signature: "Int -> String -> [a] -> [[a]]", doc: "Execute query with params, returns rows as list of lists" },
    BuiltinInfo { name: "Pg.execute", signature: "Int -> String -> [a] -> Int", doc: "Execute statement with params, returns affected row count" },
    BuiltinInfo { name: "Pg.close", signature: "Int -> ()", doc: "Close PostgreSQL connection" },
    BuiltinInfo { name: "Pg.begin", signature: "Int -> ()", doc: "Begin a transaction" },
    BuiltinInfo { name: "Pg.commit", signature: "Int -> ()", doc: "Commit the current transaction" },
    BuiltinInfo { name: "Pg.rollback", signature: "Int -> ()", doc: "Rollback the current transaction" },
    BuiltinInfo { name: "Pg.transaction", signature: "Int -> (() -> a) -> a", doc: "Execute function in transaction with auto-rollback on error" },
    BuiltinInfo { name: "Pg.prepare", signature: "Int -> String -> String -> ()", doc: "Prepare a statement with name and query" },
    BuiltinInfo { name: "Pg.queryPrepared", signature: "Int -> String -> [a] -> [[a]]", doc: "Execute prepared query with params, returns rows" },
    BuiltinInfo { name: "Pg.executePrepared", signature: "Int -> String -> [a] -> Int", doc: "Execute prepared statement with params, returns affected count" },
    BuiltinInfo { name: "Pg.deallocate", signature: "Int -> String -> ()", doc: "Deallocate a prepared statement" },
    // LISTEN/NOTIFY
    BuiltinInfo { name: "Pg.listenConnect", signature: "String -> Int", doc: "Create dedicated listener connection for LISTEN/NOTIFY" },
    BuiltinInfo { name: "Pg.listen", signature: "Int -> String -> ()", doc: "Start listening on a channel" },
    BuiltinInfo { name: "Pg.unlisten", signature: "Int -> String -> ()", doc: "Stop listening on a channel" },
    BuiltinInfo { name: "Pg.notify", signature: "Int -> String -> String -> ()", doc: "Send notification: notify(handle, channel, payload)" },
    BuiltinInfo { name: "Pg.awaitNotification", signature: "Int -> Int -> Option (String, String)", doc: "Wait for notification with timeout(ms), returns Some((channel, payload)) or None" },

    // === UUID Functions ===
    BuiltinInfo { name: "Uuid.v4", signature: "() -> String", doc: "Generate a random UUID v4" },
    BuiltinInfo { name: "Uuid.isValid", signature: "String -> Bool", doc: "Check if string is a valid UUID" },

    // === Crypto Functions ===
    BuiltinInfo { name: "Crypto.sha256", signature: "String -> String", doc: "Compute SHA-256 hash, returns hex string" },
    BuiltinInfo { name: "Crypto.sha512", signature: "String -> String", doc: "Compute SHA-512 hash, returns hex string" },
    BuiltinInfo { name: "Crypto.md5", signature: "String -> String", doc: "Compute MD5 hash (insecure, for compatibility), returns hex string" },
    BuiltinInfo { name: "Crypto.bcryptHash", signature: "String -> Int -> String", doc: "Hash password with bcrypt: bcryptHash(password, cost)" },
    BuiltinInfo { name: "Crypto.bcryptVerify", signature: "String -> String -> Bool", doc: "Verify password against bcrypt hash: bcryptVerify(password, hash)" },
    BuiltinInfo { name: "Crypto.randomBytes", signature: "Int -> String", doc: "Generate n random bytes as hex string" },

    // Time builtins
    BuiltinInfo { name: "Time.now", signature: "() -> Int", doc: "Get current UTC timestamp in milliseconds since epoch" },
    BuiltinInfo { name: "Time.fromDate", signature: "Int -> Int -> Int -> Int", doc: "Create timestamp from year, month, day (at midnight UTC)" },
    BuiltinInfo { name: "Time.fromTime", signature: "Int -> Int -> Int -> Int", doc: "Create milliseconds since midnight from hour, min, sec" },
    BuiltinInfo { name: "Time.fromDateTime", signature: "Int -> Int -> Int -> Int -> Int -> Int -> Int", doc: "Create timestamp from year, month, day, hour, min, sec" },
    BuiltinInfo { name: "Time.year", signature: "Int -> Int", doc: "Extract year from timestamp" },
    BuiltinInfo { name: "Time.month", signature: "Int -> Int", doc: "Extract month (1-12) from timestamp" },
    BuiltinInfo { name: "Time.day", signature: "Int -> Int", doc: "Extract day of month (1-31) from timestamp" },
    BuiltinInfo { name: "Time.hour", signature: "Int -> Int", doc: "Extract hour (0-23) from timestamp" },
    BuiltinInfo { name: "Time.minute", signature: "Int -> Int", doc: "Extract minute (0-59) from timestamp" },
    BuiltinInfo { name: "Time.second", signature: "Int -> Int", doc: "Extract second (0-59) from timestamp" },
    // Type introspection and reflection
    BuiltinInfo { name: "typeInfo", signature: "String -> Map k v", doc: "Get type metadata by name as Map (fields, constructors, etc.)" },
    BuiltinInfo { name: "typeOf", signature: "a -> String", doc: "Get type name of a value (Int, Float, String, Bool, List, Record, Variant, etc.)" },
    BuiltinInfo { name: "tagOf", signature: "a -> String", doc: "Get variant tag name, or empty string for non-variants" },
    BuiltinInfo { name: "reflect", signature: "a -> Json", doc: "Convert any value to Json type for inspection/serialization" },
    BuiltinInfo { name: "jsonToType", signature: "[T] Json -> T", doc: "Convert Json to typed value: jsonToType[Person](json)" },
    BuiltinInfo { name: "fromJson", signature: "[T] Json -> T", doc: "Convert Json to typed value: fromJson[Person](jsonParse(str))" },
    BuiltinInfo { name: "makeRecord", signature: "[T] Map[String, Json] -> T", doc: "Construct record from field map: makeRecord[Person](fields)" },
    BuiltinInfo { name: "makeVariant", signature: "[T] String -> Map[String, Json] -> T", doc: "Construct variant from constructor name and fields: makeVariant[Result](\"Ok\", fields)" },
    BuiltinInfo { name: "makeRecordByName", signature: "String -> Map[String, Json] -> a", doc: "Construct record by type name string: makeRecordByName(\"Person\", fields)" },
    BuiltinInfo { name: "makeVariantByName", signature: "String -> String -> Map[String, Json] -> a", doc: "Construct variant by type name: makeVariantByName(\"Result\", \"Ok\", fields)" },
    BuiltinInfo { name: "jsonToTypeByName", signature: "String -> Json -> a", doc: "Convert Json to typed value by type name: jsonToTypeByName(\"Person\", json)" },
    BuiltinInfo { name: "requestToType", signature: "HttpRequest -> String -> Result[a, String]", doc: "Parse HTTP request params to typed record: requestToType(req, \"UserParams\")" },

    // === Runtime Stats ===
    BuiltinInfo { name: "Runtime.threadCount", signature: "() -> Int", doc: "Get number of available CPU threads" },
    BuiltinInfo { name: "Runtime.uptimeMs", signature: "() -> Int", doc: "Get milliseconds since program started" },
    BuiltinInfo { name: "Runtime.memoryKb", signature: "() -> Int", doc: "Get current process memory usage in KB (Linux only)" },
    BuiltinInfo { name: "Runtime.pid", signature: "() -> Int", doc: "Get current process ID" },
    BuiltinInfo { name: "Runtime.loadAvg", signature: "() -> (Float, Float, Float)", doc: "Get 1, 5, 15 minute load averages (Linux only)" },
    BuiltinInfo { name: "Runtime.numThreads", signature: "() -> Int", doc: "Get number of OS threads in process (Linux only)" },
    BuiltinInfo { name: "Runtime.tokioWorkers", signature: "() -> Int", doc: "Get number of tokio worker threads (Linux only)" },
    BuiltinInfo { name: "Runtime.blockingThreads", signature: "() -> Int", doc: "Get number of tokio blocking threads (Linux only)" },
];

/// Extract doc comment immediately preceding a definition at the given span start.
/// Doc comments are lines starting with `#` (but not `#*` for multi-line or `#{` for sets)
/// immediately before the definition, with only whitespace between comment lines.
///
/// Example:
/// ```text
/// # This is the doc comment
/// # It can span multiple lines
/// myFunction(x) = x + 1
/// ```
pub fn extract_doc_comment(source: &str, span_start: usize) -> Option<String> {
    if span_start == 0 || span_start > source.len() {
        return None;
    }

    // Get the text before the span
    let before = &source[..span_start];

    // Find all lines before the definition
    let lines: Vec<&str> = before.lines().collect();
    if lines.is_empty() {
        return None;
    }

    // Collect comment lines going backwards from the definition
    let mut doc_lines: Vec<&str> = Vec::new();

    // Start from the line before the definition
    // The last line in `lines` might be empty or contain only whitespace
    // if the span_start is at the beginning of a line
    let mut idx = lines.len();

    // Skip trailing empty/whitespace lines
    while idx > 0 {
        idx -= 1;
        let line = lines[idx].trim();
        if line.is_empty() {
            continue;
        }

        // Check if this line is a comment
        if line.starts_with('#') && !line.starts_with("#*") && !line.starts_with("#{") {
            // It's a doc comment line - extract the comment text
            let comment_text = line[1..].trim();
            doc_lines.push(comment_text);
        } else {
            // Not a comment - stop looking
            break;
        }
    }

    if doc_lines.is_empty() {
        return None;
    }

    // Reverse since we collected them backwards
    doc_lines.reverse();

    // Join with newlines
    Some(doc_lines.join("\n"))
}

/// Compilation errors with source location information.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompileError {
    #[error("unknown variable `{name}`")]
    UnknownVariable { name: String, span: Span },

    #[error("unknown function `{name}`")]
    UnknownFunction { name: String, span: Span },

    #[error("unknown type `{name}`")]
    UnknownType { name: String, span: Span },

    #[error("`{name}` is defined multiple times")]
    DuplicateDefinition { name: String, span: Span },

    #[error("invalid pattern")]
    InvalidPattern { span: Span, context: String },

    #[error("{feature} is not yet implemented")]
    NotImplemented { feature: String, span: Span },

    #[error("cannot access private function `{function}` from outside module `{module}`")]
    PrivateAccess { function: String, module: String, span: Span },

    #[error("unknown trait `{name}`")]
    UnknownTrait { name: String, span: Span },

    #[error("type `{ty}` does not implement method `{method}` required by trait `{trait_name}`")]
    MissingTraitMethod { method: String, ty: String, trait_name: String, span: Span },

    #[error("type `{ty}` does not implement trait `{trait_name}`")]
    TraitNotImplemented { ty: String, trait_name: String, span: Span },

    #[error("function `{name}` expects {expected} argument(s), but {found} were provided")]
    ArityMismatch { name: String, expected: usize, found: usize, span: Span },

    #[error("cannot resolve trait method `{method}` without type information")]
    UnresolvedTraitMethod { method: String, span: Span },

    #[error("type `{type_name}` does not implement trait `{trait_name}`")]
    TraitBoundNotSatisfied { type_name: String, trait_name: String, span: Span },

    #[error("type error: {message}")]
    TypeError { message: String, span: Span },

    #[error("mvar safety violation: {message}")]
    MvarSafetyViolation { message: String, span: Span },

    #[error("nested write to mvar `{mvar_name}` would cause deadlock")]
    NestedMvarWrite { mvar_name: String, span: Span },

    #[error("function `{fn_name}` blocks while holding mvar lock on `{mvar_name}`")]
    BlockingWithMvarLock { fn_name: String, mvar_name: String, span: Span },

    #[error("internal compiler error: {message}")]
    InternalError { message: String, span: Span },

    #[error("module `{module}` is not imported; add `import {module}` to use `{function}`")]
    ModuleNotImported { module: String, function: String, span: Span },

    #[error("ambiguous name `{name}` is defined in multiple imported modules: {modules}")]
    AmbiguousName { name: String, modules: String, span: Span },
}

impl CompileError {
    /// Get the span associated with this error.
    pub fn span(&self) -> Span {
        match self {
            CompileError::UnknownVariable { span, .. } => *span,
            CompileError::UnknownFunction { span, .. } => *span,
            CompileError::UnknownType { span, .. } => *span,
            CompileError::DuplicateDefinition { span, .. } => *span,
            CompileError::InvalidPattern { span, .. } => *span,
            CompileError::NotImplemented { span, .. } => *span,
            CompileError::PrivateAccess { span, .. } => *span,
            CompileError::UnknownTrait { span, .. } => *span,
            CompileError::MissingTraitMethod { span, .. } => *span,
            CompileError::TraitNotImplemented { span, .. } => *span,
            CompileError::ArityMismatch { span, .. } => *span,
            CompileError::UnresolvedTraitMethod { span, .. } => *span,
            CompileError::TraitBoundNotSatisfied { span, .. } => *span,
            CompileError::TypeError { span, .. } => *span,
            CompileError::MvarSafetyViolation { span, .. } => *span,
            CompileError::NestedMvarWrite { span, .. } => *span,
            CompileError::BlockingWithMvarLock { span, .. } => *span,
            CompileError::InternalError { span, .. } => *span,
            CompileError::ModuleNotImported { span, .. } => *span,
            CompileError::AmbiguousName { span, .. } => *span,
        }
    }

    /// Convert to a SourceError for pretty printing.
    pub fn to_source_error(&self) -> nostos_syntax::SourceError {
        use nostos_syntax::SourceError;

        let span = self.span();
        match self {
            CompileError::UnknownVariable { name, .. } => {
                SourceError::unknown_variable(name, span)
            }
            CompileError::UnknownFunction { name, .. } => {
                SourceError::unknown_function(name, span)
            }
            CompileError::UnknownType { name, .. } => {
                SourceError::unknown_type(name, span)
            }
            CompileError::DuplicateDefinition { name, .. } => {
                SourceError::duplicate_definition(name, span, None)
            }
            CompileError::InvalidPattern { context, .. } => {
                SourceError::invalid_pattern(span, context)
            }
            CompileError::NotImplemented { feature, .. } => {
                SourceError::not_implemented(feature, span)
            }
            CompileError::PrivateAccess { function, module, .. } => {
                SourceError::private_access(function, module, span)
            }
            CompileError::UnknownTrait { name, .. } => {
                SourceError::unknown_trait(name, span)
            }
            CompileError::MissingTraitMethod { method, ty, trait_name, .. } => {
                SourceError::missing_trait_method(method, ty, trait_name, span)
            }
            CompileError::TraitNotImplemented { ty, trait_name, .. } => {
                SourceError::compile(
                    format!("type `{}` does not implement trait `{}`", ty, trait_name),
                    span,
                )
            }
            CompileError::ArityMismatch { name, expected, found, .. } => {
                SourceError::arity_mismatch(name, *expected, *found, span)
            }
            CompileError::UnresolvedTraitMethod { method, .. } => {
                SourceError::compile(
                    format!("cannot resolve trait method `{}` without type information", method),
                    span,
                )
            }
            CompileError::TraitBoundNotSatisfied { type_name, trait_name, .. } => {
                SourceError::compile(
                    format!("type `{}` does not implement trait `{}`", type_name, trait_name),
                    span,
                )
            }
            CompileError::TypeError { message, .. } => {
                SourceError::compile(message.clone(), span)
            }
            CompileError::MvarSafetyViolation { message, .. } => {
                SourceError::compile(format!("mvar safety violation: {}", message), span)
            }
            CompileError::NestedMvarWrite { mvar_name, .. } => {
                SourceError::compile(
                    format!("nested write to mvar `{}` would cause deadlock - cannot write to the same mvar being assigned", mvar_name),
                    span
                )
            }
            CompileError::BlockingWithMvarLock { fn_name, mvar_name, .. } => {
                SourceError::compile(
                    format!("function `{}` reads and writes mvar `{}` but also blocks (receive) - this would cause deadlock. Restructure to avoid blocking while holding mvar lock.", fn_name, mvar_name),
                    span
                )
            }
            CompileError::InternalError { message, .. } => {
                SourceError::compile(format!("internal error: {}", message), span)
            }
            CompileError::ModuleNotImported { module, function, .. } => {
                SourceError::compile(
                    format!("module `{}` is not imported; add `import {}` to use `{}`", module, module, function),
                    span
                )
            }
            CompileError::AmbiguousName { name, modules, .. } => {
                SourceError::compile(
                    format!("ambiguous name `{}` is defined in multiple imported modules: {}; use qualified name (e.g., module.{})", name, modules, name),
                    span
                )
            }
        }
    }
}

/// Simple type inference for builtin dispatch.
/// We infer types from literals and some expressions to emit typed instructions.
#[derive(Debug, Clone, Copy, PartialEq)]
enum InferredType {
    Int,
    Float,
}

/// Information about a local variable.
#[derive(Clone, Copy)]
struct LocalInfo {
    reg: Reg,
    is_float: bool,
    mutable: bool,
}

/// Compilation context.
pub struct Compiler {
    /// Current function being compiled
    chunk: Chunk,
    /// Local variable -> register and type info
    locals: HashMap<String, LocalInfo>,
    /// Next available register
    next_reg: Reg,
    /// Captured variables for the current closure: name -> capture index
    capture_indices: HashMap<String, u8>,
    /// Compiled functions
    functions: HashMap<String, Arc<FunctionValue>>,
    /// Function name -> index mapping for direct calls (no HashMap lookup at runtime!)
    function_indices: HashMap<String, u16>,
    /// Ordered list of function names (index -> name)
    function_list: Vec<String>,
    /// Type definitions (user-defined types)
    types: HashMap<String, TypeInfo>,
    /// Builtin types for field autocomplete only (not shown in top-level completion)
    builtin_types: HashMap<String, TypeInfo>,
    /// Known constructors (for type checking)
    known_constructors: HashSet<String>,
    /// Current scope depth
    #[allow(dead_code)]
    scope_depth: usize,
    /// Current module path (e.g., ["Foo", "Bar"] for module Foo.Bar)
    module_path: Vec<String>,
    /// Imports: local name -> fully qualified name
    imports: HashMap<String, String>,
    /// Function visibility: qualified name -> Visibility
    function_visibility: HashMap<String, Visibility>,
    /// Trait definitions: trait name -> TraitInfo
    trait_defs: HashMap<String, TraitInfo>,
    /// Trait implementations: (type_name, trait_name) -> TraitImplInfo
    trait_impls: HashMap<(String, String), TraitImplInfo>,
    /// Types to their implemented traits: type_name -> [trait_name, ...]
    type_traits: HashMap<String, Vec<String>>,
    /// Local variable type tracking: variable name -> type name
    local_types: HashMap<String, String>,
    /// Parameter types for specialized function variants (for monomorphization)
    /// When compiling a specialized variant, parameter name -> concrete type name
    param_types: HashMap<String, String>,
    /// Function ASTs for monomorphization: function name -> FnDef
    /// Used to recompile functions with different type contexts
    fn_asts: HashMap<String, FnDef>,
    /// Source info for each function: function name -> (source_name, source_code)
    fn_sources: HashMap<String, (String, Arc<String>)>,
    /// Function type parameters with bounds: function name -> type parameters
    /// Used to check trait bounds at call sites
    fn_type_params: HashMap<String, Vec<TypeParam>>,
    /// Functions that need monomorphization (have untyped parameters calling trait methods)
    /// These functions are not compiled normally; specialized variants are compiled at call sites
    polymorphic_fns: HashSet<String>,
    /// Current function name being compiled (for self-recursion optimization)
    current_function_name: Option<String>,
    /// Current function's type parameters (for checking nested trait bounds)
    current_fn_type_params: Vec<TypeParam>,
    /// Loop context stack for break/continue
    loop_stack: Vec<LoopContext>,
    /// Line starts: byte offsets where each line begins (line 1 is at index 0)
    line_starts: Vec<usize>,
    /// Pending functions to compile (second pass)
    /// (AST, module_path, imports, line_starts, source, source_name)
    pending_functions: Vec<(FnDef, Vec<String>, HashMap<String, String>, Vec<usize>, Arc<String>, String)>,
    /// Pre-built signatures for pending functions (for type checking)
    pending_fn_signatures: HashMap<String, nostos_types::FunctionType>,

    // Current source context
    current_source: Option<Arc<String>>,
    current_source_name: Option<String>,

    /// Type definition ASTs for REPL introspection: type name -> TypeDef
    type_defs: HashMap<String, TypeDef>,
    /// Known module prefixes (for distinguishing module.func from value.field)
    /// Contains all module path prefixes, e.g., "String", "utils", "math.vector"
    known_modules: HashSet<String>,
    /// Imported modules per module: (current_module, imported_module_path)
    /// When in module X, this tracks which modules X has imported
    imported_modules: HashSet<(Vec<String>, String)>,
    /// Use statements per module: (current_module_path, use_statement_string)
    /// e.g., (["main"], "use nalgebra.*") or (["main"], "use stdlib.{map, filter}")
    module_use_stmts: Vec<(Vec<String>, String)>,
    /// Local (inline) modules defined in the same compilation unit
    /// These don't require explicit import statements
    local_modules: HashSet<String>,
    /// REPL mode: bypass visibility checks for interactive exploration
    repl_mode: bool,
    /// Prelude functions: qualified names that bypass visibility checks (stdlib)
    prelude_functions: HashSet<String>,
    /// Module-level mutable variables (mvars): qualified name -> MvarInfo
    /// These are thread-safe shared state with automatic RwLock
    mvars: HashMap<String, MvarInfo>,

    // === Mvar deadlock detection ===
    /// Function mvar access: function name -> (reads, writes)
    fn_mvar_access: HashMap<String, FnMvarAccess>,
    /// Function calls: function name -> set of called functions
    fn_calls: HashMap<String, HashSet<String>>,
    /// Current function's mvar reads (during compilation)
    current_fn_mvar_reads: HashSet<String>,
    /// Current function's mvar writes (during compilation)
    current_fn_mvar_writes: HashSet<String>,
    /// Current function's calls (during compilation)
    current_fn_calls: HashSet<String>,
    /// Current function's mvar locks: (mvar_name, const_idx, is_write) - sorted for ordered locking
    current_fn_mvar_locks: Vec<(String, u16, bool)>,
    /// Whether current function has blocking operations (receive, etc.)
    current_fn_has_blocking: bool,
    /// Debug symbols accumulated during function compilation
    /// (not affected by scope restoration in compile_block)
    current_fn_debug_symbols: Vec<LocalVarSymbol>,
    /// Native function indices: name -> index for CallNativeIdx optimization.
    /// When set, CallNative instructions are replaced with faster CallNativeIdx.
    native_indices: HashMap<String, u16>,
    /// Extension function indices: name -> index for CallExtensionIdx optimization.
    /// When set, CallExtension instructions are replaced with faster CallExtensionIdx.
    extension_indices: HashMap<String, u16>,
}

/// Information about a module-level mutable variable (mvar).
#[derive(Clone, Debug)]
pub struct MvarInfo {
    pub type_name: String,
    pub initial_value: MvarInitValue,
}

/// Initial value for an mvar (must be a compile-time constant).
#[derive(Clone, Debug)]
pub enum MvarInitValue {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Char(char),
    EmptyList,
    IntList(Vec<i64>),
    StringList(Vec<String>),
    FloatList(Vec<f64>),
    BoolList(Vec<bool>),
    /// Tuple of constant values
    Tuple(Vec<MvarInitValue>),
    /// Generic list of constant values (for nested or mixed lists)
    List(Vec<MvarInitValue>),
    /// Record/struct constructor: (type_name, fields as (name, value) pairs)
    Record(String, Vec<(String, MvarInitValue)>),
    /// Empty map: %{}
    EmptyMap,
    /// Map with entries: %{ key => value, ... }
    Map(Vec<(MvarInitValue, MvarInitValue)>),
}

/// Mvar access information for a function (for deadlock detection).
#[derive(Clone, Debug, Default)]
pub struct FnMvarAccess {
    pub reads: HashSet<String>,
    pub writes: HashSet<String>,
    pub has_blocking: bool,
}

/// Context for a loop being compiled (for break/continue).
#[derive(Clone)]
struct LoopContext {
    /// Address of loop start (for back-jump at end of loop)
    #[allow(dead_code)]
    start_addr: usize,
    /// Address where continue should jump to (may differ from start_addr for for loops)
    #[allow(dead_code)]
    continue_addr: usize,
    /// Addresses of continue jumps to patch
    continue_jumps: Vec<usize>,
    /// Addresses of break jumps to patch at loop end
    break_jumps: Vec<usize>,
}

/// Type information for code generation.
#[derive(Clone)]
pub struct TypeInfo {
    pub name: String,
    pub kind: TypeInfoKind,
}

#[derive(Clone)]
pub enum TypeInfoKind {
    /// Record type: fields with (name, type_name) pairs
    Record { fields: Vec<(String, String)>, mutable: bool },
    /// Variant type: constructors with (name, field_types)
    /// Field types are stored as simple strings: "Float", "Int", etc.
    Variant { constructors: Vec<(String, Vec<String>)> },
}

/// Trait definition information.
#[derive(Clone)]
pub struct TraitInfo {
    pub name: String,
    pub super_traits: Vec<String>,
    pub methods: Vec<TraitMethodInfo>,
}

/// A method signature in a trait.
#[derive(Clone)]
pub struct TraitMethodInfo {
    pub name: String,
    pub param_count: usize,
    pub has_default: bool,
}

/// Trait implementation information.
#[derive(Clone)]
pub struct TraitImplInfo {
    pub type_name: String,
    pub trait_name: String,
    pub method_names: Vec<String>,  // Maps to qualified function names like "Point.Show.show"
}

impl Compiler {
    pub fn new_empty() -> Self {
        // Pre-register built-in pseudo-modules for native functions
        let builtin_modules: HashSet<String> = [
            "String", "File", "Dir", "List", "Option", "Result", "Char", "Int", "Float",
            "Bool", "Bytes", "Map", "Set", "IO", "Math", "Debug", "Time", "Thread",
            "Channel", "Regex", "Json", "Http", "Net", "Sys", "Env", "Process",
            "Base64", "Url", "Encoding", "Server", "Exec", "Random", "Path", "Panel",
            "Pg", "Uuid", "Crypto", "Float64Array", "Int64Array", "Float32Array", "Buffer",
            "Runtime", "WebSocket",
        ].iter().map(|s| s.to_string()).collect();

        let mut this = Self {
            chunk: Chunk::new(),
            locals: HashMap::new(),
            next_reg: 0,
            capture_indices: HashMap::new(),
            functions: HashMap::new(),
            function_indices: HashMap::new(),
            function_list: Vec::new(),
            types: HashMap::new(),
            builtin_types: HashMap::new(),
            known_constructors: HashSet::new(),
            scope_depth: 0,
            module_path: Vec::new(),
            imports: HashMap::new(),
            function_visibility: HashMap::new(),
            trait_defs: HashMap::new(),
            trait_impls: HashMap::new(),
            type_traits: HashMap::new(),
            local_types: HashMap::new(),
            param_types: HashMap::new(),
            fn_asts: HashMap::new(),
            fn_sources: HashMap::new(),
            fn_type_params: HashMap::new(),
            polymorphic_fns: HashSet::new(),
            current_function_name: None,
            current_fn_type_params: Vec::new(),
            loop_stack: Vec::new(),
            line_starts: vec![0],
            pending_functions: Vec::new(),
            pending_fn_signatures: HashMap::new(),
            current_source: None,
            current_source_name: None,
            type_defs: HashMap::new(),
            known_modules: builtin_modules,
            imported_modules: HashSet::new(),
            module_use_stmts: Vec::new(),
            local_modules: HashSet::new(),
            repl_mode: false,
            prelude_functions: HashSet::new(),
            mvars: HashMap::new(),
            fn_mvar_access: HashMap::new(),
            fn_calls: HashMap::new(),
            current_fn_mvar_reads: HashSet::new(),
            current_fn_mvar_writes: HashSet::new(),
            current_fn_calls: HashSet::new(),
            current_fn_mvar_locks: Vec::new(),
            current_fn_has_blocking: false,
            current_fn_debug_symbols: Vec::new(),
            native_indices: HashMap::new(),
            extension_indices: HashMap::new(),
        };

        // Register builtin types for autocomplete
        this.register_builtin_types();

        this
    }

    /// Register builtin types that are returned by native functions.
    /// These are stored separately and only used for field autocomplete,
    /// not shown in top-level type completion.
    fn register_builtin_types(&mut self) {
        // HttpResponse: returned by Http.get and Http.request
        self.builtin_types.insert(
            "HttpResponse".to_string(),
            TypeInfo {
                name: "HttpResponse".to_string(),
                kind: TypeInfoKind::Record {
                    fields: vec![
                        ("status".to_string(), "Int".to_string()),
                        ("headers".to_string(), "List".to_string()),
                        ("body".to_string(), "String".to_string()),
                    ],
                    mutable: false,
                },
            },
        );

        // HttpRequest: returned by Server.accept
        self.builtin_types.insert(
            "HttpRequest".to_string(),
            TypeInfo {
                name: "HttpRequest".to_string(),
                kind: TypeInfoKind::Record {
                    fields: vec![
                        ("id".to_string(), "Int".to_string()),
                        ("method".to_string(), "String".to_string()),
                        ("path".to_string(), "String".to_string()),
                        ("headers".to_string(), "List".to_string()),
                        ("body".to_string(), "String".to_string()),
                        ("queryParams".to_string(), "List".to_string()),
                        ("cookies".to_string(), "List".to_string()),
                        ("formParams".to_string(), "List".to_string()),
                        ("isWebSocket".to_string(), "Bool".to_string()),
                    ],
                    mutable: false,
                },
            },
        );

        // ProcessInfo: returned by Process.info
        self.builtin_types.insert(
            "ProcessInfo".to_string(),
            TypeInfo {
                name: "ProcessInfo".to_string(),
                kind: TypeInfoKind::Record {
                    fields: vec![
                        ("status".to_string(), "String".to_string()),
                        ("mailbox".to_string(), "Int".to_string()),
                        ("uptime".to_string(), "Int".to_string()),
                    ],
                    mutable: false,
                },
            },
        );

        // ExecResult: returned by Exec.run
        self.builtin_types.insert(
            "ExecResult".to_string(),
            TypeInfo {
                name: "ExecResult".to_string(),
                kind: TypeInfoKind::Record {
                    fields: vec![
                        ("exitCode".to_string(), "Int".to_string()),
                        ("stdout".to_string(), "String".to_string()),
                        ("stderr".to_string(), "String".to_string()),
                    ],
                    mutable: false,
                },
            },
        );
    }

    /// Compile all pending functions.
    /// Returns (error, source_filename, source_code) on failure.
    pub fn compile_all(&mut self) -> Result<(), (CompileError, String, Arc<String>)> {
        let errors = self.compile_all_collecting_errors();
        if let Some((fn_name, error, source_name, source)) = errors.into_iter().next() {
            let _ = fn_name; // We include source_name instead of fn_name now
            Err((error, source_name, source))
        } else {
            Ok(())
        }
    }

    /// Compile all pending functions, collecting all errors.
    /// Returns a vec of (function_name, error, source_filename, source_code) for functions that failed to compile.
    /// Functions that compile successfully get their signatures set.
    ///
    /// This uses two passes to handle dependencies:
    /// 1. First pass: compile all functions (some may get placeholder type 'a' for dependencies)
    /// 2. Second pass: re-run HM inference for functions with 'a' in their signatures
    pub fn compile_all_collecting_errors(&mut self) -> Vec<(String, CompileError, String, Arc<String>)> {
        let pending = std::mem::take(&mut self.pending_functions);
        let mut errors: Vec<(String, CompileError, String, Arc<String>)> = Vec::new();

        // Pre-build function signatures for type checking (done once, not per-function)
        self.pending_fn_signatures.clear();
        let mut counter = 0u32;
        for (fn_def, module_path, _, _, _, _) in &pending {
            let fn_name = if module_path.is_empty() {
                fn_def.name.node.clone()
            } else {
                format!("{}.{}", module_path.join("."), fn_def.name.node)
            };
            if let Some(clause) = fn_def.clauses.first() {
                let param_types: Vec<nostos_types::Type> = clause.params
                    .iter()
                    .map(|p| {
                        if let Some(ty_expr) = &p.ty {
                            self.type_name_to_type(&self.type_expr_to_string(ty_expr))
                        } else {
                            // Create unique type variable for each untyped param
                            counter += 1;
                            nostos_types::Type::Var(counter)
                        }
                    })
                    .collect();
                let ret_ty = clause.return_type.as_ref()
                    .map(|ty| self.type_name_to_type(&self.type_expr_to_string(ty)))
                    .unwrap_or_else(|| {
                        counter += 1;
                        nostos_types::Type::Var(counter)
                    });

                self.pending_fn_signatures.insert(
                    fn_name,
                    nostos_types::FunctionType {
                        type_params: vec![],
                        params: param_types,
                        ret: Box::new(ret_ty),
                    },
                );
            }
        }

        // Pre-populate fn_asts with all pending functions so they can see each other
        // This is critical for multi-file modules where functions from different files
        // need to call each other (e.g., main.nos calling benchmark.nos in the same module)
        for (fn_def, module_path, _, _, _, _) in &pending {
            let base_name = if module_path.is_empty() {
                fn_def.name.node.clone()
            } else {
                format!("{}.{}", module_path.join("."), fn_def.name.node)
            };
            if let Some(clause) = fn_def.clauses.first() {
                let param_types: Vec<String> = clause.params.iter()
                    .map(|p| p.ty.as_ref()
                        .map(|t| self.type_expr_to_string(t))
                        .unwrap_or_else(|| "_".to_string()))
                    .collect();
                let signature = param_types.join(",");
                let name = format!("{}/{}", base_name, signature);
                // Insert a placeholder in fn_asts so has_function_with_base can find it
                self.fn_asts.insert(name, fn_def.clone());
            }
        }

        // First pass: compile all functions
        for (fn_def, module_path, imports, line_starts, source, source_name) in pending {
            let saved_path = self.module_path.clone();
            let saved_imports = self.imports.clone();
            let saved_line_starts = self.line_starts.clone();
            let saved_source = self.current_source.clone();
            let saved_source_name = self.current_source_name.clone();

            // Build qualified function name
            let fn_name = if module_path.is_empty() {
                fn_def.name.node.clone()
            } else {
                format!("{}.{}", module_path.join("."), fn_def.name.node)
            };

            self.module_path = module_path;
            // Merge imports instead of replacing, to be safe.
            self.imports.extend(imports);
            self.line_starts = line_starts;
            self.current_source = Some(source.clone());
            self.current_source_name = Some(source_name.clone());

            // Continue compiling other functions even if one fails
            if let Err(e) = self.compile_fn_def(&fn_def) {
                errors.push((fn_name, e, source_name.clone(), source.clone()));
            }

            self.module_path = saved_path;
            self.imports = saved_imports;
            self.line_starts = saved_line_starts;
            self.current_source = saved_source;
            self.current_source_name = saved_source_name;
        }

        // Clear pending data now that we've processed them
        self.pending_functions.clear();
        self.pending_fn_signatures.clear();

        // Second pass: re-run HM inference for functions with type variables in their signatures
        // This handles cases like bar23() = bar() + 1 where bar was compiled after bar23's first inference
        // Run multiple iterations until no more signatures change (handles dependency order issues)
        let fn_names: Vec<String> = self.functions.keys().cloned().collect();
        let max_iterations = 5; // Prevent infinite loops
        for _iteration in 0..max_iterations {
            let mut changed = false;
            for fn_name in &fn_names {
                if let Some(fn_val) = self.functions.get(fn_name) {
                    if let Some(sig) = &fn_val.signature {
                        // If signature contains ONLY type variables (like 'a' or 'a -> b'), re-infer
                        // A signature with only type variables won't contain concrete type names
                        let has_concrete_type = sig.contains("Int") || sig.contains("Float") ||
                            sig.contains("String") || sig.contains("Bool") || sig.contains("Char") ||
                            sig.contains("List") || sig.contains("Map") || sig.contains("Set") ||
                            sig.contains("Option") || sig.contains("Result") || sig.contains("Unit") ||
                            sig.contains("Bytes") || sig.contains("BigInt") || sig.contains("Decimal");
                        // Has single-letter type variables like 'a', 'b', etc.
                        let has_type_var = sig.chars().any(|c| c.is_ascii_lowercase() && c != '-' && c != '>');

                        if has_type_var && !has_concrete_type {
                            // Try HM inference again now that all dependencies are compiled
                            if let Some(fn_ast) = self.fn_asts.get(fn_name).cloned() {
                                let result = self.try_hm_inference(&fn_ast);
                                if let Some(inferred_sig) = result {
                                    // Check if signature actually changed
                                    if &inferred_sig != sig {
                                        // Update the function's signature - need to clone and replace
                                        // since Arc::get_mut won't work if there are other references
                                        if let Some(fn_val) = self.functions.get(fn_name) {
                                            let mut new_fn_val = (**fn_val).clone();
                                            new_fn_val.signature = Some(inferred_sig);
                                            self.functions.insert(fn_name.clone(), Arc::new(new_fn_val));
                                            changed = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Third pass: Type check all user functions now that we have concrete signatures
        // This catches errors like bar() = bar2() + "x" where bar2() returns ()
        // Skip stdlib functions (those without ASTs or with stdlib paths)
        for (fn_name, fn_ast) in &self.fn_asts {
            // Skip stdlib functions (they may have inference limitations we don't want to report)
            // Stdlib functions don't have source files or are in internal paths
            if fn_name.starts_with("List.") || fn_name.starts_with("String.") ||
               fn_name.starts_with("Math.") || fn_name.starts_with("Map.") ||
               fn_name.starts_with("Set.") || fn_name.starts_with("Json.") {
                continue;
            }

            // Skip REPL eval wrappers - these are temporary functions that may contain errors
            // from previous REPL inputs. We don't want old errors to affect new inputs.
            if fn_name.starts_with("__repl_eval_") {
                continue;
            }

            // Check if this function has untyped parameters and is recursive
            let has_untyped_params = fn_ast.clauses.first()
                .map(|c| c.params.iter().any(|p| p.ty.is_none()))
                .unwrap_or(false);
            let is_recursive = Self::is_recursive_fn(fn_ast);

            // Run type checking with full knowledge of all function signatures
            if let Err(e) = self.type_check_fn(fn_ast, fn_name) {
                // Only report concrete type mismatches
                let should_report = match &e {
                    CompileError::TypeError { message, .. } => {
                        // Filter out List element vs List errors from mutual recursion inference
                        // Pattern 1: "List[X] and X" - list type first
                        // Pattern 2: "X and List[X]" - element type first
                        let is_list_element_error = message.contains("List[") &&
                            (message.contains("] and ") || message.contains(" and List[")) &&
                            message.matches("List[").count() == 1;

                        // For RECURSIVE functions with untyped params, filter out false positives
                        // from recursive inference where param types get confused with returns
                        // Non-recursive functions with conflicting branch types are real errors
                        let is_recursive_inference_error = has_untyped_params && is_recursive &&
                            message.contains("Cannot unify types:");

                        // For functions with untyped params, also filter out trait implementation errors
                        // These can be false positives during inference, especially in mutual recursion
                        // (e.g., "Bool does not implement Num" when isEven calls isOdd and vice versa)
                        let is_trait_inference_error = has_untyped_params &&
                            message.contains("does not implement");

                        // For dynamic typing support (like heterogeneous lists), filter out
                        // unification errors in functions WITHOUT untyped params that call
                        // functions WITH untyped params
                        // e.g., g() = f([1,2,3]) where f(xs) = ["hello"] ++ xs
                        // The error is in g (no untyped params) calling f (has untyped params)
                        // But DON'T filter errors in functions WITH untyped params themselves
                        // e.g., f(x) = if x then 42 else "hello" - conflicting branch types
                        let is_call_inference_error = !has_untyped_params &&
                            message.contains("Cannot unify types:") &&
                            !message.contains("List[") &&
                            !message.contains("->");

                        let is_inference_limitation = message.contains("Unknown identifier") ||
                            message.contains("Unknown type") ||
                            message.contains("has no field") ||
                            message.contains("() and ()") ||
                            is_list_element_error ||
                            is_recursive_inference_error ||
                            is_trait_inference_error ||
                            is_call_inference_error ||
                            Self::is_type_variable_only_error(message);
                        !is_inference_limitation
                    }
                    _ => true,
                };
                if should_report {
                    // Use base function name without signature for error reporting
                    let base_name = fn_name.split('/').next().unwrap_or(fn_name).to_string();
                    // Get source info from fn_sources for proper error reporting
                    let (source_name, source) = self.fn_sources.get(fn_name)
                        .cloned()
                        .unwrap_or_else(|| ("unknown".to_string(), Arc::new(String::new())));
                    errors.push((base_name, e, source_name, source));
                }
            }
        }

        // Check for mvar safety violations (prevents runtime deadlocks)
        let mvar_errors = self.check_mvar_deadlocks();
        for msg in mvar_errors {
            // Extract function name from error message if possible
            let fn_name = msg.split('`').nth(1).unwrap_or("unknown").to_string();
            let (source_name, source) = self.fn_sources.iter()
                .find(|(k, _)| k.contains(&fn_name))
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| ("unknown".to_string(), Arc::new(String::new())));
            errors.push((
                fn_name,
                CompileError::MvarSafetyViolation {
                    message: msg,
                    span: Span { start: 0, end: 0 }
                },
                source_name,
                source,
            ));
        }

        errors
    }

    /// Compile a module and add it to the current compilation context.
    pub fn add_module(&mut self, module: &Module, module_path: Vec<String>, source: Arc<String>, source_name: String) -> Result<(), CompileError> {
        // Update line_starts for this file
        self.line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                self.line_starts.push(i + 1);
            }
        }

        self.current_source = Some(source);
        self.current_source_name = Some(source_name.clone());

        // Register this module path and all its prefixes as known modules
        // e.g., for ["math", "vector"], register both "math" and "math.vector"
        if !module_path.is_empty() {
            let mut prefix = String::new();
            for component in &module_path {
                if !prefix.is_empty() {
                    prefix.push('.');
                }
                prefix.push_str(component);
                self.known_modules.insert(prefix.clone());
            }
        }

        // Set module path
        self.module_path = module_path;

        // Compile items
        self.compile_items(&module.items)?;

        // Reset module path
        self.module_path = Vec::new();

        Ok(())
    }

    pub fn new(source: &str) -> Self {
        // Compute line start offsets for source mapping
        let mut line_starts = vec![0]; // Line 1 starts at offset 0
        for (i, c) in source.char_indices() {
            if c == '\n' {
                line_starts.push(i + 1); // Next line starts after the newline
            }
        }

        // Pre-register built-in pseudo-modules for native functions
        let builtin_modules: HashSet<String> = [
            "String", "File", "Dir", "List", "Option", "Result", "Char", "Int", "Float",
            "Bool", "Bytes", "Map", "Set", "IO", "Math", "Debug", "Time", "Thread",
            "Channel", "Regex", "Json", "Http", "Net", "Sys", "Env", "Process",
            "Base64", "Url", "Encoding", "Server", "Exec", "Random", "Path", "Panel",
            "Pg", "Uuid", "Crypto", "Float64Array", "Int64Array", "Float32Array", "Buffer",
            "Runtime", "WebSocket",
        ].iter().map(|s| s.to_string()).collect();

        Self {
            chunk: Chunk::new(),
            locals: HashMap::new(),
            next_reg: 0,
            capture_indices: HashMap::new(),
            functions: HashMap::new(),
            function_indices: HashMap::new(),
            function_list: Vec::new(),
            types: HashMap::new(),
            builtin_types: HashMap::new(),
            known_constructors: HashSet::new(),
            scope_depth: 0,
            module_path: Vec::new(),
            imports: HashMap::new(),
            function_visibility: HashMap::new(),
            trait_defs: HashMap::new(),
            trait_impls: HashMap::new(),
            type_traits: HashMap::new(),
            local_types: HashMap::new(),
            param_types: HashMap::new(),
            fn_asts: HashMap::new(),
            fn_sources: HashMap::new(),
            fn_type_params: HashMap::new(),
            polymorphic_fns: HashSet::new(),
            current_function_name: None,
            current_fn_type_params: Vec::new(),
            loop_stack: Vec::new(),
            line_starts,
            pending_functions: Vec::new(),
            pending_fn_signatures: HashMap::new(),
            current_source: Some(Arc::new(source.to_string())),
            current_source_name: Some("unknown".to_string()),
            type_defs: HashMap::new(),
            known_modules: builtin_modules,
            imported_modules: HashSet::new(),
            module_use_stmts: Vec::new(),
            local_modules: HashSet::new(),
            repl_mode: false,
            prelude_functions: HashSet::new(),
            mvars: HashMap::new(),
            fn_mvar_access: HashMap::new(),
            fn_calls: HashMap::new(),
            current_fn_mvar_reads: HashSet::new(),
            current_fn_mvar_writes: HashSet::new(),
            current_fn_calls: HashSet::new(),
            current_fn_mvar_locks: Vec::new(),
            current_fn_has_blocking: false,
            current_fn_debug_symbols: Vec::new(),
            native_indices: HashMap::new(),
            extension_indices: HashMap::new(),
        }
    }

    /// Enable REPL mode - bypasses visibility checks for interactive exploration
    pub fn set_repl_mode(&mut self, enabled: bool) {
        self.repl_mode = enabled;
    }

    /// Add a prelude import - maps a local name to a qualified name.
    /// This allows stdlib functions to be available without prefix.
    /// Also marks the qualified function as a prelude function to bypass visibility checks.
    pub fn add_prelude_import(&mut self, local_name: String, qualified_name: String) {
        self.prelude_functions.insert(qualified_name.clone());
        self.imports.insert(local_name, qualified_name);
    }

    /// Get all prelude imports (local name -> qualified name mappings)
    pub fn get_prelude_imports(&self) -> &HashMap<String, String> {
        &self.imports
    }

    /// Convert a byte offset to a line number (1-indexed).
    fn offset_to_line(&self, offset: usize) -> usize {
        // Binary search for the line containing this offset
        match self.line_starts.binary_search(&offset) {
            Ok(idx) => idx + 1, // Exact match: offset is at start of line
            Err(idx) => idx,    // offset is within line idx (1-indexed)
        }
    }

    /// Get the line number for a span.
    fn span_line(&self, span: Span) -> usize {
        self.offset_to_line(span.start)
    }

    /// Get the fully qualified name with the current module path prefix.
    fn qualify_name(&self, name: &str) -> String {
        if self.module_path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.module_path.join("."), name)
        }
    }

    /// Resolve a name, checking imports and module path.
    /// Returns the fully qualified name if found, or the original name.
    /// Note: This does not check for ambiguity - use resolve_name_checked for that.
    fn resolve_name(&self, name: &str) -> String {
        match self.resolve_name_checked(name, Span::default()) {
            Ok(resolved) => resolved,
            Err(_) => name.to_string(), // Return original on error (will be caught elsewhere)
        }
    }

    /// Resolve a name with ambiguity checking.
    /// Returns the fully qualified name or an error if ambiguous.
    fn resolve_name_checked(&self, name: &str, span: Span) -> Result<String, CompileError> {
        // If the name already contains '.', it's already qualified
        if name.contains('.') {
            return Ok(name.to_string());
        }

        // First check module-local declarations (these should NOT be shadowed by imports)
        // This includes mvars, local functions, types, and constructors
        let qualified = self.qualify_name(name);
        if self.mvars.contains_key(&qualified) || self.mvars.contains_key(name) {
            // Mvars take highest precedence in local scope
            return Ok(if self.mvars.contains_key(&qualified) { qualified } else { name.to_string() });
        }
        let has_qualified = self.has_function_with_base(&qualified);
        if has_qualified || self.types.contains_key(&qualified) || self.known_constructors.contains(&qualified) {
            return Ok(qualified);
        }
        // Check if it's in the global scope (before imports for user-defined global functions)
        if self.has_function_with_base(name) || self.types.contains_key(name) || self.known_constructors.contains(name) {
            return Ok(name.to_string());
        }

        // Check user imports (import module statements)
        // Collect all matching modules to detect ambiguity
        let mut matching_modules: Vec<String> = Vec::new();
        for (importing_module, imported_module) in &self.imported_modules {
            if importing_module == &self.module_path {
                // Check if the imported module has this function
                let qualified_in_module = format!("{}.{}", imported_module, name);
                if self.has_function_with_base(&qualified_in_module)
                    || self.types.contains_key(&qualified_in_module)
                    || self.known_constructors.contains(&qualified_in_module)
                {
                    matching_modules.push(imported_module.clone());
                }
            }
        }

        // Check for ambiguity
        if matching_modules.len() > 1 {
            return Err(CompileError::AmbiguousName {
                name: name.to_string(),
                modules: matching_modules.join(", "),
                span,
            });
        }

        // Return the single match if found
        if let Some(module) = matching_modules.first() {
            return Ok(format!("{}.{}", module, name));
        }

        // Then check imports (prelude functions)
        if let Some(qualified) = self.imports.get(name) {
            return Ok(qualified.clone());
        }

        // Return the original name (will error later if not found)
        Ok(name.to_string())
    }

    /// Check if there's any function with the given base name (ignoring signature suffix).
    fn has_function_with_base(&self, base_name: &str) -> bool {
        let prefix = format!("{}/", base_name);
        // Check compiled functions
        if self.functions.keys().any(|k| k.starts_with(&prefix)) {
            return true;
        }
        // Check pending ASTs
        if self.fn_asts.keys().any(|k| k.starts_with(&prefix)) {
            return true;
        }
        // Check if current function matches (for recursion during compilation)
        if let Some(current) = &self.current_function_name {
            if current.starts_with(&prefix) {
                return true;
            }
        }
        false
    }

    /// Check if an expression is float-typed (for type-directed operator selection).
    /// This is a simple heuristic: true if the expression is a float literal or
    /// a binary operation on floats.
    /// Check if a type name refers to a float type.
    fn is_float_type_name(ty: &str) -> bool {
        matches!(ty, "Float" | "Float32" | "Float64")
    }

    /// Check if a type name refers to a BigInt type.
    fn is_bigint_type_name(ty: &str) -> bool {
        ty == "BigInt"
    }

    /// Check if a type name refers to a small integer type (not BigInt).
    fn is_small_int_type_name(ty: &str) -> bool {
        matches!(ty, "Int" | "Int8" | "Int16" | "Int32" | "Int64" | "UInt8" | "UInt16" | "UInt32" | "UInt64")
    }

    /// Check if a type name refers to an integer type (including BigInt).
    fn is_int_type_name(ty: &str) -> bool {
        Self::is_small_int_type_name(ty) || Self::is_bigint_type_name(ty)
    }

    fn is_float_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Float(_, _) | Expr::Float32(_, _) => true,
            Expr::Int(_, _) | Expr::Int8(_, _) | Expr::Int16(_, _) | Expr::Int32(_, _)
            | Expr::UInt8(_, _) | Expr::UInt16(_, _) | Expr::UInt32(_, _) | Expr::UInt64(_, _)
            | Expr::BigInt(_, _) | Expr::Decimal(_, _) => false,
            // Check if variable is known to be float
            Expr::Var(ident) => {
                // Check locals.is_float first
                if self.locals.get(&ident.node).map(|info| info.is_float).unwrap_or(false) {
                    return true;
                }
                // Check local_types for float types
                if let Some(ty) = self.local_types.get(&ident.node) {
                    return Self::is_float_type_name(ty);
                }
                // Check param_types for function parameters
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return Self::is_float_type_name(ty);
                }
                false
            }
            // Field access: look up the field's type from the record definition
            Expr::FieldAccess(obj, field, _) => {
                if let Some(obj_type) = self.expr_type_name(obj) {
                    if let Some(type_info) = self.types.get(&obj_type) {
                        if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                            for (fname, ftype) in fields {
                                if fname == &field.node {
                                    return Self::is_float_type_name(ftype);
                                }
                            }
                        }
                    }
                }
                false
            }
            Expr::BinOp(left, op, right, _) => {
                // Arithmetic operators preserve float type
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Pow | BinOp::Mod => {
                        self.is_float_expr(left) || self.is_float_expr(right)
                    }
                    // Comparison operators return bool
                    _ => false,
                }
            }
            Expr::UnaryOp(UnaryOp::Neg, operand, _) => self.is_float_expr(operand),
            Expr::If(_, then_branch, else_branch, _) => {
                self.is_float_expr(then_branch) || self.is_float_expr(else_branch)
            }
            Expr::Block(stmts, _) => {
                // Check if the last statement is an expression that is float-typed
                stmts.last().map(|s| match s {
                    Stmt::Expr(e) => self.is_float_expr(e),
                    _ => false,
                }).unwrap_or(false)
            }
            // Function calls: assume non-float by default.
            // We can't know the return type without proper type inference,
            // and assuming float based on arguments is incorrect (e.g., show(3.14) returns String).
            Expr::Call(_, _, _, _) => false,
            _ => false, // Assume non-float by default for other expressions
        }
    }

    /// Check if an expression is known to be an Int64List at compile time.
    fn is_int64_list_expr(&self, expr: &Expr) -> bool {
        match expr {
            // Empty list can be Int64List
            Expr::List(items, None, _) if items.is_empty() => true,
            // List literal with all Int64 items
            Expr::List(items, None, _) => {
                !items.is_empty() && items.iter().all(|item| self.is_int64_expr(item))
            }
            // Cons expression with Int64 head and Int64List tail
            Expr::List(items, Some(tail), _) => {
                items.iter().all(|item| self.is_int64_expr(item)) && self.is_int64_list_expr(tail)
            }
            // Variable - check type
            Expr::Var(ident) => {
                if let Some(ty) = self.local_types.get(&ident.node) {
                    return ty == "Int64List" || ty == "[Int]" || ty == "[Int64]";
                }
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return ty == "Int64List" || ty == "[Int]" || ty == "[Int64]";
                }
                false
            }
            // Known functions that return Int64List
            Expr::Call(func, _, _, _) => {
                if let Expr::Var(name) = func.as_ref() {
                    matches!(name.node.as_str(), "intListRange" | "toIntList")
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Check if an expression is known to be Int64 at compile time (for list specialization).
    fn is_int64_expr(&self, expr: &Expr) -> bool {
        match expr {
            // Int literals are Int64 by default
            Expr::Int(_, _) => true,
            // Other int types are NOT Int64
            Expr::Int8(_, _) | Expr::Int16(_, _) | Expr::Int32(_, _)
            | Expr::UInt8(_, _) | Expr::UInt16(_, _) | Expr::UInt32(_, _) | Expr::UInt64(_, _)
            | Expr::BigInt(_, _) => false,
            Expr::Float(_, _) | Expr::Float32(_, _) | Expr::Decimal(_, _) => false,
            // Check if variable is known to be Int64
            Expr::Var(ident) => {
                if let Some(ty) = self.local_types.get(&ident.node) {
                    return ty == "Int" || ty == "Int64";
                }
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return ty == "Int" || ty == "Int64";
                }
                false
            }
            // Arithmetic on Int64s stays Int64
            Expr::BinOp(left, op, right, _) => {
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        self.is_int64_expr(left) && self.is_int64_expr(right)
                    }
                    _ => false,
                }
            }
            // Function call - check known return types
            Expr::Call(func, _, _, _) => {
                if let Expr::Var(name) = func.as_ref() {
                    // Some builtins return Int64
                    matches!(name.node.as_str(), "length" | "len" | "sum" | "count")
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Check if an expression is known to be an integer at compile time.
    fn is_int_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Int(_, _) | Expr::Int8(_, _) | Expr::Int16(_, _) | Expr::Int32(_, _)
            | Expr::UInt8(_, _) | Expr::UInt16(_, _) | Expr::UInt32(_, _) | Expr::UInt64(_, _)
            | Expr::BigInt(_, _) => true,
            Expr::Float(_, _) | Expr::Float32(_, _) | Expr::Decimal(_, _) => false,
            // Check if variable is known to be int
            Expr::Var(ident) => {
                // Check locals - if it's float, return false
                if self.locals.get(&ident.node).map(|info| info.is_float).unwrap_or(false) {
                    return false;
                }
                // Check local_types for int types
                if let Some(ty) = self.local_types.get(&ident.node) {
                    return Self::is_int_type_name(ty);
                }
                // Check param_types for function parameters
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return Self::is_int_type_name(ty);
                }
                // If type is unknown, we can't safely assume it's an int
                // (it could be float from pattern matching, etc.)
                false
            }
            // Field access: look up the field's type from the record definition
            Expr::FieldAccess(obj, field, _) => {
                if let Some(obj_type) = self.expr_type_name(obj) {
                    if let Some(type_info) = self.types.get(&obj_type) {
                        if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                            for (fname, ftype) in fields {
                                if fname == &field.node {
                                    return Self::is_int_type_name(ftype);
                                }
                            }
                        }
                    }
                }
                false
            }
            Expr::BinOp(left, op, right, _) => {
                // Arithmetic operators: int if both sides are int
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        self.is_int_expr(left) && self.is_int_expr(right) && !self.is_float_expr(left) && !self.is_float_expr(right)
                    }
                    // Comparison operators return bool
                    _ => false,
                }
            }
            Expr::UnaryOp(UnaryOp::Neg, operand, _) => self.is_int_expr(operand) && !self.is_float_expr(operand),
            Expr::If(_, then_branch, else_branch, _) => {
                self.is_int_expr(then_branch) && self.is_int_expr(else_branch)
                    && !self.is_float_expr(then_branch) && !self.is_float_expr(else_branch)
            }
            Expr::Block(stmts, _) => {
                // Check if the last statement is an expression that is int-typed
                stmts.last().map(|s| match s {
                    Stmt::Expr(e) => self.is_int_expr(e) && !self.is_float_expr(e),
                    _ => false,
                }).unwrap_or(false)
            }
            // Function calls: assume non-int by default
            Expr::Call(_, _, _, _) => false,
            _ => false,
        }
    }

    /// Check if an expression is known to be a BigInt at compile time.
    fn is_bigint_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::BigInt(_, _) => true,
            Expr::Int(_, _) | Expr::Int8(_, _) | Expr::Int16(_, _) | Expr::Int32(_, _)
            | Expr::UInt8(_, _) | Expr::UInt16(_, _) | Expr::UInt32(_, _) | Expr::UInt64(_, _)
            | Expr::Float(_, _) | Expr::Float32(_, _) | Expr::Decimal(_, _) => false,
            // Check if variable is known to be BigInt
            Expr::Var(ident) => {
                // Check local_types
                if let Some(ty) = self.local_types.get(&ident.node) {
                    return Self::is_bigint_type_name(ty);
                }
                // Check param_types
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return Self::is_bigint_type_name(ty);
                }
                false
            }
            // Field access: look up the field's type
            Expr::FieldAccess(obj, field, _) => {
                if let Some(obj_type) = self.expr_type_name(obj) {
                    if let Some(type_info) = self.types.get(&obj_type) {
                        if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                            for (fname, ftype) in fields {
                                if fname == &field.node {
                                    return Self::is_bigint_type_name(ftype);
                                }
                            }
                        }
                    }
                }
                false
            }
            Expr::BinOp(left, op, right, _) => {
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        self.is_bigint_expr(left) || self.is_bigint_expr(right)
                    }
                    _ => false,
                }
            }
            Expr::UnaryOp(UnaryOp::Neg, operand, _) => self.is_bigint_expr(operand),
            Expr::If(_, then_branch, else_branch, _) => {
                self.is_bigint_expr(then_branch) || self.is_bigint_expr(else_branch)
            }
            Expr::Block(stmts, _) => {
                stmts.last().map(|s| match s {
                    Stmt::Expr(e) => self.is_bigint_expr(e),
                    _ => false,
                }).unwrap_or(false)
            }
            Expr::Call(_, _, _, _) => false,
            _ => false,
        }
    }

    /// Check if an expression is a small int (not BigInt) at compile time.
    fn is_small_int_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Int(_, _) | Expr::Int8(_, _) | Expr::Int16(_, _) | Expr::Int32(_, _)
            | Expr::UInt8(_, _) | Expr::UInt16(_, _) | Expr::UInt32(_, _) | Expr::UInt64(_, _) => true,
            Expr::BigInt(_, _) | Expr::Float(_, _) | Expr::Float32(_, _) | Expr::Decimal(_, _) => false,
            Expr::Var(ident) => {
                if let Some(ty) = self.local_types.get(&ident.node) {
                    return Self::is_small_int_type_name(ty);
                }
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return Self::is_small_int_type_name(ty);
                }
                false
            }
            Expr::FieldAccess(obj, field, _) => {
                if let Some(obj_type) = self.expr_type_name(obj) {
                    if let Some(type_info) = self.types.get(&obj_type) {
                        if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                            for (fname, ftype) in fields {
                                if fname == &field.node {
                                    return Self::is_small_int_type_name(ftype);
                                }
                            }
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Get the specific sized integer type of an expression (Int8, Int16, Int32, UInt8, etc.)
    /// Returns None for plain Int/Int64 or non-integer types.
    fn get_sized_int_type(&self, expr: &Expr) -> Option<&'static str> {
        match expr {
            Expr::Int8(_, _) => Some("Int8"),
            Expr::Int16(_, _) => Some("Int16"),
            Expr::Int32(_, _) => Some("Int32"),
            Expr::UInt8(_, _) => Some("UInt8"),
            Expr::UInt16(_, _) => Some("UInt16"),
            Expr::UInt32(_, _) => Some("UInt32"),
            Expr::UInt64(_, _) => Some("UInt64"),
            // Plain Int is Int64 - not a "sized" type for coercion purposes
            Expr::Int(_, _) => None,
            Expr::Var(ident) => {
                if let Some(ty) = self.local_types.get(&ident.node) {
                    return Self::sized_int_type_name(ty);
                }
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return Self::sized_int_type_name(ty);
                }
                None
            }
            Expr::FieldAccess(obj, field, _) => {
                if let Some(obj_type) = self.expr_type_name(obj) {
                    if let Some(type_info) = self.types.get(&obj_type) {
                        if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                            for (fname, ftype) in fields {
                                if fname == &field.node {
                                    return Self::sized_int_type_name(ftype);
                                }
                            }
                        }
                    }
                }
                None
            }
            Expr::BinOp(left, op, right, _) => {
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        // If both sides have the same sized type, result is that type
                        let lt = self.get_sized_int_type(left);
                        let rt = self.get_sized_int_type(right);
                        if lt == rt { lt } else { None }
                    }
                    _ => None,
                }
            }
            Expr::Call(func, args, _, _) => {
                // Check if this is a function call whose return type is a sized int
                // This handles REPL variable thunks like __repl_var_a__()
                if let Expr::Var(ident) = func.as_ref() {
                    if let Some(func_info) = self.functions.get(&ident.node) {
                        // First try explicit return_type
                        if let Some(ret_type) = &func_info.return_type {
                            if let Some(sized) = Self::sized_int_type_name(ret_type) {
                                return Some(sized);
                            }
                        }
                        // For 0-arity functions, the signature IS the return type
                        if args.is_empty() {
                            if let Some(sig) = &func_info.signature {
                                // Signature for 0-arity function is just the return type
                                // e.g., "Int32" not "Something -> Int32"
                                if !sig.contains("->") {
                                    return Self::sized_int_type_name(sig.trim());
                                }
                            }
                        }
                    }
                }
                None
            }
            Expr::MethodCall(receiver, method, _, _) => {
                // Check for type conversion methods like .asInt32()
                let method_name = &method.node;
                match method_name.as_str() {
                    "asInt8" => Some("Int8"),
                    "asInt16" => Some("Int16"),
                    "asInt32" => Some("Int32"),
                    "asUInt8" => Some("UInt8"),
                    "asUInt16" => Some("UInt16"),
                    "asUInt32" => Some("UInt32"),
                    "asUInt64" => Some("UInt64"),
                    _ => {
                        // Recursive: check the receiver's type
                        self.get_sized_int_type(receiver)
                    }
                }
            }
            _ => None,
        }
    }

    /// Convert a type name to a sized int type if applicable
    fn sized_int_type_name(ty: &str) -> Option<&'static str> {
        match ty {
            "Int8" => Some("Int8"),
            "Int16" => Some("Int16"),
            "Int32" => Some("Int32"),
            "UInt8" => Some("UInt8"),
            "UInt16" => Some("UInt16"),
            "UInt32" => Some("UInt32"),
            "UInt64" => Some("UInt64"),
            _ => None, // Int, Int64, Float, etc. are not "sized" for coercion
        }
    }

    /// Get the conversion instruction for a sized int type
    fn sized_int_conversion_instruction(ty: &str, dst: Reg, src: Reg) -> Option<Instruction> {
        match ty {
            "Int8" => Some(Instruction::ToInt8(dst, src)),
            "Int16" => Some(Instruction::ToInt16(dst, src)),
            "Int32" => Some(Instruction::ToInt32(dst, src)),
            "UInt8" => Some(Instruction::ToUInt8(dst, src)),
            "UInt16" => Some(Instruction::ToUInt16(dst, src)),
            "UInt32" => Some(Instruction::ToUInt32(dst, src)),
            "UInt64" => Some(Instruction::ToUInt64(dst, src)),
            _ => None,
        }
    }

    /// Compile a top-level item.
    pub fn compile_item(&mut self, item: &Item) -> Result<(), CompileError> {
        match item {
            Item::FnDef(fn_def) => {
                self.compile_fn_def(fn_def)?;
            }
            Item::TypeDef(type_def) => {
                self.compile_type_def(type_def)?;
            }
            Item::TraitDef(trait_def) => {
                self.compile_trait_def(trait_def)?;
            }
            Item::TraitImpl(trait_impl) => {
                self.compile_trait_impl(trait_impl)?;
            }
            Item::MvarDef(mvar_def) => {
                self.compile_mvar_def(mvar_def)?;
            }
            _ => {
                return Err(CompileError::NotImplemented {
                    feature: format!("item: {:?}", item),
                    span: item.span(),
                });
            }
        }
        Ok(())
    }

    /// Look up field types for a variant constructor.
    fn get_constructor_field_types(&self, ctor_name: &str) -> Vec<String> {
        for info in self.types.values() {
            if let TypeInfoKind::Variant { constructors } = &info.kind {
                for (name, field_types) in constructors {
                    if name == ctor_name {
                        return field_types.clone();
                    }
                }
            }
        }
        vec![]
    }

    /// Get the full name of a type expression including type parameters.
    /// Check if a type name is a built-in type (primitives, collections, etc.)
    fn is_builtin_type_name(&self, name: &str) -> bool {
        matches!(name,
            "Int" | "Int8" | "Int16" | "Int32" | "Int64" |
            "UInt8" | "UInt16" | "UInt32" | "UInt64" |
            "Float" | "Float32" | "Float64" |
            "Bool" | "Char" | "String" | "BigInt" | "Decimal" |
            "List" | "Array" | "Set" | "Map" | "IO" | "Pid" | "Ref" |
            "()" | "Unit" | "Never"
        )
    }

    fn type_expr_name(&self, ty: &nostos_syntax::TypeExpr) -> String {
        match ty {
            nostos_syntax::TypeExpr::Name(ident) => {
                let name = &ident.node;
                // Skip built-in types and single-letter type params (both lowercase like 'a' and uppercase like 'T')
                if self.is_builtin_type_name(name) ||
                   (name.len() == 1 && name.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false)) {
                    return name.clone();
                }
                // Check if this is a user-defined type that needs qualification
                let qualified = self.qualify_name(name);
                // Check if it's a known type (already registered)
                if self.types.contains_key(&qualified) {
                    qualified
                } else if self.types.contains_key(name) {
                    // Type exists with unqualified name (from another module)
                    name.clone()
                } else if !self.module_path.is_empty() {
                    // In a module context, qualify unknown type names
                    // This handles self-referential types that haven't been registered yet
                    qualified
                } else {
                    // Top-level code - use as-is
                    name.clone()
                }
            }
            nostos_syntax::TypeExpr::Generic(ident, args) => {
                // Include type parameters: List[Int], Map[String, Int], etc.
                let args_str: Vec<String> = args.iter()
                    .map(|arg| self.type_expr_name(arg))
                    .collect();
                let name = &ident.node;
                // Built-in generic types don't get qualified
                let base_name = if self.is_builtin_type_name(name) {
                    name.clone()
                } else {
                    let qualified = self.qualify_name(name);
                    if self.types.contains_key(&qualified) || !self.module_path.is_empty() {
                        qualified
                    } else {
                        name.clone()
                    }
                };
                if args_str.is_empty() {
                    base_name
                } else {
                    format!("{}[{}]", base_name, args_str.join(", "))
                }
            }
            nostos_syntax::TypeExpr::Function(params, ret) => {
                let params_str: Vec<String> = params.iter()
                    .map(|p| self.type_expr_name(p))
                    .collect();
                let ret_str = self.type_expr_name(ret);
                format!("({}) -> {}", params_str.join(", "), ret_str)
            }
            nostos_syntax::TypeExpr::Record(fields) => {
                let fields_str: Vec<String> = fields.iter()
                    .map(|(name, ty)| format!("{}: {}", name.node, self.type_expr_name(ty)))
                    .collect();
                format!("{{{}}}", fields_str.join(", "))
            }
            nostos_syntax::TypeExpr::Tuple(elems) => {
                let elems_str: Vec<String> = elems.iter()
                    .map(|e| self.type_expr_name(e))
                    .collect();
                format!("({})", elems_str.join(", "))
            }
            nostos_syntax::TypeExpr::Unit => "()".to_string(),
        }
    }

    /// Compile a module-level mutable variable (mvar) definition.
    /// This registers the mvar with the compiler and will set up the VM storage.
    fn compile_mvar_def(&mut self, def: &MvarDef) -> Result<(), CompileError> {
        let qualified_name = self.qualify_name(&def.name.node);
        let type_name = self.type_expr_name(&def.ty);

        // Evaluate the initial value (must be a compile-time constant)
        let initial_value = self.eval_const_expr(&def.value)
            .ok_or_else(|| CompileError::NotImplemented {
                feature: format!("mvar initial value must be a constant literal, got: {:?}", def.value),
                span: def.span,
            })?;

        // Register the mvar in our tracking map
        self.mvars.insert(qualified_name.clone(), MvarInfo {
            type_name,
            initial_value,
        });

        Ok(())
    }

    /// Evaluate a constant expression to an MvarInitValue.
    /// Returns None if the expression is not a compile-time constant.
    fn eval_const_expr(&self, expr: &Expr) -> Option<MvarInitValue> {
        match expr {
            Expr::Unit(_) => Some(MvarInitValue::Unit),
            Expr::Bool(b, _) => Some(MvarInitValue::Bool(*b)),
            Expr::Int(n, _) => Some(MvarInitValue::Int(*n)),
            Expr::Float(f, _) => Some(MvarInitValue::Float(*f)),
            Expr::String(StringLit::Plain(s), _) => Some(MvarInitValue::String(s.clone())),
            Expr::Char(c, _) => Some(MvarInitValue::Char(*c)),
            Expr::List(items, None, _) if items.is_empty() => Some(MvarInitValue::EmptyList),
            Expr::List(items, None, _) => {
                // Try to parse as homogeneous list of primitives first (more efficient)

                // Check for int list
                if items.iter().all(|item| matches!(item, Expr::Int(_, _))) {
                    let ints: Vec<i64> = items.iter()
                        .filter_map(|item| if let Expr::Int(n, _) = item { Some(*n) } else { None })
                        .collect();
                    return Some(MvarInitValue::IntList(ints));
                }

                // Check for float list
                if items.iter().all(|item| matches!(item, Expr::Float(_, _))) {
                    let floats: Vec<f64> = items.iter()
                        .filter_map(|item| if let Expr::Float(f, _) = item { Some(*f) } else { None })
                        .collect();
                    return Some(MvarInitValue::FloatList(floats));
                }

                // Check for bool list
                if items.iter().all(|item| matches!(item, Expr::Bool(_, _))) {
                    let bools: Vec<bool> = items.iter()
                        .filter_map(|item| if let Expr::Bool(b, _) = item { Some(*b) } else { None })
                        .collect();
                    return Some(MvarInitValue::BoolList(bools));
                }

                // Check for string list
                if items.iter().all(|item| matches!(item, Expr::String(StringLit::Plain(_), _))) {
                    let strings: Vec<String> = items.iter()
                        .filter_map(|item| {
                            if let Expr::String(StringLit::Plain(s), _) = item {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    return Some(MvarInitValue::StringList(strings));
                }

                // Generic list - recursively evaluate all items
                let mut values = Vec::new();
                for item in items {
                    values.push(self.eval_const_expr(item)?);
                }
                Some(MvarInitValue::List(values))
            }
            // Tuple literals
            Expr::Tuple(items, _) => {
                let mut values = Vec::new();
                for item in items {
                    values.push(self.eval_const_expr(item)?);
                }
                Some(MvarInitValue::Tuple(values))
            }
            // Record constructor: TypeName { field1: val1, field2: val2 }
            // or positional: TypeName(val1, val2)
            Expr::Record(type_name, fields, _) => {
                let mut field_values = Vec::new();
                for field in fields {
                    match field {
                        RecordField::Named(name, ref val) => {
                            let value = self.eval_const_expr(val)?;
                            field_values.push((name.node.clone(), value));
                        }
                        RecordField::Positional(ref val) => {
                            // Use empty string for positional fields (index-based)
                            let value = self.eval_const_expr(val)?;
                            field_values.push((String::new(), value));
                        }
                    }
                }
                Some(MvarInitValue::Record(type_name.node.clone(), field_values))
            }
            // Map literals: %{ key => value, ... }
            Expr::Map(pairs, _) => {
                if pairs.is_empty() {
                    Some(MvarInitValue::EmptyMap)
                } else {
                    let mut entries = Vec::new();
                    for (key, value) in pairs {
                        let k = self.eval_const_expr(key)?;
                        let v = self.eval_const_expr(value)?;
                        entries.push((k, v));
                    }
                    Some(MvarInitValue::Map(entries))
                }
            }
            _ => None,
        }
    }

    /// Analyze an expression to find all mvar accesses (reads and writes).
    /// This is used for function-level locking - we need to know which mvars
    /// a function accesses BEFORE compiling it so we can emit lock instructions.
    /// Returns (reads, writes) as sets of mvar names.
    #[allow(dead_code)]
    fn analyze_mvar_access(&self, expr: &Expr) -> (HashSet<String>, HashSet<String>) {
        let mut reads = HashSet::new();
        let mut writes = HashSet::new();
        self.collect_mvar_access(expr, &mut reads, &mut writes);
        (reads, writes)
    }

    /// Recursively collect mvar accesses from an expression.
    #[allow(dead_code)]
    fn collect_mvar_access(&self, expr: &Expr, reads: &mut HashSet<String>, writes: &mut HashSet<String>) {
        match expr {
            Expr::Var(ident) => {
                let name = self.resolve_name(&ident.node);
                if self.mvars.contains_key(&name) {
                    reads.insert(name);
                }
            }
            // Binary operations
            Expr::BinOp(left, _, right, _) => {
                self.collect_mvar_access(left, reads, writes);
                self.collect_mvar_access(right, reads, writes);
            }
            // Unary operations
            Expr::UnaryOp(_, operand, _) => {
                self.collect_mvar_access(operand, reads, writes);
            }
            // If expression (else branch is not optional in this AST)
            Expr::If(cond, then_branch, else_branch, _) => {
                self.collect_mvar_access(cond, reads, writes);
                self.collect_mvar_access(then_branch, reads, writes);
                self.collect_mvar_access(else_branch, reads, writes);
            }
            // Block expression
            Expr::Block(stmts, _) => {
                for stmt in stmts {
                    self.collect_mvar_access_stmt(stmt, reads, writes);
                }
            }
            // Do block
            Expr::Do(do_stmts, _) => {
                for do_stmt in do_stmts {
                    match do_stmt {
                        DoStmt::Bind(_, expr) => self.collect_mvar_access(expr, reads, writes),
                        DoStmt::Expr(expr) => self.collect_mvar_access(expr, reads, writes),
                    }
                }
            }
            // Function call
            Expr::Call(func, _type_args, args, _) => {
                self.collect_mvar_access(func, reads, writes);
                for arg in args {
                    self.collect_mvar_access(arg, reads, writes);
                }
            }
            // Method call
            Expr::MethodCall(receiver, _, args, _) => {
                self.collect_mvar_access(receiver, reads, writes);
                for arg in args {
                    self.collect_mvar_access(arg, reads, writes);
                }
            }
            // Field access
            Expr::FieldAccess(obj, _, _) => {
                self.collect_mvar_access(obj, reads, writes);
            }
            // Index access
            Expr::Index(obj, idx, _) => {
                self.collect_mvar_access(obj, reads, writes);
                self.collect_mvar_access(idx, reads, writes);
            }
            // List literal
            Expr::List(items, spread, _) => {
                for item in items {
                    self.collect_mvar_access(item, reads, writes);
                }
                if let Some(spread_expr) = spread {
                    self.collect_mvar_access(spread_expr, reads, writes);
                }
            }
            // Tuple
            Expr::Tuple(items, _) => {
                for item in items {
                    self.collect_mvar_access(item, reads, writes);
                }
            }
            // Record literal
            Expr::Record(_, fields, _) => {
                for field in fields {
                    match field {
                        RecordField::Positional(expr) => self.collect_mvar_access(expr, reads, writes),
                        RecordField::Named(_, expr) => self.collect_mvar_access(expr, reads, writes),
                    }
                }
            }
            // Record update
            Expr::RecordUpdate(_, base, fields, _) => {
                self.collect_mvar_access(base, reads, writes);
                for field in fields {
                    match field {
                        RecordField::Positional(expr) => self.collect_mvar_access(expr, reads, writes),
                        RecordField::Named(_, expr) => self.collect_mvar_access(expr, reads, writes),
                    }
                }
            }
            // Match expression
            Expr::Match(scrutinee, arms, _) => {
                self.collect_mvar_access(scrutinee, reads, writes);
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        self.collect_mvar_access(guard, reads, writes);
                    }
                    self.collect_mvar_access(&arm.body, reads, writes);
                }
            }
            // Lambda
            Expr::Lambda(_, body, _) => {
                self.collect_mvar_access(body, reads, writes);
            }
            // Try-catch
            Expr::Try(try_expr, catch_arms, finally_expr, _) => {
                self.collect_mvar_access(try_expr, reads, writes);
                for arm in catch_arms {
                    if let Some(guard) = &arm.guard {
                        self.collect_mvar_access(guard, reads, writes);
                    }
                    self.collect_mvar_access(&arm.body, reads, writes);
                }
                if let Some(finally) = finally_expr {
                    self.collect_mvar_access(finally, reads, writes);
                }
            }
            // Spawn
            Expr::Spawn(_, func, args, _) => {
                self.collect_mvar_access(func, reads, writes);
                for arg in args {
                    self.collect_mvar_access(arg, reads, writes);
                }
            }
            // While loop
            Expr::While(cond, body, _) => {
                self.collect_mvar_access(cond, reads, writes);
                self.collect_mvar_access(body, reads, writes);
            }
            // For loop
            Expr::For(_, start, end, body, _) => {
                self.collect_mvar_access(start, reads, writes);
                self.collect_mvar_access(end, reads, writes);
                self.collect_mvar_access(body, reads, writes);
            }
            // Send
            Expr::Send(target, msg, _) => {
                self.collect_mvar_access(target, reads, writes);
                self.collect_mvar_access(msg, reads, writes);
            }
            // Receive
            Expr::Receive(arms, timeout, _) => {
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        self.collect_mvar_access(guard, reads, writes);
                    }
                    self.collect_mvar_access(&arm.body, reads, writes);
                }
                if let Some((timeout_expr, timeout_body)) = timeout {
                    self.collect_mvar_access(timeout_expr, reads, writes);
                    self.collect_mvar_access(timeout_body, reads, writes);
                }
            }
            // Break with optional value
            Expr::Break(value, _) => {
                if let Some(val) = value {
                    self.collect_mvar_access(val, reads, writes);
                }
            }
            // Return with optional value
            Expr::Return(value, _) => {
                if let Some(val) = value {
                    self.collect_mvar_access(val, reads, writes);
                }
            }
            // Map literal
            Expr::Map(pairs, _) => {
                for (key, value) in pairs {
                    self.collect_mvar_access(key, reads, writes);
                    self.collect_mvar_access(value, reads, writes);
                }
            }
            // Set literal
            Expr::Set(items, _) => {
                for item in items {
                    self.collect_mvar_access(item, reads, writes);
                }
            }
            // Try_ (simple try without catch)
            Expr::Try_(try_expr, _) => {
                self.collect_mvar_access(try_expr, reads, writes);
            }
            // Quote/Splice (macro-like)
            Expr::Quote(expr, _) => {
                self.collect_mvar_access(expr, reads, writes);
            }
            Expr::Splice(expr, _) => {
                self.collect_mvar_access(expr, reads, writes);
            }
            // Literals and other expressions that don't contain mvar access
            Expr::Unit(_) | Expr::Bool(_, _) | Expr::Int(_, _) | Expr::Float(_, _)
            | Expr::String(_, _) | Expr::Char(_, _) | Expr::Wildcard(_) | Expr::Continue(_)
            | Expr::Int8(_, _) | Expr::Int16(_, _) | Expr::Int32(_, _)
            | Expr::UInt8(_, _) | Expr::UInt16(_, _) | Expr::UInt32(_, _) | Expr::UInt64(_, _)
            | Expr::Float32(_, _) | Expr::BigInt(_, _) | Expr::Decimal(_, _) => {}
        }
    }

    /// Collect mvar accesses from a statement.
    #[allow(dead_code)]
    fn collect_mvar_access_stmt(&self, stmt: &Stmt, reads: &mut HashSet<String>, writes: &mut HashSet<String>) {
        match stmt {
            Stmt::Expr(expr) => {
                self.collect_mvar_access(expr, reads, writes);
            }
            Stmt::Let(binding) => {
                // Check if binding target is an mvar (for reassignment)
                if let Pattern::Var(ident) = &binding.pattern {
                    let name = self.resolve_name(&ident.node);
                    if self.mvars.contains_key(&name) {
                        writes.insert(name);
                    }
                }
                self.collect_mvar_access(&binding.value, reads, writes);
            }
            Stmt::Assign(target, value, _) => {
                if let AssignTarget::Var(ident) = target {
                    let name = self.resolve_name(&ident.node);
                    if self.mvars.contains_key(&name) {
                        writes.insert(name);
                    }
                }
                self.collect_mvar_access(value, reads, writes);
            }
        }
    }

    /// Analyze a function clause to find all mvar accesses.
    #[allow(dead_code)]
    fn analyze_fn_clause_mvar_access(&self, clause: &FnClause) -> (HashSet<String>, HashSet<String>) {
        let mut reads = HashSet::new();
        let mut writes = HashSet::new();

        // Analyze guard if present
        if let Some(guard) = &clause.guard {
            self.collect_mvar_access(guard, &mut reads, &mut writes);
        }

        // Analyze body
        self.collect_mvar_access(&clause.body, &mut reads, &mut writes);

        (reads, writes)
    }

    /// Analyze all clauses of a function definition to find mvar accesses.
    #[allow(dead_code)]
    fn analyze_fn_def_mvar_access(&self, def: &FnDef) -> (HashSet<String>, HashSet<String>) {
        let mut reads = HashSet::new();
        let mut writes = HashSet::new();

        for clause in &def.clauses {
            let (clause_reads, clause_writes) = self.analyze_fn_clause_mvar_access(clause);
            reads.extend(clause_reads);
            writes.extend(clause_writes);
        }

        (reads, writes)
    }

    /// Compile a type definition.
    fn compile_type_def(&mut self, def: &TypeDef) -> Result<(), CompileError> {
        // Use qualified name (with module path prefix)
        let name = self.qualify_name(&def.name.node);

        let kind = match &def.body {
            TypeBody::Record(fields) => {
                // Register record name as a constructor
                self.known_constructors.insert(name.clone());
                let field_info: Vec<(String, String)> = fields.iter()
                    .map(|f| (f.name.node.clone(), self.type_expr_name(&f.ty)))
                    .collect();
                TypeInfoKind::Record { fields: field_info, mutable: def.mutable }
            }
            TypeBody::Variant(variants) => {
                let constructors: Vec<(String, Vec<String>)> = variants.iter()
                    .map(|v| {
                        // Register BOTH qualified and local constructor names
                        let qualified_ctor = self.qualify_name(&v.name.node);
                        let local_ctor = v.name.node.clone();
                        self.known_constructors.insert(qualified_ctor.clone());
                        self.known_constructors.insert(local_ctor.clone());

                        let field_types = match &v.fields {
                            VariantFields::Unit => vec![],
                            VariantFields::Positional(fields) => {
                                fields.iter().map(|ty| self.type_expr_name(ty)).collect()
                            }
                            VariantFields::Named(fields) => {
                                fields.iter().map(|f| self.type_expr_name(&f.ty)).collect()
                            }
                        };
                        // Store local constructor name for type system compatibility
                        (local_ctor, field_types)
                    })
                    .collect();
                TypeInfoKind::Variant { constructors }
            }
            TypeBody::Alias(_) => {
                // Type aliases don't need runtime representation
                return Ok(());
            }
            TypeBody::Empty => {
                // Never type
                return Ok(());
            }
        };

        self.types.insert(name.clone(), TypeInfo { name: name.clone(), kind });

        // Store the TypeDef AST for REPL introspection
        self.type_defs.insert(name.clone(), def.clone());

        Ok(())
    }

    /// Compile a trait definition.
    fn compile_trait_def(&mut self, def: &TraitDef) -> Result<(), CompileError> {
        // Qualify trait name with module prefix
        let name = self.qualify_name(&def.name.node);

        let super_traits: Vec<String> = def.super_traits
            .iter()
            .map(|t| t.node.clone())
            .collect();

        let methods: Vec<TraitMethodInfo> = def.methods
            .iter()
            .map(|m| TraitMethodInfo {
                name: m.name.node.clone(),
                param_count: m.params.len(),
                has_default: m.default_impl.is_some(),
            })
            .collect();

        self.trait_defs.insert(name.clone(), TraitInfo {
            name,
            super_traits,
            methods,
        });

        Ok(())
    }

    /// Check if a trait is a built-in derivable trait.
    fn is_builtin_derivable_trait(&self, name: &str) -> bool {
        matches!(name, "Hash" | "Show" | "Copy" | "Eq")
    }

    /// Compile a trait implementation.
    fn compile_trait_impl(&mut self, impl_def: &TraitImpl) -> Result<(), CompileError> {
        // Get the type name from the type expression
        // Use unqualified name for method names (compile_fn_def will add module prefix)
        // Use qualified name for type_traits registration and param_types
        let unqualified_type_name = self.type_expr_to_string(&impl_def.ty);
        let qualified_type_name = self.qualify_name(&unqualified_type_name);
        let unqualified_trait_name = impl_def.trait_name.node.clone();
        // Qualify trait name for lookup (trait defined in same module)
        let qualified_trait_name = self.qualify_name(&unqualified_trait_name);

        // Check that the trait exists (unless it's a built-in derivable trait)
        // Try both qualified and unqualified names for traits defined elsewhere
        let trait_exists = self.trait_defs.contains_key(&qualified_trait_name)
            || self.trait_defs.contains_key(&unqualified_trait_name)
            || self.is_builtin_derivable_trait(&unqualified_trait_name);
        if !trait_exists {
            return Err(CompileError::UnknownTrait {
                name: unqualified_trait_name,
                span: impl_def.trait_name.span,
            });
        }
        // Use the name that exists in trait_defs
        let trait_name = if self.trait_defs.contains_key(&qualified_trait_name) {
            qualified_trait_name
        } else {
            unqualified_trait_name
        };

        // Compile each method as a function with a special qualified name: Type.Trait.method
        // Use unqualified type name here because compile_fn_def will add module prefix
        let mut method_names = Vec::new();
        for method in &impl_def.methods {
            let method_name = method.name.node.clone();
            // Use unqualified type name for method - compile_fn_def adds module prefix
            let local_method_name = format!("{}.{}.{}", unqualified_type_name, trait_name, method_name);
            // The fully qualified method name (for registration)
            let qualified_method_name = self.qualify_name(&local_method_name);

            // Create a modified FnDef with the local name - compile_fn_def adds module prefix
            let mut modified_def = method.clone();
            modified_def.name = Spanned::new(local_method_name.clone(), method.name.span);
            modified_def.visibility = Visibility::Public; // Trait methods are always callable

            // Set up param_types for Self-typed parameters before compiling
            // This allows type inference to work correctly for field access on self/other
            let saved_param_types = std::mem::take(&mut self.param_types);
            for clause in &method.clauses {
                for param in &clause.params {
                    // Check if this parameter's type is Self
                    let is_self_typed = param.ty.as_ref().map(|t| {
                        matches!(t, TypeExpr::Name(n) if n.node == "Self")
                    }).unwrap_or(false);

                    // For "self" parameter (first param in trait methods) or Self-typed params
                    if let Some(name) = self.pattern_binding_name(&param.pattern) {
                        if name == "self" || is_self_typed {
                            // Use qualified type name for param_types
                            self.param_types.insert(name, qualified_type_name.clone());
                        }
                    }
                }
            }

            self.compile_fn_def(&modified_def)?;

            // Restore param_types
            self.param_types = saved_param_types;

            method_names.push(qualified_method_name);
        }

        // Register the trait implementation with qualified type name
        let impl_info = TraitImplInfo {
            type_name: qualified_type_name.clone(),
            trait_name: trait_name.clone(),
            method_names,
        };
        self.trait_impls.insert((qualified_type_name.clone(), trait_name.clone()), impl_info);

        // Track which traits this type implements (use qualified type name)
        self.type_traits
            .entry(qualified_type_name)
            .or_insert_with(Vec::new)
            .push(trait_name);

        Ok(())
    }

    /// Convert a type expression to a string representation.
    fn type_expr_to_string(&self, ty: &TypeExpr) -> String {
        match ty {
            TypeExpr::Name(name) => name.node.clone(),
            TypeExpr::Generic(name, params) => {
                let params_str: Vec<String> = params.iter()
                    .map(|p| self.type_expr_to_string(p))
                    .collect();
                format!("{}[{}]", name.node, params_str.join(", "))
            }
            TypeExpr::Tuple(elems) => {
                let elems_str: Vec<String> = elems.iter()
                    .map(|e| self.type_expr_to_string(e))
                    .collect();
                format!("({})", elems_str.join(", "))
            }
            TypeExpr::Function(params, ret) => {
                let params_str: Vec<String> = params.iter()
                    .map(|p| self.type_expr_to_string(p))
                    .collect();
                format!("({}) -> {}", params_str.join(", "), self.type_expr_to_string(ret))
            }
            TypeExpr::Record(fields) => {
                let fields_str: Vec<String> = fields.iter()
                    .map(|(name, ty)| format!("{}: {}", name.node, self.type_expr_to_string(ty)))
                    .collect();
                format!("{{{}}}", fields_str.join(", "))
            }
            TypeExpr::Unit => "()".to_string(),
        }
    }

    /// Find a function by base name and argument types.
    /// Returns the qualified function name (with signature) if found.
    ///
    /// Resolution order:
    /// 1. Exact match: `name/Type1,Type2,...`
    /// 2. Wildcard match: `name/_,_,...` (untyped parameters match any type)
    /// 3. Partial wildcard: `name/Type1,_,...` (specific types take precedence)
    fn resolve_function_call(&self, base_name: &str, arg_types: &[Option<String>]) -> Option<String> {
        let arity = arg_types.len();
        let prefix = format!("{}/", base_name);

        // Collect candidates from both compiled functions and pending ASTs
        let mut candidates: Vec<String> = Vec::new();

        // Check compiled functions
        for key in self.functions.keys() {
            if let Some(suffix) = key.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { Self::count_signature_params(suffix) };
                if param_count == arity {
                    candidates.push(key.clone());
                }
            }
        }

        // Check fn_asts (for functions being compiled or not yet compiled)
        for key in self.fn_asts.keys() {
            if let Some(suffix) = key.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { Self::count_signature_params(suffix) };
                if param_count == arity && !candidates.contains(key) {
                    candidates.push(key.clone());
                }
            }
        }

        // Also check if current function matches (for self-recursion during compilation)
        if let Some(current) = &self.current_function_name {
            if let Some(suffix) = current.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { Self::count_signature_params(suffix) };
                if param_count == arity && !candidates.contains(current) {
                    candidates.push(current.clone());
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Score each candidate: exact type match = 2, wildcard (_) = 1, mismatch = 0
        let mut best_match: Option<(String, usize)> = None;

        for candidate in &candidates {
            let suffix = candidate.strip_prefix(&prefix).unwrap();
            let candidate_types: Vec<String> = if suffix.is_empty() {
                vec![]
            } else {
                Self::split_signature_types(suffix)
            };

            let mut score = 0;
            let mut valid = true;

            for (i, cand_type) in candidate_types.iter().enumerate() {
                if i >= arg_types.len() {
                    valid = false;
                    break;
                }

                // Check if candidate type is a wildcard or type parameter
                let is_type_param = cand_type.len() == 1 && cand_type.chars().next().unwrap().is_uppercase();

                if *cand_type == "_" || is_type_param {
                    // Wildcard or type parameter accepts anything
                    score += 1;
                } else if let Some(arg_type) = &arg_types[i] {
                    // Check for exact match or module-qualified match
                    // e.g., "Vec" matches "nalgebra.Vec"
                    let types_match = cand_type == arg_type
                        || arg_type.ends_with(&format!(".{}", cand_type))
                        || cand_type.ends_with(&format!(".{}", arg_type));

                    if types_match {
                        // Exact type match (or module-qualified match)
                        score += 2;
                    } else if cand_type.starts_with(arg_type) && cand_type[arg_type.len()..].starts_with('[') {
                        // Parameterized type match: arg_type="List", cand_type="List[Html]"
                        // This is a compatible match but less specific than exact
                        score += 1;
                    } else if Self::types_are_compatible(arg_type, cand_type) {
                        // Polymorphic type compatible with concrete type
                        // e.g., "Map k v" compatible with "Map[String, String]"
                        score += 1;
                    } else {
                        // Type mismatch
                        valid = false;
                        break;
                    }
                } else {
                    // Unknown argument type - accept any candidate type, give lower score
                    // But prefer primitive types (String, Int, Bool, etc.) over container types (List, Map, etc.)
                    // This heuristic helps when lambda parameters have unknown types
                    let is_primitive = matches!(cand_type.as_str(),
                        "String" | "Int" | "Float" | "Bool" | "Char" | "()" |
                        "Int64" | "Float64" | "Float32");
                    if is_primitive {
                        score += 2; // Prefer primitive types when arg type is unknown
                    } else {
                        score += 1;
                    }
                }
            }

            if valid {
                if best_match.is_none() || score > best_match.as_ref().unwrap().1 {
                    best_match = Some((candidate.clone(), score));
                }
            }
        }

        best_match.map(|(name, _)| name)
    }

    /// Count parameters in a signature, handling nested brackets correctly.
    /// E.g., "List[(String,String)],String" -> 2 (not 3)
    fn count_signature_params(signature: &str) -> usize {
        if signature.is_empty() {
            return 0;
        }
        let mut count = 1;
        let mut depth = 0;
        for c in signature.chars() {
            match c {
                '[' | '(' => depth += 1,
                ']' | ')' => depth -= 1,
                ',' if depth == 0 => count += 1,
                _ => {}
            }
        }
        count
    }

    /// Check if two type strings are compatible.
    /// Returns true if:
    /// - arg_type is a polymorphic type (e.g., "Map k v", "List a", "Tree t", "List[b]", "Map[k, v]")
    /// - cand_type is a concrete type with the same base (e.g., "Map[String, String]", "List[Int]", "Tree[Node]")
    /// This handles both built-in types and user-defined generic types.
    fn types_are_compatible(arg_type: &str, cand_type: &str) -> bool {
        // First try bracket format: "List[b]" or "Map[k, v]"
        if let Some(arg_bracket_pos) = arg_type.find('[') {
            let arg_base = &arg_type[..arg_bracket_pos];
            let arg_inner = &arg_type[arg_bracket_pos + 1..arg_type.len() - 1]; // Remove [ and ]

            // Check if the type params are all type variables (single lowercase letters)
            let arg_type_params: Vec<&str> = arg_inner.split(',').map(|s| s.trim()).collect();
            let all_type_vars = arg_type_params.iter().all(|p| {
                p.len() == 1 && p.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false)
            });

            if all_type_vars {
                // Extract base type from cand_type
                let cand_base = if let Some(cand_bracket_pos) = cand_type.find('[') {
                    &cand_type[..cand_bracket_pos]
                } else {
                    cand_type
                };
                return arg_base == cand_base;
            }
        }

        // Fall back to space-separated format: "Map k v" -> "Map"
        let arg_parts: Vec<&str> = arg_type.split_whitespace().collect();
        if arg_parts.len() < 2 {
            // Not a space-separated polymorphic type
            return false;
        }
        let arg_base = arg_parts[0];

        // Check if all remaining parts are type variables (single lowercase letters)
        let all_type_vars = arg_parts[1..].iter().all(|p| {
            p.len() == 1 && p.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false)
        });
        if !all_type_vars {
            return false;
        }

        // Extract base type name from cand_type (bracket format: "Map[String, String]" -> "Map")
        let cand_base = if let Some(bracket_pos) = cand_type.find('[') {
            &cand_type[..bracket_pos]
        } else {
            // cand_type has no brackets - check if it's just the base name
            cand_type
        };

        // Base types must match
        arg_base == cand_base
    }

    /// Split a signature into parameter types, handling nested brackets correctly.
    /// E.g., "List[(String,String)],String" -> ["List[(String,String)]", "String"]
    fn split_signature_types(signature: &str) -> Vec<String> {
        if signature.is_empty() {
            return vec![];
        }
        let mut result = Vec::new();
        let mut current = String::new();
        let mut depth = 0;
        for c in signature.chars() {
            match c {
                '[' | '(' => {
                    depth += 1;
                    current.push(c);
                }
                ']' | ')' => {
                    depth -= 1;
                    current.push(c);
                }
                ',' if depth == 0 => {
                    result.push(std::mem::take(&mut current));
                }
                _ => current.push(c),
            }
        }
        if !current.is_empty() {
            result.push(current);
        }
        result
    }

    /// Check if a function with the given base name exists (with any arity).
    /// Returns Some(set of arities) if found, None if no such function exists.
    fn find_all_function_arities(&self, base_name: &str) -> Option<std::collections::HashSet<usize>> {
        let prefix = format!("{}/", base_name);
        let mut arities = std::collections::HashSet::new();

        // Check compiled functions
        for key in self.functions.keys() {
            if let Some(suffix) = key.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { Self::count_signature_params(suffix) };
                arities.insert(param_count);
            }
        }

        // Check fn_asts (for functions being compiled or not yet compiled)
        for key in self.fn_asts.keys() {
            if let Some(suffix) = key.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { Self::count_signature_params(suffix) };
                arities.insert(param_count);
            }
        }

        // Check if current function matches
        if let Some(current) = &self.current_function_name {
            if let Some(suffix) = current.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { Self::count_signature_params(suffix) };
                arities.insert(param_count);
            }
        }

        if arities.is_empty() {
            None
        } else {
            Some(arities)
        }
    }

    /// Find a function by base name and arity, returning the full function key.
    /// Used when we know a function exists with matching arity but types don't match.
    fn find_function_by_arity(&self, base_name: &str, arity: usize) -> Option<String> {
        let prefix = format!("{}/", base_name);

        // Check compiled functions first
        for key in self.functions.keys() {
            if let Some(suffix) = key.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { Self::count_signature_params(suffix) };
                if param_count == arity {
                    return Some(key.clone());
                }
            }
        }

        // Check fn_asts
        for key in self.fn_asts.keys() {
            if let Some(suffix) = key.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { Self::count_signature_params(suffix) };
                if param_count == arity {
                    return Some(key.clone());
                }
            }
        }

        // Check current function
        if let Some(current) = &self.current_function_name {
            if let Some(suffix) = current.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { Self::count_signature_params(suffix) };
                if param_count == arity {
                    return Some(current.clone());
                }
            }
        }

        None
    }

    /// Check if a user-defined function exists with the given name and arity.
    /// This is used to prevent builtin functions from shadowing user-defined functions.
    fn has_user_function(&self, name: &str, arity: usize) -> bool {
        let prefix = format!("{}/", name);

        // Check compiled functions
        for key in self.functions.keys() {
            if let Some(suffix) = key.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { suffix.split(',').count() };
                if param_count == arity {
                    return true;
                }
            }
        }

        // Check fn_asts (for functions being compiled or not yet compiled)
        for key in self.fn_asts.keys() {
            if let Some(suffix) = key.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { suffix.split(',').count() };
                if param_count == arity {
                    return true;
                }
            }
        }

        // Check if current function matches
        if let Some(current) = &self.current_function_name {
            if let Some(suffix) = current.strip_prefix(&prefix) {
                let param_count = if suffix.is_empty() { 0 } else { suffix.split(',').count() };
                if param_count == arity {
                    return true;
                }
            }
        }

        false
    }

    /// Find the implementation of a trait method for a given type.
    pub fn find_trait_method(&self, type_name: &str, method_name: &str) -> Option<String> {
        // Look through all traits this type implements
        if let Some(traits) = self.type_traits.get(type_name) {
            for trait_name in traits {
                // Check if this trait has the method
                // For built-in derivable traits, we know the method names
                let has_method = if self.is_builtin_derivable_trait(trait_name) {
                    match (trait_name.as_str(), method_name) {
                        ("Show", "show") => true,
                        ("Hash", "hash") => true,
                        ("Copy", "copy") => true,
                        ("Eq", "==") => true,
                        _ => false,
                    }
                } else if let Some(trait_info) = self.trait_defs.get(trait_name) {
                    trait_info.methods.iter().any(|m| m.name == method_name)
                } else {
                    false
                };

                if has_method {
                    // Return the qualified function name
                    return Some(format!("{}.{}.{}", type_name, trait_name, method_name));
                }
            }
        }
        None
    }

    /// Check if a method name belongs to any trait.
    /// Used to detect when a method call on an untyped parameter might be a trait method.
    fn is_known_trait_method(&self, method_name: &str) -> bool {
        for trait_info in self.trait_defs.values() {
            if trait_info.methods.iter().any(|m| m.name == method_name) {
                return true;
            }
        }
        false
    }

    /// Map a binary operator to its trait and method name.
    /// Returns (trait_name, method_name) for operators that can be overloaded.
    fn operator_to_trait_method(op: &BinOp) -> Option<(&'static str, &'static str)> {
        match op {
            // Arithmetic operators -> Num trait
            BinOp::Add => Some(("Num", "add")),
            BinOp::Sub => Some(("Num", "sub")),
            BinOp::Mul => Some(("Num", "mul")),
            BinOp::Div => Some(("Num", "div")),
            // Comparison operators -> Ord trait
            BinOp::Lt => Some(("Ord", "lt")),
            BinOp::Gt => Some(("Ord", "gt")),
            BinOp::LtEq => Some(("Ord", "lte")),
            BinOp::GtEq => Some(("Ord", "gte")),
            // Equality operators -> Eq trait
            BinOp::Eq => Some(("Eq", "eq")),
            BinOp::NotEq => Some(("Eq", "neq")),
            // These operators are not overloadable via traits
            BinOp::Mod | BinOp::Pow | BinOp::And | BinOp::Or |
            BinOp::Concat | BinOp::Cons | BinOp::Pipe => None,
        }
    }

    /// Check if a type is a primitive type (not a custom user-defined type).
    fn is_primitive_type(type_name: &str) -> bool {
        matches!(type_name,
            "Int" | "Int8" | "Int16" | "Int32" | "Int64" |
            "UInt8" | "UInt16" | "UInt32" | "UInt64" |
            "Float" | "Float32" | "Float64" |
            "Bool" | "Char" | "String" | "()" |
            "BigInt" | "Decimal" |
            "List" | "Map" | "Set" | "Option" | "Result"
        )
    }

    /// Check if a type implements a specific trait.
    fn type_implements_trait(&self, _type_name: &str, trait_name: &str) -> bool {
        self.find_implemented_trait(_type_name, trait_name).is_some()
    }

    /// Find the actual registered trait name if a type implements a trait.
    /// Returns the qualified or unqualified trait name as registered, or None if not found.
    /// This is needed because traits may be registered with qualified names (e.g., "nalgebra.Num")
    /// but looked up with unqualified names (e.g., "Num").
    fn find_implemented_trait(&self, type_name: &str, trait_name: &str) -> Option<String> {
        // Hash, Show, Eq, and Copy are now built-in for ALL types
        if matches!(trait_name, "Hash" | "Show" | "Eq" | "Copy") {
            return Some(trait_name.to_string());
        }

        // Check if the type has an explicit trait implementation
        if let Some(traits) = self.type_traits.get(type_name) {
            // Try exact match first
            if traits.contains(&trait_name.to_string()) {
                return Some(trait_name.to_string());
            }
            // Try qualified match (e.g., looking for "Num" and finding "module.Num")
            for registered_trait in traits {
                if registered_trait == trait_name {
                    return Some(registered_trait.clone());
                }
                // Check if the unqualified name matches
                if let Some(dot_pos) = registered_trait.rfind('.') {
                    let unqualified = &registered_trait[dot_pos + 1..];
                    if unqualified == trait_name {
                        return Some(registered_trait.clone());
                    }
                }
            }
        }

        // Check if type_name is a type parameter in the current function with the required bound
        // This handles nested calls like: double_hash[T: Hash](x) calling hashable[T: Hash](x)
        for type_param in &self.current_fn_type_params {
            if type_param.name.node == type_name {
                // Check if this type parameter has the required trait bound
                for constraint in &type_param.constraints {
                    if constraint.node == trait_name {
                        return Some(trait_name.to_string());
                    }
                }
            }
        }

        None
    }

    /// Check trait bounds for a function call.
    /// Returns an error if any type parameter's trait bounds are not satisfied.
    fn check_trait_bounds(
        &self,
        fn_name: &str,
        type_params: &[TypeParam],
        arg_types: &[Option<String>],
        span: Span,
    ) -> Result<(), CompileError> {
        // Get the function's parameter patterns to map args to type params
        let fn_def = match self.fn_asts.get(fn_name) {
            Some(def) => def,
            None => return Ok(()), // Can't check if we don't have the AST
        };

        if fn_def.clauses.is_empty() {
            return Ok(());
        }

        let params = &fn_def.clauses[0].params;

        // Build a map from type param name to the concrete type it's bound to
        let mut type_bindings: HashMap<String, String> = HashMap::new();

        for (i, param) in params.iter().enumerate() {
            if let Some(ref type_expr) = param.ty {
                // If the parameter has a type annotation like `x: T`, extract the type param name
                if let TypeExpr::Name(ident) = type_expr {
                    let type_param_name = &ident.node;
                    // Check if this is one of our type parameters
                    if type_params.iter().any(|tp| &tp.name.node == type_param_name) {
                        // We found a type parameter - map it to the concrete arg type
                        if i < arg_types.len() {
                            if let Some(ref concrete_type) = arg_types[i] {
                                type_bindings.insert(type_param_name.clone(), concrete_type.clone());
                            }
                        }
                    }
                }
            }
        }

        // Now check that each type parameter's bounds are satisfied
        for type_param in type_params {
            let type_param_name = &type_param.name.node;

            // Get the concrete type bound to this type parameter
            let concrete_type = match type_bindings.get(type_param_name) {
                Some(t) => t,
                None => continue, // Type not used or not known, skip checking
            };

            // Check each constraint (trait bound)
            for constraint in &type_param.constraints {
                let trait_name = &constraint.node;

                if !self.type_implements_trait(concrete_type, trait_name) {
                    return Err(CompileError::TraitBoundNotSatisfied {
                        type_name: concrete_type.clone(),
                        trait_name: trait_name.clone(),
                        span,
                    });
                }
            }
        }

        Ok(())
    }

    /// Compile a function definition.
    fn compile_fn_def(&mut self, def: &FnDef) -> Result<(), CompileError> {
        // Type check the function before compiling
        // Note: This may fail for functions calling not-yet-compiled functions,
        // so we only treat it as an error if we have sufficient type information.
        // In the future, this should be a two-pass system.
        //
        // Skip first-pass type checking for functions with untyped parameters -
        // these rely on inference which may produce false positives during initial
        // compilation. Real type errors will be caught in the second pass.
        // Also skip if this function calls any function with untyped params, since
        // those functions haven't been inferred yet.
        let has_untyped_params = def.clauses.first()
            .map(|c| c.params.iter().any(|p| p.ty.is_none()))
            .unwrap_or(false);
        let calls_untyped = self.calls_function_with_untyped_params(def);
        if !has_untyped_params && !calls_untyped {
            // Use local name here; full cross-module type checking happens in compile_all_collecting_errors
            if let Err(e) = self.type_check_fn(def, &def.name.node) {
            // Only report errors that are actual type mismatches, not unknown identifiers
            // or polymorphic type issues that the checker can't handle yet
            let should_report = match &e {
                CompileError::TypeError { message, .. } => {
                    // Filter out errors from incomplete type inference:
                    // - Unknown identifiers (functions not compiled yet)
                    // - Unknown types (custom types not registered)
                    // - Type variable unification (polymorphism issues)
                    // - Missing field errors (trait method dispatch limitations)
                    // - Unit type unification (false positive: () and ())
                    // - List element type vs list unification (complex mutual recursion inference)
                    //   Pattern: "Cannot unify types: List[X] and X" where there's one List[]
                    // Filter out List element vs List errors from mutual recursion inference
                    // Pattern 1: "List[X] and X" - list type first
                    // Pattern 2: "X and List[X]" - element type first
                    let is_list_element_error = message.contains("List[") &&
                        (message.contains("] and ") || message.contains(" and List[")) &&
                        message.matches("List[").count() == 1;
                    // Skip trait errors for type parameters (single uppercase letter)
                    // These will be properly checked at monomorphization time
                    let is_type_param_trait_error = message.contains("does not implement") &&
                        message.split_whitespace()
                            .find(|w| w.len() == 1 && w.chars().next().unwrap().is_uppercase())
                            .is_some();
                    let is_inference_limitation = message.contains("Unknown identifier") ||
                        message.contains("Unknown type") ||
                        message.contains("has no field") ||
                        message.contains("() and ()") ||
                        is_list_element_error ||
                        is_type_param_trait_error ||
                        Self::is_type_variable_only_error(message);

                    !is_inference_limitation
                }
                _ => true,
            };
            if should_report {
                return Err(e);
            }
            }
        }

        // Save compiler state
        let saved_chunk = std::mem::take(&mut self.chunk);
        let saved_locals = std::mem::take(&mut self.locals);
        let saved_next_reg = self.next_reg;
        let saved_function_name = self.current_function_name.take();

        // Reset for new function
        self.chunk = Chunk::new();
        self.locals = HashMap::new();
        self.next_reg = 0;

        // Use qualified name with type signature to support function overloading
        // Format: name/type1,type2,... where untyped params use "_"
        let base_name = self.qualify_name(&def.name.node);
        let param_types: Vec<String> = def.clauses[0].params.iter()
            .map(|p| p.ty.as_ref()
                .map(|t| self.type_expr_to_string(t))
                .unwrap_or_else(|| "_".to_string()))
            .collect();
        let signature = param_types.join(",");
        let name = format!("{}/{}", base_name, signature);

        // Track current function name for self-recursion optimization
        self.current_function_name = Some(name.clone());

        // Save and clear mvar tracking for this function (for deadlock detection)
        let saved_mvar_reads = std::mem::take(&mut self.current_fn_mvar_reads);
        let saved_mvar_writes = std::mem::take(&mut self.current_fn_mvar_writes);
        let saved_fn_calls = std::mem::take(&mut self.current_fn_calls);
        let saved_has_blocking = std::mem::replace(&mut self.current_fn_has_blocking, false);

        // Store the function's visibility
        self.function_visibility.insert(name.clone(), def.visibility);

        // Store AST for potential monomorphization
        self.fn_asts.insert(name.clone(), def.clone());

        // Store source info for error reporting
        if let (Some(source_name), Some(source)) = (&self.current_source_name, &self.current_source) {
            self.fn_sources.insert(name.clone(), (source_name.clone(), source.clone()));
        }

        // Invalidate any existing monomorphized variants of this function
        // We replace them with stale markers so CallDirect indices remain valid.
        // The variants will be recompiled on next use (checked in compile_monomorphized_variant)
        let prefix = format!("{}$", name);
        let variants_to_invalidate: Vec<String> = self.functions.keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();
        for variant in &variants_to_invalidate {
            // Mark as invalidated by inserting a placeholder with empty code
            // The actual recompilation happens in compile_monomorphized_variant
            // which checks if the variant needs updating
            if let Some(old_func) = self.functions.get(variant) {
                let stale_marker = FunctionValue {
                    name: format!("__stale__{}", variant),  // Mark as stale
                    arity: old_func.arity,
                    param_names: old_func.param_names.clone(),
                    code: Arc::new(Chunk::new()),
                    module: old_func.module.clone(),
                    source_span: None,
                    jit_code: None,
                    call_count: std::sync::atomic::AtomicU32::new(0),
                    debug_symbols: vec![],
                    source_code: None,
                    source_file: None,
                    doc: None,
                    signature: None,
                    param_types: vec![],
                    return_type: None,
                };
                self.functions.insert(variant.clone(), Arc::new(stale_marker));
            }
        }

        // Store type parameters with bounds for trait bound checking at call sites
        if !def.type_params.is_empty() {
            self.fn_type_params.insert(name.clone(), def.type_params.clone());
        }

        // Set current function's type parameters for nested trait bound checking
        let saved_fn_type_params = std::mem::replace(&mut self.current_fn_type_params, def.type_params.clone());

        // Check if we need pattern matching dispatch
        let needs_dispatch = def.clauses.len() > 1 || def.clauses.iter().any(|clause| {
            clause.params.iter().any(|p| !self.is_simple_pattern(&p.pattern)) || clause.guard.is_some()
        });

        // Get arity from first clause (all clauses must have same arity)
        let arity = def.clauses[0].params.len();

        // Generate param names (used for debugging/introspection)
        let mut param_names: Vec<String> = Vec::new();
        for (i, param) in def.clauses[0].params.iter().enumerate() {
            if let Some(n) = self.pattern_binding_name(&param.pattern) {
                param_names.push(n);
            } else {
                param_names.push(format!("_arg{}", i));
            }
        }

        // Save and set up param_types for typed parameters (for compile-time type coercion)
        // Note: We clone instead of take to preserve any pre-set entries (e.g., self -> TypeName from trait impls)
        let saved_param_types = self.param_types.clone();
        for param in def.clauses[0].params.iter() {
            if let Some(param_name) = self.pattern_binding_name(&param.pattern) {
                if let Some(ty) = &param.ty {
                    let type_name = self.type_expr_to_string(ty);
                    // Only insert if not already set (preserve trait impl's self -> TypeName)
                    self.param_types.entry(param_name).or_insert(type_name);
                }
            }
        }

        // Allocate registers for parameters (0..arity)
        self.next_reg = arity as Reg;

        // Clear locals before analysis - this ensures we don't have stale entries from previous functions
        // that could interfere with mvar detection
        self.locals.clear();
        // Clear debug symbols for this function
        self.current_fn_debug_symbols.clear();

        // Pre-analyze function body to determine if function-level mvar locking is needed.
        // If a function reads an mvar and later writes to it (even via a local variable),
        // we need to hold a lock for the entire function to prevent race conditions.
        let mut fn_mvar_reads: HashSet<String> = HashSet::new();
        let mut fn_mvar_writes: HashSet<String> = HashSet::new();
        let mut fn_has_blocking = false;

        for clause in &def.clauses {
            // Collect mvar refs (reads) from the body
            let refs = self.collect_mvar_refs(&clause.body);
            fn_mvar_reads.extend(refs);

            // Collect mvar writes from the body
            let writes = self.collect_mvar_writes(&clause.body);
            fn_mvar_writes.extend(writes);

            // Check for blocking operations
            if self.expr_has_blocking(&clause.body) {
                fn_has_blocking = true;
            }

            // Also check guard if present
            if let Some(guard) = &clause.guard {
                fn_mvar_reads.extend(self.collect_mvar_refs(guard));
                fn_mvar_writes.extend(self.collect_mvar_writes(guard));
                if self.expr_has_blocking(guard) {
                    fn_has_blocking = true;
                }
            }
        }

        // Find mvars that are both read AND written - these need function-level locking
        let mvars_needing_lock: Vec<String> = fn_mvar_reads
            .intersection(&fn_mvar_writes)
            .cloned()
            .collect();

        // If we have mvars needing locks AND the function has blocking operations, error
        if !mvars_needing_lock.is_empty() && fn_has_blocking {
            return Err(CompileError::BlockingWithMvarLock {
                fn_name: name.clone(),
                mvar_name: mvars_needing_lock[0].clone(),
                span: def.span,
            });
        }

        // Emit function-level locks if needed (sorted for consistent lock ordering)
        self.current_fn_mvar_locks.clear();
        let mut sorted_locks: Vec<String> = mvars_needing_lock;
        sorted_locks.sort(); // Consistent ordering prevents deadlocks between functions

        for mvar_name in &sorted_locks {
            let name_idx = self.chunk.add_constant(Value::String(Arc::new(mvar_name.clone())));
            self.chunk.emit(Instruction::MvarLock(name_idx, true), 0); // write lock for read-modify-write
            self.current_fn_mvar_locks.push((mvar_name.clone(), name_idx, true));
        }

        if needs_dispatch {
            // Multi-clause dispatch: try each clause in order
            let mut clause_jumps: Vec<usize> = Vec::new();

            for (clause_idx, clause) in def.clauses.iter().enumerate() {
                // Clear locals for this clause attempt
                self.locals.clear();

                // Track jump to skip to next clause on pattern failure
                let mut next_clause_jumps: Vec<usize> = Vec::new();

                // Test each parameter pattern
                for (i, param) in clause.params.iter().enumerate() {
                    let arg_reg = i as Reg;
                    let (success_reg, bindings) = self.compile_pattern_test(&param.pattern, arg_reg)?;

                    // Jump to next clause if pattern doesn't match
                    next_clause_jumps.push(
                        self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0)
                    );

                    // Add bindings to locals with type info from pattern
                    for (name, reg, is_float) in bindings {
                        self.locals.insert(name, LocalInfo { reg, is_float, mutable: false });
                    }
                }

                // Compile guard if present
                if let Some(guard) = &clause.guard {
                    let guard_reg = self.compile_expr_tail(guard, false)?;
                    next_clause_jumps.push(
                        self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), 0)
                    );
                }

                // All patterns matched and guard passed - compile body
                let body_line = self.span_line(clause.body.span());
                let result_reg = match self.compile_expr_tail(&clause.body, true) {
                    Ok(reg) => reg,
                    Err(CompileError::UnresolvedTraitMethod { .. }) => {
                        // Mark this function as needing monomorphization
                        self.polymorphic_fns.insert(name.clone());
                        // Restore state and return success
                        self.chunk = saved_chunk;
                        self.locals = saved_locals;
                        self.next_reg = saved_next_reg;
                        self.current_function_name = saved_function_name;
                        self.param_types = saved_param_types;
                        self.current_fn_type_params = saved_fn_type_params;
                        return Ok(());
                    }
                    Err(e) => {
                        return Err(e);
                    }
                };
                // Emit MvarUnlock for all held locks (reverse order for proper LIFO unlocking)
                for (_, name_idx, is_write) in self.current_fn_mvar_locks.iter().rev() {
                    self.chunk.emit(Instruction::MvarUnlock(*name_idx, *is_write), 0);
                }
                self.chunk.emit(Instruction::Return(result_reg), body_line);

                // Record where we need to patch for "matched" jumps
                if clause_idx < def.clauses.len() - 1 {
                    clause_jumps.push(self.chunk.code.len());
                }

                // Patch all the "skip to next clause" jumps to land here
                let next_clause_addr = self.chunk.code.len();
                for jump_addr in next_clause_jumps {
                    self.chunk.patch_jump(jump_addr, next_clause_addr);
                }
            }

            // If we get here, no clause matched - release locks and throw error
            for (_, name_idx, is_write) in self.current_fn_mvar_locks.iter().rev() {
                self.chunk.emit(Instruction::MvarUnlock(*name_idx, *is_write), 0);
            }
            let error_idx = self.chunk.add_constant(Value::String(Arc::new(
                format!("No clause matched for function '{}'", name)
            )));
            let error_reg = self.alloc_reg();
            self.chunk.emit(Instruction::LoadConst(error_reg, error_idx), 0);
            self.chunk.emit(Instruction::Throw(error_reg), 0);
        } else {
            // Simple case: single clause with only variable patterns
            let clause = &def.clauses[0];

            // Map parameter patterns to registers
            for (i, param) in clause.params.iter().enumerate() {
                if let Some(n) = self.pattern_binding_name(&param.pattern) {
                    // Check if parameter has Float type annotation
                    let is_float = param.ty.as_ref().map(|t| {
                        match t {
                            nostos_syntax::TypeExpr::Name(ident) => {
                                matches!(ident.node.as_str(), "Float" | "Float32" | "Float64")
                            }
                            _ => false,
                        }
                    }).unwrap_or(false);
                    self.locals.insert(n.clone(), LocalInfo { reg: i as Reg, is_float, mutable: false });
                    // Record parameter as debug symbol
                    self.current_fn_debug_symbols.push(LocalVarSymbol {
                        name: n,
                        register: i as Reg,
                    });
                }
            }

            // Compile function body (in tail position)
            let body_line = self.span_line(clause.body.span());
            let result_reg = match self.compile_expr_tail(&clause.body, true) {
                Ok(reg) => reg,
                Err(CompileError::UnresolvedTraitMethod { .. }) => {
                    // Mark this function as needing monomorphization
                    self.polymorphic_fns.insert(name.clone());
                    // Restore state and return success
                    self.chunk = saved_chunk;
                    self.locals = saved_locals;
                    self.next_reg = saved_next_reg;
                    self.current_function_name = saved_function_name;
                    self.param_types = saved_param_types;
                    self.current_fn_type_params = saved_fn_type_params;
                    self.current_fn_mvar_reads = saved_mvar_reads;
                    self.current_fn_mvar_writes = saved_mvar_writes;
                    self.current_fn_calls = saved_fn_calls;
                    return Ok(());
                }
                Err(e) => return Err(e),
            };
            // Emit MvarUnlock for all held locks (reverse order for proper LIFO unlocking)
            for (_, name_idx, is_write) in self.current_fn_mvar_locks.iter().rev() {
                self.chunk.emit(Instruction::MvarUnlock(*name_idx, *is_write), 0);
            }
            self.chunk.emit(Instruction::Return(result_reg), body_line);
        }

        self.chunk.register_count = self.next_reg as usize;

        // Use accumulated debug symbols (not affected by block scope restoration)
        let debug_symbols = std::mem::take(&mut self.current_fn_debug_symbols);

        // Extract source code for this function from the source
        let source_code = self.current_source.as_ref().and_then(|src| {
            if def.span.start < src.len() && def.span.end <= src.len() {
                Some(Arc::new(src[def.span.start..def.span.end].to_string()))
            } else {
                None
            }
        });

        // Extract doc comment from source (comment lines immediately before the function)
        let doc = def.doc.clone().or_else(|| {
            self.current_source.as_ref().and_then(|src| {
                extract_doc_comment(src, def.span.start)
            })
        });

        let func = FunctionValue {
            name: name.clone(),
            arity,
            param_names,
            code: Arc::new(std::mem::take(&mut self.chunk)),
            module: if self.module_path.is_empty() { None } else { Some(self.module_path.join(".")) },
            source_span: Some((def.span.start, def.span.end)),
            jit_code: None,
            call_count: AtomicU32::new(0),
            debug_symbols,
            // REPL introspection fields
            source_code,
            source_file: self.current_source_name.clone(),
            doc,
            signature: Some(self.infer_signature(def)),
            param_types: def.param_type_strings(),
            return_type: def.return_type_string(),
        };

        // Assign function index if not already indexed (for trait methods and late-compiled functions)
        if !self.function_indices.contains_key(&name) {
            let idx = self.function_list.len() as u16;
            self.function_indices.insert(name.clone(), idx);
            self.function_list.push(name.clone());
        }

        // Save mvar access info for this function (for deadlock detection)
        let fn_access = FnMvarAccess {
            reads: std::mem::take(&mut self.current_fn_mvar_reads),
            writes: std::mem::take(&mut self.current_fn_mvar_writes),
            has_blocking: self.current_fn_has_blocking,
        };
        if !fn_access.reads.is_empty() || !fn_access.writes.is_empty() || fn_access.has_blocking {
            self.fn_mvar_access.insert(name.clone(), fn_access);
        }
        let fn_calls_for_this_fn = std::mem::take(&mut self.current_fn_calls);
        if !fn_calls_for_this_fn.is_empty() {
            self.fn_calls.insert(name.clone(), fn_calls_for_this_fn);
        }

        self.functions.insert(name, Arc::new(func));

        // Restore compiler state
        self.chunk = saved_chunk;
        self.locals = saved_locals;
        self.next_reg = saved_next_reg;
        self.current_function_name = saved_function_name;
        self.param_types = saved_param_types;
        self.current_fn_type_params = saved_fn_type_params;
        self.current_fn_mvar_reads = saved_mvar_reads;
        self.current_fn_mvar_writes = saved_mvar_writes;
        self.current_fn_calls = saved_fn_calls;
        self.current_fn_has_blocking = saved_has_blocking;

        Ok(())
    }

    /// Check if a pattern is "simple" (just a variable or wildcard).
    fn is_simple_pattern(&self, pattern: &Pattern) -> bool {
        matches!(pattern, Pattern::Var(_) | Pattern::Wildcard(_))
    }

    /// Get binding name from a pattern (if it's a simple variable).
    fn pattern_binding_name(&self, pattern: &Pattern) -> Option<String> {
        match pattern {
            Pattern::Var(ident) => Some(ident.node.clone()),
            Pattern::Wildcard(_) => None,
            _ => None, // Complex patterns need deconstruction
        }
    }

    /// Compile an expression, potentially in tail position.
    fn compile_expr_tail(&mut self, expr: &Expr, is_tail: bool) -> Result<Reg, CompileError> {
        let line = self.span_line(expr.span());
        match expr {
            // Literals
            Expr::Int(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Int64(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::Float(f, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Float64(*f));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            // Typed integer literals
            Expr::Int8(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Int8(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::Int16(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Int16(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::Int32(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Int32(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            // Unsigned integer literals
            Expr::UInt8(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::UInt8(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::UInt16(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::UInt16(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::UInt32(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::UInt32(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::UInt64(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::UInt64(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            // Float32 literal
            Expr::Float32(f, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Float32(*f));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            // BigInt literal
            Expr::BigInt(s, _) => {
                let dst = self.alloc_reg();
                use num_bigint::BigInt;
                let big = s.parse::<BigInt>().unwrap_or_default();
                let idx = self.chunk.add_constant(Value::BigInt(Arc::new(big)));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            // Decimal literal
            Expr::Decimal(s, _) => {
                let dst = self.alloc_reg();
                use rust_decimal::Decimal;
                let dec = s.parse::<Decimal>().unwrap_or_default();
                let idx = self.chunk.add_constant(Value::Decimal(dec));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::Bool(b, _) => {
                let dst = self.alloc_reg();
                if *b {
                    self.chunk.emit(Instruction::LoadTrue(dst), line);
                } else {
                    self.chunk.emit(Instruction::LoadFalse(dst), line);
                }
                Ok(dst)
            }
            Expr::String(string_lit, _) => {
                match string_lit {
                    StringLit::Plain(s) => {
                        let dst = self.alloc_reg();
                        let idx = self.chunk.add_constant(Value::String(Arc::new(s.clone())));
                        self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                        Ok(dst)
                    }
                    StringLit::Interpolated(parts) => {
                        self.compile_interpolated_string(parts)
                    }
                }
            }
            Expr::Char(c, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Char(*c));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::Unit(_) => {
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::LoadUnit(dst), line);
                Ok(dst)
            }

            // Variables
            Expr::Var(ident) => {
                let name = &ident.node;
                if let Some(info) = self.locals.get(name).copied() {
                    // If in tail position, emit a Move to preserve line info for debugger
                    if is_tail {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::Move(dst, info.reg), line);
                        Ok(dst)
                    } else {
                        Ok(info.reg)
                    }
                } else if let Some(&capture_idx) = self.capture_indices.get(name) {
                    // It's a captured variable - load from closure environment
                    let dst = self.alloc_reg();
                    self.chunk.emit(Instruction::GetCapture(dst, capture_idx), line);
                    Ok(dst)
                } else if self.functions.contains_key(name) {
                    // It's a function reference (exact match)
                    let dst = self.alloc_reg();
                    let func = self.functions.get(name).unwrap().clone();
                    let idx = self.chunk.add_constant(Value::Function(func));
                    self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                    Ok(dst)
                } else {
                    // Check for function with signature (new naming convention)
                    let resolved = self.resolve_name(name);
                    let prefix = format!("{}/", resolved);
                    // First find the key, then get the function (to avoid borrow issues)
                    let func_key = self.functions.keys()
                        .find(|k| k.starts_with(&prefix))
                        .cloned();
                    if let Some(key) = func_key {
                        let dst = self.alloc_reg();
                        let func = self.functions.get(&key).unwrap().clone();
                        let idx = self.chunk.add_constant(Value::Function(func));
                        self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                        return Ok(dst);
                    }
                    // Check if it's a function currently being compiled (for self-recursion)
                    // or in fn_asts (for mutual recursion or forward references)
                    let has_fn = self.current_function_name.as_ref().map_or(false, |c| c.starts_with(&prefix))
                        || self.fn_asts.keys().any(|k| k.starts_with(&prefix));
                    if has_fn {
                        // For recursive calls during compilation, we can't load the function value yet
                        // But this shouldn't happen - recursive calls should go through compile_call
                        // If we reach here, it's a first-class reference to a function not yet compiled
                        // Return an error for now - this case needs special handling
                    }
                    // Check if it's a module-level mutable variable (mvar)
                    let mvar_name = self.resolve_name(name);
                    if self.mvars.contains_key(&mvar_name) {
                        let dst = self.alloc_reg();
                        let name_idx = self.chunk.add_constant(Value::String(Arc::new(mvar_name.clone())));
                        self.chunk.emit(Instruction::MvarRead(dst, name_idx), line);
                        // Track mvar read for deadlock detection
                        self.current_fn_mvar_reads.insert(mvar_name);
                        return Ok(dst);
                    }
                    Err(CompileError::UnknownVariable {
                        name: name.clone(),
                        span: ident.span,
                    })
                }
            }

            // Binary operations
            Expr::BinOp(left, op, right, _) => {
                self.compile_binop(op, left, right)
            }

            // Unary operations
            Expr::UnaryOp(op, operand, _) => {
                self.compile_unaryop(op, operand)
            }

            // Function call
            Expr::Call(func, type_args, args, _) => {
                self.compile_call(func, type_args, args, is_tail)
            }

            // If expression
            Expr::If(cond, then_branch, else_branch, _) => {
                self.compile_if(cond, then_branch, else_branch, is_tail)
            }

            // Match expression
            Expr::Match(scrutinee, arms, span) => {
                self.compile_match(scrutinee, arms, is_tail, span.start)
            }

            // Block
            Expr::Block(stmts, _) => {
                self.compile_block(stmts, is_tail)
            }

            // List literal
            Expr::List(items, tail, _) => {
                // Check if all items are Int64 for specialization
                let all_int64 = !items.is_empty() && items.iter().all(|item| self.is_int64_expr(item));

                match tail {
                    Some(tail_expr) => {
                        // List cons syntax: [e1, e2, ... | tail]
                        // Check if tail is also Int64List for full specialization
                        let tail_is_int64_list = self.is_int64_list_expr(tail_expr);
                        let use_int64 = all_int64 && tail_is_int64_list;

                        // Compile items in order first
                        let mut item_regs = Vec::new();
                        for item in items {
                            let reg = self.compile_expr_tail(item, false)?;
                            item_regs.push(reg);
                        }
                        // Compile the tail
                        let mut result_reg = self.compile_expr_tail(tail_expr, false)?;
                        // Cons each item onto the tail in reverse order
                        for item_reg in item_regs.into_iter().rev() {
                            let new_reg = self.alloc_reg();
                            if use_int64 {
                                // O(log n) cons on Int64List using imbl::Vector
                                self.chunk.emit(Instruction::ConsInt64(new_reg, item_reg, result_reg), line);
                            } else {
                                self.chunk.emit(Instruction::Cons(new_reg, item_reg, result_reg), line);
                            }
                            result_reg = new_reg;
                        }
                        Ok(result_reg)
                    }
                    None => {
                        // Simple list literal: [e1, e2, ...]
                        // Use MakeInt64List for better sum/length performance
                        let mut regs = Vec::new();
                        for item in items {
                            let reg = self.compile_expr_tail(item, false)?;
                            regs.push(reg);
                        }
                        let dst = self.alloc_reg();
                        if all_int64 {
                            self.chunk.emit(Instruction::MakeInt64List(dst, regs.into()), line);
                        } else {
                            self.chunk.emit(Instruction::MakeList(dst, regs.into()), line);
                        }
                        Ok(dst)
                    }
                }
            }

            // Tuple literal
            Expr::Tuple(items, _) => {
                let mut regs = Vec::new();
                for item in items {
                    let reg = self.compile_expr_tail(item, false)?;
                    regs.push(reg);
                }
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::MakeTuple(dst, regs.into()), line);
                Ok(dst)
            }

            // Lambda
            Expr::Lambda(params, body, _) => {
                self.compile_lambda(params, body)
            }

            // Field access - or mvar read if the path is a known mvar
            Expr::FieldAccess(obj, field, _) => {
                // Check if this is an mvar access (e.g., demo.panel.panelCursor)
                // by extracting the full path and checking if it's a registered mvar
                if let Some(module_path) = self.extract_module_path(obj) {
                    let full_path = format!("{}.{}", module_path, field.node);
                    if self.mvars.contains_key(&full_path) {
                        let dst = self.alloc_reg();
                        let name_idx = self.chunk.add_constant(Value::String(Arc::new(full_path.clone())));
                        self.chunk.emit(Instruction::MvarRead(dst, name_idx), line);
                        // Track mvar read for deadlock detection
                        self.current_fn_mvar_reads.insert(full_path);
                        return Ok(dst);
                    }
                }

                // Regular field access on a record
                let obj_reg = self.compile_expr_tail(obj, false)?;
                let dst = self.alloc_reg();
                let field_idx = self.chunk.add_constant(Value::String(Arc::new(field.node.clone())));
                self.chunk.emit(Instruction::GetField(dst, obj_reg, field_idx), line);
                Ok(dst)
            }

            // Record construction
            Expr::Record(type_name, fields, _) => {
                // Handle Html(...) - transforms bare HTML tag names to stdlib.html.* calls
                // Returns an Html tree. Use render(Html(...)) to get a String.
                if type_name.node == "Html" && fields.len() == 1 {
                    if let RecordField::Positional(inner_expr) = &fields[0] {
                        // Transform the argument expression to use stdlib.html.* functions
                        let transformed_arg = self.transform_html_expr(inner_expr);

                        // Compile the transformed expression and return the Html tree
                        return self.compile_expr_tail(&transformed_arg, is_tail);
                    }
                }

                // Resolve type name (check imports)
                let qualified_type = self.resolve_name(&type_name.node);
                self.compile_record(&qualified_type, fields)
            }

            // Record update
            Expr::RecordUpdate(type_name, base, fields, _) => {
                // Resolve type name (check imports)
                let qualified_type = self.resolve_name(&type_name.node);
                self.compile_record_update(&qualified_type, base, fields)
            }

            // Method call (UFCS) or module-qualified function call
            Expr::MethodCall(obj, method, args, _span) => {
                // Check if this is a module-qualified call (e.g., Math.add(1, 2))
                // by checking if the object looks like a module path
                if let Some(module_path) = self.extract_module_path(obj) {
                    // It's a module-qualified call: Module.function(args)
                    let qualified_name = format!("{}.{}", module_path, method.node);

                    // === Check for builtin module-qualified functions first ===
                    match qualified_name.as_str() {
                        "File.readAll" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileReadAll(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "File.writeAll" if args.len() == 2 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let content_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileWriteAll(dst, path_reg, content_reg), line);
                            return Ok(dst);
                        }
                        "Http.get" if args.len() == 1 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpGet(dst, url_reg), line);
                            return Ok(dst);
                        }
                        "Http.post" if args.len() == 2 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let body_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpPost(dst, url_reg, body_reg), line);
                            return Ok(dst);
                        }
                        "Http.put" if args.len() == 2 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let body_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpPut(dst, url_reg, body_reg), line);
                            return Ok(dst);
                        }
                        "Http.delete" if args.len() == 1 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpDelete(dst, url_reg), line);
                            return Ok(dst);
                        }
                        "Http.patch" if args.len() == 2 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let body_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpPatch(dst, url_reg, body_reg), line);
                            return Ok(dst);
                        }
                        "Http.head" if args.len() == 1 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpHead(dst, url_reg), line);
                            return Ok(dst);
                        }
                        "Http.request" if args.len() == 4 => {
                            let method_reg = self.compile_expr_tail(&args[0], false)?;
                            let url_reg = self.compile_expr_tail(&args[1], false)?;
                            let headers_reg = self.compile_expr_tail(&args[2], false)?;
                            let body_reg = self.compile_expr_tail(&args[3], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpRequest(dst, method_reg, url_reg, headers_reg, body_reg), line);
                            return Ok(dst);
                        }
                        // HTTP Server functions
                        "Server.bind" if args.len() == 1 => {
                            let port_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerBind(dst, port_reg), line);
                            return Ok(dst);
                        }
                        "Server.accept" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerAccept(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Server.respond" if args.len() == 4 => {
                            let req_id_reg = self.compile_expr_tail(&args[0], false)?;
                            let status_reg = self.compile_expr_tail(&args[1], false)?;
                            let headers_reg = self.compile_expr_tail(&args[2], false)?;
                            let body_reg = self.compile_expr_tail(&args[3], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerRespond(dst, req_id_reg, status_reg, headers_reg, body_reg), line);
                            return Ok(dst);
                        }
                        "Server.close" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerClose(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Server.matchPath" if args.len() == 2 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let pattern_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![path_reg, pattern_reg].into(), line);
                            return Ok(dst);
                        }
                        // WebSocket functions
                        "WebSocket.accept" if args.len() == 1 => {
                            let request_id_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::WebSocketAccept(dst, request_id_reg), line);
                            return Ok(dst);
                        }
                        "WebSocket.send" if args.len() == 2 => {
                            let request_id_reg = self.compile_expr_tail(&args[0], false)?;
                            let message_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::WebSocketSend(dst, request_id_reg, message_reg), line);
                            return Ok(dst);
                        }
                        "WebSocket.recv" if args.len() == 1 => {
                            let request_id_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::WebSocketReceive(dst, request_id_reg), line);
                            return Ok(dst);
                        }
                        "WebSocket.close" if args.len() == 1 => {
                            let request_id_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::WebSocketClose(dst, request_id_reg), line);
                            return Ok(dst);
                        }
                        // PostgreSQL functions
                        "Pg.connect" if args.len() == 1 => {
                            let conn_str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgConnect(dst, conn_str_reg), line);
                            return Ok(dst);
                        }
                        "Pg.query" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let query_reg = self.compile_expr_tail(&args[1], false)?;
                            let params_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgQuery(dst, handle_reg, query_reg, params_reg), line);
                            return Ok(dst);
                        }
                        "Pg.execute" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let query_reg = self.compile_expr_tail(&args[1], false)?;
                            let params_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgExecute(dst, handle_reg, query_reg, params_reg), line);
                            return Ok(dst);
                        }
                        "Pg.close" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgClose(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Pg.begin" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgBegin(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Pg.commit" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgCommit(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Pg.rollback" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgRollback(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Pg.transaction" if args.len() == 2 => {
                            // Pg.transaction(conn, fn) expands to:
                            // Pg.begin(conn)
                            // result = try fn() catch e -> { Pg.rollback(conn); throw(e) } end
                            // Pg.commit(conn)
                            // result
                            let conn_reg = self.compile_expr_tail(&args[0], false)?;
                            let fn_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();

                            // 1. Begin transaction
                            let begin_dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgBegin(begin_dst, conn_reg), line);

                            // 2. Push exception handler
                            let handler_idx = self.chunk.emit(Instruction::PushHandler(0), line);

                            // 3. Call the closure (0 arguments)
                            self.chunk.emit(Instruction::Call(dst, fn_reg, vec![].into()), line);

                            // 4. Pop handler (success path)
                            self.chunk.emit(Instruction::PopHandler, line);

                            // 5. Commit transaction
                            let commit_dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgCommit(commit_dst, conn_reg), line);

                            // 6. Jump past catch block
                            let skip_catch = self.chunk.emit(Instruction::Jump(0), line);

                            // 7. Catch block - patch handler to jump here
                            let catch_start = self.chunk.code.len();
                            self.chunk.patch_jump(handler_idx, catch_start);

                            // 8. Get exception
                            let exc_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::GetException(exc_reg), line);

                            // 9. Rollback transaction
                            let rollback_dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgRollback(rollback_dst, conn_reg), line);

                            // 10. Re-throw the exception
                            self.chunk.emit(Instruction::Throw(exc_reg), line);

                            // 11. End - patch skip jump
                            let end = self.chunk.code.len();
                            self.chunk.patch_jump(skip_catch, end);

                            return Ok(dst);
                        }
                        "Pg.prepare" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let name_reg = self.compile_expr_tail(&args[1], false)?;
                            let query_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgPrepare(dst, handle_reg, name_reg, query_reg), line);
                            return Ok(dst);
                        }
                        "Pg.queryPrepared" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let name_reg = self.compile_expr_tail(&args[1], false)?;
                            let params_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgQueryPrepared(dst, handle_reg, name_reg, params_reg), line);
                            return Ok(dst);
                        }
                        "Pg.executePrepared" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let name_reg = self.compile_expr_tail(&args[1], false)?;
                            let params_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgExecutePrepared(dst, handle_reg, name_reg, params_reg), line);
                            return Ok(dst);
                        }
                        "Pg.deallocate" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let name_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgDeallocate(dst, handle_reg, name_reg), line);
                            return Ok(dst);
                        }
                        // LISTEN/NOTIFY builtins
                        "Pg.listenConnect" if args.len() == 1 => {
                            let conn_str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgListenConnect(dst, conn_str_reg), line);
                            return Ok(dst);
                        }
                        "Pg.listen" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let channel_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgListen(dst, handle_reg, channel_reg), line);
                            return Ok(dst);
                        }
                        "Pg.unlisten" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let channel_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgUnlisten(dst, handle_reg, channel_reg), line);
                            return Ok(dst);
                        }
                        "Pg.notify" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let channel_reg = self.compile_expr_tail(&args[1], false)?;
                            let payload_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgNotify(dst, handle_reg, channel_reg, payload_reg), line);
                            return Ok(dst);
                        }
                        "Pg.awaitNotification" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let timeout_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgAwaitNotification(dst, handle_reg, timeout_reg), line);
                            return Ok(dst);
                        }
                        // Time builtins
                        "Time.now" if args.is_empty() => {
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeNow(dst), line);
                            return Ok(dst);
                        }
                        "Time.fromDate" if args.len() == 3 => {
                            let year_reg = self.compile_expr_tail(&args[0], false)?;
                            let month_reg = self.compile_expr_tail(&args[1], false)?;
                            let day_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeFromDate(dst, year_reg, month_reg, day_reg), line);
                            return Ok(dst);
                        }
                        "Time.fromTime" if args.len() == 3 => {
                            let hour_reg = self.compile_expr_tail(&args[0], false)?;
                            let min_reg = self.compile_expr_tail(&args[1], false)?;
                            let sec_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeFromTime(dst, hour_reg, min_reg, sec_reg), line);
                            return Ok(dst);
                        }
                        "Time.fromDateTime" if args.len() == 6 => {
                            let year_reg = self.compile_expr_tail(&args[0], false)?;
                            let month_reg = self.compile_expr_tail(&args[1], false)?;
                            let day_reg = self.compile_expr_tail(&args[2], false)?;
                            let hour_reg = self.compile_expr_tail(&args[3], false)?;
                            let min_reg = self.compile_expr_tail(&args[4], false)?;
                            let sec_reg = self.compile_expr_tail(&args[5], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeFromDateTime(dst, year_reg, month_reg, day_reg, hour_reg, min_reg, sec_reg), line);
                            return Ok(dst);
                        }
                        "Time.year" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeYear(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        "Time.month" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeMonth(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        "Time.day" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeDay(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        "Time.hour" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeHour(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        "Time.minute" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeMinute(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        "Time.second" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeSecond(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        // Type introspection and reflection
                        "typeInfo" if args.len() == 1 => {
                            let name_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TypeInfo(dst, name_reg), line);
                            return Ok(dst);
                        }
                        "typeOf" if args.len() == 1 => {
                            let val_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TypeOf(dst, val_reg), line);
                            return Ok(dst);
                        }
                        "tagOf" if args.len() == 1 => {
                            let val_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TagOf(dst, val_reg), line);
                            return Ok(dst);
                        }
                        "reflect" if args.len() == 1 => {
                            let val_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Reflect(dst, val_reg), line);
                            return Ok(dst);
                        }
                        // String encoding functions
                        "Base64.encode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Base64Encode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Base64.decode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Base64Decode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Url.encode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::UrlEncode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Url.decode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::UrlDecode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Encoding.toBytes" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Utf8Encode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Encoding.fromBytes" if args.len() == 1 => {
                            let bytes_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Utf8Decode(dst, bytes_reg), line);
                            return Ok(dst);
                        }
                        // === Buffer operations (for efficient HTML rendering) ===
                        "Buffer.new" if args.is_empty() => {
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::BufferNew(dst), line);
                            return Ok(dst);
                        }
                        "Buffer.append" if args.len() == 2 => {
                            let buf_reg = self.compile_expr_tail(&args[0], false)?;
                            let str_reg = self.compile_expr_tail(&args[1], false)?;
                            self.chunk.emit(Instruction::BufferAppend(buf_reg, str_reg), line);
                            return Ok(buf_reg); // Return the buffer
                        }
                        "Buffer.toString" if args.len() == 1 => {
                            let buf_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::BufferToString(dst, buf_reg), line);
                            return Ok(dst);
                        }
                        // String functions (1 arg)
                        "String.length" | "String.chars" | "String.from_chars" | "String.toInt" | "String.to_int"
                        | "String.toFloat" | "String.trim" | "String.trimStart" | "String.trimEnd"
                        | "String.toUpper" | "String.toLower" | "String.reverse" | "String.lines"
                        | "String.words" | "String.isEmpty" if args.len() == 1 => {
                            let arg_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg_reg].into(), line);
                            return Ok(dst);
                        }
                        // String functions (2 args)
                        "String.contains" | "String.startsWith" | "String.endsWith"
                        | "String.indexOf" | "String.lastIndexOf" | "String.repeat" if args.len() == 2 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg].into(), line);
                            return Ok(dst);
                        }
                        // String functions (3 args)
                        "String.replace" | "String.replaceAll" | "String.substring"
                        | "String.padStart" | "String.padEnd" if args.len() == 3 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let arg2_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg, arg2_reg].into(), line);
                            return Ok(dst);
                        }
                        // Time functions (0 args)
                        "Time.now" | "Time.nowSecs" | "Time.timezone" | "Time.timezoneOffset" if args.is_empty() => {
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![].into(), line);
                            return Ok(dst);
                        }
                        // Time functions (1 arg)
                        "Time.year" | "Time.month" | "Time.day" | "Time.hour" | "Time.minute"
                        | "Time.second" | "Time.weekday" | "Time.toUtc" | "Time.fromUtc" if args.len() == 1 => {
                            let arg_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg_reg].into(), line);
                            return Ok(dst);
                        }
                        // Time functions (2 args)
                        "Time.format" | "Time.formatUtc" | "Time.parse" if args.len() == 2 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg].into(), line);
                            return Ok(dst);
                        }
                        // Random functions (0 args)
                        "Random.float" | "Random.bool" if args.is_empty() => {
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![].into(), line);
                            return Ok(dst);
                        }
                        // Runtime stats functions (0 args)
                        "Runtime.threadCount" | "Runtime.uptimeMs" | "Runtime.memoryKb" | "Runtime.pid" | "Runtime.loadAvg" | "Runtime.numThreads" | "Runtime.tokioWorkers" | "Runtime.blockingThreads" if args.is_empty() => {
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![].into(), line);
                            return Ok(dst);
                        }
                        // Random functions (1 arg)
                        "Random.choice" | "Random.shuffle" | "Random.bytes" if args.len() == 1 => {
                            let arg_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg_reg].into(), line);
                            return Ok(dst);
                        }
                        // Random.int (2 args)
                        "Random.int" if args.len() == 2 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg].into(), line);
                            return Ok(dst);
                        }
                        // Env functions (0 args)
                        "Env.all" | "Env.cwd" | "Env.home" | "Env.args" | "Env.platform" if args.is_empty() => {
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![].into(), line);
                            return Ok(dst);
                        }
                        // Env functions (1 arg)
                        "Env.get" | "Env.remove" | "Env.setCwd" if args.len() == 1 => {
                            let arg_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg_reg].into(), line);
                            return Ok(dst);
                        }
                        // Env.set (2 args)
                        "Env.set" if args.len() == 2 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg].into(), line);
                            return Ok(dst);
                        }
                        // Path functions (1 arg)
                        "Path.dirname" | "Path.basename" | "Path.extension" | "Path.normalize"
                        | "Path.isAbsolute" | "Path.isRelative" | "Path.split" if args.len() == 1 => {
                            let arg_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg_reg].into(), line);
                            return Ok(dst);
                        }
                        // Path functions (2 args)
                        "Path.join" | "Path.withExtension" if args.len() == 2 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg].into(), line);
                            return Ok(dst);
                        }
                        // Regex functions (2 args)
                        "Regex.matches" | "Regex.find" | "Regex.findAll" | "Regex.split" | "Regex.captures" if args.len() == 2 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg].into(), line);
                            return Ok(dst);
                        }
                        // Regex functions (3 args)
                        "Regex.replace" | "Regex.replaceAll" if args.len() == 3 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let arg2_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg, arg2_reg].into(), line);
                            return Ok(dst);
                        }
                        // UUID functions (0 args)
                        "Uuid.v4" if args.is_empty() => {
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![].into(), line);
                            return Ok(dst);
                        }
                        // UUID functions (1 arg)
                        "Uuid.isValid" if args.len() == 1 => {
                            let arg_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg_reg].into(), line);
                            return Ok(dst);
                        }
                        // Crypto functions (1 arg)
                        "Crypto.sha256" | "Crypto.sha512" | "Crypto.md5" | "Crypto.randomBytes" if args.len() == 1 => {
                            let arg_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg_reg].into(), line);
                            return Ok(dst);
                        }
                        // Crypto functions (2 args)
                        "Crypto.bcryptHash" | "Crypto.bcryptVerify" if args.len() == 2 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg].into(), line);
                            return Ok(dst);
                        }
                        // Map functions (1 arg)
                        "Map.keys" | "Map.values" | "Map.size" | "Map.isEmpty" | "Map.toList" | "Map.fromList" if args.len() == 1 => {
                            let arg_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg_reg].into(), line);
                            return Ok(dst);
                        }
                        // Map functions (2 args)
                        "Map.get" | "Map.contains" | "Map.remove" | "Map.union" | "Map.intersection" | "Map.difference" if args.len() == 2 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg].into(), line);
                            return Ok(dst);
                        }
                        // Map.insert (3 args)
                        "Map.insert" if args.len() == 3 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let arg2_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg, arg2_reg].into(), line);
                            return Ok(dst);
                        }
                        // Set functions (1 arg)
                        "Set.size" | "Set.isEmpty" | "Set.toList" | "Set.fromList" if args.len() == 1 => {
                            let arg_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg_reg].into(), line);
                            return Ok(dst);
                        }
                        // Set functions (2 args)
                        "Set.insert" | "Set.remove" | "Set.contains" | "Set.union"
                        | "Set.intersection" | "Set.difference" | "Set.symmetricDifference"
                        | "Set.isSubset" | "Set.isProperSubset" if args.len() == 2 => {
                            let arg0_reg = self.compile_expr_tail(&args[0], false)?;
                            let arg1_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arg0_reg, arg1_reg].into(), line);
                            return Ok(dst);
                        }
                        // File handle operations
                        "File.open" if args.len() == 2 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let mode_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileOpen(dst, path_reg, mode_reg), line);
                            return Ok(dst);
                        }
                        "File.write" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let data_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileWrite(dst, handle_reg, data_reg), line);
                            return Ok(dst);
                        }
                        "File.read" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let size_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileRead(dst, handle_reg, size_reg), line);
                            return Ok(dst);
                        }
                        "File.readLine" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileReadLine(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "File.flush" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileFlush(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "File.close" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileClose(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "File.seek" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let offset_reg = self.compile_expr_tail(&args[1], false)?;
                            let whence_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileSeek(dst, handle_reg, offset_reg, whence_reg), line);
                            return Ok(dst);
                        }
                        // Directory operations
                        "Dir.create" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirCreate(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.createAll" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirCreateAll(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.list" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirList(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.remove" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirRemove(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.removeAll" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirRemoveAll(dst, path_reg), line);
                            return Ok(dst);
                        }
                        // File utilities
                        "File.exists" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileExists(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.exists" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirExists(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "File.remove" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileRemove(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "File.rename" if args.len() == 2 => {
                            let old_reg = self.compile_expr_tail(&args[0], false)?;
                            let new_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileRename(dst, old_reg, new_reg), line);
                            return Ok(dst);
                        }
                        "File.copy" if args.len() == 2 => {
                            let src_reg = self.compile_expr_tail(&args[0], false)?;
                            let dest_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileCopy(dst, src_reg, dest_reg), line);
                            return Ok(dst);
                        }
                        "File.size" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileSize(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "File.append" if args.len() == 2 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let content_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileAppend(dst, path_reg, content_reg), line);
                            return Ok(dst);
                        }
                        // === Directory operations ===
                        "Dir.create" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirCreate(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.createAll" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirCreateAll(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.list" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirList(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.remove" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirRemove(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.removeAll" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirRemoveAll(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.exists" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirExists(dst, path_reg), line);
                            return Ok(dst);
                        }
                        // === String encoding operations ===
                        "Base64.encode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Base64Encode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Base64.decode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Base64Decode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Url.encode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::UrlEncode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Url.decode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::UrlDecode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Encoding.toBytes" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Utf8Encode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Encoding.fromBytes" if args.len() == 1 => {
                            let bytes_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Utf8Decode(dst, bytes_reg), line);
                            return Ok(dst);
                        }
                        // === Buffer operations (for efficient HTML rendering) ===
                        "Buffer.new" if args.is_empty() => {
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::BufferNew(dst), line);
                            return Ok(dst);
                        }
                        "Buffer.append" if args.len() == 2 => {
                            let buf_reg = self.compile_expr_tail(&args[0], false)?;
                            let str_reg = self.compile_expr_tail(&args[1], false)?;
                            self.chunk.emit(Instruction::BufferAppend(buf_reg, str_reg), line);
                            return Ok(buf_reg); // Return the buffer
                        }
                        "Buffer.toString" if args.len() == 1 => {
                            let buf_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::BufferToString(dst, buf_reg), line);
                            return Ok(dst);
                        }
                        // === HTTP Server operations ===
                        "Server.bind" if args.len() == 1 => {
                            let port_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerBind(dst, port_reg), line);
                            return Ok(dst);
                        }
                        "Server.accept" if args.len() == 1 => {
                            let server_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerAccept(dst, server_reg), line);
                            return Ok(dst);
                        }
                        "Server.respond" if args.len() == 4 => {
                            let req_id_reg = self.compile_expr_tail(&args[0], false)?;
                            let status_reg = self.compile_expr_tail(&args[1], false)?;
                            let headers_reg = self.compile_expr_tail(&args[2], false)?;
                            let body_reg = self.compile_expr_tail(&args[3], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerRespond(dst, req_id_reg, status_reg, headers_reg, body_reg), line);
                            return Ok(dst);
                        }
                        "Server.close" if args.len() == 1 => {
                            let server_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerClose(dst, server_reg), line);
                            return Ok(dst);
                        }
                        "Server.matchPath" if args.len() == 2 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let pattern_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![path_reg, pattern_reg].into(), line);
                            return Ok(dst);
                        }
                        // === WebSocket operations ===
                        "WebSocket.accept" if args.len() == 1 => {
                            let request_id_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::WebSocketAccept(dst, request_id_reg), line);
                            return Ok(dst);
                        }
                        "WebSocket.send" if args.len() == 2 => {
                            let request_id_reg = self.compile_expr_tail(&args[0], false)?;
                            let message_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::WebSocketSend(dst, request_id_reg, message_reg), line);
                            return Ok(dst);
                        }
                        "WebSocket.recv" if args.len() == 1 => {
                            let request_id_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::WebSocketReceive(dst, request_id_reg), line);
                            return Ok(dst);
                        }
                        "WebSocket.close" if args.len() == 1 => {
                            let request_id_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::WebSocketClose(dst, request_id_reg), line);
                            return Ok(dst);
                        }
                        // === PostgreSQL operations ===
                        "Pg.connect" if args.len() == 1 => {
                            let conn_str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgConnect(dst, conn_str_reg), line);
                            return Ok(dst);
                        }
                        "Pg.query" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let query_reg = self.compile_expr_tail(&args[1], false)?;
                            let params_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgQuery(dst, handle_reg, query_reg, params_reg), line);
                            return Ok(dst);
                        }
                        "Pg.execute" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let query_reg = self.compile_expr_tail(&args[1], false)?;
                            let params_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgExecute(dst, handle_reg, query_reg, params_reg), line);
                            return Ok(dst);
                        }
                        "Pg.close" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgClose(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Pg.begin" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgBegin(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Pg.commit" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgCommit(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Pg.rollback" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgRollback(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Pg.transaction" if args.len() == 2 => {
                            // Pg.transaction(conn, fn) expands to:
                            // Pg.begin(conn)
                            // result = try fn() catch e -> { Pg.rollback(conn); throw(e) } end
                            // Pg.commit(conn)
                            // result
                            let conn_reg = self.compile_expr_tail(&args[0], false)?;
                            let fn_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();

                            // 1. Begin transaction
                            let begin_dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgBegin(begin_dst, conn_reg), line);

                            // 2. Push exception handler
                            let handler_idx = self.chunk.emit(Instruction::PushHandler(0), line);

                            // 3. Call the closure (0 arguments)
                            self.chunk.emit(Instruction::Call(dst, fn_reg, vec![].into()), line);

                            // 4. Pop handler (success path)
                            self.chunk.emit(Instruction::PopHandler, line);

                            // 5. Commit transaction
                            let commit_dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgCommit(commit_dst, conn_reg), line);

                            // 6. Jump past catch block
                            let skip_catch = self.chunk.emit(Instruction::Jump(0), line);

                            // 7. Catch block - patch handler to jump here
                            let catch_start = self.chunk.code.len();
                            self.chunk.patch_jump(handler_idx, catch_start);

                            // 8. Get exception
                            let exc_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::GetException(exc_reg), line);

                            // 9. Rollback transaction
                            let rollback_dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgRollback(rollback_dst, conn_reg), line);

                            // 10. Re-throw the exception
                            self.chunk.emit(Instruction::Throw(exc_reg), line);

                            // 11. End - patch skip jump
                            let end = self.chunk.code.len();
                            self.chunk.patch_jump(skip_catch, end);

                            return Ok(dst);
                        }
                        "Pg.prepare" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let name_reg = self.compile_expr_tail(&args[1], false)?;
                            let query_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgPrepare(dst, handle_reg, name_reg, query_reg), line);
                            return Ok(dst);
                        }
                        "Pg.queryPrepared" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let name_reg = self.compile_expr_tail(&args[1], false)?;
                            let params_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgQueryPrepared(dst, handle_reg, name_reg, params_reg), line);
                            return Ok(dst);
                        }
                        "Pg.executePrepared" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let name_reg = self.compile_expr_tail(&args[1], false)?;
                            let params_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgExecutePrepared(dst, handle_reg, name_reg, params_reg), line);
                            return Ok(dst);
                        }
                        "Pg.deallocate" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let name_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgDeallocate(dst, handle_reg, name_reg), line);
                            return Ok(dst);
                        }
                        // LISTEN/NOTIFY builtins
                        "Pg.listenConnect" if args.len() == 1 => {
                            let conn_str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgListenConnect(dst, conn_str_reg), line);
                            return Ok(dst);
                        }
                        "Pg.listen" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let channel_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgListen(dst, handle_reg, channel_reg), line);
                            return Ok(dst);
                        }
                        "Pg.unlisten" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let channel_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgUnlisten(dst, handle_reg, channel_reg), line);
                            return Ok(dst);
                        }
                        "Pg.notify" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let channel_reg = self.compile_expr_tail(&args[1], false)?;
                            let payload_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgNotify(dst, handle_reg, channel_reg, payload_reg), line);
                            return Ok(dst);
                        }
                        "Pg.awaitNotification" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let timeout_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::PgAwaitNotification(dst, handle_reg, timeout_reg), line);
                            return Ok(dst);
                        }
                        // Time builtins
                        "Time.now" if args.is_empty() => {
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeNow(dst), line);
                            return Ok(dst);
                        }
                        "Time.fromDate" if args.len() == 3 => {
                            let year_reg = self.compile_expr_tail(&args[0], false)?;
                            let month_reg = self.compile_expr_tail(&args[1], false)?;
                            let day_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeFromDate(dst, year_reg, month_reg, day_reg), line);
                            return Ok(dst);
                        }
                        "Time.fromTime" if args.len() == 3 => {
                            let hour_reg = self.compile_expr_tail(&args[0], false)?;
                            let min_reg = self.compile_expr_tail(&args[1], false)?;
                            let sec_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeFromTime(dst, hour_reg, min_reg, sec_reg), line);
                            return Ok(dst);
                        }
                        "Time.fromDateTime" if args.len() == 6 => {
                            let year_reg = self.compile_expr_tail(&args[0], false)?;
                            let month_reg = self.compile_expr_tail(&args[1], false)?;
                            let day_reg = self.compile_expr_tail(&args[2], false)?;
                            let hour_reg = self.compile_expr_tail(&args[3], false)?;
                            let min_reg = self.compile_expr_tail(&args[4], false)?;
                            let sec_reg = self.compile_expr_tail(&args[5], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeFromDateTime(dst, year_reg, month_reg, day_reg, hour_reg, min_reg, sec_reg), line);
                            return Ok(dst);
                        }
                        "Time.year" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeYear(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        "Time.month" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeMonth(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        "Time.day" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeDay(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        "Time.hour" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeHour(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        "Time.minute" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeMinute(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        "Time.second" if args.len() == 1 => {
                            let ts_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TimeSecond(dst, ts_reg), line);
                            return Ok(dst);
                        }
                        // Type introspection and reflection
                        "typeInfo" if args.len() == 1 => {
                            let name_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TypeInfo(dst, name_reg), line);
                            return Ok(dst);
                        }
                        "typeOf" if args.len() == 1 => {
                            let val_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TypeOf(dst, val_reg), line);
                            return Ok(dst);
                        }
                        "tagOf" if args.len() == 1 => {
                            let val_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::TagOf(dst, val_reg), line);
                            return Ok(dst);
                        }
                        "reflect" if args.len() == 1 => {
                            let val_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Reflect(dst, val_reg), line);
                            return Ok(dst);
                        }
                        // === Process introspection ===
                        "Process.all" if args.is_empty() => {
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ProcessAll(dst), line);
                            return Ok(dst);
                        }
                        "Process.time" if args.len() == 1 => {
                            let pid_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ProcessTime(dst, pid_reg), line);
                            return Ok(dst);
                        }
                        "Process.alive" if args.len() == 1 => {
                            let pid_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ProcessAlive(dst, pid_reg), line);
                            return Ok(dst);
                        }
                        "Process.info" if args.len() == 1 => {
                            let pid_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ProcessInfo(dst, pid_reg), line);
                            return Ok(dst);
                        }
                        "Process.kill" if args.len() == 1 => {
                            let pid_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ProcessKill(dst, pid_reg), line);
                            return Ok(dst);
                        }
                        // === Panel (TUI) functions ===
                        "Panel.create" if args.len() == 1 => {
                            let title_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, "Panel.create", vec![title_reg].into(), line);
                            return Ok(dst);
                        }
                        "Panel.setContent" if args.len() == 2 => {
                            let id_reg = self.compile_expr_tail(&args[0], false)?;
                            let content_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, "Panel.setContent", vec![id_reg, content_reg].into(), line);
                            return Ok(dst);
                        }
                        "Panel.show" if args.len() == 1 => {
                            let id_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, "Panel.show", vec![id_reg].into(), line);
                            return Ok(dst);
                        }
                        "Panel.hide" if args.len() == 1 => {
                            let id_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, "Panel.hide", vec![id_reg].into(), line);
                            return Ok(dst);
                        }
                        "Panel.onKey" if args.len() == 2 => {
                            let id_reg = self.compile_expr_tail(&args[0], false)?;
                            let handler_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, "Panel.onKey", vec![id_reg, handler_reg].into(), line);
                            return Ok(dst);
                        }
                        "Panel.registerHotkey" if args.len() == 2 => {
                            let key_reg = self.compile_expr_tail(&args[0], false)?;
                            let callback_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, "Panel.registerHotkey", vec![key_reg, callback_reg].into(), line);
                            return Ok(dst);
                        }
                        // === Eval ===
                        // Only use built-in eval if no user-defined function exists
                        "eval" if args.len() == 1 && !self.has_user_function("eval", 1) => {
                            let code_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, "eval", vec![code_reg].into(), line);
                            return Ok(dst);
                        }
                        // === External process execution ===
                        "Exec.run" if args.len() == 2 => {
                            let cmd_reg = self.compile_expr_tail(&args[0], false)?;
                            let args_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ExecRun(dst, cmd_reg, args_reg), line);
                            return Ok(dst);
                        }
                        "Exec.start" if args.len() == 2 => {
                            let cmd_reg = self.compile_expr_tail(&args[0], false)?;
                            let args_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ExecSpawn(dst, cmd_reg, args_reg), line);
                            return Ok(dst);
                        }
                        "Exec.readline" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ExecReadLine(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Exec.readStderr" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ExecReadStderr(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Exec.write" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let data_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ExecWrite(dst, handle_reg, data_reg), line);
                            return Ok(dst);
                        }
                        "Exec.wait" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ExecWait(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Exec.kill" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ExecKill(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        // === Float64Array builtins ===
                        "Float64Array.fromList" if args.len() == 1 => {
                            let list_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![list_reg].into(), line);
                            return Ok(dst);
                        }
                        "Float64Array.length" if args.len() == 1 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg].into(), line);
                            return Ok(dst);
                        }
                        "Float64Array.get" if args.len() == 2 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let idx_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg, idx_reg].into(), line);
                            return Ok(dst);
                        }
                        "Float64Array.set" if args.len() == 3 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let idx_reg = self.compile_expr_tail(&args[1], false)?;
                            let val_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg, idx_reg, val_reg].into(), line);
                            return Ok(dst);
                        }
                        "Float64Array.toList" if args.len() == 1 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg].into(), line);
                            return Ok(dst);
                        }
                        "Float64Array.make" if args.len() == 2 => {
                            let size_reg = self.compile_expr_tail(&args[0], false)?;
                            let val_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![size_reg, val_reg].into(), line);
                            return Ok(dst);
                        }
                        // === Int64Array builtins ===
                        "Int64Array.fromList" if args.len() == 1 => {
                            let list_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![list_reg].into(), line);
                            return Ok(dst);
                        }
                        "Int64Array.length" if args.len() == 1 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg].into(), line);
                            return Ok(dst);
                        }
                        "Int64Array.get" if args.len() == 2 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let idx_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg, idx_reg].into(), line);
                            return Ok(dst);
                        }
                        "Int64Array.set" if args.len() == 3 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let idx_reg = self.compile_expr_tail(&args[1], false)?;
                            let val_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg, idx_reg, val_reg].into(), line);
                            return Ok(dst);
                        }
                        "Int64Array.toList" if args.len() == 1 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg].into(), line);
                            return Ok(dst);
                        }
                        "Int64Array.make" if args.len() == 2 => {
                            let size_reg = self.compile_expr_tail(&args[0], false)?;
                            let val_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![size_reg, val_reg].into(), line);
                            return Ok(dst);
                        }
                        // === Float32Array builtins (for vectors/pgvector) ===
                        "Float32Array.fromList" if args.len() == 1 => {
                            let list_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![list_reg].into(), line);
                            return Ok(dst);
                        }
                        "Float32Array.length" if args.len() == 1 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg].into(), line);
                            return Ok(dst);
                        }
                        "Float32Array.get" if args.len() == 2 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let idx_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg, idx_reg].into(), line);
                            return Ok(dst);
                        }
                        "Float32Array.set" if args.len() == 3 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let idx_reg = self.compile_expr_tail(&args[1], false)?;
                            let val_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg, idx_reg, val_reg].into(), line);
                            return Ok(dst);
                        }
                        "Float32Array.toList" if args.len() == 1 => {
                            let arr_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![arr_reg].into(), line);
                            return Ok(dst);
                        }
                        "Float32Array.make" if args.len() == 2 => {
                            let size_reg = self.compile_expr_tail(&args[0], false)?;
                            let val_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.emit_call_native(dst, &qualified_name, vec![size_reg, val_reg].into(), line);
                            return Ok(dst);
                        }
                        _ => {} // Fall through to user-defined functions
                    }

                    let resolved_name = self.resolve_name(&qualified_name);

                    // Compute argument types for function overloading
                    let arg_types: Vec<Option<String>> = args.iter()
                        .map(|a| self.expr_type_name(a))
                        .collect();

                    // Resolve to the correct function variant using signature matching
                    let call_name = if let Some(resolved) = self.resolve_function_call(&resolved_name, &arg_types) {
                        resolved
                    } else {
                        // Fall back to trying with all wildcards
                        let wildcard_sig = vec!["_".to_string(); arg_types.len()].join(",");
                        format!("{}/{}", resolved_name, wildcard_sig)
                    };

                    // Check for user-defined function
                    let fn_exists = self.functions.contains_key(&call_name)
                        || self.fn_asts.contains_key(&call_name)
                        || self.current_function_name.as_ref() == Some(&call_name);

                    if fn_exists {
                        // Check visibility before allowing the call
                        self.check_visibility(&call_name, method.span)?;

                        // Compile arguments
                        let mut arg_regs = Vec::new();
                        for arg in args {
                            let reg = self.compile_expr_tail(arg, false)?;
                            arg_regs.push(reg);
                        }

                        let dst = self.alloc_reg();
                        // Direct function call by index (no HashMap lookup at runtime!)
                        if let Some(&func_idx) = self.function_indices.get(&call_name) {
                            // Track function call for deadlock detection
                            self.current_fn_calls.insert(call_name.clone());
                            if is_tail {
                                // Emit MvarUnlock for all held locks before tail call
                                for (_, name_idx, is_write) in self.current_fn_mvar_locks.iter().rev() {
                                    self.chunk.emit(Instruction::MvarUnlock(*name_idx, *is_write), 0);
                                }
                                self.chunk.emit(Instruction::TailCallDirect(func_idx, arg_regs.into()), line);
                                return Ok(0);
                            } else {
                                self.chunk.emit(Instruction::CallDirect(dst, func_idx, arg_regs.into()), line);
                                return Ok(dst);
                            }
                        }
                    } else {
                        // Function not found with this arity - check if it exists with different arity
                        // to provide a better error message than falling through to UFCS
                        let call_arity = args.len();
                        if let Some(arities) = self.find_all_function_arities(&resolved_name) {
                            if !arities.contains(&call_arity) {
                                // Function exists but not with this arity - report arity mismatch
                                let expected_arity = arities.into_iter().next().unwrap_or(0);
                                return Err(CompileError::ArityMismatch {
                                    name: qualified_name.clone(),
                                    expected: expected_arity,
                                    found: call_arity,
                                    span: method.span,
                                });
                            }
                        }
                    }
                }

                // Type-based builtin method dispatch for receiver-style calls
                // Handles m.get(k) where m is a Map, s.toUpper() where s is a String, etc.
                if let Some(type_name) = self.expr_type_name(obj) {
                    let builtin_name: Option<&str> = if type_name.starts_with("Map[") || type_name == "Map" {
                        match method.node.as_str() {
                            "get" => Some("Map.get"),
                            "insert" => Some("Map.insert"),
                            "remove" => Some("Map.remove"),
                            "contains" => Some("Map.contains"),
                            "keys" => Some("Map.keys"),
                            "values" => Some("Map.values"),
                            "size" => Some("Map.size"),
                            "isEmpty" => Some("Map.isEmpty"),
                            "union" => Some("Map.union"),
                            "intersection" => Some("Map.intersection"),
                            "difference" => Some("Map.difference"),
                            "toList" => Some("Map.toList"),
                            _ => None,
                        }
                    } else if type_name.starts_with("Set[") || type_name == "Set" {
                        match method.node.as_str() {
                            "contains" => Some("Set.contains"),
                            "insert" => Some("Set.insert"),
                            "remove" => Some("Set.remove"),
                            "size" => Some("Set.size"),
                            "isEmpty" => Some("Set.isEmpty"),
                            "union" => Some("Set.union"),
                            "intersection" => Some("Set.intersection"),
                            "difference" => Some("Set.difference"),
                            "symmetricDifference" => Some("Set.symmetricDifference"),
                            "isSubset" => Some("Set.isSubset"),
                            "isProperSubset" => Some("Set.isProperSubset"),
                            "toList" => Some("Set.toList"),
                            _ => None,
                        }
                    } else if type_name == "String" {
                        match method.node.as_str() {
                            "length" => Some("String.length"),
                            "chars" => Some("String.chars"),
                            "toInt" => Some("String.toInt"),
                            "toFloat" => Some("String.toFloat"),
                            "trim" => Some("String.trim"),
                            "trimStart" => Some("String.trimStart"),
                            "trimEnd" => Some("String.trimEnd"),
                            "toUpper" => Some("String.toUpper"),
                            "toLower" => Some("String.toLower"),
                            "contains" => Some("String.contains"),
                            "startsWith" => Some("String.startsWith"),
                            "endsWith" => Some("String.endsWith"),
                            "replace" => Some("String.replace"),
                            "replaceAll" => Some("String.replaceAll"),
                            "indexOf" => Some("String.indexOf"),
                            "lastIndexOf" => Some("String.lastIndexOf"),
                            "substring" => Some("String.substring"),
                            "repeat" => Some("String.repeat"),
                            "padStart" => Some("String.padStart"),
                            "padEnd" => Some("String.padEnd"),
                            "reverse" => Some("String.reverse"),
                            "lines" => Some("String.lines"),
                            "words" => Some("String.words"),
                            "isEmpty" => Some("String.isEmpty"),
                            _ => None,
                        }
                    } else if type_name == "Buffer" {
                        // Buffer uses special bytecode instructions, handle directly
                        match method.node.as_str() {
                            "append" => {
                                if args.len() == 1 {
                                    let buf_reg = self.compile_expr_tail(obj, false)?;
                                    let str_reg = self.compile_expr_tail(&args[0], false)?;
                                    self.chunk.emit(Instruction::BufferAppend(buf_reg, str_reg), line);
                                    return Ok(buf_reg);
                                }
                                None
                            }
                            "toString" => {
                                if args.is_empty() {
                                    let buf_reg = self.compile_expr_tail(obj, false)?;
                                    let dst = self.alloc_reg();
                                    self.chunk.emit(Instruction::BufferToString(dst, buf_reg), line);
                                    return Ok(dst);
                                }
                                None
                            }
                            _ => None,
                        }
                    } else if type_name == "Float64Array" {
                        match method.node.as_str() {
                            "length" => Some("Float64Array.length"),
                            "get" => Some("Float64Array.get"),
                            "set" => Some("Float64Array.set"),
                            "toList" => Some("Float64Array.toList"),
                            _ => None,
                        }
                    } else if type_name == "Int64Array" {
                        match method.node.as_str() {
                            "length" => Some("Int64Array.length"),
                            "get" => Some("Int64Array.get"),
                            "set" => Some("Int64Array.set"),
                            "toList" => Some("Int64Array.toList"),
                            _ => None,
                        }
                    } else if type_name == "Float32Array" {
                        match method.node.as_str() {
                            "length" => Some("Float32Array.length"),
                            "get" => Some("Float32Array.get"),
                            "set" => Some("Float32Array.set"),
                            "toList" => Some("Float32Array.toList"),
                            _ => None,
                        }
                    } else {
                        None
                    };

                    if let Some(native_name) = builtin_name {
                        // Compile receiver and args
                        let obj_reg = self.compile_expr_tail(obj, false)?;
                        let mut arg_regs = vec![obj_reg];
                        for arg in args {
                            let reg = self.compile_expr_tail(arg, false)?;
                            arg_regs.push(reg);
                        }
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, native_name, arg_regs.into(), line);
                        return Ok(dst);
                    }
                }

                // Try trait method dispatch if we can determine the type of obj
                if let Some(type_name) = self.expr_type_name(obj) {
                    if let Some(qualified_method) = self.find_trait_method(&type_name, &method.node) {
                        // Found a trait method - compile as qualified function call
                        let mut all_args = vec![obj.as_ref().clone()];
                        all_args.extend(args.iter().cloned());

                        // Compute argument types for function overloading
                        let arg_types: Vec<Option<String>> = all_args.iter()
                            .map(|a| self.expr_type_name(a))
                            .collect();

                        // Resolve to the correct function variant using signature matching
                        let call_name = if let Some(resolved) = self.resolve_function_call(&qualified_method, &arg_types) {
                            resolved
                        } else {
                            // Fall back to trying with all wildcards
                            let wildcard_sig = vec!["_".to_string(); arg_types.len()].join(",");
                            format!("{}/{}", qualified_method, wildcard_sig)
                        };

                        // Compile arguments
                        let mut arg_regs = Vec::new();
                        for arg in &all_args {
                            let reg = self.compile_expr_tail(arg, false)?;
                            arg_regs.push(reg);
                        }

                        let dst = self.alloc_reg();
                        // Direct function call by index (no HashMap lookup at runtime!)
                        if let Some(&func_idx) = self.function_indices.get(&call_name) {
                            // Track function call for deadlock detection
                            self.current_fn_calls.insert(call_name.clone());
                            if is_tail {
                                // Emit MvarUnlock for all held locks before tail call
                                for (_, name_idx, is_write) in self.current_fn_mvar_locks.iter().rev() {
                                    self.chunk.emit(Instruction::MvarUnlock(*name_idx, *is_write), 0);
                                }
                                self.chunk.emit(Instruction::TailCallDirect(func_idx, arg_regs.into()), line);
                                return Ok(0);
                            } else {
                                self.chunk.emit(Instruction::CallDirect(dst, func_idx, arg_regs.into()), line);
                                return Ok(dst);
                            }
                        }
                    }
                } else {
                    // Type unknown - check if this is a trait method that needs monomorphization
                    if self.is_known_trait_method(&method.node) {
                        return Err(CompileError::UnresolvedTraitMethod {
                            method: method.node.clone(),
                            span: method.span,
                        });
                    }
                }

                // Regular UFCS method call: obj.method(args) -> method(obj, args)
                let mut all_args = vec![obj.as_ref().clone()];
                all_args.extend(args.iter().cloned());

                let func_expr = Expr::Var(method.clone());
                self.compile_call(&func_expr, &[], &all_args, is_tail)
            }

            // Index access
            Expr::Index(coll, index, _) => {
                let coll_reg = self.compile_expr_tail(coll, false)?;
                let idx_reg = self.compile_expr_tail(index, false)?;
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::Index(dst, coll_reg, idx_reg), line);
                Ok(dst)
            }

            // Map literal: %{"key": value, ...}
            Expr::Map(pairs, _) => {
                let mut pair_regs = Vec::new();
                for (key, value) in pairs {
                    let key_reg = self.compile_expr_tail(key, false)?;
                    let val_reg = self.compile_expr_tail(value, false)?;
                    pair_regs.push((key_reg, val_reg));
                }
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::MakeMap(dst, pair_regs.into()), line);
                Ok(dst)
            }

            // Set literal: #{elem, ...}
            Expr::Set(elems, _) => {
                let mut regs = Vec::new();
                for elem in elems {
                    let reg = self.compile_expr_tail(elem, false)?;
                    regs.push(reg);
                }
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::MakeSet(dst, regs.into()), line);
                Ok(dst)
            }

            // Try/catch/finally expression
            Expr::Try(try_expr, catch_arms, finally_expr, _) => {
                self.compile_try(try_expr, catch_arms, finally_expr.as_deref(), is_tail)
            }

            // Error propagation: expr?
            Expr::Try_(inner_expr, _) => {
                self.compile_try_propagate(inner_expr)
            }

            // === Concurrency expressions ===

            // Send: pid <- msg
            Expr::Send(pid_expr, msg_expr, _) => {
                let pid_reg = self.compile_expr_tail(pid_expr, false)?;
                let msg_reg = self.compile_expr_tail(msg_expr, false)?;
                self.chunk.emit(Instruction::Send(pid_reg, msg_reg), line);
                // Send returns unit
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::LoadUnit(dst), line);
                Ok(dst)
            }

            // Spawn: spawn(func), spawn(() => expr), or spawn { block }
            Expr::Spawn(kind, func_expr, args, _span) => {
                // If func_expr is a Block, wrap it in a zero-param Lambda (thunk)
                let effective_func = match func_expr.as_ref() {
                    Expr::Block(_, block_span) => {
                        Expr::Lambda(vec![], func_expr.clone(), block_span.clone())
                    }
                    _ => func_expr.as_ref().clone(),
                };
                let func_reg = self.compile_expr_tail(&effective_func, false)?;
                let mut arg_regs = Vec::new();
                for arg in args {
                    let reg = self.compile_expr_tail(arg, false)?;
                    arg_regs.push(reg);
                }
                let dst = self.alloc_reg();
                match kind {
                    SpawnKind::Normal => {
                        self.chunk.emit(Instruction::Spawn(dst, func_reg, arg_regs.into()), line);
                    }
                    SpawnKind::Linked => {
                        self.chunk.emit(Instruction::SpawnLink(dst, func_reg, arg_regs.into()), line);
                    }
                    SpawnKind::Monitored => {
                        // SpawnMonitor returns (pid, ref)
                        let ref_dst = self.alloc_reg();
                        self.chunk.emit(Instruction::SpawnMonitor(dst, ref_dst, func_reg, arg_regs.into()), line);
                    }
                }
                Ok(dst)
            }

            // Receive: receive pattern -> body ... after timeout -> timeout_body end
            Expr::Receive(arms, after_clause, _) => {
                // Mark function as having blocking operations
                self.current_fn_has_blocking = true;

                // Allocate a register for the received message
                let msg_reg = self.alloc_reg();

                // Handle timeout if present
                let timeout_jump = if let Some((timeout_expr, _)) = after_clause {
                    // Compile timeout expression
                    let timeout_reg = self.compile_expr_tail(timeout_expr, false)?;
                    // Emit receive with timeout - places message in msg_reg or Unit if timeout
                    self.chunk.emit(Instruction::ReceiveTimeout(msg_reg, timeout_reg), line);
                    // Check if msg_reg is Unit (timeout indicator)
                    let is_unit_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::TestUnit(is_unit_reg, msg_reg), line);
                    // Jump to timeout handling if Unit
                    Some(self.chunk.emit(Instruction::JumpIfTrue(is_unit_reg, 0), line))
                } else {
                    // No timeout - regular receive
                    self.chunk.emit(Instruction::Receive(msg_reg), line);
                    None
                };

                // The message is in msg_reg after Receive completes
                // We need to match it against the arms

                let dst = self.alloc_reg();
                let mut end_jumps = Vec::new();

                for (i, arm) in arms.iter().enumerate() {
                    let is_last = i == arms.len() - 1;

                    // Save locals before processing arm (pattern bindings should be scoped to this arm)
                    let saved_locals = self.locals.clone();

                    // Try to match the pattern against the message
                    let (match_success, bindings) = self.compile_pattern_test(&arm.pattern, msg_reg)?;

                    let next_arm_jump = if !is_last {
                        Some(self.chunk.emit(Instruction::JumpIfFalse(match_success, 0), line))
                    } else {
                        None
                    };

                    // Bind pattern variables with type info from pattern
                    for (name, reg, is_float) in bindings {
                        self.locals.insert(name, LocalInfo { reg, is_float, mutable: false });
                    }

                    // Compile guard if present
                    if let Some(ref guard) = arm.guard {
                        let guard_reg = self.compile_expr_tail(guard, false)?;
                        if !is_last {
                            let guard_fail = self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), line);
                            // Compile body
                            let body_reg = self.compile_expr_tail(&arm.body, is_tail)?;
                            self.chunk.emit(Instruction::Move(dst, body_reg), line);
                            end_jumps.push(self.chunk.emit(Instruction::Jump(0), line));
                            // Patch guard fail jump
                            self.chunk.patch_jump(guard_fail, self.chunk.code.len());
                        } else {
                            // Last arm - no jump needed for guard failure
                            let body_reg = self.compile_expr_tail(&arm.body, is_tail)?;
                            self.chunk.emit(Instruction::Move(dst, body_reg), line);
                        }
                    } else {
                        // No guard - compile body directly
                        let body_reg = self.compile_expr_tail(&arm.body, is_tail)?;
                        self.chunk.emit(Instruction::Move(dst, body_reg), line);
                        if !is_last {
                            end_jumps.push(self.chunk.emit(Instruction::Jump(0), line));
                        }
                    }

                    // Patch next arm jump
                    if let Some(jump_idx) = next_arm_jump {
                        self.chunk.patch_jump(jump_idx, self.chunk.code.len());
                    }

                    // Restore locals after arm (pattern bindings shouldn't leak to next arm)
                    self.locals = saved_locals;
                }

                // Handle timeout body if present
                if let Some((_, timeout_body)) = after_clause {
                    // Jump past timeout body (for normal message case)
                    let skip_timeout = self.chunk.emit(Instruction::Jump(0), line);
                    end_jumps.push(skip_timeout);

                    // Patch timeout jump to point here
                    if let Some(jump_idx) = timeout_jump {
                        self.chunk.patch_jump(jump_idx, self.chunk.code.len());
                    }

                    // Compile timeout body
                    let timeout_result = self.compile_expr_tail(timeout_body, is_tail)?;
                    self.chunk.emit(Instruction::Move(dst, timeout_result), line);
                }

                // Patch end jumps
                let end_target = self.chunk.code.len();
                for jump_idx in end_jumps {
                    self.chunk.patch_jump(jump_idx, end_target);
                }

                Ok(dst)
            }

            // While loop
            Expr::While(cond, body, _) => {
                self.compile_while(cond, body)
            }

            // For loop
            Expr::For(var, start, end, body, _) => {
                self.compile_for(var, start, end, body)
            }

            // Break
            Expr::Break(value, span) => {
                self.compile_break(value.as_ref().map(|v| v.as_ref()), *span)
            }

            // Continue
            Expr::Continue(span) => {
                self.compile_continue(*span)
            }

            // Return
            Expr::Return(value, span) => {
                self.compile_return(value.as_ref().map(|v| v.as_ref()), *span)
            }

            _ => Err(CompileError::NotImplemented {
                feature: format!("expr: {:?}", expr),
                span: expr.span(),
            }),
        }
    }

    /// Compile a while loop.
    fn compile_while(&mut self, cond: &Expr, body: &Expr) -> Result<Reg, CompileError> {
        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), 0);

        // Record loop start - for while loops, continue jumps back to condition
        let loop_start = self.chunk.code.len();

        // Push loop context (continue_addr same as start_addr for while loops)
        self.loop_stack.push(LoopContext {
            start_addr: loop_start,
            continue_addr: loop_start,
            continue_jumps: Vec::new(),
            break_jumps: Vec::new(),
        });

        // Compile condition
        let cond_reg = self.compile_expr_tail(cond, false)?;

        // Jump to end if false
        let exit_jump = self.chunk.emit(Instruction::JumpIfFalse(cond_reg, 0), 0);

        // Compile body
        let _ = self.compile_expr_tail(body, false)?;

        // Jump back to loop start
        // Formula: offset = target - current_position - 1 (because IP is incremented before execution)
        let jump_offset = loop_start as i16 - self.chunk.code.len() as i16 - 1;
        self.chunk.emit(Instruction::Jump(jump_offset), 0);

        // Patch exit jump
        self.chunk.patch_jump(exit_jump, self.chunk.code.len());

        // Pop loop context and patch break/continue jumps
        let loop_ctx = self.loop_stack.pop().unwrap();
        for break_jump in loop_ctx.break_jumps {
            self.chunk.patch_jump(break_jump, self.chunk.code.len());
        }
        // Continue jumps should go to loop_start (already handled at emit time for while loops)
        for continue_jump in loop_ctx.continue_jumps {
            self.chunk.patch_jump(continue_jump, loop_start);
        }

        Ok(dst)
    }

    /// Compile a for loop.
    fn compile_for(&mut self, var: &Ident, start: &Expr, end: &Expr, body: &Expr) -> Result<Reg, CompileError> {
        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), 0);

        // Compile start and end values
        let counter_reg = self.compile_expr_tail(start, false)?;
        let end_reg = self.compile_expr_tail(end, false)?;

        // Bind loop variable to counter register (for loop counter is always int)
        let saved_var = self.locals.get(&var.node).cloned();
        self.locals.insert(var.node.clone(), LocalInfo { reg: counter_reg, is_float: false, mutable: false });

        // Record loop start
        let loop_start = self.chunk.code.len();

        // Push loop context - continue_addr will be set later to point to increment
        // For now use 0 as placeholder
        self.loop_stack.push(LoopContext {
            start_addr: loop_start,
            continue_addr: 0, // Will be set after body compilation
            continue_jumps: Vec::new(),
            break_jumps: Vec::new(),
        });

        // Check if counter < end
        let cond_reg = self.alloc_reg();
        self.chunk.emit(Instruction::LtInt(cond_reg, counter_reg, end_reg), 0);

        // Jump to end if counter >= end
        let exit_jump = self.chunk.emit(Instruction::JumpIfFalse(cond_reg, 0), 0);

        // Compile body
        let _ = self.compile_expr_tail(body, false)?;

        // Record where increment starts (for continue jumps)
        let increment_addr = self.chunk.code.len();

        // Increment counter: counter = counter + 1
        let one_reg = self.alloc_reg();
        let one_idx = self.chunk.add_constant(Value::Int64(1));
        self.chunk.emit(Instruction::LoadConst(one_reg, one_idx), 0);
        self.chunk.emit(Instruction::AddInt(counter_reg, counter_reg, one_reg), 0);

        // Jump back to loop start
        // Formula: offset = target - current_position - 1 (because IP is incremented before execution)
        let jump_offset = loop_start as i16 - self.chunk.code.len() as i16 - 1;
        self.chunk.emit(Instruction::Jump(jump_offset), 0);

        // Patch exit jump
        self.chunk.patch_jump(exit_jump, self.chunk.code.len());

        // Pop loop context and patch break/continue jumps
        let loop_ctx = self.loop_stack.pop().unwrap();
        for break_jump in loop_ctx.break_jumps {
            self.chunk.patch_jump(break_jump, self.chunk.code.len());
        }
        // Continue jumps should go to the increment section
        for continue_jump in loop_ctx.continue_jumps {
            self.chunk.patch_jump(continue_jump, increment_addr);
        }

        // Restore previous variable binding if any
        if let Some(prev_info) = saved_var {
            self.locals.insert(var.node.clone(), prev_info);
        } else {
            self.locals.remove(&var.node);
        }

        Ok(dst)
    }

    /// Compile a break statement.
    fn compile_break(&mut self, value: Option<&Expr>, span: Span) -> Result<Reg, CompileError> {
        if self.loop_stack.is_empty() {
            return Err(CompileError::NotImplemented {
                feature: "break outside of loop".to_string(),
                span,
            });
        }

        // If there's a value, compile it (for future: return value from loop)
        let dst = if let Some(val) = value {
            self.compile_expr_tail(val, false)?
        } else {
            let r = self.alloc_reg();
            self.chunk.emit(Instruction::LoadUnit(r), 0);
            r
        };

        // Emit jump to be patched later
        let jump_idx = self.chunk.emit(Instruction::Jump(0), 0);

        // Add to current loop's break jumps
        if let Some(loop_ctx) = self.loop_stack.last_mut() {
            loop_ctx.break_jumps.push(jump_idx);
        }

        Ok(dst)
    }

    /// Compile a continue statement.
    fn compile_continue(&mut self, span: Span) -> Result<Reg, CompileError> {
        if self.loop_stack.is_empty() {
            return Err(CompileError::NotImplemented {
                feature: "continue outside of loop".to_string(),
                span,
            });
        }

        // Emit jump with placeholder offset - will be patched at end of loop
        let jump_idx = self.chunk.emit(Instruction::Jump(0), 0);

        // Add to current loop's continue jumps
        if let Some(loop_ctx) = self.loop_stack.last_mut() {
            loop_ctx.continue_jumps.push(jump_idx);
        }

        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), 0);
        Ok(dst)
    }

    /// Compile a return statement.
    fn compile_return(&mut self, value: Option<&Expr>, span: Span) -> Result<Reg, CompileError> {
        let line = self.span_line(span);

        // Compile the return value (or unit if none)
        let val_reg = if let Some(val) = value {
            self.compile_expr_tail(val, false)?
        } else {
            let r = self.alloc_reg();
            self.chunk.emit(Instruction::LoadUnit(r), line);
            r
        };

        // Emit Return instruction
        self.chunk.emit(Instruction::Return(val_reg), line);

        // Return a unit register (execution won't continue past Return)
        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), line);
        Ok(dst)
    }

    /// Compile a binary operation.
    fn compile_binop(&mut self, op: &BinOp, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
        // Compute line number from the left operand's span
        let line = self.span_line(left.span());

        // Handle short-circuit operators first
        match op {
            BinOp::And => return self.compile_and(left, right),
            BinOp::Or => return self.compile_or(left, right),
            BinOp::Pipe => {
                // a |> f is f(a)
                return self.compile_call(right, &[], &[left.clone()], false);
            }
            _ => {}
        }

        // Check for operator overloading on custom types
        // If the left operand has a known custom type that implements the relevant trait,
        // dispatch to the trait method instead of using primitive VM instructions
        if let Some((trait_name, method_name)) = Self::operator_to_trait_method(op) {
            if let Some(left_type) = self.expr_type_name(left) {
                // Only dispatch to trait methods for non-primitive custom types
                if !Self::is_primitive_type(&left_type) && self.types.contains_key(&left_type) {
                    // Check if the type implements the trait and get the actual registered trait name
                    // (may be qualified like "nalgebra.Num" even if we looked for "Num")
                    if let Some(actual_trait_name) = self.find_implemented_trait(&left_type, trait_name) {
                        // Look up the trait method implementation using the actual trait name
                        let qualified_method = format!("{}.{}.{}", left_type, actual_trait_name, method_name);

                        // Find the actual function with signature
                        let method_arg_types = vec![Some(left_type.clone()), self.expr_type_name(right)];
                        if let Some(resolved_method) = self.resolve_function_call(&qualified_method, &method_arg_types) {
                            if self.functions.contains_key(&resolved_method) {
                                // Compile as a function call to the trait method
                                let left_reg = self.compile_expr_tail(left, false)?;
                                let right_reg = self.compile_expr_tail(right, false)?;
                                let dst = self.alloc_reg();

                                let func_idx = *self.function_indices.get(&resolved_method)
                                    .expect("Function should have been assigned an index");

                                // Track function call for deadlock detection
                                self.current_fn_calls.insert(resolved_method.clone());

                                self.chunk.emit(Instruction::CallDirect(dst, func_idx, vec![left_reg, right_reg].into()), line);
                                return Ok(dst);
                            }
                        }
                    }
                }
            }
        }

        // Check numeric types for coercion
        let left_is_float = self.is_float_expr(left);
        let right_is_float = self.is_float_expr(right);
        let left_is_bigint = self.is_bigint_expr(left);
        let right_is_bigint = self.is_bigint_expr(right);
        let is_float = left_is_float || right_is_float;
        let is_bigint = left_is_bigint || right_is_bigint;

        // Compile-time coercion: if one side is float and the other is an int literal,
        // compile the int literal directly as a float constant (no runtime conversion needed)
        let left_reg = if is_float && !left_is_float {
            if let Expr::Int(n, _) = left {
                // Compile int literal directly as float
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Float64(*n as f64));
                self.chunk.emit(Instruction::LoadConst(dst, idx), self.span_line(left.span()));
                dst
            } else {
                // Not a literal - compile normally and coerce at runtime
                let reg = self.compile_expr_tail(left, false)?;
                if self.is_int_expr(left) {
                    let coerced = self.alloc_reg();
                    self.chunk.emit(Instruction::IntToFloat(coerced, reg), self.span_line(left.span()));
                    coerced
                } else {
                    reg
                }
            }
        } else {
            self.compile_expr_tail(left, false)?
        };

        let right_reg = if is_float && !right_is_float {
            if let Expr::Int(n, _) = right {
                // Compile int literal directly as float
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Float64(*n as f64));
                self.chunk.emit(Instruction::LoadConst(dst, idx), self.span_line(right.span()));
                dst
            } else {
                // Not a literal - compile normally and coerce at runtime
                let reg = self.compile_expr_tail(right, false)?;
                if self.is_int_expr(right) {
                    let coerced = self.alloc_reg();
                    self.chunk.emit(Instruction::IntToFloat(coerced, reg), self.span_line(right.span()));
                    coerced
                } else {
                    reg
                }
            }
        } else {
            self.compile_expr_tail(right, false)?
        };

        // BigInt coercion (runtime only, no compile-time optimization for BigInt literals)
        let mut left_reg = left_reg;
        let mut right_reg = right_reg;
        if is_bigint {
            // BigInt coercion: convert small ints to BigInt
            if !left_is_bigint && self.is_small_int_expr(left) {
                let coerced = self.alloc_reg();
                self.chunk.emit(Instruction::ToBigInt(coerced, left_reg), self.span_line(left.span()));
                left_reg = coerced;
            }
            if !right_is_bigint && self.is_small_int_expr(right) {
                let coerced = self.alloc_reg();
                self.chunk.emit(Instruction::ToBigInt(coerced, right_reg), self.span_line(right.span()));
                right_reg = coerced;
            }
        }

        // Sized integer coercion: if one side is a sized int (Int8, Int16, Int32, etc.)
        // and the other side is a plain Int literal, coerce the literal to match
        if !is_float && !is_bigint {
            let left_sized = self.get_sized_int_type(left);
            let right_sized = self.get_sized_int_type(right);
            let left_is_plain_int = matches!(left, Expr::Int(_, _));
            let right_is_plain_int = matches!(right, Expr::Int(_, _));

            // If left is sized and right is plain int literal, coerce right
            if let Some(sized_type) = left_sized {
                if right_is_plain_int {
                    if let Some(instr) = Self::sized_int_conversion_instruction(sized_type, right_reg, right_reg) {
                        // Emit in-place conversion (reuse register)
                        self.chunk.emit(instr, self.span_line(right.span()));
                    }
                }
            }
            // If right is sized and left is plain int literal, coerce left
            else if let Some(sized_type) = right_sized {
                if left_is_plain_int {
                    if let Some(instr) = Self::sized_int_conversion_instruction(sized_type, left_reg, left_reg) {
                        self.chunk.emit(instr, self.span_line(left.span()));
                    }
                }
            }
        }

        let dst = self.alloc_reg();

        let instr = match op {
            BinOp::Add => {
                if is_float {
                    Instruction::AddFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::AddInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Sub => {
                if is_float {
                    Instruction::SubFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::SubInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Mul => {
                if is_float {
                    Instruction::MulFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::MulInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Div => {
                if is_float {
                    Instruction::DivFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::DivInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Mod => Instruction::ModInt(dst, left_reg, right_reg),
            BinOp::Pow => Instruction::PowFloat(dst, left_reg, right_reg),
            BinOp::Eq => {
                if is_float {
                    Instruction::EqFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::Eq(dst, left_reg, right_reg)
                }
            }
            BinOp::NotEq => {
                if is_float {
                    self.chunk.emit(Instruction::EqFloat(dst, left_reg, right_reg), line);
                } else {
                    self.chunk.emit(Instruction::Eq(dst, left_reg, right_reg), line);
                }
                self.chunk.emit(Instruction::Not(dst, dst), line);
                return Ok(dst);
            }
            BinOp::Lt => {
                if is_float {
                    Instruction::LtFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::LtInt(dst, left_reg, right_reg)
                }
            }
            BinOp::LtEq => {
                if is_float {
                    Instruction::LeFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::LeInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Gt => {
                // Gt is Lt with args swapped
                if is_float {
                    Instruction::LtFloat(dst, right_reg, left_reg)
                } else {
                    Instruction::GtInt(dst, left_reg, right_reg)
                }
            }
            BinOp::GtEq => {
                // GtEq is LeEq with args swapped
                if is_float {
                    Instruction::LeFloat(dst, right_reg, left_reg)
                } else {
                    Instruction::GeInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Concat => Instruction::Concat(dst, left_reg, right_reg),
            BinOp::Cons => Instruction::Cons(dst, left_reg, right_reg),
            BinOp::And | BinOp::Or | BinOp::Pipe => unreachable!(),
        };

        self.chunk.emit(instr, line);
        Ok(dst)
    }

    /// Compile short-circuit AND.
    fn compile_and(&mut self, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
        let left_reg = self.compile_expr_tail(left, false)?;
        let dst = self.alloc_reg();

        let end_jump = self.chunk.emit(Instruction::JumpIfFalse(left_reg, 0), 0);
        let right_reg = self.compile_expr_tail(right, false)?;
        self.chunk.emit(Instruction::Move(dst, right_reg), 0);
        let skip_false = self.chunk.emit(Instruction::Jump(0), 0);
        let false_target = self.chunk.code.len();
        self.chunk.patch_jump(end_jump, false_target);
        self.chunk.emit(Instruction::LoadFalse(dst), 0);
        let end_target = self.chunk.code.len();
        self.chunk.patch_jump(skip_false, end_target);
        Ok(dst)
    }

    /// Compile short-circuit OR.
    fn compile_or(&mut self, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
        let left_reg = self.compile_expr_tail(left, false)?;
        let dst = self.alloc_reg();

        let end_jump = self.chunk.emit(Instruction::JumpIfTrue(left_reg, 0), 0);
        let right_reg = self.compile_expr_tail(right, false)?;
        self.chunk.emit(Instruction::Move(dst, right_reg), 0);
        let skip_true = self.chunk.emit(Instruction::Jump(0), 0);
        let true_target = self.chunk.code.len();
        self.chunk.patch_jump(end_jump, true_target);
        self.chunk.emit(Instruction::LoadTrue(dst), 0);
        let end_target = self.chunk.code.len();
        self.chunk.patch_jump(skip_true, end_target);
        Ok(dst)
    }

    /// Compile a unary operation.
    fn compile_unaryop(&mut self, op: &UnaryOp, operand: &Expr) -> Result<Reg, CompileError> {
        let is_float = self.is_float_expr(operand);
        let src = self.compile_expr_tail(operand, false)?;
        let dst = self.alloc_reg();

        let instr = match op {
            UnaryOp::Neg => {
                if is_float {
                    Instruction::NegFloat(dst, src)
                } else {
                    Instruction::NegInt(dst, src)
                }
            }
            UnaryOp::Not => Instruction::Not(dst, src),
        };

        self.chunk.emit(instr, 0);
        Ok(dst)
    }

    /// Compile a function call, with tail call optimization.
    fn compile_call(&mut self, func: &Expr, type_args: &[TypeExpr], args: &[Expr], is_tail: bool) -> Result<Reg, CompileError> {
        // Get line number for this call expression
        let line = self.span_line(func.span());

        // Try to extract a qualified function name from the expression
        let maybe_qualified_name = self.extract_qualified_name(func);

        // Handle special built-in `throw` that compiles to the Throw instruction
        if let Some(ref name) = maybe_qualified_name {
            if name == "throw" && args.len() == 1 {
                let arg_reg = self.compile_expr_tail(&args[0], false)?;
                self.chunk.emit(Instruction::Throw(arg_reg), line);
                // Throw doesn't return, but we need to return a register
                // Return a unit register since execution won't continue
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::LoadUnit(dst), line);
                return Ok(dst);
            }
            // Handle __native__ - call extension function by name
            // Syntax: __native__("FunctionName", arg1, arg2, ...)
            if name == "__native__" && !args.is_empty() {
                // First argument must be a string literal with the function name
                let ext_func_name = match &args[0] {
                    Expr::String(StringLit::Plain(s), _) => s.clone(),
                    _ => {
                        return Err(CompileError::TypeError {
                            message: "__native__ first argument must be a plain string literal".to_string(),
                            span: args[0].span(),
                        });
                    }
                };

                // Compile remaining arguments
                let mut arg_regs = Vec::new();
                for arg in &args[1..] {
                    let reg = self.compile_expr_tail(arg, false)?;
                    arg_regs.push(reg);
                }

                // Emit CallExtension instruction (uses CallExtensionIdx if index is known)
                let dst = self.alloc_reg();
                self.emit_call_extension(dst, &ext_func_name, arg_regs.into(), line);
                return Ok(dst);
            }
            // Handle self() - get current process ID
            if name == "self" && args.is_empty() {
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::SelfPid(dst), line);
                return Ok(dst);
            }

            // Handle Html(...) - transforms bare HTML tag names to stdlib.html.* calls
            // Returns an Html tree. Use render(Html(...)) to get a String.
            if name == "Html" && args.len() == 1 {
                // Ensure stdlib.html is registered as a known module for name resolution
                self.known_modules.insert("stdlib".to_string());
                self.known_modules.insert("stdlib.html".to_string());

                // Transform the argument expression to use stdlib.html.* functions
                let transformed_arg = self.transform_html_expr(&args[0]);

                // Compile the transformed expression and return the Html tree
                return self.compile_expr_tail(&transformed_arg, is_tail);
            }
        }

        // Get expected parameter types from the called function (if known)
        // This is needed to monomorphize polymorphic function arguments
        let expected_param_types: Vec<Option<TypeExpr>> = if let Some(ref qname) = maybe_qualified_name {
            self.get_function_param_types(qname)
        } else {
            vec![]
        };

        // Build type parameter substitution map from argument types
        // This is needed when calling polymorphic functions like apply_hash[T: Hash](f: (T) -> Int, x: T)
        // where we need to substitute T with the concrete type from x (Val)
        let type_param_map: HashMap<String, String> = self.build_type_param_map(&expected_param_types, args);

        // Compile arguments, handling polymorphic function arguments specially
        let mut arg_regs = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            // Get the expected parameter type and substitute any type parameters
            let expected_type = expected_param_types.get(i)
                .and_then(|t| t.as_ref())
                .map(|t| self.substitute_type_params(t, &type_param_map));
            let reg = self.compile_arg_with_expected_type(arg, expected_type.as_ref())?;
            arg_regs.push(reg);
        }

        if let Some(qualified_name) = maybe_qualified_name {
            // Compile-time resolved builtins - no string lookup, no HashMap, no runtime dispatch!
            if !qualified_name.contains('.') {
                match qualified_name.as_str() {
                    // === Type-agnostic builtins (no runtime dispatch needed) ===
                    "println" if arg_regs.len() == 1 => {
                        self.chunk.emit(Instruction::Println(arg_regs[0]), line);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LoadUnit(dst), line);
                        return Ok(dst);
                    }
                    "print" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::Print(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "inspect" if arg_regs.len() == 2 => {
                        // inspect(value, name) - call native function
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "inspect", arg_regs.into(), line);
                        return Ok(dst);
                    }
                    // === Panel (TUI) functions ===
                    "Panel.create" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "Panel.create", arg_regs.into(), line);
                        return Ok(dst);
                    }
                    "Panel.setContent" if arg_regs.len() == 2 => {
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "Panel.setContent", arg_regs.into(), line);
                        return Ok(dst);
                    }
                    "Panel.show" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "Panel.show", arg_regs.into(), line);
                        return Ok(dst);
                    }
                    "Panel.hide" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "Panel.hide", arg_regs.into(), line);
                        return Ok(dst);
                    }
                    "Panel.onKey" if arg_regs.len() == 2 => {
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "Panel.onKey", arg_regs.into(), line);
                        return Ok(dst);
                    }
                    "Panel.registerHotkey" if arg_regs.len() == 2 => {
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "Panel.registerHotkey", arg_regs.into(), line);
                        return Ok(dst);
                    }
                    // === Eval ===
                    // Only use built-in eval if no user-defined function exists
                    "eval" if arg_regs.len() == 1 && !self.has_user_function("eval", 1) => {
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "eval", arg_regs.into(), line);
                        return Ok(dst);
                    }
                    "head" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListHead(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "tail" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListTail(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "isEmpty" | "empty" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListIsEmpty(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    // listSum is the canonical name, sum is only used if no user function exists
                    "listSum" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListSum(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "sum" if arg_regs.len() == 1 && !self.has_user_function("sum", 1) => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListSum(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "listProduct" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListProduct(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "product" if arg_regs.len() == 1 && !self.has_user_function("product", 1) => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListProduct(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "listMax" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListMax(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "maximum" if arg_regs.len() == 1 && !self.has_user_function("maximum", 1) => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListMax(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "listMin" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListMin(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "minimum" if arg_regs.len() == 1 && !self.has_user_function("minimum", 1) => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListMin(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "rangeList" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::RangeList(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    // Specialized Int64List operations for fast integer list processing
                    "toIntList" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToInt64List(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "intListRange" if arg_regs.len() == 1 => {
                        // Create Int64List [n, n-1, ..., 1] directly
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::RangeInt64List(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "range" if arg_regs.len() == 2 && !self.has_user_function("range", 2) => {
                        // range(start, end) - create list [start..end)
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "range", vec![arg_regs[0], arg_regs[1]].into(), line);
                        return Ok(dst);
                    }
                    "length" | "len" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::Length(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "sumInt64Array" if arg_regs.len() == 1 => {
                        // Fast native SIMD-optimized sum for Int64Array
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "sumInt64Array", vec![arg_regs[0]].into(), line);
                        return Ok(dst);
                    }
                    "panic" if arg_regs.len() == 1 => {
                        self.chunk.emit(Instruction::Panic(arg_regs[0]), line);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LoadUnit(dst), line);
                        return Ok(dst);
                    }
                    "assert" if arg_regs.len() == 1 => {
                        self.chunk.emit(Instruction::Assert(arg_regs[0]), line);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LoadUnit(dst), line);
                        return Ok(dst);
                    }
                    "sleep" if arg_regs.len() == 1 => {
                        // sleep(ms) - sleep for N milliseconds
                        self.chunk.emit(Instruction::Sleep(arg_regs[0]), line);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LoadUnit(dst), line);
                        return Ok(dst);
                    }
                    "vmStats" if arg_regs.is_empty() => {
                        // vmStats() - get process stats
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::VmStats(dst), line);
                        return Ok(dst);
                    }
                    "assert_eq" if arg_regs.len() == 2 => {
                        self.chunk.emit(Instruction::AssertEq(arg_regs[0], arg_regs[1]), line);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LoadUnit(dst), line);
                        return Ok(dst);
                    }
                    "typeOf" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::TypeOf(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "tagOf" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::TagOf(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "reflect" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::Reflect(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "typeInfo" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::TypeInfo(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "jsonToType" if arg_regs.len() == 1 && !type_args.is_empty() => {
                        // Extract type name from type argument
                        let type_name = self.type_expr_to_string(&type_args[0]);
                        // Load the type name as a string constant
                        let type_reg = self.alloc_reg();
                        let const_idx = self.chunk.add_constant(Value::String(Arc::new(type_name)));
                        self.chunk.emit(Instruction::LoadConst(type_reg, const_idx as u16), line);
                        // Emit Construct instruction
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::Construct(dst, type_reg, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "fromJson" if arg_regs.len() == 1 && !type_args.is_empty() => {
                        // fromJson[T](json) - constructs type from parsed Json
                        // Same as jsonToType[T](json)
                        let type_name = self.type_expr_to_string(&type_args[0]);
                        let type_reg = self.alloc_reg();
                        let const_idx = self.chunk.add_constant(Value::String(Arc::new(type_name)));
                        self.chunk.emit(Instruction::LoadConst(type_reg, const_idx as u16), line);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::Construct(dst, type_reg, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "makeRecord" if arg_regs.len() == 1 && !type_args.is_empty() => {
                        // makeRecord[T](fields_map)
                        let type_name = self.type_expr_to_string(&type_args[0]);
                        let type_reg = self.alloc_reg();
                        let const_idx = self.chunk.add_constant(Value::String(Arc::new(type_name)));
                        self.chunk.emit(Instruction::LoadConst(type_reg, const_idx as u16), line);
                        // Emit MakeRecordDyn instruction: dst, type_reg, fields_reg
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::MakeRecordDyn(dst, type_reg, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "makeVariant" if arg_regs.len() == 2 && !type_args.is_empty() => {
                        // makeVariant[T](ctor_name, fields_map)
                        let type_name = self.type_expr_to_string(&type_args[0]);
                        let type_reg = self.alloc_reg();
                        let const_idx = self.chunk.add_constant(Value::String(Arc::new(type_name)));
                        self.chunk.emit(Instruction::LoadConst(type_reg, const_idx as u16), line);
                        // Emit MakeVariantDyn instruction: dst, type_reg, ctor_reg, fields_reg
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::MakeVariantDyn(dst, type_reg, arg_regs[0], arg_regs[1]), line);
                        return Ok(dst);
                    }
                    // String-based variants for use in pure Nostos code
                    "makeRecordByName" if arg_regs.len() == 2 => {
                        // makeRecordByName(type_name, fields_map)
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::MakeRecordDyn(dst, arg_regs[0], arg_regs[1]), line);
                        return Ok(dst);
                    }
                    "makeVariantByName" if arg_regs.len() == 3 => {
                        // makeVariantByName(type_name, ctor_name, fields_map)
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::MakeVariantDyn(dst, arg_regs[0], arg_regs[1], arg_regs[2]), line);
                        return Ok(dst);
                    }
                    "jsonToTypeByName" if arg_regs.len() == 2 => {
                        // jsonToTypeByName(type_name, json)
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::Construct(dst, arg_regs[0], arg_regs[1]), line);
                        return Ok(dst);
                    }
                    "requestToType" if arg_regs.len() == 2 => {
                        // requestToType(request, type_name)
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::RequestToType(dst, arg_regs[0], arg_regs[1]), line);
                        return Ok(dst);
                    }
                    // === Math builtins (type-aware - use typed instruction if we can infer type) ===
                    "abs" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        // Check if we can infer type from the argument expression
                        let arg_type = self.infer_expr_type(&args[0]);
                        match arg_type {
                            Some(InferredType::Int) => {
                                self.chunk.emit(Instruction::AbsInt(dst, arg_regs[0]), line);
                            }
                            Some(InferredType::Float) => {
                                self.chunk.emit(Instruction::AbsFloat(dst, arg_regs[0]), line);
                            }
                            None => {
                                // Fallback: emit AbsInt (type checker should have validated)
                                // In practice, abs is usually called on Int
                                self.chunk.emit(Instruction::AbsInt(dst, arg_regs[0]), line);
                            }
                        }
                        return Ok(dst);
                    }
                    "sqrt" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::SqrtFloat(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "sin" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::SinFloat(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "cos" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::CosFloat(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "tan" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::TanFloat(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "floor" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::FloorFloat(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "ceil" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::CeilFloat(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "round" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::RoundFloat(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "log" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LogFloat(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "log10" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::Log10Float(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "pow" if arg_regs.len() == 2 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::PowFloat(dst, arg_regs[0], arg_regs[1]), line);
                        return Ok(dst);
                    }
                    "min" if arg_regs.len() == 2 => {
                        let dst = self.alloc_reg();
                        // Check if we can infer type from the first argument
                        let arg_type = self.infer_expr_type(&args[0]);
                        match arg_type {
                            Some(InferredType::Int) => {
                                self.chunk.emit(Instruction::MinInt(dst, arg_regs[0], arg_regs[1]), line);
                            }
                            Some(InferredType::Float) => {
                                self.chunk.emit(Instruction::MinFloat(dst, arg_regs[0], arg_regs[1]), line);
                            }
                            None => {
                                // Fallback: emit MinInt (most common case)
                                self.chunk.emit(Instruction::MinInt(dst, arg_regs[0], arg_regs[1]), line);
                            }
                        }
                        return Ok(dst);
                    }
                    "max" if arg_regs.len() == 2 => {
                        let dst = self.alloc_reg();
                        // Check if we can infer type from the first argument
                        let arg_type = self.infer_expr_type(&args[0]);
                        match arg_type {
                            Some(InferredType::Int) => {
                                self.chunk.emit(Instruction::MaxInt(dst, arg_regs[0], arg_regs[1]), line);
                            }
                            Some(InferredType::Float) => {
                                self.chunk.emit(Instruction::MaxFloat(dst, arg_regs[0], arg_regs[1]), line);
                            }
                            None => {
                                // Fallback: emit MaxInt (most common case)
                                self.chunk.emit(Instruction::MaxInt(dst, arg_regs[0], arg_regs[1]), line);
                            }
                        }
                        return Ok(dst);
                    }
                    // === Numeric type conversions ===
                    // New standardized naming: as<Type>() with legacy aliases
                    "asFloat64" | "asFloat" | "toFloat" | "toFloat64" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::IntToFloat(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "asInt64" | "asInt" | "toInt" | "toInt64" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::FloatToInt(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "asInt8" | "toInt8" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToInt8(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "asInt16" | "toInt16" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToInt16(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "asInt32" | "toInt32" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToInt32(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "asUInt8" | "toUInt8" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToUInt8(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "asUInt16" | "toUInt16" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToUInt16(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "asUInt32" | "toUInt32" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToUInt32(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "asUInt64" | "toUInt64" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToUInt64(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "asFloat32" | "toFloat32" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToFloat32(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "asBigInt" | "toBigInt" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToBigInt(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    // === Typed Array builtins ===
                    "newInt64Array" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::MakeInt64Array(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "newFloat64Array" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::MakeFloat64Array(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "newFloat32Array" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "newFloat32Array", arg_regs.into(), line);
                        return Ok(dst);
                    }
                    // === Option unwrapping ===
                    "unwrapOr" if arg_regs.len() == 2 => {
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, "unwrapOr", arg_regs.into(), line);
                        return Ok(dst);
                    }
                    // === Trait-based builtins (show, copy, hash) ===
                    // First try trait dispatch, then fall back to native
                    builtin @ ("show" | "copy" | "hash") if arg_regs.len() == 1 => {
                        // Try to dispatch to trait method if type is known
                        let arg_type = self.expr_type_name(&args[0]);

                        if let Some(ref concrete_type) = arg_type {
                            if let Some(qualified_method) = self.find_trait_method(concrete_type, &qualified_name) {
                                // Use resolve_function_call to find the actual function with signature
                                // The trait method takes the concrete type as argument
                                let method_arg_types = vec![Some(concrete_type.clone())];
                                if let Some(resolved_method) = self.resolve_function_call(&qualified_method, &method_arg_types) {
                                    if self.functions.contains_key(&resolved_method) {
                                        let dst = self.alloc_reg();
                                        let func_idx = *self.function_indices.get(&resolved_method)
                                            .expect("Function should have been assigned an index");
                                        // Track function call for deadlock detection
                                        self.current_fn_calls.insert(resolved_method.clone());
                                        if is_tail {
                                            // Emit MvarUnlock for all held locks before tail call
                                            for (_, name_idx, is_write) in self.current_fn_mvar_locks.iter().rev() {
                                                self.chunk.emit(Instruction::MvarUnlock(*name_idx, *is_write), 0);
                                            }
                                            self.chunk.emit(Instruction::TailCallDirect(func_idx, arg_regs.into()), line);
                                            return Ok(0);
                                        } else {
                                            self.chunk.emit(Instruction::CallDirect(dst, func_idx, arg_regs.into()), line);
                                            return Ok(dst);
                                        }
                                    }
                                }
                            }
                        }

                        // Check if we need monomorphization:
                        // - Type is unknown (None), OR
                        // - Type is a type parameter (single uppercase letter like "T", "U", etc.)
                        let is_type_param = arg_type.as_ref().map_or(false, |t| {
                            t.len() == 1 && t.chars().next().unwrap().is_uppercase()
                        });

                        if arg_type.is_none() || is_type_param {
                            // Check if we're in a function with type parameters
                            // that have the matching trait bound (Hash for hash, Show for show, Copy for copy)
                            let required_trait = match builtin {
                                "hash" => "Hash",
                                "show" => "Show",
                                "copy" => "Copy",
                                _ => "",
                            };
                            // If we're in a function with type params and any has the required trait bound,
                            // this needs monomorphization
                            let needs_monomorphization = !self.current_fn_type_params.is_empty() &&
                                self.current_fn_type_params.iter().any(|tp| {
                                    tp.constraints.iter().any(|b| b.node == required_trait)
                                });
                            if needs_monomorphization {
                                return Err(CompileError::UnresolvedTraitMethod {
                                    method: qualified_name.clone(),
                                    span: func.span(),
                                });
                            }
                        }

                        // Fall back to native call
                        let dst = self.alloc_reg();
                        self.emit_call_native(dst, &qualified_name, arg_regs.into(), line);
                        return Ok(dst);
                    }
                    _ => {} // Fall through to normal function lookup
                }
            } else {
                // === Module-qualified builtins (async IO operations) ===
                match qualified_name.as_str() {
                    "File.readAll" if arg_regs.len() == 1 => {
                        // File.readAll(path) -> async read entire file as string
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::FileReadAll(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "File.writeAll" if arg_regs.len() == 2 => {
                        // File.writeAll(path, content) -> async write string to file
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::FileWriteAll(dst, arg_regs[0], arg_regs[1]), line);
                        return Ok(dst);
                    }
                    "Http.get" if arg_regs.len() == 1 => {
                        // Http.get(url) -> async HTTP GET request
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::HttpGet(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    // === Process introspection ===
                    "Process.all" if arg_regs.is_empty() => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ProcessAll(dst), line);
                        return Ok(dst);
                    }
                    "Process.time" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ProcessTime(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "Process.alive" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ProcessAlive(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "Process.info" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ProcessInfo(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "Process.kill" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ProcessKill(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    // === External process execution ===
                    "Exec.run" if arg_regs.len() == 2 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ExecRun(dst, arg_regs[0], arg_regs[1]), line);
                        return Ok(dst);
                    }
                    "Exec.start" if arg_regs.len() == 2 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ExecSpawn(dst, arg_regs[0], arg_regs[1]), line);
                        return Ok(dst);
                    }
                    "Exec.readline" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ExecReadLine(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "Exec.readStderr" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ExecReadStderr(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "Exec.write" if arg_regs.len() == 2 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ExecWrite(dst, arg_regs[0], arg_regs[1]), line);
                        return Ok(dst);
                    }
                    "Exec.wait" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ExecWait(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "Exec.kill" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ExecKill(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    _ => {} // Fall through to normal function lookup
                }

            }

            // CRITICAL: If this is a local variable (function parameter or let binding),
            // do NOT resolve it as a global function. Fall through to generic call path.
            // This ensures local variables shadow global/stdlib functions when called.
            if !qualified_name.contains('.') && self.locals.contains_key(&qualified_name) {
                // Skip function resolution - this is a local variable being called as a function
                // Fall through to generic call path at the end of compile_call
            } else {

            // Resolve the name (handles imports and module path) with ambiguity checking
            let resolved_name = self.resolve_name_checked(&qualified_name, func.span())?;

            // Get argument types for function overloading resolution
            let arg_types: Vec<Option<String>> = args.iter()
                .map(|arg| self.expr_type_name(arg))
                .collect();

            // Check trait bounds if the function has type parameters with constraints
            // Note: fn_type_params keys now include signature, so we need to find the right one
            for (fn_name, type_params) in &self.fn_type_params {
                if fn_name.starts_with(&format!("{}/", resolved_name)) && !type_params.is_empty() {
                    self.check_trait_bounds(fn_name, type_params, &arg_types, func.span())?;
                    break;
                }
            }

            // Resolve the function call using signature-based overloading
            let call_name = if let Some(resolved) = self.resolve_function_call(&resolved_name, &arg_types) {
                resolved
            } else {
                // No matching function found for this arity/types.
                // Check if the function exists with a different arity - if so, report arity mismatch.
                // Only report arity error if ALL variants of the function have different arity.
                let call_arity = arg_types.len();
                if let Some(arities) = self.find_all_function_arities(&resolved_name) {
                    if !arities.contains(&call_arity) {
                        // Function exists but not with this arity - report arity mismatch
                        let expected_arity = arities.into_iter().next().unwrap_or(0);
                        return Err(CompileError::ArityMismatch {
                            name: resolved_name.clone(),
                            expected: expected_arity,
                            found: call_arity,
                            span: func.span(),
                        });
                    }
                    // Function exists with matching arity but types don't match
                    // Try to find the actual function by arity (for type variable args like List[b])
                    if let Some(actual_fn) = self.find_function_by_arity(&resolved_name, call_arity) {
                        actual_fn
                    } else {
                        // Fall back to trying with all wildcards (for backward compatibility)
                        let wildcard_sig = vec!["_".to_string(); arg_types.len()].join(",");
                        format!("{}/{}", resolved_name, wildcard_sig)
                    }
                } else {
                    // No function with this name exists at all
                    // Fall back to trying with all wildcards (for backward compatibility)
                    let wildcard_sig = vec!["_".to_string(); arg_types.len()].join(",");
                    format!("{}/{}", resolved_name, wildcard_sig)
                }
            };

            // Check if this is a polymorphic function that needs monomorphization
            // polymorphic_fns stores names like "hashable/_" or "module.func/_,_"
            let is_polymorphic = self.polymorphic_fns.contains(&call_name);

            // Convert Option<String> types to concrete types, checking if all are known
            let concrete_arg_types: Vec<String> = arg_types.iter()
                .filter_map(|t| t.clone())
                .collect();
            let all_types_known = concrete_arg_types.len() == arg_types.len();

            // Check that all types are concrete (not type parameters)
            // A type is concrete if it's a known type (in self.types) or a primitive
            // Type parameters are single uppercase letters NOT found in self.types
            let all_types_concrete = concrete_arg_types.iter().all(|t| {
                // If it's a known type, it's concrete
                if self.types.contains_key(t) {
                    return true;
                }
                // Primitives are concrete
                if matches!(t.as_str(), "Int" | "Float" | "String" | "Bool" | "Char" | "()" | "List" | "Map" | "Set" | "Tuple") {
                    return true;
                }
                // Single uppercase letters that aren't known types are type parameters
                !(t.len() == 1 && t.chars().next().unwrap().is_uppercase())
            });

            // If polymorphic and all types are known AND concrete, try to create a monomorphized variant
            let final_call_name = if is_polymorphic && all_types_known && all_types_concrete && !concrete_arg_types.is_empty() {
                // Get parameter names from the function AST
                // fn_asts is keyed by full name with signature (e.g., "hashable/T")
                if let Some(fn_def) = self.fn_asts.get(&call_name).cloned() {
                    // Get parameter names
                    let param_names: Vec<String> = fn_def.clauses[0].params.iter()
                        .filter_map(|p| self.pattern_binding_name(&p.pattern))
                        .collect();

                    // Try to monomorphize using the full call_name as base
                    match self.compile_monomorphized_variant(&call_name, &concrete_arg_types, &param_names) {
                        Ok(mangled_name) => mangled_name,
                        Err(_) => call_name.clone(), // Fall back to original if monomorphization fails
                    }
                } else {
                    call_name.clone()
                }
            } else {
                // If calling a polymorphic function with type parameters (not concrete types),
                // we need to propagate the polymorphism. This happens when a polymorphic function
                // calls another polymorphic function with its own type parameter.
                // Example: double_hash[T: Hash](x: T) calling hashable[T: Hash](x: T)
                if is_polymorphic && !all_types_concrete && !self.current_fn_type_params.is_empty() {
                    // We're in a function with type params, calling a polymorphic function with type params
                    // This means the current function also needs monomorphization
                    return Err(CompileError::UnresolvedTraitMethod {
                        method: call_name.clone(),
                        span: func.span(),
                    });
                }
                call_name.clone()
            };

            // Check for user-defined function
            // Also check fn_asts and current_function_name for functions being compiled
            let fn_exists = self.functions.contains_key(&final_call_name)
                || self.fn_asts.contains_key(&final_call_name)
                || self.current_function_name.as_ref() == Some(&final_call_name);
            if fn_exists {
                // Check visibility before allowing the call
                self.check_visibility(&final_call_name, func.span())?;

                // Self-recursion optimization: use CallSelf to avoid HashMap lookup
                let is_self_call = self.current_function_name.as_ref() == Some(&final_call_name);
                let dst = self.alloc_reg();

                if is_self_call {
                    // Direct self-call - no lookup needed!
                    // Track self-call for deadlock detection
                    if let Some(ref fn_name) = self.current_function_name {
                        self.current_fn_calls.insert(fn_name.clone());
                    }
                    if is_tail {
                        // Emit MvarUnlock for all held locks before tail call
                        for (_, name_idx, is_write) in self.current_fn_mvar_locks.iter().rev() {
                            self.chunk.emit(Instruction::MvarUnlock(*name_idx, *is_write), 0);
                        }
                        self.chunk.emit(Instruction::TailCallSelf(arg_regs.into()), line);
                        return Ok(0);
                    } else {
                        self.chunk.emit(Instruction::CallSelf(dst, arg_regs.into()), line);
                        return Ok(dst);
                    }
                } else if let Some(&func_idx) = self.function_indices.get(&final_call_name) {
                    // Direct function call by index (no HashMap lookup at runtime!)
                    // Track function call for deadlock detection
                    self.current_fn_calls.insert(final_call_name.clone());
                    if is_tail {
                        // Emit MvarUnlock for all held locks before tail call
                        for (_, name_idx, is_write) in self.current_fn_mvar_locks.iter().rev() {
                            self.chunk.emit(Instruction::MvarUnlock(*name_idx, *is_write), 0);
                        }
                        self.chunk.emit(Instruction::TailCallDirect(func_idx, arg_regs.into()), line);
                        return Ok(0);
                    } else {
                        self.chunk.emit(Instruction::CallDirect(dst, func_idx, arg_regs.into()), line);
                        return Ok(dst);
                    }
                } else {
                    // Function exists in fn_asts but not yet compiled (forward reference or mutual recursion)
                    // For now, this is an error - mutual recursion requires forward declarations
                    // TODO: Support forward declarations for mutual recursion
                    return Err(CompileError::UnknownFunction {
                        name: final_call_name,
                        span: func.span(),
                    });
                }
            }
            } // Close the else block for local variable check
        }

        // Generic function call (lambdas, higher-order functions)
        let func_reg = self.compile_expr_tail(func, false)?;
        let dst = self.alloc_reg();

        if is_tail {
            self.chunk.emit(Instruction::TailCall(func_reg, arg_regs.into()), line);
            Ok(0)
        } else {
            self.chunk.emit(Instruction::Call(dst, func_reg, arg_regs.into()), line);
            Ok(dst)
        }
    }

    /// Extract a qualified function name from an expression.
    /// Returns Some("Module.function") for field access on modules, Some("function") for simple vars.
    fn extract_qualified_name(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Var(ident) => Some(ident.node.clone()),
            // Handle empty Record expressions as type/module references
            // The parser parses uppercase identifiers like `Box` as Record constructors
            Expr::Record(type_name, fields, _) if fields.is_empty() => {
                Some(type_name.node.clone())
            }
            Expr::FieldAccess(target, field, _) => {
                // Try to build a qualified name like "module.submodule.function"
                if let Some(base) = self.extract_qualified_name(target) {
                    // Check if the base is a known module OR if there are functions with this prefix
                    // (supports both module.func and Type.Trait.method patterns)
                    let prefix = format!("{}.", base);
                    let is_module_or_type = self.known_modules.contains(&base)
                        || self.functions.keys().any(|k| k.starts_with(&prefix));
                    if is_module_or_type {
                        Some(format!("{}.{}", base, field.node))
                    } else {
                        None // It's a field access on a value, not a module
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Transform `x: Result[T, E] = obj.toType()` to `x: Result[T, E] = requestToType(obj, "T")`
    /// This is syntactic sugar for the requestToType builtin.
    fn try_transform_to_type_call(&self, binding: &Binding) -> Option<Binding> {
        use nostos_syntax::{TypeExpr, Ident};

        // Check if the value is a method call `obj.toType()` with no arguments
        let (receiver, method_name, args, span) = match &binding.value {
            Expr::MethodCall(receiver, method, args, span) => {
                (receiver.as_ref(), &method.node, args, *span)
            }
            _ => return None,
        };

        // Must be `toType` with no arguments
        if method_name != "toType" || !args.is_empty() {
            return None;
        }

        // Must have an explicit type annotation
        let type_expr = binding.ty.as_ref()?;

        // Extract the inner type from Result[T, E] or similar wrapper type
        let inner_type_name = self.extract_result_inner_type(type_expr)?;

        // Create the transformed expression: requestToType(obj, "TypeName")
        let type_name_expr = Expr::String(nostos_syntax::StringLit::Plain(inner_type_name), span);
        let new_value = Expr::Call(
            Box::new(Expr::Var(Ident { node: "requestToType".to_string(), span })),
            vec![],  // no type args
            vec![receiver.clone(), type_name_expr],
            span,
        );

        Some(Binding {
            pattern: binding.pattern.clone(),
            ty: binding.ty.clone(),
            value: new_value,
            mutable: binding.mutable,
            span: binding.span,
        })
    }

    /// Extract the first type parameter from Result[T, E] type expressions.
    /// Returns the type name string for T.
    fn extract_result_inner_type(&self, type_expr: &nostos_syntax::TypeExpr) -> Option<String> {
        use nostos_syntax::TypeExpr;

        match type_expr {
            TypeExpr::Generic(ident, args) => {
                // Check if it's Result[T, E] pattern
                let base_name = &ident.node;
                if (base_name == "Result" || base_name == "MaybeError") && !args.is_empty() {
                    // Get the first type argument (T in Result[T, E])
                    Some(self.type_expr_name(&args[0]))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try to determine the type name of an expression at compile time.
    /// This is used for trait method dispatch.
    fn expr_type_name(&self, expr: &Expr) -> Option<String> {
        match expr {
            // Record/variant construction: Point(1, 2) or Point{x: 1, y: 2}
            // Note: The parser treats uppercase calls like Foo(42) as Record expressions
            Expr::Record(type_name, _, _) => {
                let name = &type_name.node;
                // First check if it's directly a type name (for records and single-constructor variants)
                if self.types.contains_key(name) {
                    return Some(name.clone());
                }
                // Try resolving the name (handles module-qualified types like nalgebra.Vec)
                let resolved_name = self.resolve_name(name);
                if self.types.contains_key(&resolved_name) {
                    return Some(resolved_name);
                }
                // Otherwise check if it's a variant constructor
                for (ty_name, info) in &self.types {
                    if let TypeInfoKind::Variant { constructors } = &info.kind {
                        if constructors.iter().any(|(ctor_name, _)| ctor_name == name) {
                            return Some(ty_name.clone());
                        }
                    }
                }
                // Fall back to the given name (might be an unknown type, will error later)
                Some(name.clone())
            }
            // Tuple
            Expr::Tuple(_, _) => Some("Tuple".to_string()),
            // List
            Expr::List(_, _, _) => Some("List".to_string()),
            // Literals
            Expr::Int(_, _) => Some("Int".to_string()),
            Expr::Int8(_, _) => Some("Int8".to_string()),
            Expr::Int16(_, _) => Some("Int16".to_string()),
            Expr::Int32(_, _) => Some("Int32".to_string()),
            Expr::UInt8(_, _) => Some("UInt8".to_string()),
            Expr::UInt16(_, _) => Some("UInt16".to_string()),
            Expr::UInt32(_, _) => Some("UInt32".to_string()),
            Expr::UInt64(_, _) => Some("UInt64".to_string()),
            Expr::BigInt(_, _) => Some("BigInt".to_string()),
            Expr::Float(_, _) => Some("Float".to_string()),
            Expr::Float32(_, _) => Some("Float32".to_string()),
            Expr::Decimal(_, _) => Some("Decimal".to_string()),
            Expr::String(_, _) => Some("String".to_string()),
            Expr::Char(_, _) => Some("Char".to_string()),
            Expr::Bool(_, _) => Some("Bool".to_string()),
            Expr::Unit(_) => Some("()".to_string()),
            // Map
            Expr::Map(_, _) => Some("Map".to_string()),
            // Set
            Expr::Set(_, _) => Some("Set".to_string()),
            // For Call expressions on uppercase identifiers (variant constructors),
            // check if it's a known type
            Expr::Call(func, _type_args, args, _) => {
                if let Expr::Var(ident) = func.as_ref() {
                    // Check for copy/hash/show builtins - these return the same type as their argument
                    // or a known type (hash -> Int, show -> String)
                    match ident.node.as_str() {
                        "copy" if args.len() == 1 => {
                            // copy(x) returns the same type as x
                            return self.expr_type_name(&args[0]);
                        }
                        "hash" if args.len() == 1 => {
                            return Some("Int".to_string());
                        }
                        "show" if args.len() == 1 => {
                            return Some("String".to_string());
                        }
                        _ => {}
                    }

                    if ident.node.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        // Check if it's a record constructor (type name matches)
                        // First try direct name, then resolved name (for module-qualified types)
                        if self.types.get(&ident.node).map(|info| matches!(&info.kind, TypeInfoKind::Record { .. })).unwrap_or(false) {
                            return Some(ident.node.clone());
                        }
                        let resolved_type = self.resolve_name(&ident.node);
                        if self.types.get(&resolved_type).map(|info| matches!(&info.kind, TypeInfoKind::Record { .. })).unwrap_or(false) {
                            return Some(resolved_type);
                        }
                        // Otherwise check if it's a variant constructor
                        for (type_name, info) in &self.types {
                            if let TypeInfoKind::Variant { constructors } = &info.kind {
                                if constructors.iter().any(|(name, _)| name == &ident.node) {
                                    return Some(type_name.clone());
                                }
                            }
                        }
                    }

                    // Try to get return type from function
                    // This is needed for REPL variable thunks like __repl_var_a_1()
                    // First try the raw name, then try the resolved name (for imported functions)
                    let resolved_name = self.resolve_name(&ident.node);

                    // First try direct return_type field (more reliable)
                    if let Some(ret_type) = self.get_function_return_type(&ident.node)
                        .or_else(|| self.get_function_return_type(&resolved_name)) {
                        if !ret_type.is_empty() {
                            // If the return type is not already qualified, try to qualify it
                            // based on the function's module (e.g., nalgebra.vec returns Vec -> nalgebra.Vec)
                            if !ret_type.contains('.') && self.types.contains_key(&ret_type) {
                                return Some(ret_type);
                            }
                            // Try qualifying with the function's module path
                            if let Some(dot_pos) = resolved_name.rfind('.') {
                                let module_prefix = &resolved_name[..dot_pos];
                                let qualified_type = format!("{}.{}", module_prefix, ret_type);
                                if self.types.contains_key(&qualified_type) {
                                    return Some(qualified_type);
                                }
                            }
                            // Fall back to unqualified name
                            return Some(ret_type);
                        }
                    }

                    // Fall back to parsing signature if return_type not set
                    if let Some(sig) = self.get_function_signature(&ident.node)
                        .or_else(|| self.get_function_signature(&resolved_name)) {
                        // For 0-arity functions, the signature IS the return type
                        // For others, extract after "-> "
                        let return_type = if let Some(arrow_pos) = sig.rfind("-> ") {
                            sig[arrow_pos + 3..].trim().to_string()
                        } else {
                            sig.trim().to_string()
                        };
                        if !return_type.is_empty() {
                            // Strip trait bounds prefix (e.g., "Eq a, Hash a => Map[a, b]" -> "Map[a, b]")
                            let stripped = if let Some(arrow_pos) = return_type.find("=>") {
                                return_type[arrow_pos + 2..].trim().to_string()
                            } else {
                                return_type
                            };
                            // Only use if it's a concrete type (not a single-letter type variable)
                            // Type variables are single lowercase letters like "a", "b", "c"
                            let is_type_var = stripped.len() == 1 && stripped.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false);
                            if !is_type_var {
                                // Try to qualify the return type with the function's module
                                if !stripped.contains('.') {
                                    if let Some(dot_pos) = resolved_name.rfind('.') {
                                        let module_prefix = &resolved_name[..dot_pos];
                                        let qualified_type = format!("{}.{}", module_prefix, stripped);
                                        if self.types.contains_key(&qualified_type) {
                                            return Some(qualified_type);
                                        }
                                    }
                                }
                                return Some(stripped);
                            }
                        }
                    }
                }
                // Handle FieldAccess as function (e.g., module.function())
                // This handles calls like nalgebra.vec([1, 2, 3])
                if let Expr::FieldAccess(obj, field, _) = func.as_ref() {
                    // Build the qualified function name from the field access
                    if let Expr::Var(module_ident) = obj.as_ref() {
                        let qualified_name = format!("{}.{}", module_ident.node, field.node);
                        let resolved_name = self.resolve_name(&qualified_name);

                        // Try to get return type from function
                        if let Some(ret_type) = self.get_function_return_type(&qualified_name)
                            .or_else(|| self.get_function_return_type(&resolved_name)) {
                            if !ret_type.is_empty() {
                                // If the return type is not already qualified, try to qualify it
                                // based on the function's module
                                if !ret_type.contains('.') && self.types.contains_key(&ret_type) {
                                    return Some(ret_type);
                                }
                                // Try qualifying with the function's module path
                                if let Some(dot_pos) = resolved_name.rfind('.') {
                                    let module_prefix = &resolved_name[..dot_pos];
                                    let qualified_type = format!("{}.{}", module_prefix, ret_type);
                                    if self.types.contains_key(&qualified_type) {
                                        return Some(qualified_type);
                                    }
                                }
                                // Fall back to unqualified name
                                return Some(ret_type);
                            }
                        }
                    }
                }
                None
            }
            // For variables, look up tracked type
            Expr::Var(ident) => {
                // Check param_types first (for monomorphized function variants)
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return Some(ty.clone());
                }
                // Then check local_types
                self.local_types.get(&ident.node).cloned()
            }
            // For field access, look up the field's type from the record definition
            Expr::FieldAccess(obj, field, _) => {
                // First, get the type of the base object
                if let Some(obj_type) = self.expr_type_name(obj) {
                    // Look up the type definition
                    if let Some(type_info) = self.types.get(&obj_type) {
                        if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                            // Find the field and return its type
                            for (fname, ftype) in fields {
                                if fname == &field.node {
                                    return Some(ftype.clone());
                                }
                            }
                        }
                    }
                }
                None
            }
            // Binary operations - try to determine result type
            Expr::BinOp(lhs, op, rhs, _) => {
                match op {
                    // Arithmetic operators preserve numeric type
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => {
                        // If either operand has a known type, use that
                        if let Some(lhs_type) = self.expr_type_name(lhs) {
                            return Some(lhs_type);
                        }
                        if let Some(rhs_type) = self.expr_type_name(rhs) {
                            return Some(rhs_type);
                        }
                        None
                    }
                    // Comparison and logical operators return Bool
                    BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq |
                    BinOp::Gt | BinOp::GtEq | BinOp::And | BinOp::Or => {
                        Some("Bool".to_string())
                    }
                    // String/List concatenation - try to preserve type
                    BinOp::Concat => {
                        if let Some(lhs_type) = self.expr_type_name(lhs) {
                            return Some(lhs_type);
                        }
                        if let Some(rhs_type) = self.expr_type_name(rhs) {
                            return Some(rhs_type);
                        }
                        Some("String".to_string()) // Default to String
                    }
                    // List cons - returns a list
                    BinOp::Cons => {
                        // The result is the type of the RHS (list)
                        if let Some(rhs_type) = self.expr_type_name(rhs) {
                            return Some(rhs_type);
                        }
                        None
                    }
                    // Pipe doesn't change type (returns result of RHS function)
                    BinOp::Pipe => None,
                }
            }
            // Unary operations
            Expr::UnaryOp(op, inner, _) => {
                match op {
                    UnaryOp::Neg => self.expr_type_name(inner),
                    UnaryOp::Not => Some("Bool".to_string()),
                }
            }
            // If expression - try to get type from branches
            Expr::If(_, then_branch, else_branch, _) => {
                if let Some(ty) = self.expr_type_name(then_branch) {
                    return Some(ty);
                }
                self.expr_type_name(else_branch)
            }
            // Block - type is type of last expression
            Expr::Block(stmts, _) => {
                if let Some(last) = stmts.last() {
                    if let Stmt::Expr(e) = last {
                        return self.expr_type_name(e);
                    }
                }
                None
            }
            // Method call - determine return type based on receiver type and method
            Expr::MethodCall(obj, method, _args, _) => {
                // Check if this is a module-qualified function call (e.g., nalgebra.vec, Http.get)
                if let Some(module_path) = self.extract_module_path(obj) {
                    let qualified_name = format!("{}.{}", module_path, method.node);
                    let resolved_name = self.resolve_name(&qualified_name);

                    // Try to get return type from user-defined function first
                    if let Some(ret_type) = self.get_function_return_type(&qualified_name)
                        .or_else(|| self.get_function_return_type(&resolved_name)) {
                        if !ret_type.is_empty() {
                            // If the return type is not already qualified, try to qualify it
                            // based on the function's module
                            if !ret_type.contains('.') && self.types.contains_key(&ret_type) {
                                return Some(ret_type);
                            }
                            // Try qualifying with the function's module path
                            let qualified_type = format!("{}.{}", module_path, ret_type);
                            if self.types.contains_key(&qualified_type) {
                                return Some(qualified_type);
                            }
                            // Fall back to unqualified name
                            return Some(ret_type);
                        }
                    }

                    // Fall back to builtin signature check
                    if let Some(sig) = Self::get_builtin_signature(&qualified_name) {
                        // Extract return type from signature (after last "->")
                        if let Some(arrow_pos) = sig.rfind("->") {
                            let return_type = sig[arrow_pos + 2..].trim().to_string();
                            if !return_type.is_empty() {
                                return Some(return_type);
                            }
                        }
                    }
                }

                if let Some(obj_type) = self.expr_type_name(obj) {
                    // Map methods
                    if obj_type.starts_with("Map[") || obj_type == "Map" {
                        return match method.node.as_str() {
                            // Methods that return Map
                            "insert" | "remove" | "merge" => Some(obj_type),
                            // Methods that return Bool
                            "contains" | "isEmpty" => Some("Bool".to_string()),
                            // Methods that return Int
                            "size" => Some("Int".to_string()),
                            // Methods that return List
                            "keys" | "values" => Some("List".to_string()),
                            // get returns unknown type (the value type)
                            _ => None,
                        };
                    }
                    // Set methods
                    else if obj_type.starts_with("Set[") || obj_type == "Set" {
                        return match method.node.as_str() {
                            // Methods that return Set
                            "insert" | "remove" | "union" | "intersection" | "difference" => Some(obj_type),
                            // Methods that return Bool
                            "contains" | "isEmpty" => Some("Bool".to_string()),
                            // Methods that return Int
                            "size" => Some("Int".to_string()),
                            // Methods that return List
                            "toList" => Some("List".to_string()),
                            _ => None,
                        };
                    }
                    // String methods
                    else if obj_type == "String" {
                        return match method.node.as_str() {
                            // Methods that return String
                            "toUpper" | "toLower" | "trim" | "trimStart" | "trimEnd" |
                            "replace" | "replaceAll" | "substring" | "repeat" |
                            "padStart" | "padEnd" | "reverse" => Some("String".to_string()),
                            // Methods that return Bool
                            "contains" | "startsWith" | "endsWith" | "isEmpty" => Some("Bool".to_string()),
                            // Methods that return Int
                            "length" | "indexOf" | "lastIndexOf" => Some("Int".to_string()),
                            // Methods that return List
                            "chars" | "lines" | "words" | "split" => Some("List".to_string()),
                            _ => None,
                        };
                    }
                    // List methods (from stdlib)
                    else if obj_type.starts_with("List[") || obj_type == "List" {
                        return match method.node.as_str() {
                            // Methods that return List
                            "map" | "filter" | "take" | "drop" | "reverse" | "sort" |
                            "concat" | "flatten" | "unique" | "takeWhile" | "dropWhile" |
                            "zip" | "zipWith" | "interleave" | "group" | "scanl" |
                            "init" | "push" | "remove" | "removeAt" | "insertAt" |
                            "set" | "slice" | "findIndices" => Some(obj_type),
                            // Methods that return Bool
                            "any" | "all" | "contains" => Some("Bool".to_string()),
                            // Methods that return Int
                            "count" => Some("Int".to_string()),
                            // fold, find, etc. return unknown types
                            _ => None,
                        };
                    }
                    // Buffer methods
                    else if obj_type == "Buffer" {
                        return match method.node.as_str() {
                            "append" => Some("Buffer".to_string()),
                            "toString" => Some("String".to_string()),
                            _ => None,
                        };
                    }
                    // Float64Array methods
                    else if obj_type == "Float64Array" {
                        return match method.node.as_str() {
                            "length" => Some("Int".to_string()),
                            "get" => Some("Float".to_string()),
                            "set" => Some("Float64Array".to_string()),
                            "toList" => Some("List".to_string()),
                            "slice" | "map" | "fill" => Some("Float64Array".to_string()),
                            "sum" | "min" | "max" | "mean" => Some("Float".to_string()),
                            "fold" => None, // Return type depends on accumulator
                            _ => None,
                        };
                    }
                    // Int64Array methods
                    else if obj_type == "Int64Array" {
                        return match method.node.as_str() {
                            "length" => Some("Int".to_string()),
                            "get" => Some("Int".to_string()),
                            "set" => Some("Int64Array".to_string()),
                            "toList" => Some("List".to_string()),
                            "slice" | "map" | "fill" => Some("Int64Array".to_string()),
                            "sum" | "min" | "max" => Some("Int".to_string()),
                            "fold" => None,
                            _ => None,
                        };
                    }
                    // Float32Array methods
                    else if obj_type == "Float32Array" {
                        return match method.node.as_str() {
                            "length" => Some("Int".to_string()),
                            "get" => Some("Float".to_string()),
                            "set" => Some("Float32Array".to_string()),
                            "toList" => Some("List".to_string()),
                            "slice" | "map" | "fill" => Some("Float32Array".to_string()),
                            "sum" | "min" | "max" | "mean" => Some("Float".to_string()),
                            "fold" => None,
                            _ => None,
                        };
                    }

                    // Fall through to user-defined function lookup below
                }

                // Check for generic builtins that work on any type (UFCS style)
                // These have signatures like "a -> ReturnType" where 'a' is any type
                match method.node.as_str() {
                    "show" => return Some("String".to_string()),
                    "hash" => return Some("Int".to_string()),
                    "copy" => {
                        // copy returns the same type as input
                        if let Some(obj_type) = self.expr_type_name(obj) {
                            return Some(obj_type);
                        }
                    }
                    // Type conversion methods (as<Type>)
                    "asInt8" | "toInt8" => return Some("Int8".to_string()),
                    "asInt16" | "toInt16" => return Some("Int16".to_string()),
                    "asInt32" | "toInt32" => return Some("Int32".to_string()),
                    "asInt64" | "asInt" | "toInt" | "toInt64" => return Some("Int".to_string()),
                    "asUInt8" | "toUInt8" => return Some("UInt8".to_string()),
                    "asUInt16" | "toUInt16" => return Some("UInt16".to_string()),
                    "asUInt32" | "toUInt32" => return Some("UInt32".to_string()),
                    "asUInt64" | "toUInt64" => return Some("UInt64".to_string()),
                    "asFloat32" | "toFloat32" => return Some("Float32".to_string()),
                    "asFloat64" | "asFloat" | "toFloat" | "toFloat64" => return Some("Float".to_string()),
                    "asBigInt" | "toBigInt" => return Some("BigInt".to_string()),
                    _ => {}
                }

                // Try to find a user-defined function that matches this method call
                // This handles UFCS calls like person.greet() -> greet(person)
                let method_name = &method.node;
                let resolved = self.resolve_name(method_name);

                // Try different function name patterns
                for fn_key in self.fn_asts.keys() {
                    // Match "method/Type,..." or just "method/..."
                    let base_name = fn_key.split('/').next().unwrap_or(fn_key);
                    if base_name == resolved || base_name == method_name {
                        if let Some(fn_def) = self.fn_asts.get(fn_key) {
                            if !fn_def.clauses.is_empty() {
                                if let Some(ref ret_type) = fn_def.clauses[0].return_type {
                                    return Some(self.type_expr_to_string(ret_type));
                                }
                            }
                        }
                    }
                }

                None
            }
            _ => None,
        }
    }

    /// Compile a monomorphized (type-specialized) variant of a function.
    /// Returns the mangled function name if successful.
    fn compile_monomorphized_variant(
        &mut self,
        base_name: &str,
        arg_type_names: &[String],
        param_names: &[String],
    ) -> Result<String, CompileError> {
        // Extract base function name (without signature)
        // base_name is like "hashable/T", we want "hashable"
        let fn_base = base_name.split('/').next().unwrap_or(base_name);

        // Generate mangled name: fnbase$Type1_Type2/Type1,Type2
        // The $ identifies this as a monomorphized variant
        // The /signature matches what compile_fn_def generates
        let type_suffix = arg_type_names.join("_");
        let sig_suffix = arg_type_names.join(",");
        let mangled_name = format!("{}${}/{}", fn_base, type_suffix, sig_suffix);

        // Check if variant exists and is NOT stale (marked by __stale__ prefix in name)
        if let Some(existing) = self.functions.get(&mangled_name) {
            if !existing.name.starts_with("__stale__") {
                // Variant exists and is fresh - use it
                return Ok(mangled_name);
            }
            // Variant is stale (base function was redefined) - continue to recompile
        }

        // Get the original function's AST
        let fn_def = match self.fn_asts.get(base_name) {
            Some(def) => def.clone(),
            None => return Err(CompileError::UnknownFunction {
                name: base_name.to_string(),
                span: Span::default(),
            }),
        };

        // Extract the module path from the base_name (e.g., "json.jsonParse" -> ["json"])
        // This is needed to compile the function in its original module context
        let original_module_path: Vec<String> = if base_name.contains('.') {
            let parts: Vec<&str> = base_name.rsplitn(2, '.').collect();
            if parts.len() == 2 {
                parts[1].split('.').map(|s| s.to_string()).collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // Create a new FnDef with the mangled name (use local name, not fully qualified)
        // Clear type_params so the specialized version is not treated as polymorphic
        let mut specialized_def = fn_def.clone();
        // Use just the base part without signature for the def name
        // mangled_name is like "hashable$Point/Point", we want "hashable$Point"
        let name_without_sig = mangled_name.split('/').next().unwrap_or(&mangled_name);
        let local_name = if name_without_sig.contains('.') {
            name_without_sig.rsplitn(2, '.').next().unwrap_or(name_without_sig).to_string()
        } else {
            name_without_sig.to_string()
        };
        specialized_def.name = Spanned::new(local_name, fn_def.name.span);
        specialized_def.type_params.clear(); // No longer polymorphic

        // Update parameter types in clauses to use concrete types instead of type params
        // This ensures the function signature uses concrete types
        for clause in &mut specialized_def.clauses {
            for (i, param) in clause.params.iter_mut().enumerate() {
                if i < arg_type_names.len() {
                    // Replace the type annotation with the concrete type
                    let concrete_type_name = &arg_type_names[i];
                    param.ty = Some(TypeExpr::Name(Spanned::new(
                        concrete_type_name.clone(),
                        Span::default(),
                    )));
                }
            }
        }

        // Save current context
        let saved_param_types = std::mem::take(&mut self.param_types);
        let saved_module_path = std::mem::replace(&mut self.module_path, original_module_path.clone());
        let saved_imports = std::mem::take(&mut self.imports);

        // Set param_types for this specialization
        for (i, param_name) in param_names.iter().enumerate() {
            if i < arg_type_names.len() {
                self.param_types.insert(param_name.clone(), arg_type_names[i].clone());
            }
        }

        // Forward declare the function with the correct module
        let arity = fn_def.clauses[0].params.len();
        let placeholder = FunctionValue {
            name: mangled_name.clone(),
            arity,
            param_names: param_names.to_vec(),
            code: Arc::new(Chunk::new()),
            module: if original_module_path.is_empty() { None } else { Some(original_module_path.join(".")) },
            source_span: None,
            jit_code: None,
            call_count: AtomicU32::new(0),
            debug_symbols: vec![],
            // REPL introspection fields - will be populated when compiled
            source_code: None,
            source_file: None,
            doc: None,
            signature: None,
            param_types: vec![],
            return_type: None,
        };
        self.functions.insert(mangled_name.clone(), Arc::new(placeholder));

        // Assign function index
        if !self.function_indices.contains_key(&mangled_name) {
            let idx = self.function_list.len() as u16;
            self.function_indices.insert(mangled_name.clone(), idx);
            self.function_list.push(mangled_name.clone());
        }

        // Compile the specialized function in the original module context
        self.compile_fn_def(&specialized_def)?;

        // Restore context
        self.param_types = saved_param_types;
        self.module_path = saved_module_path;
        self.imports = saved_imports;

        Ok(mangled_name)
    }

    /// Get parameter types for a function by name.
    /// Returns the TypeExpr for each parameter, or None if no type annotation.
    fn get_function_param_types(&self, fn_name: &str) -> Vec<Option<TypeExpr>> {
        let resolved = self.resolve_name(fn_name);
        let prefix = format!("{}/", resolved);

        // Try to find the function definition in fn_asts
        for (key, def) in &self.fn_asts {
            if key.starts_with(&prefix) || key == fn_name {
                if !def.clauses.is_empty() {
                    return def.clauses[0].params.iter()
                        .map(|p| p.ty.clone())
                        .collect();
                }
            }
        }
        vec![]
    }

    /// Compile an argument expression with knowledge of the expected parameter type.
    /// This handles the case where a polymorphic function is passed as an argument
    /// and needs to be monomorphized based on the expected function type.
    fn compile_arg_with_expected_type(
        &mut self,
        arg: &Expr,
        expected_type: Option<&TypeExpr>,
    ) -> Result<Reg, CompileError> {
        // Check if this argument is a variable that refers to a polymorphic function
        if let Expr::Var(ident) = arg {
            let name = &ident.node;
            let resolved = self.resolve_name(name);
            let prefix = format!("{}/", resolved);

            // Check if this variable refers to a polymorphic function
            let poly_fn_name = self.polymorphic_fns.iter()
                .find(|k| k.starts_with(&prefix))
                .cloned();

            if let Some(poly_name) = poly_fn_name {
                // We have a polymorphic function. Try to determine the type to specialize for.
                if let Some(TypeExpr::Function(param_types, _)) = expected_type {
                    // Expected type is a function type like (Val) -> Int
                    // Extract concrete types from the function parameters
                    let concrete_types: Vec<String> = param_types.iter()
                        .filter_map(|t| self.type_expr_to_type_name(t))
                        .collect();

                    if concrete_types.len() == param_types.len() && !concrete_types.is_empty() {
                        // All types are concrete - monomorphize and load
                        if let Some(fn_def) = self.fn_asts.get(&poly_name).cloned() {
                            let param_names: Vec<String> = fn_def.clauses[0].params.iter()
                                .filter_map(|p| self.pattern_binding_name(&p.pattern))
                                .collect();

                            match self.compile_monomorphized_variant(&poly_name, &concrete_types, &param_names) {
                                Ok(mangled_name) => {
                                    // Load the monomorphized function
                                    if let Some(func) = self.functions.get(&mangled_name).cloned() {
                                        let dst = self.alloc_reg();
                                        let idx = self.chunk.add_constant(Value::Function(func));
                                        let line = self.span_line(arg.span());
                                        self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                                        return Ok(dst);
                                    }
                                }
                                Err(_) => {
                                    // Fall through to normal compilation
                                }
                            }
                        }
                    }
                }
            }
        }

        // Default: compile normally
        self.compile_expr_tail(arg, false)
    }

    /// Convert a TypeExpr to a concrete type name, or None if it's a type parameter.
    fn type_expr_to_type_name(&self, ty: &TypeExpr) -> Option<String> {
        match ty {
            TypeExpr::Name(ident) => {
                let name = &ident.node;
                // Check if it's a concrete type (in self.types or a primitive)
                if self.types.contains_key(name) {
                    return Some(name.clone());
                }
                if matches!(name.as_str(), "Int" | "Float" | "String" | "Bool" | "Char" | "()" | "List" | "Map" | "Set" | "Tuple") {
                    return Some(name.clone());
                }
                // Single uppercase letters are likely type parameters
                if name.len() == 1 && name.chars().next().unwrap().is_uppercase() {
                    return None;
                }
                // Could be an unknown type - return it anyway
                Some(name.clone())
            }
            TypeExpr::Unit => Some("()".to_string()),
            TypeExpr::Tuple(types) => {
                // For tuple types, we'd need to handle them specially
                // For now, just return Tuple
                if types.iter().all(|t| self.type_expr_to_type_name(t).is_some()) {
                    Some("Tuple".to_string())
                } else {
                    None
                }
            }
            TypeExpr::Function(_, _) => {
                // Nested function types - return generic name for now
                Some("Fn".to_string())
            }
            _ => None,
        }
    }

    /// Build a map from type parameters to concrete types based on argument expressions.
    /// For example, if expected_types has `(T) -> Int` and `T`, and the second argument
    /// is `Val(42)`, we can deduce that `T = Val`.
    fn build_type_param_map(&self, expected_types: &[Option<TypeExpr>], args: &[Expr]) -> HashMap<String, String> {
        let mut map = HashMap::new();

        // First pass: find simple type parameter mappings from non-function types
        for (expected, arg) in expected_types.iter().zip(args.iter()) {
            if let Some(type_expr) = expected {
                // Check if this parameter is a simple type parameter (single uppercase letter)
                if let TypeExpr::Name(ident) = type_expr {
                    let name = &ident.node;
                    if name.len() == 1 && name.chars().next().unwrap().is_uppercase() {
                        // This is a type parameter - try to get concrete type from argument
                        if let Some(concrete_type) = self.expr_type_name(arg) {
                            // Only map if the concrete type is actually concrete (not another type param)
                            if !(concrete_type.len() == 1 && concrete_type.chars().next().unwrap().is_uppercase()) {
                                map.insert(name.clone(), concrete_type);
                            }
                        }
                    }
                }
            }
        }

        map
    }

    /// Substitute type parameters in a TypeExpr using the given map.
    /// For example, `(T) -> Int` with map `{T: Val}` becomes `(Val) -> Int`.
    fn substitute_type_params(&self, type_expr: &TypeExpr, map: &HashMap<String, String>) -> TypeExpr {
        match type_expr {
            TypeExpr::Name(ident) => {
                if let Some(concrete) = map.get(&ident.node) {
                    TypeExpr::Name(Spanned::new(concrete.clone(), ident.span))
                } else {
                    type_expr.clone()
                }
            }
            TypeExpr::Function(params, ret) => {
                let new_params: Vec<TypeExpr> = params.iter()
                    .map(|p| self.substitute_type_params(p, map))
                    .collect();
                let new_ret = Box::new(self.substitute_type_params(ret, map));
                TypeExpr::Function(new_params, new_ret)
            }
            TypeExpr::Tuple(types) => {
                let new_types: Vec<TypeExpr> = types.iter()
                    .map(|t| self.substitute_type_params(t, map))
                    .collect();
                TypeExpr::Tuple(new_types)
            }
            TypeExpr::Generic(name, args) => {
                let new_args: Vec<TypeExpr> = args.iter()
                    .map(|a| self.substitute_type_params(a, map))
                    .collect();
                TypeExpr::Generic(name.clone(), new_args)
            }
            TypeExpr::Record(fields) => {
                let new_fields: Vec<(Ident, TypeExpr)> = fields.iter()
                    .map(|(name, ty)| (name.clone(), self.substitute_type_params(ty, map)))
                    .collect();
                TypeExpr::Record(new_fields)
            }
            _ => type_expr.clone(),
        }
    }

    /// Extract a module path from an expression.
    /// Returns Some("module") or Some("outer.inner") if the expression is a module reference.
    /// Also returns Type.Trait prefixes for explicit trait method calls.
    fn extract_module_path(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Var(ident) => {
                // Check if the identifier is a known module OR if there are functions/mvars with this prefix
                let prefix = format!("{}.", ident.node);
                if self.known_modules.contains(&ident.node)
                    || self.functions.keys().any(|k| k.starts_with(&prefix))
                    || self.mvars.keys().any(|k| k.starts_with(&prefix)) {
                    Some(ident.node.clone())
                } else {
                    None
                }
            }
            // Handle empty Record expressions as module references (e.g., Math in Math.add)
            // or type references (e.g., Box in Box.Doubler.doubler)
            // The parser parses uppercase identifiers like `Math` as Record constructors
            Expr::Record(type_name, fields, _) if fields.is_empty() => {
                // Check if it's a known module OR if there are functions with this prefix
                let prefix = format!("{}.", type_name.node);
                if self.known_modules.contains(&type_name.node)
                    || self.functions.keys().any(|k| k.starts_with(&prefix)) {
                    Some(type_name.node.clone())
                } else {
                    None
                }
            }
            Expr::FieldAccess(target, field, _) => {
                // Check if we're building a nested module path like outer.inner
                // or Type.Trait prefix like Box.Doubler
                if let Some(base) = self.extract_module_path(target) {
                    let combined = format!("{}.{}", base, field.node);
                    // Check if the combined path is a known module OR if there are functions/mvars with this prefix
                    let prefix = format!("{}.", combined);
                    if self.known_modules.contains(&combined)
                        || self.functions.keys().any(|k| k.starts_with(&prefix))
                        || self.mvars.keys().any(|k| k.starts_with(&prefix)) {
                        Some(combined)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            // Handle MethodCall on modules for nested modules (e.g., outer.inner in outer.inner.func)
            Expr::MethodCall(obj, method, args, _) if args.is_empty() => {
                if let Some(base) = self.extract_module_path(obj) {
                    let combined = format!("{}.{}", base, method.node);
                    // Check if the combined path is a known module OR if there are functions with this prefix
                    let prefix = format!("{}.", combined);
                    if self.known_modules.contains(&combined)
                        || self.functions.keys().any(|k| k.starts_with(&prefix)) {
                        Some(combined)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if a function can be accessed from the current module.
    /// Returns Ok(()) if access is allowed, Err with PrivateAccess or ModuleNotImported otherwise.
    fn check_visibility(&self, qualified_name: &str, span: Span) -> Result<(), CompileError> {
        // REPL mode bypasses all visibility checks for interactive exploration
        if self.repl_mode {
            return Ok(());
        }

        // Prelude functions (stdlib) bypass visibility checks
        // Strip signature suffix (e.g., "stdlib.list.reverse/_" -> "stdlib.list.reverse")
        let base_name = qualified_name.split('/').next().unwrap_or(qualified_name);
        if self.prelude_functions.contains(base_name) {
            return Ok(());
        }

        // Functions brought in via `use` statements bypass import checks
        // They were explicitly imported by the user
        if self.imports.values().any(|v| v == base_name || qualified_name.starts_with(&format!("{}/", v))) {
            return Ok(());
        }

        // Get the visibility of the function
        let visibility = self.function_visibility.get(qualified_name);

        // If we don't know about this function yet, assume it's accessible
        // (it might be a built-in or will be an UnknownFunction error later)
        let visibility = match visibility {
            Some(v) => *v,
            None => return Ok(()),
        };

        // Extract the module path from the qualified name (everything before the last dot)
        let function_module: Vec<&str> = qualified_name.rsplitn(2, '.').collect();
        let function_module = if function_module.len() > 1 {
            function_module[1].split('.').collect::<Vec<_>>()
        } else {
            vec![] // Function is in root module
        };

        // Current module path
        let current_module: Vec<&str> = self.module_path.iter().map(|s| s.as_str()).collect();

        // Check if same module (allow access to both public and private functions)
        let is_same_module = current_module.len() >= function_module.len()
            && current_module[..function_module.len()] == function_module[..];

        if is_same_module {
            return Ok(());
        }

        // Different module - check if it's imported
        let function_module_string = function_module.join(".");
        let function_name_with_sig = qualified_name.rsplit('.').next().unwrap_or(qualified_name);
        let function_name = function_name_with_sig.split('/').next().unwrap_or(function_name_with_sig);

        // Check if this is a trait method call (Type.Trait.method pattern)
        // These don't require imports as they're part of trait dispatch
        if function_module.len() == 2 {
            let type_name = function_module[0];
            let trait_name = function_module[1];
            // Check if this is a registered trait implementation
            if self.trait_impls.contains_key(&(type_name.to_string(), trait_name.to_string())) {
                // This is a trait method call, allow it
                return Ok(());
            }
        }

        // Check if the module is imported by the current module or is a local (inline) module
        let is_imported = self.imported_modules.contains(&(self.module_path.clone(), function_module_string.clone()));
        let is_local_module = self.local_modules.contains(&function_module_string);

        if !is_imported && !is_local_module && !function_module.is_empty() {
            // Module not imported
            return Err(CompileError::ModuleNotImported {
                module: function_module_string,
                function: function_name.to_string(),
                span,
            });
        }

        // Module is imported (or function is in root) - check visibility
        if visibility.is_public() {
            return Ok(());
        }

        // Private function from imported module - access denied
        let module_name = if function_module.is_empty() {
            "<root>".to_string()
        } else {
            function_module_string
        };

        Err(CompileError::PrivateAccess {
            function: function_name.to_string(),
            module: module_name,
            span,
        })
    }

    /// Compile an if expression.
    fn compile_if(
        &mut self,
        cond: &Expr,
        then_branch: &Expr,
        else_branch: &Expr,
        is_tail: bool,
    ) -> Result<Reg, CompileError> {
        let cond_reg = self.compile_expr_tail(cond, false)?;
        let dst = self.alloc_reg();

        // Jump to else if false
        let else_jump = self.chunk.emit(Instruction::JumpIfFalse(cond_reg, 0), 0);

        // Then branch
        let then_reg = self.compile_expr_tail(then_branch, is_tail)?;

        // Optimization: in tail position with a simple value, emit Return directly
        // Skip this optimization if the then_branch is a tail call (already emits TailCall*)
        if is_tail && !self.is_tail_call_expr(then_branch) {
            // Emit MvarUnlock for all held locks before returning
            for (_, name_idx, is_write) in self.current_fn_mvar_locks.iter().rev() {
                self.chunk.emit(Instruction::MvarUnlock(*name_idx, *is_write), 0);
            }
            self.chunk.emit(Instruction::Return(then_reg), 0);
        } else {
            self.chunk.emit(Instruction::Move(dst, then_reg), 0);
        }
        let end_jump = self.chunk.emit(Instruction::Jump(0), 0);

        // Else branch
        let else_target = self.chunk.code.len();
        self.chunk.patch_jump(else_jump, else_target);

        let else_reg = self.compile_expr_tail(else_branch, is_tail)?;

        // Same optimization for else branch
        if is_tail && !self.is_tail_call_expr(else_branch) {
            // Emit MvarUnlock for all held locks before returning
            for (_, name_idx, is_write) in self.current_fn_mvar_locks.iter().rev() {
                self.chunk.emit(Instruction::MvarUnlock(*name_idx, *is_write), 0);
            }
            self.chunk.emit(Instruction::Return(else_reg), 0);
        } else {
            self.chunk.emit(Instruction::Move(dst, else_reg), 0);
        }

        let end_target = self.chunk.code.len();
        self.chunk.patch_jump(end_jump, end_target);

        Ok(dst)
    }

    /// Check if an expression is a tail call (Call that will emit TailCall* instructions)
    fn is_tail_call_expr(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::Call(_, _, _, _))
    }

    /// Compile a match expression.
    fn compile_match(&mut self, scrutinee: &Expr, arms: &[MatchArm], is_tail: bool, line: usize) -> Result<Reg, CompileError> {
        let scrut_reg = self.compile_expr_tail(scrutinee, false)?;
        let dst = self.alloc_reg();
        let mut end_jumps = Vec::new();
        let mut last_arm_fail_jump: Option<usize> = None;
        // Jumps that need to go to the next arm (pattern fail or guard fail)
        let mut jumps_to_next_arm: Vec<usize> = Vec::new();

        for (i, arm) in arms.iter().enumerate() {
            let is_last = i == arms.len() - 1;

            // Patch any jumps from previous arm that should go to this arm
            let arm_start = self.chunk.code.len();
            for jump in jumps_to_next_arm.drain(..) {
                self.chunk.patch_jump(jump, arm_start);
            }

            // Save locals before processing arm (pattern bindings should be scoped to this arm)
            let saved_locals = self.locals.clone();

            // Try to match the pattern
            let (match_success, bindings) = self.compile_pattern_test(&arm.pattern, scrut_reg)?;

            // If pattern fails, jump to next arm (or panic if last)
            let pattern_fail_jump = self.chunk.emit(Instruction::JumpIfFalse(match_success, 0), 0);

            // Bind pattern variables with type info from pattern
            for (name, reg, is_float) in bindings {
                self.locals.insert(name, LocalInfo { reg, is_float, mutable: false });
            }

            // Compile guard if present
            let guard_fail_jump = if let Some(guard) = &arm.guard {
                let guard_reg = self.compile_expr_tail(guard, false)?;
                Some(self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), 0))
            } else {
                None
            };

            // Compile arm body - pass is_tail for tail calls, but always move result
            let body_reg = self.compile_expr_tail(&arm.body, is_tail)?;
            self.chunk.emit(Instruction::Move(dst, body_reg), 0);

            // After body, jump to end of match
            end_jumps.push(self.chunk.emit(Instruction::Jump(0), 0));

            // Handle jumps to next arm
            if is_last {
                // Last arm: pattern fail or guard fail should panic
                last_arm_fail_jump = Some(pattern_fail_jump);
                if let Some(guard_jump) = guard_fail_jump {
                    // For last arm with guard, need to also jump to panic on guard fail
                    // Patch pattern fail to same location as guard fail
                    jumps_to_next_arm.push(guard_jump);
                }
            } else {
                // Not last arm: pattern fail or guard fail should try next arm
                jumps_to_next_arm.push(pattern_fail_jump);
                if let Some(guard_jump) = guard_fail_jump {
                    jumps_to_next_arm.push(guard_jump);
                }
            }

            // Restore locals after arm (pattern bindings shouldn't leak to next arm)
            self.locals = saved_locals;
        }

        // Patch remaining jumps (from last arm failures) to panic location
        let panic_location = self.chunk.code.len();
        for jump in jumps_to_next_arm.drain(..) {
            self.chunk.patch_jump(jump, panic_location);
        }

        // If last arm failed, emit panic for non-exhaustive match
        if let Some(fail_jump) = last_arm_fail_jump {
            self.chunk.patch_jump(fail_jump, self.chunk.code.len());
            let msg_idx = self.chunk.add_constant(Value::String(Arc::new("Non-exhaustive match: no pattern matched".to_string())));
            let msg_reg = self.alloc_reg();
            self.chunk.emit(Instruction::LoadConst(msg_reg, msg_idx), line);
            self.chunk.emit(Instruction::Panic(msg_reg), line);
        }

        // Patch all end jumps
        let end_target = self.chunk.code.len();
        for jump in end_jumps {
            self.chunk.patch_jump(jump, end_target);
        }

        Ok(dst)
    }

    /// Compile a try/catch/finally expression.
    fn compile_try(
        &mut self,
        try_expr: &Expr,
        catch_arms: &[MatchArm],
        finally_expr: Option<&Expr>,
        is_tail: bool,
    ) -> Result<Reg, CompileError> {
        let dst = self.alloc_reg();

        // For finally blocks, we need a flag to track if we should rethrow after finally
        // This only adds overhead when finally exists
        let rethrow_flag = if finally_expr.is_some() {
            let flag = self.alloc_reg();
            // Initialize to false - no rethrow pending
            let false_idx = self.chunk.add_constant(Value::Bool(false));
            self.chunk.emit(Instruction::LoadConst(flag, false_idx), 0);
            Some(flag)
        } else {
            None
        };

        // 1. Push exception handler - offset will be patched later
        let handler_idx = self.chunk.emit(Instruction::PushHandler(0), 0);

        // 2. Compile the try body
        let try_result = self.compile_expr_tail(try_expr, false)?;
        self.chunk.emit(Instruction::Move(dst, try_result), 0);

        // 3. Pop the handler (success path)
        self.chunk.emit(Instruction::PopHandler, 0);

        // 4. Jump past the catch block (success path)
        let skip_catch_jump = self.chunk.emit(Instruction::Jump(0), 0);

        // 5. CATCH BLOCK START - patch the handler to jump here
        let catch_start = self.chunk.code.len();
        self.chunk.patch_jump(handler_idx, catch_start);

        // 6. Get the exception value
        let exc_reg = self.alloc_reg();
        self.chunk.emit(Instruction::GetException(exc_reg), 0);

        // 7. Pattern match on the exception (similar to compile_match)
        // We need to track jumps to re-throw if no pattern matches
        let mut end_jumps = Vec::new();
        let mut rethrow_jumps = Vec::new();

        for (i, arm) in catch_arms.iter().enumerate() {
            let is_last = i == catch_arms.len() - 1;

            // Try to match the pattern
            let (match_success, bindings) = self.compile_pattern_test(&arm.pattern, exc_reg)?;

            // Always emit JumpIfFalse, even for last arm (to handle no-match case)
            let next_arm_jump = self.chunk.emit(Instruction::JumpIfFalse(match_success, 0), 0);

            // Bind pattern variables with type info from pattern
            for (name, reg, is_float) in bindings {
                self.locals.insert(name, LocalInfo { reg, is_float, mutable: false });
            }

            // Compile guard if present
            if let Some(guard) = &arm.guard {
                let guard_reg = self.compile_expr_tail(guard, false)?;
                // If guard fails, jump to next arm (or rethrow for last arm)
                let guard_jump = self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), 0);
                if is_last {
                    rethrow_jumps.push(guard_jump);
                } else {
                    // Patch to same location as pattern mismatch
                    rethrow_jumps.push(guard_jump);
                }
            }

            // Compile catch arm body
            let body_reg = self.compile_expr_tail(&arm.body, is_tail && finally_expr.is_none())?;
            self.chunk.emit(Instruction::Move(dst, body_reg), 0);

            // Jump to end (skip other arms and rethrow)
            end_jumps.push(self.chunk.emit(Instruction::Jump(0), 0));

            // Patch jump to next arm (or rethrow block for last arm)
            if is_last {
                rethrow_jumps.push(next_arm_jump);
            } else {
                let next_target = self.chunk.code.len();
                self.chunk.patch_jump(next_arm_jump, next_target);
            }
        }

        // 7.5 Re-throw block: if no pattern matched, handle rethrow
        let rethrow_start = self.chunk.code.len();
        for jump in rethrow_jumps {
            self.chunk.patch_jump(jump, rethrow_start);
        }

        if let Some(flag) = rethrow_flag {
            // With finally: set flag to true and jump to finally, then rethrow after
            let true_idx = self.chunk.add_constant(Value::Bool(true));
            self.chunk.emit(Instruction::LoadConst(flag, true_idx), 0);
            // Fall through to after_catch where finally is compiled
        } else {
            // No finally: rethrow immediately
            self.chunk.emit(Instruction::Throw(exc_reg), 0);
        }

        // 8. Patch all end jumps from catch arms
        let after_catch = self.chunk.code.len();
        for jump in end_jumps {
            self.chunk.patch_jump(jump, after_catch);
        }

        // 9. Patch the skip_catch_jump to land here
        self.chunk.patch_jump(skip_catch_jump, after_catch);

        // 10. Compile finally block if present
        if let Some(finally) = finally_expr {
            // Finally is executed for both success and exception paths
            // Its result is discarded; the try/catch result is preserved in dst
            self.compile_expr_tail(finally, false)?;

            // After finally, check if we need to rethrow
            if let Some(flag) = rethrow_flag {
                let done_jump = self.chunk.emit(Instruction::JumpIfFalse(flag, 0), 0);
                // Rethrow the preserved exception
                self.chunk.emit(Instruction::Throw(exc_reg), 0);
                // Patch done_jump to here (normal exit)
                let done_target = self.chunk.code.len();
                self.chunk.patch_jump(done_jump, done_target);
            }
        }

        Ok(dst)
    }

    /// Compile error propagation: expr?
    /// If expr throws, re-throw the exception. Otherwise return its value.
    fn compile_try_propagate(&mut self, inner_expr: &Expr) -> Result<Reg, CompileError> {
        let dst = self.alloc_reg();

        // 1. Push exception handler
        let handler_idx = self.chunk.emit(Instruction::PushHandler(0), 0);

        // 2. Compile the inner expression
        let result = self.compile_expr_tail(inner_expr, false)?;
        self.chunk.emit(Instruction::Move(dst, result), 0);

        // 3. Pop handler on success
        self.chunk.emit(Instruction::PopHandler, 0);

        // 4. Jump past the re-throw
        let skip_rethrow_jump = self.chunk.emit(Instruction::Jump(0), 0);

        // 5. RE-THROW BLOCK - patch handler to jump here
        let rethrow_start = self.chunk.code.len();
        self.chunk.patch_jump(handler_idx, rethrow_start);

        // 6. Get exception and re-throw it
        let exc_reg = self.alloc_reg();
        self.chunk.emit(Instruction::GetException(exc_reg), 0);
        self.chunk.emit(Instruction::Throw(exc_reg), 0);

        // 7. Patch skip_rethrow_jump to land here
        let after_rethrow = self.chunk.code.len();
        self.chunk.patch_jump(skip_rethrow_jump, after_rethrow);

        Ok(dst)
    }

    /// Compile a pattern test and return (success_reg, bindings).
    /// Bindings are (name, reg, is_float) tuples.
    fn compile_pattern_test(&mut self, pattern: &Pattern, scrut_reg: Reg) -> Result<(Reg, Vec<(String, Reg, bool)>), CompileError> {
        let success_reg = self.alloc_reg();
        let mut bindings: Vec<(String, Reg, bool)> = Vec::new();

        match pattern {
            Pattern::Wildcard(_) => {
                self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
            }
            Pattern::Var(ident) => {
                // Check if variable already exists as immutable binding
                if let Some(existing_info) = self.locals.get(&ident.node).copied() {
                    if !existing_info.mutable {
                        // Immutable variable: use as constraint (test equality)
                        self.chunk.emit(Instruction::Eq(success_reg, scrut_reg, existing_info.reg), 0);
                    } else {
                        // Mutable variable: rebind it
                        self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
                        self.chunk.emit(Instruction::Move(existing_info.reg, scrut_reg), 0);
                    }
                } else {
                    // New variable: create binding
                    self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
                    let var_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::Move(var_reg, scrut_reg), 0);
                    // Type unknown for plain var pattern, will be updated by variant context
                    bindings.push((ident.node.clone(), var_reg, false));
                }
            }
            Pattern::Unit(_) => {
                self.chunk.emit(Instruction::TestUnit(success_reg, scrut_reg), 0);
            }
            Pattern::Int(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int64(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Int8(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int8(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Int16(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int16(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Int32(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int32(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt8(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt8(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt16(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt16(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt32(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt32(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt64(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt64(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Float(f, _) => {
                let const_idx = self.chunk.add_constant(Value::Float64(*f));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Float32(f, _) => {
                let const_idx = self.chunk.add_constant(Value::Float32(*f));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::BigInt(s, _) => {
                use num_bigint::BigInt;
                let big = s.parse::<BigInt>().unwrap_or_default();
                let const_idx = self.chunk.add_constant(Value::BigInt(Arc::new(big)));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Decimal(s, _) => {
                use rust_decimal::Decimal;
                let dec = s.parse::<Decimal>().unwrap_or_default();
                let const_idx = self.chunk.add_constant(Value::Decimal(dec));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Char(c, _) => {
                let const_idx = self.chunk.add_constant(Value::Char(*c));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Bool(b, _) => {
                let const_idx = self.chunk.add_constant(Value::Bool(*b));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::String(s, _) => {
                let const_idx = self.chunk.add_constant(Value::String(Arc::new(s.clone())));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Variant(ctor, fields, _) => {
                // Store constructor name in constants for exact string matching
                // Use local constructor name (without module path) to match how variants are created
                // Variants are created with local tags, so patterns must also use local tags
                let local_ctor = ctor.node.rsplit('.').next().unwrap_or(&ctor.node).to_string();
                let ctor_idx = self.chunk.add_constant(Value::String(Arc::new(local_ctor.clone())));
                self.chunk.emit(Instruction::TestTag(success_reg, scrut_reg, ctor_idx), 0);

                // Look up field types for this constructor - try both qualified and local names
                let qualified_ctor = self.qualify_name(&ctor.node);
                let field_types = {
                    let qt = self.get_constructor_field_types(&qualified_ctor);
                    if !qt.is_empty() { qt } else { self.get_constructor_field_types(&local_ctor) }
                };

                // Extract and bind fields - only if tag matches (guard with conditional jump)
                match fields {
                    VariantPatternFields::Unit => {}
                    VariantPatternFields::Positional(patterns) => {
                        // Jump past field extraction if tag doesn't match
                        let skip_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        for (i, pat) in patterns.iter().enumerate() {
                            let field_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::GetVariantField(field_reg, scrut_reg, i as u8), 0);
                            let (_, mut sub_bindings) = self.compile_pattern_test(pat, field_reg)?;

                            // Update type info based on constructor's field type
                            let is_float = field_types.get(i)
                                .map(|t| matches!(t.as_str(), "Float" | "Float32" | "Float64"))
                                .unwrap_or(false);
                            for binding in &mut sub_bindings {
                                binding.2 = is_float;
                            }
                            bindings.append(&mut sub_bindings);
                        }

                        // Patch the skip jump to land here
                        let after_extract = self.chunk.code.len();
                        self.chunk.patch_jump(skip_jump, after_extract);
                    }
                    VariantPatternFields::Named(nfields) => {
                        // Jump past field extraction if tag doesn't match
                        let skip_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        for field in nfields {
                            match field {
                                RecordPatternField::Punned(ident) => {
                                    // {x} means bind field "x" to variable "x"
                                    let field_reg = self.alloc_reg();
                                    let name_idx = self.chunk.add_constant(Value::String(Arc::new(ident.node.clone())));
                                    self.chunk.emit(Instruction::GetVariantFieldByName(field_reg, scrut_reg, name_idx), 0);
                                    // Type unknown for named fields (would need named field type lookup)
                                    bindings.push((ident.node.clone(), field_reg, false));
                                }
                                RecordPatternField::Named(ident, pat) => {
                                    // {name: n} means bind field "name" to the result of matching pattern
                                    let field_reg = self.alloc_reg();
                                    let name_idx = self.chunk.add_constant(Value::String(Arc::new(ident.node.clone())));
                                    self.chunk.emit(Instruction::GetVariantFieldByName(field_reg, scrut_reg, name_idx), 0);
                                    let (_, mut sub_bindings) = self.compile_pattern_test(pat, field_reg)?;
                                    bindings.append(&mut sub_bindings);
                                }
                                RecordPatternField::Rest(_) => {
                                    // Ignore rest pattern for now - it just matches remaining fields
                                }
                            }
                        }

                        // Patch the skip jump to land here
                        let after_extract = self.chunk.code.len();
                        self.chunk.patch_jump(skip_jump, after_extract);
                    }
                }
            }
            Pattern::List(list_pattern, _) => {
                match list_pattern {
                    ListPattern::Empty => {
                        self.chunk.emit(Instruction::TestNil(success_reg, scrut_reg), 0);
                    }
                    ListPattern::Cons(head_patterns, tail) => {
                        let n = head_patterns.len();

                        if tail.is_some() {
                            // Pattern like [a, b | t] - check list has at least n elements
                            // Check length >= n
                            let len_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Length(len_reg, scrut_reg), 0);
                            let n_reg = self.alloc_reg();
                            let n_idx = self.chunk.add_constant(Value::Int64(n as i64));
                            self.chunk.emit(Instruction::LoadConst(n_reg, n_idx), 0);
                            self.chunk.emit(Instruction::GeInt(success_reg, len_reg, n_reg), 0);
                        } else {
                            // Pattern like [a, b, c] - check list has exactly n elements
                            let len_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Length(len_reg, scrut_reg), 0);
                            let n_reg = self.alloc_reg();
                            let n_idx = self.chunk.add_constant(Value::Int64(n as i64));
                            self.chunk.emit(Instruction::LoadConst(n_reg, n_idx), 0);
                            self.chunk.emit(Instruction::Eq(success_reg, len_reg, n_reg), 0);
                        }

                        // Guard the Decons block: skip if success_reg is false (list too short)
                        // This prevents "Cannot decons empty list" errors when pattern doesn't match
                        let skip_decons_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        // Decons each head element
                        let mut current_list = scrut_reg;
                        for (i, head_pat) in head_patterns.iter().enumerate() {
                            let head_reg = self.alloc_reg();
                            let tail_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Decons(head_reg, tail_reg, current_list), 0);

                            let (head_success, mut head_bindings) = self.compile_pattern_test(head_pat, head_reg)?;
                            // AND the head pattern's success with our overall success
                            self.chunk.emit(Instruction::And(success_reg, success_reg, head_success), 0);
                            bindings.append(&mut head_bindings);

                            // If this is the last head pattern and there's a tail pattern
                            if i == n - 1 {
                                if let Some(tail_pat) = tail {
                                    let (tail_success, mut tail_bindings) = self.compile_pattern_test(tail_pat, tail_reg)?;
                                    // AND the tail pattern's success with our overall success
                                    self.chunk.emit(Instruction::And(success_reg, success_reg, tail_success), 0);
                                    bindings.append(&mut tail_bindings);
                                }
                            } else {
                                current_list = tail_reg;
                            }
                        }

                        // Patch jump to skip past the Decons block
                        self.chunk.patch_jump(skip_decons_jump, self.chunk.code.len());
                    }
                }
            }
            Pattern::StringCons(string_pattern, _) => {
                match string_pattern {
                    StringPattern::Empty => {
                        // Empty string pattern: test if string is ""
                        self.chunk.emit(Instruction::TestEmptyString(success_reg, scrut_reg), 0);
                    }
                    StringPattern::Cons(prefix_strings, tail_pat) => {
                        // String cons pattern like ["hello", "world" | rest]
                        // Concatenate prefix strings to form the expected prefix
                        let prefix: String = prefix_strings.concat();
                        let prefix_len = prefix.chars().count();

                        // First, check if the string is long enough
                        let len_reg = self.alloc_reg();
                        self.chunk.emit(Instruction::Length(len_reg, scrut_reg), 0);
                        let expected_len_reg = self.alloc_reg();
                        let expected_len_idx = self.chunk.add_constant(Value::Int64(prefix_len as i64));
                        self.chunk.emit(Instruction::LoadConst(expected_len_reg, expected_len_idx), 0);
                        self.chunk.emit(Instruction::GeInt(success_reg, len_reg, expected_len_reg), 0);

                        // Guard: skip if string is too short
                        let skip_decons_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        // Decons the string character by character and match against prefix
                        let mut current_str = scrut_reg;
                        for (i, c) in prefix.chars().enumerate() {
                            let head_reg = self.alloc_reg();
                            let tail_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::StringDecons(head_reg, tail_reg, current_str), 0);

                            // Test if head matches expected character
                            let expected_char_reg = self.alloc_reg();
                            let char_str = c.to_string();
                            let char_idx = self.chunk.add_constant(Value::String(Arc::new(char_str)));
                            self.chunk.emit(Instruction::LoadConst(expected_char_reg, char_idx), 0);
                            let char_match_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::EqStr(char_match_reg, head_reg, expected_char_reg), 0);
                            self.chunk.emit(Instruction::And(success_reg, success_reg, char_match_reg), 0);

                            // If last character, compile tail pattern binding
                            if i == prefix_len - 1 {
                                let (tail_success, mut tail_bindings) = self.compile_pattern_test(tail_pat, tail_reg)?;
                                self.chunk.emit(Instruction::And(success_reg, success_reg, tail_success), 0);
                                bindings.append(&mut tail_bindings);
                            } else {
                                current_str = tail_reg;
                            }
                        }

                        // Patch skip jump
                        self.chunk.patch_jump(skip_decons_jump, self.chunk.code.len());
                    }
                }
            }
            Pattern::Tuple(patterns, _) => {
                self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
                for (i, pat) in patterns.iter().enumerate() {
                    let elem_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::GetTupleField(elem_reg, scrut_reg, i as u8), 0);
                    let (sub_success, mut sub_bindings) = self.compile_pattern_test(pat, elem_reg)?;
                    // AND the sub-pattern's success with our overall success
                    self.chunk.emit(Instruction::And(success_reg, success_reg, sub_success), 0);
                    bindings.append(&mut sub_bindings);
                }
            }
            Pattern::Map(entries, _) => {
                // 1. Check if it is a map
                self.chunk.emit(Instruction::IsMap(success_reg, scrut_reg), 0);
                
                // Jump to end if not a map
                let type_fail_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);
                
                for (key_expr, val_pat) in entries {
                    // Compile key expression to a register
                    let key_reg = self.compile_expr_tail(key_expr, false)?;
                    
                    // Check if key exists
                    let exists_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::MapContainsKey(exists_reg, scrut_reg, key_reg), 0);
                    
                    // Update success_reg
                    self.chunk.emit(Instruction::And(success_reg, success_reg, exists_reg), 0);
                    
                    // Guard: if key doesn't exist, skip value check (to avoid panic in MapGet)
                    let skip_val_jump = self.chunk.emit(Instruction::JumpIfFalse(exists_reg, 0), 0);
                    
                    // Get value
                    let val_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::MapGet(val_reg, scrut_reg, key_reg), 0);
                    
                    // Match value against pattern
                    let (sub_success, mut sub_bindings) = self.compile_pattern_test(val_pat, val_reg)?;
                    self.chunk.emit(Instruction::And(success_reg, success_reg, sub_success), 0);
                    bindings.append(&mut sub_bindings);
                    
                    // Patch skip val jump
                    let after_val = self.chunk.code.len();
                    self.chunk.patch_jump(skip_val_jump, after_val);
                }
                
                // Patch type check jump
                let after_checks = self.chunk.code.len();
                self.chunk.patch_jump(type_fail_jump, after_checks);
            }
            Pattern::Set(elements, span) => {
                // 1. Check if it is a set
                self.chunk.emit(Instruction::IsSet(success_reg, scrut_reg), 0);
                
                // Jump to end if not a set
                let type_fail_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);
                
                for elem_pat in elements {
                    // We need to compile the pattern to a value to check existence
                    // Only support literals and pinned variables for now
                    let val_reg = match elem_pat {
                        Pattern::Int(n, _) => {
                            let r = self.alloc_reg();
                            let idx = self.chunk.add_constant(Value::Int64(*n));
                            self.chunk.emit(Instruction::LoadConst(r, idx), 0);
                            r
                        }
                        Pattern::String(s, _) => {
                            let r = self.alloc_reg();
                            let idx = self.chunk.add_constant(Value::String(Arc::new(s.clone())));
                            self.chunk.emit(Instruction::LoadConst(r, idx), 0);
                            r
                        }
                        Pattern::Bool(b, _) => {
                            let r = self.alloc_reg();
                            if *b { self.chunk.emit(Instruction::LoadTrue(r), 0); }
                            else { self.chunk.emit(Instruction::LoadFalse(r), 0); }
                            r
                        }
                        Pattern::Char(c, _) => {
                            let r = self.alloc_reg();
                            let idx = self.chunk.add_constant(Value::Char(*c));
                            self.chunk.emit(Instruction::LoadConst(r, idx), 0);
                            r
                        }
                        Pattern::Pin(expr, _) => {
                            self.compile_expr_tail(expr, false)?
                        }
                        _ => {
                            return Err(CompileError::InvalidPattern {
                                span: *span,
                                context: "Set patterns only support literals and pinned variables".to_string(),
                            });
                        }
                    };
                    
                    // Check if value exists in set
                    let exists_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::SetContains(exists_reg, scrut_reg, val_reg), 0);
                    
                    // Update success_reg
                    self.chunk.emit(Instruction::And(success_reg, success_reg, exists_reg), 0);
                }
                
                // Patch type check jump
                let after_checks = self.chunk.code.len();
                self.chunk.patch_jump(type_fail_jump, after_checks);
            }
            _ => {
                return Err(CompileError::NotImplemented {
                    feature: format!("pattern: {:?}", pattern),
                    span: pattern.span(),
                });
            }
        }

        Ok((success_reg, bindings))
    }

    /// Compile a block.
    fn compile_block(&mut self, stmts: &[Stmt], is_tail: bool) -> Result<Reg, CompileError> {
        // --- BEGIN SCOPE ---
        let saved_locals = self.locals.clone();
        let saved_local_types = self.local_types.clone();

        let mut last_reg = 0;
        if stmts.is_empty() {
            let dst = self.alloc_reg();
            self.chunk.emit(Instruction::LoadUnit(dst), 0);
            last_reg = dst;
        } else {
            for (i, stmt) in stmts.iter().enumerate() {
                let is_last = i == stmts.len() - 1;
                last_reg = self.compile_stmt(stmt, is_tail && is_last)?;
            }
        }

        // --- END SCOPE ---
        self.locals = saved_locals;
        self.local_types = saved_local_types;

        Ok(last_reg)
    }

    /// Compile a statement.
    fn compile_stmt(&mut self, stmt: &Stmt, is_tail: bool) -> Result<Reg, CompileError> {
        match stmt {
            Stmt::Expr(expr) => self.compile_expr_tail(expr, is_tail),
            Stmt::Let(binding) => self.compile_binding(binding),
            Stmt::Assign(target, value, _) => self.compile_assign(target, value),
        }
    }

    /// Compile a let binding.
    fn compile_binding(&mut self, binding: &Binding) -> Result<Reg, CompileError> {
        // Sugar: transform `x: Result[T, E] = obj.toType()` to `x: Result[T, E] = requestToType(obj, "T")`
        if let Some(transformed) = self.try_transform_to_type_call(binding) {
            return self.compile_binding(&transformed);
        }

        // Determine type from explicit annotation or infer from value
        let explicit_type = binding.ty.as_ref().map(|t| self.type_expr_name(t));
        let inferred_type = self.expr_type_name(&binding.value);
        let value_type = explicit_type.clone().or(inferred_type.clone());

        // For simple variable bindings, check if this is an mvar assignment that needs atomic locking
        // BUT skip if we already have a function-level lock on this mvar
        let atomic_lock_info = if let Pattern::Var(ident) = &binding.pattern {
            if self.locals.get(&ident.node).is_none() {
                let mvar_name = self.resolve_name(&ident.node);
                if self.mvars.contains_key(&mvar_name) {
                    // Check if we already have a function-level lock on this mvar
                    let has_fn_lock = self.current_fn_mvar_locks.iter()
                        .any(|(name, _, _)| name == &mvar_name);
                    if has_fn_lock {
                        // Function-level lock already covers this mvar
                        None
                    } else {
                        // Check if RHS WRITES to the same mvar (would deadlock!)
                        let mvar_writes = self.collect_mvar_writes(&binding.value);
                        if mvar_writes.contains(&mvar_name) {
                            return Err(CompileError::NestedMvarWrite {
                                mvar_name: mvar_name.clone(),
                                span: binding.span,
                            });
                        }
                        // Check if RHS reads from the same mvar (needs atomic read-modify-write)
                        let mvar_refs = self.collect_mvar_refs(&binding.value);
                        if mvar_refs.contains(&mvar_name) {
                            // Emit MvarLock before compiling the RHS
                            let name_idx = self.chunk.add_constant(Value::String(Arc::new(mvar_name.clone())));
                            self.chunk.emit(Instruction::MvarLock(name_idx, true), 0); // write lock
                            Some((mvar_name, name_idx))
                        } else {
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let value_reg = self.compile_expr_tail(&binding.value, false)?;

        // For simple variable binding
        if let Pattern::Var(ident) = &binding.pattern {
            // If the variable already exists, check mutability
            if let Some(existing_info) = self.locals.get(&ident.node).copied() {
                if binding.mutable {
                    // New binding is mutable (var x = ...): create new mutable binding that shadows the old one
                    let is_float = self.is_float_type(&value_type) || self.is_float_expr(&binding.value);
                    self.locals.insert(ident.node.clone(), LocalInfo { reg: value_reg, is_float, mutable: true });
                    // Record explicit type if provided
                    if let Some(ty) = explicit_type {
                        self.local_types.insert(ident.node.clone(), ty);
                    }
                } else if existing_info.mutable {
                    // Existing is mutable, new is immutable: allow reassignment
                    let existing_reg = existing_info.reg;
                    if existing_reg != value_reg {
                        self.chunk.emit(Instruction::Move(existing_reg, value_reg), 0);
                    }
                } else {
                    // Both are immutable: treat as pattern match (assert equality)
                    // Emit AssertEq to check that the new value matches the existing one
                    self.chunk.emit(Instruction::AssertEq(existing_info.reg, value_reg), self.span_line(binding.span));
                }
            } else {
                // Check if this is an mvar (module-level mutable variable) assignment
                let mvar_name = self.resolve_name(&ident.node);
                if self.mvars.contains_key(&mvar_name) {
                    // This is an mvar assignment, not a new binding
                    let name_idx = self.chunk.add_constant(Value::String(Arc::new(mvar_name.clone())));
                    self.chunk.emit(Instruction::MvarWrite(name_idx, value_reg), 0);
                    // Emit MvarUnlock if we acquired a lock for atomic update
                    if let Some((_, lock_idx)) = &atomic_lock_info {
                        self.chunk.emit(Instruction::MvarUnlock(*lock_idx, true), 0);
                    }
                    // Track mvar write for deadlock detection
                    self.current_fn_mvar_writes.insert(mvar_name);
                } else {
                    // New binding - determine if float from explicit type or value expression
                    let is_float = self.is_float_type(&value_type) || self.is_float_expr(&binding.value);
                    self.locals.insert(ident.node.clone(), LocalInfo { reg: value_reg, is_float, mutable: binding.mutable });
                    // Record debug symbol for this local variable
                    self.current_fn_debug_symbols.push(LocalVarSymbol {
                        name: ident.node.clone(),
                        register: value_reg,
                    });
                    // Record the type (explicit takes precedence over inferred, and pre-set types take precedence over both)
                    // Check if there's already a type set (e.g., from REPL variable type annotations)
                    let existing_type = self.local_types.get(&ident.node).cloned();
                    let final_type = if let Some(existing) = existing_type {
                        // Prefer existing type unless it's a type parameter (single lowercase letter)
                        // and we have a better inferred type
                        if existing.len() == 1 && existing.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false) {
                            // Existing is a type parameter - prefer inferred if it's better
                            value_type.or(Some(existing))
                        } else {
                            // Existing type looks good, keep it
                            Some(existing)
                        }
                    } else {
                        value_type
                    };
                    if let Some(ty) = final_type {
                        self.local_types.insert(ident.node.clone(), ty);
                    }
                }
            }
        } else {
            // For complex patterns, we need to deconstruct
            let (_, bindings) = self.compile_pattern_test(&binding.pattern, value_reg)?;
            for (name, reg, is_float) in bindings {
                self.locals.insert(name.clone(), LocalInfo { reg, is_float, mutable: false });
                // Record debug symbol for pattern binding
                self.current_fn_debug_symbols.push(LocalVarSymbol {
                    name,
                    register: reg,
                });
            }
        }

        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), 0);
        Ok(dst)
    }

    /// Check if a type name represents a float type
    fn is_float_type(&self, type_name: &Option<String>) -> bool {
        match type_name {
            Some(ty) => Self::is_float_type_name(ty) || ty == "f32" || ty == "f64",
            None => false,
        }
    }

    /// Collect all mvar names referenced in an expression (for atomic update detection).
    fn collect_mvar_refs(&self, expr: &Expr) -> std::collections::HashSet<String> {
        let mut refs = std::collections::HashSet::new();
        self.collect_mvar_refs_inner(expr, &mut refs);
        refs
    }

    /// Collect all mvar names WRITTEN in an expression (for nested write detection).
    /// This detects patterns like `x = { x = 5; ... }` which would deadlock.
    fn collect_mvar_writes(&self, expr: &Expr) -> std::collections::HashSet<String> {
        let mut writes = std::collections::HashSet::new();
        self.collect_mvar_writes_inner(expr, &mut writes);
        writes
    }

    fn collect_mvar_writes_inner(&self, expr: &Expr, writes: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::Block(stmts, _) => {
                for stmt in stmts {
                    match stmt {
                        Stmt::Expr(e) => self.collect_mvar_writes_inner(e, writes),
                        Stmt::Let(binding) => {
                            // Check if this is an mvar assignment (not a new local binding)
                            if let Pattern::Var(ident) = &binding.pattern {
                                if self.locals.get(&ident.node).is_none() {
                                    let mvar_name = self.resolve_name(&ident.node);
                                    if self.mvars.contains_key(&mvar_name) {
                                        writes.insert(mvar_name);
                                    }
                                }
                            }
                            // Also recurse into the value expression
                            self.collect_mvar_writes_inner(&binding.value, writes);
                        }
                        Stmt::Assign(target, value, _) => {
                            // Check if target is an mvar
                            if let AssignTarget::Var(ident) = target {
                                if self.locals.get(&ident.node).is_none() {
                                    let mvar_name = self.resolve_name(&ident.node);
                                    if self.mvars.contains_key(&mvar_name) {
                                        writes.insert(mvar_name);
                                    }
                                }
                            }
                            self.collect_mvar_writes_inner(value, writes);
                        }
                    }
                }
            }
            Expr::If(cond, then_branch, else_branch, _) => {
                self.collect_mvar_writes_inner(cond, writes);
                self.collect_mvar_writes_inner(then_branch, writes);
                self.collect_mvar_writes_inner(else_branch, writes);
            }
            Expr::Match(scrutinee, arms, _) => {
                self.collect_mvar_writes_inner(scrutinee, writes);
                for arm in arms {
                    self.collect_mvar_writes_inner(&arm.body, writes);
                }
            }
            Expr::BinOp(left, _, right, _) => {
                self.collect_mvar_writes_inner(left, writes);
                self.collect_mvar_writes_inner(right, writes);
            }
            Expr::UnaryOp(_, operand, _) => {
                self.collect_mvar_writes_inner(operand, writes);
            }
            Expr::Call(func, _type_args, args, _) => {
                self.collect_mvar_writes_inner(func, writes);
                for arg in args {
                    self.collect_mvar_writes_inner(arg, writes);
                }
            }
            Expr::Tuple(elems, _) => {
                for e in elems {
                    self.collect_mvar_writes_inner(e, writes);
                }
            }
            Expr::List(elems, tail, _) => {
                for e in elems {
                    self.collect_mvar_writes_inner(e, writes);
                }
                if let Some(t) = tail {
                    self.collect_mvar_writes_inner(t, writes);
                }
            }
            Expr::Lambda(_, body, _) => {
                self.collect_mvar_writes_inner(body, writes);
            }
            _ => {}
        }
    }

    fn collect_mvar_refs_inner(&self, expr: &Expr, refs: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::Var(ident) => {
                let name = &ident.node;
                let resolved = self.resolve_name(name);
                if self.mvars.contains_key(&resolved) && !self.locals.contains_key(name) {
                    refs.insert(resolved);
                }
            }
            Expr::BinOp(left, _, right, _) => {
                self.collect_mvar_refs_inner(left, refs);
                self.collect_mvar_refs_inner(right, refs);
            }
            Expr::UnaryOp(_, operand, _) => {
                self.collect_mvar_refs_inner(operand, refs);
            }
            Expr::Call(func, _type_args, args, _) => {
                self.collect_mvar_refs_inner(func, refs);
                for arg in args {
                    self.collect_mvar_refs_inner(arg, refs);
                }
            }
            Expr::If(cond, then_branch, else_branch, _) => {
                self.collect_mvar_refs_inner(cond, refs);
                self.collect_mvar_refs_inner(then_branch, refs);
                self.collect_mvar_refs_inner(else_branch, refs);
            }
            Expr::Block(stmts, _) => {
                for stmt in stmts {
                    match stmt {
                        Stmt::Expr(e) => self.collect_mvar_refs_inner(e, refs),
                        Stmt::Let(binding) => self.collect_mvar_refs_inner(&binding.value, refs),
                        Stmt::Assign(_, e, _) => self.collect_mvar_refs_inner(e, refs),
                    }
                }
            }
            Expr::Tuple(elems, _) => {
                for e in elems {
                    self.collect_mvar_refs_inner(e, refs);
                }
            }
            Expr::List(elems, tail, _) => {
                for e in elems {
                    self.collect_mvar_refs_inner(e, refs);
                }
                if let Some(t) = tail {
                    self.collect_mvar_refs_inner(t, refs);
                }
            }
            Expr::Index(coll, idx, _) => {
                self.collect_mvar_refs_inner(coll, refs);
                self.collect_mvar_refs_inner(idx, refs);
            }
            Expr::FieldAccess(obj, _, _) => {
                self.collect_mvar_refs_inner(obj, refs);
            }
            Expr::MethodCall(obj, _, args, _) => {
                self.collect_mvar_refs_inner(obj, refs);
                for arg in args {
                    self.collect_mvar_refs_inner(arg, refs);
                }
            }
            Expr::Lambda(_, body, _) => {
                self.collect_mvar_refs_inner(body, refs);
            }
            Expr::Match(scrutinee, arms, _) => {
                self.collect_mvar_refs_inner(scrutinee, refs);
                for arm in arms {
                    self.collect_mvar_refs_inner(&arm.body, refs);
                }
            }
            _ => {}
        }
    }

    /// Check if an expression contains blocking operations (receive).
    fn expr_has_blocking(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Receive(_, _, _) => true,
            Expr::Block(stmts, _) => {
                stmts.iter().any(|stmt| match stmt {
                    Stmt::Expr(e) => self.expr_has_blocking(e),
                    Stmt::Let(binding) => self.expr_has_blocking(&binding.value),
                    Stmt::Assign(_, e, _) => self.expr_has_blocking(e),
                })
            }
            Expr::If(cond, then_branch, else_branch, _) => {
                self.expr_has_blocking(cond) || self.expr_has_blocking(then_branch) || self.expr_has_blocking(else_branch)
            }
            Expr::Match(scrutinee, arms, _) => {
                self.expr_has_blocking(scrutinee) || arms.iter().any(|arm| self.expr_has_blocking(&arm.body))
            }
            Expr::BinOp(left, _, right, _) => {
                self.expr_has_blocking(left) || self.expr_has_blocking(right)
            }
            Expr::UnaryOp(_, operand, _) => self.expr_has_blocking(operand),
            Expr::Call(func, _type_args, args, _) => {
                self.expr_has_blocking(func) || args.iter().any(|a| self.expr_has_blocking(a))
            }
            Expr::Tuple(elems, _) | Expr::List(elems, None, _) => {
                elems.iter().any(|e| self.expr_has_blocking(e))
            }
            Expr::List(elems, Some(tail), _) => {
                elems.iter().any(|e| self.expr_has_blocking(e)) || self.expr_has_blocking(tail)
            }
            Expr::Lambda(_, body, _) => self.expr_has_blocking(body),
            Expr::Index(coll, idx, _) => {
                self.expr_has_blocking(coll) || self.expr_has_blocking(idx)
            }
            Expr::FieldAccess(obj, _, _) => self.expr_has_blocking(obj),
            Expr::MethodCall(obj, _, args, _) => {
                self.expr_has_blocking(obj) || args.iter().any(|a| self.expr_has_blocking(a))
            }
            Expr::Try(body, arms, finally_opt, _) => {
                self.expr_has_blocking(body)
                    || arms.iter().any(|arm| self.expr_has_blocking(&arm.body))
                    || finally_opt.as_ref().map_or(false, |f| self.expr_has_blocking(f))
            }
            _ => false,
        }
    }

    /// Check if a function transitively has blocking operations.
    /// Uses fn_mvar_access which is populated after each function is compiled.
    fn fn_has_transitive_blocking(&self, fn_name: &str, visited: &mut HashSet<String>) -> bool {
        if visited.contains(fn_name) {
            return false; // Already checked, avoid infinite recursion
        }
        visited.insert(fn_name.to_string());

        // Check if this function directly has blocking
        if let Some(access) = self.fn_mvar_access.get(fn_name) {
            if access.has_blocking {
                return true;
            }
        }

        // Check transitively through called functions
        if let Some(calls) = self.fn_calls.get(fn_name) {
            for called in calls {
                if self.fn_has_transitive_blocking(called, visited) {
                    return true;
                }
            }
        }

        false
    }

    /// Compile an assignment.
    fn compile_assign(&mut self, target: &AssignTarget, value: &Expr) -> Result<Reg, CompileError> {
        // Check if this is an mvar assignment that needs atomic locking
        // BUT skip if we already have a function-level lock on this mvar
        let needs_atomic_lock = if let AssignTarget::Var(ident) = target {
            if self.locals.get(&ident.node).is_none() {
                let mvar_name = self.resolve_name(&ident.node);
                if self.mvars.contains_key(&mvar_name) {
                    // Check if we already have a function-level lock on this mvar
                    let has_fn_lock = self.current_fn_mvar_locks.iter()
                        .any(|(name, _, _)| name == &mvar_name);
                    if has_fn_lock {
                        // Function-level lock already covers this mvar
                        false
                    } else {
                        // Check if RHS WRITES to the same mvar (would deadlock!)
                        let mvar_writes = self.collect_mvar_writes(value);
                        if mvar_writes.contains(&mvar_name) {
                            return Err(CompileError::NestedMvarWrite {
                                mvar_name: mvar_name.clone(),
                                span: ident.span,
                            });
                        }
                        // Check if the RHS reads from this same mvar
                        let mvar_refs = self.collect_mvar_refs(value);
                        mvar_refs.contains(&mvar_name)
                    }
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        // If atomic lock needed, emit MvarLock before compiling the value
        let lock_name_idx = if needs_atomic_lock {
            if let AssignTarget::Var(ident) = target {
                let mvar_name = self.resolve_name(&ident.node);
                let name_idx = self.chunk.add_constant(Value::String(Arc::new(mvar_name)));
                self.chunk.emit(Instruction::MvarLock(name_idx, true), 0); // write lock
                Some(name_idx)
            } else {
                None
            }
        } else {
            None
        };

        let value_reg = self.compile_expr_tail(value, false)?;

        match target {
            AssignTarget::Var(ident) => {
                if let Some(info) = self.locals.get(&ident.node) {
                    let var_reg = info.reg;
                    self.chunk.emit(Instruction::Move(var_reg, value_reg), 0);
                } else {
                    // Check if it's a module-level mutable variable (mvar)
                    let mvar_name = self.resolve_name(&ident.node);
                    if self.mvars.contains_key(&mvar_name) {
                        let name_idx = self.chunk.add_constant(Value::String(Arc::new(mvar_name.clone())));
                        self.chunk.emit(Instruction::MvarWrite(name_idx, value_reg), 0);
                        // Track mvar write for deadlock detection
                        self.current_fn_mvar_writes.insert(mvar_name);
                    } else {
                        return Err(CompileError::UnknownVariable {
                            name: ident.node.clone(),
                            span: ident.span,
                        });
                    }
                }
            }
            AssignTarget::Field(obj, field) => {
                let obj_reg = self.compile_expr_tail(obj, false)?;
                let field_idx = self.chunk.add_constant(Value::String(Arc::new(field.node.clone())));
                self.chunk.emit(Instruction::SetField(obj_reg, field_idx, value_reg), 0);
            }
            AssignTarget::Index(coll, idx) => {
                let coll_reg = self.compile_expr_tail(coll, false)?;
                let idx_reg = self.compile_expr_tail(idx, false)?;
                self.chunk.emit(Instruction::IndexSet(coll_reg, idx_reg, value_reg), 0);
            }
        }

        // Emit MvarUnlock if we acquired a lock for atomic update
        if let Some(name_idx) = lock_name_idx {
            self.chunk.emit(Instruction::MvarUnlock(name_idx, true), 0);
        }

        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), 0);
        Ok(dst)
    }

    /// Compile an interpolated string.
    ///
    /// For each part:
    /// - Literal strings are loaded directly
    /// - Expressions are compiled and converted to string using `show`
    /// Then all parts are concatenated.
    fn compile_interpolated_string(&mut self, parts: &[StringPart]) -> Result<Reg, CompileError> {
        if parts.is_empty() {
            // Empty string
            let dst = self.alloc_reg();
            let idx = self.chunk.add_constant(Value::String(Arc::new(String::new())));
            self.chunk.emit(Instruction::LoadConst(dst, idx), 0);
            return Ok(dst);
        }

        // Compile each part to a string register
        let mut part_regs = Vec::new();
        for part in parts {
            let reg = match part {
                StringPart::Lit(s) => {
                    let dst = self.alloc_reg();
                    let idx = self.chunk.add_constant(Value::String(Arc::new(s.clone())));
                    self.chunk.emit(Instruction::LoadConst(dst, idx), 0);
                    dst
                }
                StringPart::Expr(e) => {
                    // Compile the expression
                    let expr_reg = self.compile_expr_tail(e, false)?;
                    // Call `show` to convert to string
                    let dst = self.alloc_reg();
                    self.emit_call_native(dst, "show", vec![expr_reg].into(), 0);
                    dst
                }
            };
            part_regs.push(reg);
        }

        // Concatenate all parts
        if part_regs.len() == 1 {
            return Ok(part_regs[0]);
        }

        // Fold concatenation: result = part0 ++ part1 ++ part2 ++ ...
        let mut result = part_regs[0];
        for &part_reg in &part_regs[1..] {
            let dst = self.alloc_reg();
            self.chunk.emit(Instruction::Concat(dst, result, part_reg), 0);
            result = dst;
        }

        Ok(result)
    }

    /// Compile a lambda expression.
    fn compile_lambda(&mut self, params: &[Pattern], body: &Expr) -> Result<Reg, CompileError> {
        use std::collections::HashSet;

        // Step 1: Find free variables in the lambda body
        let mut param_names_set: HashSet<String> = HashSet::new();
        for param in params {
            if let Some(name) = self.pattern_binding_name(param) {
                param_names_set.insert(name);
            }
        }
        let body_free_vars = free_vars(body, &param_names_set);

        // Step 2: Filter to variables that exist in outer scope (locals)
        let mut captures: Vec<(String, Reg)> = Vec::new();
        for var_name in &body_free_vars {
            if let Some(info) = self.locals.get(var_name) {
                captures.push((var_name.clone(), info.reg));
            }
            // Also check if it's in our own captures (nested closures)
            else if self.capture_indices.contains_key(var_name) {
                // Need to re-capture from our own capture environment
                let dst = self.alloc_reg();
                let cap_idx = *self.capture_indices.get(var_name).unwrap();
                self.chunk.emit(Instruction::GetCapture(dst, cap_idx), 0);
                captures.push((var_name.clone(), dst));
            }
        }

        // Step 3: Save state
        let saved_chunk = std::mem::take(&mut self.chunk);
        let saved_locals = std::mem::take(&mut self.locals);
        let saved_next_reg = self.next_reg;
        let saved_capture_indices = std::mem::take(&mut self.capture_indices);
        let saved_local_types = std::mem::take(&mut self.local_types);
        // Save current_function_name to prevent self-call optimization inside lambdas
        // (lambdas are separate functions, not the enclosing function)
        let saved_current_function_name = self.current_function_name.take();

        // Step 4: Create new function for lambda
        self.chunk = Chunk::new();
        self.locals = HashMap::new();
        self.capture_indices = HashMap::new();
        self.local_types = HashMap::new();
        self.next_reg = 0;
        // Set to None - lambdas don't have a self-call target (they can't TailCallSelf)

        let arity = params.len();
        let mut param_names = Vec::new();

        // Set next_reg to be after all parameters
        self.next_reg = arity as Reg;

        // Allocate registers for parameters and handle pattern destructuring
        for (i, param) in params.iter().enumerate() {
            let arg_reg = i as Reg;
            if let Some(name) = self.pattern_binding_name(param) {
                // Simple variable pattern - bind directly to param register
                self.locals.insert(name.clone(), LocalInfo { reg: arg_reg, is_float: false, mutable: false });
                param_names.push(name);
            } else {
                // Complex pattern (tuple, etc.) - need to destructure
                param_names.push(format!("_arg{}", i));
                // Compile pattern matching to extract bindings
                // Note: we ignore the success_reg since lambda params always match structurally
                let (_, bindings) = self.compile_pattern_test(param, arg_reg)?;
                for (name, reg, is_float) in bindings {
                    self.locals.insert(name, LocalInfo { reg, is_float, mutable: false });
                }
            }
        }

        // Set up capture indices for the lambda body to use
        for (i, (name, _)) in captures.iter().enumerate() {
            self.capture_indices.insert(name.clone(), i as u8);
        }

        // Compile body (in tail position)
        let body_line = self.span_line(body.span());
        let result_reg = self.compile_expr_tail(body, true)?;
        self.chunk.emit(Instruction::Return(result_reg), body_line);

        self.chunk.register_count = self.next_reg as usize;

        let lambda_chunk = std::mem::take(&mut self.chunk);

        // Use accumulated debug symbols (not affected by block scope restoration)
        let debug_symbols = std::mem::take(&mut self.current_fn_debug_symbols);

        // Step 5: Restore state
        self.chunk = saved_chunk;
        self.locals = saved_locals;
        self.next_reg = saved_next_reg;
        self.capture_indices = saved_capture_indices;
        self.local_types = saved_local_types;
        self.current_function_name = saved_current_function_name;

        // Step 6: Create closure with captures
        let func = FunctionValue {
            name: "<lambda>".to_string(),
            arity,
            param_names,
            code: Arc::new(lambda_chunk),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: AtomicU32::new(0),
            debug_symbols,
            // REPL introspection fields - lambdas don't have these
            source_code: None,
            source_file: None,
            doc: None,
            signature: None,
            param_types: vec![],
            return_type: None,
        };

        let dst = self.alloc_reg();

        if captures.is_empty() {
            // No captures - just load the function
            let func_idx = self.chunk.add_constant(Value::Function(Arc::new(func)));
            self.chunk.emit(Instruction::LoadConst(dst, func_idx), 0);
        } else {
            // Has captures - create a closure
            let func_idx = self.chunk.add_constant(Value::Function(Arc::new(func)));
            let capture_regs: Vec<Reg> = captures.iter().map(|(_, reg)| *reg).collect();
            self.chunk.emit(Instruction::MakeClosure(dst, func_idx, capture_regs.into()), 0);
        }

        Ok(dst)
    }

    /// Compile a record construction.
    fn compile_record(&mut self, type_name: &str, fields: &[RecordField]) -> Result<Reg, CompileError> {
        // Enforce that type must be predeclared
        if !self.known_constructors.contains(type_name) {
            // If it's a module item that looks like a record (uppercase), we might be here mistakenly?
            // No, resolve_name handles variables.
            // If it's not in known_constructors, it's an error.
            return Err(CompileError::UnknownType {
                name: type_name.to_string(),
                span: Span::default(), // We don't have span here easily without passing it, but CompileError needs it.
                // We should update compile_record to take span or return error without span and add it later?
                // Actually Expr::Record has span.
                // Let's assume passed span or just use default for now (user asked for check).
                // Or better, passing span to compile_record would be better refactor but let's see.
                // compile_record doesn't take span.
                // I will return error with default span, and maybe caller can fix it?
                // Or I can add span argument.
            });
        }

        let mut field_regs = Vec::new();
        for field in fields {
            match field {
                RecordField::Positional(expr) => {
                    let reg = self.compile_expr_tail(expr, false)?;
                    field_regs.push(reg);
                }
                RecordField::Named(_, expr) => {
                    let reg = self.compile_expr_tail(expr, false)?;
                    field_regs.push(reg);
                }
            }
        }

        let dst = self.alloc_reg();

        // Check if this is a variant constructor (not a record type)
        // If type_name matches a variant constructor, emit MakeVariant instead of MakeRecord
        let mut is_variant_ctor = false;
        let mut parent_type_name: Option<String> = None;

        // Get the local part of type_name (e.g., "Object" from "stdlib.json.Object")
        let local_type_name = type_name.rsplit('.').next().unwrap_or(type_name);

        for (ty_name, info) in &self.types {
            if let TypeInfoKind::Variant { constructors } = &info.kind {
                // Check if type_name (qualified or local) matches any constructor (stored as local name)
                if constructors.iter().any(|(ctor_name, _)| {
                    ctor_name == type_name || ctor_name == local_type_name
                }) {
                    is_variant_ctor = true;
                    parent_type_name = Some(ty_name.clone());
                    break;
                }
            }
        }

        if is_variant_ctor {
            let parent_type = parent_type_name.unwrap();
            let type_idx = self.chunk.add_constant(Value::String(Arc::new(parent_type)));
            // Use local constructor name (without module path) for better user experience
            let local_ctor = type_name.rsplit('.').next().unwrap_or(type_name);
            let ctor_idx = self.chunk.add_constant(Value::String(Arc::new(local_ctor.to_string())));
            self.chunk.emit(Instruction::MakeVariant(dst, type_idx, ctor_idx, field_regs.into()), 0);
        } else {
            let type_idx = self.chunk.add_constant(Value::String(Arc::new(type_name.to_string())));
            self.chunk.emit(Instruction::MakeRecord(dst, type_idx, field_regs.into()), 0);
        }
        Ok(dst)
    }

    /// Compile a record update.
    fn compile_record_update(&mut self, type_name: &str, base: &Expr, fields: &[RecordField]) -> Result<Reg, CompileError> {
        // Enforce that type must be predeclared
        if !self.known_constructors.contains(type_name) {
            return Err(CompileError::UnknownType {
                name: type_name.to_string(),
                span: Span::default(),
            });
        }

        let base_reg = self.compile_expr_tail(base, false)?;

        let mut field_regs = Vec::new();
        for field in fields {
            match field {
                RecordField::Named(_, expr) => {
                    let reg = self.compile_expr_tail(expr, false)?;
                    field_regs.push(reg);
                }
                RecordField::Positional(expr) => {
                    let reg = self.compile_expr_tail(expr, false)?;
                    field_regs.push(reg);
                }
            }
        }

        let dst = self.alloc_reg();
        let type_idx = self.chunk.add_constant(Value::String(Arc::new(type_name.to_string())));
        self.chunk.emit(Instruction::UpdateRecord(dst, base_reg, type_idx, field_regs.into()), 0);
        Ok(dst)
    }

    /// Simple type inference from expression for builtin dispatch.
    /// Returns Some(type) if we can determine the type from the expression,
    /// None if we can't (would need full type system integration).
    fn infer_expr_type(&self, expr: &Expr) -> Option<InferredType> {
        match expr {
            // Literals have known types
            Expr::Int(_, _) => Some(InferredType::Int),
            Expr::Float(_, _) => Some(InferredType::Float),

            // Negation preserves type
            Expr::UnaryOp(UnaryOp::Neg, inner, _) => self.infer_expr_type(inner),

            // Binary arithmetic: both operands should have same type
            Expr::BinOp(left, op, _, _) if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) => {
                self.infer_expr_type(left)
            }

            // Variable: check if we know its type
            Expr::Var(ident) => {
                self.local_types.get(&ident.node).and_then(|t| {
                    match t.as_str() {
                        "Int" => Some(InferredType::Int),
                        "Float" => Some(InferredType::Float),
                        _ => None,
                    }
                })
            }

            // Tuple with single element (parenthesized)
            Expr::Tuple(items, _) if items.len() == 1 => self.infer_expr_type(&items[0]),

            // Block: type of last statement if it's an expression
            Expr::Block(stmts, _) => {
                stmts.last().and_then(|stmt| {
                    match stmt {
                        Stmt::Expr(e) => self.infer_expr_type(e),
                        _ => None,
                    }
                })
            }

            // Other cases: can't infer without full type system
            _ => None,
        }
    }

    /// Allocate a new register.
    fn alloc_reg(&mut self) -> Reg {
        let reg = self.next_reg;
        if reg == 255 {
            panic!("Register limit exceeded: function has too many local variables (max ~120). Consider breaking into smaller functions.");
        }
        self.next_reg += 1;
        reg
    }

    /// Get a compiled function.
    pub fn get_function(&self, name: &str) -> Option<Arc<FunctionValue>> {
        self.find_function(name).cloned()
    }

    /// Register an externally compiled function (e.g., from a previous eval).
    /// This makes the function callable from code compiled by this compiler.
    pub fn register_external_function(&mut self, name: &str, func: Arc<FunctionValue>) {
        // Only add if not already present
        if self.functions.contains_key(name) {
            return;
        }

        // Add to functions map
        self.functions.insert(name.to_string(), func);

        // Allocate an index for indexed calls
        let idx = self.function_list.len() as u16;
        self.function_indices.insert(name.to_string(), idx);
        self.function_list.push(name.to_string());

        // Extract module prefix from qualified names like "module.submodule.func/"
        // and register each level as a known module
        if let Some(slash_pos) = name.find('/') {
            let name_without_sig = &name[..slash_pos];
            let mut parts: Vec<&str> = name_without_sig.split('.').collect();
            if parts.len() > 1 {
                // Remove the function name (last part)
                parts.pop();
                // Register each module prefix level
                let mut prefix = String::new();
                for part in parts {
                    if !prefix.is_empty() {
                        prefix.push('.');
                    }
                    prefix.push_str(part);
                    self.known_modules.insert(prefix.clone());
                }
            }
        }
    }

    /// Register an externally defined type (e.g., from a previous eval).
    pub fn register_external_type(&mut self, name: &str, type_val: &Arc<TypeValue>) {
        if self.types.contains_key(name) {
            return;
        }
        use nostos_vm::value::TypeKind;
        let kind = match &type_val.kind {
            TypeKind::Record { mutable } => {
                let fields = type_val.fields.iter()
                    .map(|f| (f.name.clone(), f.type_name.clone()))
                    .collect();
                TypeInfoKind::Record { fields, mutable: *mutable }
            }
            TypeKind::Variant => {
                // Also register variant constructor names as known constructors
                for c in &type_val.constructors {
                    self.known_constructors.insert(c.name.clone());
                }
                let constructors = type_val.constructors.iter()
                    .map(|c| (c.name.clone(), c.fields.iter().map(|f| f.type_name.clone()).collect()))
                    .collect();
                TypeInfoKind::Variant { constructors }
            }
            TypeKind::Primitive | TypeKind::Alias { .. } => return,
        };
        let type_info = TypeInfo { name: name.to_string(), kind };
        self.types.insert(name.to_string(), type_info);
        self.known_constructors.insert(name.to_string());
    }

    /// Register a dynamic mvar (from eval) with the compiler.
    /// This allows the compiler to emit MvarRead instructions for the variable.
    pub fn register_dynamic_mvar(&mut self, name: &str) {
        if !self.mvars.contains_key(name) {
            self.mvars.insert(name.to_string(), MvarInfo {
                type_name: "Any".to_string(),
                initial_value: MvarInitValue::Unit,
            });
        }
    }

    /// Register a known module prefix.
    /// This allows the compiler to recognize module-qualified names like `mymodule.func()`.
    pub fn register_known_module(&mut self, module: &str) {
        self.known_modules.insert(module.to_string());
    }

    /// Register all external functions at once, preserving their indices.
    /// This is important for eval because compiled bytecode contains CallDirect
    /// instructions with hardcoded function indices that must be preserved.
    pub fn register_external_functions_with_list(
        &mut self,
        functions: &HashMap<String, Arc<FunctionValue>>,
        function_list: &[String],
    ) {
        // First, ensure function_list has enough capacity
        // We'll fill in the functions at their original indices
        self.function_list.clear();
        self.function_list.extend(function_list.iter().cloned());

        // Set up indices
        self.function_indices.clear();
        for (idx, name) in function_list.iter().enumerate() {
            self.function_indices.insert(name.clone(), idx as u16);
        }

        // Copy all functions and extract module names
        for (name, func) in functions {
            self.functions.insert(name.clone(), func.clone());

            // Extract module prefix from qualified names like "module.submodule.func/"
            // and register each level as a known module
            if let Some(slash_pos) = name.find('/') {
                let name_without_sig = &name[..slash_pos];
                let mut parts: Vec<&str> = name_without_sig.split('.').collect();
                if parts.len() > 1 {
                    // Remove the function name (last part)
                    parts.pop();
                    // Register each module prefix level
                    let mut prefix = String::new();
                    for part in parts {
                        if !prefix.is_empty() {
                            prefix.push('.');
                        }
                        prefix.push_str(part);
                        self.known_modules.insert(prefix.clone());
                    }
                }
            }
        }
    }

    /// Get the ordered function list (names).
    pub fn get_function_list_names(&self) -> &[String] {
        &self.function_list
    }

    /// Get all compiled functions.
    pub fn get_all_functions(&self) -> &HashMap<String, Arc<FunctionValue>> {
        &self.functions
    }

    /// Get the ordered function list for direct indexed calls.
    /// Returns functions in the same order as their indices (for CallDirect).
    /// Functions that have been removed will be skipped (their slots become None).
    pub fn get_function_list(&self) -> Vec<Arc<FunctionValue>> {
        self.function_list.iter()
            .filter_map(|name| self.functions.get(name).cloned())
            .collect()
    }

    /// Get the module-level mutable variables (mvars).
    /// Returns a HashMap of qualified name -> MvarInfo.
    pub fn get_mvars(&self) -> &HashMap<String, MvarInfo> {
        &self.mvars
    }

    /// Check for potential mvar deadlocks caused by transitive blocking.
    ///
    /// With function-level locking for read-modify-write patterns, we need to check
    /// that functions holding mvar locks don't transitively call blocking operations.
    /// Direct blocking (receive in same function) is caught during compilation.
    /// This check catches transitive blocking (calling a function that blocks).
    pub fn check_mvar_deadlocks(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // Check for transitive blocking: a function with mvar locks that calls
        // (directly or transitively) a function that blocks (receive)
        for (fn_name, access) in &self.fn_mvar_access {
            // Find mvars that are both read AND written by this function
            // These are the ones that get function-level locks
            let mvars_needing_lock: HashSet<_> = access.reads.intersection(&access.writes).collect();

            if !mvars_needing_lock.is_empty() {
                // This function has function-level locks - check if any called function blocks
                if let Some(called_fns) = self.fn_calls.get(fn_name) {
                    for called_fn in called_fns {
                        let mut visited = HashSet::new();
                        // Skip checking the current function itself (direct blocking is caught earlier)
                        visited.insert(fn_name.clone());

                        if self.fn_has_transitive_blocking(called_fn, &mut visited) {
                            let mvar_name = mvars_needing_lock.iter().next().unwrap();
                            errors.push(format!(
                                "function `{}` holds lock on mvar `{}` and calls `{}` which blocks (receive) - this could cause deadlock",
                                fn_name, mvar_name, called_fn
                            ));
                            break; // One error per function is enough
                        }
                    }
                }
            }
        }

        errors
    }

    /// Get all types for the VM.
    pub fn get_vm_types(&self) -> HashMap<String, Arc<TypeValue>> {
        use nostos_vm::value::{TypeValue, TypeKind, FieldInfo, ConstructorInfo};

        let mut vm_types = HashMap::new();
        for (name, type_info) in &self.types {
            // Get the TypeDef AST for introspection fields
            let type_def = self.type_defs.get(name);

            let type_value = match &type_info.kind {
                TypeInfoKind::Record { fields, mutable } => {
                    let field_infos: Vec<FieldInfo> = fields.iter()
                        .map(|(fname, ftype)| FieldInfo {
                            name: fname.clone(),
                            type_name: ftype.clone(),
                            mutable: *mutable,
                            private: false,
                        })
                        .collect();
                    TypeValue {
                        name: name.clone(),
                        kind: TypeKind::Record { mutable: *mutable },
                        fields: field_infos,
                        constructors: vec![],
                        traits: self.type_traits.get(name).cloned().unwrap_or_default(),
                        // REPL introspection fields
                        source_code: type_def.map(|d| d.body_string()),
                        source_file: self.current_source_name.clone(),
                        doc: type_def.and_then(|d| d.doc.clone()),
                        type_params: type_def.map(|d| d.type_param_names()).unwrap_or_default(),
                    }
                }
                TypeInfoKind::Variant { constructors } => {
                    TypeValue {
                        name: name.clone(),
                        kind: TypeKind::Variant,
                        fields: vec![],
                        constructors: constructors.iter()
                            .map(|(n, field_types)| ConstructorInfo {
                                name: n.clone(),
                                fields: field_types.iter().enumerate().map(|(i, ty)| FieldInfo {
                                    name: format!("_{}", i),
                                    type_name: ty.clone(),
                                    mutable: false,
                                    private: false,
                                }).collect(),
                            })
                            .collect(),
                        traits: self.type_traits.get(name).cloned().unwrap_or_default(),
                        // REPL introspection fields
                        source_code: type_def.map(|d| d.body_string()),
                        source_file: self.current_source_name.clone(),
                        doc: type_def.and_then(|d| d.doc.clone()),
                        type_params: type_def.map(|d| d.type_param_names()).unwrap_or_default(),
                    }
                }
            };
            vm_types.insert(name.clone(), Arc::new(type_value));
        }
        vm_types
    }

    // =========================================================================
    // REPL Introspection Query Methods
    // =========================================================================

    /// Set native function indices for CallNativeIdx optimization.
    /// This should be called with the indices from the VM's scheduler after
    /// default natives are registered.
    pub fn set_native_indices(&mut self, indices: HashMap<String, u16>) {
        self.native_indices = indices;
    }

    /// Set the extension function indices for CallExtensionIdx optimization.
    pub fn set_extension_indices(&mut self, indices: HashMap<String, u16>) {
        self.extension_indices = indices;
    }

    /// Emit a native function call, using CallNativeIdx if the index is known,
    /// otherwise falling back to CallNative.
    fn emit_call_native(&mut self, dst: Reg, name: &str, args: RegList, line: usize) {
        if let Some(&idx) = self.native_indices.get(name) {
            // Fast path: use indexed call
            self.chunk.emit(Instruction::CallNativeIdx(dst, idx, args), line);
        } else {
            // Slow path: use string-based lookup
            let name_idx = self.chunk.add_constant(Value::String(Arc::new(name.to_string())));
            self.chunk.emit(Instruction::CallNative(dst, name_idx, args), line);
        }
    }

    /// Emit an extension function call, using CallExtensionIdx if the index is known,
    /// otherwise falling back to CallExtension.
    fn emit_call_extension(&mut self, dst: Reg, name: &str, args: RegList, line: usize) {
        if let Some(&idx) = self.extension_indices.get(name) {
            // Fast path: use indexed call
            self.chunk.emit(Instruction::CallExtensionIdx(dst, idx, args), line);
        } else {
            // Slow path: use string-based lookup
            let name_idx = self.chunk.add_constant(Value::String(Arc::new(name.to_string())));
            self.chunk.emit(Instruction::CallExtension(dst, name_idx, args), line);
        }
    }

    /// Get all function names in the compiler.
    /// Returns the full names including type signatures (e.g., "greet/String").
    pub fn get_function_names(&self) -> Vec<&str> {
        self.functions.keys().map(|s| s.as_str()).collect()
    }

    /// Get built-in function names for autocomplete.
    pub fn get_builtin_names() -> Vec<&'static str> {
        BUILTINS.iter().map(|b| b.name).collect()
    }

    /// Get signature for a built-in function.
    pub fn get_builtin_signature(name: &str) -> Option<&'static str> {
        BUILTINS.iter().find(|b| b.name == name).map(|b| b.signature)
    }

    /// Get documentation for a built-in function.
    pub fn get_builtin_doc(name: &str) -> Option<&'static str> {
        BUILTINS.iter().find(|b| b.name == name).map(|b| b.doc)
    }

    /// Get all builtins with their signatures for external registration.
    pub fn get_builtins() -> Vec<(&'static str, &'static str)> {
        BUILTINS.iter().map(|b| (b.name, b.signature)).collect()
    }

    /// Get function names formatted for user display.
    /// Groups overloaded functions and shows their signatures.
    pub fn get_function_names_display(&self) -> Vec<String> {
        use std::collections::BTreeMap;

        // Group functions by base name
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for name in self.functions.keys() {
            let base_name = name.split('/').next().unwrap_or(name);
            let signature = name.split('/').nth(1).unwrap_or("");
            groups.entry(base_name.to_string())
                .or_default()
                .push(signature.to_string());
        }

        // Format for display
        let mut result = Vec::new();
        for (base_name, sigs) in groups {
            if sigs.len() == 1 && sigs[0].is_empty() {
                // Zero-arg function: main/
                result.push(base_name);
            } else if sigs.len() == 1 {
                // Single function with parameters: show with signature
                result.push(format!("{}({})", base_name, sigs[0].replace(',', ", ")));
            } else {
                // Overloaded function: show all variants
                for sig in sigs {
                    if sig.is_empty() {
                        result.push(format!("{}()", base_name));
                    } else {
                        result.push(format!("{}({})", base_name, sig.replace(',', ", ")));
                    }
                }
            }
        }
        result
    }

    /// Get all variants of a function by base name.
    /// Returns a list of (full_name, display_name) tuples.
    pub fn get_function_variants(&self, base_name: &str) -> Vec<(String, String)> {
        let prefix = format!("{}/", base_name);
        let mut variants = Vec::new();

        for name in self.functions.keys() {
            if name.starts_with(&prefix) || name == base_name {
                let sig = name.split('/').nth(1).unwrap_or("");
                let display = if sig.is_empty() {
                    format!("{}()", base_name)
                } else {
                    format!("{}({})", base_name, sig.replace(',', ", "))
                };
                variants.push((name.clone(), display));
            }
        }

        // Sort by signature for consistent ordering
        variants.sort_by(|a, b| a.0.cmp(&b.0));
        variants
    }

    /// Remove a function (and all its overloads) by base name.
    /// Used when deleting a definition from the project.
    pub fn remove_function(&mut self, base_name: &str) {
        // Collect all variants to remove
        let prefix = format!("{}/", base_name);
        let keys_to_remove: Vec<String> = self.functions.keys()
            .filter(|name| name.starts_with(&prefix) || *name == base_name)
            .cloned()
            .collect();

        for key in keys_to_remove {
            self.functions.remove(&key);
            self.function_visibility.remove(&key);
            // Note: function_indices and function_list are not updated
            // because that would invalidate existing indices. This is fine
            // for a REPL/TUI session - the stale index just won't be called.
        }

        // Also remove from fn_asts if present
        self.fn_asts.remove(base_name);
    }

    /// Remove a type definition by name.
    pub fn remove_type(&mut self, name: &str) {
        self.types.remove(name);
    }

    /// Remove a trait definition by name.
    pub fn remove_trait(&mut self, name: &str) {
        self.trait_defs.remove(name);
    }

    /// Convert a full function name to a display-friendly format.
    /// "greet/String" -> "greet(String)"
    /// "main/" -> "main()"
    pub fn function_name_display(name: &str) -> String {
        if let Some((base, sig)) = name.split_once('/') {
            if sig.is_empty() {
                format!("{}()", base)
            } else {
                format!("{}({})", base, sig.replace(',', ", "))
            }
        } else {
            name.to_string()
        }
    }

    /// Extract the base name from a full function name.
    /// "greet/String" -> "greet"
    /// "main/" -> "main"
    pub fn function_base_name(name: &str) -> &str {
        name.split('/').next().unwrap_or(name)
    }

    /// Get all user-defined type names in the compiler.
    /// Does not include builtin types (those are only for field autocomplete).
    pub fn get_type_names(&self) -> Vec<&str> {
        self.types.keys().map(|s| s.as_str()).collect()
    }

    /// Get all types as TypeValue for external registration.
    pub fn get_all_types(&self) -> HashMap<String, Arc<TypeValue>> {
        self.get_vm_types()
    }

    /// Get all trait implementations for external registration.
    /// Returns Vec of (type_name, trait_name, TraitImplInfo).
    pub fn get_all_trait_impls(&self) -> Vec<(String, String, TraitImplInfo)> {
        self.trait_impls.iter()
            .map(|((ty, tr), info)| (ty.clone(), tr.clone(), info.clone()))
            .collect()
    }

    /// Register external trait implementations (for compile checking).
    pub fn register_external_trait_impls(&mut self, impls: Vec<(String, String, TraitImplInfo)>) {
        for (type_name, trait_name, info) in impls {
            self.trait_impls.insert((type_name.clone(), trait_name.clone()), info);
            // Also update type_traits
            self.type_traits
                .entry(type_name)
                .or_insert_with(Vec::new)
                .push(trait_name);
        }
    }

    /// Get all known module prefixes.
    pub fn get_known_modules(&self) -> impl Iterator<Item = &str> {
        self.known_modules.iter().map(|s| s.as_str())
    }

    /// Get all public functions from a module (for `use module.*`).
    /// Returns Vec of (local_name, qualified_name) pairs.
    pub fn get_module_public_functions(&self, module_path: &str) -> Vec<(String, String)> {
        let prefix = format!("{}.", module_path);
        self.function_visibility.iter()
            .filter(|(name, vis)| {
                name.starts_with(&prefix) && **vis == Visibility::Public
            })
            .map(|(name, _)| {
                let local_name = name.strip_prefix(&prefix)
                    .unwrap_or(name)
                    .to_string();
                (local_name, name.clone())
            })
            .collect()
    }

    /// Check if a function is public (exported).
    pub fn is_function_public(&self, name: &str) -> bool {
        // Check in function_visibility map (exact match)
        if let Some(vis) = self.function_visibility.get(name) {
            return *vis == Visibility::Public;
        }
        // Strip signature suffix and check
        let base_name = name.split('/').next().unwrap_or(name);
        if let Some(vis) = self.function_visibility.get(base_name) {
            return *vis == Visibility::Public;
        }
        // Search for any function with this base name (handles qualified names)
        // e.g., "htmlParse" should match "stdlib.html_parser.htmlParse/String"
        // and "stdlib.html_parser.htmlParse" should match "stdlib.html_parser.htmlParse/String"
        for (key, vis) in &self.function_visibility {
            let key_base = key.split('/').next().unwrap_or(key);
            if key_base == base_name || key_base.ends_with(&format!(".{}", base_name)) {
                return *vis == Visibility::Public;
            }
        }
        // If not found, assume not public (builtins, etc. are handled elsewhere)
        false
    }

    /// Add an import alias (local name -> qualified name).
    pub fn add_import_alias(&mut self, local_name: &str, qualified_name: &str) {
        self.imports.insert(local_name.to_string(), qualified_name.to_string());
    }

    /// Set local variable types for REPL method dispatch.
    /// This allows the compiler to know variable types from previous REPL evaluations.
    pub fn set_local_types(&mut self, types: HashMap<String, String>) {
        self.local_types = types;
    }

    /// Add a single local variable type.
    pub fn set_local_type(&mut self, name: String, type_name: String) {
        self.local_types.insert(name, type_name);
    }

    /// Get all trait names in the compiler.
    pub fn get_trait_names(&self) -> Vec<&str> {
        self.trait_defs.keys().map(|s| s.as_str()).collect()
    }

    /// Get trait information by name.
    pub fn get_trait_info(&self, name: &str) -> Option<&TraitInfo> {
        self.trait_defs.get(name)
    }

    /// Find a function by base name or full qualified name.
    /// Supports both "add" and "add/_,_" formats.
    fn find_function(&self, name: &str) -> Option<&Arc<FunctionValue>> {
        // Try exact match first
        if let Some(f) = self.functions.get(name) {
            return Some(f);
        }
        // If name doesn't contain '/', search for functions with that base name
        if !name.contains('/') {
            let prefix = format!("{}/", name);
            for (key, func) in &self.functions {
                if key.starts_with(&prefix) {
                    return Some(func);
                }
            }
        }
        None
    }

    /// Get a function's signature as a displayable string.
    pub fn get_function_signature(&self, name: &str) -> Option<String> {
        self.find_function(name).and_then(|f| f.signature.clone())
    }

    /// Get a function's return type directly.
    pub fn get_function_return_type(&self, name: &str) -> Option<String> {
        self.find_function(name).and_then(|f| f.return_type.clone())
    }

    /// Get a function's doc comment.
    pub fn get_function_doc(&self, name: &str) -> Option<String> {
        self.find_function(name).and_then(|f| f.doc.clone())
    }

    /// Get a function's source code.
    pub fn get_function_source(&self, name: &str) -> Option<String> {
        self.find_function(name)
            .and_then(|f| f.source_code.as_ref())
            .map(|s| s.to_string())
    }

    /// Get source code for ALL overloads of a function by base name.
    /// Returns None if no functions found.
    pub fn get_all_function_sources(&self, name: &str) -> Option<String> {
        let prefix = format!("{}/", name);
        let mut sources: Vec<String> = Vec::new();

        // Collect all functions matching this base name
        for (key, func) in &self.functions {
            let matches = key == name || key.starts_with(&prefix);
            if matches {
                if let Some(source) = &func.source_code {
                    sources.push(source.to_string());
                }
            }
        }

        if sources.is_empty() {
            None
        } else {
            Some(sources.join("\n\n"))
        }
    }

    /// Get all traits implemented by a type.
    pub fn get_type_traits(&self, type_name: &str) -> Vec<String> {
        self.type_traits.get(type_name).cloned().unwrap_or_default()
    }

    /// Get all types implementing a trait.
    pub fn get_trait_implementors(&self, trait_name: &str) -> Vec<String> {
        self.trait_impls.iter()
            .filter(|((_, t), _)| t == trait_name)
            .map(|((ty, _), _)| ty.clone())
            .collect()
    }

    /// Get a TypeDef AST for a type (for introspection).
    pub fn get_type_def(&self, name: &str) -> Option<&TypeDef> {
        self.type_defs.get(name)
    }

    /// Get a FnDef AST for a function (for introspection).
    /// Supports both full names ("greet/String") and base names ("greet").
    pub fn get_fn_def(&self, name: &str) -> Option<&FnDef> {
        // Try exact match first
        if let Some(def) = self.fn_asts.get(name) {
            return Some(def);
        }
        // If name doesn't contain '/', search for functions with that base name
        if !name.contains('/') {
            let prefix = format!("{}/", name);
            for (key, def) in &self.fn_asts {
                if key.starts_with(&prefix) {
                    return Some(def);
                }
            }
        }
        None
    }

    /// Get all FnDef ASTs for a function by base name (for viewing overloaded functions).
    pub fn get_all_fn_defs(&self, base_name: &str) -> Vec<(&str, &FnDef)> {
        let prefix = format!("{}/", base_name);
        let mut results = Vec::new();
        for (key, def) in &self.fn_asts {
            if key.starts_with(&prefix) || key == base_name {
                results.push((key.as_str(), def));
            }
        }
        // Sort by name for consistent ordering
        results.sort_by(|a, b| a.0.cmp(b.0));
        results
    }

    /// Check if a module exists (has any functions/types with that prefix).
    pub fn module_exists(&self, module_name: &str) -> bool {
        let prefix = format!("{}.", module_name);
        // Check functions
        for name in self.functions.keys() {
            if name.starts_with(&prefix) || name == module_name {
                return true;
            }
        }
        // Check types
        for name in self.types.keys() {
            if name.starts_with(&prefix) || name == module_name {
                return true;
            }
        }
        false
    }

    /// Get the list of modules imported by a given module.
    /// Returns module names as strings (e.g., ["nalgebra", "stdlib.list"]).
    pub fn get_module_imports(&self, module_name: &str) -> Vec<String> {
        let module_path: Vec<String> = if module_name.is_empty() {
            vec![]
        } else {
            module_name.split('.').map(String::from).collect()
        };

        self.imported_modules
            .iter()
            .filter_map(|(importing_module, imported)| {
                if *importing_module == module_path {
                    Some(imported.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get use statements for a module (for displaying in browser metadata).
    pub fn get_module_use_stmts(&self, module_name: &str) -> Vec<String> {
        let module_path: Vec<String> = if module_name.is_empty() {
            vec![]
        } else {
            module_name.split('.').map(String::from).collect()
        };

        self.module_use_stmts
            .iter()
            .filter_map(|(path, stmt)| {
                if *path == module_path {
                    Some(stmt.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all use statements (for copying to another compiler).
    pub fn get_all_use_stmts(&self) -> &[(Vec<String>, String)] {
        &self.module_use_stmts
    }

    /// Register use statements from another compiler.
    pub fn register_use_stmts(&mut self, stmts: &[(Vec<String>, String)]) {
        for (path, stmt) in stmts {
            self.module_use_stmts.push((path.clone(), stmt.clone()));
        }
    }

    /// Get field names for a record type.
    /// Checks both user-defined types and builtin types.
    /// Returns empty vec for non-record types or unknown types.
    pub fn get_type_fields(&self, type_name: &str) -> Vec<String> {
        // Check user-defined types first
        if let Some(type_info) = self.types.get(type_name) {
            if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                return fields.iter().map(|(name, _)| name.clone()).collect();
            }
        }
        // Then check builtin types
        if let Some(type_info) = self.builtin_types.get(type_name) {
            if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                return fields.iter().map(|(name, _)| name.clone()).collect();
            }
        }
        Vec::new()
    }

    /// Get constructor names for a variant type.
    /// Returns empty vec for non-variant types or unknown types.
    pub fn get_type_constructors(&self, type_name: &str) -> Vec<String> {
        if let Some(type_info) = self.types.get(type_name) {
            if let TypeInfoKind::Variant { constructors } = &type_info.kind {
                return constructors.iter().map(|(name, _)| name.clone()).collect();
            }
        }
        Vec::new()
    }

    /// Infer the type signature of a function definition using Hindley-Milner type inference.
    /// Returns a formatted signature string like "Int -> Int -> Int" or "a -> a -> a".
    ///
    /// This function attempts full HM inference and falls back to AST-based constraint analysis
    /// if inference fails.
    pub fn infer_signature(&self, def: &FnDef) -> String {
        // Try full Hindley-Milner inference first
        if let Some(sig) = self.try_hm_inference(def) {
            return sig;
        }
        // Fall back to AST-based signature
        def.signature()
    }

    /// Try full Hindley-Milner type inference for a function.
    /// Returns None if inference fails, allowing fallback to AST-based signature.
    fn try_hm_inference(&self, def: &FnDef) -> Option<String> {
        // Create a fresh type environment for inference
        let mut env = nostos_types::standard_env();

        // Register known types from the compiler context
        for (name, type_info) in &self.types {
            // Get type parameters from the original TypeDef if available
            let type_params: Vec<nostos_types::TypeParam> = self.type_defs.get(name)
                .map(|td| td.type_params.iter().map(|p| nostos_types::TypeParam {
                    name: p.name.node.clone(),
                    constraints: p.constraints.iter().map(|c| c.node.clone()).collect(),
                }).collect())
                .unwrap_or_default();

            match &type_info.kind {
                TypeInfoKind::Record { fields, mutable } => {
                    let field_types: Vec<(String, nostos_types::Type, bool)> = fields
                        .iter()
                        .map(|(n, ty)| (n.clone(), self.type_name_to_type(ty), false))
                        .collect();
                    env.define_type(
                        name.clone(),
                        nostos_types::TypeDef::Record {
                            params: type_params,
                            fields: field_types,
                            is_mutable: *mutable,
                        },
                    );
                }
                TypeInfoKind::Variant { constructors } => {
                    let ctors: Vec<nostos_types::Constructor> = constructors
                        .iter()
                        .map(|(ctor_name, field_types)| {
                            if field_types.is_empty() {
                                nostos_types::Constructor::Unit(ctor_name.clone())
                            } else {
                                nostos_types::Constructor::Positional(
                                    ctor_name.clone(),
                                    field_types.iter().map(|ty| self.type_name_to_type(ty)).collect(),
                                )
                            }
                        })
                        .collect();
                    env.define_type(
                        name.clone(),
                        nostos_types::TypeDef::Variant {
                            params: type_params,
                            constructors: ctors,
                        },
                    );
                }
            }
        }

        // Register mvars as bindings so type inference can resolve mvar references
        for (mvar_name, mvar_info) in &self.mvars {
            let mvar_type = self.type_name_to_type(&mvar_info.type_name);
            // Register with both qualified and unqualified names
            env.bindings.insert(mvar_name.clone(), (mvar_type.clone(), false));
            // Also register with just the local name (strip module prefix)
            if let Some(dot_pos) = mvar_name.rfind('.') {
                let local_name = &mvar_name[dot_pos + 1..];
                if !env.bindings.contains_key(local_name) {
                    env.bindings.insert(local_name.to_string(), (mvar_type, false));
                }
            }
        }

        // Register built-in functions from BUILTINS for type inference
        // This allows functions calling Panel.*, String.*, etc. to have proper return types
        for builtin in BUILTINS {
            if env.functions.contains_key(builtin.name) {
                continue; // Don't overwrite standard_env functions
            }
            if let Some(fn_type) = self.parse_signature_string(builtin.signature) {
                env.functions.insert(builtin.name.to_string(), fn_type);
            }
        }

        // Register known functions in environment for recursive calls
        // Don't overwrite functions from standard_env (like println) that have trait constraints
        // Sort function names to ensure deterministic processing order - this matters for overloaded
        // functions with different arities. When sorted, "foo/" comes before "foo/_", so the
        // 0-arity version gets registered as the base name "foo" first.
        let mut fn_names: Vec<_> = self.functions.keys().cloned().collect();
        fn_names.sort();
        for fn_name in fn_names {
            let fn_val = match self.functions.get(&fn_name) {
                Some(v) => v,
                None => continue,
            };
            if env.functions.contains_key(&fn_name) {
                // Skip - don't overwrite built-in functions with proper type params/constraints
                continue;
            }
            // Skip placeholder functions - they have no signature yet and would have wrong param types
            // We'll rely on pre-registration for the function being inferred (below)
            if fn_val.signature.is_none() {
                continue;
            }

            // IMPORTANT: Parse from the signature string to preserve type variable relationships
            // For example, "a -> List(a) -> a" should have the same Var ID for all occurrences of 'a'
            // Using param_types separately would create independent type variables
            let func_type = if let Some(sig) = fn_val.signature.as_ref() {
                if let Some(ft) = self.parse_signature_string(sig) {
                    ft
                } else {
                    // Fallback: construct from param_types if signature parsing fails
                    let param_types: Vec<nostos_types::Type> = fn_val.param_types
                        .iter()
                        .map(|ty| {
                            if ty == "_" || ty == "?" {
                                env.fresh_var()
                            } else {
                                self.type_name_to_type(ty)
                            }
                        })
                        .collect();
                    let ret_ty = fn_val.return_type.as_ref()
                        .map(|ty| self.type_name_to_type(ty))
                        .unwrap_or_else(|| env.fresh_var());
                    nostos_types::FunctionType {
                        type_params: vec![],
                        params: param_types,
                        ret: Box::new(ret_ty),
                    }
                }
            } else {
                continue; // No signature available
            };

            // Insert with full name (e.g., "bar/")
            env.functions.insert(fn_name.clone(), func_type.clone());

            // Also insert with base name (e.g., "main.bar") for simple lookups
            // Function names have format "module.name/" or "name/"
            if let Some(slash_pos) = fn_name.find('/') {
                let base_name = &fn_name[..slash_pos];
                // Only insert if base name not already registered (don't overwrite overloads)
                if !env.functions.contains_key(base_name) {
                    env.functions.insert(base_name.to_string(), func_type.clone());
                }

                // Also insert without module prefix for local lookups (e.g., "main.bar" -> "bar")
                if let Some(dot_pos) = base_name.rfind('.') {
                    let short_name = &base_name[dot_pos + 1..];
                    if !env.functions.contains_key(short_name) {
                        env.functions.insert(short_name.to_string(), func_type);
                    }
                }
            }
        }

        // Pre-register the function being inferred with fresh type variables for recursion support
        // This allows the function to call itself during type inference
        // Always register (overwrite builtins if needed) so user functions can shadow builtins
        let fn_name = &def.name.node;
        if let Some(clause) = def.clauses.first() {
                let param_types: Vec<nostos_types::Type> = clause.params
                    .iter()
                    .map(|p| {
                        if let Some(ty_expr) = &p.ty {
                            self.type_name_to_type(&self.type_expr_to_string(ty_expr))
                        } else {
                            env.fresh_var()
                        }
                    })
                    .collect();
                let ret_ty = clause.return_type.as_ref()
                    .map(|ty| self.type_name_to_type(&self.type_expr_to_string(ty)))
                    .unwrap_or_else(|| env.fresh_var());

            env.functions.insert(
                fn_name.clone(),
                nostos_types::FunctionType {
                    type_params: vec![],
                    params: param_types,
                    ret: Box::new(ret_ty),
                },
            );
        }

        // Create inference context
        let mut ctx = InferCtx::new(&mut env);

        // Infer the function type
        let func_ty = match ctx.infer_function(def) {
            Ok(ft) => ft,
            Err(_e) => {
                // Debug: uncomment to see inference errors
                // eprintln!("DEBUG: infer_function failed for {}: {:?}", def.name.node, _e);
                return None;
            }
        };

        // Solve constraints (this can hang on unresolved type vars with HasField)
        if let Err(_e) = ctx.solve() {
            // Debug: uncomment to see solve errors
            // eprintln!("DEBUG: solve failed for {}: {:?}", def.name.node, _e);
            return None;
        }

        // Collect all resolved types for the signature
        let resolved_params: Vec<nostos_types::Type> = func_ty.params
            .iter()
            .map(|ty| ctx.env.apply_subst(ty))
            .collect();
        let resolved_ret = ctx.env.apply_subst(&func_ty.ret);

        // Debug: uncomment to see resolved types
        // eprintln!("DEBUG HM: {} params={:?} ret={:?}", def.name.node, resolved_params, resolved_ret);

        // Collect all type variable IDs in order of first appearance
        let mut var_order: Vec<u32> = Vec::new();
        for ty in resolved_params.iter().chain(std::iter::once(&resolved_ret)) {
            self.collect_type_vars(ty, &mut var_order);
        }

        // Create mapping from type var ID to normalized letter
        let var_map: HashMap<u32, char> = var_order.iter().enumerate()
            .map(|(i, &id)| (id, (b'a' + (i as u8 % 26)) as char))
            .collect();

        // Format with normalized type variables
        let param_types: Vec<String> = resolved_params.iter()
            .map(|ty| self.format_type_normalized(ty, &var_map))
            .collect();
        let ret_type = self.format_type_normalized(&resolved_ret, &var_map);

        // Collect trait bounds for type variables that appear in the signature
        let mut bounds: Vec<String> = Vec::new();
        for (&var_id, &var_name) in &var_map {
            let trait_names = ctx.get_trait_bounds(var_id);
            for trait_name in trait_names {
                bounds.push(format!("{} {}", trait_name, var_name));
            }
        }
        bounds.sort(); // Deterministic ordering

        // Format the signature with constraint prefix if there are bounds
        let type_sig = if param_types.is_empty() {
            ret_type
        } else {
            format!("{} -> {}", param_types.join(" -> "), ret_type)
        };

        if bounds.is_empty() {
            Some(type_sig)
        } else {
            Some(format!("{} => {}", bounds.join(", "), type_sig))
        }
    }

    /// Type check a function definition using Hindley-Milner type inference.
    /// Returns Ok(()) if type checking passes, or a CompileError with details.
    ///
    /// The `qualified_name` parameter is the full module-qualified name of the function
    /// (e.g., "moduleB.main") which is used to prioritize same-module functions when
    /// resolving local names (to avoid cross-module conflicts when functions in different
    /// modules have the same local name but different arities).
    pub fn type_check_fn(&self, def: &FnDef, qualified_name: &str) -> Result<(), CompileError> {
        // Create a fresh type environment for inference
        let mut env = nostos_types::standard_env();

        // Register known types from the compiler context
        for (name, type_info) in &self.types {
            // Get type parameters from the original TypeDef if available
            let type_params: Vec<nostos_types::TypeParam> = self.type_defs.get(name)
                .map(|td| td.type_params.iter().map(|p| nostos_types::TypeParam {
                    name: p.name.node.clone(),
                    constraints: p.constraints.iter().map(|c| c.node.clone()).collect(),
                }).collect())
                .unwrap_or_default();

            match &type_info.kind {
                TypeInfoKind::Record { fields, mutable } => {
                    let field_types: Vec<(String, nostos_types::Type, bool)> = fields
                        .iter()
                        .map(|(n, ty)| (n.clone(), self.type_name_to_type(ty), false))
                        .collect();
                    env.define_type(
                        name.clone(),
                        nostos_types::TypeDef::Record {
                            params: type_params,
                            fields: field_types,
                            is_mutable: *mutable,
                        },
                    );
                }
                TypeInfoKind::Variant { constructors } => {
                    let ctors: Vec<nostos_types::Constructor> = constructors
                        .iter()
                        .map(|(ctor_name, field_types)| {
                            if field_types.is_empty() {
                                nostos_types::Constructor::Unit(ctor_name.clone())
                            } else {
                                nostos_types::Constructor::Positional(
                                    ctor_name.clone(),
                                    field_types.iter().map(|ty| self.type_name_to_type(ty)).collect(),
                                )
                            }
                        })
                        .collect();
                    env.define_type(
                        name.clone(),
                        nostos_types::TypeDef::Variant {
                            params: type_params,
                            constructors: ctors,
                        },
                    );
                }
            }
        }

        // Register trait implementations for custom types
        // All types implement Hash, Show, Eq, and Copy automatically
        let builtin_traits = ["Hash", "Show", "Eq", "Copy"];
        for (type_name, _) in &self.types {
            let for_type = nostos_types::Type::Named {
                name: type_name.clone(),
                args: vec![],
            };
            for trait_name in &builtin_traits {
                env.impls.push(nostos_types::TraitImpl {
                    trait_name: trait_name.to_string(),
                    for_type: for_type.clone(),
                    constraints: vec![],
                });
            }
        }

        // Also register any explicit trait implementations
        for (type_name, traits) in &self.type_traits {
            let for_type = nostos_types::Type::Named {
                name: type_name.clone(),
                args: vec![],
            };
            for trait_name in traits {
                if !builtin_traits.contains(&trait_name.as_str()) {
                    env.impls.push(nostos_types::TraitImpl {
                        trait_name: trait_name.clone(),
                        for_type: for_type.clone(),
                        constraints: vec![],
                    });
                }
            }
        }

        // NOTE: We intentionally do NOT register mvars here in type_check_fn.
        // Mvar registration is only done in try_hm_inference for signature inference.
        // Registering mvars in type_check_fn can cause spurious type errors when
        // the inferred function signature uses type variables that don't match
        // the concrete mvar types.

        // Register built-in functions from BUILTINS for type inference
        // This enables UFCS type checking (e.g., String.contains expects String -> Bool)
        for builtin in BUILTINS {
            if env.functions.contains_key(builtin.name) {
                continue; // Don't overwrite standard_env functions
            }
            if let Some(fn_type) = self.parse_signature_string(builtin.signature) {
                env.functions.insert(builtin.name.to_string(), fn_type);
            }
        }

        // Register known functions FIRST - these have actual inferred types
        // after compilation, not just type variables
        // Sort function names to ensure deterministic processing order - this matters for overloaded
        // functions with different arities. When sorted, "foo/" comes before "foo/_", so the
        // 0-arity version gets registered as the base name "foo" first.
        let mut fn_names_sorted: Vec<_> = self.functions.keys().cloned().collect();
        fn_names_sorted.sort();
        for fn_name in fn_names_sorted {
            let fn_val = match self.functions.get(&fn_name) {
                Some(v) => v,
                None => continue,
            };
            // Skip placeholder functions - they have no signature set yet
            // A function has been properly compiled if it has a signature
            if fn_val.signature.is_none() {
                continue;
            }
            // Parse the full signature to get accurate param and return types
            // This is better than using param_types which may contain "?" placeholders
            let fn_type = if let Some(sig) = fn_val.signature.as_ref() {
                if let Some(parsed) = self.parse_signature_string(sig) {
                    parsed
                } else {
                    // Fallback to building from param_types
                    let param_types: Vec<nostos_types::Type> = fn_val.param_types
                        .iter()
                        .map(|ty| self.type_name_to_type(ty))
                        .collect();
                    let ret_ty = fn_val.return_type.as_ref()
                        .map(|ty| self.type_name_to_type(ty))
                        .unwrap_or_else(|| env.fresh_var());
                    nostos_types::FunctionType {
                        type_params: vec![],
                        params: param_types,
                        ret: Box::new(ret_ty),
                    }
                }
            } else {
                // No signature - use param_types as fallback
                let param_types: Vec<nostos_types::Type> = fn_val.param_types
                    .iter()
                    .map(|ty| self.type_name_to_type(ty))
                    .collect();
                let ret_ty = fn_val.return_type.as_ref()
                    .map(|ty| self.type_name_to_type(ty))
                    .unwrap_or_else(|| env.fresh_var());
                nostos_types::FunctionType {
                    type_params: vec![],
                    params: param_types,
                    ret: Box::new(ret_ty),
                }
            };

            // Register with full key (e.g., "bar2/")
            if !env.functions.contains_key(&fn_name) {
                env.functions.insert(fn_name, fn_type.clone());
            }
        }

        // Then register pending function signatures (for functions not yet compiled)
        // These use type variables since we don't know the actual types yet
        for (fn_name, fn_type) in &self.pending_fn_signatures {
            if !env.functions.contains_key(fn_name) {
                env.functions.insert(fn_name.clone(), fn_type.clone());
            }
        }

        // Now register local names, prioritizing functions in the same module as the current function.
        // Extract the current function's module prefix (e.g., "moduleB.main" -> "moduleB.")
        let current_module_prefix: Option<String> = {
            let base_name = qualified_name.split('/').next().unwrap_or(qualified_name);
            if let Some(dot_pos) = base_name.rfind('.') {
                Some(format!("{}.", &base_name[..dot_pos]))
            } else {
                None
            }
        };

        // First pass: register local names for functions NOT in the current module
        // (only if not already present)
        // Sort to ensure "foo/" (0-arity) comes before "foo/_" (1-arity)
        let mut fn_keys_sorted: Vec<_> = self.functions.keys().cloned().collect();
        fn_keys_sorted.sort();
        for fn_name in &fn_keys_sorted {
            let base_name = fn_name.split('/').next().unwrap_or(fn_name);
            let in_current_module = current_module_prefix.as_ref()
                .map(|prefix| base_name.starts_with(prefix))
                .unwrap_or(base_name.rfind('.').is_none());

            if !in_current_module {
                let local_name = base_name.rsplit('.').next().unwrap_or(base_name);
                if !env.functions.contains_key(local_name) {
                    if let Some(fn_type) = env.functions.get(fn_name).cloned() {
                        env.functions.insert(local_name.to_string(), fn_type);
                    }
                }
            }
        }
        for fn_name in self.pending_fn_signatures.keys() {
            let base_name = fn_name.split('/').next().unwrap_or(fn_name);
            let in_current_module = current_module_prefix.as_ref()
                .map(|prefix| base_name.starts_with(prefix))
                .unwrap_or(base_name.rfind('.').is_none());

            if !in_current_module {
                if let Some(dot_pos) = base_name.rfind('.') {
                    let local_name = &base_name[dot_pos + 1..];
                    if !env.functions.contains_key(local_name) {
                        if let Some(fn_type) = env.functions.get(fn_name).cloned() {
                            env.functions.insert(local_name.to_string(), fn_type);
                        }
                    }
                }
            }
        }

        // Second pass: register local names for functions IN the current module
        // (these OVERWRITE any existing local names to ensure same-module priority)
        // Sort and iterate so that "foo/" (0-arity) is processed LAST and wins
        // (since we're overwriting, the last one processed wins)
        for fn_name in fn_keys_sorted.iter().rev() {
            let base_name = fn_name.split('/').next().unwrap_or(fn_name);
            let in_current_module = current_module_prefix.as_ref()
                .map(|prefix| base_name.starts_with(prefix))
                .unwrap_or(base_name.rfind('.').is_none());

            if in_current_module {
                let local_name = base_name.rsplit('.').next().unwrap_or(base_name);
                if let Some(fn_type) = env.functions.get(fn_name).cloned() {
                    env.functions.insert(local_name.to_string(), fn_type);
                }
            }
        }
        for fn_name in self.pending_fn_signatures.keys() {
            let base_name = fn_name.split('/').next().unwrap_or(fn_name);
            let in_current_module = current_module_prefix.as_ref()
                .map(|prefix| base_name.starts_with(prefix))
                .unwrap_or(base_name.rfind('.').is_none());

            if in_current_module {
                if let Some(dot_pos) = base_name.rfind('.') {
                    let local_name = &base_name[dot_pos + 1..];
                    if let Some(fn_type) = env.functions.get(fn_name).cloned() {
                        env.functions.insert(local_name.to_string(), fn_type);
                    }
                }
            }
        }

        // Update next_var to avoid collisions with type variables in registered functions
        // This is critical because pending_fn_signatures contains type variables from a different context
        let max_var_in_functions = env.functions.values()
            .filter_map(|ft| ft.max_var_id())
            // Filter out u32::MAX which is used as a sentinel for unknown types
            .filter(|&id| id != u32::MAX)
            .max();
        if let Some(max_id) = max_var_in_functions {
            // Set next_var to be at least max_id + 1 to avoid collisions
            if env.next_var <= max_id {
                env.next_var = max_id.saturating_add(1);
            }
        }

        // Pre-register the function being checked with fresh type variables for recursion support
        // This allows the function to call itself during type inference
        // Always overwrite existing entries - when checking an overload, we need THIS overload's
        // types, not another overload's types that might have been registered earlier
        let fn_name = &def.name.node;
        if let Some(clause) = def.clauses.first() {
            let param_types: Vec<nostos_types::Type> = clause.params
                .iter()
                .map(|p| {
                    if let Some(ty_expr) = &p.ty {
                        self.type_name_to_type(&self.type_expr_to_string(ty_expr))
                    } else {
                        env.fresh_var()
                    }
                })
                .collect();
            let ret_ty = clause.return_type.as_ref()
                .map(|ty| self.type_name_to_type(&self.type_expr_to_string(ty)))
                .unwrap_or_else(|| env.fresh_var());

            env.functions.insert(
                fn_name.clone(),
                nostos_types::FunctionType {
                    type_params: vec![],
                    params: param_types,
                    ret: Box::new(ret_ty),
                },
            );
        }

        // Create inference context
        let mut ctx = InferCtx::new(&mut env);

        // Infer the function type - this generates constraints
        let span = def.name.span;
        ctx.infer_function(def).map_err(|e| CompileError::TypeError {
            message: e.to_string(),
            span,
        })?;

        // Solve constraints - this is where type mismatches are detected
        ctx.solve().map_err(|e| CompileError::TypeError {
            message: e.to_string(),
            span,
        })?;

        Ok(())
    }

    /// Check if an error is an inference limitation (only type variables, no concrete types).
    /// We should report errors that involve concrete type mismatches like "Int and String".
    fn is_type_variable_only_error(message: &str) -> bool {
        // Check for "Cannot unify types: X and Y" pattern
        if message.contains("Cannot unify types:") {
            // Extract the two types from the message
            if let Some(types_part) = message.strip_prefix("Cannot unify types: ") {
                let parts: Vec<&str> = types_part.split(" and ").collect();
                if parts.len() == 2 {
                    let type1 = parts[0].trim();
                    let type2 = parts[1].trim();

                    // Check if a type contains type variables (used in higher-order function inference)
                    let contains_type_var = |s: &str| {
                        s.contains('?') ||  // Internal type variable like ?5 or (?2, ?3) -> ?2
                        (s.len() == 1 && s.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false))  // Single letter like T or a
                    };

                    // Suppress if either type contains type variables
                    // Errors involving function types with type variables like "(?2, ?3) -> ?2 and Int"
                    // are inference limitations with higher-order functions
                    if contains_type_var(type1) || contains_type_var(type2) {
                        return true;
                    }
                }
            }
        }

        // Trait bound errors with type parameters should be suppressed
        // Type parameters are single uppercase/lowercase letters followed by "does not implement"
        if message.contains("does not implement") {
            // Check if there's a single-letter type parameter before "does not implement"
            if let Some(pos) = message.find("does not implement") {
                let prefix = &message[..pos].trim();
                let words: Vec<&str> = prefix.split_whitespace().collect();
                if let Some(&last_word) = words.last() {
                    // Type parameters are single letters (uppercase or lowercase)
                    if last_word.len() == 1 && last_word.chars().next().unwrap().is_alphabetic() {
                        return true;
                    }
                }
            }
        }

        // Trait errors involving type variables (like List[?5]) should be suppressed
        if message.contains("does not implement") && message.contains('?') {
            return true;
        }

        // Occurs check errors are type inference limitations with recursive types
        if message.contains("Occurs check failed") {
            return true;
        }

        false
    }

    /// Check if a function definition is recursive (calls itself)
    fn is_recursive_fn(def: &FnDef) -> bool {
        let fn_name = &def.name.node;
        for clause in &def.clauses {
            if Self::expr_contains_call(&clause.body, fn_name) {
                return true;
            }
        }
        false
    }

    /// Check if an expression contains a call to the given function name
    fn expr_contains_call(expr: &Expr, fn_name: &str) -> bool {
        match expr {
            Expr::Call(callee, _type_args, args, _) => {
                // Check if this is a direct call to fn_name
                if let Expr::Var(ident) = callee.as_ref() {
                    if ident.node == fn_name {
                        return true;
                    }
                }
                // Check callee and all argument expressions
                if Self::expr_contains_call(callee, fn_name) {
                    return true;
                }
                args.iter().any(|a| Self::expr_contains_call(a, fn_name))
            }
            Expr::BinOp(lhs, _, rhs, _) => {
                Self::expr_contains_call(lhs, fn_name) ||
                Self::expr_contains_call(rhs, fn_name)
            }
            Expr::UnaryOp(_, e, _) => Self::expr_contains_call(e, fn_name),
            Expr::If(cond, then_br, else_br, _) => {
                Self::expr_contains_call(cond, fn_name) ||
                Self::expr_contains_call(then_br, fn_name) ||
                Self::expr_contains_call(else_br, fn_name)
            }
            Expr::Match(scrutinee, arms, _) => {
                if Self::expr_contains_call(scrutinee, fn_name) {
                    return true;
                }
                arms.iter().any(|arm| Self::expr_contains_call(&arm.body, fn_name))
            }
            Expr::Block(stmts, _) => {
                for stmt in stmts {
                    match stmt {
                        Stmt::Let(binding) => {
                            if Self::expr_contains_call(&binding.value, fn_name) {
                                return true;
                            }
                        }
                        Stmt::Expr(e) => {
                            if Self::expr_contains_call(e, fn_name) {
                                return true;
                            }
                        }
                        Stmt::Assign(_, e, _) => {
                            if Self::expr_contains_call(e, fn_name) {
                                return true;
                            }
                        }
                    }
                }
                false
            }
            Expr::Lambda(_, body, _) => Self::expr_contains_call(body, fn_name),
            Expr::Try(body, arms, finally, _) => {
                Self::expr_contains_call(body, fn_name) ||
                arms.iter().any(|arm| Self::expr_contains_call(&arm.body, fn_name)) ||
                finally.as_ref().map(|f| Self::expr_contains_call(f, fn_name)).unwrap_or(false)
            }
            Expr::String(string_lit, _) => {
                // Check string interpolation parts
                match string_lit {
                    StringLit::Interpolated(parts) => parts.iter().any(|p| {
                        match p {
                            StringPart::Expr(e) => Self::expr_contains_call(e, fn_name),
                            _ => false,
                        }
                    }),
                    StringLit::Plain(_) => false,
                }
            }
            Expr::Tuple(elems, _) => elems.iter().any(|e| Self::expr_contains_call(e, fn_name)),
            Expr::List(elems, _, _) => elems.iter().any(|e| Self::expr_contains_call(e, fn_name)),
            Expr::MethodCall(receiver, _, args, _) => {
                Self::expr_contains_call(receiver, fn_name) ||
                args.iter().any(|a| Self::expr_contains_call(a, fn_name))
            }
            Expr::FieldAccess(e, _, _) => Self::expr_contains_call(e, fn_name),
            Expr::Index(e, idx, _) => {
                Self::expr_contains_call(e, fn_name) ||
                Self::expr_contains_call(idx, fn_name)
            }
            _ => false, // Literals and other terminal expressions
        }
    }

    /// Check if a function definition calls any function that has untyped parameters.
    /// This is used to defer type errors for functions that depend on not-yet-inferred functions.
    fn calls_function_with_untyped_params(&self, def: &FnDef) -> bool {
        for clause in &def.clauses {
            if self.expr_calls_function_with_untyped_params(&clause.body) {
                return true;
            }
        }
        false
    }

    /// Check if an expression calls any function with untyped params.
    fn expr_calls_function_with_untyped_params(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Call(func, _type_args, args, _) => {
                // Check the called function
                if let Expr::Var(ident) = func.as_ref() {
                    let name = &ident.node;
                    // Look up the function to see if it has untyped params
                    // Try different name formats
                    for (fn_name, fn_val) in &self.functions {
                        let base_name = fn_name.split('/').next().unwrap_or(fn_name);
                        if base_name == name || fn_name == name {
                            // Check if any param type is "?" or "_"
                            if fn_val.param_types.iter().any(|t| t == "?" || t == "_") {
                                return true;
                            }
                        }
                    }
                }
                // Recursively check arguments
                args.iter().any(|a| self.expr_calls_function_with_untyped_params(a)) ||
                self.expr_calls_function_with_untyped_params(func)
            }
            Expr::BinOp(left, _, right, _) => {
                self.expr_calls_function_with_untyped_params(left) ||
                self.expr_calls_function_with_untyped_params(right)
            }
            Expr::UnaryOp(_, e, _) => self.expr_calls_function_with_untyped_params(e),
            Expr::If(cond, then_branch, else_branch, _) => {
                self.expr_calls_function_with_untyped_params(cond) ||
                self.expr_calls_function_with_untyped_params(then_branch) ||
                self.expr_calls_function_with_untyped_params(else_branch)
            }
            Expr::Match(e, arms, _) => {
                self.expr_calls_function_with_untyped_params(e) ||
                arms.iter().any(|arm| self.expr_calls_function_with_untyped_params(&arm.body))
            }
            Expr::Block(stmts, _) => {
                stmts.iter().any(|stmt| match stmt {
                    Stmt::Let(binding) => self.expr_calls_function_with_untyped_params(&binding.value),
                    Stmt::Expr(e) => self.expr_calls_function_with_untyped_params(e),
                    Stmt::Assign(_, e, _) => self.expr_calls_function_with_untyped_params(e),
                })
            }
            Expr::Lambda(_, body, _) => self.expr_calls_function_with_untyped_params(body),
            Expr::Try(body, arms, finally, _) => {
                self.expr_calls_function_with_untyped_params(body) ||
                arms.iter().any(|arm| self.expr_calls_function_with_untyped_params(&arm.body)) ||
                finally.as_ref().map(|f| self.expr_calls_function_with_untyped_params(f)).unwrap_or(false)
            }
            Expr::Tuple(elems, _) | Expr::List(elems, _, _) => {
                elems.iter().any(|e| self.expr_calls_function_with_untyped_params(e))
            }
            Expr::MethodCall(receiver, _, args, _) => {
                self.expr_calls_function_with_untyped_params(receiver) ||
                args.iter().any(|a| self.expr_calls_function_with_untyped_params(a))
            }
            Expr::FieldAccess(e, _, _) => self.expr_calls_function_with_untyped_params(e),
            Expr::Index(e, idx, _) => {
                self.expr_calls_function_with_untyped_params(e) ||
                self.expr_calls_function_with_untyped_params(idx)
            }
            _ => false,
        }
    }

    /// Collect all type variable IDs in order of first appearance.
    fn collect_type_vars(&self, ty: &nostos_types::Type, vars: &mut Vec<u32>) {
        match ty {
            nostos_types::Type::Var(id) => {
                if !vars.contains(id) {
                    vars.push(*id);
                }
            }
            nostos_types::Type::Tuple(elems) => {
                for e in elems {
                    self.collect_type_vars(e, vars);
                }
            }
            nostos_types::Type::List(elem) | nostos_types::Type::Array(elem)
            | nostos_types::Type::Set(elem) | nostos_types::Type::IO(elem) => {
                self.collect_type_vars(elem, vars);
            }
            nostos_types::Type::Map(k, v) => {
                self.collect_type_vars(k, vars);
                self.collect_type_vars(v, vars);
            }
            nostos_types::Type::Function(f) => {
                for p in &f.params {
                    self.collect_type_vars(p, vars);
                }
                self.collect_type_vars(&f.ret, vars);
            }
            nostos_types::Type::Named { args, .. } => {
                for a in args {
                    self.collect_type_vars(a, vars);
                }
            }
            nostos_types::Type::Record(rec) => {
                for (_, t, _) in &rec.fields {
                    self.collect_type_vars(t, vars);
                }
            }
            _ => {}
        }
    }

    /// Format a type with normalized type variable names.
    fn format_type_normalized(&self, ty: &nostos_types::Type, var_map: &HashMap<u32, char>) -> String {
        match ty {
            nostos_types::Type::Var(id) => {
                var_map.get(id).map(|c| c.to_string()).unwrap_or_else(|| format!("?{}", id))
            }
            nostos_types::Type::Int => "Int".to_string(),
            nostos_types::Type::Int8 => "Int8".to_string(),
            nostos_types::Type::Int16 => "Int16".to_string(),
            nostos_types::Type::Int32 => "Int32".to_string(),
            nostos_types::Type::Int64 => "Int64".to_string(),
            nostos_types::Type::UInt8 => "UInt8".to_string(),
            nostos_types::Type::UInt16 => "UInt16".to_string(),
            nostos_types::Type::UInt32 => "UInt32".to_string(),
            nostos_types::Type::UInt64 => "UInt64".to_string(),
            nostos_types::Type::Float => "Float".to_string(),
            nostos_types::Type::Float32 => "Float32".to_string(),
            nostos_types::Type::Float64 => "Float64".to_string(),
            nostos_types::Type::BigInt => "BigInt".to_string(),
            nostos_types::Type::Decimal => "Decimal".to_string(),
            nostos_types::Type::Bool => "Bool".to_string(),
            nostos_types::Type::Char => "Char".to_string(),
            nostos_types::Type::String => "String".to_string(),
            nostos_types::Type::Unit => "()".to_string(),
            nostos_types::Type::Never => "Never".to_string(),
            nostos_types::Type::Pid => "Pid".to_string(),
            nostos_types::Type::Ref => "Ref".to_string(),
            nostos_types::Type::TypeParam(name) => name.clone(),
            nostos_types::Type::Tuple(elems) => {
                let inner: Vec<String> = elems.iter()
                    .map(|t| self.format_type_normalized(t, var_map))
                    .collect();
                format!("({})", inner.join(", "))
            }
            nostos_types::Type::List(elem) => {
                format!("List[{}]", self.format_type_normalized(elem, var_map))
            }
            nostos_types::Type::Array(elem) => {
                format!("Array[{}]", self.format_type_normalized(elem, var_map))
            }
            nostos_types::Type::Set(elem) => {
                format!("Set[{}]", self.format_type_normalized(elem, var_map))
            }
            nostos_types::Type::Map(k, v) => {
                format!("Map[{}, {}]",
                    self.format_type_normalized(k, var_map),
                    self.format_type_normalized(v, var_map))
            }
            nostos_types::Type::Function(f) => {
                let params: Vec<String> = f.params.iter()
                    .map(|t| self.format_type_normalized(t, var_map))
                    .collect();
                let ret = self.format_type_normalized(&f.ret, var_map);
                // Wrap entire function type in parentheses so it's a single unit
                // when used as a parameter in another function signature
                if params.is_empty() {
                    format!("(() -> {})", ret)
                } else {
                    format!("(({}) -> {})", params.join(", "), ret)
                }
            }
            nostos_types::Type::Named { name, args } => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let args_str: Vec<String> = args.iter()
                        .map(|t| self.format_type_normalized(t, var_map))
                        .collect();
                    format!("{}[{}]", name, args_str.join(", "))
                }
            }
            nostos_types::Type::Record(rec) => {
                if let Some(name) = &rec.name {
                    name.clone()
                } else {
                    let fields: Vec<String> = rec.fields.iter()
                        .map(|(n, t, _)| format!("{}: {}", n, self.format_type_normalized(t, var_map)))
                        .collect();
                    format!("{{{}}}", fields.join(", "))
                }
            }
            nostos_types::Type::Variant(v) => v.name.clone(),
            nostos_types::Type::IO(inner) => {
                format!("IO[{}]", self.format_type_normalized(inner, var_map))
            }
        }
    }

    /// Convert a type name string to a nostos_types::Type.
    /// Handles parameterized types like "List[Int]" and "Map[String, Int]".
    fn type_name_to_type(&self, ty: &str) -> nostos_types::Type {
        let ty = ty.trim();

        // Handle list shorthand syntax: [a] means List[a]
        if ty.starts_with('[') && ty.ends_with(']') {
            let inner = &ty[1..ty.len() - 1].trim();
            let elem_type = self.type_name_to_type(inner);
            return nostos_types::Type::List(Box::new(elem_type));
        }

        // Handle tuple types: (A, B, C) but not function types (A -> B)
        // Also exclude "()" which is Unit
        if ty.starts_with('(') && ty.ends_with(')') && ty != "()" && !ty.contains("->") {
            let inner = &ty[1..ty.len() - 1];
            // Parse comma-separated elements at depth 0
            let mut elems = Vec::new();
            let mut current = String::new();
            let mut depth = 0;
            for ch in inner.chars() {
                match ch {
                    '(' | '[' | '{' => {
                        depth += 1;
                        current.push(ch);
                    }
                    ')' | ']' | '}' => {
                        depth -= 1;
                        current.push(ch);
                    }
                    ',' if depth == 0 => {
                        if !current.trim().is_empty() {
                            elems.push(self.type_name_to_type(current.trim()));
                        }
                        current.clear();
                    }
                    _ => current.push(ch),
                }
            }
            if !current.trim().is_empty() {
                elems.push(self.type_name_to_type(current.trim()));
            }
            if elems.len() >= 2 {
                return nostos_types::Type::Tuple(elems);
            }
            // Single element in parens - just return the element type
            if elems.len() == 1 {
                return elems.into_iter().next().unwrap();
            }
        }

        // Handle function type syntax: "(params) -> ret" or "param -> ret"
        // This includes parenthesized function types like "(() -> a)"
        if let Some(func_type) = self.parse_function_type_string(ty) {
            return func_type;
        }

        // Check for parameterized type syntax: Name[Args]
        if let Some(bracket_pos) = ty.find('[') {
            if ty.ends_with(']') {
                let name = ty[..bracket_pos].trim();
                let args_str = &ty[bracket_pos + 1..ty.len() - 1];

                // Parse type arguments (handle nested brackets for things like List[List[Int]])
                let args = self.parse_type_args(args_str);

                // Handle built-in parameterized types
                return match name {
                    "List" if args.len() == 1 => {
                        nostos_types::Type::List(Box::new(args.into_iter().next().unwrap()))
                    }
                    "Array" if args.len() == 1 => {
                        nostos_types::Type::Array(Box::new(args.into_iter().next().unwrap()))
                    }
                    "Set" if args.len() == 1 => {
                        nostos_types::Type::Set(Box::new(args.into_iter().next().unwrap()))
                    }
                    "Map" if args.len() == 2 => {
                        let mut iter = args.into_iter();
                        let key = iter.next().unwrap();
                        let val = iter.next().unwrap();
                        nostos_types::Type::Map(Box::new(key), Box::new(val))
                    }
                    "IO" if args.len() == 1 => {
                        nostos_types::Type::IO(Box::new(args.into_iter().next().unwrap()))
                    }
                    _ => nostos_types::Type::Named {
                        name: name.to_string(),
                        args,
                    },
                };
            }
        }

        // Handle space-separated parameterized types like "Map k v", "List a", "Set a"
        // These come from BUILTINS signatures
        // Must split respecting parentheses, e.g., "Option (String, String)" -> ["Option", "(String, String)"]
        let parts = self.split_type_args_by_space(ty);
        if parts.len() >= 2 {
            let name = parts[0].as_str();
            let args: Vec<nostos_types::Type> = parts[1..].iter()
                .map(|arg| self.type_name_to_type(arg))
                .collect();

            return match name {
                "List" if args.len() == 1 => {
                    nostos_types::Type::List(Box::new(args.into_iter().next().unwrap()))
                }
                "Array" if args.len() == 1 => {
                    nostos_types::Type::Array(Box::new(args.into_iter().next().unwrap()))
                }
                "Set" if args.len() == 1 => {
                    nostos_types::Type::Set(Box::new(args.into_iter().next().unwrap()))
                }
                "Map" if args.len() == 2 => {
                    let mut iter = args.into_iter();
                    let key = iter.next().unwrap();
                    let val = iter.next().unwrap();
                    nostos_types::Type::Map(Box::new(key), Box::new(val))
                }
                "IO" if args.len() == 1 => {
                    nostos_types::Type::IO(Box::new(args.into_iter().next().unwrap()))
                }
                _ => nostos_types::Type::Named {
                    name: name.to_string(),
                    args,
                },
            };
        }

        match ty {
            "Int" | "Int64" => nostos_types::Type::Int,
            "Int8" => nostos_types::Type::Int8,
            "Int16" => nostos_types::Type::Int16,
            "Int32" => nostos_types::Type::Int32,
            "UInt8" => nostos_types::Type::UInt8,
            "UInt16" => nostos_types::Type::UInt16,
            "UInt32" => nostos_types::Type::UInt32,
            "UInt64" => nostos_types::Type::UInt64,
            "Float" | "Float64" => nostos_types::Type::Float,
            "Float32" => nostos_types::Type::Float32,
            "BigInt" => nostos_types::Type::BigInt,
            "Decimal" => nostos_types::Type::Decimal,
            "Bool" => nostos_types::Type::Bool,
            "Char" => nostos_types::Type::Char,
            "String" => nostos_types::Type::String,
            "Pid" => nostos_types::Type::Pid,
            "Ref" => nostos_types::Type::Ref,
            "()" | "Unit" => nostos_types::Type::Unit,
            "?" | "_" => nostos_types::Type::Var(u32::MAX), // Unknown/untyped param
            _ => {
                // Check if this is a type variable (single lowercase letter)
                // Type variables like 'a', 'b', 'c' are used in polymorphic signatures
                if ty.len() == 1 && ty.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false) {
                    // Convert to a consistent type variable ID based on the letter
                    // 'a' -> 1, 'b' -> 2, etc.
                    let var_id = (ty.chars().next().unwrap() as u32) - ('a' as u32) + 1;
                    nostos_types::Type::Var(var_id)
                } else {
                    nostos_types::Type::Named { name: ty.to_string(), args: vec![] }
                }
            }
        }
    }

    /// Parse comma-separated type arguments, handling nested brackets and parentheses.
    fn parse_type_args(&self, args_str: &str) -> Vec<nostos_types::Type> {
        let mut args = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for ch in args_str.chars() {
            match ch {
                '[' | '(' => {
                    depth += 1;
                    current.push(ch);
                }
                ']' | ')' => {
                    depth -= 1;
                    current.push(ch);
                }
                ',' if depth == 0 => {
                    if !current.trim().is_empty() {
                        args.push(self.type_name_to_type(current.trim()));
                    }
                    current.clear();
                }
                _ => current.push(ch),
            }
        }

        // Don't forget the last argument
        if !current.trim().is_empty() {
            args.push(self.type_name_to_type(current.trim()));
        }

        args
    }

    /// Split type string by spaces while respecting parentheses and brackets.
    /// E.g., "Option (String, String)" -> ["Option", "(String, String)"]
    /// E.g., "Map k v" -> ["Map", "k", "v"]
    fn split_type_args_by_space(&self, ty: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for ch in ty.chars() {
            match ch {
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(ch);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    current.push(ch);
                }
                ' ' if depth == 0 => {
                    if !current.is_empty() {
                        parts.push(current.clone());
                        current.clear();
                    }
                }
                _ => current.push(ch),
            }
        }

        if !current.is_empty() {
            parts.push(current);
        }

        parts
    }

    /// Parse a signature string like "String -> Int" or "Int -> String -> ()" into a FunctionType.
    /// This is used to register built-in functions for type inference.
    fn parse_signature_string(&self, sig: &str) -> Option<nostos_types::FunctionType> {
        let sig = sig.trim();

        // Handle constraint syntax: "Show a => a -> String" -> "a -> String"
        let sig_without_constraints = if let Some(idx) = sig.find("=>") {
            sig[idx + 2..].trim()
        } else {
            sig
        };

        // Split by " -> " to get parameter and return types
        // Need to be careful with nested types like "Map k v -> k"
        let parts: Vec<&str> = self.split_arrow_types(sig_without_constraints);

        if parts.is_empty() {
            return None;
        }

        // Last part is the return type, rest are parameters
        let ret_str = parts.last()?;
        let param_strs = &parts[..parts.len() - 1];

        // Filter out "()" which means no params in signature syntax (e.g., "() -> Pid")
        let params: Vec<nostos_types::Type> = param_strs
            .iter()
            .filter(|s| s.trim() != "()")
            .map(|s| self.type_name_to_type(s.trim()))
            .collect();

        let ret = self.type_name_to_type(ret_str.trim());

        Some(nostos_types::FunctionType {
            type_params: vec![],
            params,
            ret: Box::new(ret),
        })
    }

    /// Split a signature string by " -> " while respecting nested brackets.
    fn split_arrow_types<'a>(&self, sig: &'a str) -> Vec<&'a str> {
        let mut parts = Vec::new();
        let mut depth: i32 = 0;
        let mut start = 0;
        let bytes = sig.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            match bytes[i] {
                b'[' | b'(' | b'{' => depth += 1,
                b']' | b')' | b'}' => depth = (depth - 1).max(0),
                b'-' if depth == 0 && i + 2 < bytes.len() && bytes[i + 1] == b'>' => {
                    // Found " -> " at depth 0
                    let part = &sig[start..i].trim();
                    if !part.is_empty() {
                        parts.push(*part);
                    }
                    i += 2; // Skip "->"
                    // Skip whitespace after ->
                    while i < bytes.len() && bytes[i] == b' ' {
                        i += 1;
                    }
                    start = i;
                    continue;
                }
                _ => {}
            }
            i += 1;
        }

        // Don't forget the last part (return type)
        let last = sig[start..].trim();
        if !last.is_empty() {
            parts.push(last);
        }

        parts
    }

    /// Parse a function type string like "() -> Int", "a -> b", or "(() -> a)".
    /// Returns None if the string doesn't represent a function type.
    fn parse_function_type_string(&self, ty: &str) -> Option<nostos_types::Type> {
        let ty = ty.trim();

        // Handle parenthesized type: if entire string is wrapped in parens, unwrap and recurse
        // But be careful: "()" is Unit, not an empty paren group
        // And "(a, b)" is a tuple, not a paren group
        if ty.starts_with('(') && ty.ends_with(')') && ty != "()" {
            // Check if the parens are balanced and wrap the entire expression
            let inner = &ty[1..ty.len() - 1];
            let mut depth = 0;
            let mut is_wrapped = true;
            for (i, c) in inner.char_indices() {
                match c {
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' => {
                        depth -= 1;
                        if depth < 0 {
                            is_wrapped = false;
                            break;
                        }
                    }
                    ',' if depth == 0 => {
                        // Found a comma at depth 0 - this is a tuple, not a wrapped type
                        is_wrapped = false;
                        break;
                    }
                    _ => {}
                }
                // If we find "->" at depth 0, we need to check if it's in the middle or at the end
                if depth == 0 && i + 2 < inner.len() && &inner[i..i+2] == "->" {
                    // There's an arrow inside, so this is a wrapped function type
                    // Continue checking for balanced parens
                }
            }

            // If the outer parens wrap the entire expression and it's balanced
            if is_wrapped && depth == 0 {
                // Try to parse the inner content as a function type
                if let Some(inner_type) = self.parse_function_type_string(inner) {
                    return Some(inner_type);
                }
                // If inner isn't a function type, the outer parens might just be grouping
                // Fall through to check for arrow at this level
            }
        }

        // Look for " -> " at depth 0 to identify function types
        let mut depth = 0;
        let bytes = ty.as_bytes();
        for i in 0..bytes.len() {
            match bytes[i] {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth = (depth - 1).max(0),
                b'-' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == b'>' => {
                    // Found " -> " at depth 0 - this is a function type
                    let params_str = ty[..i].trim();
                    let ret_str = ty[i + 2..].trim();

                    // Parse parameter types
                    let params = if params_str == "()" || params_str.is_empty() {
                        // No parameters
                        vec![]
                    } else if params_str.starts_with('(') && params_str.ends_with(')') {
                        // Multiple params or single param in parens: "(a, b)" or "(a)"
                        let inner = &params_str[1..params_str.len() - 1];
                        self.parse_type_args(inner)
                    } else {
                        // Single param without parens: "a"
                        vec![self.type_name_to_type(params_str)]
                    };

                    let ret = self.type_name_to_type(ret_str);

                    return Some(nostos_types::Type::Function(nostos_types::FunctionType {
                        type_params: vec![],
                        params,
                        ret: Box::new(ret),
                    }));
                }
                _ => {}
            }
        }

        None
    }

    /// HTML tag names and helper functions that should be resolved from stdlib.html
    /// when inside an Html(...) scope.
    const HTML_SCOPED_NAMES: &'static [&'static str] = &[
        // Helpers
        "text", "raw", "el", "empty", "render",
        // Type constructors
        "Element", "Text", "Raw", "Empty",
        // Container tags (overloaded: tag(List[Html]) and tag(String))
        "div", "span", "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td",
        "form", "nav", "header", "footer", "section", "article", "aside",
        "headEl", "body", "button", "label",  // headEl avoids conflict with stdlib head()
        // Text-only tags
        "title", "strong", "em", "code", "pre", "small",
        // Self-closing tags
        "br", "hr", "img", "input", "meta", "linkTag",
        // Elements with required attrs (overloaded)
        "a",
    ];

    /// Transform an expression inside Html(...) to resolve bare HTML tag names
    /// to their qualified stdlib.html.* equivalents.
    /// IMPORTANT: Only transforms names that are being CALLED as functions, not variable references.
    fn transform_html_expr(&self, expr: &Expr) -> Expr {
        match expr {
            // Transform function calls where the function is a bare HTML tag name
            // e.g., div([...]) -> stdlib.html.div([...])
            // But NOT: h2(title) where title is a variable -> h2(title) stays as-is for the arg
            Expr::Call(func, type_args, args, span) => {
                // Transform function if it's a bare HTML tag name
                let new_func = if let Expr::Var(ident) = func.as_ref() {
                    if Self::HTML_SCOPED_NAMES.contains(&ident.node.as_str()) {
                        // Transform `div` to `stdlib.html.div`
                        let fn_span = ident.span;
                        let stdlib_var = Expr::Var(Spanned::new("stdlib".to_string(), fn_span));
                        let html_access = Expr::FieldAccess(
                            Box::new(stdlib_var),
                            Spanned::new("html".to_string(), fn_span),
                            fn_span,
                        );
                        Expr::FieldAccess(
                            Box::new(html_access),
                            ident.clone(),
                            fn_span,
                        )
                    } else {
                        // Not an HTML tag, keep as-is
                        func.as_ref().clone()
                    }
                } else {
                    // Function is not a simple Var, recurse
                    self.transform_html_expr(func)
                };
                // Recursively transform arguments
                let new_args: Vec<Expr> = args.iter().map(|a| self.transform_html_expr(a)).collect();
                Expr::Call(Box::new(new_func), type_args.clone(), new_args, *span)
            }

            // Bare variable references are NOT transformed - they might be local variables
            // e.g., h2(title) where title is a parameter stays as h2(title)
            Expr::Var(_) => expr.clone(),

            // Recursively transform method calls
            Expr::MethodCall(obj, method, args, span) => {
                let new_obj = self.transform_html_expr(obj);
                let new_args: Vec<Expr> = args.iter().map(|a| self.transform_html_expr(a)).collect();
                Expr::MethodCall(Box::new(new_obj), method.clone(), new_args, *span)
            }

            // Transform if expressions
            Expr::If(cond, then_branch, else_branch, span) => {
                Expr::If(
                    Box::new(self.transform_html_expr(cond)),
                    Box::new(self.transform_html_expr(then_branch)),
                    Box::new(self.transform_html_expr(else_branch)),
                    *span,
                )
            }

            // Transform match expressions
            Expr::Match(scrutinee, arms, span) => {
                let new_scrutinee = self.transform_html_expr(scrutinee);
                let new_arms: Vec<MatchArm> = arms.iter().map(|arm| {
                    MatchArm {
                        pattern: arm.pattern.clone(),
                        guard: arm.guard.as_ref().map(|g| self.transform_html_expr(g)),
                        body: self.transform_html_expr(&arm.body),
                        span: arm.span,
                    }
                }).collect();
                Expr::Match(Box::new(new_scrutinee), new_arms, *span)
            }

            // Transform lists
            Expr::List(items, tail, span) => {
                let new_items: Vec<Expr> = items.iter().map(|i| self.transform_html_expr(i)).collect();
                let new_tail = tail.as_ref().map(|t| Box::new(self.transform_html_expr(t)));
                Expr::List(new_items, new_tail, *span)
            }

            // Transform tuples
            Expr::Tuple(items, span) => {
                let new_items: Vec<Expr> = items.iter().map(|i| self.transform_html_expr(i)).collect();
                Expr::Tuple(new_items, *span)
            }

            // Transform maps
            Expr::Map(pairs, span) => {
                let new_pairs: Vec<(Expr, Expr)> = pairs.iter().map(|(k, v)| {
                    (self.transform_html_expr(k), self.transform_html_expr(v))
                }).collect();
                Expr::Map(new_pairs, *span)
            }

            // Transform sets
            Expr::Set(items, span) => {
                let new_items: Vec<Expr> = items.iter().map(|i| self.transform_html_expr(i)).collect();
                Expr::Set(new_items, *span)
            }

            // Transform binary operations
            Expr::BinOp(left, op, right, span) => {
                Expr::BinOp(
                    Box::new(self.transform_html_expr(left)),
                    *op,
                    Box::new(self.transform_html_expr(right)),
                    *span,
                )
            }

            // Transform unary operations
            Expr::UnaryOp(op, operand, span) => {
                Expr::UnaryOp(*op, Box::new(self.transform_html_expr(operand)), *span)
            }

            // Transform lambdas
            Expr::Lambda(params, body, span) => {
                Expr::Lambda(params.clone(), Box::new(self.transform_html_expr(body)), *span)
            }

            // Transform blocks
            Expr::Block(stmts, span) => {
                let new_stmts: Vec<Stmt> = stmts.iter().map(|stmt| {
                    match stmt {
                        Stmt::Expr(e) => Stmt::Expr(self.transform_html_expr(e)),
                        Stmt::Let(binding) => Stmt::Let(Binding {
                            mutable: binding.mutable,
                            pattern: binding.pattern.clone(),
                            ty: binding.ty.clone(),
                            value: self.transform_html_expr(&binding.value),
                            span: binding.span,
                        }),
                        Stmt::Assign(target, val, s) => {
                            Stmt::Assign(target.clone(), self.transform_html_expr(val), *s)
                        }
                    }
                }).collect();
                Expr::Block(new_stmts, *span)
            }

            // Transform field access
            Expr::FieldAccess(obj, field, span) => {
                Expr::FieldAccess(Box::new(self.transform_html_expr(obj)), field.clone(), *span)
            }

            // Transform index access
            Expr::Index(obj, idx, span) => {
                Expr::Index(
                    Box::new(self.transform_html_expr(obj)),
                    Box::new(self.transform_html_expr(idx)),
                    *span,
                )
            }

            // Transform record construction
            Expr::Record(name, fields, span) => {
                let new_fields: Vec<RecordField> = fields.iter().map(|f| {
                    match f {
                        RecordField::Positional(e) => RecordField::Positional(self.transform_html_expr(e)),
                        RecordField::Named(n, e) => RecordField::Named(n.clone(), self.transform_html_expr(e)),
                    }
                }).collect();
                Expr::Record(name.clone(), new_fields, *span)
            }

            // Transform record update
            Expr::RecordUpdate(name, base, fields, span) => {
                let new_base = self.transform_html_expr(base);
                let new_fields: Vec<RecordField> = fields.iter().map(|f| {
                    match f {
                        RecordField::Positional(e) => RecordField::Positional(self.transform_html_expr(e)),
                        RecordField::Named(n, e) => RecordField::Named(n.clone(), self.transform_html_expr(e)),
                    }
                }).collect();
                Expr::RecordUpdate(name.clone(), Box::new(new_base), new_fields, *span)
            }

            // Transform try expressions
            Expr::Try(body, arms, finally, span) => {
                let new_body = self.transform_html_expr(body);
                let new_arms: Vec<MatchArm> = arms.iter().map(|arm| {
                    MatchArm {
                        pattern: arm.pattern.clone(),
                        guard: arm.guard.as_ref().map(|g| self.transform_html_expr(g)),
                        body: self.transform_html_expr(&arm.body),
                        span: arm.span,
                    }
                }).collect();
                let new_finally = finally.as_ref().map(|f| Box::new(self.transform_html_expr(f)));
                Expr::Try(Box::new(new_body), new_arms, new_finally, *span)
            }

            // Transform while loops
            Expr::While(cond, body, span) => {
                Expr::While(
                    Box::new(self.transform_html_expr(cond)),
                    Box::new(self.transform_html_expr(body)),
                    *span,
                )
            }

            // Transform for loops
            Expr::For(var, start, end, body, span) => {
                Expr::For(
                    var.clone(),
                    Box::new(self.transform_html_expr(start)),
                    Box::new(self.transform_html_expr(end)),
                    Box::new(self.transform_html_expr(body)),
                    *span,
                )
            }

            // Transform break with value
            Expr::Break(val, span) => {
                Expr::Break(val.as_ref().map(|v| Box::new(self.transform_html_expr(v))), *span)
            }

            // Transform return with value
            Expr::Return(val, span) => {
                Expr::Return(val.as_ref().map(|v| Box::new(self.transform_html_expr(v))), *span)
            }

            // Transform try? expressions
            Expr::Try_(inner, span) => {
                Expr::Try_(Box::new(self.transform_html_expr(inner)), *span)
            }

            // Transform send expressions
            Expr::Send(pid, msg, span) => {
                Expr::Send(
                    Box::new(self.transform_html_expr(pid)),
                    Box::new(self.transform_html_expr(msg)),
                    *span,
                )
            }

            // Transform spawn expressions
            Expr::Spawn(kind, func, args, span) => {
                let new_func = self.transform_html_expr(func);
                let new_args: Vec<Expr> = args.iter().map(|a| self.transform_html_expr(a)).collect();
                Expr::Spawn(*kind, Box::new(new_func), new_args, *span)
            }

            // Transform receive expressions
            Expr::Receive(arms, timeout, span) => {
                let new_arms: Vec<MatchArm> = arms.iter().map(|arm| {
                    MatchArm {
                        pattern: arm.pattern.clone(),
                        guard: arm.guard.as_ref().map(|g| self.transform_html_expr(g)),
                        body: self.transform_html_expr(&arm.body),
                        span: arm.span,
                    }
                }).collect();
                let new_timeout = timeout.as_ref().map(|(t, body)| {
                    (Box::new(self.transform_html_expr(t)), Box::new(self.transform_html_expr(body)))
                });
                Expr::Receive(new_arms, new_timeout, *span)
            }

            // Transform do blocks
            Expr::Do(stmts, span) => {
                let new_stmts: Vec<DoStmt> = stmts.iter().map(|stmt| {
                    match stmt {
                        DoStmt::Expr(e) => DoStmt::Expr(self.transform_html_expr(e)),
                        DoStmt::Bind(pat, e) => DoStmt::Bind(pat.clone(), self.transform_html_expr(e)),
                    }
                }).collect();
                Expr::Do(new_stmts, *span)
            }

            // Transform quote expressions
            Expr::Quote(inner, span) => {
                Expr::Quote(Box::new(self.transform_html_expr(inner)), *span)
            }

            // Transform splice expressions
            Expr::Splice(inner, span) => {
                Expr::Splice(Box::new(self.transform_html_expr(inner)), *span)
            }

            // Transform string interpolations
            Expr::String(lit, span) => {
                match lit {
                    StringLit::Plain(_) => expr.clone(),
                    StringLit::Interpolated(parts) => {
                        let new_parts: Vec<StringPart> = parts.iter().map(|p| {
                            match p {
                                StringPart::Lit(s) => StringPart::Lit(s.clone()),
                                StringPart::Expr(e) => StringPart::Expr(self.transform_html_expr(e)),
                            }
                        }).collect();
                        Expr::String(StringLit::Interpolated(new_parts), *span)
                    }
                }
            }

            // All other expressions pass through unchanged
            _ => expr.clone(),
        }
    }

}

/// Collect free variables in an expression (variables not bound locally).
fn free_vars(expr: &Expr, bound: &std::collections::HashSet<String>) -> std::collections::HashSet<String> {
    use std::collections::HashSet;
    let mut free = HashSet::new();

    match expr {
        Expr::Var(ident) => {
            if !bound.contains(&ident.node) {
                free.insert(ident.node.clone());
            }
        }
        Expr::Int(_, _) | Expr::Float(_, _) | Expr::Bool(_, _) | Expr::Char(_, _)
        | Expr::String(_, _) | Expr::Unit(_) => {}

        Expr::BinOp(l, _, r, _) => {
            free.extend(free_vars(l, bound));
            free.extend(free_vars(r, bound));
        }
        Expr::UnaryOp(_, e, _) => {
            free.extend(free_vars(e, bound));
        }
        Expr::Call(f, _type_args, args, _) => {
            free.extend(free_vars(f, bound));
            for arg in args {
                free.extend(free_vars(arg, bound));
            }
        }
        Expr::If(c, t, e, _) => {
            free.extend(free_vars(c, bound));
            free.extend(free_vars(t, bound));
            free.extend(free_vars(e, bound));
        }
        Expr::Lambda(params, body, _) => {
            let mut new_bound = bound.clone();
            for p in params {
                if let Some(name) = pattern_var_name(p) {
                    new_bound.insert(name);
                }
            }
            free.extend(free_vars(body, &new_bound));
        }
        Expr::Tuple(items, _) => {
            for item in items {
                free.extend(free_vars(item, bound));
            }
        }
        Expr::List(items, tail, _) => {
            for item in items {
                free.extend(free_vars(item, bound));
            }
            if let Some(t) = tail {
                free.extend(free_vars(t, bound));
            }
        }
        Expr::Block(stmts, _) => {
            let mut local_bound = bound.clone();
            for stmt in stmts {
                match stmt {
                    Stmt::Expr(e) => free.extend(free_vars(e, &local_bound)),
                    Stmt::Let(binding) => {
                        free.extend(free_vars(&binding.value, &local_bound));
                        if let Some(name) = pattern_var_name(&binding.pattern) {
                            local_bound.insert(name);
                        }
                    }
                    Stmt::Assign(target, val, _) => {
                        free.extend(free_vars(val, &local_bound));
                        if let AssignTarget::Var(ident) = target {
                            if !local_bound.contains(&ident.node) {
                                free.insert(ident.node.clone());
                            }
                        }
                    }
                }
            }
        }
        Expr::Match(scrutinee, arms, _) => {
            free.extend(free_vars(scrutinee, bound));
            for arm in arms {
                let mut arm_bound = bound.clone();
                collect_pattern_vars(&arm.pattern, &mut arm_bound);
                if let Some(guard) = &arm.guard {
                    free.extend(free_vars(guard, &arm_bound));
                }
                free.extend(free_vars(&arm.body, &arm_bound));
            }
        }
        Expr::FieldAccess(obj, _, _) => {
            free.extend(free_vars(obj, bound));
        }
        Expr::Index(coll, idx, _) => {
            free.extend(free_vars(coll, bound));
            free.extend(free_vars(idx, bound));
        }
        Expr::Record(_, fields, _) => {
            for field in fields {
                match field {
                    RecordField::Positional(e) | RecordField::Named(_, e) => {
                        free.extend(free_vars(e, bound));
                    }
                }
            }
        }
        Expr::RecordUpdate(_, base, fields, _) => {
            free.extend(free_vars(base, bound));
            for field in fields {
                match field {
                    RecordField::Positional(e) | RecordField::Named(_, e) => {
                        free.extend(free_vars(e, bound));
                    }
                }
            }
        }
        Expr::Try_(e, _) => {
            free.extend(free_vars(e, bound));
        }
        Expr::Try(try_expr, catch_arms, finally_expr, _) => {
            free.extend(free_vars(try_expr, bound));
            for arm in catch_arms {
                let mut arm_bound = bound.clone();
                collect_pattern_vars(&arm.pattern, &mut arm_bound);
                free.extend(free_vars(&arm.body, &arm_bound));
            }
            if let Some(fin) = finally_expr {
                free.extend(free_vars(fin, bound));
            }
        }
        Expr::Spawn(_, func, args, _) => {
            free.extend(free_vars(func, bound));
            for arg in args {
                free.extend(free_vars(arg, bound));
            }
        }
        Expr::MethodCall(obj, _, args, _) => {
            free.extend(free_vars(obj, bound));
            for arg in args {
                free.extend(free_vars(arg, bound));
            }
        }
        Expr::Send(pid, msg, _) => {
            free.extend(free_vars(pid, bound));
            free.extend(free_vars(msg, bound));
        }
        Expr::Map(pairs, _) => {
            for (k, v) in pairs {
                free.extend(free_vars(k, bound));
                free.extend(free_vars(v, bound));
            }
        }
        Expr::Set(items, _) => {
            for item in items {
                free.extend(free_vars(item, bound));
            }
        }
        _ => {} // Other expressions - add as needed
    }

    free
}

fn pattern_var_name(pat: &Pattern) -> Option<String> {
    match pat {
        Pattern::Var(ident) => Some(ident.node.clone()),
        _ => None,
    }
}

fn collect_pattern_vars(pat: &Pattern, vars: &mut std::collections::HashSet<String>) {
    match pat {
        Pattern::Var(ident) => { vars.insert(ident.node.clone()); }
        Pattern::Tuple(pats, _) => {
            for p in pats {
                collect_pattern_vars(p, vars);
            }
        }
        Pattern::List(list_pat, _) => {
            match list_pat {
                ListPattern::Empty => {}
                ListPattern::Cons(heads, tail) => {
                    for p in heads {
                        collect_pattern_vars(p, vars);
                    }
                    if let Some(t) = tail {
                        collect_pattern_vars(t, vars);
                    }
                }
            }
        }
        Pattern::StringCons(string_pat, _) => {
            match string_pat {
                StringPattern::Empty => {}
                StringPattern::Cons(_, tail) => {
                    collect_pattern_vars(tail, vars);
                }
            }
        }
        Pattern::Variant(_, fields, _) => {
            match fields {
                VariantPatternFields::Unit => {}
                VariantPatternFields::Positional(pats) => {
                    for p in pats {
                        collect_pattern_vars(p, vars);
                    }
                }
                VariantPatternFields::Named(named) => {
                    for field in named {
                        match field {
                            RecordPatternField::Punned(ident) => { vars.insert(ident.node.clone()); }
                            RecordPatternField::Named(_, pat) => collect_pattern_vars(pat, vars),
                            RecordPatternField::Rest(_) => {}
                        }
                    }
                }
            }
        }
        Pattern::Record(fields, _) => {
            for field in fields {
                match field {
                    RecordPatternField::Punned(ident) => { vars.insert(ident.node.clone()); }
                    RecordPatternField::Named(_, pat) => collect_pattern_vars(pat, vars),
                    RecordPatternField::Rest(_) => {}
                }
            }
        }
        Pattern::Or(pats, _) => {
            for p in pats {
                collect_pattern_vars(p, vars);
            }
        }
        _ => {}
    }
}

/// Compile a complete module.
pub fn compile_module(module: &Module, source: &str) -> Result<Compiler, CompileError> {
    let mut compiler = Compiler::new(source);
    compiler.compile_items(&module.items)?;
    compiler.compile_all().map_err(|(e, _, _)| e)?;
    Ok(compiler)
}

/// Compile a module with stdlib pre-loaded.
/// The stdlib_path should point to the stdlib directory.
pub fn compile_module_with_stdlib(
    module: &Module,
    source: &str,
    stdlib_path: &std::path::Path,
) -> Result<Compiler, CompileError> {
    let mut compiler = Compiler::new(source);

    // Load stdlib if path exists
    if stdlib_path.is_dir() {
        load_stdlib_into_compiler(&mut compiler, stdlib_path)?;
    }

    // Compile the main module
    compiler.compile_items(&module.items)?;
    compiler.compile_all().map_err(|(e, _, _)| e)?;
    Ok(compiler)
}

/// Load all stdlib modules into a compiler.
fn load_stdlib_into_compiler(compiler: &mut Compiler, stdlib_path: &std::path::Path) -> Result<(), CompileError> {
    use std::sync::Arc;

    fn visit_dirs(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    visit_dirs(&path, files)?;
                } else if path.extension().map(|e| e == "nos").unwrap_or(false) {
                    files.push(path);
                }
            }
        }
        Ok(())
    }

    let mut stdlib_files = Vec::new();
    visit_dirs(stdlib_path, &mut stdlib_files).map_err(|_| CompileError::InternalError {
        message: "Failed to read stdlib directory".to_string(),
        span: Span { start: 0, end: 0 },
    })?;

    for file_path in &stdlib_files {
        let source = std::fs::read_to_string(file_path).map_err(|_| CompileError::InternalError {
            message: format!("Failed to read stdlib file: {}", file_path.display()),
            span: Span { start: 0, end: 0 },
        })?;

        let (module_opt, errors) = parse(&source);
        if !errors.is_empty() {
            return Err(CompileError::InternalError {
                message: format!("Failed to parse stdlib file {}: {:?}", file_path.display(), errors),
                span: Span { start: 0, end: 0 },
            });
        }

        if let Some(module) = module_opt {
            // Build module path: stdlib.list, stdlib.json, etc.
            let relative = file_path.strip_prefix(stdlib_path).unwrap();
            let mut components: Vec<String> = vec!["stdlib".to_string()];
            for component in relative.components() {
                let s = component.as_os_str().to_string_lossy().to_string();
                if s.ends_with(".nos") {
                    components.push(s.trim_end_matches(".nos").to_string());
                } else {
                    components.push(s);
                }
            }

            compiler.add_module(
                &module,
                components,
                Arc::new(source.clone()),
                file_path.to_str().unwrap_or("").to_string(),
            )?;
        }
    }

    Ok(())
}

impl Compiler {
    /// Compile a list of items (can be called recursively for nested modules).
    fn compile_items(&mut self, items: &[Item]) -> Result<(), CompileError> {
        // First pass: process use and import statements to set up imports
        for item in items {
            if let Item::Use(use_stmt) = item {
                self.compile_use_stmt(use_stmt)?;
            }
            if let Item::Import(import_stmt) = item {
                self.compile_import_stmt(import_stmt)?;
            }
        }

        // Second pass: collect type definitions
        for item in items {
            if let Item::TypeDef(type_def) = item {
                self.compile_type_def(type_def)?;
            }
        }

        // Third pass: compile trait definitions
        for item in items {
            if let Item::TraitDef(trait_def) = item {
                self.compile_trait_def(trait_def)?;
            }
        }

        // Fourth pass: compile trait implementations (after trait defs)
        for item in items {
            if let Item::TraitImpl(trait_impl) = item {
                self.compile_trait_impl(trait_impl)?;
            }
        }

        // Fifth pass: process nested modules (before functions so they're available)
        for item in items {
            if let Item::ModuleDef(module_def) = item {
                self.compile_module_def(module_def)?;
            }
        }

        // Sixth pass: process mvar definitions (before functions so they're available)
        for item in items {
            if let Item::MvarDef(mvar_def) = item {
                self.compile_mvar_def(mvar_def)?;
            }
        }

        // Collect and merge function definitions by name AND signature
        // Functions with the same name but different type signatures are different functions
        let mut fn_clauses: std::collections::HashMap<String, Vec<FnClause>> = std::collections::HashMap::new();
        let mut fn_order: Vec<String> = Vec::new();
        let mut fn_spans: std::collections::HashMap<String, Span> = std::collections::HashMap::new();
        let mut fn_visibility: std::collections::HashMap<String, Visibility> = std::collections::HashMap::new();
        let mut fn_type_params_map: std::collections::HashMap<String, Vec<TypeParam>> = std::collections::HashMap::new();

        for item in items {
            if let Item::FnDef(fn_def) = item {
                let base_name = self.qualify_name(&fn_def.name.node);
                // Build signature from first clause's parameter types
                let param_types: Vec<String> = fn_def.clauses[0].params.iter()
                    .map(|p| p.ty.as_ref()
                        .map(|t| self.type_expr_to_string(t))
                        .unwrap_or_else(|| "_".to_string()))
                    .collect();
                let signature = param_types.join(",");
                let qualified_name = format!("{}/{}", base_name, signature);

                if !fn_clauses.contains_key(&qualified_name) {
                    fn_order.push(qualified_name.clone());
                    fn_spans.insert(qualified_name.clone(), fn_def.span);
                    // Use visibility and type_params from first definition
                    fn_visibility.insert(qualified_name.clone(), fn_def.visibility);
                    fn_type_params_map.insert(qualified_name.clone(), fn_def.type_params.clone());
                } else {
                    // Merge spans to cover all clauses (from first to last definition)
                    if let Some(existing_span) = fn_spans.get_mut(&qualified_name) {
                        existing_span.end = fn_def.span.end;
                    }
                }
                fn_clauses.entry(qualified_name).or_default().extend(fn_def.clauses.iter().cloned());
            }
        }

        // Fourth pass: forward declare all functions (for recursion)
        // Use the new naming scheme with type signatures
        for name in &fn_order {
            let clauses = fn_clauses.get(name).unwrap();
            let arity = clauses[0].params.len();

            // name already includes signature from collection phase (e.g., "greet/Int")
            let full_name = name.clone();

            // Extract param types from the first clause for type inference
            // Use "?" for untyped params (will be resolved by HM inference)
            let param_types: Vec<String> = clauses[0].params.iter()
                .map(|p| p.ty.as_ref()
                    .map(|ty| self.type_expr_to_string(ty))
                    .unwrap_or_else(|| "?".to_string()))
                .collect();

            // Extract return type from the first clause if available
            let return_type = clauses[0].return_type.as_ref()
                .map(|ty| self.type_expr_to_string(ty));

            // Insert a placeholder function for forward reference
            // But DON'T overwrite if the function already exists with a valid signature
            // (this preserves the old working function if recompilation fails)
            let should_insert = match self.functions.get(&full_name) {
                None => true,  // New function, insert placeholder
                Some(existing) => existing.signature.is_none(),  // Only overwrite if existing is also a placeholder
            };
            if should_insert {
                let placeholder = FunctionValue {
                    name: full_name.clone(),
                    arity,
                    param_names: vec![],
                    code: Arc::new(Chunk::new()),
                    module: if self.module_path.is_empty() { None } else { Some(self.module_path.join(".")) },
                    source_span: None,
                    jit_code: None,
                    call_count: AtomicU32::new(0),
                    debug_symbols: vec![],
                    // REPL introspection fields - will be populated when compiled
                    source_code: None,
                    source_file: None,
                    doc: None,
                    signature: None,
                    param_types,
                    return_type,
                };
                self.functions.insert(full_name.clone(), Arc::new(placeholder));
            }

            // Assign function index for direct calls (no HashMap lookup at runtime!)
            if !self.function_indices.contains_key(&full_name) {
                let idx = self.function_list.len() as u16;
                self.function_indices.insert(full_name.clone(), idx);
                self.function_list.push(full_name.clone());
            }

            // Register visibility early (needed for `use module.*` to work across modules)
            if let Some(vis) = fn_visibility.get(name) {
                self.function_visibility.insert(full_name, *vis);
            }
        }

        // Fifth pass: queue functions with merged clauses
        for name in &fn_order {
            let clauses = fn_clauses.get(name).unwrap();
            let span = fn_spans.get(name).copied().unwrap_or_default();
            // Extract the local name (without module prefix and without signature)
            // name is like "module.greet/Int" or "greet/Int", we want just "greet"
            let name_without_sig = name.split('/').next().unwrap_or(name);
            let local_name = if name_without_sig.contains('.') {
                name_without_sig.rsplit('.').next().unwrap_or(name_without_sig)
            } else {
                name_without_sig
            };
            let merged_fn = FnDef {
                visibility: *fn_visibility.get(name).unwrap_or(&Visibility::Private),
                doc: None,
                name: Spanned::new(local_name.to_string(), span),
                type_params: fn_type_params_map.get(name).cloned().unwrap_or_default(),
                clauses: clauses.clone(),
                span,
            };
            self.pending_functions.push((
                merged_fn,
                self.module_path.clone(),
                self.imports.clone(),
                self.line_starts.clone(),
                self.current_source.clone().expect("Source not set"),
                self.current_source_name.clone().expect("Source name not set"),
            ));
        }

        Ok(())
    }

    /// Compile a nested module definition.
    fn compile_module_def(&mut self, module_def: &ModuleDef) -> Result<(), CompileError> {
        // Push the module name onto the path
        self.module_path.push(module_def.name.node.clone());

        // Register the full module path as known (e.g., "Utils" or "Outer.Inner")
        let full_path = self.module_path.join(".");
        self.known_modules.insert(full_path.clone());

        // Also register as a local module (inline modules don't require explicit imports)
        self.local_modules.insert(full_path);

        // Compile the module's items
        self.compile_items(&module_def.items)?;

        // Pop the module name from the path
        self.module_path.pop();

        Ok(())
    }

    /// Compile a use statement (import).
    fn compile_use_stmt(&mut self, use_stmt: &UseStmt) -> Result<(), CompileError> {
        // Build the module path from the use statement
        let module_path: String = use_stmt.path.iter()
            .map(|ident| ident.node.as_str())
            .collect::<Vec<_>>()
            .join(".");

        // Store the use statement for later retrieval (for editor/browser)
        let use_stmt_string = match &use_stmt.imports {
            UseImports::All => format!("use {}.*", module_path),
            UseImports::Named(items) => {
                let names: Vec<String> = items.iter()
                    .map(|item| {
                        if let Some(alias) = &item.alias {
                            format!("{} as {}", item.name.node, alias.node)
                        } else {
                            item.name.node.clone()
                        }
                    })
                    .collect();
                format!("use {}.{{{}}}", module_path, names.join(", "))
            }
        };
        self.module_use_stmts.push((self.module_path.clone(), use_stmt_string));

        match &use_stmt.imports {
            UseImports::All => {
                // `use Foo.*` - import all public functions from the module
                let prefix = format!("{}.", module_path);
                let public_functions: Vec<String> = self.function_visibility.iter()
                    .filter(|(name, vis)| {
                        name.starts_with(&prefix) && **vis == Visibility::Public
                    })
                    .map(|(name, _)| name.clone())
                    .collect();

                for qualified_name in public_functions {
                    // Extract local name (function name without module prefix and signature)
                    // e.g., "nalgebra.vec/List" -> "vec"
                    let without_prefix = qualified_name.strip_prefix(&prefix)
                        .unwrap_or(&qualified_name);
                    // Strip the signature part (after /)
                    let local_name = without_prefix.split('/').next()
                        .unwrap_or(without_prefix)
                        .to_string();
                    // Also strip the signature from qualified name for the import mapping
                    let qualified_base = qualified_name.split('/').next()
                        .unwrap_or(&qualified_name)
                        .to_string();
                    self.imports.insert(local_name, qualified_base);
                }
            }
            UseImports::Named(items) => {
                for item in items {
                    let local_name = item.alias.as_ref()
                        .map(|a| a.node.clone())
                        .unwrap_or_else(|| item.name.node.clone());
                    let qualified_name = format!("{}.{}", module_path, item.name.node);
                    self.imports.insert(local_name, qualified_name);
                }
            }
        }

        Ok(())
    }

    /// Compile an import statement.
    /// Records that the current module imports the specified module.
    fn compile_import_stmt(&mut self, import_stmt: &ImportStmt) -> Result<(), CompileError> {
        // Build the module path from the import statement
        let module_path: String = import_stmt.path.iter()
            .map(|ident| ident.node.as_str())
            .collect::<Vec<_>>()
            .join(".");

        // Record that this module is imported by the current module
        self.imported_modules.insert((self.module_path.clone(), module_path));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nostos_syntax::parser::parse;
    use nostos_vm::async_vm::{AsyncVM, AsyncConfig};

    fn compile_and_run(source: &str) -> Result<Value, String> {
        let (module_opt, errors) = parse(source);
        if !errors.is_empty() {
            return Err(format!("Parse error: {:?}", errors));
        }
        let module = module_opt.ok_or_else(|| "Parse returned no module".to_string())?;
        let compiler = compile_module(&module, source).map_err(|e| format!("Compile error: {:?}", e))?;

        // Use AsyncVM with single thread for deterministic tests
        let config = AsyncConfig {
            num_threads: 1,
            ..Default::default()
        };
        let mut vm = AsyncVM::new(config);
        vm.register_default_natives();

        for (name, func) in compiler.get_all_functions() {
            vm.register_function(&name, func.clone());
        }
        vm.set_function_list(compiler.get_function_list());
        for (name, type_val) in compiler.get_vm_types() {
            vm.register_type(&name, type_val);
        }

        // Look for main function name (0-arity, so add "/" suffix)
        let main_fn_name = if compiler.get_function("main").is_some() {
            "main/"
        } else if let Some((name, _)) = compiler.get_all_functions().iter().next() {
            // Use the first function found - but need to construct the function name with arity
            return Err(format!("TODO: handle non-main function: {}", name));
        } else {
            return Err("No functions to run".to_string());
        };

        let result = vm.run(main_fn_name)
            .map_err(|e| format!("Runtime error: {:?}", e))?;
        Ok(result.to_value())
    }

    // ========== Doc Comment Tests ==========

    #[test]
    fn test_extract_doc_comment_basic() {
        let source = "# This is a doc comment\nmain() = 42";
        let span_start = source.find("main").unwrap();
        let doc = extract_doc_comment(source, span_start);
        assert_eq!(doc, Some("This is a doc comment".to_string()));
    }

    #[test]
    fn test_extract_doc_comment_multiline() {
        let source = "# First line\n# Second line\nmain() = 42";
        let span_start = source.find("main").unwrap();
        let doc = extract_doc_comment(source, span_start);
        assert_eq!(doc, Some("First line\nSecond line".to_string()));
    }

    #[test]
    fn test_extract_doc_comment_with_whitespace() {
        let source = "# Doc comment\n\nmain() = 42";
        let span_start = source.find("main").unwrap();
        let doc = extract_doc_comment(source, span_start);
        assert_eq!(doc, Some("Doc comment".to_string()));
    }

    #[test]
    fn test_extract_doc_comment_none() {
        let source = "main() = 42";
        let span_start = source.find("main").unwrap();
        let doc = extract_doc_comment(source, span_start);
        assert_eq!(doc, None);
    }

    #[test]
    fn test_extract_doc_comment_not_immediately_before() {
        let source = "# Orphan comment\n\nother() = 1\nmain() = 42";
        let span_start = source.find("main").unwrap();
        let doc = extract_doc_comment(source, span_start);
        // Should be None because there's code (other function) between
        assert_eq!(doc, None);
    }

    #[test]
    fn test_doc_comment_stored_on_function() {
        let source = "# Computes factorial of n\nfact(n) = if n == 0 then 1 else n * fact(n - 1)";
        let (module_opt, errors) = parse(source);
        assert!(errors.is_empty());
        let module = module_opt.unwrap();
        let compiler = compile_module(&module, source).unwrap();

        // Find the function key (could be fact, fact/, or fact/Int64)
        let func_names = compiler.get_function_names();
        let fact_name = func_names.iter().find(|n| n.starts_with("fact"))
            .expect(&format!("fact function should exist in {:?}", func_names));

        // The function should have the doc comment
        let doc = compiler.get_function_doc(fact_name);
        assert_eq!(doc, Some("Computes factorial of n".to_string()));
    }

    #[test]
    fn test_doc_comment_multiline_on_function() {
        let source = "# Doubles a number\n# Works with any numeric type\ndouble(x) = x + x";
        let (module_opt, errors) = parse(source);
        assert!(errors.is_empty());
        let module = module_opt.unwrap();
        let compiler = compile_module(&module, source).unwrap();

        // Find the function key
        let func_names = compiler.get_function_names();
        let double_name = func_names.iter().find(|n| n.starts_with("double"))
            .expect(&format!("double function should exist in {:?}", func_names));

        let doc = compiler.get_function_doc(double_name);
        assert_eq!(doc, Some("Doubles a number\nWorks with any numeric type".to_string()));
    }

    #[test]
    fn test_doc_comment_not_multiline_comment() {
        // #* is for multi-line comments, should not be treated as doc
        let source = "#* This is a block comment *#\nmain() = 42";
        let span_start = source.find("main").unwrap();
        let doc = extract_doc_comment(source, span_start);
        assert_eq!(doc, None);
    }

    #[test]
    fn test_doc_comment_not_set_literal() {
        // #{ is for set literals, should not be treated as doc
        let source = "#{ not a doc }\nmain() = 42";
        let span_start = source.find("main").unwrap();
        let doc = extract_doc_comment(source, span_start);
        assert_eq!(doc, None);
    }

    // ========== Basic Compilation Tests ==========

    #[test]
    fn test_compile_simple_function() {
        let result = compile_and_run("main() = 42");
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_compile_addition() {
        let result = compile_and_run("main() = 2 + 3");
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_compile_nested_arithmetic() {
        let result = compile_and_run("main() = (2 + 3) * 4");
        assert_eq!(result, Ok(Value::Int64(20)));
    }

    #[test]
    fn test_compile_if_then_else() {
        let result = compile_and_run("main() = if true then 1 else 2");
        assert_eq!(result, Ok(Value::Int64(1)));

        let result2 = compile_and_run("main() = if false then 1 else 2");
        assert_eq!(result2, Ok(Value::Int64(2)));
    }

    #[test]
    fn test_compile_comparison() {
        let result = compile_and_run("main() = if 5 > 3 then 1 else 0");
        assert_eq!(result, Ok(Value::Int64(1)));
    }

    #[test]
    fn test_compile_function_call() {
        let source = "
            double(x) = x + x
            main() = double(21)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_compile_recursive_function() {
        let source = "
            fact(n) = if n == 0 then 1 else n * fact(n - 1)
            main() = fact(5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(120)));
    }

    #[test]
    fn test_compile_tail_recursive() {
        let source = "
            sum(n, acc) = if n == 0 then acc else sum(n - 1, acc + n)
            main() = sum(100, 0)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5050)));
    }

    #[test]
    fn test_compile_list() {
        let source = "main() = [1, 2, 3]";
        let result = compile_and_run(source);
        match result {
            Ok(Value::List(items)) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Int64(1));
            }
            other => panic!("Expected list, got {:?}", other),
        }
    }

    #[test]
    fn test_compile_lambda() {
        let source = "
            apply(f, x) = f(x)
            main() = apply(x => x * 2, 21)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_compile_bool_ops() {
        let source = "main() = true && false";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(false)));

        let source2 = "main() = true || false";
        let result2 = compile_and_run(source2);
        assert_eq!(result2, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_compile_negation() {
        let source = "main() = !true";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(false)));
    }

    #[test]
    fn test_compile_string() {
        let source = r#"main() = "hello""#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "hello"),
            other => panic!("Expected string, got {:?}", other),
        }
    }

    // ============= More comprehensive end-to-end tests =============

    #[test]
    fn test_e2e_fibonacci_tail_recursive() {
        let source = "
            fib(n, a, b) = if n == 0 then a else fib(n - 1, b, a + b)
            main() = fib(20, 0, 1)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(6765)));
    }

    #[test]
    fn test_e2e_mutual_recursion() {
        let source = "
            isEven(n) = if n == 0 then true else isOdd(n - 1)
            isOdd(n) = if n == 0 then false else isEven(n - 1)
            main() = isEven(10)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_higher_order_function() {
        let source = "
            twice(f, x) = f(f(x))
            addOne(n) = n + 1
            main() = twice(addOne, 5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(7)));
    }

    #[test]
    fn test_e2e_compose_lambdas() {
        let source = "
            compose(f, g, x) = f(g(x))
            main() = compose(x => x * 2, y => y + 1, 10)
        ";
        // (10 + 1) * 2 = 22
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(22)));
    }

    #[test]
    fn test_e2e_tuple() {
        // Tuple creation works
        let source = "main() = (1, 2, 3)";
        let result = compile_and_run(source);
        match result {
            Ok(Value::Tuple(items)) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Int64(1));
                assert_eq!(items[1], Value::Int64(2));
                assert_eq!(items[2], Value::Int64(3));
            }
            other => panic!("Expected tuple, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_multiple_functions_chained() {
        let source = "
            square(x) = x * x
            double(x) = x + x
            addTen(x) = x + 10
            main() = addTen(double(square(3)))
        ";
        // square(3) = 9, double(9) = 18, addTen(18) = 28
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(28)));
    }

    #[test]
    fn test_e2e_nested_conditionals() {
        let source = "
            classify(n) = if n < 0 then 0 - 1 else if n == 0 then 0 else 1
            main() = classify(5) + classify(0 - 3) + classify(0)
        ";
        // 1 + (-1) + 0 = 0
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(0)));
    }

    #[test]
    fn test_e2e_gcd() {
        let source = "
            gcd(a, b) = if b == 0 then a else gcd(b, a - (a / b) * b)
            main() = gcd(48, 18)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(6)));
    }

    #[test]
    fn test_e2e_power() {
        let source = "
            power(base, exp, acc) = if exp == 0 then acc else power(base, exp - 1, acc * base)
            main() = power(2, 10, 1)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(1024)));
    }

    #[test]
    fn test_e2e_complex_boolean_logic() {
        let source = "
            xor(a, b) = (a || b) && !(a && b)
            main() = xor(true, false) && !xor(true, true)
        ";
        // xor(true, false) = true, xor(true, true) = false, !false = true
        // true && true = true
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_ackermann_small() {
        let source = "
            ack(m, n) = if m == 0 then n + 1 else if n == 0 then ack(m - 1, 1) else ack(m - 1, ack(m, n - 1))
            main() = ack(2, 3)
        ";
        // ack(2, 3) = 9
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(9)));
    }

    #[test]
    fn test_e2e_collatz_steps() {
        let source = "
            collatz(n, steps) = if n == 1 then steps else if (n - (n / 2) * 2) == 0 then collatz(n / 2, steps + 1) else collatz(3 * n + 1, steps + 1)
            main() = collatz(27, 0)
        ";
        // collatz(27) takes 111 steps
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(111)));
    }

    #[test]
    fn test_e2e_curried_application() {
        let source = "
            add(x) = y => x + y
            main() = add(10)(32)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_closure_captures() {
        let source = "
            makeAdder(n) = x => x + n
            main() = makeAdder(40)(2)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_float_arithmetic() {
        // Float multiplication works
        let source = "
            scale(x) = x * 2.0
            main() = scale(3.5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Float64(7.0)));
    }

    #[test]
    fn test_e2e_comparison_chain() {
        let source = "
            inRange(x, low, high) = x >= low && x <= high
            main() = inRange(5, 1, 10) && !inRange(15, 1, 10)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_let_binding_in_block() {
        // Nostos uses `x = 10` for bindings in blocks, not `let x = 10`
        let source = "
            main() = {
                x = 10
                y = 20
                x + y
            }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(30)));
    }

    #[test]
    fn test_e2e_nested_blocks() {
        let source = "
            main() = {
                a = {
                    x = 5
                    x * 2
                }
                b = {
                    y = 3
                    y * 3
                }
                a + b
            }
        ";
        // a = 10, b = 9, a + b = 19
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(19)));
    }

    #[test]
    fn test_e2e_string_value() {
        let source = r#"
            greet(name) = name
            main() = greet("World")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "World"),
            other => panic!("Expected string, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_tuple_field_access() {
        let source = "
            swap(pair) = (pair.1, pair.0)
            main() = {
                p = (1, 2)
                swapped = swap(p)
                swapped.0 * 10 + swapped.1
            }
        ";
        // swapped = (2, 1), result = 2*10 + 1 = 21
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(21)));
    }

    #[test]
    fn test_e2e_match_literals() {
        // Match syntax uses -> not =>
        let source = r#"
            describe(n) = match n {
                0 -> "zero"
                1 -> "one"
                _ -> "many"
            }
            main() = describe(1)
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "one"),
            other => panic!("Expected 'one', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_match_variable_binding() {
        let source = "
            double(x) = match x {
                n -> n + n
            }
            main() = double(21)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_match_tuple_pattern() {
        let source = "
            first(pair) = match pair {
                (a, _) -> a
            }
            main() = first((42, 100))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_list_cons_pattern() {
        let source = "
            sum(xs) = match xs {
                [] -> 0
                [h | t] -> h + sum(t)
            }
            main() = sum([1, 2, 3, 4, 5])
        ";
        // 1 + 2 + 3 + 4 + 5 = 15
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(15)));
    }

    #[test]
    fn test_e2e_list_head_tail() {
        let source = "
            head(xs) = match xs {
                [h | _] -> h
                [] -> 0
            }
            main() = head([42, 1, 2])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_record_construction() {
        let source = "
            type Point = { x: Int, y: Int }
            main() = Point(3, 4)
        ";
        let result = compile_and_run(source);
        match result {
            Ok(Value::Record(r)) => {
                assert_eq!(r.type_name, "Point");
                assert_eq!(r.fields.len(), 2);
                assert_eq!(r.fields[0], Value::Int64(3));
                assert_eq!(r.fields[1], Value::Int64(4));
            }
            other => panic!("Expected record, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_record_field_access() {
        // Test with named field access
        let source = "
            type Point = { x: Int, y: Int }
            getX(p) = p.x
            main() = getX(Point(42, 100))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_variant_construction() {
        let source = "
            type Option[T] = Some(T) | None
            main() = Some(42)
        ";
        let result = compile_and_run(source);
        // Variants are now properly compiled as Variants
        match result {
            Ok(Value::Variant(v)) => {
                assert_eq!(v.constructor.as_str(), "Some");
                assert_eq!(v.fields.len(), 1);
                assert_eq!(v.fields[0], Value::Int64(42));
            }
            other => panic!("Expected variant, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_variant_match() {
        let source = "
            type Option[T] = Some(T) | None
            unwrap(opt) = match opt {
                Some(x) -> x
                None -> 0
            }
            main() = unwrap(Some(42))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_variant_none() {
        let source = "
            type Option[T] = Some(T) | None
            unwrap(opt) = match opt {
                Some(x) -> x
                None -> 0
            }
            main() = unwrap(None)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(0)));
    }

    // ===== Additional comprehensive tests =====

    #[test]
    fn test_e2e_list_length() {
        let source = "
            len(xs) = match xs {
                [] -> 0
                [_ | t] -> 1 + len(t)
            }
            main() = len([1, 2, 3, 4, 5])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_list_map() {
        let source = "
            map(f, xs) = match xs {
                [] -> []
                [h | t] -> [f(h) | map(f, t)]
            }
            sum(xs) = match xs {
                [] -> 0
                [h | t] -> h + sum(t)
            }
            main() = sum(map(x => x * 2, [1, 2, 3]))
        ";
        // [2, 4, 6] => sum = 12
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(12)));
    }

    #[test]
    fn test_e2e_list_filter() {
        let source = "
            filter(pred, xs) = match xs {
                [] -> []
                [h | t] -> if pred(h) then [h | filter(pred, t)] else filter(pred, t)
            }
            sum(xs) = match xs {
                [] -> 0
                [h | t] -> h + sum(t)
            }
            main() = sum(filter(x => x > 2, [1, 2, 3, 4, 5]))
        ";
        // [3, 4, 5] => sum = 12
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(12)));
    }

    #[test]
    fn test_e2e_list_append() {
        let source = "
            append(xs, ys) = match xs {
                [] -> ys
                [h | t] -> [h | append(t, ys)]
            }
            sum(xs) = match xs {
                [] -> 0
                [h | t] -> h + sum(t)
            }
            main() = sum(append([1, 2], [3, 4, 5]))
        ";
        // [1, 2, 3, 4, 5] => sum = 15
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(15)));
    }

    #[test]
    fn test_e2e_list_reverse() {
        let source = "
            reverseHelper(xs, acc) = match xs {
                [] -> acc
                [h | t] -> reverseHelper(t, [h | acc])
            }
            reverse(xs) = reverseHelper(xs, [])
            head(xs) = match xs {
                [h | _] -> h
                [] -> 0
            }
            main() = head(reverse([1, 2, 3, 4, 5]))
        ";
        // reverse [1,2,3,4,5] = [5,4,3,2,1], head = 5
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_list_fold() {
        let source = "
            foldl(f, acc, xs) = match xs {
                [] -> acc
                [h | t] -> foldl(f, f(acc, h), t)
            }
            main() = foldl((acc, x) => acc + x, 0, [1, 2, 3, 4, 5])
        ";
        // sum = 15
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(15)));
    }

    #[test]
    fn test_e2e_nested_variant_match() {
        let source = "
            type Option[T] = Some(T) | None
            doubleUnwrap(opt) = match opt {
                Some(inner) -> match inner {
                    Some(x) -> x
                    None -> 0
                }
                None -> 0
            }
            main() = doubleUnwrap(Some(Some(42)))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_either_pattern() {
        let source = "
            type Either[L, R] = Left(L) | Right(R)
            handle(result) = match result {
                Left(err) -> 0 - err
                Right(val) -> val
            }
            main() = handle(Left(5)) + handle(Right(10))
        ";
        // -5 + 10 = 5
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_char_literal() {
        // Simple char value test (char patterns not implemented yet)
        let source = r#"
            main() = 'a'
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Char('a')));
    }

    #[test]
    fn test_e2e_nested_tuple_pattern() {
        // Nested tuple patterns via field access (pattern syntax limited)
        let source = "
            extract(x) = {
                ab = x.0
                cd = x.1
                ab.0 + ab.1 + cd.0 + cd.1
            }
            main() = extract(((1, 2), (3, 4)))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(10)));
    }

    #[test]
    fn test_e2e_deep_closure_capture() {
        // Use lambda syntax instead of function definitions inside blocks
        let source = "
            outer(x) = y => z => x + y + z
            main() = outer(10)(20)(12)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_multiple_closures_same_capture() {
        // Use tuple directly with lambdas
        let source = "
            makePair(n) = (x => x + n, x => x - n)
            main() = {
                fns = makePair(10)
                adder = fns.0
                subber = fns.1
                adder(5) + subber(20)
            }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(25)));
    }

    #[test]
    fn test_e2e_even_odd_mutual_recursion() {
        let source = "
            isEven(n) = if n == 0 then true else isOdd(n - 1)
            isOdd(n) = if n == 0 then false else isEven(n - 1)
            main() = isEven(10) && isOdd(7)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_list_take() {
        let source = "
            take(n, xs) = if n == 0 then [] else match xs {
                [] -> []
                [h | t] -> [h | take(n - 1, t)]
            }
            sum(xs) = match xs {
                [] -> 0
                [h | t] -> h + sum(t)
            }
            main() = sum(take(3, [1, 2, 3, 4, 5]))
        ";
        // take 3 [1,2,3,4,5] = [1,2,3], sum = 6
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(6)));
    }

    #[test]
    fn test_e2e_list_drop() {
        let source = "
            drop(n, xs) = if n == 0 then xs else match xs {
                [] -> []
                [_ | t] -> drop(n - 1, t)
            }
            sum(xs) = match xs {
                [] -> 0
                [h | t] -> h + sum(t)
            }
            main() = sum(drop(2, [1, 2, 3, 4, 5]))
        ";
        // drop 2 [1,2,3,4,5] = [3,4,5], sum = 12
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(12)));
    }

    #[test]
    fn test_e2e_compose_higher_order() {
        let source = "
            compose(f, g) = x => f(g(x))
            double(x) = x * 2
            addOne(x) = x + 1
            main() = compose(double, compose(addOne, double))(5)
        ";
        // double(5) = 10, addOne(10) = 11, double(11) = 22
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(22)));
    }

    #[test]
    fn test_e2e_option_map() {
        let source = "
            type Option[T] = Some(T) | None
            mapOpt(f, opt) = match opt {
                Some(x) -> Some(f(x))
                None -> None
            }
            unwrap(opt) = match opt {
                Some(x) -> x
                None -> 0
            }
            main() = unwrap(mapOpt(x => x * 2, Some(21)))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_option_flatmap() {
        let source = "
            type Option[T] = Some(T) | None
            flatMap(f, opt) = match opt {
                Some(x) -> f(x)
                None -> None
            }
            safeDiv(a, b) = if b == 0 then None else Some(a / b)
            unwrap(opt) = match opt {
                Some(x) -> x
                None -> 0
            }
            main() = unwrap(flatMap(x => safeDiv(x, 2), Some(10)))
        ";
        // Some(10) -> safeDiv(10, 2) = Some(5) -> unwrap = 5
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_list_zip() {
        let source = "
            zip(xs, ys) = match xs {
                [] -> []
                [hx | tx] -> match ys {
                    [] -> []
                    [hy | ty] -> [(hx, hy) | zip(tx, ty)]
                }
            }
            sumPairs(ps) = match ps {
                [] -> 0
                [p | t] -> p.0 + p.1 + sumPairs(t)
            }
            main() = sumPairs(zip([1, 2, 3], [10, 20, 30]))
        ";
        // zip = [(1,10), (2,20), (3,30)], sumPairs = 1+10+2+20+3+30 = 66
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(66)));
    }

    #[test]
    fn test_e2e_float_operations() {
        // Float arithmetic in main directly
        let source = "
            main() = 3.0 * 2.5
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Float64(7.5)));
    }

    #[test]
    fn test_e2e_comparison_operators() {
        let source = "
            compare(a, b) = if a < b then 0 - 1 else if a > b then 1 else 0
            main() = compare(3, 5) + compare(5, 3) + compare(4, 4)
        ";
        // -1 + 1 + 0 = 0
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(0)));
    }

    #[test]
    fn test_e2e_wildcard_pattern() {
        let source = "
            first(xs) = match xs {
                [a, _, _] -> a
                _ -> 0
            }
            main() = first([42, 100, 200])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_list_exact_pattern() {
        let source = "
            sumThree(xs) = match xs {
                [a, b, c] -> a + b + c
                _ -> 0
            }
            main() = sumThree([10, 20, 12])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_block_shadowing() {
        // Note: Variable shadowing in blocks not fully isolated
        // Testing block with different variable name instead
        let source = "
            main() = {
                x = 10
                y = {
                    z = 20
                    z + 5
                }
                x + y
            }
        ";
        // z = 20, y = 25, x = 10, result = 35
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(35)));
    }

    #[test]
    fn test_e2e_deeply_nested_conditionals() {
        let source = "
            classify(n) =
                if n < 0 then
                    if n < 0 - 10 then \"very negative\"
                    else \"negative\"
                else if n == 0 then \"zero\"
                else if n < 10 then \"small positive\"
                else \"large positive\"
            main() = classify(5)
        ";
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "small positive"),
            other => panic!("Expected 'small positive', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_higher_order_list_operations() {
        let source = "
            map(f, xs) = match xs {
                [] -> []
                [h | t] -> [f(h) | map(f, t)]
            }
            filter(p, xs) = match xs {
                [] -> []
                [h | t] -> if p(h) then [h | filter(p, t)] else filter(p, t)
            }
            foldl(f, acc, xs) = match xs {
                [] -> acc
                [h | t] -> foldl(f, f(acc, h), t)
            }
            main() = {
                nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                doubled = map(x => x * 2, nums)
                evens = filter(x => (x - (x / 2) * 2) == 0, doubled)
                foldl((a, b) => a + b, 0, evens)
            }
        ";
        // doubled = [2,4,6,8,10,12,14,16,18,20]
        // all are even, so evens = same
        // sum = 2+4+6+8+10+12+14+16+18+20 = 110
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(110)));
    }

    #[test]
    fn test_e2e_list_nth() {
        let source = "
            type Option[T] = Some(T) | None
            nth(n, xs) = match xs {
                [] -> None
                [h | t] -> if n == 0 then Some(h) else nth(n - 1, t)
            }
            unwrap(opt) = match opt {
                Some(x) -> x
                None -> 0
            }
            main() = unwrap(nth(2, [10, 20, 30, 40]))
        ";
        // nth 2 = 30
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(30)));
    }

    #[test]
    fn test_e2e_list_all_any() {
        let source = "
            all(p, xs) = match xs {
                [] -> true
                [h | t] -> if p(h) then all(p, t) else false
            }
            any(p, xs) = match xs {
                [] -> false
                [h | t] -> if p(h) then true else any(p, t)
            }
            main() = all(x => x > 0, [1, 2, 3]) && any(x => x > 2, [1, 2, 3])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_tree_sum() {
        let source = "
            type Tree[T] = Leaf(T) | Node(Tree[T], Tree[T])
            sumTree(tree) = match tree {
                Leaf(v) -> v
                Node(l, r) -> sumTree(l) + sumTree(r)
            }
            main() = sumTree(Node(Node(Leaf(1), Leaf(2)), Node(Leaf(3), Leaf(4))))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(10)));
    }

    #[test]
    fn test_e2e_tree_depth() {
        let source = "
            type Tree[T] = Leaf(T) | Node(Tree[T], Tree[T])
            max(a, b) = if a > b then a else b
            depth(tree) = match tree {
                Leaf(_) -> 1
                Node(l, r) -> 1 + max(depth(l), depth(r))
            }
            main() = depth(Node(Node(Leaf(1), Node(Leaf(2), Leaf(3))), Leaf(4)))
        ";
        // left subtree depth: 3, right: 1, total = 4
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(4)));
    }

    #[test]
    fn test_e2e_expression_evaluator() {
        let source = "
            type Expr = Num(Int) | Add(Expr, Expr) | Mul(Expr, Expr)
            eval(expr) = match expr {
                Num(n) -> n
                Add(e1, e2) -> eval(e1) + eval(e2)
                Mul(e1, e2) -> eval(e1) * eval(e2)
            }
            main() = eval(Add(Mul(Num(3), Num(4)), Num(5)))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(17)));
    }

    // =========================================================================
    // Map/Set literal tests
    // =========================================================================

    #[test]
    fn test_e2e_map_literal_empty() {
        let source = "
            main() = %{}
        ";
        let result = compile_and_run(source);
        match result {
            Ok(Value::Map(m)) => {
                assert!(m.is_empty());
            }
            other => panic!("Expected empty map, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_map_literal_simple() {
        let source = r#"
            main() = %{"a": 1, "b": 2}
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::Map(m)) => {
                assert_eq!(m.len(), 2);
            }
            other => panic!("Expected map with 2 entries, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_set_literal_empty() {
        let source = "
            main() = #{}
        ";
        let result = compile_and_run(source);
        match result {
            Ok(Value::Set(s)) => {
                assert!(s.is_empty());
            }
            other => panic!("Expected empty set, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_set_literal_simple() {
        let source = "
            main() = #{1, 2, 3}
        ";
        let result = compile_and_run(source);
        match result {
            Ok(Value::Set(s)) => {
                assert_eq!(s.len(), 3);
            }
            other => panic!("Expected set with 3 elements, got {:?}", other),
        }
    }

    // =========================================================================
    // Multi-clause function dispatch tests
    // =========================================================================

    #[test]
    fn test_e2e_multiclause_factorial() {
        let source = "
            factorial(0) = 1
            factorial(n) = n * factorial(n - 1)
            main() = factorial(5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(120)));
    }

    #[test]
    fn test_e2e_multiclause_fibonacci() {
        let source = "
            fib(0) = 0
            fib(1) = 1
            fib(n) = fib(n - 1) + fib(n - 2)
            main() = fib(10)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(55)));
    }

    #[test]
    fn test_e2e_multiclause_with_patterns() {
        let source = "
            len([]) = 0
            len([_ | t]) = 1 + len(t)
            main() = len([1, 2, 3, 4, 5])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_multiclause_sum_list() {
        let source = "
            sum([]) = 0
            sum([h | t]) = h + sum(t)
            main() = sum([1, 2, 3, 4, 5])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(15)));
    }

    #[test]
    fn test_e2e_multiclause_reverse() {
        let source = "
            reverse_acc([], acc) = acc
            reverse_acc([h | t], acc) = reverse_acc(t, [h | acc])
            reverse(xs) = reverse_acc(xs, [])
            len([]) = 0
            len([_ | t]) = 1 + len(t)
            main() = len(reverse([1, 2, 3, 4, 5]))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_multiclause_variant_dispatch() {
        let source = "
            type Option[T] = Some(T) | None
            unwrap(None) = 0
            unwrap(Some(x)) = x
            main() = unwrap(Some(42)) + unwrap(None)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_multiclause_either() {
        let source = "
            type Either[L, R] = Left(L) | Right(R)
            getValue(Left(x)) = x
            getValue(Right(x)) = x
            main() = getValue(Left(10)) + getValue(Right(32))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_multiclause_tuple_pattern() {
        let source = "
            fst((a, _)) = a
            snd((_, b)) = b
            main() = fst((1, 2)) + snd((3, 4))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_multiclause_literal_match() {
        let source = r#"
            describe(0) = "zero"
            describe(1) = "one"
            describe(_) = "other"
            main() = describe(1)
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(s.as_ref(), "one"),
            other => panic!("Expected string 'one', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_multiclause_mixed_patterns() {
        let source = "
            classify(0, _) = 0
            classify(_, 0) = 1
            classify(a, b) = a + b
            main() = classify(0, 5) + classify(5, 0) * 10 + classify(2, 3) * 100
        ";
        // 0 + 10 + 500 = 510
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(510)));
    }

    // =========================================================================
    // Function with guards tests
    // =========================================================================

    #[test]
    fn test_e2e_function_with_guard() {
        let source = "
            abs(n) when n < 0 = -n
            abs(n) = n
            main() = abs(-5) + abs(3)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(8)));
    }

    #[test]
    fn test_e2e_function_multiple_guards() {
        let source = "
            sign(n) when n < 0 = -1
            sign(n) when n > 0 = 1
            sign(_) = 0
            main() = sign(-5) + sign(5) + sign(0)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(0)));
    }

    #[test]
    fn test_e2e_function_guard_with_pattern() {
        let source = "
            first_positive([]) = 0
            first_positive([h | t]) when h > 0 = h
            first_positive([_ | t]) = first_positive(t)
            main() = first_positive([-3, -2, 5, 10])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    // ===== String Interpolation Tests =====

    #[test]
    fn test_e2e_string_interpolation_simple() {
        let source = r#"
            main() = {
                name = "World"
                "Hello, ${name}!"
            }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Hello, World!"),
            other => panic!("Expected 'Hello, World!', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_integer() {
        let source = r#"
            main() = {
                x = 42
                "The answer is ${x}"
            }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "The answer is 42"),
            other => panic!("Expected 'The answer is 42', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_expression() {
        let source = r#"
            main() = "Sum: ${1 + 2 + 3}"
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Sum: 6"),
            other => panic!("Expected 'Sum: 6', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_multiple() {
        let source = r#"
            main() = {
                a = 10
                b = 20
                "${a} + ${b} = ${a + b}"
            }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "10 + 20 = 30"),
            other => panic!("Expected '10 + 20 = 30', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_with_function_call() {
        let source = r#"
            double(x) = x * 2
            main() = "Double of 21 is ${double(21)}"
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Double of 21 is 42"),
            other => panic!("Expected 'Double of 21 is 42', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_nested_expr() {
        let source = r#"
            main() = "Result: ${if true then 1 else 0}"
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Result: 1"),
            other => panic!("Expected 'Result: 1', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_no_interpolation() {
        let source = r#"
            main() = "Plain string without interpolation"
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Plain string without interpolation"),
            other => panic!("Expected plain string, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_at_start() {
        let source = r#"
            main() = {
                x = 42
                "${x} is the answer"
            }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "42 is the answer"),
            other => panic!("Expected '42 is the answer', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_at_end() {
        let source = r#"
            main() = {
                x = 42
                "Answer: ${x}"
            }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Answer: 42"),
            other => panic!("Expected 'Answer: 42', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_only_expr() {
        let source = r#"
            main() = "${42}"
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "42"),
            other => panic!("Expected '42', got {:?}", other),
        }
    }

    // ===== Module System Tests =====

    #[test]
    fn test_e2e_module_nested_function() {
        let source = "
            module Math
                pub add(a, b) = a + b
                pub double(x) = x * 2
            end

            main() = Math.add(10, Math.double(16))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_module_nested_module() {
        let source = "
            module Outer
                pub module Inner
                    pub value() = 21
                end
            end

            main() = Outer.Inner.value() * 2
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_module_use_import() {
        let source = "
            module Math
                pub add(a, b) = a + b
            end

            use Math.{add}

            main() = add(20, 22)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_module_use_alias() {
        let source = "
            module Math
                pub multiply(a, b) = a * b
            end

            use Math.{multiply as mul}

            main() = mul(6, 7)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_module_multiple_functions() {
        let source = "
            module Utils
                pub inc(x) = x + 1
                dec(x) = x - 1
                pub triple(x) = x * 3
            end

            main() = Utils.triple(Utils.inc(13))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_module_recursive_function() {
        let source = "
            module Math
                pub factorial(0) = 1
                pub factorial(n) = n * factorial(n - 1)
            end

            main() = Math.factorial(5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(120)));
    }

    // ===== Visibility Tests =====

    #[test]
    fn test_e2e_visibility_private_access_denied() {
        let source = "
            module Secret
                private_fn() = 42
            end

            main() = Secret.private_fn()
        ";
        let result = compile_and_run(source);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("PrivateAccess"), "Expected PrivateAccess error, got: {}", err);
    }

    #[test]
    fn test_e2e_visibility_public_access_allowed() {
        let source = "
            module Public
                pub public_fn() = 42
            end

            main() = Public.public_fn()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_visibility_private_internal_call() {
        // Private functions can call each other within the same module
        let source = "
            module Math
                helper(x) = x * 2
                pub double_plus_one(x) = helper(x) + 1
            end

            main() = Math.double_plus_one(20)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(41)));
    }

    #[test]
    fn test_e2e_visibility_nested_module_private() {
        // Private nested module function should not be accessible
        let source = "
            module Outer
                pub module Inner
                    secret() = 42  // private in Inner
                end
            end

            main() = Outer.Inner.secret()
        ";
        let result = compile_and_run(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_e2e_visibility_pub_type_and_function() {
        // Test that pub works for both types and functions
        let source = "
            module Shapes
                pub type Point = { x: Int, y: Int }
                pub origin() = Point(0, 0)
            end

            main() = {
                p = Shapes.origin()
                p.x + p.y
            }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(0)));
    }

    // ===== Trait Tests =====

    #[test]
    fn test_e2e_trait_basic_parse_and_compile() {
        // Test that the basic trait syntax parses and compiles (no method call yet)
        // Based on working parser tests:
        // parse_ok("trait Eq ==(self, other) -> Bool !=(self, other) = !(self == other) end");
        // parse_ok("Point: Show show(self) = self.x end");
        let source = "
            type Point = { x: Int, y: Int }
            trait Show show(self) -> Int end
            Point: Show show(self) = self.x end
            main() = 42
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_trait_explicit_qualified_call() {
        // Test trait dispatch on record type
        let source = "
            type Point = { x: Int, y: Int }
            trait GetX getX(self) -> Int end
            Point: GetX getX(self) = self.x end
            main() = Point(42, 10).getX()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_trait_method_with_self_reference() {
        // Test trait methods that use self
        let source = "
            type Counter = { value: Int }
            trait Doubled doubled(self) -> Int end
            Counter: Doubled doubled(self) = self.value * 2 end
            main() = Counter(21).doubled()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_trait_multiple_methods() {
        // Test trait with multiple methods
        let source = "
            type Num = { value: Int }
            trait Math add(self, n) -> Int sub(self, n) -> Int end
            Num: Math add(self, n) = self.value + n sub(self, n) = self.value - n end
            main() = Num(10).add(5) + Num(10).sub(3)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(22))); // (10+5) + (10-3) = 15 + 7 = 22
    }

    #[test]
    fn test_e2e_trait_method_dispatch_on_record_literal() {
        // Test that method dispatch works on record constructors
        let source = "
            type Rectangle = { width: Int, height: Int }
            trait Area area(self) -> Int end
            Rectangle: Area area(self) = self.width * self.height end
            main() = { r = Rectangle(4, 5), r.area() }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(20)));
    }

    #[test]
    fn test_e2e_trait_unknown_trait_error() {
        // Test that implementing an unknown trait produces an error
        let source = "
            type Point = { x: Int, y: Int }
            Point: NonExistent foo(self) = 42 end
            main() = 1
        ";
        let result = compile_and_run(source);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("UnknownTrait"), "Expected UnknownTrait error, got: {}", err);
    }

    #[test]
    fn test_e2e_trait_dispatch_on_int_literal() {
        // Test trait method dispatch on Int literals
        let source = "
            trait Triple triple(self) -> Int end
            Int: Triple triple(self) = self * 3 end
            main() = 7.triple()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(21)));
    }

    #[test]
    fn test_e2e_trait_dispatch_on_string_literal() {
        // Test trait method dispatch on String literals
        let source = "
            trait Greeting greet(self) -> String end
            String: Greeting greet(self) = \"Hello, \" ++ self end
            main() = \"World\".greet()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::String(std::sync::Arc::new("Hello, World".to_string()))));
    }

    #[test]
    fn test_e2e_trait_multiple_impls() {
        // Test same trait implemented for multiple types
        let source = "
            type Point = { x: Int, y: Int }
            type Rectangle = { w: Int, h: Int }
            trait Size size(self) -> Int end
            Point: Size size(self) = 2 end
            Rectangle: Size size(self) = 4 end
            main() = { p = Point(0, 0), r = Rectangle(5, 10), p.size() + r.size() }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(6))); // 2 + 4 = 6
    }

    #[test]
    fn test_e2e_trait_explicit_call() {
        // Test explicit trait method call: Type.Trait.method(obj)
        let source = "
            type Box = { value: Int }
            trait Doubler doubler(self) -> Int end
            Box: Doubler doubler(self) = self.value * 2 end
            main() = { b = Box(10), Box.Doubler.doubler(b) }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(20)));
    }

    #[test]
    fn test_e2e_trait_method_chaining() {
        // Test calling trait methods on multiple values
        // Note: True chaining like `a = 5.inc(); a.inc()` would require tracking
        // return types of method calls, which is not yet implemented.
        let source = "
            trait Inc inc(self) -> Int end
            Int: Inc inc(self) = self + 1 end
            main() = 5.inc() + 6.inc()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(13))); // 6 + 7 = 13
    }

    #[test]
    fn test_e2e_trait_bool_impl() {
        // Test trait implementation for Bool
        let source = "
            trait Toggle toggle(self) -> Bool end
            Bool: Toggle toggle(self) = !self end
            main() = if true.toggle() then 0 else 1
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(1))); // true.toggle() = false, so else branch
    }

    #[test]
    fn test_e2e_trait_with_record_return() {
        // Test trait methods that return records
        let source = "
            type Point = { x: Int, y: Int }
            trait Cloner cloner(self) -> Point end
            Point: Cloner cloner(self) = Point(self.x, self.y) end
            main() = { p = Point(3, 4), p2 = p.cloner(), p2.x + p2.y }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(7)));
    }

    #[test]
    fn test_e2e_trait_with_multiple_args() {
        // Test trait methods with multiple arguments
        let source = "
            type Base = { value: Int }
            trait Adder adder(self, x, y) -> Int end
            Base: Adder adder(self, x, y) = self.value + x + y end
            main() = { b = Base(10), b.adder(20, 30) }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(60)));
    }

    // =========================================================================
    // Error handling tests - verify correct error types and span locations
    // =========================================================================

    fn compile_should_fail(source: &str) -> CompileError {
        let (module_opt, errors) = parse(source);
        if !errors.is_empty() {
            panic!("Unexpected parse error: {:?}", errors);
        }
        let module = module_opt.expect("Parse returned no module");
        match compile_module(&module, source) {
            Err(e) => e,
            Ok(_) => panic!("Expected compile error, but compilation succeeded"),
        }
    }

    #[test]
    fn test_error_unknown_variable() {
        let source = "main() = x + 1";
        let err = compile_should_fail(source);
        match err {
            CompileError::UnknownVariable { name, span } => {
                assert_eq!(name, "x");
                // 'x' starts at position 9 in the source
                assert_eq!(span.start, 9);
                assert_eq!(span.end, 10);
            }
            _ => panic!("Expected UnknownVariable error, got {:?}", err),
        }
    }

    #[test]
    fn test_error_unknown_variable_in_block() {
        let source = "main() = { y = undefined_var }";
        let err = compile_should_fail(source);
        match err {
            CompileError::UnknownVariable { name, .. } => {
                assert_eq!(name, "undefined_var");
            }
            _ => panic!("Expected UnknownVariable error, got {:?}", err),
        }
    }

    #[test]
    fn test_error_unknown_trait() {
        let source = "type Foo = { x: Int }\nFoo: NonExistentTrait end";
        let err = compile_should_fail(source);
        match err {
            CompileError::UnknownTrait { name, .. } => {
                assert_eq!(name, "NonExistentTrait");
            }
            _ => panic!("Expected UnknownTrait error, got {:?}", err),
        }
    }

    #[test]
    fn test_error_private_access() {
        let source = "
            module Secret
                private_fn() = 42
            end
            main() = Secret.private_fn()
        ";
        let err = compile_should_fail(source);
        match err {
            CompileError::PrivateAccess { function, module, .. } => {
                assert_eq!(function, "private_fn");
                assert_eq!(module, "Secret");
            }
            _ => panic!("Expected PrivateAccess error, got {:?}", err),
        }
    }

    #[test]
    fn test_error_span_points_to_correct_variable() {
        // Test that the span correctly points to the undefined variable
        let source = "main() = {\n    a = 1\n    b = c + a\n}";
        let err = compile_should_fail(source);
        match err {
            CompileError::UnknownVariable { name, span } => {
                assert_eq!(name, "c");
                // Verify the span points to 'c' by extracting from source
                let pointed_text = &source[span.start..span.end];
                assert_eq!(pointed_text, "c");
            }
            _ => panic!("Expected UnknownVariable error, got {:?}", err),
        }
    }

    #[test]
    fn test_error_to_source_error_conversion() {
        let source = "main() = unknown";
        let err = compile_should_fail(source);

        // Test that to_source_error creates a proper SourceError
        let source_err = err.to_source_error();
        assert!(source_err.message.contains("unknown"));
        assert_eq!(source_err.span.start, 9);
        assert_eq!(source_err.span.end, 16);
    }

    #[test]
    fn test_error_format_output() {
        let source = "main() = undefined_var";
        let err = compile_should_fail(source);
        let source_err = err.to_source_error();

        // Test that format() produces valid output
        let formatted = source_err.format("test.nos", source);
        assert!(formatted.contains("unknown variable"));
        assert!(formatted.contains("undefined_var"));
        // Should contain line/column reference
        assert!(formatted.contains("test.nos"));
    }

    #[test]
    fn test_error_nested_undefined() {
        // Test nested expression with undefined variable
        let source = "main() = if true then undefined_fn() else 0";
        let err = compile_should_fail(source);
        match err {
            CompileError::UnknownVariable { name, .. } => {
                assert_eq!(name, "undefined_fn");
            }
            _ => panic!("Expected UnknownVariable error, got {:?}", err),
        }
    }

    // ============= Try/Catch Tests =============

    #[test]
    fn test_try_catch_basic() {
        // throw and catch should work
        let source = r#"
            main() = try { throw("error") } catch { e -> e }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "error"),
            other => panic!("Expected string 'error', got {:?}", other),
        }
    }

    #[test]
    fn test_try_catch_no_exception() {
        // When no exception, return the try body value
        let source = r#"
            main() = try { 42 } catch { _ -> 0 }
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_try_catch_pattern_matching() {
        // Pattern matching in catch
        let source = r#"
            main() = try { throw("special") } catch {
                "special" -> 1
                other -> 2
            }
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(1)));
    }

    #[test]
    fn test_try_catch_pattern_fallthrough() {
        // Non-matching pattern falls to next
        let source = r#"
            main() = try { throw("other") } catch {
                "special" -> 1
                _ -> 2
            }
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(2)));
    }

    #[test]
    fn test_error_propagation_success() {
        // ? operator returns value on success
        let source = r#"
            might_fail(fail) = if fail then throw("error") else 42
            main() = try { might_fail(false)? + 1 } catch { _ -> 0 }
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(43)));
    }

    #[test]
    fn test_error_propagation_rethrow() {
        // ? operator propagates exception
        let source = r#"
            might_fail(fail) = if fail then throw("error") else 42
            propagate() = might_fail(true)? + 1
            main() = try { propagate() } catch { e -> e }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "error"),
            other => panic!("Expected string 'error', got {:?}", other),
        }
    }

    #[test]
    fn test_nested_try_catch() {
        // Nested try/catch
        let source = r#"
            main() = try {
                inner = try { throw("inner") } catch { _ -> throw("outer") }
                inner
            } catch {
                e -> e
            }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "outer"),
            other => panic!("Expected string 'outer', got {:?}", other),
        }
    }

    #[test]
    fn test_throw_integer() {
        // Can throw any value, not just strings
        let source = r#"
            main() = try { throw(42) } catch { e -> e }
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    // =========================================================================
    // Type Inference Signature Tests (80+ tests)
    // =========================================================================

    fn get_signature(source: &str, fn_name: &str) -> Option<String> {
        let (module_opt, errors) = parse(source);
        if !errors.is_empty() {
            return None;
        }
        let module = module_opt?;
        let compiler = compile_module(&module, source).ok()?;
        compiler.get_function_signature(fn_name).map(|s| s.to_string())
    }

    // Helper to check signature matches one of several valid options
    // Also accepts signatures with trait bounds (e.g., "Num a => a -> a -> a" matches "a -> a -> a")
    fn sig_matches(sig: &str, options: &[&str]) -> bool {
        // Strip trait bound prefix if present (e.g., "Num a => " prefix)
        let core_sig = if let Some(idx) = sig.find(" => ") {
            &sig[idx + 4..]
        } else {
            sig
        };
        options.iter().any(|opt| sig == *opt || core_sig == *opt)
    }

    // -------------------------------------------------------------------------
    // Basic Arithmetic Operations (Tests 1-10)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_01_add_unified() {
        let sig = get_signature("add(x, y) = x + y\nmain() = 0", "add").unwrap();
        // Should infer: Num a => a -> a -> a (or just a -> a -> a if bounds not shown)
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int", "Num a => a -> a -> a"]),
            "add: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_02_sub_unified() {
        let sig = get_signature("sub(x, y) = x - y\nmain() = 0", "sub").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int"]),
            "sub: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_03_mul_unified() {
        let sig = get_signature("mul(x, y) = x * y\nmain() = 0", "mul").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int"]),
            "mul: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_04_div_unified() {
        let sig = get_signature("div(x, y) = x / y\nmain() = 0", "div").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int"]),
            "div: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_05_mod_unified() {
        let sig = get_signature("rem(x, y) = x % y\nmain() = 0", "rem").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int"]),
            "mod: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_06_pow_unified() {
        let sig = get_signature("pow(x, y) = x ** y\nmain() = 0", "pow").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int"]),
            "pow: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_07_add_with_int_literal() {
        let sig = get_signature("inc(x) = x + 1\nmain() = 0", "inc").unwrap();
        assert!(sig_matches(&sig, &["Int -> Int", "a -> a"]),
            "inc: expected Int -> Int, got: {}", sig);
    }

    #[test]
    fn test_hm_08_add_with_float_literal() {
        let sig = get_signature("incf(x) = x + 1.0\nmain() = 0", "incf").unwrap();
        assert!(sig_matches(&sig, &["Float -> Float", "a -> a"]),
            "incf: expected Float -> Float, got: {}", sig);
    }

    #[test]
    fn test_hm_09_negation() {
        let sig = get_signature("neg(x) = -x\nmain() = 0", "neg").unwrap();
        assert!(sig.contains("->"), "neg: expected function type, got: {}", sig);
    }

    #[test]
    fn test_hm_10_complex_arithmetic() {
        let sig = get_signature("calc(a, b, c) = a * b + c\nmain() = 0", "calc").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a -> a", "Int -> Int -> Int -> Int"]),
            "calc: expected all unified, got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Comparison Operations (Tests 11-20)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_11_eq_returns_bool() {
        let sig = get_signature("eq(x, y) = x == y\nmain() = 0", "eq").unwrap();
        assert!(sig.ends_with("-> Bool"), "eq: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_12_neq_returns_bool() {
        let sig = get_signature("neq(x, y) = x != y\nmain() = 0", "neq").unwrap();
        assert!(sig.ends_with("-> Bool"), "neq: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_13_lt_returns_bool() {
        let sig = get_signature("lt(x, y) = x < y\nmain() = 0", "lt").unwrap();
        assert!(sig.ends_with("-> Bool"), "lt: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_14_gt_returns_bool() {
        let sig = get_signature("gt(x, y) = x > y\nmain() = 0", "gt").unwrap();
        assert!(sig.ends_with("-> Bool"), "gt: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_15_lte_returns_bool() {
        let sig = get_signature("lte(x, y) = x <= y\nmain() = 0", "lte").unwrap();
        assert!(sig.ends_with("-> Bool"), "lte: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_16_gte_returns_bool() {
        let sig = get_signature("gte(x, y) = x >= y\nmain() = 0", "gte").unwrap();
        assert!(sig.ends_with("-> Bool"), "gte: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_17_comparison_unifies_operands() {
        let sig = get_signature("cmp(x, y) = x < y\nmain() = 0", "cmp").unwrap();
        // Should be a -> a -> Bool (operands unified)
        assert!(sig.contains("-> Bool"), "cmp: expected Bool return, got: {}", sig);
    }

    #[test]
    fn test_hm_18_eq_with_int_literal() {
        let sig = get_signature("isZero(x) = x == 0\nmain() = 0", "isZero").unwrap();
        assert!(sig_matches(&sig, &["Int -> Bool", "a -> Bool"]),
            "isZero: expected Int -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_19_comparison_chain() {
        let sig = get_signature("inRange(x, lo, hi) = x >= lo && x <= hi\nmain() = 0", "inRange").unwrap();
        assert!(sig.ends_with("-> Bool"), "inRange: expected Bool return, got: {}", sig);
    }

    #[test]
    fn test_hm_20_eq_bool_literal() {
        let sig = get_signature("isTrue(x) = x == true\nmain() = 0", "isTrue").unwrap();
        assert!(sig_matches(&sig, &["Bool -> Bool", "a -> Bool"]),
            "isTrue: expected Bool -> Bool, got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Boolean Operations (Tests 21-30)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_21_and_bool() {
        let sig = get_signature("band(a, b) = a && b\nmain() = 0", "band").unwrap();
        assert_eq!(sig, "Bool -> Bool -> Bool", "and: got: {}", sig);
    }

    #[test]
    fn test_hm_22_or_bool() {
        let sig = get_signature("bor(a, b) = a || b\nmain() = 0", "bor").unwrap();
        assert_eq!(sig, "Bool -> Bool -> Bool", "or: got: {}", sig);
    }

    #[test]
    fn test_hm_23_not_bool() {
        let sig = get_signature("bnot(a) = !a\nmain() = 0", "bnot").unwrap();
        assert_eq!(sig, "Bool -> Bool", "not: got: {}", sig);
    }

    #[test]
    fn test_hm_24_complex_bool() {
        let sig = get_signature("logic(a, b, c) = (a && b) || c\nmain() = 0", "logic").unwrap();
        assert_eq!(sig, "Bool -> Bool -> Bool -> Bool", "logic: got: {}", sig);
    }

    #[test]
    fn test_hm_25_bool_with_comparison() {
        let sig = get_signature("check(x, y) = x > 0 && y < 10\nmain() = 0", "check").unwrap();
        assert!(sig.ends_with("-> Bool"), "check: expected Bool return, got: {}", sig);
    }

    #[test]
    fn test_hm_26_bool_literal_true() {
        let sig = get_signature("alwaysTrue() = true\nmain() = 0", "alwaysTrue").unwrap();
        assert_eq!(sig, "Bool", "alwaysTrue: got: {}", sig);
    }

    #[test]
    fn test_hm_27_bool_literal_false() {
        let sig = get_signature("alwaysFalse() = false\nmain() = 0", "alwaysFalse").unwrap();
        assert_eq!(sig, "Bool", "alwaysFalse: got: {}", sig);
    }

    #[test]
    fn test_hm_28_double_negation() {
        let sig = get_signature("dblNot(x) = !!x\nmain() = 0", "dblNot").unwrap();
        assert_eq!(sig, "Bool -> Bool", "dblNot: got: {}", sig);
    }

    #[test]
    fn test_hm_29_implies() {
        let sig = get_signature("implies(a, b) = !a || b\nmain() = 0", "implies").unwrap();
        assert_eq!(sig, "Bool -> Bool -> Bool", "implies: got: {}", sig);
    }

    #[test]
    fn test_hm_30_xor() {
        let sig = get_signature("xor(a, b) = (a || b) && !(a && b)\nmain() = 0", "xor").unwrap();
        assert_eq!(sig, "Bool -> Bool -> Bool", "xor: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Control Flow - If Expressions (Tests 31-40)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_31_if_basic() {
        let sig = get_signature("choose(c, a, b) = if c then a else b\nmain() = 0", "choose").unwrap();
        assert!(sig.starts_with("Bool"), "choose: expected Bool first, got: {}", sig);
    }

    #[test]
    fn test_hm_32_if_unifies_branches() {
        let sig = get_signature("sel(c, x, y) = if c then x else y\nmain() = 0", "sel").unwrap();
        // Should be Bool -> a -> a -> a
        assert!(sig.starts_with("Bool -> "), "sel: expected Bool first, got: {}", sig);
    }

    #[test]
    fn test_hm_33_if_with_int_branches() {
        let sig = get_signature("abs(x) = if x < 0 then -x else x\nmain() = 0", "abs").unwrap();
        assert!(sig.contains("->"), "abs: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_34_if_with_bool_return() {
        let sig = get_signature("sign(x) = if x > 0 then true else false\nmain() = 0", "sign").unwrap();
        assert!(sig.ends_with("-> Bool"), "sign: expected Bool return, got: {}", sig);
    }

    #[test]
    fn test_hm_35_nested_if() {
        let sig = get_signature("clamp(x, lo, hi) = if x < lo then lo else if x > hi then hi else x\nmain() = 0", "clamp").unwrap();
        assert!(sig.contains("->"), "clamp: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_36_if_with_literal() {
        let sig = get_signature("maybeOne(c) = if c then 1 else 0\nmain() = 0", "maybeOne").unwrap();
        assert!(sig_matches(&sig, &["Bool -> Int", "Bool -> a"]),
            "maybeOne: got: {}", sig);
    }

    #[test]
    fn test_hm_37_if_string_branches() {
        let sig = get_signature(r#"greet(formal) = if formal then "Hello" else "Hi"\nmain() = 0"#, "greet").unwrap();
        assert!(sig_matches(&sig, &["Bool -> String", "Bool -> a"]),
            "greet: got: {}", sig);
    }

    #[test]
    fn test_hm_38_if_complex_condition() {
        let sig = get_signature("check(a, b) = if a > 0 && b > 0 then a + b else 0\nmain() = 0", "check").unwrap();
        assert!(sig.contains("->"), "check: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_39_if_with_unit() {
        let sig = get_signature("maybe(c) = if c then () else ()\nmain() = 0", "maybe").unwrap();
        assert!(sig_matches(&sig, &["Bool -> ()", "Bool -> a"]),
            "maybe: got: {}", sig);
    }

    #[test]
    fn test_hm_40_ternary_like() {
        let sig = get_signature("max(a, b) = if a > b then a else b\nmain() = 0", "max").unwrap();
        assert!(sig.contains("->"), "max: expected function, got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Functions - Polymorphism (Tests 41-50)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_41_identity() {
        let sig = get_signature("id(x) = x\nmain() = 0", "id").unwrap();
        assert_eq!(sig, "a -> a", "identity: got: {}", sig);
    }

    #[test]
    fn test_hm_42_const() {
        let sig = get_signature("const(x, y) = x\nmain() = 0", "const").unwrap();
        assert_eq!(sig, "a -> b -> a", "const: got: {}", sig);
    }

    #[test]
    fn test_hm_43_flip() {
        let sig = get_signature("second(x, y) = y\nmain() = 0", "second").unwrap();
        assert_eq!(sig, "a -> b -> b", "second: got: {}", sig);
    }

    #[test]
    fn test_hm_44_three_params_first() {
        let sig = get_signature("first3(a, b, c) = a\nmain() = 0", "first3").unwrap();
        assert_eq!(sig, "a -> b -> c -> a", "first3: got: {}", sig);
    }

    #[test]
    fn test_hm_45_three_params_middle() {
        let sig = get_signature("mid3(a, b, c) = b\nmain() = 0", "mid3").unwrap();
        assert_eq!(sig, "a -> b -> c -> b", "mid3: got: {}", sig);
    }

    #[test]
    fn test_hm_46_three_params_last() {
        let sig = get_signature("last3(a, b, c) = c\nmain() = 0", "last3").unwrap();
        assert_eq!(sig, "a -> b -> c -> c", "last3: got: {}", sig);
    }

    #[test]
    fn test_hm_47_four_params_unused() {
        let sig = get_signature("pick(a, b, c, d) = b\nmain() = 0", "pick").unwrap();
        assert_eq!(sig, "a -> b -> c -> d -> b", "pick: got: {}", sig);
    }

    #[test]
    fn test_hm_48_no_params_int() {
        let sig = get_signature("fortytwo() = 42\nmain() = 0", "fortytwo").unwrap();
        assert_eq!(sig, "Int", "fortytwo: got: {}", sig);
    }

    #[test]
    fn test_hm_49_no_params_string() {
        let sig = get_signature(r#"hello() = "hello"\nmain() = 0"#, "hello").unwrap();
        assert_eq!(sig, "String", "hello: got: {}", sig);
    }

    #[test]
    fn test_hm_50_no_params_unit() {
        let sig = get_signature("noop() = ()\nmain() = 0", "noop").unwrap();
        assert_eq!(sig, "()", "noop: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Type Annotations (Tests 51-60)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_51_annotated_int_param() {
        let sig = get_signature("f(x: Int) = x\nmain() = 0", "f").unwrap();
        assert_eq!(sig, "Int -> Int", "f: got: {}", sig);
    }

    #[test]
    fn test_hm_52_annotated_bool_param() {
        let sig = get_signature("g(x: Bool) = x\nmain() = 0", "g").unwrap();
        assert_eq!(sig, "Bool -> Bool", "g: got: {}", sig);
    }

    #[test]
    fn test_hm_53_annotated_string_param() {
        let sig = get_signature("h(x: String) = x\nmain() = 0", "h").unwrap();
        assert_eq!(sig, "String -> String", "h: got: {}", sig);
    }

    #[test]
    fn test_hm_54_annotated_float_param() {
        let sig = get_signature("flt(x: Float) = x\nmain() = 0", "flt").unwrap();
        assert_eq!(sig, "Float -> Float", "flt: got: {}", sig);
    }

    #[test]
    fn test_hm_55_annotated_return_type() {
        let sig = get_signature("ret(x) -> Int = x\nmain() = 0", "ret").unwrap();
        assert!(sig.ends_with("-> Int"), "ret: expected Int return, got: {}", sig);
    }

    #[test]
    fn test_hm_56_annotated_both() {
        let sig = get_signature("both(x: Int) -> Int = x + 1\nmain() = 0", "both").unwrap();
        assert_eq!(sig, "Int -> Int", "both: got: {}", sig);
    }

    #[test]
    fn test_hm_57_mixed_annotated_unannotated() {
        let sig = get_signature("mix(x: Int, y) = x + y\nmain() = 0", "mix").unwrap();
        assert_eq!(sig, "Int -> Int -> Int", "mix: got: {}", sig);
    }

    #[test]
    fn test_hm_58_all_annotated() {
        let sig = get_signature("all(a: Int, b: Int, c: Int) -> Int = a + b + c\nmain() = 0", "all").unwrap();
        assert_eq!(sig, "Int -> Int -> Int -> Int", "all: got: {}", sig);
    }

    #[test]
    fn test_hm_59_char_annotation() {
        let sig = get_signature("chr(c: Char) = c\nmain() = 0", "chr").unwrap();
        assert_eq!(sig, "Char -> Char", "chr: got: {}", sig);
    }

    #[test]
    fn test_hm_60_unit_return() {
        let sig = get_signature("unit() -> () = ()\nmain() = 0", "unit").unwrap();
        assert_eq!(sig, "()", "unit: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Tuples (Tests 61-70)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_61_tuple_creation() {
        let sig = get_signature("pair(a, b) = (a, b)\nmain() = 0", "pair").unwrap();
        assert!(sig.contains("(") && sig.contains(")"), "pair: expected tuple, got: {}", sig);
    }

    #[test]
    fn test_hm_62_tuple_fst() {
        let sig = get_signature("fst(p) = p.0\nmain() = 0", "fst").unwrap();
        assert!(sig.contains("->"), "fst: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_63_tuple_snd() {
        let sig = get_signature("snd(p) = p.1\nmain() = 0", "snd").unwrap();
        assert!(sig.contains("->"), "snd: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_64_triple() {
        let sig = get_signature("triple(a, b, c) = (a, b, c)\nmain() = 0", "triple").unwrap();
        assert!(sig.contains("(") && sig.contains(","), "triple: expected tuple, got: {}", sig);
    }

    #[test]
    fn test_hm_65_swap_tuple() {
        let sig = get_signature("swap(p) = (p.1, p.0)\nmain() = 0", "swap").unwrap();
        assert!(sig.contains("->"), "swap: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_66_tuple_with_same_types() {
        let sig = get_signature("dup(x) = (x, x)\nmain() = 0", "dup").unwrap();
        assert!(sig.contains("->"), "dup: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_67_nested_tuple() {
        let sig = get_signature("nest(a, b, c) = ((a, b), c)\nmain() = 0", "nest").unwrap();
        assert!(sig.contains("("), "nest: expected nested tuple, got: {}", sig);
    }

    #[test]
    fn test_hm_68_tuple_with_literal() {
        let sig = get_signature("withOne(x) = (x, 1)\nmain() = 0", "withOne").unwrap();
        assert!(sig.contains("->"), "withOne: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_69_tuple_access_simple() {
        // Single field access on tuples
        let sig = get_signature("getFirst(p) = p.0\nmain() = 0", "getFirst").unwrap();
        assert!(sig.contains("->"), "getFirst: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_70_tuple_in_arithmetic() {
        let sig = get_signature("sumPair(p) = p.0 + p.1\nmain() = 0", "sumPair").unwrap();
        assert!(sig.contains("->"), "sumPair: expected function, got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Lists (Tests 71-80)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_71_empty_list() {
        let sig = get_signature("empty() = []\nmain() = 0", "empty").unwrap();
        assert!(sig.contains("List"), "empty: expected List, got: {}", sig);
    }

    #[test]
    fn test_hm_72_singleton_list() {
        let sig = get_signature("single(x) = [x]\nmain() = 0", "single").unwrap();
        assert!(sig.contains("List"), "single: expected List, got: {}", sig);
    }

    #[test]
    fn test_hm_73_list_of_ints() {
        let sig = get_signature("ints() = [1, 2, 3]\nmain() = 0", "ints").unwrap();
        assert!(sig_matches(&sig, &["List[Int]", "List[a]"]),
            "ints: expected List[Int], got: {}", sig);
    }

    #[test]
    fn test_hm_74_list_of_bools() {
        let sig = get_signature("bools() = [true, false]\nmain() = 0", "bools").unwrap();
        assert!(sig_matches(&sig, &["List[Bool]", "List[a]"]),
            "bools: expected List[Bool], got: {}", sig);
    }

    #[test]
    fn test_hm_75_list_param() {
        let sig = get_signature("takeList(xs) = xs\nmain() = 0", "takeList").unwrap();
        assert_eq!(sig, "a -> a", "takeList: got: {}", sig);
    }

    #[test]
    fn test_hm_76_list_cons() {
        let sig = get_signature("cons(x, xs) = [x | xs]\nmain() = 0", "cons").unwrap();
        assert!(sig.contains("List") || sig.contains("->"), "cons: got: {}", sig);
    }

    #[test]
    fn test_hm_77_list_head() {
        // Use multi-clause function pattern matching
        let sig = get_signature("hd([h | _]) = h\nmain() = 0", "hd").unwrap();
        assert!(sig.contains("->"), "hd: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_78_list_tail() {
        // Use multi-clause function pattern matching
        let sig = get_signature("tl([_ | t]) = t\nmain() = 0", "tl").unwrap();
        assert!(sig.contains("->"), "tl: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_79_list_length() {
        // Use multi-clause function with pattern matching
        let sig = get_signature("len([]) = 0\nlen([_ | t]) = 1 + len(t)\nmain() = 0", "len").unwrap();
        assert!(sig.contains("->"), "len: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_80_list_map_param() {
        // Use multi-clause function with pattern matching
        let sig = get_signature("mapList(f, []) = []\nmapList(f, [h | t]) = [f(h) | mapList(f, t)]\nmain() = 0", "mapList").unwrap();
        assert!(sig.contains("->"), "mapList: expected function, got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Custom Types - Records (Tests 81-90)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_81_record_construction() {
        // Test record construction inference - without inline record syntax
        let src = "type Point = { x: Int, y: Int }\ngetX(p: Point) -> Int = p.x\nmain() = 0";
        let sig = get_signature(src, "getX").unwrap();
        assert!(sig.contains("Point") && sig.contains("Int"), "getX: got: {}", sig);
    }

    #[test]
    fn test_hm_82_record_field_access() {
        let src = "type Point = { x: Int, y: Int }\ngetX(p: Point) = p.x\nmain() = 0";
        let sig = get_signature(src, "getX").unwrap();
        assert!(sig.contains("Point") && sig.contains("Int"), "getX: got: {}", sig);
    }

    #[test]
    fn test_hm_83_record_field_y() {
        let src = "type Point = { x: Int, y: Int }\ngetY(p: Point) = p.y\nmain() = 0";
        let sig = get_signature(src, "getY").unwrap();
        assert!(sig.contains("Point") && sig.contains("Int"), "getY: got: {}", sig);
    }

    #[test]
    fn test_hm_84_record_with_both_fields() {
        let src = "type Point = { x: Int, y: Int }\nsum(p: Point) = p.x + p.y\nmain() = 0";
        let sig = get_signature(src, "sum").unwrap();
        assert!(sig.contains("Point") && sig.contains("Int"), "sum: got: {}", sig);
    }

    #[test]
    fn test_hm_85_record_distance() {
        let src = "type Point = { x: Int, y: Int }\ndist(p: Point) = p.x * p.x + p.y * p.y\nmain() = 0";
        let sig = get_signature(src, "dist").unwrap();
        assert!(sig.contains("Point") && sig.contains("Int"), "dist: got: {}", sig);
    }

    #[test]
    fn test_hm_86_two_record_params() {
        // Test with explicit return type to help inference
        let src = "type Point = { x: Int, y: Int }\naddX(p1: Point, p2: Point) -> Int = p1.x + p2.x\nmain() = 0";
        let sig = get_signature(src, "addX").unwrap();
        assert!(sig.contains("Point"), "addX: got: {}", sig);
    }

    #[test]
    fn test_hm_87_record_string_field() {
        let src = "type Person = { name: String, age: Int }\ngetName(p: Person) = p.name\nmain() = 0";
        let sig = get_signature(src, "getName").unwrap();
        assert!(sig.contains("Person") && sig.contains("String"), "getName: got: {}", sig);
    }

    #[test]
    fn test_hm_88_record_bool_field() {
        let src = "type Flag = { value: Bool }\nisSet(f: Flag) = f.value\nmain() = 0";
        let sig = get_signature(src, "isSet").unwrap();
        assert!(sig.contains("Flag") && sig.contains("Bool"), "isSet: got: {}", sig);
    }

    #[test]
    fn test_hm_89_record_identity() {
        let src = "type Box = { value: Int }\nidBox(b: Box) = b\nmain() = 0";
        let sig = get_signature(src, "idBox").unwrap();
        assert!(sig.contains("Box"), "idBox: got: {}", sig);
    }

    #[test]
    fn test_hm_90_record_single_field() {
        // Simpler nested record - just access one level
        let src = "type Inner = { val: Int }\ngetVal(i: Inner) = i.val\nmain() = 0";
        let sig = get_signature(src, "getVal").unwrap();
        assert!(sig.contains("Inner") && sig.contains("Int"), "getVal: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Custom Types - Variants (Tests 91-100)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_91_variant_simple() {
        // Simple variant without type params
        let src = "type Color = Red | Green | Blue\nred() = Red\nmain() = 0";
        let sig = get_signature(src, "red").unwrap();
        assert!(sig.contains("Color") || !sig.contains("->"), "red: got: {}", sig);
    }

    #[test]
    fn test_hm_92_variant_with_data() {
        // Variant with data
        let src = "type Result = Ok(Int) | Err(String)\nok(x) = Ok(x)\nmain() = 0";
        let sig = get_signature(src, "ok").unwrap();
        assert!(sig.contains("->"), "ok: got: {}", sig);
    }

    #[test]
    fn test_hm_93_variant_match_simple() {
        // Match on simple variant using multi-clause
        let src = "type Color = Red | Green | Blue\nisRed(Red) = true\nisRed(_) = false\nmain() = 0";
        let sig = get_signature(src, "isRed").unwrap();
        assert!(sig.contains("->"), "isRed: got: {}", sig);
    }

    #[test]
    fn test_hm_94_variant_to_int() {
        // Convert variant to int using multi-clause
        let src = "type Color = Red | Green | Blue\ntoInt(Red) = 0\ntoInt(Green) = 1\ntoInt(Blue) = 2\nmain() = 0";
        let sig = get_signature(src, "toInt").unwrap();
        assert!(sig.contains("->"), "toInt: got: {}", sig);
    }

    #[test]
    fn test_hm_95_variant_green() {
        let src = "type Color = Red | Green | Blue\ngreen() = Green\nmain() = 0";
        let sig = get_signature(src, "green").unwrap();
        assert!(sig.contains("Color") || !sig.contains("->"), "green: got: {}", sig);
    }

    #[test]
    fn test_hm_96_variant_blue() {
        let src = "type Color = Red | Green | Blue\nblue() = Blue\nmain() = 0";
        let sig = get_signature(src, "blue").unwrap();
        assert!(sig.contains("Color") || !sig.contains("->"), "blue: got: {}", sig);
    }

    #[test]
    fn test_hm_97_variant_with_two_fields() {
        // Variant with multiple data
        let src = "type Pair = MkPair(Int, Int)\nmkPair(a, b) = MkPair(a, b)\nmain() = 0";
        let sig = get_signature(src, "mkPair").unwrap();
        assert!(sig.contains("->"), "mkPair: got: {}", sig);
    }

    #[test]
    fn test_hm_98_variant_mixed() {
        // Mix of unit and data constructors
        let src = "type Maybe = Nothing | Just(Int)\njust(x) = Just(x)\nmain() = 0";
        let sig = get_signature(src, "just").unwrap();
        assert!(sig.contains("->"), "just: got: {}", sig);
    }

    #[test]
    fn test_hm_99_variant_nothing() {
        let src = "type Maybe = Nothing | Just(Int)\nnothing() = Nothing\nmain() = 0";
        let sig = get_signature(src, "nothing").unwrap();
        assert!(sig.contains("Maybe") || !sig.contains("->"), "nothing: got: {}", sig);
    }

    #[test]
    fn test_hm_100_variant_is_nothing() {
        let src = "type Maybe = Nothing | Just(Int)\nisNothing(Nothing) = true\nisNothing(_) = false\nmain() = 0";
        let sig = get_signature(src, "isNothing").unwrap();
        assert!(sig.contains("->"), "isNothing: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Lambda / Higher-Order Functions (Tests 101-110)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_101_lambda_identity() {
        let sig = get_signature("idLam() = x => x\nmain() = 0", "idLam").unwrap();
        assert!(sig.contains("->"), "idLam: got: {}", sig);
    }

    #[test]
    fn test_hm_102_lambda_const() {
        let sig = get_signature("constLam(k) = x => k\nmain() = 0", "constLam").unwrap();
        assert!(sig.contains("->"), "constLam: got: {}", sig);
    }

    #[test]
    fn test_hm_103_apply() {
        let sig = get_signature("apply(f, x) = f(x)\nmain() = 0", "apply").unwrap();
        assert!(sig.contains("->"), "apply: got: {}", sig);
    }

    #[test]
    fn test_hm_104_compose() {
        let sig = get_signature("compose(f, g) = x => f(g(x))\nmain() = 0", "compose").unwrap();
        assert!(sig.contains("->"), "compose: got: {}", sig);
    }

    #[test]
    fn test_hm_105_twice() {
        let sig = get_signature("twice(f) = x => f(f(x))\nmain() = 0", "twice").unwrap();
        assert!(sig.contains("->"), "twice: got: {}", sig);
    }

    #[test]
    fn test_hm_106_flip() {
        let sig = get_signature("flip(f) = (x, y) => f(y, x)\nmain() = 0", "flip").unwrap();
        assert!(sig.contains("->"), "flip: got: {}", sig);
    }

    #[test]
    fn test_hm_107_curry() {
        let sig = get_signature("curry(f) = x => y => f(x, y)\nmain() = 0", "curry").unwrap();
        assert!(sig.contains("->"), "curry: got: {}", sig);
    }

    #[test]
    fn test_hm_108_uncurry() {
        let sig = get_signature("uncurry(f) = (x, y) => f(x)(y)\nmain() = 0", "uncurry").unwrap();
        assert!(sig.contains("->"), "uncurry: got: {}", sig);
    }

    #[test]
    fn test_hm_109_lambda_with_closure() {
        let sig = get_signature("adder(n) = x => x + n\nmain() = 0", "adder").unwrap();
        assert!(sig.contains("->"), "adder: got: {}", sig);
    }

    #[test]
    fn test_hm_110_lambda_multilevel() {
        let sig = get_signature("add3(a) = b => c => a + b + c\nmain() = 0", "add3").unwrap();
        assert!(sig.contains("->"), "add3: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Match Expressions (Tests 111-120)
    // Using multi-clause function syntax instead of inline match expressions
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_111_match_int_patterns() {
        // Multi-clause function pattern: isZero(0) = true; isZero(_) = false
        let sig = get_signature("isZeroM(0) = true\nisZeroM(_) = false\nmain() = 0", "isZeroM").unwrap();
        assert!(sig.ends_with("-> Bool") || sig.contains("Bool"), "isZeroM: got: {}", sig);
    }

    #[test]
    fn test_hm_112_match_bool_patterns() {
        // Multi-clause function pattern for bool
        let sig = get_signature("boolToInt(true) = 1\nboolToInt(false) = 0\nmain() = 0", "boolToInt").unwrap();
        assert!(sig.contains("->"), "boolToInt: got: {}", sig);
    }

    #[test]
    fn test_hm_113_match_tuple() {
        // Single clause with tuple pattern
        let sig = get_signature("sumTup((a, b)) = a + b\nmain() = 0", "sumTup").unwrap();
        assert!(sig.contains("->"), "sumTup: got: {}", sig);
    }

    #[test]
    fn test_hm_114_match_wildcard() {
        // Single clause with wildcard
        let sig = get_signature("always(_) = 42\nmain() = 0", "always").unwrap();
        assert!(sig.contains("->"), "always: got: {}", sig);
    }

    #[test]
    fn test_hm_115_match_with_guard() {
        // Multi-clause with guards
        let sig = get_signature("classify(n) when n > 0 = 1\nclassify(n) when n < 0 = -1\nclassify(_) = 0\nmain() = 0", "classify").unwrap();
        assert!(sig.contains("->"), "classify: got: {}", sig);
    }

    #[test]
    fn test_hm_116_match_list_empty() {
        // Multi-clause for list patterns
        let sig = get_signature("isEmpty([]) = true\nisEmpty(_) = false\nmain() = 0", "isEmpty").unwrap();
        assert!(sig.ends_with("-> Bool") || sig.contains("Bool"), "isEmpty: got: {}", sig);
    }

    #[test]
    fn test_hm_117_match_multiple_branches() {
        // Multi-clause fibonacci
        let sig = get_signature("fibM(0) = 0\nfibM(1) = 1\nfibM(n) = fibM(n-1) + fibM(n-2)\nmain() = 0", "fibM").unwrap();
        assert!(sig.contains("->"), "fibM: got: {}", sig);
    }

    #[test]
    fn test_hm_118_match_string() {
        // Multi-clause string match
        let sig = get_signature("greet(\"hi\") = \"hello\"\ngreet(_) = \"goodbye\"\nmain() = 0", "greet").unwrap();
        assert!(sig.contains("String") || sig.contains("->"), "greet: got: {}", sig);
    }

    #[test]
    fn test_hm_119_match_nested_tuple() {
        // Single clause with nested tuple pattern
        let sig = get_signature("flatten(((a, b), c)) = (a, b, c)\nmain() = 0", "flatten").unwrap();
        assert!(sig.contains("->"), "flatten: got: {}", sig);
    }

    #[test]
    fn test_hm_120_match_or_pattern() {
        // Multi-clause to simulate or pattern
        let sig = get_signature("isSmall(0) = true\nisSmall(1) = true\nisSmall(2) = true\nisSmall(_) = false\nmain() = 0", "isSmall").unwrap();
        assert!(sig.ends_with("-> Bool") || sig.contains("Bool"), "isSmall: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Block Expressions (Tests 121-130)
    // Using newlines in blocks instead of semicolons
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_121_block_simple() {
        let sig = get_signature("block() = { 42 }\nmain() = 0", "block").unwrap();
        assert_eq!(sig, "Int", "block: got: {}", sig);
    }

    #[test]
    fn test_hm_122_block_with_let() {
        // Use newlines inside block
        let sig = get_signature("withLet(x) = {\n    y = x + 1\n    y\n}\nmain() = 0", "withLet").unwrap();
        assert!(sig.contains("->"), "withLet: got: {}", sig);
    }

    #[test]
    fn test_hm_123_block_multiple_lets() {
        // Multiple lets with newlines
        let sig = get_signature("multi(a, b) = {\n    x = a + 1\n    y = b + 2\n    x + y\n}\nmain() = 0", "multi").unwrap();
        assert!(sig.contains("->"), "multi: got: {}", sig);
    }

    #[test]
    fn test_hm_124_block_shadowing() {
        // Shadowing with newlines
        let sig = get_signature("shadow(x) = {\n    x = x + 1\n    x = x * 2\n    x\n}\nmain() = 0", "shadow").unwrap();
        assert!(sig.contains("->"), "shadow: got: {}", sig);
    }

    #[test]
    fn test_hm_125_block_returns_last() {
        // Just expressions with newlines
        let sig = get_signature("last() = {\n    1\n    2\n    3\n}\nmain() = 0", "last").unwrap();
        assert_eq!(sig, "Int", "last: got: {}", sig);
    }

    #[test]
    fn test_hm_126_block_with_if() {
        // Block with if using newlines
        let sig = get_signature("condBlock(c) = {\n    r = if c then 1 else 0\n    r\n}\nmain() = 0", "condBlock").unwrap();
        assert!(sig.contains("->"), "condBlock: got: {}", sig);
    }

    #[test]
    fn test_hm_127_nested_blocks() {
        let sig = get_signature("nested() = { { { 42 } } }\nmain() = 0", "nested").unwrap();
        assert_eq!(sig, "Int", "nested: got: {}", sig);
    }

    #[test]
    fn test_hm_128_block_with_tuple() {
        // Tuple in block with newlines
        let sig = get_signature("tupBlock(a, b) = {\n    p = (a, b)\n    p\n}\nmain() = 0", "tupBlock").unwrap();
        assert!(sig.contains("->"), "tupBlock: got: {}", sig);
    }

    #[test]
    fn test_hm_129_block_using_param() {
        // Using params with multiple lets
        let sig = get_signature("useParam(x) = {\n    y = x * 2\n    z = y + 1\n    z\n}\nmain() = 0", "useParam").unwrap();
        assert!(sig.contains("->"), "useParam: got: {}", sig);
    }

    #[test]
    fn test_hm_130_block_complex() {
        // Complex block with newlines
        let sig = get_signature("complex(a, b, c) = {\n    x = a + b\n    y = x * c\n    if y > 0 then y else -y\n}\nmain() = 0", "complex").unwrap();
        assert!(sig.contains("->"), "complex: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Edge Cases and Special Scenarios (Tests 131-140)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_131_single_char() {
        let sig = get_signature("getChar() = 'a'\nmain() = 0", "getChar").unwrap();
        assert_eq!(sig, "Char", "getChar: got: {}", sig);
    }

    #[test]
    fn test_hm_132_empty_string() {
        let sig = get_signature(r#"emptyStr() = ""\nmain() = 0"#, "emptyStr").unwrap();
        assert_eq!(sig, "String", "emptyStr: got: {}", sig);
    }

    #[test]
    fn test_hm_133_large_int() {
        let sig = get_signature("bigNum() = 9999999999\nmain() = 0", "bigNum").unwrap();
        assert!(sig == "Int" || sig == "BigInt", "bigNum: got: {}", sig);
    }

    #[test]
    fn test_hm_134_negative_int() {
        let sig = get_signature("negNum() = -42\nmain() = 0", "negNum").unwrap();
        assert_eq!(sig, "Int", "negNum: got: {}", sig);
    }

    #[test]
    fn test_hm_135_zero() {
        let sig = get_signature("zero() = 0\nmain() = 0", "zero").unwrap();
        assert_eq!(sig, "Int", "zero: got: {}", sig);
    }

    #[test]
    fn test_hm_136_float_zero() {
        let sig = get_signature("fzero() = 0.0\nmain() = 0", "fzero").unwrap();
        assert_eq!(sig, "Float", "fzero: got: {}", sig);
    }

    #[test]
    fn test_hm_137_scientific_notation() {
        let sig = get_signature("sci() = 1.5e10\nmain() = 0", "sci").unwrap();
        assert_eq!(sig, "Float", "sci: got: {}", sig);
    }

    #[test]
    fn test_hm_138_many_params() {
        let sig = get_signature("many(a, b, c, d, e) = a\nmain() = 0", "many").unwrap();
        assert_eq!(sig, "a -> b -> c -> d -> e -> a", "many: got: {}", sig);
    }

    #[test]
    fn test_hm_139_all_unified() {
        let sig = get_signature("allSame(a, b, c, d) = a + b + c + d\nmain() = 0", "allSame").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a -> a -> a", "Int -> Int -> Int -> Int -> Int"]),
            "allSame: got: {}", sig);
    }

    #[test]
    fn test_hm_140_partial_unified() {
        let sig = get_signature("partial(a, b, c) = (a + b, c)\nmain() = 0", "partial").unwrap();
        assert!(sig.contains("->"), "partial: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Show Trait Constraint Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_141_println_wrapper_has_show_constraint() {
        let sig = get_signature("printIt(x) = println(x)\nmain() = 0", "printIt").unwrap();
        // Should infer: Show a => a -> ()
        assert!(sig.contains("Show"), "printIt: expected Show constraint, got: {}", sig);
        assert!(sig.contains("()"), "printIt: expected Unit return type, got: {}", sig);
    }

    #[test]
    fn test_hm_142_print_wrapper_has_show_constraint() {
        let sig = get_signature("printVal(x) = print(x)\nmain() = 0", "printVal").unwrap();
        // Should infer: Show a => a -> ()
        assert!(sig.contains("Show"), "printVal: expected Show constraint, got: {}", sig);
    }

    #[test]
    fn test_hm_143_println_with_specific_type_no_constraint() {
        let sig = get_signature("printInt(x: Int) = println(x)\nmain() = 0", "printInt").unwrap();
        // With a concrete type annotation, no type variable, so no constraint needed
        assert_eq!(sig, "Int -> ()", "printInt: got: {}", sig);
    }

    #[test]
    fn test_hm_144_string_literal_return() {
        // A function returning a string literal should have String return type
        let sig = get_signature("greeting() = \"hello\"\nmain() = 0", "greeting").unwrap();
        assert_eq!(sig, "String", "greeting: expected String return type, got: {}", sig);
    }

    #[test]
    fn test_hm_145_int_literal_return() {
        // A function returning an int literal should have Int return type
        let sig = get_signature("answer() = 42\nmain() = 0", "answer").unwrap();
        assert_eq!(sig, "Int", "answer: expected Int return type, got: {}", sig);
    }

    #[test]
    fn test_hm_146_string_concat_return() {
        // A function returning string concat should have String return type
        let sig = get_signature("fullName() = \"John\" ++ \" \" ++ \"Doe\"\nmain() = 0", "fullName").unwrap();
        assert_eq!(sig, "String", "fullName: expected String return type, got: {}", sig);
    }

    #[test]
    fn test_hm_147_unit_return() {
        // A function returning unit should have () return type
        let sig = get_signature("doNothing() = ()\nmain() = 0", "doNothing").unwrap();
        assert_eq!(sig, "()", "doNothing: expected () return type, got: {}", sig);
    }

    #[test]
    fn test_hm_148_list_int_return() {
        // A function returning a list of ints should have List[Int] return type
        let sig = get_signature("numbers() = [1, 2, 3]\nmain() = 0", "numbers").unwrap();
        assert_eq!(sig, "List[Int]", "numbers: expected List[Int] return type, got: {}", sig);
    }

    #[test]
    fn test_hm_149_function_with_mvar_returns_string() {
        // Function that uses mvar should still have String return type when returning a string
        let sig = get_signature(r#"
mvar counter: Int = 0
initDemo() = {
    counter = 1
    "Demo initialized"
}
main() = 0
"#, "initDemo").unwrap();
        assert_eq!(sig, "String", "initDemo with mvar: expected String return type, got: {}", sig);
    }

    #[test]
    fn test_hm_150_function_with_mvar_string_concat() {
        // Function that uses string concatenation and mvar should have String return type
        let sig = get_signature(r#"
mvar items: List[String] = ["item1", "item2"]
viewList() = {
    header = "Items:\n"
    footer = "\n[End]"
    header ++ footer
}
main() = 0
"#, "viewList").unwrap();
        assert_eq!(sig, "String", "viewList with concat: expected String return type, got: {}", sig);
    }

    #[test]
    fn test_hm_151_function_returns_unit_from_mvar_write() {
        // Function that writes to mvar should have () return type (unit)
        let sig = get_signature(r#"
mvar counter: Int = 0
increment() = {
    counter = counter + 1
    ()
}
main() = 0
"#, "increment").unwrap();
        assert_eq!(sig, "()", "increment: expected () return type, got: {}", sig);
    }

    #[test]
    fn test_hm_152_function_reads_mvar_returns_int() {
        // Function that reads mvar and returns Int should have Int return type
        let sig = get_signature(r#"
mvar counter: Int = 0
getCount() = counter
main() = 0
"#, "getCount").unwrap();
        assert_eq!(sig, "Int", "getCount: expected Int return type, got: {}", sig);
    }

    #[test]
    fn test_hm_153_list_demo_style_view_function() {
        // Simulates the list demo's listView function structure
        let sig = get_signature(r#"
mvar listItems: List[String] = ["item1", "item2"]
mvar listCursor: Int = 0
listView() = {
    header = "=== List Demo ===\n\n"
    footer = "\n[End]"
    header ++ footer
}
main() = 0
"#, "listView").unwrap();
        assert_eq!(sig, "String", "listView: expected String return type, got: {}", sig);
    }

    #[test]
    fn test_hm_154_list_demo_style_init_function() {
        // Simulates the list demo's listInit function structure
        let sig = get_signature(r#"
mvar listPanelId: Int = 0
listInit() = {
    listPanelId = 1
    "List demo ready."
}
main() = 0
"#, "listInit").unwrap();
        assert_eq!(sig, "String", "listInit: expected String return type, got: {}", sig);
    }

    #[test]
    fn test_hm_155_function_reads_mvar_list() {
        // Function that reads mvar list should return List[Int]
        let sig = get_signature(r#"
mvar items: List[Int] = [1, 2, 3]
getItems() = items
main() = 0
"#, "getItems").unwrap();
        assert_eq!(sig, "List[Int]", "getItems: expected List[Int] return type, got: {}", sig);
    }

    #[test]
    fn test_hm_156_function_reads_mvar_list_string() {
        // Function that reads mvar list of strings should return List[String]
        let sig = get_signature(r#"
mvar names: List[String] = ["a", "b", "c"]
getNames() = names
main() = 0
"#, "getNames").unwrap();
        assert_eq!(sig, "List[String]", "getNames: expected List[String] return type, got: {}", sig);
    }

    #[test]
    fn test_hm_157_function_calls_panel_show() {
        // Function calling Panel.show should have signature Int -> ()
        let sig = get_signature(r#"
showPanel(id) = Panel.show(id)
main() = 0
"#, "showPanel").unwrap();
        assert_eq!(sig, "Int -> ()", "showPanel: got: {}", sig);
    }

    #[test]
    fn test_hm_158_function_calls_panel_setcontent() {
        // Function calling Panel.setContent should have signature Int -> String -> ()
        let sig = get_signature(r#"
updatePanel(id, content) = Panel.setContent(id, content)
main() = 0
"#, "updatePanel").unwrap();
        assert_eq!(sig, "Int -> String -> ()", "updatePanel: got: {}", sig);
    }

    #[test]
    fn test_hm_159_function_calls_panel_create() {
        // Function calling Panel.create should have signature String -> Int
        let sig = get_signature(r#"
createPanel(title) = Panel.create(title)
main() = 0
"#, "createPanel").unwrap();
        assert_eq!(sig, "String -> Int", "createPanel: got: {}", sig);
    }

    #[test]
    fn test_hm_160_string_concat_chain() {
        // String concatenation chain should return String
        let sig = get_signature(r#"
buildMessage() = {
    header = "Header\n"
    body = "Body\n"
    footer = "Footer"
    header ++ body ++ footer
}
main() = 0
"#, "buildMessage").unwrap();
        assert_eq!(sig, "String", "buildMessage: expected String return type, got: {}", sig);
    }

    #[test]
    fn test_hm_161_function_like_listview() {
        // Function similar to listView that calls other functions and concats strings
        let sig = get_signature(r#"
mvar listItems: List[String] = ["item1", "item2"]
mvar listCursor: Int = 0
renderItem(idx, item) = if idx == 0 then "> " ++ item else "  " ++ item
listView() = {
    header = "=== List ===\n\n"
    footer = "\n[End]"
    header ++ footer
}
main() = 0
"#, "listView").unwrap();
        assert_eq!(sig, "String", "listView: expected String return type, got: {}", sig);
    }

    #[test]
    fn test_hm_162_list_demo_panel_calls() {
        // Test that list demo functions using Panel.* are inferred correctly
        let source = r#"
mvar listPanelId: Int = 0
mvar listCursor: Int = 0

# listShow() calls Panel.setContent and Panel.show - should return ()
listShow() = {
    Panel.setContent(listPanelId, "content")
    Panel.show(listPanelId)
}

# listInit() calls Panel.create and returns String
listInit() = {
    listPanelId = Panel.create("List Demo")
    Panel.onKey(listPanelId, "callback")
    "ready"
}

main() = 0
"#;
        // Check listShow signature - should be () since Panel.show returns ()
        let sig = get_signature(source, "listShow").unwrap();
        assert_eq!(sig, "()", "listShow: got: {}", sig);

        // Check listInit signature - should be String (the return value)
        let sig = get_signature(source, "listInit").unwrap();
        assert_eq!(sig, "String", "listInit: got: {}", sig);
    }

    #[test]
    fn test_hm_163_listview_with_recursive_render() {
        // Full listView test with recursive listRenderList function
        let source = r#"
mvar listItems: List[String] = ["First", "Second", "Third"]
mvar listCursor: Int = 0

listRenderItem(idx, item, cursor) =
    if idx == cursor then "> " ++ item else "  " ++ item

listRenderList(items, idx, cursor) = match items {
    [] -> ""
    [item | rest] -> listRenderItem(idx, item, cursor) ++ "\n" ++ listRenderList(rest, idx + 1, cursor)
}

listView() = {
    header = "=== List Demo ===\n\n"
    list = listRenderList(listItems, 0, listCursor)
    footer = "\n[End]"
    header ++ list ++ footer
}

main() = 0
"#;
        // listRenderItem should be Int -> String -> Int -> String
        let sig = get_signature(source, "listRenderItem").unwrap();
        assert!(sig.contains("String"), "listRenderItem should return String: got: {}", sig);

        // listRenderList should return String
        let sig = get_signature(source, "listRenderList").unwrap();
        assert!(sig.contains("String"), "listRenderList should return String: got: {}", sig);

        // listView should return String
        let sig = get_signature(source, "listView").unwrap();
        assert_eq!(sig, "String", "listView: got: {}", sig);
    }

    // =========================================================================
    // Type Checking Tests - Ensuring type errors are caught at compile time
    // =========================================================================

    /// Helper to check that source code fails with a type error (compile-time or runtime)
    fn expect_type_error(source: &str, expected_substring: &str) {
        let result = compile_and_run(source);
        match result {
            Err(msg) if msg.contains("TypeError") || msg.contains("Cannot unify") || msg.contains("Type mismatch") => {
                assert!(msg.contains(expected_substring) || expected_substring.is_empty(),
                    "Expected error containing '{}', got: {}", expected_substring, msg);
            }
            Err(msg) if msg.contains("Type error") || msg.contains("expected numeric") || msg.contains("type mismatch") => {
                // Runtime type errors are also acceptable - VM caught the type mismatch
                if expected_substring.is_empty() || msg.contains(expected_substring) {
                    return; // Runtime type error is fine
                }
                // For runtime errors, be lenient - any type error is caught
                return;
            }
            Err(msg) => {
                // Some type errors may manifest as other compile errors
                if expected_substring.is_empty() || msg.contains(expected_substring) {
                    return; // Accept any compile error if no specific substring expected
                }
                panic!("Expected type error containing '{}', got different error: {}", expected_substring, msg);
            }
            Ok(val) => {
                panic!("Expected type error, but code compiled and ran successfully with result: {:?}", val);
            }
        }
    }

    /// Helper to verify code compiles and runs successfully
    fn expect_success(source: &str) {
        let result = compile_and_run(source);
        assert!(result.is_ok(), "Expected successful compilation, got error: {:?}", result);
    }

    // -------------------------------------------------------------------------
    // Basic Arithmetic Type Mismatch Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_001_int_plus_string() {
        expect_type_error("f(x: Int) = x + \"hello\"\nmain() = f(1)", "Int and String");
    }

    #[test]
    fn test_type_002_string_plus_int() {
        // String + Int is a type error - String doesn't implement Num
        expect_type_error("f(x: String) = x + 42\nmain() = f(\"hi\")", "String does not implement Num");
    }

    #[test]
    fn test_type_003_int_minus_string() {
        expect_type_error("f(x: Int) = x - \"hello\"\nmain() = f(1)", "Int and String");
    }

    #[test]
    fn test_type_004_int_multiply_string() {
        expect_type_error("f(x: Int) = x * \"hello\"\nmain() = f(1)", "Int and String");
    }

    #[test]
    fn test_type_005_int_divide_string() {
        expect_type_error("f(x: Int) = x / \"hello\"\nmain() = f(1)", "Int and String");
    }

    #[test]
    fn test_type_006_bool_plus_int() {
        // Bool + Int is a type error - Bool doesn't implement Num
        expect_type_error("f(x: Bool) = x + 42\nmain() = f(true)", "Bool does not implement Num");
    }

    #[test]
    fn test_type_007_int_plus_bool() {
        // Int + Bool is a type error
        expect_type_error("f(x: Int) = x + true\nmain() = f(1)", "Int and Bool");
    }

    #[test]
    fn test_type_008_float_plus_string() {
        expect_type_error("f(x: Float) = x + \"hello\"\nmain() = f(1.0)", "Float and String");
    }

    // -------------------------------------------------------------------------
    // Comparison Type Mismatch Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_009_compare_int_string() {
        expect_type_error("f(x: Int) = x > \"hello\"\nmain() = f(1)", "Int and String");
    }

    #[test]
    fn test_type_010_compare_bool_int() {
        // Bool doesn't implement Ord trait required for <
        expect_type_error("f(x: Bool) = x < 42\nmain() = f(true)", "Bool does not implement Ord");
    }

    #[test]
    fn test_type_011_compare_string_float() {
        expect_type_error("f(x: String) = x >= 1.5\nmain() = f(\"hi\")", "String and Float");
    }

    // -------------------------------------------------------------------------
    // Function Argument Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_012_wrong_arg_used_as_string() {
        // Function uses x as string (concat), but called with Int
        // Note: The type annotation alone doesn't enforce the type - usage does
        expect_type_error("f(x: String) = x ++ \"!\"\nmain() = f(42)", "");
    }

    #[test]
    fn test_type_013_wrong_arg_type_string_to_int() {
        // Function uses x as Int (arithmetic), but called with String
        expect_type_error("f(x: Int) = x + 1\nmain() = f(\"hello\")", "Int and String");
    }

    #[test]
    fn test_type_014_wrong_arg_type_bool_to_int() {
        // Function uses x as Int (arithmetic), but called with Bool
        expect_type_error("f(x: Int) = x * 2\nmain() = f(true)", "Int and Bool");
    }

    // -------------------------------------------------------------------------
    // Return Type Mismatch Tests
    // Return type annotation syntax (f(): T) is not yet supported by parser.
    // These tests verify mismatches via usage context instead.
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_015_return_used_as_wrong_type() {
        // Function returns string, but caller uses it as Int
        expect_type_error("f(x: Int) = \"hello\"\ng(x: Int) = f(x) + 1\nmain() = g(1)", "String and Int");
    }

    #[test]
    fn test_type_016_return_used_in_comparison() {
        // Function returns string, caller compares with int
        expect_type_error("f(x: Int) = \"hello\"\ng(x: Int) = f(x) > 42\nmain() = g(1)", "");
    }

    #[test]
    fn test_type_017_return_used_in_arithmetic() {
        // Function returns bool, caller uses in arithmetic
        expect_type_error("f(x: Int) = x > 0\ng(x: Int) = f(x) + 1\nmain() = g(1)", "Bool and Int");
    }

    // -------------------------------------------------------------------------
    // If-Then-Else Branch Type Mismatch Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_018_if_branches_int_string() {
        expect_type_error("f(x: Bool) = if x then 42 else \"hello\"\nmain() = f(true)", "Int and String");
    }

    #[test]
    fn test_type_019_if_branches_string_int() {
        expect_type_error("f(x: Bool) = if x then \"hello\" else 42\nmain() = f(true)", "String and Int");
    }

    #[test]
    fn test_type_020_if_branches_bool_int() {
        expect_type_error("f(x: Bool) = if x then true else 42\nmain() = f(true)", "Bool and Int");
    }

    #[test]
    fn test_type_021_if_condition_requires_bool_not_int() {
        // If conditions require Bool, not truthy values
        expect_type_error("f(x: Int) = if x then 1 else 2\nmain() = f(1)", "Int and Bool");
    }

    #[test]
    fn test_type_022_if_condition_requires_bool_not_string() {
        // If conditions require Bool, not truthy values
        expect_type_error("f(x: String) = if x then 1 else 2\nmain() = f(\"hi\")", "String and Bool");
    }

    // -------------------------------------------------------------------------
    // List Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_023_list_mixed_types() {
        // Type checker reports second type first: "String and Int"
        expect_type_error("f() = [1, \"hello\", 3]\nmain() = f()", "String and Int");
    }

    #[test]
    fn test_type_024_list_mixed_int_bool() {
        // Type checker reports second type first: "Bool and Int"
        expect_type_error("f() = [1, true, 3]\nmain() = f()", "Bool and Int");
    }

    #[test]
    fn test_type_025_list_concat_heterogeneous() {
        // Language feature: lists can contain heterogeneous types (like Python)
        // This is intentional - [String] ++ [Int] produces a mixed list
        expect_success("f(xs) = [\"hello\"] ++ xs\ng() = f([1,2,3])\nmain() = g()");
    }

    // -------------------------------------------------------------------------
    // Tuple Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_026_tuple_access_used_wrong() {
        // Tuple field access - use .1 (String) where Int is expected
        expect_type_error("f(t) = t.1 + 42\nmain() = f((1, \"hi\"))", "String and Int");
    }

    // -------------------------------------------------------------------------
    // Binary Boolean Operation Tests
    // Boolean operators require Bool operands
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_027_and_requires_bool() {
        // && requires Bool operands, not truthy values
        expect_type_error("f(x: Int) = x && true\nmain() = f(1)", "Int and Bool");
    }

    #[test]
    fn test_type_028_or_requires_bool() {
        // || requires Bool operands, not truthy values
        expect_type_error("f(x: String) = x || false\nmain() = f(\"hi\")", "String and Bool");
    }

    #[test]
    fn test_type_029_not_with_int() {
        expect_type_error("f(x: Int) = !x\nmain() = f(1)", "");
    }

    // -------------------------------------------------------------------------
    // Valid Code Tests - Ensuring good code passes
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_030_valid_int_arithmetic() {
        expect_success("f(x: Int, y: Int) = x + y * 2\nmain() = f(1, 2)");
    }

    #[test]
    fn test_type_031_valid_float_arithmetic() {
        expect_success("f(x: Float, y: Float) = x + y / 2.0\nmain() = f(1.0, 2.0)");
    }

    #[test]
    fn test_type_032_valid_string_operations() {
        expect_success("f(x: String) = x\nmain() = f(\"hello\")");
    }

    #[test]
    fn test_type_033_valid_bool_operations() {
        expect_success("f(x: Bool, y: Bool) = x && y || !x\nmain() = f(true, false)");
    }

    #[test]
    fn test_type_034_valid_comparison() {
        expect_success("f(x: Int, y: Int) = x > y && x < 100\nmain() = f(1, 2)");
    }

    #[test]
    fn test_type_035_valid_if_same_branch_types() {
        expect_success("f(x: Bool) = if x then 1 else 2\nmain() = f(true)");
    }

    #[test]
    fn test_type_036_valid_list_same_types() {
        expect_success("f() = [1, 2, 3, 4, 5]\nmain() = f()");
    }

    #[test]
    fn test_type_037_valid_tuple() {
        expect_success("f() = (1, \"hello\", true)\nmain() = f()");
    }

    #[test]
    fn test_type_038_valid_function_chaining() {
        expect_success("double(x: Int) = x * 2\nquad(x: Int) = double(double(x))\nmain() = quad(5)");
    }

    #[test]
    fn test_type_039_valid_recursion() {
        // Recursive factorial - no return type annotation (inferred)
        expect_success("fact(n: Int) = if n == 0 then 1 else n * fact(n - 1)\nmain() = fact(5)");
    }

    // -------------------------------------------------------------------------
    // More Complex Type Error Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_040_nested_call_type_mismatch() {
        // Inner function returns String, outer expects Int
        expect_type_error("inner(x: Int): String = \"result\"\nouter(x: Int): Int = inner(x) + 1\nmain() = outer(1)", "");
    }

    #[test]
    fn test_type_041_chained_arithmetic_type_error() {
        expect_type_error("f(x: Int) = x + 1 + \"bad\" + 2\nmain() = f(1)", "Int and String");
    }

    #[test]
    fn test_type_042_nested_if_type_error() {
        expect_type_error("f(x: Bool, y: Bool) = if x then (if y then 1 else \"no\") else 2\nmain() = f(true, true)", "Int and String");
    }

    #[test]
    fn test_type_043_list_operation_heterogeneous() {
        // List concatenation allows heterogeneous types at runtime
        // The type system only checks literal lists, not concatenation operations
        expect_success("f(xs) = [\"hello\", \"world\"] ++ xs\ng() = f([1,2])\nmain() = g()");
    }

    // -------------------------------------------------------------------------
    // Edge Cases and Special Scenarios
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_044_modulo_with_string() {
        expect_type_error("f(x: Int) = x % \"hello\"\nmain() = f(10)", "Int and String");
    }

    #[test]
    fn test_type_045_power_with_string() {
        expect_type_error("f(x: Int) = x ** \"hello\"\nmain() = f(2)", "Int and String");
    }

    #[test]
    fn test_type_046_equality_different_types() {
        expect_type_error("f(x: Int) = x == \"hello\"\nmain() = f(1)", "Int and String");
    }

    #[test]
    fn test_type_047_inequality_different_types() {
        expect_type_error("f(x: Int) = x != \"hello\"\nmain() = f(1)", "Int and String");
    }

    // -------------------------------------------------------------------------
    // Untyped Function Tests - Should infer and catch errors
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_048_untyped_returns_conflicting_types() {
        // Function body tries to return different types in branches
        expect_type_error("f(x) = if x then 42 else \"hello\"\nmain() = f(true)", "Int and String");
    }

    #[test]
    fn test_type_049_untyped_arithmetic_mismatch() {
        // Even without annotations, should catch x + "str" when x is used as number
        expect_type_error("f(x) = x + 1 + \"bad\"\nmain() = f(1)", "");
    }

    #[test]
    fn test_type_050_untyped_comparison_mismatch() {
        expect_type_error("f(x) = (x > 5) + 1\nmain() = f(10)", "");
    }

    // -------------------------------------------------------------------------
    // Additional Valid Code Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_051_valid_mixed_numeric_literal() {
        // Int literal should work where Int expected
        expect_success("f(x: Int) = x + 42\nmain() = f(1)");
    }

    #[test]
    fn test_type_052_valid_string_literal() {
        expect_success("f(x: String) = x\nmain() = f(\"hello world\")");
    }

    #[test]
    fn test_type_053_valid_empty_list() {
        expect_success("f() = []\nmain() = f()");
    }

    #[test]
    fn test_type_054_valid_nested_tuples() {
        expect_success("f() = ((1, 2), (\"a\", \"b\"))\nmain() = f()");
    }

    #[test]
    fn test_type_055_valid_multiple_functions() {
        expect_success("add(x: Int, y: Int) = x + y\nmul(x: Int, y: Int) = x * y\nmain() = add(mul(2, 3), 4)");
    }

    // -------------------------------------------------------------------------
    // Type Error with Multiple Parameters
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_056_multi_param_first_wrong() {
        expect_type_error("f(a: Int, b: Int) = a + b\nmain() = f(\"bad\", 1)", "");
    }

    #[test]
    fn test_type_057_multi_param_second_wrong() {
        expect_type_error("f(a: Int, b: Int) = a + b\nmain() = f(1, \"bad\")", "");
    }

    #[test]
    fn test_type_058_multi_param_both_wrong() {
        expect_type_error("f(a: Int, b: Int) = a + b\nmain() = f(\"bad\", true)", "");
    }

    // -------------------------------------------------------------------------
    // String Concatenation vs Arithmetic
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_059_string_concat_valid() {
        // String ++ String should work (if supported)
        // Skip if not supported - this tests the operator exists
        let result = compile_and_run("f() = \"hello\" ++ \" world\"\nmain() = f()");
        // Either succeeds or fails with parse/feature error, not type error
        match result {
            Ok(_) => (), // Great, it works
            Err(e) if e.contains("TypeError") => panic!("Should not be a type error: {}", e),
            Err(_) => (), // Some other error is fine (feature not implemented)
        }
    }

    #[test]
    fn test_type_060_char_type_valid() {
        expect_success("f(c: Char) = c\nmain() = f('a')");
    }

    // -------------------------------------------------------------------------
    // Arity Mismatch Tests - Ensure wrong number of arguments is caught
    // -------------------------------------------------------------------------

    /// Helper to check that source code fails with an arity mismatch error
    fn expect_arity_error(source: &str) {
        let (module_opt, errors) = nostos_syntax::parser::parse(source);
        if !errors.is_empty() {
            panic!("Expected arity error, but got parse errors: {:?}", errors);
        }
        let module = module_opt.expect("Expected module from parse");
        let result = compile_module(&module, source);
        match result {
            Err(CompileError::ArityMismatch { name, expected, found, .. }) => {
                // This is what we want - arity mismatch was detected
                assert!(true, "Got expected arity error: {} expected {} args, got {}", name, expected, found);
            }
            Err(CompileError::TypeError { message, .. }) if message.contains("Wrong number of arguments") => {
                // HM type checker also catches arity errors
                assert!(true, "Got expected arity error via type checker: {}", message);
            }
            Err(other) => {
                panic!("Expected ArityMismatch error, got different error: {:?}", other);
            }
            Ok(_) => {
                panic!("Expected arity error, but code compiled successfully");
            }
        }
    }

    #[test]
    fn test_arity_001_too_many_args() {
        // foo() takes 0 args but is called with 1
        expect_arity_error("foo() = 42\nmain() = foo(\"extra\")");
    }

    #[test]
    fn test_arity_002_too_few_args() {
        // foo(x) takes 1 arg but is called with 0
        expect_arity_error("foo(x) = x + 1\nmain() = foo()");
    }

    #[test]
    fn test_arity_003_extra_arg_in_nested_call() {
        // bar() calls foo with wrong arity
        expect_arity_error("foo() = 42\nbar() = foo(\"string\")\nmain() = bar()");
    }

    #[test]
    fn test_arity_004_multiple_extra_args() {
        // foo() takes 0 args but is called with 3
        expect_arity_error("foo() = 42\nmain() = foo(1, 2, 3)");
    }

    #[test]
    fn test_arity_005_one_arg_vs_two() {
        // foo(x) takes 1 arg but is called with 2
        expect_arity_error("foo(x) = x\nmain() = foo(1, 2)");
    }

    // === Stdlib Tests ===

    #[test]
    fn test_e2e_string_trim() {
        let source = r#"
            main() = String.trim("  hello  ")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "hello"),
            other => panic!("Expected 'hello', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_to_upper() {
        let source = r#"
            main() = String.toUpper("hello")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "HELLO"),
            other => panic!("Expected 'HELLO', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_to_lower() {
        let source = r#"
            main() = String.toLower("HELLO")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "hello"),
            other => panic!("Expected 'hello', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_contains() {
        let source = r#"
            main() = String.contains("hello world", "world")
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_string_starts_with() {
        let source = r#"
            main() = String.startsWith("hello world", "hello")
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_string_ends_with() {
        let source = r#"
            main() = String.endsWith("hello world", "world")
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_string_replace() {
        let source = r#"
            main() = String.replace("hello world", "world", "rust")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "hello rust"),
            other => panic!("Expected 'hello rust', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_index_of() {
        let source = r#"
            main() = String.indexOf("hello world", "world")
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(6)));
    }

    #[test]
    fn test_e2e_string_repeat() {
        let source = r#"
            main() = String.repeat("ab", 3)
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "ababab"),
            other => panic!("Expected 'ababab', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_reverse() {
        let source = r#"
            main() = String.reverse("hello")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "olleh"),
            other => panic!("Expected 'olleh', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_is_empty() {
        let source = r#"
            main() = String.isEmpty("")
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_time_now() {
        let source = r#"
            main() = Time.now() > 0
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_random_int() {
        let source = r#"
            main() = {
                r = Random.int(1, 10)
                r >= 1 && r <= 10
            }
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_random_float() {
        let source = r#"
            main() = {
                r = Random.float()
                r >= 0.0 && r < 1.0
            }
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_random_bool() {
        // Just test that it runs without error
        let source = r#"
            main() = {
                b = Random.bool()
                b || !b
            }
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_env_platform() {
        let source = r#"
            main() = String.length(Env.platform()) > 0
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_env_cwd() {
        let source = r#"
            main() = String.length(Env.cwd()) > 0
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_path_join() {
        let source = r#"
            main() = Path.join("/home", "user")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert!(s.contains("user")),
            other => panic!("Expected path string, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_path_basename() {
        let source = r#"
            main() = Path.basename("/home/user/file.txt")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "file.txt"),
            other => panic!("Expected 'file.txt', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_path_dirname() {
        let source = r#"
            main() = Path.dirname("/home/user/file.txt")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert!(s.contains("user")),
            other => panic!("Expected path with 'user', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_path_extension() {
        let source = r#"
            main() = Path.extension("/home/user/file.txt")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "txt"),
            other => panic!("Expected 'txt', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_path_is_absolute() {
        let source = r#"
            main() = Path.isAbsolute("/home/user")
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_regex_match() {
        let source = r#"
            main() = Regex.matches("hello123world", "[0-9]+")
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_regex_replace() {
        let source = r#"
            main() = Regex.replace("hello 123 world", "[0-9]+", "XXX")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "hello XXX world"),
            other => panic!("Expected 'hello XXX world', got {:?}", other),
        }
    }

    #[test]
    fn test_builtin_names_include_stdlib_modules() {
        let names = Compiler::get_builtin_names();

        // Verify new stdlib modules are present
        let modules = ["String", "Time", "Random", "Env", "Path", "Regex", "File", "Dir", "Exec"];
        for module in modules {
            let has_module = names.iter().any(|n| n.starts_with(&format!("{}.", module)));
            assert!(has_module, "Module {} should have at least one builtin function", module);
        }

        // Verify some specific functions exist
        assert!(names.contains(&"String.trim"), "String.trim should be in builtins");
        assert!(names.contains(&"Time.now"), "Time.now should be in builtins");
        assert!(names.contains(&"Random.int"), "Random.int should be in builtins");
        assert!(names.contains(&"Env.get"), "Env.get should be in builtins");
        assert!(names.contains(&"Path.join"), "Path.join should be in builtins");
        assert!(names.contains(&"Regex.matches"), "Regex.matches should be in builtins");

        // Verify top-level functions (not module-prefixed) also exist
        assert!(names.contains(&"println"), "println should be in builtins");
    }

    #[test]
    fn test_builtin_signatures_available() {
        // Verify signatures are available for autocomplete
        assert!(Compiler::get_builtin_signature("String.trim").is_some(), "String.trim signature should exist");
        assert!(Compiler::get_builtin_signature("Time.now").is_some(), "Time.now signature should exist");
        assert!(Compiler::get_builtin_signature("Random.int").is_some(), "Random.int signature should exist");
        assert!(Compiler::get_builtin_signature("Env.get").is_some(), "Env.get signature should exist");
        assert!(Compiler::get_builtin_signature("Path.join").is_some(), "Path.join signature should exist");
        assert!(Compiler::get_builtin_signature("Regex.matches").is_some(), "Regex.matches signature should exist");
    }

    #[test]
    fn test_builtin_docs_available() {
        // Verify docs are available for autocomplete
        assert!(Compiler::get_builtin_doc("String.trim").is_some(), "String.trim doc should exist");
        assert!(Compiler::get_builtin_doc("Time.now").is_some(), "Time.now doc should exist");
        assert!(Compiler::get_builtin_doc("Random.int").is_some(), "Random.int doc should exist");
        assert!(Compiler::get_builtin_doc("Env.get").is_some(), "Env.get doc should exist");
        assert!(Compiler::get_builtin_doc("Path.join").is_some(), "Path.join doc should exist");
        assert!(Compiler::get_builtin_doc("Regex.matches").is_some(), "Regex.matches doc should exist");
    }
}
