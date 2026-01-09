# Async Runtime

Nostos is built on **Tokio**, the leading async runtime for Rust. This means *everything* is non-blocking - I/O, network calls, database queries, even mvar locks. Understanding this architecture helps you write efficient, scalable programs.

## The Big Picture

Traditional languages block threads when waiting for I/O. If you have 100 database queries in flight, you need 100 OS threads - expensive!

Nostos uses **async/await under the hood**. When a process waits for I/O, it *yields* to the runtime, allowing other processes to run. A handful of OS threads can serve millions of concurrent operations.

## Everything is Non-Blocking

These operations look synchronous but are actually async under the hood:

```nostos
# All these operations are non-blocking!
main() = {
    # File I/O - yields while waiting for disk
    content = File.read("data.txt")

    # HTTP request - yields while waiting for response
    response = Http.get("https://api.example.com")

    # Database query - yields while PostgreSQL processes
    conn = Pg.connect("host=localhost user=postgres")
    rows = Pg.query(conn, "SELECT * FROM users", [])

    # Sleep - yields, doesn't block the thread
    sleep(1000)

    # Message receive - yields until message arrives
    receive { msg -> println(msg) }
}
```

You don't write `await` keywords - the runtime handles it automatically. Your code reads like synchronous code but executes asynchronously.

## MVars: Async Locks

Even mvar locks are non-blocking. When a process tries to access an mvar that's locked by another process, it *yields* instead of blocking the thread.

```nostos
mvar counter: Int = 0

increment() = {
    # This lock is async!
    # If another process holds the lock, we yield - not block
    counter = counter + 1
    counter
}

main() = {
    me = self()

    # Spawn 1000 workers all trying to increment
    for i = 0 to 1000 {
        spawn {
            increment()
            me <- "done"
        }
    }

    # Wait for all - workers yield while waiting for lock
    for i = 0 to 1000 {
        receive { "done" -> () }
    }

    println("Final: " ++ show(counter))  # Always 1000
}
```

**Key insight:** Because locks are async, you can have thousands of processes contending for the same mvar without blocking OS threads. The runtime schedules them efficiently.

## Implications for Your Code

### 1. Spawn Freely

Since processes are cheap and I/O is non-blocking, don't hesitate to spawn:

```nostos
# Fetch 100 URLs in parallel - only uses a few OS threads!
fetch_all(urls) = {
    me = self()
    urls.each(url => spawn {
        result = Http.get(url)
        me <- (url, result)
    })

    # Collect results
    urls.map(_ => receive { (url, result) -> (url, result) })
}
```

### 2. Don't Fear Blocking Operations

In Nostos, "blocking" operations don't actually block. The runtime handles the async plumbing:

```nostos
# This looks like it blocks, but doesn't!
process_order(order_id) = {
    # Each step yields while waiting
    order = Pg.query(conn, "SELECT * FROM orders WHERE id = $1", [order_id])
    customer = Pg.query(conn, "SELECT * FROM customers WHERE id = $1", [order.customer_id])
    inventory = Http.post("https://inventory.api/check", order.items)

    # Other processes run while we wait for each step
    finalize(order, customer, inventory)
}
```

### 3. Long-Running Computation

CPU-intensive work *does* block a thread. For heavy computation, consider breaking it up or running in a separate process:

```nostos
# Long computation - runs on its own thread
heavy_work() = {
    # This blocks a thread but doesn't affect other processes
    result = expensive_calculation(1000000)
    result
}

main() = {
    me = self()

    # Spawn heavy work so it doesn't block main
    spawn { me <- heavy_work() }

    # Main can continue doing other things
    println("Computing in background...")

    receive { result -> println("Got: " ++ show(result)) }
}
```

## The Runtime Architecture

Here's what happens under the hood:

1. **Tokio Multi-Threaded Runtime** - A pool of OS threads (usually = CPU cores) handles all work
2. **Green Thread Scheduler** - Nostos processes (green threads) are multiplexed onto these threads
3. **Work Stealing** - Idle threads steal work from busy ones, keeping all cores utilized
4. **Async I/O** - All I/O uses epoll/kqueue/IOCP - never blocks threads

## Performance Benefits

- **Memory Efficiency** - 100,000 processes use a fraction of the memory that 100,000 OS threads would require
- **High Concurrency** - Handle millions of connections with a small thread pool
- **Low Latency** - No thread context switching overhead - just lightweight coroutine switches
- **CPU Utilization** - Work stealing keeps all cores busy without manual load balancing

## Summary

Nostos gives you the simplicity of synchronous code with the performance of async. You don't need to think about `async/await`, futures, or callback hell. Just write straightforward code and let the runtime handle the complexity.
