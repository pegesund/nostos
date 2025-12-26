# Concurrency

Nostos embraces an Erlang-style actor model for concurrency. This means lightweight processes, message passing, and shared-nothing architecture for robust, fault-tolerant systems.

## Spawning Processes

The `spawn` function creates a new, independent process and returns its Process ID (PID). Each process has its own heap and stack.

```nostos
worker_task() = {
    println("Worker process started!")
    # ... do some work ...
}

main() = {
    pid = spawn { worker_task() }
    println("Spawned worker with PID: " ++ show(pid))
}
```

## Scaling: 100,000 Processes

Nostos processes are extremely lightweight. You can spawn 100,000 processes in under a second.

```nostos
type Message = Increase(Int) | Nothing

report_counter(parent, current_counter) = {
    parent <- Increase(current_counter)
}

main() = {
    me = self()
    num_workers = 100000
    var total = 0

    # Spawn all workers
    for i = 0 to num_workers {
        spawn { report_counter(me, i) }
    }
    println("All workers spawned!")

    # Collect results
    for i = 0 to num_workers {
        receive {
            Increase(value) -> { total = total + value }
            Nothing -> ()
        }
    }

    println("Total: " ++ show(total))  # 4999950000
}
```

### Why So Fast?

Unlike OS threads (which have ~1MB stack each), Nostos processes are **green threads** scheduled by the runtime. They start with tiny stacks that grow as needed, and the runtime multiplexes them across CPU cores using work-stealing.

## Message Passing

Processes communicate by sending and receiving messages. The `<-` operator sends a message to a PID, and `receive` blocks until a matching message arrives.

```nostos
responder(parent_pid) = {
    receive {
        msg -> {
            println("Responder received: " ++ show(msg))
            parent_pid <- "Hello from responder!"
        }
    }
}

main() = {
    me = self()
    pid = spawn { responder(me) }

    pid <- "Ping!"

    received_response = receive { response -> response }
    println("Main received: " ++ show(received_response))
}
```

## Pattern Matching & Timeouts

`receive` can use pattern matching to select specific messages and can include a timeout clause.

```nostos
alarm_clock(duration_ms) = {
    receive {
        _ -> println("Alarm dismissed early!")
        after duration_ms -> println("Wake up!")
    }
}

main() = {
    pid = spawn { alarm_clock(2000) }
    sleep(1000)
}
```

## MVars: Shared Mutable State

While message passing is preferred, sometimes you need shared state. **MVars** provide thread-safe shared state with automatic locking.

```nostos
# Declare a module-level mvar
mvar counter: Int = 0

increment() = {
    counter = counter + 1
    counter
}

decrement() = {
    counter = counter - 1
    counter
}

main() = {
    println(increment())  # 1
    println(increment())  # 2
    println(decrement())  # 1
}
```

### Concurrent Access

MVars are safe to use from multiple processes.

```nostos
mvar total: Int = 0

add_to_total(n) = {
    total = total + n
    total
}

worker(parent, amount) = {
    add_to_total(amount)
    parent <- "done"
}

main() = {
    me = self()

    for i = 0 to 10 {
        spawn { worker(me, 10) }
    }

    for i = 0 to 10 {
        receive { "done" -> () }
    }

    println("Total: " ++ show(total))  # Always 100
}
```

### MVar Best Practices

- Keep mvar operations short to minimize lock contention
- Prefer message passing for complex coordination
- Use mvars for simple counters, flags, or shared config
- MVars work with any type: `mvar cache: Map[String, Int] = %{}`
