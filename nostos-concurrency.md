# Nostos Concurrency

Actor-based concurrency with lightweight processes, message passing, and fault tolerance.

*Inspired by Erlang/Elixir, but fully typed.*

---

## Core Concepts

- **Processes** — Lightweight, isolated units of execution (not OS threads)
- **Messages** — Typed, immutable data sent between processes
- **Mailboxes** — Per-process queue of incoming messages
- **Selective receive** — Pattern matching on message types
- **Links & monitors** — Failure propagation and notification
- **Supervisors** — Automatic restart strategies

---

## Processes

### Spawning

```
# Spawn with lambda
pid = spawn(() => 
  println("Hello from process!")
)

# Spawn with function
pid = spawn(worker)

# Spawn with function and args
pid = spawn(worker, [arg1, arg2])

# Spawn linked (dies together)
pid = spawn_link(worker)

# Spawn monitored (get notified on death)
{pid, ref} = spawn_monitor(worker)
```

### Process Identity

```
self()                         # => current process pid
alive?(pid)                    # => true/false
processes()                    # => list of all pids
```

---

## Message Passing

### Defining Message Types

```
# Define message protocol as types
type CounterMsg =
  | Inc(Pid)
  | Dec(Pid)
  | Get(Pid)
  | Set(Int, Pid)
  | Stop

type CounterReply =
  | Value(Int)
  | Ok
```

### Send

Non-blocking, fire-and-forget:

```
pid <- Inc(self())
pid <- Get(self())
pid <- Stop
```

### Receive

Blocking, with pattern matching on types:

```
receive
  Inc(sender) -> handleInc(sender)
  Dec(sender) -> handleDec(sender)
  Get(sender) -> handleGet(sender)
  Stop -> cleanup()
end
```

### Receive with Timeout

```
receive
  Value(n) -> handle(n)
  Ok -> done()
after 5000 ->
  panic(Timeout)
end

# Timeout 0 = check mailbox without blocking
receive
  msg -> handle(msg)
after 0 ->
  noMessages()
end
```

### Selective Receive

Messages are matched in order. Non-matching messages stay in mailbox:

```
type Priority = Urgent(String) | Normal(String)

# Only handle urgent messages first
receive
  Urgent(msg) -> handleUrgent(msg)
end

# Then handle normal messages
receive
  Normal(msg) -> handleNormal(msg)
end
```

### Pin Operator in Receive

Match against existing values:

```
type Request = Get(Ref, Pid)
type Response = Reply(Ref, Data)

ref = makeRef()
pid <- Get(ref, self())

receive
  Reply(^ref, data) -> data    # must match our ref
after 5000 ->
  panic(Timeout)
end
```

---

## Patterns

### Ping-Pong

```
type PingMsg = Ping(Pid) | Stop
type PongMsg = Pong(Pid)

pong() =
  receive
    Ping(sender) ->
      sender <- Pong(self())
      pong()
    Stop ->
      ()
  end

ping(pong_pid, 0) =
  pong_pid <- Stop

ping(pong_pid, n) =
  pong_pid <- Ping(self())
  receive
    Pong(_) -> ping(pong_pid, n - 1)
  end

main() =
  pong_pid = spawn(pong)
  ping(pong_pid, 5)
```

### Stateful Server

```
type CounterMsg =
  | Inc(Pid)
  | Dec(Pid)
  | Get(Pid)
  | Set(Int, Pid)
  | Stop

type CounterReply =
  | Value(Int)
  | Ok

counter() = loop(0)

loop(state) =
  receive
    Inc(sender) ->
      sender <- Value(state + 1)
      loop(state + 1)
      
    Dec(sender) ->
      sender <- Value(state - 1)
      loop(state - 1)
      
    Get(sender) ->
      sender <- Value(state)
      loop(state)
      
    Set(value, sender) ->
      sender <- Ok
      loop(value)
      
    Stop ->
      state
  end

# Usage
c = spawn(counter)
c <- Inc(self())
c <- Inc(self())
c <- Get(self())
receive Value(n) -> n end      # => 2
```

### Synchronous Call Helper

```
# Generic request/response wrapper
type Call[M] = Call(M, Ref, Pid)
type Reply[R] = Reply(Ref, R)

call(pid, msg) =
  ref = makeRef()
  pid <- Call(msg, ref, self())
  receive
    Reply(^ref, response) -> response
  after 5000 ->
    panic(Timeout)
  end

cast(pid, msg) =
  pid <- msg
  ()

# Server message types
type ServerMsg = GetValue | SetValue(Int) | Stop
type ServerReply = Value(Int) | Ok

# Server side
serverLoop(state) =
  receive
    Call(GetValue, ref, sender) ->
      sender <- Reply(ref, Value(state))
      serverLoop(state)
      
    Call(SetValue(value), ref, sender) ->
      sender <- Reply(ref, Ok)
      serverLoop(value)
      
    Stop ->
      state
  end

# Client side
value = call(server, GetValue)
call(server, SetValue(42))
cast(server, Stop)
```

---

## Named Processes

```
# Register a name (using strings)
register("counter", pid)

# Look up by name
pid = whereis("counter")       # => Some(pid) or None

# Send to named process
lookup("counter") <- Inc(self())

# Unregister
unregister("counter")

# List all names
registered()                   # => ["counter", "logger", ...]
```

---

## Links and Monitors

### Links (Bidirectional)

When linked processes die, both die:

```
# Link after spawn
pid = spawn(worker)
link(pid)

# Or spawn already linked
pid = spawn_link(worker)

# Unlink
unlink(pid)
```

### Monitors (Unidirectional)

Get notified when a process dies:

```
type MonitorMsg = Down(Ref, Pid, ExitReason)

ref = monitor(pid)

receive
  Down(^ref, pid, reason) ->
    println("Process died: " ++ reason.show())
end

# Stop monitoring
demonitor(ref)
```

### Trapping Exits

Convert exit signals to messages:

```
type ExitMsg = Exit(Pid, ExitReason)

Process.trapExit(true)

spawn_link(worker)

receive
  Exit(pid, reason) ->
    println("Linked process exited: " ++ reason.show())
    # Can restart or clean up
end
```

---

## Supervisors

Automatic restart of failed processes.

### Types

```
type RestartStrategy = OneForOne | OneForAll | RestForOne

type RestartOption = Always | Transient | Never

type ChildSpec = Child{
  id: String,
  start: () -> (),
  restart: RestartOption
}
```

### Basic Supervisor

```
children = [
  Child(id: "worker1", start: worker1, restart: Always),
  Child(id: "worker2", start: worker2, restart: Always),
  Child(id: "worker3", start: worker3, restart: Transient)
]

sup = Supervisor.start(children, OneForOne)
```

### Restart Strategies

| Strategy | Behavior |
|----------|----------|
| `OneForOne` | Only restart the failed child |
| `OneForAll` | Restart all children if one fails |
| `RestForOne` | Restart failed child and all started after it |

### Restart Options

| Option | Behavior |
|--------|----------|
| `Always` | Always restart (default) |
| `Transient` | Restart only on abnormal exit |
| `Never` | Never restart |

### Supervisor Trees

```
#        root_sup
#       /        \
#   worker_sup   logger
#   /    |    \
#  w1   w2    w3

type ChildType = Worker | Supervisor

root_children = [
  Child(id: "worker_sup", start: workerSupervisor, type: Supervisor),
  Child(id: "logger", start: logger, type: Worker)
]

root = Supervisor.start(root_children, OneForOne)
```

---

## Process Introspection

```
type ProcessStatus = Running | Waiting | Suspended

type ProcessInfo = {
  status: ProcessStatus,
  mailboxSize: Int,
  memory: Int,
  links: List[Pid],
  monitors: List[Ref]
}

# Process info
Process.info(pid)              # => ProcessInfo

# Mailbox inspection (debugging)
Process.messages(pid)          # => list of queued messages

# All processes
processes()                    # => [pid1, pid2, ...]
processes().length()           # => count

# Process.info on self
Process.info(self())
```

---

## Error Handling

### Exit Reasons

```
type ExitReason = Normal | Shutdown | Error(String)

# Normal exit
exit(Normal)

# Abnormal exit
exit(Error("something went wrong"))

# Shutdown
exit(Shutdown)

# Kill (cannot be trapped)
Process.kill(pid)
```

### Try in Processes

```
safeWorker() =
  try {
    riskyOperation()
  } catch
    Error(reason) ->
      println("Caught: " ++ reason)
      safeWorker()              # retry
  end
```

---

## Hot Code Reload

Update process code while running:

```
type ServerMsgV1 = Get(Pid) | Upgrade
type ServerMsgV2 = Get(Pid) | GetDouble(Pid) | Upgrade
type ServerReply = Value(Int)

# Version 1
serverV1(state) =
  receive
    Get(sender) ->
      sender <- Value(state)
      serverV1(state)
    Upgrade ->
      serverV2(state)           # switch to new version
  end

# Version 2 (new feature)
serverV2(state) =
  receive
    Get(sender) ->
      sender <- Value(state)
      serverV2(state)
    GetDouble(sender) ->        # new!
      sender <- Value(state * 2)
      serverV2(state)
  end

# Trigger upgrade
server <- Upgrade
```

---

## Runtime

### Scheduler

```
type Priority = Low | Normal | High

# Yield explicitly (usually not needed)
Process.yield()

# Sleep
Process.sleep(1000)            # milliseconds

# Set process priority
Process.setPriority(High)
```

### System Limits

```
# Get/set process limit
System.processLimit()          # => 262144
System.setProcessLimit(500000)

# Memory per process
Process.setMaxHeapSize(1_000_000)
```

---

## VM Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Scheduler                          │
│                                                         │
│   Run Queue:  [P1] -> [P3] -> [P7] -> [P2] -> ...      │
│   Wait Queue: [P4 (receive)] [P5 (sleep)] ...          │
│                                                         │
│   - Preemptive (reduction counting)                    │
│   - Work stealing across cores                         │
│   - Fair scheduling                                    │
└─────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │ Process  │    │ Process  │    │ Process  │
   │          │    │          │    │          │
   │ Mailbox  │    │ Mailbox  │    │ Mailbox  │
   │ Stack    │    │ Stack    │    │ Stack    │
   │ Heap     │    │ Heap     │    │ Heap     │
   │ Links    │    │ Links    │    │ Links    │
   └──────────┘    └──────────┘    └──────────┘
        │                │                │
        ▼                ▼                ▼
   Independent GC   Independent GC   Independent GC
```

### Design Points

| Aspect | Choice |
|--------|--------|
| Processes | Green threads, not OS threads |
| Scheduling | Preemptive, reduction-based |
| Memory | Per-process heap, isolated GC |
| Messages | Copied between processes (no sharing) |
| Scaling | Work-stealing across CPU cores |
| Failure | Let it crash + supervision |

---

## Summary

```
# Define message types
type Msg = Request(Pid) | Response(Int) | Stop

# Spawn
pid = spawn(fn)
pid = spawn_link(fn)
(pid, ref) = spawn_monitor(fn)

# Send
pid <- Request(self())

# Receive
receive
  Response(n) -> handle(n)
  Stop -> cleanup()
after timeout ->
  handleTimeout()
end

# Identity
self()
alive?(pid)

# Naming
register("name", pid)
whereis("name")
lookup("name") <- msg

# Linking
link(pid)
unlink(pid)
ref = monitor(pid)
demonitor(ref)

# Supervision
Supervisor.start(children, OneForOne)

# Introspection
Process.info(pid)
processes()
```

The philosophy: **let it crash**, supervise and restart. Build robust systems from processes that can fail independently.
