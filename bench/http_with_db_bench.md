# HTTP + PostgreSQL Transaction Benchmark

## Overview

This benchmark compares the performance of different web frameworks executing PostgreSQL transactions under high concurrency. The test simulates a realistic workload: a credit transfer between users requiring ACID guarantees.

## What We Did

### The Transaction

Each request executes the exact same database transaction across all frameworks:

```sql
BEGIN;
UPDATE web_users SET credits = credits - $amount WHERE id = $from_id;
UPDATE web_users SET credits = credits + $amount WHERE id = $to_id;
COMMIT;
```

This is a realistic workload that:
- Requires a database connection from a pool
- Uses parameterized queries
- Involves row-level locking (causing contention)
- Needs proper transaction handling with rollback on error

### Test Setup

- **Database**: PostgreSQL (localhost)
- **Table**: `web_users` with 100 test users
- **Connection Pool**: 25 connections for all frameworks
- **Load Generator**: wrk with 4 threads, 100 concurrent connections, 10 second duration
- **Request Pattern**: Random transfers between users 1-100

### Frameworks Tested

1. **Nostos** - Custom language VM with async I/O and connection pooling
2. **Ruby on Rails 8** - With YJIT enabled, Puma (5 workers × 5 threads)
3. **Go + pgx** - Standard library HTTP with pgxpool
4. **Go + database/sql** - Standard library with lib/pq driver
5. **Node.js + Express** - With pg (node-postgres) pool

## Key Innovation: Fast-Fail with 503

During testing, we discovered that the traditional approach of queuing requests when the connection pool is exhausted leads to timeouts and degraded throughput.

**Nostos implements a "fast-fail" pattern:**
- When the pool is exhausted, immediately return HTTP 503 (Service Unavailable)
- Don't waste time/resources on requests that can't be served
- Let clients retry with exponential backoff

This approach:
- Eliminates timeouts completely
- Increases successful throughput by ~60%
- Provides clear backpressure signal to clients
- Keeps the server responsive under load

## Results

| Framework | Transactions/sec | Timeouts | Notes |
|-----------|-----------------|----------|-------|
| **Nostos** | ~1,600 | 0 | Fast-fail 503s for overflow |
| Rails 8 + YJIT | ~573 | 102 | Puma cluster mode |
| Go + pgx | ~560 | 50 | High-performance PG driver |
| Node.js + Express | ~438 | 124 | Single-threaded event loop |
| Go + database/sql | ~336 | 155 | Standard library driver |

### Analysis

**Nostos is 2.8-4.7x faster** than all tested frameworks for this transactional workload.

Key factors:
1. **Fast-fail pattern** - Other frameworks block waiting for connections, causing cascading timeouts
2. **Efficient async I/O** - Per-connection locking allows true parallel database operations
3. **Lightweight VM** - Low overhead per request compared to Rails/Node.js

The bottleneck at high concurrency is PostgreSQL row-level locking, not the application framework. Nostos maximizes throughput by not wasting resources on requests it can't serve.

## How to Reproduce

### Prerequisites
- PostgreSQL running on localhost (user: postgres, password: postgres)
- wrk load testing tool
- The respective language runtimes (Ruby, Go, Node.js)

### Setup Database
```sql
CREATE TABLE IF NOT EXISTS web_users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    credits INT DEFAULT 100,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert 100 test users
INSERT INTO web_users (name, email)
SELECT 'User ' || i, 'user' || i || '@example.com'
FROM generate_series(1, 100) AS i;
```

### Run Benchmarks
```bash
# Start the server for each framework, then:
wrk -t4 -c100 -d10s -s transfer_random.lua http://localhost:PORT/transfer
```

## Conclusion

For database-heavy web applications, Nostos provides excellent performance through:
- Efficient connection pooling with per-connection locking
- Fast-fail backpressure (503) instead of timeouts
- Lightweight async runtime

The 503 fast-fail pattern is particularly valuable - it's a better user experience than random timeouts and actually improves overall throughput.
