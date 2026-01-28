#!/usr/bin/env ruby
# Fibonacci benchmark

def fib(n)
  if n <= 1
    n
  else
    fib(n - 1) + fib(n - 2)
  end
end

start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
result = fib(40)
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start

puts "TIME:#{(elapsed * 1000).to_i}"
puts "RESULT:#{result}"
