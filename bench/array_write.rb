#!/usr/bin/env ruby
# Array write benchmark - heavy writing to array
# Fills an array with computed values and sums them

def fill_array(arr, size)
  (0...size).each do |i|
    arr[i] = i * i + i * 3 + 7
  end
end

def sum_array(arr, size)
  total = 0
  (0...size).each do |i|
    total += arr[i]
  end
  total
end

def run_iteration(size)
  arr = Array.new(size, 0)
  fill_array(arr, size)
  sum_array(arr, size)
end

def benchmark(iterations, size)
  total = 0
  iterations.times do
    total += run_iteration(size)
  end
  total
end

size = 10000
iterations = 100
result = benchmark(iterations, size)
puts result
