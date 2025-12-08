# nostos_crystal_example.cr

# Define the channel type. Messages will either be a Tuple(Symbol, Int32)
# representing { :increase, value } or a Symbol representing :nothing.
alias MessageType = Tuple(Symbol, Int32) | Symbol

# Worker fiber: sends its counter value to the channel as a tagged message
def worker(channel : Channel(MessageType), counter : Int32)
  channel.send({:increase, counter}) # This sends a Tuple(Symbol, Int32)
end

# Main program
num_workers = 100_000

# Create a new channel with the specified Union type for messages
channel = Channel(MessageType).new

puts "Spawning #{num_workers} lightweight Crystal fibers..."

# Spawn workers
num_workers.times do |i|
  spawn do
    worker(channel, i)
  end
end

puts "All workers spawned! Collecting results..."

# Collect results from the channel and sum them
total_sum = 0_i64 # Int64 to prevent overflow

# Loop num_workers times to receive all messages
num_workers.times do
  message = channel.receive # Receive message from the channel
  
  # Use a case statement to pattern match on the received message
  case message
  when Tuple(Symbol, Int32) # Check if it's a tuple of Symbol and Int32
    if message[0] == :increase # Then check the symbol
      counter_value = message[1] # Extract the counter value
      total_sum += counter_value
    else
      puts "Received unexpected tuple message: #{message}"
    end
  when Symbol # Check if it's a plain symbol
    if message == :nothing
      nil # Do nothing, as per Nostos Nothing -> ()
    else
      puts "Received unexpected symbol message: #{message}"
    end
  else
    # This branch would catch any other types if the channel type allowed them,
    # or if an unexpected message was sent.
    puts "Received unexpected message type: #{message.class} with value #{message}"
  end
end

puts "All workers completed! Total sum of counters: #{total_sum}"