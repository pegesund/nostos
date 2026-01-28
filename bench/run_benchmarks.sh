#!/bin/bash
# Benchmark runner for Nostos vs Python vs Ruby vs Java
# Measures computation time only (excludes startup time)

set -e

# Force C locale for consistent decimal handling
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOSTOS_BIN="${SCRIPT_DIR}/../target/release/nostos"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Nostos Benchmark Suite${NC}"
echo -e "${BLUE}    (Computation time only)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if nostos binary exists, build if not
if [ ! -f "$NOSTOS_BIN" ]; then
    echo -e "${YELLOW}Building Nostos in release mode...${NC}"
    cd "${SCRIPT_DIR}/.."
    cargo build --release --quiet
    cd "$SCRIPT_DIR"
fi

# Compile Java benchmark if needed
if [ ! -f "${SCRIPT_DIR}/Fib.class" ] || [ "${SCRIPT_DIR}/Fib.java" -nt "${SCRIPT_DIR}/Fib.class" ]; then
    echo -e "${YELLOW}Compiling Java benchmark...${NC}"
    javac "${SCRIPT_DIR}/Fib.java"
fi

# Function to run a benchmark and capture internal time
# Each program outputs TIME:<milliseconds> and RESULT:<value>
run_benchmark() {
    local name=$1
    local cmd=$2

    # Run 3 times and take the best
    local best_time=999999999
    local result=""

    for i in 1 2 3; do
        local output=$($cmd 2>&1)
        local time_ms=$(echo "$output" | grep "^TIME:" | cut -d: -f2)
        result=$(echo "$output" | grep "^RESULT:" | cut -d: -f2)

        if [ -n "$time_ms" ] && [ "$time_ms" -lt "$best_time" ]; then
            best_time=$time_ms
        fi
    done

    # Convert ms to seconds for display
    local time_secs=$(echo "scale=3; $best_time / 1000" | bc)
    printf "  %-12s %8.3fs  (result: %s)\n" "$name:" "$time_secs" "$result"

    # Return the time in ms via a temp file
    echo "$best_time" > /tmp/bench_time_$$
}

# Run Fibonacci benchmark
echo -e "${GREEN}Fibonacci(40) - Recursive${NC}"
echo "  Running 3 iterations each, showing best time..."
echo "  (Startup time excluded - measures computation only)"
echo ""

run_benchmark "Nostos" "$NOSTOS_BIN ${SCRIPT_DIR}/fib.nos"
nostos_time=$(cat /tmp/bench_time_$$)

run_benchmark "Python" "python3 ${SCRIPT_DIR}/fib.py"
python_time=$(cat /tmp/bench_time_$$)

run_benchmark "Ruby" "ruby ${SCRIPT_DIR}/fib.rb"
ruby_time=$(cat /tmp/bench_time_$$)

run_benchmark "Java" "java -cp ${SCRIPT_DIR} Fib"
java_time=$(cat /tmp/bench_time_$$)

rm -f /tmp/bench_time_$$

echo ""
echo -e "${YELLOW}Comparison:${NC}"
echo "----------------------------------------"

# Calculate ratios (using ms values)
python_cmp=$(echo "$nostos_time < $python_time" | bc)
ruby_cmp=$(echo "$nostos_time < $ruby_time" | bc)
java_cmp=$(echo "$nostos_time < $java_time" | bc)

if [ "$python_cmp" -eq 1 ]; then
    inv_ratio=$(echo "scale=1; $python_time / $nostos_time" | bc)
    printf "  Nostos is ${GREEN}%sx faster${NC} than Python\n" "$inv_ratio"
else
    ratio=$(echo "scale=1; $nostos_time / $python_time" | bc)
    printf "  Nostos is ${RED}%sx slower${NC} than Python\n" "$ratio"
fi

if [ "$ruby_cmp" -eq 1 ]; then
    inv_ratio=$(echo "scale=1; $ruby_time / $nostos_time" | bc)
    printf "  Nostos is ${GREEN}%sx faster${NC} than Ruby\n" "$inv_ratio"
else
    ratio=$(echo "scale=1; $nostos_time / $ruby_time" | bc)
    printf "  Nostos is ${RED}%sx slower${NC} than Ruby\n" "$ratio"
fi

if [ "$java_cmp" -eq 1 ]; then
    inv_ratio=$(echo "scale=1; $java_time / $nostos_time" | bc)
    printf "  Nostos is ${GREEN}%sx faster${NC} than Java\n" "$inv_ratio"
else
    ratio=$(echo "scale=1; $nostos_time / $java_time" | bc)
    printf "  Nostos is ${RED}%sx slower${NC} than Java\n" "$ratio"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Summary${NC}"
echo -e "${BLUE}========================================${NC}"
nostos_secs=$(echo "scale=3; $nostos_time / 1000" | bc)
python_secs=$(echo "scale=3; $python_time / 1000" | bc)
ruby_secs=$(echo "scale=3; $ruby_time / 1000" | bc)
java_secs=$(echo "scale=3; $java_time / 1000" | bc)
printf "  %-12s %8.3fs\n" "Nostos:" "$nostos_secs"
printf "  %-12s %8.3fs\n" "Python:" "$python_secs"
printf "  %-12s %8.3fs\n" "Ruby:" "$ruby_secs"
printf "  %-12s %8.3fs\n" "Java:" "$java_secs"
echo ""
