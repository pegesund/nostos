#!/bin/bash
# Array write benchmark runner for Nostos vs Python vs Ruby

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
echo -e "${BLUE}    Array Write Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "  1000 iterations x 100000 element arrays"
echo "  Each iteration: create, fill with computed values, sum"
echo ""
echo -e "${YELLOW}Note:${NC} Nostos uses tail-call optimized recursion."
echo "      Python/Ruby use native for-loops (no TCO support)."
echo ""

# Check if nostos binary exists, build if not
if [ ! -f "$NOSTOS_BIN" ]; then
    echo -e "${YELLOW}Building Nostos in release mode...${NC}"
    cd "${SCRIPT_DIR}/.."
    cargo build --release --quiet
    cd "$SCRIPT_DIR"
fi

# Function to run a benchmark and capture time
run_benchmark() {
    local name=$1
    local cmd=$2

    # Run 3 times and take the best
    local best_time=999999
    local result=""

    for i in 1 2 3; do
        local start=$(date +%s.%N)
        result=$($cmd 2>&1 | head -1)
        local end=$(date +%s.%N)
        local elapsed=$(echo "$end - $start" | bc)

        if (( $(echo "$elapsed < $best_time" | bc -l) )); then
            best_time=$elapsed
        fi
    done

    printf "  %-12s %8.3fs  (result: %s)\n" "$name:" "$best_time" "$result"
    echo "$best_time" > /tmp/bench_time_$$
}

echo -e "${GREEN}Running benchmarks...${NC}"
echo "  Running 3 iterations each, showing best time..."
echo ""

run_benchmark "Nostos(rec)" "$NOSTOS_BIN ${SCRIPT_DIR}/array_write.nos"
nostos_time=$(cat /tmp/bench_time_$$)

run_benchmark "Nostos(loop)" "$NOSTOS_BIN ${SCRIPT_DIR}/array_write_loop.nos"
nostos_loop_time=$(cat /tmp/bench_time_$$)

run_benchmark "Python" "python3 ${SCRIPT_DIR}/array_write.py"
python_time=$(cat /tmp/bench_time_$$)

run_benchmark "Ruby" "ruby ${SCRIPT_DIR}/array_write.rb"
ruby_time=$(cat /tmp/bench_time_$$)

rm -f /tmp/bench_time_$$

echo ""
echo -e "${YELLOW}Comparison (loop JIT version):${NC}"
echo "----------------------------------------"

python_cmp=$(echo "$nostos_loop_time < $python_time" | bc)
ruby_cmp=$(echo "$nostos_loop_time < $ruby_time" | bc)
rec_cmp=$(echo "$nostos_loop_time < $nostos_time" | bc)

if [ "$python_cmp" -eq 1 ]; then
    inv_ratio=$(echo "scale=1; $python_time / $nostos_loop_time" | bc)
    printf "  Nostos(loop) is ${GREEN}%sx faster${NC} than Python\n" "$inv_ratio"
else
    ratio=$(echo "scale=1; $nostos_loop_time / $python_time" | bc)
    printf "  Nostos(loop) is ${RED}%sx slower${NC} than Python\n" "$ratio"
fi

if [ "$ruby_cmp" -eq 1 ]; then
    inv_ratio=$(echo "scale=1; $ruby_time / $nostos_loop_time" | bc)
    printf "  Nostos(loop) is ${GREEN}%sx faster${NC} than Ruby\n" "$inv_ratio"
else
    ratio=$(echo "scale=1; $nostos_loop_time / $ruby_time" | bc)
    printf "  Nostos(loop) is ${RED}%sx slower${NC} than Ruby\n" "$ratio"
fi

if [ "$rec_cmp" -eq 1 ]; then
    inv_ratio=$(echo "scale=1; $nostos_time / $nostos_loop_time" | bc)
    printf "  Nostos(loop) is ${GREEN}%sx faster${NC} than Nostos(rec)\n" "$inv_ratio"
else
    ratio=$(echo "scale=1; $nostos_loop_time / $nostos_time" | bc)
    printf "  Nostos(loop) is ${RED}%sx slower${NC} than Nostos(rec)\n" "$ratio"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Summary${NC}"
echo -e "${BLUE}========================================${NC}"
printf "  %-14s %8.3fs  (tail-recursive)\n" "Nostos(rec):" "$nostos_time"
printf "  %-14s %8.3fs  (loop + JIT)\n" "Nostos(loop):" "$nostos_loop_time"
printf "  %-14s %8.3fs  (for-loops)\n" "Python:" "$python_time"
printf "  %-14s %8.3fs  (for-loops)\n" "Ruby:" "$ruby_time"
echo ""
