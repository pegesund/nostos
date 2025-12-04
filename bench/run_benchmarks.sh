#!/bin/bash
# Benchmark runner for Nostos vs Python vs Ruby

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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Nostos Benchmark Suite${NC}"
echo -e "${BLUE}========================================${NC}"
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

    printf "  %-16s %8.3fs  (result: %s)\n" "$name:" "$best_time" "$result"
    # Return the time via a temp file to avoid subshell issues
    echo "$best_time" > /tmp/bench_time_$$
}

# Run Fibonacci benchmark
echo -e "${GREEN}Fibonacci(35) - Recursive${NC}"
echo "  Running 3 iterations each, showing best time..."
echo ""

run_benchmark "Nostos (VM)" "$NOSTOS_BIN ${SCRIPT_DIR}/fib.nos"
nostos_vm_time=$(cat /tmp/bench_time_$$)

run_benchmark "Nostos (Runtime)" "$NOSTOS_BIN --runtime ${SCRIPT_DIR}/fib.nos"
nostos_rt_time=$(cat /tmp/bench_time_$$)

run_benchmark "Python" "python3 ${SCRIPT_DIR}/fib.py"
python_time=$(cat /tmp/bench_time_$$)

run_benchmark "Ruby" "ruby ${SCRIPT_DIR}/fib.rb"
ruby_time=$(cat /tmp/bench_time_$$)

rm -f /tmp/bench_time_$$

echo ""
echo -e "${YELLOW}Comparison (VM mode):${NC}"
echo "----------------------------------------"

# Calculate ratios for VM
python_ratio_vm=$(echo "scale=1; $nostos_vm_time / $python_time" | bc)
ruby_ratio_vm=$(echo "scale=1; $nostos_vm_time / $ruby_time" | bc)

python_cmp_vm=$(echo "$nostos_vm_time < $python_time" | bc)
ruby_cmp_vm=$(echo "$nostos_vm_time < $ruby_time" | bc)

if [ "$python_cmp_vm" -eq 1 ]; then
    inv_ratio=$(echo "scale=1; $python_time / $nostos_vm_time" | bc)
    printf "  Nostos VM is ${GREEN}%sx faster${NC} than Python\n" "$inv_ratio"
else
    printf "  Nostos VM is ${RED}%sx slower${NC} than Python\n" "$python_ratio_vm"
fi

if [ "$ruby_cmp_vm" -eq 1 ]; then
    inv_ratio=$(echo "scale=1; $ruby_time / $nostos_vm_time" | bc)
    printf "  Nostos VM is ${GREEN}%sx faster${NC} than Ruby\n" "$inv_ratio"
else
    printf "  Nostos VM is ${RED}%sx slower${NC} than Ruby\n" "$ruby_ratio_vm"
fi

echo ""
echo -e "${CYAN}Comparison (Runtime mode):${NC}"
echo "----------------------------------------"

# Calculate ratios for Runtime
python_ratio_rt=$(echo "scale=1; $nostos_rt_time / $python_time" | bc)
ruby_ratio_rt=$(echo "scale=1; $nostos_rt_time / $ruby_time" | bc)

python_cmp_rt=$(echo "$nostos_rt_time < $python_time" | bc)
ruby_cmp_rt=$(echo "$nostos_rt_time < $ruby_time" | bc)

if [ "$python_cmp_rt" -eq 1 ]; then
    inv_ratio=$(echo "scale=1; $python_time / $nostos_rt_time" | bc)
    printf "  Nostos Runtime is ${GREEN}%sx faster${NC} than Python\n" "$inv_ratio"
else
    printf "  Nostos Runtime is ${RED}%sx slower${NC} than Python\n" "$python_ratio_rt"
fi

if [ "$ruby_cmp_rt" -eq 1 ]; then
    inv_ratio=$(echo "scale=1; $ruby_time / $nostos_rt_time" | bc)
    printf "  Nostos Runtime is ${GREEN}%sx faster${NC} than Ruby\n" "$inv_ratio"
else
    printf "  Nostos Runtime is ${RED}%sx slower${NC} than Ruby\n" "$ruby_ratio_rt"
fi

# Compare VM vs Runtime
echo ""
echo -e "${YELLOW}VM vs Runtime:${NC}"
echo "----------------------------------------"
vm_rt_cmp=$(echo "$nostos_vm_time < $nostos_rt_time" | bc)
if [ "$vm_rt_cmp" -eq 1 ]; then
    ratio=$(echo "scale=1; $nostos_rt_time / $nostos_vm_time" | bc)
    printf "  VM is ${GREEN}%sx faster${NC} than Runtime\n" "$ratio"
else
    ratio=$(echo "scale=1; $nostos_vm_time / $nostos_rt_time" | bc)
    printf "  Runtime is ${GREEN}%sx faster${NC} than VM\n" "$ratio"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Summary${NC}"
echo -e "${BLUE}========================================${NC}"
printf "  %-16s %8.3fs\n" "Nostos (VM):" "$nostos_vm_time"
printf "  %-16s %8.3fs\n" "Nostos (Runtime):" "$nostos_rt_time"
printf "  %-16s %8.3fs\n" "Python:" "$python_time"
printf "  %-16s %8.3fs\n" "Ruby:" "$ruby_time"
echo ""
