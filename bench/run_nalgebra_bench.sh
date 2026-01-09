#!/bin/bash
# Nalgebra benchmark: Rust vs Nostos vs NumPy

set -e

export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
NOSTOS_BIN="${PROJECT_ROOT}/target/release/nostos"
RUST_BENCH="${SCRIPT_DIR}/nalgebra_rust/target/release/nalgebra_bench"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Nalgebra Benchmark${NC}"
echo -e "${BLUE}  Rust vs Nostos vs NumPy${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check/build nostos
if [ ! -f "$NOSTOS_BIN" ]; then
    echo -e "${YELLOW}Building Nostos in release mode...${NC}"
    cd "$PROJECT_ROOT"
    cargo build --release --quiet
    cd "$SCRIPT_DIR"
fi

# Check/build Rust benchmark
if [ ! -f "$RUST_BENCH" ]; then
    echo -e "${YELLOW}Building Rust nalgebra benchmark...${NC}"
    cd "${SCRIPT_DIR}/nalgebra_rust"
    cargo build --release --quiet
    cd "$SCRIPT_DIR"
fi

# Check Python venv
if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv "${SCRIPT_DIR}/venv"
    source "${SCRIPT_DIR}/venv/bin/activate"
    pip install numpy --quiet
else
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

# Verify numpy is installed
python3 -c "import numpy; print(f'Using NumPy {numpy.__version__}')"

echo ""

# Function to run benchmark and measure time
run_benchmark() {
    local name=$1
    local cmd=$2
    local runs=${3:-3}

    local best_time=999999
    local result=""

    for i in $(seq 1 $runs); do
        local start=$(date +%s.%N)
        # Capture output, filter for numeric result (checksum)
        result=$($cmd 2>/dev/null | grep -E '^[0-9]+\.[0-9]+$' | head -1)
        local end=$(date +%s.%N)
        local elapsed=$(echo "$end - $start" | bc)

        if (( $(echo "$elapsed < $best_time" | bc -l) )); then
            best_time=$elapsed
        fi
    done

    printf "  %-15s %8.3fs  (result: %s)\n" "$name:" "$best_time" "$result"
    echo "$best_time" > /tmp/bench_time_$$
}

echo -e "${GREEN}Running benchmarks (3 runs each, best time shown)...${NC}"
echo ""

# Run Rust nalgebra benchmark
echo -e "${YELLOW}Rust nalgebra (native):${NC}"
run_benchmark "rust" "${RUST_BENCH}"
rust_time=$(cat /tmp/bench_time_$$)

echo ""

# Run NumPy benchmark
echo -e "${YELLOW}NumPy (Python + BLAS):${NC}"
run_benchmark "numpy" "python3 ${SCRIPT_DIR}/nalgebra_numpy.py"
numpy_time=$(cat /tmp/bench_time_$$)

echo ""

# Run Nostos nalgebra benchmark
echo -e "${YELLOW}Nostos nalgebra:${NC}"
cd "$PROJECT_ROOT"
run_benchmark "nostos" "${NOSTOS_BIN} examples/nalgebra-project/"
nostos_time=$(cat /tmp/bench_time_$$)

rm -f /tmp/bench_time_$$

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

printf "  %-15s %8.3fs\n" "Rust:" "$rust_time"
printf "  %-15s %8.3fs\n" "NumPy:" "$numpy_time"
printf "  %-15s %8.3fs\n" "Nostos:" "$nostos_time"
echo ""

# Calculate comparisons
echo -e "${YELLOW}Comparisons:${NC}"

# Nostos vs Rust (shows Nostos overhead)
nostos_vs_rust=$(echo "scale=2; $nostos_time / $rust_time" | bc)
echo -e "  Nostos vs Rust:  ${RED}${nostos_vs_rust}x slower${NC} (Nostos overhead)"

# Nostos vs NumPy
if (( $(echo "$nostos_time < $numpy_time" | bc -l) )); then
    ratio=$(echo "scale=2; $numpy_time / $nostos_time" | bc)
    echo -e "  Nostos vs NumPy: ${GREEN}${ratio}x faster${NC}"
else
    ratio=$(echo "scale=2; $nostos_time / $numpy_time" | bc)
    echo -e "  Nostos vs NumPy: ${RED}${ratio}x slower${NC}"
fi

# Rust vs NumPy
if (( $(echo "$rust_time < $numpy_time" | bc -l) )); then
    ratio=$(echo "scale=2; $numpy_time / $rust_time" | bc)
    echo -e "  Rust vs NumPy:   ${GREEN}${ratio}x faster${NC}"
else
    ratio=$(echo "scale=2; $rust_time / $numpy_time" | bc)
    echo -e "  Rust vs NumPy:   ${RED}${ratio}x slower${NC}"
fi

echo ""

# Notes
echo -e "${YELLOW}Notes:${NC}"
echo "  - Rust: Pure nalgebra, no runtime overhead"
echo "  - NumPy: Python + optimized BLAS/LAPACK"
echo "  - Nostos: ~0.5s startup + FFI overhead to nalgebra"
echo ""
