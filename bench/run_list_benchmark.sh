#!/bin/bash
# List iteration benchmark - Nostos vs Python vs Haskell
# Tests head/tail pattern matching performance

set -e

export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOSTOS_BIN="${SCRIPT_DIR}/../target/release/nostos"
HASKELL_BIN="${SCRIPT_DIR}/list_iterate_hs"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    List Pattern Matching Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Build Nostos if needed
if [ ! -f "$NOSTOS_BIN" ]; then
    echo -e "${YELLOW}Building Nostos in release mode...${NC}"
    cd "${SCRIPT_DIR}/.."
    cargo build --release --quiet
    cd "$SCRIPT_DIR"
fi

# Build Haskell if needed
if [ ! -f "$HASKELL_BIN" ] || [ "${SCRIPT_DIR}/list_iterate.hs" -nt "$HASKELL_BIN" ]; then
    echo -e "${YELLOW}Compiling Haskell with -O2...${NC}"
    ghc -O2 -o "$HASKELL_BIN" "${SCRIPT_DIR}/list_iterate.hs" 2>/dev/null
fi

echo "Running list iteration (50K elements, 9 traversals)..."
echo ""

# Run Nostos
echo -e "${GREEN}Nostos:${NC}"
nostos_start=$(date +%s.%N)
nostos_result=$($NOSTOS_BIN "${SCRIPT_DIR}/list_iterate.nos" 2>&1 | head -1)
nostos_end=$(date +%s.%N)
nostos_time=$(echo "$nostos_end - $nostos_start" | bc)
printf "  Time: %8.3fs  (result: %s)\n" "$nostos_time" "$nostos_result"

# Run Haskell
echo -e "${GREEN}Haskell:${NC}"
haskell_start=$(date +%s.%N)
haskell_result=$("$HASKELL_BIN" 2>&1 | head -1)
haskell_end=$(date +%s.%N)
haskell_time=$(echo "$haskell_end - $haskell_start" | bc)
printf "  Time: %8.3fs  (result: %s)\n" "$haskell_time" "$haskell_result"

# Run Python
echo -e "${GREEN}Python:${NC}"
python_start=$(date +%s.%N)
python_result=$(python3 "${SCRIPT_DIR}/list_iterate.py" 2>&1 | head -1)
python_end=$(date +%s.%N)
python_time=$(echo "$python_end - $python_start" | bc)
printf "  Time: %8.3fs  (result: %s)\n" "$python_time" "$python_result"

echo ""
echo -e "${YELLOW}Comparison:${NC}"
echo "----------------------------------------"

haskell_ratio=$(echo "scale=1; $nostos_time / $haskell_time" | bc)
python_ratio=$(echo "scale=1; $python_time / $nostos_time" | bc)

if (( $(echo "$nostos_time < $haskell_time" | bc -l) )); then
    inv_ratio=$(echo "scale=1; $haskell_time / $nostos_time" | bc)
    echo -e "  Nostos is ${GREEN}${inv_ratio}x faster${NC} than Haskell"
else
    echo -e "  Nostos is ${YELLOW}${haskell_ratio}x slower${NC} than Haskell"
fi

echo -e "  Nostos is ${GREEN}${python_ratio}x faster${NC} than Python"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Summary${NC}"
echo -e "${BLUE}========================================${NC}"
printf "  %-12s %8.3fs\n" "Nostos:" "$nostos_time"
printf "  %-12s %8.3fs\n" "Haskell:" "$haskell_time"
printf "  %-12s %8.3fs\n" "Python:" "$python_time"
echo ""
