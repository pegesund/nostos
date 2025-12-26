#!/bin/bash
# List sum benchmark - Nostos vs Haskell
# Tests tail-recursive sum with JIT optimization (50M elements)
# Note: Python is ~200x slower than Nostos JIT, so we skip it

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
echo -e "${BLUE}    List Sum Benchmark (50M elements)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Note: Python is ~200x slower than Nostos JIT, so we skip it${NC}"
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

echo "Running tail-recursive sum on 50M elements..."
echo ""

# Run Nostos
echo -e "${GREEN}Nostos (with JIT):${NC}"
nostos_start=$(date +%s.%N)
nostos_output=$($NOSTOS_BIN "${SCRIPT_DIR}/list_iterate.nos" 2>&1)
nostos_end=$(date +%s.%N)
nostos_time=$(echo "$nostos_end - $nostos_start" | bc)
nostos_len=$(echo "$nostos_output" | head -1)
nostos_sum=$(echo "$nostos_output" | head -2 | tail -1)
printf "  Time: %8.3fs  (len: %s, sum: %s)\n" "$nostos_time" "$nostos_len" "$nostos_sum"

# Run Haskell
echo -e "${GREEN}Haskell (GHC -O2):${NC}"
haskell_start=$(date +%s.%N)
haskell_output=$("$HASKELL_BIN" 2>&1)
haskell_end=$(date +%s.%N)
haskell_time=$(echo "$haskell_end - $haskell_start" | bc)
haskell_len=$(echo "$haskell_output" | head -1)
haskell_sum=$(echo "$haskell_output" | head -2 | tail -1)
printf "  Time: %8.3fs  (len: %s, sum: %s)\n" "$haskell_time" "$haskell_len" "$haskell_sum"

echo ""
echo -e "${YELLOW}Comparison:${NC}"
echo "----------------------------------------"

if (( $(echo "$nostos_time < $haskell_time" | bc -l) )); then
    ratio=$(echo "scale=1; $haskell_time / $nostos_time" | bc)
    echo -e "  Nostos is ${GREEN}${ratio}x faster${NC} than Haskell"
else
    ratio=$(echo "scale=1; $nostos_time / $haskell_time" | bc)
    echo -e "  Nostos is ${YELLOW}${ratio}x slower${NC} than Haskell"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Summary${NC}"
echo -e "${BLUE}========================================${NC}"
printf "  %-20s %8.3fs\n" "Nostos (JIT):" "$nostos_time"
printf "  %-20s %8.3fs\n" "Haskell (GHC -O2):" "$haskell_time"
printf "  %-20s %8s\n" "Python:" "~200x slower (skipped)"
echo ""
