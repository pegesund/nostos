#!/bin/bash
# Verification script: compares Nostos and Python outputs
# Usage: ./verify.sh [test_number]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NOSTOS_DIR="$SCRIPT_DIR/nostos"
PYTHON_DIR="$SCRIPT_DIR/python"
NOSTOS_BIN="${SCRIPT_DIR}/../target/release/nostos"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

passed=0
failed=0
skipped=0

run_test() {
    local test_name="$1"
    local nos_file="$NOSTOS_DIR/${test_name}.nos"
    local py_file="$PYTHON_DIR/${test_name}.py"

    if [ ! -f "$nos_file" ]; then
        echo -e "${YELLOW}SKIP${NC} $test_name: No Nostos file"
        ((skipped++))
        return
    fi

    if [ ! -f "$py_file" ]; then
        echo -e "${YELLOW}SKIP${NC} $test_name: No Python file"
        ((skipped++))
        return
    fi

    # Run Nostos program
    nos_output=$(timeout 5 "$NOSTOS_BIN" "$nos_file" 2>&1)
    nos_exit=$?

    # Run Python program
    py_output=$(timeout 5 python3 "$py_file" 2>&1)
    py_exit=$?

    # Compare outputs (trim whitespace)
    nos_output_trimmed=$(echo "$nos_output" | tr -d '[:space:]')
    py_output_trimmed=$(echo "$py_output" | tr -d '[:space:]')

    if [ "$nos_exit" -ne 0 ] && [ "$nos_exit" -ne 124 ]; then
        echo -e "${RED}FAIL${NC} $test_name: Nostos error"
        echo "  Nostos output: $nos_output"
        ((failed++))
    elif [ "$py_exit" -ne 0 ] && [ "$py_exit" -ne 124 ]; then
        echo -e "${RED}FAIL${NC} $test_name: Python error"
        echo "  Python output: $py_output"
        ((failed++))
    elif [ "$nos_output_trimmed" != "$py_output_trimmed" ]; then
        echo -e "${RED}FAIL${NC} $test_name: Output mismatch"
        echo "  Nostos: $nos_output"
        echo "  Python: $py_output"
        ((failed++))
    else
        echo -e "${GREEN}PASS${NC} $test_name: $nos_output"
        ((passed++))
    fi
}

echo "========================================"
echo "  Nostos Language Verification Suite"
echo "========================================"
echo ""

if [ -n "$1" ]; then
    # Run specific test
    run_test "$1"
else
    # Run all tests
    for nos_file in "$NOSTOS_DIR"/*.nos; do
        test_name=$(basename "$nos_file" .nos)
        run_test "$test_name"
    done
fi

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
echo -e "  ${GREEN}Passed${NC}: $passed"
echo -e "  ${RED}Failed${NC}: $failed"
echo -e "  ${YELLOW}Skipped${NC}: $skipped"
echo "========================================"

if [ "$failed" -gt 0 ]; then
    exit 1
fi
exit 0
