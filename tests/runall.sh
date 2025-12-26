#!/bin/bash
# Run all .nos test files and verify expected results
# Usage: ./runall.sh [--verbose] [--stop-on-fail] [directory]
#
# The binary output format differs from test expectations:
# - Binary prints strings with quotes: "hello"
# - Tests expect without quotes: hello
# - Binary prints tuples as: (42, "hello")
# - Tests expect: (42, hello)
#
# This script normalizes output to match test expectations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$ROOT_DIR/target/release/nostos"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

VERBOSE=false
STOP_ON_FAIL=false
TEST_DIR="$SCRIPT_DIR"

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v) VERBOSE=true; shift ;;
        --stop-on-fail|-s) STOP_ON_FAIL=true; shift ;;
        *) TEST_DIR="$1"; shift ;;
    esac
done

if [ ! -f "$BINARY" ]; then
    echo "Building release binary..."
    (cd "$ROOT_DIR" && cargo build --release)
fi

# Normalize output to match test harness format:
# - Remove quotes from strings (but keep char quotes like 'a')
# - Handle nested structures (tuples, lists)
normalize_output() {
    local input="$1"
    # Remove quotes around strings, but carefully:
    # "hello" -> hello
    # (42, "hello") -> (42, hello)
    # ["a", "b"] -> [a, b]
    # But keep: 'a' (char literals)
    echo "$input" | sed -E '
        # Remove standalone string quotes
        s/^"(.*)"$/\1/
        # Remove string quotes inside structures (after comma-space or opening paren/bracket)
        s/\(([0-9]+), "([^"]*)"\)/(\1, \2)/g
        s/\(([0-9]+), "([^"]*)", ([a-z]+)\)/(\1, \2, \3)/g
        s/\("([^"]*)", ([0-9]+)\)/(\1, \2)/g
        s/\["([^"]*)"/[\1/g
        s/, "([^"]*)"/, \1/g
        s/"([^"]*)"\]$/\1]/g
    '
}

passed=0
failed=0
skipped=0
declare -a failures

while IFS= read -r -d '' file; do
    # Skip timeout tests (intentionally slow/hang)
    if [[ "$file" == *"/timeout/"* ]]; then
        ((skipped++))
        $VERBOSE && echo -e "${YELLOW}SKIP${NC}: $file (timeout test)"
        continue
    fi

    # Read expected value or error
    expect=$(grep "^# expect:" "$file" 2>/dev/null | head -1 | sed 's/^# expect: //')
    expect_error=$(grep "^# expect_error:" "$file" 2>/dev/null | head -1 | sed 's/^# expect_error: //')

    if [ -z "$expect" ] && [ -z "$expect_error" ]; then
        ((skipped++))
        $VERBOSE && echo -e "${YELLOW}SKIP${NC}: $file (no expect comment)"
        continue
    fi

    # Run the test with timeout
    output=$(timeout 10 "$BINARY" "$file" 2>&1) || true

    if [ -n "$expect_error" ]; then
        # Test expects an error - check if error message contains expected string
        if echo "$output" | grep -qF "$expect_error"; then
            ((passed++))
            $VERBOSE && echo -e "${GREEN}PASS${NC}: $file"
        else
            ((failed++))
            msg="$file: Expected error containing '$expect_error', got: $output"
            failures+=("$msg")
            echo -e "${RED}FAIL${NC}: $msg"
            $STOP_ON_FAIL && exit 1
        fi
    else
        # Test expects a specific value - use last line only (tests may print debug info)
        actual=$(echo "$output" | tail -1 | tr -d '\n')
        normalized=$(normalize_output "$actual")

        if [ "$normalized" = "$expect" ]; then
            ((passed++))
            $VERBOSE && echo -e "${GREEN}PASS${NC}: $file"
        else
            ((failed++))
            msg="$file: Expected '$expect', got '$actual' (normalized: '$normalized')"
            failures+=("$msg")
            echo -e "${RED}FAIL${NC}: $msg"
            $STOP_ON_FAIL && exit 1
        fi
    fi
done < <(find "$TEST_DIR" -name "*.nos" -print0 | sort -z)

echo ""
echo "========================================"
echo -e "Results: ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}, ${YELLOW}$skipped skipped${NC}"
echo "========================================"

if [ $failed -gt 0 ]; then
    echo ""
    echo "Failures:"
    for f in "${failures[@]}"; do
        echo "  - $f"
    done
    exit 1
fi

exit 0
