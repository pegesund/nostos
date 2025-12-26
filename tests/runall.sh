#!/bin/bash
# Run all .nos test files and verify expected results
# Usage: ./runall.sh [--verbose] [--stop-on-fail] [--jobs N] [directory]
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
JOBS=8
TEST_DIR="$SCRIPT_DIR"

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v) VERBOSE=true; shift ;;
        --stop-on-fail|-s) STOP_ON_FAIL=true; shift ;;
        --jobs|-j) JOBS="$2"; shift 2 ;;
        *) TEST_DIR="$1"; shift ;;
    esac
done

if [ ! -f "$BINARY" ]; then
    echo "Building release binary..."
    (cd "$ROOT_DIR" && cargo build --release)
fi

# Remove all .nostos directories to avoid picking up definition files as tests
find "$TEST_DIR" -type d -name ".nostos" -exec rm -rf {} + 2>/dev/null || true

# Export variables and functions for parallel execution
export BINARY
export VERBOSE

# Function to run a single test - outputs: STATUS|file|message
run_single_test() {
    local file="$1"

    # Skip timeout tests (intentionally slow/hang)
    if [[ "$file" == *"/timeout/"* ]]; then
        echo "SKIP|$file|timeout test"
        return
    fi

    # Read expected value or error
    local expect=$(grep "^# expect:" "$file" 2>/dev/null | head -1 | sed 's/^# expect: //')
    local expect_error=$(grep "^# expect_error:" "$file" 2>/dev/null | head -1 | sed 's/^# expect_error: //')

    if [ -z "$expect" ] && [ -z "$expect_error" ]; then
        echo "SKIP|$file|no expect comment"
        return
    fi

    # Run the test with timeout
    local output=$(timeout 10 "$BINARY" "$file" 2>&1) || true

    if [ -n "$expect_error" ]; then
        # Test expects an error - check if error message contains expected string
        if echo "$output" | grep -qF "$expect_error"; then
            echo "PASS|$file|"
        else
            echo "FAIL|$file|Expected error containing '$expect_error', got: $output"
        fi
    else
        # Test expects a specific value - use last line only (tests may print debug info)
        local actual=$(echo "$output" | tail -1 | tr -d '\n')
        # Normalize output inline
        local normalized=$(echo "$actual" | sed -E '
            s/^"(.*)"$/\1/
            s/\(([0-9]+), "([^"]*)"\)/(\1, \2)/g
            s/\(([0-9]+), "([^"]*)", ([a-z]+)\)/(\1, \2, \3)/g
            s/\("([^"]*)", ([0-9]+)\)/(\1, \2)/g
            s/\["([^"]*)"/[\1/g
            s/, "([^"]*)"/, \1/g
            s/"([^"]*)"\]$/\1]/g
        ')

        if [ "$normalized" = "$expect" ]; then
            echo "PASS|$file|"
        else
            echo "FAIL|$file|Expected '$expect', got '$actual' (normalized: '$normalized')"
        fi
    fi
}
export -f run_single_test

# Find all test files and run in parallel
RESULTS_FILE=$(mktemp)
find "$TEST_DIR" -name "*.nos" -print0 | sort -z | \
    xargs -0 -P "$JOBS" -I {} bash -c 'run_single_test "$@"' _ {} > "$RESULTS_FILE"

# Parse results
passed=0
failed=0
skipped=0
declare -a failures

while IFS='|' read -r status file message; do
    case "$status" in
        PASS)
            ((passed++))
            $VERBOSE && echo -e "${GREEN}PASS${NC}: $file"
            ;;
        FAIL)
            ((failed++))
            failures+=("$file: $message")
            echo -e "${RED}FAIL${NC}: $file: $message"
            ;;
        SKIP)
            ((skipped++))
            $VERBOSE && echo -e "${YELLOW}SKIP${NC}: $file ($message)"
            ;;
    esac
done < "$RESULTS_FILE"

rm -f "$RESULTS_FILE"

echo ""
echo "========================================"
echo -e "Results: ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}, ${YELLOW}$skipped skipped${NC} (using $JOBS parallel jobs)"
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
