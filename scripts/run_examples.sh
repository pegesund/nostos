#!/bin/bash

# A script to run all .nos files in the examples/ directory and check for panics or errors.
# A successful run is one that exits with code 0.

# Force C locale for consistent decimal handling
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOSTOS_BIN="$(dirname "${SCRIPT_DIR}")/target/release/nostos"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Nostos Example Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if nostos binary exists, build if not
if [ ! -f "$NOSTOS_BIN" ]; then
    echo -e "Building Nostos in release mode..."
    cd "${SCRIPT_DIR}"
    cargo build --release
    cd "$SCRIPT_DIR"
    echo ""
fi

EXAMPLE_FILES=$(find "$(dirname "${SCRIPT_DIR}")/examples" -name "*.nos")
FAILURES=()
PASSED_COUNT=0
FAILED_COUNT=0

for file in $EXAMPLE_FILES; do
    # Skip REPL demos and TUI-only examples (interactive mode only)
    if [[ "$file" == *"repl_demo.nos"* ]] || [[ "$file" == *"inspect_demo.nos"* ]] || [[ "$file" == *"inspector_demo.nos"* ]]; then
        echo -e "Skipping ${file#$(dirname "${SCRIPT_DIR}")/}"
        continue
    fi

    if [[ "$file" == *"http_server.nos"* ]] || [[ "$file" == *"http_server_test.nos"* ]]; then
        printf "Running %-50s" "${file#${SCRIPT_DIR}/}"
        # --- Server Test Logic ---
        # These examples spawn their own clients that trigger shutdown,
        # so we just run them with a timeout and check exit code
        if timeout 15s "$NOSTOS_BIN" "$file" > /tmp/nostos_server_output.log 2>&1; then
            echo -e "[${GREEN}PASS${NC}]"
            ((PASSED_COUNT++))
        else
            EXIT_CODE=$?
            if [ "$EXIT_CODE" -eq 124 ]; then
                echo -e "[${RED}FAIL${NC}] (Timed out)"
                FAILURES+=("${file#${SCRIPT_DIR}/}:\nServer timed out (hung).\n")
            else
                echo -e "[${RED}FAIL${NC}] (Exit code $EXIT_CODE)"
                FAILURE_MSG=$(cat /tmp/nostos_server_output.log)
                FAILURES+=("${file#${SCRIPT_DIR}/}:\n$FAILURE_MSG\n")
            fi
            ((FAILED_COUNT++))
        fi
        rm -f /tmp/nostos_server_output.log
    else
        # --- Original execution logic for non-server examples ---
        printf "Running %-50s" "${file#${SCRIPT_DIR}/}"
        # Use longer timeout for examples with external HTTP requests
        if [[ "$file" == *"http_client.nos"* ]] || \
           [[ "$file" == *"http_error_handling.nos"* ]] || \
           [[ "$file" == *"async_io_demo.nos"* ]]; then
            EXAMPLE_TIMEOUT=60s
        else
            EXAMPLE_TIMEOUT=5s
        fi
        if timeout $EXAMPLE_TIMEOUT "$NOSTOS_BIN" "$file" > /tmp/nostos_example_run.log 2>&1; then
            echo -e "[${GREEN}PASS${NC}]"
            ((PASSED_COUNT++))
        else
            echo -e "[${RED}FAIL${NC}]"
            FAILURE_MSG=$(< /tmp/nostos_example_run.log)
            FAILURES+=("${file#${SCRIPT_DIR}/}:\n$FAILURE_MSG\n")
            ((FAILED_COUNT++))
        fi
    fi
done

rm -f /tmp/nostos_example_run.log

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Example Run Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}${PASSED_COUNT} passed${NC}, ${RED}${FAILED_COUNT} failed${NC}"
echo ""

if [ ${#FAILURES[@]} -ne 0 ]; then
    echo -e "${RED}Failures:${NC}"
    for failure in "${FAILURES[@]}"; do
        echo -e "----------------------------------------"
        echo -e "$failure"
    done
    exit 1
fi

exit 0
