#!/bin/bash
# Benchmark runner for Nostos interpreter performance testing
# Saves results with timestamps to track improvements over time

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOSTOS="$SCRIPT_DIR/../target/release/nostos"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/bench_$TIMESTAMP.txt"

# Create results directory if needed
mkdir -p "$RESULTS_DIR"

# Check if nostos binary exists
if [ ! -f "$NOSTOS" ]; then
    echo "Error: nostos binary not found. Run 'cargo build --release' first."
    exit 1
fi

# Optional: description for this run
DESC="${1:-No description}"

echo "========================================"
echo "Nostos Benchmark Suite"
echo "========================================"
echo "Date: $(date)"
echo "Description: $DESC"
echo "Binary: $NOSTOS"
echo ""

# Write header to result file
{
    echo "# Nostos Benchmark Results"
    echo "# Date: $(date)"
    echo "# Description: $DESC"
    echo "# Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo ""
    echo "# Format: benchmark_name | real_time_ms | result"
    echo ""
} > "$RESULT_FILE"

# Function to run a single benchmark
run_bench() {
    local name="$1"
    local file="$2"

    # Run 3 times and take the median
    local times=()
    for i in 1 2 3; do
        # Use time command and capture real time
        local start=$(date +%s%N)
        local output=$(timeout 60 "$NOSTOS" "$file" 2>&1)
        local end=$(date +%s%N)
        local elapsed=$(( (end - start) / 1000000 ))  # Convert to ms
        times+=($elapsed)
    done

    # Sort and get median
    IFS=$'\n' sorted=($(sort -n <<<"${times[*]}")); unset IFS
    local median=${sorted[1]}

    # Get the result value (last line of output)
    local result=$(echo "$output" | tail -1)

    printf "%-35s %8d ms    result: %s\n" "$name" "$median" "$result"
    echo "$name | $median | $result" >> "$RESULT_FILE"
}

echo "Running benchmarks (3 iterations each, taking median)..."
echo ""
printf "%-35s %11s    %s\n" "Benchmark" "Time" "Result"
echo "------------------------------------------------------------------------"

# Run all benchmarks
for f in "$SCRIPT_DIR"/*.nos; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .nos)
        run_bench "$name" "$f"
    fi
done

echo "------------------------------------------------------------------------"
echo ""
echo "Results saved to: $RESULT_FILE"
echo ""

# Show comparison with previous run if exists
PREV_FILE=$(ls -t "$RESULTS_DIR"/bench_*.txt 2>/dev/null | sed -n '2p')
if [ -n "$PREV_FILE" ] && [ -f "$PREV_FILE" ]; then
    echo "Comparison with previous run ($(basename "$PREV_FILE")):"
    echo ""
    printf "%-35s %10s %10s %10s\n" "Benchmark" "Previous" "Current" "Change"
    echo "------------------------------------------------------------------------"

    while IFS='|' read -r name time result; do
        # Skip comments and empty lines
        [[ "$name" =~ ^#.*$ ]] && continue
        [[ -z "$name" ]] && continue

        name=$(echo "$name" | xargs)  # trim whitespace
        time=$(echo "$time" | xargs)

        # Find corresponding line in previous file
        prev_time=$(grep "^$name |" "$PREV_FILE" 2>/dev/null | cut -d'|' -f2 | xargs)

        if [ -n "$prev_time" ]; then
            if [ "$prev_time" -gt 0 ]; then
                change=$(echo "scale=1; (($time - $prev_time) * 100) / $prev_time" | bc 2>/dev/null)
                if [ -n "$change" ]; then
                    if (( $(echo "$change < 0" | bc -l) )); then
                        printf "%-35s %8d ms %8d ms   \033[32m%+.1f%%\033[0m\n" "$name" "$prev_time" "$time" "$change"
                    elif (( $(echo "$change > 0" | bc -l) )); then
                        printf "%-35s %8d ms %8d ms   \033[31m%+.1f%%\033[0m\n" "$name" "$prev_time" "$time" "$change"
                    else
                        printf "%-35s %8d ms %8d ms   %+.1f%%\n" "$name" "$prev_time" "$time" "$change"
                    fi
                fi
            fi
        fi
    done < "$RESULT_FILE"
    echo ""
fi

echo "Done!"
