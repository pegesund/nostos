#!/bin/bash
# Benchmark runner WITHOUT JIT - tests pure interpreter performance
# This is where our optimizations will have the most impact

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOSTOS="$SCRIPT_DIR/../target/release/nostos"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/bench_nojit_$TIMESTAMP.txt"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$NOSTOS" ]; then
    echo "Error: nostos binary not found. Run 'cargo build --release' first."
    exit 1
fi

DESC="${1:-No description}"

echo "========================================"
echo "Nostos Benchmark Suite (NO JIT)"
echo "========================================"
echo "Date: $(date)"
echo "Description: $DESC"
echo ""

{
    echo "# Nostos Benchmark Results (NO JIT)"
    echo "# Date: $(date)"
    echo "# Description: $DESC"
    echo "# Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo ""
} > "$RESULT_FILE"

run_bench() {
    local name="$1"
    local file="$2"

    local times=()
    for i in 1 2 3; do
        local start=$(date +%s%N)
        local output=$(timeout 120 "$NOSTOS" --no-jit "$file" 2>&1)
        local end=$(date +%s%N)
        local elapsed=$(( (end - start) / 1000000 ))
        times+=($elapsed)
    done

    IFS=$'\n' sorted=($(sort -n <<<"${times[*]}")); unset IFS
    local median=${sorted[1]}
    local result=$(echo "$output" | tail -1)

    printf "%-35s %8d ms    result: %s\n" "$name" "$median" "$result"
    echo "$name | $median | $result" >> "$RESULT_FILE"
}

echo "Running benchmarks WITHOUT JIT (3 iterations each)..."
echo ""
printf "%-35s %11s    %s\n" "Benchmark" "Time" "Result"
echo "------------------------------------------------------------------------"

for f in "$SCRIPT_DIR"/*.nos; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .nos)
        run_bench "$name" "$f"
    fi
done

echo "------------------------------------------------------------------------"
echo ""
echo "Results saved to: $RESULT_FILE"

# Compare with previous no-jit run
PREV_FILE=$(ls -t "$RESULTS_DIR"/bench_nojit_*.txt 2>/dev/null | sed -n '2p')
if [ -n "$PREV_FILE" ] && [ -f "$PREV_FILE" ]; then
    echo ""
    echo "Comparison with previous NO-JIT run:"
    echo ""
    printf "%-35s %10s %10s %10s\n" "Benchmark" "Previous" "Current" "Change"
    echo "------------------------------------------------------------------------"

    while IFS='|' read -r name time result; do
        [[ "$name" =~ ^#.*$ ]] && continue
        [[ -z "$name" ]] && continue
        name=$(echo "$name" | xargs)
        time=$(echo "$time" | xargs)
        prev_time=$(grep "^$name |" "$PREV_FILE" 2>/dev/null | cut -d'|' -f2 | xargs)
        if [ -n "$prev_time" ] && [ "$prev_time" -gt 0 ]; then
            change=$(echo "scale=1; (($time - $prev_time) * 100) / $prev_time" | bc 2>/dev/null)
            if [ -n "$change" ]; then
                if (( $(echo "$change < -5" | bc -l) )); then
                    printf "%-35s %8d ms %8d ms   \033[32m%+.1f%%\033[0m\n" "$name" "$prev_time" "$time" "$change"
                elif (( $(echo "$change > 5" | bc -l) )); then
                    printf "%-35s %8d ms %8d ms   \033[31m%+.1f%%\033[0m\n" "$name" "$prev_time" "$time" "$change"
                else
                    printf "%-35s %8d ms %8d ms   %+.1f%%\n" "$name" "$prev_time" "$time" "$change"
                fi
            fi
        fi
    done < "$RESULT_FILE"
fi

echo ""
echo "Done!"
