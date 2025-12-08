#!/bin/bash
# Run all .nos test files from the tests/ directory

set -e

cd "$(dirname "$0")/.."

if [ -n "$1" ]; then
    # Run tests matching a pattern
    cargo test --package nostos-compiler --test nos_files "$1"
else
    # Run all .nos tests
    cargo test --package nostos-compiler --test nos_files
fi
