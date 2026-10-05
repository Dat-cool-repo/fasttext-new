#!/usr/bin/env bash
# Rust unit tests + clippy (pure-Rust core and the PyO3 module).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJ"
export FASTTEXT_NEW_DATA="$DATA"
cargo test --release -- --test-threads=4
cargo clippy --all-targets -- -D warnings
PYO3_PYTHON="$VENV/bin/python" cargo clippy --features python --lib -- -D warnings
