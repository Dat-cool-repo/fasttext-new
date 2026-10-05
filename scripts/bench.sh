#!/usr/bin/env bash
# Build (release) and run the throughput benchmark against the C++ package.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$VENV/bin/activate"
cd "$PROJ"
maturin develop --release --uv -q
FASTTEXT_NEW_DATA="$DATA" python bench/bench.py "$@"
