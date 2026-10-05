#!/usr/bin/env bash
# Build the abi3 extension in release mode and install it into the project venv.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$VENV/bin/activate"
cd "$PROJ"
maturin develop --release --uv "$@"
