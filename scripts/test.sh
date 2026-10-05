#!/usr/bin/env bash
# Build the extension into the venv and run the pytest parity suite.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$VENV/bin/activate"
cd "$PROJ"
export FASTTEXT_NEW_DATA="$DATA"
maturin develop --release --uv -q
ls "$DATA"/fx_*.bin >/dev/null 2>&1 || python scripts/make_fixtures.py "$DATA"
python -m pytest -q -p no:cacheprovider "$@"
