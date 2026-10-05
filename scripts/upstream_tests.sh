#!/usr/bin/env bash
# Run the upstream `fasttext` 0.8.0 test suite (unit + integration tests) against the patched
# sources in vendor/fasttext/src: copies the pristine crates.io package, overlays our src/ and
# runs `cargo test`. Usage: bash scripts/upstream_tests.sh [cargo test args...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
BASE="${FASTTEXT_NEW_UPSTREAM_DIR:-$HOME/target/fasttext-python-bindings/upstream}"
SRC="$BASE/fasttext-0.8.0"
WORK="$BASE/patched-current"
if [ ! -d "$SRC" ]; then
  mkdir -p "$BASE"
  curl -sL https://crates.io/api/v1/crates/fasttext/0.8.0/download | tar xz -C "$BASE"
fi
rm -rf "$WORK"
cp -r "$SRC" "$WORK"
rm -rf "$WORK/src"
cp -r "$PROJ/vendor/fasttext/src" "$WORK/src"
cd "$WORK"
CARGO_TARGET_DIR="$HOME/target/ftpb-vendor-test" cargo test --release "$@" -- --test-threads=4 2>&1 \
  | grep -E "^test result|FAILED|panicked|^error|failures:|^    [a-z_:]+$" || true
