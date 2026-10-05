#!/usr/bin/env bash
# Regenerate vendor/fasttext-0.8.0-fixes.patch (diff of vendor/fasttext/src vs crates.io fasttext 0.8.0).
set -euo pipefail
source "$(dirname "$0")/env.sh"
SRC="${FASTTEXT_NEW_UPSTREAM_DIR:-$HOME/target/fasttext-python-bindings/upstream}/fasttext-0.8.0"
if [ ! -d "$SRC" ]; then
  mkdir -p "$(dirname "$SRC")"
  curl -sL https://crates.io/api/v1/crates/fasttext/0.8.0/download | tar xz -C "$(dirname "$SRC")"
fi
cd "$PROJ"
diff -ru "$SRC/src" vendor/fasttext/src | sed "s|$SRC/|a/|; s|vendor/fasttext/|b/|" > vendor/fasttext-0.8.0-fixes.patch || true
wc -l vendor/fasttext-0.8.0-fixes.patch
