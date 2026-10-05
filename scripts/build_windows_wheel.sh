#!/usr/bin/env bash
# Cross-compile the Windows x86_64 abi3 wheel from Linux/WSL with mingw-w64
# (needs `rustup target add x86_64-pc-windows-gnu` and the x86_64-w64-mingw32-gcc linker).
# PyO3's `generate-import-lib` feature creates the python3.dll import library, so no Windows
# Python is needed. Output: dist/fasttext_new-*-cp39-abi3-win_amd64.whl
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$VENV/bin/activate"
cd "$PROJ"
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc
maturin build --release --target x86_64-pc-windows-gnu --out dist "$@"
ls -la dist/*win_amd64.whl
