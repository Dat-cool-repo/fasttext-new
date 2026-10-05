#!/usr/bin/env bash
# One-time setup (Linux / WSL / macOS, needs uv and curl): venv, Python deps, the reference C++
# package, and test models / data in $FASTTEXT_NEW_DATA (default ./data).
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$DATA"
[ -d "$VENV" ] || uv venv --python python3.12 "$VENV"
source "$VENV/bin/activate"
uv pip install maturin pytest numpy
# Reference C++ implementation (try numpy2-compatible fork first).
uv pip install fasttext-numpy2 || uv pip install fasttext-wheel
for f in lid.176.ftz lid.176.bin; do
  [ -s "$DATA/$f" ] || curl -fL --retry 3 -o "$DATA/$f" "https://dl.fbaipublicfiles.com/fasttext/supervised-models/$f"
done
ls -la "$DATA"
python -c "import fasttext, numpy; print('fasttext ref ok', fasttext.__file__, numpy.__version__)"
# Multilingual parity corpus (10k sentences, 20 languages; not redistributed).
[ -s "$DATA/papluca_test.csv" ] || curl -fL -o "$DATA/papluca_test.csv" \
  https://huggingface.co/datasets/papluca/language-identification/resolve/main/test.csv
# Small per-loss fixture models trained with the C++ package.
python "$PROJ/scripts/make_fixtures.py" "$DATA"
# Training datasets for tests/test_training.py (cooking.stackexchange, papluca train/valid splits).
python "$PROJ/scripts/make_train_data.py" "$DATA"
# Reference outputs of the C++ package for tests/test_golden.py (used where C++ is unavailable).
python "$PROJ/scripts/make_golden.py" "$DATA"
