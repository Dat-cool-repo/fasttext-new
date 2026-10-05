# fasttext-new

**A maintained, drop-in replacement for the archived `fasttext` Python package, written in Rust,
that gives the same predictions as the original, bit for bit.**

[![wheels](https://github.com/Dat-cool-repo/fasttext-new/actions/workflows/wheels.yml/badge.svg)](https://github.com/Dat-cool-repo/fasttext-new/actions/workflows/wheels.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B%20%7C%203.14t-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

`fasttext-new` is a Python package (PyPI name `fasttext-new`, import name `fasttext_new`) that
exposes the API of the original [fastText](https://github.com/facebookresearch/fastText) Python
bindings on top of a pure-Rust fastText implementation, through [PyO3](https://pyo3.rs). It
loads existing `.bin` / `.ftz` models, predicts, trains, evaluates, quantizes and saves models
that the C++ package can read, and every one of those features is tested against the C++
package.

## Why

Facebook archived the fastText repository in March 2024, but fastText models are still a core
part of many data pipelines, above all for **language identification** (`lid.176.bin` /
`lid.176.ftz`) and **quality filtering** of pretraining data. Toolkits such as
[datatrove](https://github.com/huggingface/datatrove) and
[NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) call it.

The official Python package is unmaintained. Keeping it working means installing forks that patch
the archived C++ core (for NumPy 2, for example). It has no free-threaded wheels, and batch
scoring holds the GIL.

A pure-Rust port of fastText, the [`fasttext` crate](https://github.com/messense/fasttext-rs), had
no Python bindings. The existing Rust-backed Python bindings we tried failed our parity tests
against the C++ package. `fasttext-new` fills that gap: the same API and the same answers as the
original, maintained, with parallel batch scoring and free-threaded Python support.

The original research and planning note is in [docs/MOTIVATION.md](docs/MOTIVATION.md).

## Features

- **Same API as `fasttext`**: `load_model`, `predict`, `train_supervised`,
  `train_unsupervised`, `test`, `test_label`, `quantize`, `save_model`, word / sentence /
  input vectors, words and labels (with `include_freq`), subwords, nearest neighbours,
  analogies, `get_line`, `tokenize`, the input and output matrices, and `on_unicode_error`.
- **Identical predictions.** On `lid.176.bin` and `lid.176.ftz` the labels, their order and the
  probabilities match the C++ package exactly (max |Δp| = 0 over 10,121 multilingual lines).
- **Interoperable files.** Models saved here (including quantized `.ftz`) load in the C++
  package and predict identically there, and vice versa.
- **Fast.** Faster than the C++ package on a single thread, plus `predict_batch` that scores a
  list of texts on all cores with the GIL released.
- **Training with the GIL released**, C++-style `verbose` progress output, and Ctrl-C support.
- **Free-threaded Python** (3.14t): the module does not re-enable the GIL, and
  `predict` scales across Python threads.
- **No C++ toolchain** needed, NumPy 1 and 2 both supported, one abi3 wheel per platform for
  CPython 3.9+.
- **Compatibility shim**: `fasttext_new.install_as_fasttext()` makes `import fasttext` resolve
  to this package, so existing code runs unchanged.

## Status

Version 0.1.0, **alpha**. The full feature set above is implemented and tested on Linux x86_64
(WSL2) and Windows x86_64. Wheels for Linux aarch64 and macOS are built by the CI workflow but
have not been tested on real hardware yet. **Nothing is published on PyPI yet**, so install from
source for now (see below). See [known differences](#known-differences) for what does not match
the original.

## Installation

You need a Rust toolchain (1.85 or newer, from [rustup](https://rustup.rs)) and Python 3.9 or
newer.

```bash
# Straight from GitHub (builds the extension with maturin)
pip install "git+https://github.com/Dat-cool-repo/fasttext-new"

# Or from a clone
git clone https://github.com/Dat-cool-repo/fasttext-new
cd fasttext-new
pip install .                      # build and install a release wheel
# or, for development:
pip install maturin
maturin develop --release          # build and install into the active virtualenv
```

`pip install fasttext-new` will work once the package is published on PyPI. Wheels built by
CI are available as workflow artifacts in the meantime.

## Quick start

```python
import fasttext_new as fasttext

# Language identification with the official model
# (https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz)
model = fasttext.load_model("lid.176.ftz")
model.predict("Bonjour tout le monde, comment allez-vous ?")
# (('__label__fr',), array([0.98...]))
model.predict("Der schnelle braune Fuchs", k=3)        # top 3 labels and probabilities

# Score many documents in parallel, with the GIL released
docs = ["Hello world", "Hola mundo", "Ciao mondo"]
labels, probs = model.predict_batch(docs, k=1)          # threads=None: all cores
# labels[i]: the top-k labels of docs[i]; probs[i]: their probabilities (float32 array)

# Train, evaluate, quantize and save, with the original arguments
clf = fasttext.train_supervised(input="train.txt", lr=1.0, epoch=25, wordNgrams=2)
n, precision, recall = clf.test("valid.txt")
clf.quantize(input="train.txt", retrain=True)
clf.save_model("model.ftz")                             # also loadable by the C++ package

# Word vectors
vec = fasttext.train_unsupervised("corpus.txt", model="skipgram")
vec.get_word_vector("king")
vec.get_nearest_neighbors("king", k=5)
```

### Using it with code that does `import fasttext`

```python
import fasttext_new
fasttext_new.install_as_fasttext()

import fasttext                     # now resolves to fasttext_new
model = fasttext.load_model("lid.176.bin")
```

Call `install_as_fasttext()` before anything imports `fasttext`; libraries that import it later
get `fasttext_new` too. It does nothing if the real `fasttext` package is already imported,
unless you pass `force=True`.

## API compatibility

The Python API follows the original `fasttext` package (`fasttext/FastText.py` from the last
release):

- `predict` returns the same types: for a `str`, `(tuple[str], ndarray[float64])`; for a list,
  `(list[list[str]], list[ndarray[float32]])`. `k=-1` returns every label, and a negative
  `threshold` filters nothing, as in C++.
- `train_supervised` / `train_unsupervised` accept the same keyword and positional arguments
  with the same defaults (`thread = cpu_count() - 1`), the `snake_case` aliases
  (`word_ngrams`, `min_count`, `label_prefix`, ...), and raise `TypeError` on unknown or
  duplicate arguments. Trained models expose `lr`, `dim`, `loss`, `epoch`, ... attributes.
- `pretrainedVectors` adds the `.vec` words to the dictionary like C++.
- `test(path, k, threshold)` returns `(N, precision@k, recall@k)` and `test_label` returns
  per-label `{precision, recall, f1score}`, identical to C++ on the same model file (including
  NaN for labels that are never predicted).
- `quantize(input, qout, cutoff, retrain, epoch, lr, thread, verbose, dsub, qnorm)` has the
  original semantics, including arguments falling back to the model's own values.
- Models whose dictionary contains bytes that are not valid UTF-8 (common for models trained on
  web text) load, keep their exact bytes, and are decoded with the `on_unicode_error` handler
  you pass (`"strict"`, `"replace"`, `"ignore"`, `"surrogateescape"`, ...).

Additions that the original does not have:

- `model.predict_batch(texts, k=1, threshold=0.0, threads=None)`: parallel scoring with the
  GIL released, same return type as `predict(list)`.
- `fasttext_new.install_as_fasttext()`.

### Known differences

- **Training is not bit-identical** to C++. It cannot be: Hogwild! SGD on several threads is
  non-deterministic in both implementations. Trained models are compared by quality instead
  (see below), and vocabularies are identical.
- **Quantized files are not byte-identical** to C++'s, because the product-quantizer k-means
  differs. Quality is equivalent, and files from either implementation load and predict
  identically in the other.
- **Invalid UTF-8 in training and test files** is replaced by U+FFFD when tokenizing, while C++
  keeps the raw bytes. Only files with broken UTF-8 are affected. Models containing such words
  are handled exactly (see `on_unicode_error` above).
- `set_matrices` raises `NotImplementedError`.
- The `loss` attribute of a trained model is a string (`"softmax"`, `"hs"`, ...), not the C++
  pybind enum.
- Autotune (`autotuneValidationFile`, `autotuneDuration`, ...) uses the Rust crate's
  implementation and is only smoke-tested. Its results have not been compared with C++
  autotune.
- `predict(str)` releases the GIL only for texts of 2 KB or more, because for short lines
  releasing it costs more than it saves. Use `predict_batch` for throughput.

## Parity and accuracy

The test suite compares `fasttext_new` against the C++ package (`fasttext-numpy2` 0.10.4) on the
same inputs.

**Inference.** On `lid.176.bin` and `lid.176.ftz` over 10,121 multilingual lines: identical
labels in identical order, max |Δp| = 0. On 13 small fixture models covering every loss
(softmax, hierarchical softmax, one-vs-all, negative sampling), word n-grams up to 3, and
quantized models with and without `qout` / `qnorm` / pruning: identical labels, max |Δp| ≤ 6e-8.
Word, sentence and input vectors, tokenization and subwords are compared too.

**Training quality** (same arguments, `thread=4`, P@1 on a held-out split, one run each;
run-to-run standard deviation is about 0.002):

| Dataset / loss | fasttext-new | C++ | Δ |
|---|--:|--:|--:|
| cooking.stackexchange, softmax (lr 1.0, 25 epochs, wordNgrams 2) | 0.6043 | 0.6017 | +0.0026 |
| cooking.stackexchange, hs | 0.5960 | 0.5817 | +0.0143 |
| cooking.stackexchange, ova (lr 0.5) | 0.6093 | 0.6070 | +0.0023 |
| cooking.stackexchange, ns (lr 0.5) | 0.5513 | 0.5403 | +0.0110 |
| language ID (20 languages), softmax, dim 16, 3 epochs | 0.9912 | 0.9905 | +0.0007 |
| language ID, hs | 0.9883 | 0.9851 | +0.0032 |
| supervised + pretrainedVectors (language ID) | 0.8918 | 0.8916 | +0.0002 |
| skipgram, language purity of 10 nearest neighbours | 0.9872 | 0.9868 | +0.0004 |
| cbow, same metric | 0.8653 | 0.8626 | +0.0027 |

Over 3 runs each, the mean |ΔP@1| was at most 0.011. The tests allow 0.02 to 0.025 (cooking) and
0.01 (language ID). Training speed matches C++: cooking, 25 epochs, `wordNgrams=2`, 4 threads
takes 4.0 s here and 4.1 s with C++.

The hs and ns losses come out about one point better than C++ on cooking. A likely (unproven)
reason: with `thread < 10`, C++ initializes only `thread` tenths of the input matrix with random
values and leaves the rest at zero, while the Rust crate initializes all of it.

**Quantization** (cooking, dense P@1 0.605, quantizing the same `.bin` with each
implementation):

| Settings | fasttext-new | C++ |
|---|--:|--:|
| default | 0.564 | 0.565 |
| `qnorm`, `cutoff=20000`, `retrain` | 0.592 | 0.597 |
| `qout` | 0.562 | 0.561 |
| `cutoff=50000`, `dsub=4` | 0.566 | 0.564 |

## Benchmarks

`bench/bench.py`, documents per second, `k=1`, best of 5. Machine: Intel i9-13900H, WSL2
(Ubuntu 24.04), shared with other jobs, so expect about ±15% noise. Rust threads are capped at 4.
C++ is `fasttext-numpy2` 0.10.4.

Short lines (10,000 sentences, 108 characters on average):

| Model | C++ `predict(str)` loop | C++ `predict(list)` | fasttext-new `predict(str)` loop | `predict_batch`, 1 thread | `predict_batch`, 4 threads | 4 Python threads: C++ / fasttext-new |
|---|--:|--:|--:|--:|--:|--:|
| lid.176.ftz | 78k | 86k | **129k** | 157k | **325k** | 79k / 134k |
| lid.176.bin | 114k | 130k | **129k** | 163k | **338k** | 116k / 142k |

Web-page-sized documents (512 documents, about 2.1 KB each):

| Model | C++ `predict(str)` loop | fasttext-new `predict(str)` loop | `predict_batch`, 4 threads | 4 Python threads: C++ / fasttext-new |
|---|--:|--:|--:|--:|
| lid.176.ftz | 5.4k | **10.0k** | 24.3k | 5.1k / **27.0k** |
| lid.176.bin | 9.6k | **11.5k** | 30.9k | 9.4k / **37.1k** |

On free-threaded Python 3.14t, 4 Python threads calling `predict(str)` run 2.6x (`.ftz`) and
3.2x (`.bin`) faster than one thread. The C++ package does not scale with Python threads.

Run it yourself with `bash scripts/bench.sh` (add `--doc-chars 2000` for long documents).

## Platforms

| Platform | Wheel | Tested |
|---|---|---|
| Linux x86_64, abi3 (CPython 3.9+) | CI and local | Full test suite |
| Linux x86_64, free-threaded 3.14t | CI and local | Golden tests and a thread-scaling check |
| Windows x86_64, abi3 | CI (MSVC); also cross-compiled with mingw-w64 (`scripts/build_windows_wheel.sh`) | Golden tests and a smoke test on Windows with the mingw build |
| Linux aarch64, macOS x86_64 / arm64 (abi3 and 3.14t), Windows 3.14t | CI only | Golden tests and a smoke test in CI |

The CI workflow ([`.github/workflows/wheels.yml`](.github/workflows/wheels.yml)) builds abi3 and
free-threaded 3.14t wheels for all of these plus an sdist, runs `cargo fmt` / `test` /
`clippy`, and tests every wheel on its platform with Python 3.10, 3.12 and 3.14t. There is no
3.13t wheel: PyO3 0.29 supports free-threaded CPython only from 3.14. It does not publish
anything.

## Development

The helper scripts in `scripts/` are written for Linux, WSL and macOS (bash). They read their
settings from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `FASTTEXT_NEW_DATA` | `./data` | Models, corpora and training data used by the tests and benchmarks |
| `FASTTEXT_NEW_VENV` | `./.venv` | The virtualenv the scripts create and use |
| `CARGO_TARGET_DIR` | `./target` | Cargo build directory |
| `RAYON_NUM_THREADS` | `4` | Thread count for parallel prediction in tests and benchmarks |

```bash
bash scripts/setup_env.sh      # once: venv (maturin, pytest, numpy, fasttext-numpy2 as the C++
                               # reference), then downloads lid.176.{ftz,bin} and the test corpora
                               # into $FASTTEXT_NEW_DATA and builds the fixture models and golden files
bash scripts/test.sh -q        # build (release) + the whole pytest suite (a few minutes)
bash scripts/test.sh -q tests/test_parity.py     # inference parity only (about 30 s)
bash scripts/cargo_check.sh    # cargo test + clippy -D warnings (with and without PyO3)
bash scripts/upstream_tests.sh # the fasttext crate's own test suite on the patched sources
bash scripts/bench.sh          # throughput benchmark against the C++ package
bash scripts/build.sh          # just build and install the extension into the venv
bash scripts/build_windows_wheel.sh  # cross-compile a Windows x86_64 abi3 wheel with mingw-w64
```

`setup_env.sh` needs [uv](https://github.com/astral-sh/uv) and `curl`. Downloaded data is never
committed. The test data comes from:

- `lid.176.bin` / `lid.176.ftz`: the official
  [fastText language-identification models](https://fasttext.cc/docs/en/language-identification.html)
  (CC BY-SA 3.0).
- [cooking.stackexchange](https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz),
  the dataset of the official fastText tutorial.
- [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification)
  (20 languages), used as the multilingual parity corpus and for language-ID training.
- Small fixture models trained from it with the C++ package (`scripts/make_fixtures.py`) and
  recorded C++ outputs (`scripts/make_golden.py`).

The test suite has about 234 tests: `tests/test_parity.py` (inference parity against the live
C++ package), `tests/test_training.py` (training quality, `test()`, save / quantize
interoperability, argument handling, GIL release, Ctrl-C) and `tests/test_golden.py` (parity
against recorded C++ outputs, for platforms where the C++ package cannot be installed). Tests
whose models or data are missing are skipped, so a fresh clone runs `test_golden.py` with only
`lid.176.ftz` downloaded into `$FASTTEXT_NEW_DATA`.

On Windows (PowerShell), to test an installed wheel:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install dist\fasttext_new-0.1.0-cp39-abi3-win_amd64.whl pytest
$env:FASTTEXT_NEW_DATA = "data"
.venv\Scripts\python -m pytest -q -p no:cacheprovider tests\test_golden.py
```

## Project layout

```
src/model.rs          pure-Rust core: C++-compatible tokenization, predict, vectors,
                      train / quantize / save, test() with a port of C++ Meter, unit tests
src/python.rs         PyO3 bindings (GIL release, free-threading, Ctrl-C during training)
src/lib.rs            crate root
python/fasttext_new/  the fastText-compatible Python API (FastText.py) and the import shim
tests/                parity, training and golden test suites; tests/data has the small
                      checked-in sentences and golden file
bench/bench.py        throughput benchmark against the C++ package
examples/profile.rs   tokenization vs. inference profiler
scripts/              setup, build, test, benchmark and patch-maintenance scripts
vendor/fasttext/      vendored, patched copy of the `fasttext` crate 0.8.0
vendor/fasttext-0.8.0-fixes.patch   combined diff of the vendored crate against 0.8.0
upstream/             the same fixes split into a 9-patch series for the crate
docs/MOTIVATION.md    the original research and planning note
.github/workflows/    CI: wheels for every platform, Rust checks, wheel tests
```

## The vendored `fasttext` crate and upstreaming

`fasttext-new` depends on a vendored copy of the
[`fasttext` crate](https://github.com/messense/fasttext-rs) 0.8.0 in `vendor/fasttext/`, with
fixes for correctness (quantized prediction for hierarchical-softmax and negative-sampling
models was wrong, so `lid.176.ftz` predicted `zh` for French), bit-exact scores, speed and
training behaviour. Every change is marked `[fasttext-python-bindings patch]` in the source. The
crate's own test suite (455 tests) passes on the patched sources.

The fixes are prepared as a patch series for the crate in `upstream/`, described in
[UPSTREAM.md](UPSTREAM.md). They have not been submitted yet. The goal is to drop the vendored
copy once the fixes are upstream.

A few remaining crate differences are worked around in `src/model.rs` instead (for example,
`</s>` handling in word n-grams and predictions on empty text); UPSTREAM.md lists them.

## License

`fasttext-new` is dual-licensed under either of

- the MIT license ([LICENSE-MIT](LICENSE-MIT)), or
- the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)),

at your option.

Third-party material in this repository keeps its own license:

- `vendor/fasttext/` is a modified copy of the `fasttext` crate, MIT licensed, Copyright (c)
  2018 messense, which also carries fastText's original BSD license notice. See
  [vendor/fasttext/LICENSE](vendor/fasttext/LICENSE).
- `tests/data/golden/lid.176.ftz.json` records outputs of Facebook's `lid.176.ftz` model and is
  distributed under CC BY-SA 3.0, like the model. See
  [tests/data/golden/NOTICE](tests/data/golden/NOTICE).

No models or datasets are included. The pre-trained fastText models you download (such as
`lid.176.bin`) are licensed by their authors, CC BY-SA 3.0 for the language-identification
models.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

## Acknowledgements

- [facebookresearch/fastText](https://github.com/facebookresearch/fastText): the original
  library, models and Python API that this project reproduces.
- [messense/fasttext-rs](https://github.com/messense/fasttext-rs): the pure-Rust fastText
  implementation this package is built on.
- [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).
