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
  probabilities match the C++ package exactly: max |Δp| = 0 over 10,121 multilingual test
  lines, and the top-1 label and probability are bit-identical on 672,499 real web documents
  (2.9 GB of FineWeb / FineWeb-2 text in 17 languages).
- **Interoperable files.** Models saved here (including quantized `.ftz`) load in the C++
  package and predict identically there, and vice versa.
- **Fast.** On real web documents, faster than the C++ package on a single thread (1.2x with
  `lid.176.bin`), plus `predict_batch`, which scores a list of texts on all cores with the GIL
  released (2.8x with 4 threads on `.bin`, 4.2x on `.ftz`). See [Benchmarks](#benchmarks).
- **Safe with untrusted model files.** A corrupt or malicious `.bin` / `.ftz` raises
  `ValueError` instead of crashing or allocating without bound (fuzzed, see [Testing](#testing)).
- **Training with the GIL released**, C++-style `verbose` progress output, and Ctrl-C support.
- **Free-threaded Python** (3.14t): the module does not re-enable the GIL, and `predict`
  scales across Python threads (checked in CI).
- **No C++ toolchain** needed, NumPy 1 and 2 both supported, one abi3 wheel per platform for
  CPython 3.9+.
- **Compatibility shim**: `fasttext_new.install_as_fasttext()` makes `import fasttext` resolve
  to this package, so existing code runs unchanged.

## Status

Version 0.1.0, **alpha**. The full feature set above is implemented. The complete test suite,
including the comparisons with the C++ package, runs on Linux x86_64 (WSL2); Windows x86_64 was
also tested by hand. In CI, every wheel (Linux x86_64 / aarch64, macOS x86_64 / arm64, Windows
x86_64, abi3 and free-threaded 3.14t) is tested on its platform with about 250 tests that do not
need the C++ package (see [Testing](#testing)). The test suite was also run by hand on Apple
Silicon (macOS, M5 Pro). See [known differences](#known-differences) for what does not match the
original.

## Installation

```bash
pip install fasttext-new
```

Prebuilt wheels cover Linux x86_64 / aarch64, macOS x86_64 / arm64 and Windows x86_64, for
CPython 3.9+ (abi3) and free-threaded 3.14t. To build from source you need a Rust toolchain
(1.85 or newer, from [rustup](https://rustup.rs)) and Python 3.9 or newer:

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
# (train.txt / valid.txt: one example per line, "__label__<tag> <text>", e.g. the
# cooking.stackexchange data of the fastText tutorial; corpus.txt below: any raw text)
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
- **Corrupt or malicious model files are rejected** with `ValueError` where C++ trusts the file
  (and may crash or allocate without bound): every size in the file is checked against the file
  length before allocating, and the loaded shapes are validated. Files that C++ writes always
  pass; the limits only exclude models no fastText tooling produces (`dim` above 65,536,
  `maxn` or `wordNgrams` above 256, inconsistent matrix shapes, degenerate label counts).

## Parity and accuracy

The test suite compares `fasttext_new` against the C++ package (`fasttext-numpy2` 0.10.4) on the
same inputs.

**Inference.** On `lid.176.bin` and `lid.176.ftz` over 10,121 multilingual lines: identical
labels in identical order, max |Δp| = 0. On 13 small fixture models covering every loss
(softmax, hierarchical softmax, one-vs-all, negative sampling), word n-grams up to 3, and
quantized models with and without `qout` / `qnorm` / pruning: identical labels, probabilities
within 1e-5 (the test tolerance). Word, sentence and input vectors, tokenization and subwords
are compared too.

**Real web text.** `bench/real_agreement.py` ran both implementations over 672,499 documents
(2.9 GB of text: the FineWeb-2 test shards of 16 languages plus 200 MB of English FineWeb,
newlines replaced by spaces as datatrove does): the top-1 label agreed on **100%** of the
documents, and the probability was **bit-identical on 100%**, with both `lid.176.bin` and
`lid.176.ftz` (zero mismatches to investigate).

**Training quality** (same arguments, `thread=4`, P@1 on a held-out split, one run each,
measured with `tests/test_training.py` on a 4-vCPU GitHub runner):

| Dataset / loss | fasttext-new | C++ | Δ |
|---|--:|--:|--:|
| cooking.stackexchange, softmax (lr 1.0, 25 epochs, wordNgrams 2) | 0.6020 | 0.6097 | −0.0077 |
| cooking.stackexchange, hs | 0.5990 | 0.5863 | +0.0127 |
| cooking.stackexchange, ova (lr 0.5) | 0.6093 | 0.6070 | +0.0023 |
| cooking.stackexchange, ns (lr 0.5) | 0.5537 | 0.5397 | +0.0140 |
| language ID (20 languages), softmax, dim 16, 3 epochs | 0.9912 | 0.9904 | +0.0008 |
| language ID, hs | 0.9883 | 0.9852 | +0.0031 |
| supervised + pretrainedVectors (language ID) | 0.8919 | 0.8921 | −0.0002 |
| skipgram, language purity of 10 nearest neighbours | 0.9875 | 0.9868 | +0.0007 |
| cbow, same metric | 0.8655 | 0.8622 | +0.0033 |
| FineWeb-2 language ID (17 languages, first 100 characters, 306k lines), mean of 3 runs | 0.9900 | 0.9900 | +0.0001 |

Training is stochastic, so single runs differ by up to about a point; the tests allow 0.02 to
0.025 (cooking) and 0.01 (language ID). Training speed is close to C++: the FineWeb-2 classifier
(5 epochs, 4 threads) trains in 8.8 s here and 8.0 s with C++ on the same runner.

The hs and ns losses come out about one point better than C++ on cooking. A likely (unproven)
reason: with `thread < 10`, C++ initializes only `thread` tenths of the input matrix with random
values, while the Rust crate initializes all of it. The C++ package allocates that matrix
uninitialized, so the rest is zero for large matrices but heap garbage for small ones, which is
also why the C++ trainer sporadically aborts small runs with "Encountered NaN".

**Quantization** (cooking, dense P@1 0.604, quantizing the same `.bin` with each
implementation):

| Settings | fasttext-new | C++ |
|---|--:|--:|
| default | 0.568 | 0.568 |
| `qnorm`, `cutoff=20000`, `retrain` | 0.596 | 0.600 |
| `qout` | 0.568 | 0.570 |
| `cutoff=50000`, `dsub=4` | 0.560 | 0.560 |

## Benchmarks

Language identification over the 672,499 real web documents above (4.4 KB on average), with
`bench/real_agreement.py` on a GitHub-hosted `ubuntu-latest` runner (4 vCPUs). C++ is
`fasttext-numpy2` 0.10.4 `predict(list)`; fasttext-new is `predict_batch`. Documents per second
(text MB/s), and the peak RSS of the whole Python process:

| Model | C++ | fasttext-new, 1 thread | fasttext-new, 4 threads | Peak RSS, C++ / fasttext-new |
|---|--:|--:|--:|--:|
| lid.176.bin | 6,454 (28 MB/s) | 7,870 (34 MB/s) | **18,230** (80 MB/s) | 603 / 594 MB |
| lid.176.ftz | 2,803 (12 MB/s) | not measured | **11,640** (51 MB/s) | 435 / 431 MB |

`bench/bench.py` (`bash scripts/bench.sh`, add `--doc-chars 2000` for long documents) compares
the `predict(str)` loop, `predict(list)`, `predict_batch` and several Python threads on short
lines. Its numbers depend a lot on the machine; run it on yours.

## Platforms

| Platform | Wheel | Tested |
|---|---|---|
| Linux x86_64, abi3 (CPython 3.9+) | CI and local | Full test suite, including the comparisons with the C++ package (locally and in the manual `reference` workflow); in CI, about 330 tests on every push |
| Linux x86_64 / aarch64, free-threaded 3.14t | CI | About 310 tests, including thread scaling |
| Linux aarch64, abi3 | CI | About 310 tests (golden C++ outputs, self-consistency, corrupt models) |
| macOS x86_64 / arm64 (abi3 and 3.14t) | CI | About 310 tests per job; on Apple Silicon (M5 Pro) also by hand: 312 tests pass, plus the README examples |
| Windows x86_64, abi3 and 3.14t | CI (MSVC); also cross-compiled with mingw-w64 (`scripts/build_windows_wheel.sh`) | About 310 tests in CI; tested by hand with the mingw build |

The CI workflow ([`.github/workflows/wheels.yml`](.github/workflows/wheels.yml)) builds abi3 and
free-threaded 3.14t wheels for all of these plus an sdist, runs `cargo fmt` / `test` /
`clippy`, smoke-fuzzes every fuzz target, and tests every wheel on its platform with Python 3.10,
3.12 and 3.14t (and 3.9 with NumPy 1 on Linux, macOS arm64 and Windows). There is no 3.13t
wheel: PyO3 0.29 supports free-threaded CPython only from 3.14. It does not publish anything.

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
python scripts/make_tiny_models.py   # retrain the tiny committed models (C++ package)
python scripts/make_golden.py --tiny # re-record their golden C++ outputs
```

The `reference` workflow (`gh workflow run reference`) runs the steps that need the C++ package
or large downloads on a GitHub runner: the whole suite after `setup_env.sh`, the real-web-text
agreement, and the README examples in fresh virtualenvs.

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
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and
  [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (ODC-By) for the
  real-web-text checks (`bench/fetch_real_data.py`).

The only models in the repository are the tiny ones in `tests/data/models/` (5 to 48 KB each),
trained by `scripts/make_tiny_models.py` on synthetic text, with their recorded C++ outputs in
`tests/data/golden/tiny_*.json`.

See [Testing](#testing) for what the tests cover. Tests whose models or data are missing are
skipped, so a fresh clone runs the golden, self-check and corrupt-model tests with nothing
downloaded.

On Windows (PowerShell), to test an installed wheel:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install dist\fasttext_new-0.1.0-cp39-abi3-win_amd64.whl pytest
$env:FASTTEXT_NEW_DATA = "data"
.venv\Scripts\python -m pytest -q -p no:cacheprovider tests\test_golden.py tests\test_selfcheck.py tests\test_malformed.py
```

The mingw-w64 build calls `expf` / `log` from the Windows UCRT (`ucrtbase.dll`), like the MSVC
build: mingw-w64's own `expf` is about 15x slower, which made softmax training 2.5x slower than
on Linux (cooking, 25 epochs, 4 threads: 10-11 s instead of about 4.5 s; 5 epochs on 1 thread:
5-6 s instead of 2.6 s). The mingw wheel now takes 4.1-5.7 s and 2.2-2.4 s, the same as the
MSVC wheel (measured with the threads on the performance cores; with the default scheduling,
Windows also uses the efficiency cores of hybrid CPUs, which adds 1-3 s).

## Testing

| Suite | What | Needs |
|---|---|---|
| `tests/test_parity.py` | inference parity with the live C++ package: `lid.176.bin` / `.ftz` and 13 fixture models on 10,121 lines (papluca/language-identification test split, the checked-in sentences and edge cases), vectors, vocabulary, subwords, matrices, errors | C++ package, `setup_env.sh` data |
| `tests/test_training.py` | training quality vs C++ (table above), `test()` / `test_label()` equality, save / quantize interoperability in both directions, argument handling, GIL release, Ctrl-C, pretrained vectors, autotune smoke test, non-UTF-8 dictionaries | C++ package, `setup_env.sh` data |
| `tests/test_golden.py` | parity with recorded C++ outputs: `lid.176.ftz` (checked in) and the 10 tiny models (every loss, quantized with `qnorm` / `qout` / pruning, cbow): predictions incl. `k=-1`, `test()` / `test_label()`, vectors, vocabulary, subwords, nearest neighbours, analogies | nothing (`lid.176.ftz` optional) |
| `tests/test_selfcheck.py` | train → save → load → predict for every loss, quantization round trips, unsupervised models, the API surface, threads (concurrent predict, quantize during predict, GIL release, free-threaded scaling) | nothing |
| `tests/test_malformed.py` | truncated files, bad header / dictionary / matrix / quantizer fields, random byte flips, degenerate counts, fuzz regressions: always `ValueError` | nothing |
| `cargo test` | Rust unit tests and `tests/fuzz_regressions.rs` (replays `fuzz/regressions/`) | nothing |
| `scripts/upstream_tests.sh` | the `fasttext` crate's own 455 tests on the patched sources | network (crates.io) |

On every push, CI runs about 310 of these tests on each of Linux x86_64 / aarch64, macOS
x86_64 / arm64 and Windows (Python 3.9 to 3.14t), and about 330 on Linux x86_64 with the C++
package. The full suite (611 tests with the C++ package and data: 540 pass, 71 are skipped as not
applicable, e.g. a reference missing from an older local golden file) runs locally and in the manual `reference` workflow.

**Real data.** 672,499 FineWeb / FineWeb-2 documents (2.9 GB, 17 languages) through
`lid.176.bin` and `lid.176.ftz` with both implementations: 100% identical top-1 labels and
bit-identical probabilities. A 17-language classifier trained on 306k of these documents gets
the same P@1 with both (0.9900).

**Fuzzing** ([`fuzz/`](fuzz/README.md), cargo-fuzz with AddressSanitizer). Three targets:
`load_model` (arbitrary bytes as a `.bin` / `.ftz`: args, dictionary, dense and quantized
matrices, product quantizers; every inference call, save and reload), `predict` (arbitrary
valid and invalid UTF-8 through tokenization, `predict`, `predict_batch`, `test()`, vectors and
subwords with the 10 tiny models) and `train` (arbitrary training files and `.vec` files, tiny
settings, then predict / test / save / reload / quantize). Before release: 25 minutes per target
with 2 workers on a GitHub runner (330k, 1.2M and 225k executions), plus shorter local runs while
fixing; every CI run fuzzes each target for 60 seconds. Crashes found by fuzzing, all fixed and
kept as regression inputs: a Huffman-tree build reading past its array (label counts above
the 1e18 sentinel), a panic in nearest-neighbour sorting with NaN vectors, label word vectors
reading past the input matrix when `bucket = 0`, and `.vec` words with NUL bytes that made
saved models unloadable. Hardened before fuzzing, from reviewing the loader (and tested by
`test_malformed.py`): every size declared in a model file is checked against the file length
before allocating, the loaded shapes are validated, and degenerate label counts (deep or
quadratic Huffman trees, zero labels) are rejected.

**Platforms.** Linux x86_64 (WSL2) by hand and in CI; Windows x86_64 by hand (mingw build) and in
CI (MSVC); macOS arm64 by hand (Apple Silicon, M5 Pro: 312 tests, `cargo test` and the README
examples) and in CI; Linux aarch64 and macOS x86_64 in CI only.

## Project layout

```
src/model.rs          pure-Rust core: C++-compatible tokenization, predict, vectors,
                      train / quantize / save, test() with a port of C++ Meter, unit tests
src/python.rs         PyO3 bindings (GIL release, free-threading, Ctrl-C during training)
src/lib.rs            crate root
python/fasttext_new/  the fastText-compatible Python API (FastText.py) and the import shim
tests/                parity, training, golden, self-check and corrupt-model test suites;
                      tests/data has the checked-in sentences, tiny models and golden files
fuzz/                 cargo-fuzz targets and the regression inputs of the crashes they found
bench/                throughput benchmark and real-web-text checks against the C++ package
examples/profile.rs   tokenization vs. inference profiler
scripts/              setup, build, test, benchmark and patch-maintenance scripts
vendor/fasttext/      vendored, patched copy of the `fasttext` crate 0.8.0
vendor/fasttext-0.8.0-fixes.patch   combined diff of the vendored crate against 0.8.0
upstream/             the same fixes split into a 9-patch series for the crate
docs/MOTIVATION.md    the original research and planning note
.github/workflows/    CI: wheels for every platform, Rust checks, wheel tests, fuzzing, and the
                      manual `reference` workflow
```

## The vendored `fasttext` crate and upstreaming

`fasttext-new` depends on a vendored copy of the
[`fasttext` crate](https://github.com/messense/fasttext-rs) 0.8.0 in `vendor/fasttext/`, with
fixes for correctness (quantized prediction for hierarchical-softmax and negative-sampling
models was wrong, so `lid.176.ftz` predicted `zh` for French), bit-exact scores, speed,
training behaviour and robustness against corrupt model files. Every change is marked `[fasttext-python-bindings patch]` in the source. The
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
