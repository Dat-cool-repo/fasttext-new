# Upstreaming the `fasttext` crate fixes

**Status: not submitted yet. No PR or issue has been opened upstream, and the maintainer has not
been contacted.**

`fasttext-new` depends on a vendored, patched copy of crates.io
[`fasttext` 0.8.0](https://crates.io/crates/fasttext/0.8.0)
([messense/fasttext-rs](https://github.com/messense/fasttext-rs), MIT) in `vendor/fasttext/`.
This document turns those changes into a patch series that the crate could take, with a PR
description draft.

## The series (`upstream/`)

The patches are relative to the crate (repository) root. Each one is `git am`-compatible and
builds and passes the crate's full test suite (455 tests) on its own, applied on top of the
previous ones.

| # | Patch | What and why | Size |
|---|---|---|---|
| 1 | `0001-fix-quantized-hs-ns-predict` | **Bug fix.** Quantized (`.ftz`) prediction applied a softmax over the first `nlabels` output rows for every loss. That is wrong for hierarchical softmax (the rows are Huffman-tree nodes) and negative sampling, so `lid.176.ftz` predicted `zh` for French. It now delegates to the real loss, and for `qout` HS/NS models runs the loss over a lazily dequantized output matrix. | +49 −14 |
| 2 | `0002-bit-exact-scores` | Probabilities and tie order identical to C++: f64 `std_log`, a libstdc++ heap port for top-k and the HS DFS, f64 HS sigmoid, softmax division, and a sequential output dot product (SIMD behind a new `simd-dot` feature). | +131 −38 |
| 3 | `0003-raw-predict-and-input-rows` | New APIs for bindings: `predict_on_words_raw` (no label `String` per prediction; threshold passed through unclamped like C++) and `add_input_rows` (C++ `addInputVector`, dense or quantized). | +47 |
| 4 | `0004-faster-subword-hashing` | Incremental FNV-1a n-gram hashing without an allocation per n-gram, plus a cheap integer hasher for `pruneidx`. About 3x faster `.ftz` tokenization. | +40 −11 |
| 5 | `0005-training-os-threads-and-progress` | Hogwild! workers get one OS thread each instead of rayon tasks, which ran serially when the pool was smaller than `thread` (changing the lr schedule). `verbose` is now honoured with C++-style dictionary / `Progress:` output, including during `quantize(retrain=true)`. | +183 −53 |
| 6 | `0006-parallel-matrix-init` | `DenseMatrix::uniform` fills its 10 independently seeded blocks in parallel (same values). Saves about 0.5 s per run with 2M buckets. | +25 −8 |
| 7 | `0007-pretrained-vectors-join-dictionary` | `pretrainedVectors`: like C++, every `.vec` word joins the dictionary before the input matrix is sized. Before, only words already in the vocabulary got their vectors, so vocabularies differed from C++. | +35 −29 |
| 8 | `0008-quantize-args-and-memory` | `quantize(retrain=true)` keeps `epoch`/`lr`/`thread`/`verbose` in the model args like C++ (`epoch` is saved in the file). The now-unused dense input matrix (often hundreds of MB) is dropped after quantization instead of being kept next to the quantized one. | +23 |
| 9 | `0009-dictionary-non-utf8-words` | Models whose dictionary has words that are not valid UTF-8 (common for models trained on web text) used to be rejected at load. Such entries now keep their exact bytes (`Entry::raw`) for hashing, n-grams and saving, so they load, predict and round-trip like in C++. | +56 −16 |

Each patch's commit message explains the change in detail. Patches 1 and 2 are the important
ones. 3 and 4 help any language binding. 5 to 8 make training and quantization behave like C++.

### How it was verified

* **Crate tests:** `scripts/make_upstream_series.py --test` runs `cargo test --release` (455
  unit and integration tests from the crates.io package) after every patch, and checks that the
  whole series reproduces `vendor/fasttext/src` byte for byte (minus our
  `[fasttext-python-bindings patch]` marker comments, which are left out of the upstream copies).
  `scripts/upstream_tests.sh` runs the same suite on the vendored sources.
* **Against C++ fastText** (the Python package `fasttext-numpy2` 0.10.4), through the Python
  bindings in this repository (`tests/`):
  * prediction on `lid.176.bin` / `lid.176.ftz` and 13 fixture models (every loss, quantized
    with and without `qout` / `qnorm` / pruning): labels and order identical on 10,121 lines,
    max |Δp| = 0 for lid.176 and ≤ 6e-8 for the fixtures;
  * training (patches 5–8): P@1 / R@5 within 0.02 of C++ on cooking.stackexchange and within
    0.01 on 20-language identification, identical vocabularies, unsupervised neighbour quality
    within 0.005;
  * quantization: models quantized here load in C++ and predict identically, and vice versa;
  * non-UTF-8 dictionaries (patch 9): words, labels and predictions decoded with
    `replace` / `ignore` / `surrogateescape` match C++, and a model re-saved here loads in C++
    with the same raw bytes.

### Applying

```bash
git clone https://github.com/messense/fasttext-rs && cd fasttext-rs
git log --oneline -1   # check that src/ still matches the 0.8.0 release; rebase if not
git am /path/to/fasttext-new/upstream/*.patch    # or: git apply <patch> for each
cargo test --release
```

The patches were generated against the published 0.8.0 package (`src/` and `Cargo.toml.orig`),
not against a repository checkout. If upstream has moved on, regenerate or rebase them. The
`From:` line is a placeholder (`FASTTEXT_NEW_PATCH_AUTHOR` overrides it when regenerating, or use
`git commit --amend --reset-author` after `git am`). Regenerate with:

```bash
python3 scripts/make_upstream_series.py --test   # Linux / WSL; writes upstream/*.patch
```

## Suggested PRs

It is probably easiest to send these as separate PRs, in this order:

1. **#1 on its own**: a clear user-visible bug with an easy reproduction (`lid.176.ftz`).
2. **#2 + #3 + #4**: "C++ parity and speed for bindings".
3. **#5 – #8**: "training / quantization behave like C++".
4. **#9**: "load models with non-UTF-8 dictionary entries".

### PR description draft (for #1)

> **Fix quantized (.ftz) prediction for hierarchical-softmax and negative-sampling models**
>
> `predict_raw_quantized` applies a softmax over the first `nlabels` rows of the output matrix
> regardless of the loss. For hierarchical softmax those rows are internal Huffman-tree nodes,
> and negative sampling is a per-label binary logistic, so predictions from HS / NS `.ftz`
> models are wrong. The most visible case is the official language-ID model:
>
> ```rust
> let m = FastText::load_model("lid.176.ftz")?;   // loss = hs
> m.predict("Bonjour tout le monde, comment allez-vous ?", 1, 0.0)
> // before: __label__zh    after: __label__fr (0.98, same as C++ fastText)
> ```
>
> This PR mirrors C++ `Model::predict`. When the output matrix is dense (`qout = false`), it
> calls the model's loss, as for non-quantized models. When it is quantized (`qout = true`),
> softmax / one-vs-all keep the existing path, and HS / NS run their loss over a dequantized
> copy of the output matrix, built once on first use (`OnceLock`) and reset by `quantize()`.
>
> Testing: `cargo test` passes. I compared against the C++ package (`fasttext-numpy2` 0.10.4)
> on 10,121 multilingual sentences. `lid.176.ftz` now gives the same top-k labels and
> probabilities. HS and NS fixture models, with and without `qout`, also match. I can add a
> regression test against a small HS `.ftz` fixture if you'd like one in `tests/`.
>
> Context: I'm building Python bindings on top of this crate (a drop-in replacement for the
> archived `fasttext` Python package) and have a few more parity fixes queued: bit-exact
> scores, training thread handling, pretrained vectors, non-UTF-8 dictionaries. I'm happy to
> send them as follow-ups if they're welcome.

### PR description draft (for #2–#4)

> **Bit-exact C++ parity for predictions, plus two small APIs and faster tokenization**
>
> * Scores and the order of tied labels now match C++ fastText exactly: f64 `std_log`,
>   libstdc++-compatible heap for top-k and the HS DFS, f64 HS sigmoid, division in softmax,
>   sequential output dot product. The SIMD dot product is still available behind the new
>   `simd-dot` feature. Ties are common with the table sigmoid of OVA / NS models, and
>   without this the label order differs from C++.
> * `predict_on_words_raw` (no `String` per prediction) and `add_input_rows` (input rows of
>   dense or quantized models), which bindings need to reproduce C++ exactly without extra
>   allocation.
> * About 3x faster tokenization for quantized models: incremental n-gram hashing, and an
>   integer hasher for `pruneidx` instead of SipHash.
>
> Verified: `lid.176.bin` / `.ftz` predictions are identical to the C++ package (max |Δp| = 0)
> on 10k sentences. `cargo test` passes after each commit.
>
> Open question: should exact parity be the default (as here) or opt-in? The scalar dot
> product only affects the output layer, so the cost is small.

## Open questions for the maintainer

* `simd-dot`: exact parity by default (as in patch 2) or the faster SIMD dot product by default?
* The public `predict()` / `predict_on_words()` / `test_model()` clamp a negative threshold to
  0 (documented). C++ instead filters nothing and disables the HS pruning. Patch 3 follows C++
  only in the new raw API. Should the others follow too?
* The training tokenizer still turns invalid UTF-8 into U+FFFD (C++ keeps the bytes). Making it
  byte-based is a larger change than patch 9.

## Other differences found, not in the series

These are worked around in the bindings (`src/model.rs`) rather than patched in the crate:

* `FastText::predict(text)` adds `</s>` after the word n-grams are computed, so `</s>` is left
  out of the word-n-gram hashes (C++ includes it, which matters for `wordNgrams > 1`
  classifiers). It also returns nothing for empty text, while C++ predicts from `</s>`. The
  bindings re-implement C++ `Dictionary::getLine` and call `predict_on_words_raw`.
* `Dictionary::tokenize` drops tokens beyond 1,024 per line. C++ `readWord` (and the Python
  `fasttext.tokenize`) does not.
* `get_ngram_strings` leaves out the strings of n-grams pruned from a quantized model. C++
  lists every n-gram string, but only the surviving ids.
* C++ quirk, for reference: with `thread < 10`, C++ `DenseMatrix::uniform` initialises only the
  first `thread` tenths of the input matrix and leaves the rest at zero. Measured with
  `thread=4`: 40% non-zero. The crate initialises everything. This likely explains why some
  losses (hs, ns) train about 1 point of P@1 *better* here than in C++ on cooking.stackexchange.

## About the Python bindings

The original plan ([docs/MOTIVATION.md](docs/MOTIVATION.md)) included offering the Python
bindings to the crate's maintainer, either upstream or as a companion package. That conversation
has not been started. A possible first message, once the bug-fix PR is in:

> Hi! I've built `fasttext-new`, PyO3 bindings for this crate that reproduce the archived
> `fasttext` Python package's API (load / predict / train / quantize / test, abi3 and
> free-threaded wheels). It matches the C++ package exactly on lid.176 and trains to the same
> accuracy. Would you be interested in hosting the bindings in this repository (e.g. as a
> `python/` workspace member), or would you prefer it to stay a separate companion package
> that links here?
