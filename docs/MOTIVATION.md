# Motivation and original plan

This is the planning and research note that started `fasttext-new`, kept for context. The
[README](../README.md) describes what the project does today; where the two disagree, the
README is current.

## Problem

Facebook **archived fastText in March 2024**, but it is still used throughout pretraining data
pipelines:

- Language identification with `lid.176.bin` / `lid.176.ftz`.
- Quality classifiers, for example DCLM-style filters.
- Data-curation toolkits such as NVIDIA NeMo Curator and Hugging Face datatrove call it for
  language ID and quality filtering.

The official Python package is unmaintained. People install patched forks (for example
`fasttext-numpy2`) just to get it working with NumPy 2. There are no free-threaded Python wheels,
and batch scoring holds the GIL.

A pure-Rust reimplementation of fastText exists (the `fasttext` crate), but it has no Python
bindings. Writing them, with proof that they behave like the original, is this project.

## Evidence

- [facebookresearch/fastText](https://github.com/facebookresearch/fastText): repository archived
  in March 2024.
- [messense/fasttext-rs](https://github.com/messense/fasttext-rs) /
  [docs.rs/crate/fasttext](https://docs.rs/crate/fasttext/latest): the `fasttext` crate became a
  pure-Rust port in version 0.8.0 (April 2026; versions up to 0.7 wrapped the C++ library). It
  has no Python API.
- [huggingface/datatrove](https://github.com/huggingface/datatrove) and
  [NVIDIA/NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator) use fastText models for language
  identification and quality filtering.
- `fasttext-numpy2` exists to patch the archived package for NumPy 2 compatibility.

## What already existed

| Option | Why it was not enough |
|---|---|
| `fasttext` (official PyPI package) | Archived C++ core with no maintenance; NumPy 2 issues. |
| `fasttext-numpy2` / `fasttext-wheel` forks | Minimal patches on the same archived C++ core. No free-threaded wheels; batch scoring holds the GIL. |
| `fasttext` crate 0.8.0 (messense/fasttext-rs, MIT) | Pure Rust. Loads `.bin` / `.ftz`, has train / quantize / predict / vectors / nearest neighbours. No Python API. |
| `fasttext-pure-rs` crate 0.1.0 | Inference only, aimed at language ID. Smaller scope. |
| An existing Rust-backed Python package on PyPI | In our parity tests (October 2026) it could not load `lid.176.ftz`, and on `lid.176.bin` its top-1 label agreed with the C++ package on only a handful of 10,000 sentences. It also holds the `fasttext-rs` name, so this project is published as `fasttext-new`. |

**Decision:** depend on the pure-Rust `fasttext` crate, vendored with fixes (see
[UPSTREAM.md](../UPSTREAM.md)), and treat the C++ package as the reference: every feature is
tested against it.

## Original MVP scope

1. A PyO3 module exposing the **same API** as `fasttext`, so existing code keeps working:
   `load_model(path)`, `model.predict(text, k=1, threshold=0.0)`, `get_word_vector`,
   `get_sentence_vector`, `get_labels`, `get_words`. *Done.*
2. Load existing `.bin` and `.ftz` models, including `lid.176.bin` and `lid.176.ftz`. *Done.*
3. **Batch prediction** that releases the GIL: `model.predict_batch(list_of_str, k)`. *Done.*
4. abi3 wheels for Linux x86_64 / aarch64, macOS and Windows, built with maturin. *CI workflow
   written; Linux x86_64 and Windows x86_64 tested locally.*
5. A parity test suite: predictions and probabilities match the original C++ to within 1e-5 on
   a fixed corpus. *Done; lid.176 matches bit for bit.*

## Original stretch goals

- Supervised training (`train_supervised`). *Done, together with `train_unsupervised`,
  `test`, `save_model` and `quantize`.*
- Free-threaded Python wheels. *Done for 3.14t (built and tested in CI on Linux, macOS and Windows); PyO3 0.29 does not support 3.13t.*
- A `fasttext-score` CLI that scores a text column in Parquet / JSONL files in parallel. *Not
  started.*
- A datatrove filter adapter and a NeMo Curator example. *Not started.*
- Benchmarks against the C++ package. *Done (see the README).*

## Risks identified up front

- The package name `fasttext` is taken on PyPI: publish under another name and offer a
  compatibility shim (`fasttext_new.install_as_fasttext()`).
- The Rust port might not support every model variant (quantized `.ftz`, supervised training).
  In practice quantized prediction for hierarchical-softmax and negative-sampling models was
  wrong in the crate and had to be fixed (patch 1 in [UPSTREAM.md](../UPSTREAM.md)).
- Floating-point differences: the tolerance was set to 1e-5, and the crate was patched until
  predictions became bit-identical to C++.
- Licensing: fastText is MIT licensed, and so is the Rust port.
