//! fastText-compatible Python bindings on top of the pure-Rust `fasttext` crate.
//!
//! * [`model`]: pure-Rust core (C++-compatible tokenization, predict, vectors). Usable and
//!   tested without Python (`cargo test`).
//! * `python` (feature `python`, enabled by maturin): the PyO3 extension module
//!   `fasttext_new._fasttext_new`. The user-facing API lives in `python/fasttext_new/`.

pub mod model;

#[cfg(feature = "python")]
mod python;
