# Fuzzing

[cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) targets (libFuzzer, nightly Rust):

| Target | Input | Checks |
|---|---|---|
| `load_model` | arbitrary bytes as a `.bin` / `.ftz` file (args, dictionary, dense and quantized matrices, product quantizers) | loading fails cleanly or gives a model on which every inference call works, that saves and reloads with bit-identical predictions |
| `predict` | arbitrary bytes as text (valid and invalid UTF-8), with the tiny models of `tests/data/models` (every loss, quantized, unsupervised) | tokenization, `predict` / `predict_batch` agreement, sorted probabilities in [0, 1], `test()` on the raw bytes, vectors and subwords |
| `train` | arbitrary training-file contents (and optionally a `.vec` file), tiny settings, one epoch | training fails cleanly or gives a model that predicts, tests, saves, reloads and quantizes |

```bash
cargo install cargo-fuzz
cd fuzz
cargo +nightly fuzz run load_model -- -jobs=2 -rss_limit_mb=2048 -timeout=10 -max_len=65536
cargo +nightly fuzz run predict -- -jobs=2 -rss_limit_mb=2048 -timeout=10 -max_len=8192
cargo +nightly fuzz run train -- -jobs=2 -rss_limit_mb=2048 -timeout=10 -max_len=4096
```

Seed `load_model` with the models in `tests/data/models/` (`cp ../tests/data/models/tiny_* corpus/load_model/`).

`regressions/<target>/` holds the minimized inputs of every crash found so far. They are replayed
by `cargo test` (`tests/fuzz_regressions.rs` in the main crate) and, for model files, by
`tests/test_malformed.py`.
