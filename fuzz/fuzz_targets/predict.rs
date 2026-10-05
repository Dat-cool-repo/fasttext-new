//! Tokenize and predict arbitrary text (valid and invalid UTF-8) with small fixed models
//! (`tests/data/models/`, one per loss, quantized and unsupervised), and run `test()` on the
//! raw bytes as a labelled file.
#![no_main]

include!("../common/predict.rs");

libfuzzer_sys::fuzz_target!(|data: &[u8]| run(data));
