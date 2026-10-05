//! Load arbitrary bytes as a `.bin` / `.ftz` model (args, dictionary, dense and quantized
//! matrices, product quantizers). Loading must either fail cleanly or give a model that every
//! inference call can use, that saves, and that reloads with identical predictions.
#![no_main]

include!("../common/load_model.rs");

libfuzzer_sys::fuzz_target!(|data: &[u8]| run(data));
