//! Train on arbitrary training-file contents (tiny models, one epoch, one thread), optionally
//! with an arbitrary pretrained `.vec` file, then predict, test, save / reload and sometimes
//! quantize the result. Training must either fail cleanly or give a usable model.
#![no_main]

include!("../common/train.rs");

libfuzzer_sys::fuzz_target!(|data: &[u8]| run(data));
