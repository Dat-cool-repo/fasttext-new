// Train on arbitrary training-file contents (tiny models, one epoch, one thread), optionally
// with an arbitrary pretrained `.vec` file, then predict, test, save / reload and sometimes
// quantize the result. Training must either fail cleanly or give a usable model.
// Included by `fuzz_targets/train.rs` and by the main crate's `tests/fuzz_regressions.rs`.
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use fasttext_new::model::{Args, LossName, Model, ModelName};

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ftnew-fuzz-{}-{name}", std::process::id()))
}

/// Run one fuzz input (shared by the fuzz target and `tests/fuzz_regressions.rs`).
pub fn run(data: &[u8]) {
    if data.len() < 3 {
        return;
    }
    let (cfg, body) = (&data[..3], &data[3..]);
    // Optional pretrained vectors: everything after the first 0xFF byte.
    let (train, vec) = match body.iter().position(|&b| b == 0xFF) {
        Some(i) if cfg[2] & 1 == 1 => (&body[..i], Some(&body[i + 1..])),
        _ => (body, None),
    };
    let input = tmp("train.txt");
    std::fs::write(&input, train).unwrap();
    let mut args = Args {
        input: input.clone(),
        dim: 4,
        epoch: 1,
        thread: 1,
        verbose: 0,
        ..Args::default()
    };
    args.model = [ModelName::Supervised, ModelName::Cbow, ModelName::SkipGram][cfg[0] as usize % 3];
    if args.model == ModelName::Supervised {
        args.apply_supervised_defaults();
    }
    args.loss = [
        LossName::Softmax,
        LossName::HierarchicalSoftmax,
        LossName::NegativeSampling,
        LossName::OneVsAll,
    ][(cfg[0] as usize / 3) % 4];
    args.word_ngrams = 1 + (cfg[1] % 3) as i32;
    args.minn = (cfg[1] / 3 % 3) as i32;
    args.maxn = args.minn + (cfg[1] / 9 % 3) as i32;
    args.bucket = if cfg[1] & 0x80 != 0 { 0 } else { 97 };
    args.min_count = 1;
    args.ws = 2;
    args.neg = 2;
    args.lr = 0.1;
    if cfg[2] & 2 == 2 {
        args.label = "#".into();
    }
    if let Some(v) = vec {
        let p = tmp("pre.vec");
        std::fs::write(&p, v).unwrap();
        args.pretrained_vectors = p;
    }
    let quantize = cfg[2] & 4 == 4 && args.model == ModelName::Supervised;
    let Ok(m) = Model::train(args, Arc::new(AtomicBool::new(false))) else {
        return;
    };
    if m.is_supervised() {
        let line = String::from_utf8_lossy(train.split(|&b| b == b'\n').next().unwrap_or(b""));
        let p = m.predict(&line, -1, 0.0).unwrap();
        assert!(p.len() <= m.labels().len());
        m.test_bytes(train, 1, 0.0).unwrap();
    } else {
        let _ = m.nearest_neighbors("a", 2);
    }
    let mut buf = Vec::new();
    m.inner().save(&mut buf).unwrap();
    let mut m2 = Model::load_bytes(&buf).expect("reload a trained model");
    assert_eq!(m.labels(), m2.labels());
    if quantize {
        let q = Args {
            dsub: 2,
            ..Args::default()
        };
        m2.quantize(&q).unwrap();
        let mut buf = Vec::new();
        m2.inner().save(&mut buf).unwrap();
        let q = Model::load_bytes(&buf).expect("reload a quantized model");
        let _ = q.predict("a b c", 2, 0.0).unwrap();
    }
}
