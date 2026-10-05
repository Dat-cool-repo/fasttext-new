// Tokenize and predict arbitrary text (valid and invalid UTF-8) with small fixed models
// (`tests/data/models/`, one per loss, quantized and unsupervised), and run `test()` on the
// raw bytes as a labelled file.
// Included by `fuzz_targets/predict.rs` and by the main crate's `tests/fuzz_regressions.rs`.
use std::sync::OnceLock;

use fasttext_new::model::{Error, Model, tokenize};

macro_rules! models {
    ($($name:literal),*) => {
        [$(include_bytes!(concat!("../../tests/data/models/", $name)).as_slice()),*]
    };
}

fn models() -> &'static [Model] {
    static M: OnceLock<Vec<Model>> = OnceLock::new();
    M.get_or_init(|| {
        models!(
            "tiny_softmax.bin",
            "tiny_hs.bin",
            "tiny_ova.bin",
            "tiny_ns.bin",
            "tiny_nosub.bin",
            "tiny_softmax_q.ftz",
            "tiny_hs_q.ftz",
            "tiny_hs_qout.ftz",
            "tiny_ova_qout.ftz",
            "tiny_cbow.bin"
        )
        .iter()
        .map(|b| Model::load_bytes(b).expect("fixture model"))
        .collect()
    })
}

fn check_preds(m: &Model, p: &[(u32, f32)], k: i32, threshold: f32) {
    let n = m.labels().len();
    let kk = if k < 0 { n } else { k as usize };
    assert!(p.len() <= kk.min(n), "{} > {}", p.len(), kk);
    for w in p.windows(2) {
        assert!(w[0].1 >= w[1].1, "not sorted: {p:?}");
    }
    for &(l, s) in p {
        assert!((l as usize) < n);
        assert!((0.0..=1.0 + 1e-4).contains(&s), "probability {s}");
        if threshold > 0.0 {
            assert!(s >= threshold * (1.0 - 1e-6), "{s} < threshold {threshold}");
        }
    }
}

/// Run one fuzz input (shared by the fuzz target and `tests/fuzz_regressions.rs`).
pub fn run(data: &[u8]) {
    if data.len() < 2 {
        return;
    }
    let ms = models();
    let m = &ms[data[0] as usize % ms.len()];
    let k = [1, 2, 5, -1][(data[0] as usize / ms.len()) % 4];
    let threshold = [0.0f32, 0.1, 0.5, -1.0][data[1] as usize % 4];
    let raw = &data[2..];
    let text = String::from_utf8_lossy(raw);

    let toks = tokenize(&text);
    assert!(toks.iter().all(|t| !t.is_empty()));
    let _ = m.tokens(&text);
    let mut ids = Vec::new();
    m.line_ids(&text, &mut ids);

    let lines: Vec<&str> = text.split('\n').collect();
    match m.predict(&text, k, threshold) {
        Ok(p) => {
            assert!(m.is_supervised() && lines.len() == 1);
            check_preds(m, &p, k, threshold);
        }
        Err(Error::NotSupervised) => assert!(!m.is_supervised()),
        Err(Error::Newline) => assert!(lines.len() > 1),
        Err(e) => panic!("unexpected error {e}"),
    }
    if m.is_supervised() {
        let batch = m.predict_batch(&lines, k, threshold).unwrap();
        for (l, b) in lines.iter().zip(&batch) {
            let p = m.predict(l, k, threshold).unwrap();
            assert_eq!(
                p.iter().map(|x| (x.0, x.1.to_bits())).collect::<Vec<_>>(),
                b.iter().map(|x| (x.0, x.1.to_bits())).collect::<Vec<_>>()
            );
        }
        let r = m.test_bytes(raw, k.max(1), threshold).unwrap();
        assert!(r.total.predicted_gold <= r.total.predicted.min(r.total.gold));
    }
    for l in &lines {
        if let Ok(v) = m.sentence_vector(l) {
            assert_eq!(v.len(), m.dimension());
        }
    }
    for t in toks.iter().take(32) {
        assert_eq!(m.word_vector(t).len(), m.dimension());
        let (s, i) = m.subwords(t);
        assert!(s.len() >= i.len());
    }
}
