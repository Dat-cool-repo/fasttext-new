// Load arbitrary bytes as a `.bin` / `.ftz` model (args, dictionary, dense and quantized
// matrices, product quantizers). Loading must either fail cleanly or give a model that every
// inference call can use, that saves, and that reloads with identical predictions.
// Included by `fuzz_targets/load_model.rs` and by the main crate's `tests/fuzz_regressions.rs`.
use fasttext_new::model::Model;

const TEXTS: [&str; 4] = [
    "hello world",
    "",
    "apple banana café 東京 😀",
    "__label__x a b c d e f",
];

fn bits(p: &[(u32, f32)]) -> Vec<(u32, u32)> {
    p.iter().map(|&(l, s)| (l, s.to_bits())).collect()
}

fn exercise(m: &Model) {
    let nlabels = m.labels().len();
    let dim = m.dimension();
    for t in TEXTS {
        for k in [1, 2, -1] {
            if let Ok(p) = m.predict(t, k, 0.0) {
                assert!(m.is_supervised());
                let kk = if k < 0 { nlabels } else { k as usize };
                assert!(p.len() <= kk.min(nlabels));
                assert!(p.iter().all(|&(l, _)| (l as usize) < nlabels));
            }
        }
        let _ = m.predict(t, 3, 0.2);
        assert_eq!(m.word_vector(t).len(), dim);
        if let Ok(v) = m.sentence_vector(t) {
            assert_eq!(v.len(), dim);
        }
        let _ = m.subwords(t);
        let _ = m.tokens(t);
        let _ = m.word_id(t);
        let _ = m.label_id(t);
        let _ = m.subword_id(t);
    }
    if let Ok(b) = m.predict_batch(&TEXTS, 2, 0.0) {
        for (t, r) in TEXTS.iter().zip(&b) {
            assert_eq!(bits(r), bits(&m.predict(t, 2, 0.0).unwrap()));
        }
    }
    let _ = m.input_vector(0);
    let _ = m.input_vector(-1);
    let _ = m.input_vector(i32::MAX);
    if let Ok((r, c, d)) = m.input_matrix() {
        assert_eq!(r * c, d.len());
    }
    if let Ok((r, c, d)) = m.output_matrix() {
        assert_eq!(r * c, d.len());
    }
    let _ = m.test_bytes(
        b"__label__a hello world\n__label__b x y\nno label\n",
        1,
        0.0,
    );
    // Nearest neighbours precompute an nwords x dim matrix; keep the fuzzer fast.
    if m.nwords() * dim <= 1 << 16 {
        let _ = m.nearest_neighbors("hello", 3);
        let _ = m.nearest_neighbors("hello", usize::MAX);
        let _ = m.analogies("a", "b", "c", 2);
    }
}

/// Run one fuzz input (shared by the fuzz target and `tests/fuzz_regressions.rs`).
pub fn run(data: &[u8]) {
    let Ok(m) = Model::load_bytes(data) else {
        return;
    };
    exercise(&m);
    // A loaded model saves, and the saved file loads and predicts identically.
    let mut buf = Vec::new();
    m.inner().save(&mut buf).expect("save a loaded model");
    let m2 = Model::load_bytes(&buf).expect("reload a saved model");
    assert_eq!(m.labels(), m2.labels());
    assert_eq!(m.nwords(), m2.nwords());
    if m.is_supervised() {
        for t in TEXTS {
            assert_eq!(
                bits(&m.predict(t, -1, 0.0).unwrap()),
                bits(&m2.predict(t, -1, 0.0).unwrap())
            );
        }
    }
}
