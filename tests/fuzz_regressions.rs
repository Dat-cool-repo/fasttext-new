//! Replays the minimized crash inputs of the fuzz targets (`fuzz/regressions/<target>/`)
//! through the same code as the fuzz targets (`fuzz/common/`), so every crash found by fuzzing
//! stays fixed. Runs with plain `cargo test` (no nightly, no libFuzzer).

use std::path::Path;

mod load_model {
    include!("../fuzz/common/load_model.rs");
}
mod predict {
    include!("../fuzz/common/predict.rs");
}
mod train {
    include!("../fuzz/common/train.rs");
}

fn replay(target: &str, run: fn(&[u8])) -> usize {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz/regressions")
        .join(target);
    let mut n = 0;
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        if path.is_file() {
            eprintln!("replaying {}", path.display());
            run(&std::fs::read(&path).unwrap());
            n += 1;
        }
    }
    n
}

#[test]
fn load_model_regressions() {
    assert!(replay("load_model", load_model::run) > 0);
}

#[test]
fn predict_regressions() {
    replay("predict", predict::run);
}

#[test]
fn train_regressions() {
    replay("train", train::run);
}

/// A model with NaN word vectors: the nearest-neighbour sort used a comparator that is not a
/// total order with NaN (which panics in `sort_by` since Rust 1.81). NaN scores now rank last.
#[test]
fn nearest_neighbors_with_nan_vectors() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz/regressions/load_model/nan_vectors_nn_sort");
    let m = fasttext_new::model::Model::load_bytes(&std::fs::read(path).unwrap()).unwrap();
    for word in ["apple", "guitar", "zzz"] {
        let nn = m.nearest_neighbors(word, m.nwords());
        let first_nan = nn.iter().position(|(s, _)| s.is_nan()).unwrap_or(nn.len());
        assert!(
            nn[first_nan..].iter().all(|(s, _)| s.is_nan()),
            "NaN scores must come last"
        );
        assert!(
            nn[..first_nan].windows(2).all(|w| w[0].0 >= w[1].0),
            "scores must be sorted"
        );
    }
}

/// The seed models themselves go through the same checks.
#[test]
fn committed_models_pass_the_load_model_checks() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/models");
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("tiny_")
        {
            load_model::run(&std::fs::read(&path).unwrap());
        }
    }
}
