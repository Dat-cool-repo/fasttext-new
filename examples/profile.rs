//! Split single-thread predict time into tokenization vs. inference.
//! `cargo run --release --example profile -- MODEL TEXT_FILE`
use std::time::Instant;

use fasttext_new::model::Model;

fn main() {
    let mut args = std::env::args().skip(1);
    let model = Model::load(args.next().expect("model path")).unwrap();
    let text = std::fs::read_to_string(args.next().expect("text file")).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let mut ids = Vec::new();
    let t = Instant::now();
    let mut n_ids = 0;
    for l in &lines {
        model.line_ids(l, &mut ids);
        n_ids += ids.len();
    }
    let tok = t.elapsed();
    let all: Vec<Vec<i32>> = lines
        .iter()
        .map(|l| {
            let mut v = Vec::new();
            model.line_ids(l, &mut v);
            v
        })
        .collect();
    let t = Instant::now();
    for v in &all {
        std::hint::black_box(model.inner().predict_on_words_raw(v, 1, 0.0));
    }
    let inf = t.elapsed();
    let t = Instant::now();
    for l in &lines {
        std::hint::black_box(model.predict(l, 1, 0.0).unwrap());
    }
    let total = t.elapsed();
    println!(
        "{} lines, {:.1} ids/line: tokenize {:?}, inference {:?}, predict total {:?} ({:.0} lines/s)",
        lines.len(),
        n_ids as f64 / lines.len() as f64,
        tok,
        inf,
        total,
        lines.len() as f64 / total.as_secs_f64()
    );
}
