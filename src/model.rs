//! Pure-Rust core used by the Python bindings.
//!
//! This is a thin layer over the pure-Rust [`fasttext`] crate (messense/fasttext-rs).
//! The crate does the heavy lifting (model file parsing incl. product-quantized `.ftz`,
//! hierarchical softmax / softmax / one-vs-all / negative-sampling output layers).
//!
//! What this module adds is *bit-for-bit compatibility with the C++ Python bindings*:
//!
//! * The C++ `predict()` appends `"\n"` to the text and runs `Dictionary::getLine`, so the
//!   end-of-sentence token `</s>` is always part of the input **and** of the word n-gram
//!   hashes. The crate's `predict()` appends `</s>` after the word n-grams and returns
//!   nothing for empty text. We re-implement `getLine` here (same whitespace set, same
//!   ordering of ids) and call the crate's lower-level `predict_on_words`.
//! * `k = -1` means "all labels" (C++ `kAllLabelsAsTarget`).
//! * Word / sentence vectors are accumulated in the same order and normalised the same way
//!   (`sum`, then multiply by `1.0 / n`) as C++, for dense and quantized models (via the
//!   `add_input_rows` helper added to the vendored crate).
//!
//! Training, quantization and saving are delegated to the crate; [`Model::test`] is a port of
//! C++ `FastText::test` + `Meter` on top of the C++-compatible tokenization above, so its
//! numbers match the C++ package exactly for the same model file.

use std::borrow::Cow;
use std::fmt;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

use fasttext::FastText;
pub use fasttext::args::{Args, LossName, ModelName};
pub use fasttext::dictionary::Entry;
use fasttext::dictionary::EntryType;
use fasttext::matrix::{DenseMatrix, Matrix};
use rayon::prelude::*;

/// End-of-sentence token used by fastText.
pub const EOS: &str = "</s>";

/// Errors surfaced to Python.
#[derive(Debug)]
pub enum Error {
    /// The model file could not be read or parsed.
    Load(String),
    /// `predict` was called on an unsupervised (cbow / skipgram) model.
    NotSupervised,
    /// The input text contains a newline (C++ fastText processes one line at a time).
    Newline,
    /// `k` is 0 or below -1 (C++: "k needs to be 1 or higher!").
    InvalidK,
    /// The operation is not available for this kind of model (e.g. matrix of a quantized model).
    Unsupported(&'static str),
    /// Training / quantization / saving / testing failed (I/O or invalid arguments).
    Failed(String),
}

impl From<fasttext::FastTextError> for Error {
    fn from(e: fasttext::FastTextError) -> Self {
        Error::Failed(e.to_string())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Load(msg) => write!(f, "{msg}"),
            Error::NotSupervised => write!(f, "Model needs to be supervised for prediction!"),
            Error::Newline => write!(f, "predict processes one line at a time (remove '\\n')"),
            Error::InvalidK => write!(f, "k needs to be 1 or higher!"),
            Error::Unsupported(msg) => write!(f, "{msg}"),
            Error::Failed(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// A single prediction: index into [`Model::labels`] and its probability.
pub type Pred = (u32, f32);

/// C++ `Dictionary::hash`: 32-bit FNV-1a over *signed* bytes.
#[inline]
pub fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 2_166_136_261;
    for &b in bytes {
        h ^= b as i8 as i32 as u32;
        h = h.wrapping_mul(16_777_619);
    }
    h
}

/// Whitespace set of C++ `Dictionary::readWord`.
#[inline]
fn is_tok_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t' | 0x0b | 0x0c | 0)
}

/// C++ `fasttext.tokenize`: tokens split on the `readWord` whitespace set; each `\n` yields
/// `</s>`.
pub fn tokenize(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut start = None;
    for (i, &b) in bytes.iter().enumerate() {
        if is_tok_ws(b) {
            if let Some(s) = start.take() {
                out.push(text[s..i].to_string());
            }
            if b == b'\n' {
                out.push(EOS.to_string());
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        out.push(text[s..].to_string());
    }
    out
}

/// Character n-grams of C++ `Dictionary::computeSubwords` (UTF-8 aware, no 1-grams at the
/// `<` / `>` boundaries).
fn char_ngrams(word: &str, minn: i32, maxn: i32) -> Vec<String> {
    let b = word.as_bytes();
    let mut out = Vec::new();
    for i in 0..b.len() {
        if (b[i] & 0xC0) == 0x80 {
            continue;
        }
        let (mut j, mut n) = (i, 1);
        while j < b.len() && n <= maxn {
            j += 1;
            while j < b.len() && (b[j] & 0xC0) == 0x80 {
                j += 1;
            }
            if n >= minn && !(n == 1 && (i == 0 || j == b.len())) {
                out.push(word[i..j].to_string());
            }
            n += 1;
        }
    }
    out
}

/// Whitespace set of `std::istream >> std::string` in the "C" locale (`isspace`).
#[inline]
fn is_c_space(c: char) -> bool {
    matches!(c, ' ' | '\n' | '\r' | '\t' | '\x0b' | '\x0c')
}

/// Counters of C++ `Meter::Metrics` (overall or for one label).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    pub gold: u64,
    pub predicted: u64,
    pub predicted_gold: u64,
}

impl Counts {
    fn merge(&mut self, o: &Counts) {
        self.gold += o.gold;
        self.predicted += o.predicted;
        self.predicted_gold += o.predicted_gold;
    }

    /// `predictedGold / predicted` (NaN if nothing was predicted, like C++).
    pub fn precision(&self) -> f64 {
        if self.predicted == 0 {
            return f64::NAN;
        }
        self.predicted_gold as f64 / self.predicted as f64
    }

    /// `predictedGold / gold` (NaN if there is no gold label, like C++).
    pub fn recall(&self) -> f64 {
        if self.gold == 0 {
            return f64::NAN;
        }
        self.predicted_gold as f64 / self.gold as f64
    }

    /// `2 * predictedGold / (predicted + gold)` (NaN if both are 0, like C++).
    pub fn f1(&self) -> f64 {
        if self.predicted + self.gold == 0 {
            return f64::NAN;
        }
        2.0 * self.predicted_gold as f64 / (self.predicted + self.gold) as f64
    }
}

/// Result of [`Model::test`]: C++ `Meter` without the score-vs-true curves.
#[derive(Debug, Clone, PartialEq)]
pub struct TestResult {
    /// Number of examples (lines with at least one known label and one word).
    pub nexamples: u64,
    /// Totals over all labels (`precision()` / `recall()` are P@k / R@k).
    pub total: Counts,
    /// Per-label counters, indexed by label id.
    pub labels: Vec<Counts>,
}

impl TestResult {
    fn new(nlabels: usize) -> Self {
        TestResult {
            nexamples: 0,
            total: Counts::default(),
            labels: vec![Counts::default(); nlabels],
        }
    }

    /// C++ `Meter::log`.
    fn log(&mut self, gold: &[i32], preds: &[Pred]) {
        self.nexamples += 1;
        self.total.gold += gold.len() as u64;
        self.total.predicted += preds.len() as u64;
        for &(lid, _) in preds {
            let c = &mut self.labels[lid as usize];
            c.predicted += 1;
            if gold.contains(&(lid as i32)) {
                c.predicted_gold += 1;
                self.total.predicted_gold += 1;
            }
        }
        for &lid in gold {
            self.labels[lid as usize].gold += 1;
        }
    }

    fn merge(mut self, o: TestResult) -> Self {
        self.nexamples += o.nexamples;
        self.total.merge(&o.total);
        for (a, b) in self.labels.iter_mut().zip(&o.labels) {
            a.merge(b);
        }
        self
    }
}

/// A loaded or trained fastText model.
pub struct Model {
    ft: FastText,
    supervised: bool,
    nwords: i32,
    labels: Vec<String>,
    /// Some dictionary entry is not valid UTF-8 (see [`Entry::raw`]).
    has_raw: bool,
    /// Lazily computed normalised word vectors for nearest-neighbour / analogy queries
    /// (C++ `lazyComputeWordVectors`).
    word_vectors: OnceLock<DenseMatrix>,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("dim", &self.dimension())
            .field("nwords", &self.nwords)
            .field("nlabels", &self.labels.len())
            .field("quantized", &self.is_quantized())
            .finish()
    }
}

impl Model {
    /// Load a `.bin` (dense) or `.ftz` (quantized) model file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let ft = FastText::load_model(path)
            .map_err(|e| Error::Load(format!("{}: {e}", path.display())))?;
        Ok(Self::from_fasttext(ft))
    }

    /// Wrap an already-loaded crate model.
    pub fn from_fasttext(ft: FastText) -> Self {
        let supervised = ft.args().model == ModelName::Supervised;
        let nwords = ft.dict().nwords();
        let (labels, _) = ft.get_labels();
        let has_raw = ft.dict().words().iter().any(|e| e.raw.is_some());
        Model {
            ft,
            supervised,
            nwords,
            labels,
            has_raw,
            word_vectors: OnceLock::new(),
        }
    }

    /// Train a model (C++ `fasttext::FastText::train`, or autotune when
    /// `args.autotune_validation_file` is set). Setting `abort` stops training early.
    pub fn train(args: Args, abort: Arc<AtomicBool>) -> Result<Self> {
        if !args.input.is_file() {
            return Err(Error::Failed(format!(
                "{} cannot be opened for training!",
                args.input.display()
            )));
        }
        let ft = if args.has_autotune() {
            fasttext::autotune::Autotune::run(args)?
        } else {
            FastText::train_with_abort(args, abort)?
        };
        Ok(Self::from_fasttext(ft))
    }

    /// Quantize in place (C++ `FastText::quantize`). `qargs` supplies `input`, `qout`,
    /// `cutoff`, `retrain`, `epoch`, `lr`, `thread`, `verbose`, `dsub` and `qnorm`.
    pub fn quantize(&mut self, qargs: &Args) -> Result<()> {
        if !self.supervised {
            return Err(Error::Failed(
                "For now we only support quantization of supervised models".into(),
            ));
        }
        if self.is_quantized() {
            return Err(Error::Failed("The model is already quantized".into()));
        }
        if qargs.retrain && qargs.input.as_os_str().is_empty() {
            return Err(Error::Failed("Need input file path if retraining".into()));
        }
        self.ft.quantize(qargs)?;
        // The dictionary may have been pruned; labels are unchanged.
        self.nwords = self.ft.dict().nwords();
        self.has_raw = self.ft.dict().words().iter().any(|e| e.raw.is_some());
        self.word_vectors = OnceLock::new();
        Ok(())
    }

    /// Save as a C++-compatible `.bin` (or `.ftz` once quantized).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        self.ft.save_model(path).map_err(|e| {
            Error::Failed(format!(
                "{} cannot be opened for saving! ({e})",
                path.display()
            ))
        })
    }

    /// Training / model arguments.
    pub fn args(&self) -> &Args {
        self.ft.args()
    }

    /// C++ `FastText::test` on a labelled file: every line is predicted with
    /// `predict(k, threshold)` and logged into a `Meter`. Lines are processed in parallel.
    pub fn test(&self, path: impl AsRef<Path>, k: i32, threshold: f32) -> Result<TestResult> {
        if !self.supervised {
            return Err(Error::NotSupervised);
        }
        let k = self.check_k(k)?;
        let data = std::fs::read(path.as_ref())
            .map_err(|_| Error::Failed("Test file cannot be opened!".into()))?;
        let nlabels = self.labels.len();
        // Every '\n' ends a C++ line, so chunks that end right after a '\n' are independent.
        let target = (data.len() / (4 * rayon::current_num_threads().max(1))).max(1 << 16);
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < data.len() {
            let mut end = (start + target).min(data.len());
            while end < data.len() && data[end - 1] != b'\n' {
                end += 1;
            }
            chunks.push(&data[start..end]);
            start = end;
        }
        Ok(chunks
            .par_iter()
            .map(|chunk| {
                let mut res = TestResult::new(nlabels);
                let mut eval = |ids: &[i32], gold: &[i32]| {
                    if !gold.is_empty() && !ids.is_empty() {
                        res.log(gold, &self.predict_ids(ids, k, threshold));
                    }
                };
                self.for_each_example(chunk, &mut eval);
                res
            })
            .reduce(|| TestResult::new(nlabels), TestResult::merge))
    }

    /// Iterate over the examples of a labelled text like C++ `Dictionary::getLine` on a
    /// stream: the tokens up to a `\n` (read as `</s>`) or a literal `</s>` token form one
    /// example; a trailing example without `\n` gets no `</s>`. Invalid UTF-8 inside a token
    /// is replaced by U+FFFD (the crate's training tokenizer does the same).
    fn for_each_example(&self, data: &[u8], f: &mut impl FnMut(&[i32], &[i32])) {
        let dict = self.ft.dict();
        let ngrams = self.ft.args().word_ngrams;
        let (mut ids, mut gold, mut hashes) = (Vec::new(), Vec::new(), Vec::new());
        let mut ntokens = 0usize;
        let n = data.len();
        let mut i = 0;
        loop {
            while i < n && data[i] != b'\n' && is_tok_ws(data[i]) {
                i += 1;
            }
            if i >= n {
                if ntokens > 0 {
                    dict.add_word_ngrams(&mut ids, &hashes, ngrams);
                    f(&ids, &gold);
                }
                return;
            }
            let token: Cow<'_, str> = if data[i] == b'\n' {
                i += 1;
                Cow::Borrowed(EOS)
            } else {
                let start = i;
                while i < n && !is_tok_ws(data[i]) {
                    i += 1;
                }
                String::from_utf8_lossy(&data[start..i])
            };
            ntokens += 1;
            self.push_token(&token, &mut ids, &mut hashes, Some(&mut gold));
            if token == EOS {
                dict.add_word_ngrams(&mut ids, &hashes, ngrams);
                f(&ids, &gold);
                ids.clear();
                gold.clear();
                hashes.clear();
                ntokens = 0;
            }
        }
    }

    /// One token of C++ `Dictionary::getLine`: a word adds its subword ids (and its hash for
    /// the word n-grams); a label in the dictionary adds its label id.
    #[inline]
    fn push_token(
        &self,
        token: &str,
        out: &mut Vec<i32>,
        hashes: &mut Vec<i32>,
        labels: Option<&mut Vec<i32>>,
    ) {
        let dict = self.ft.dict();
        let h = fnv1a(token.as_bytes());
        let wid = dict.get_id_with_hash(token, h).unwrap_or(-1);
        let ty = if wid < 0 {
            dict.get_type_from_str(token)
        } else {
            dict.get_type_by_id(wid)
        };
        if ty == EntryType::Word {
            dict.add_subwords(out, token, wid);
            hashes.push(h as i32);
        } else if let Some(labels) = labels {
            if wid >= 0 {
                labels.push(wid - self.nwords);
            }
        }
    }

    /// Access the underlying crate model.
    pub fn inner(&self) -> &FastText {
        &self.ft
    }

    pub fn dimension(&self) -> usize {
        self.ft.get_dimension() as usize
    }

    pub fn is_quantized(&self) -> bool {
        self.ft.is_quant()
    }

    pub fn is_supervised(&self) -> bool {
        self.supervised
    }

    /// Dictionary entries: `nwords` words, then the labels.
    pub fn entries(&self) -> &[Entry] {
        self.ft.dict().words()
    }

    /// Number of words (label `i` is entry `nwords + i`).
    pub fn nwords(&self) -> usize {
        self.nwords as usize
    }

    /// Whether some dictionary entry is not valid UTF-8 (its `word` is a lossy copy).
    pub fn has_raw(&self) -> bool {
        self.has_raw
    }

    /// The entry whose (lossy) string is `word`, preferring an exact match.
    pub fn entry_for(&self, word: &str) -> Option<&Entry> {
        let dict = self.ft.dict();
        match dict.get_id(word) {
            Some(id) => Some(&dict.words()[id as usize]),
            None if self.has_raw => dict
                .words()
                .iter()
                .find(|e| e.raw.is_some() && e.word == word),
            None => None,
        }
    }

    /// Label strings in dictionary order (index = label id).
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Vocabulary words and their counts.
    pub fn words(&self) -> (Vec<String>, Vec<i64>) {
        self.ft.get_vocab()
    }

    /// Labels and their counts.
    pub fn labels_with_freq(&self) -> (Vec<String>, Vec<i64>) {
        self.ft.get_labels()
    }

    /// `(model, loss, word_ngrams, minn, maxn, bucket)` summary of the training args.
    pub fn args_summary(&self) -> (String, String, i32, i32, i32, i32) {
        let a = self.ft.args();
        let loss = match a.loss {
            LossName::HierarchicalSoftmax => "hs",
            LossName::NegativeSampling => "ns",
            LossName::Softmax => "softmax",
            LossName::OneVsAll => "ova",
        };
        (
            a.model.to_string(),
            loss.to_string(),
            a.word_ngrams,
            a.minn,
            a.maxn,
            a.bucket,
        )
    }

    pub fn word_id(&self, word: &str) -> i32 {
        self.ft.get_word_id(word).unwrap_or(-1)
    }

    pub fn label_id(&self, label: &str) -> i32 {
        // Same as C++ `FastText::getLabelId` (no check that the entry really is a label).
        match self.ft.dict().get_id(label) {
            Some(id) => id - self.nwords,
            None => -1,
        }
    }

    /// C++ `Dictionary::getLine` applied to `text + "\n"`: returns the input-matrix row ids
    /// (word ids / char n-gram buckets, `</s>`, then word n-gram buckets).
    pub fn line_ids(&self, text: &str, out: &mut Vec<i32>) {
        let dict = self.ft.dict();
        out.clear();
        let mut hashes: Vec<i32> = Vec::new();
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut i = 0;
        loop {
            while i < n && bytes[i] != b'\n' && is_tok_ws(bytes[i]) {
                i += 1;
            }
            let token = if i >= n || bytes[i] == b'\n' {
                // A newline (or the implicit trailing "\n") with an empty buffer is </s>.
                EOS
            } else {
                let start = i;
                while i < n && !is_tok_ws(bytes[i]) {
                    i += 1;
                }
                // Splitting on ASCII bytes keeps UTF-8 boundaries intact.
                &text[start..i]
            };
            self.push_token(token, out, &mut hashes, None);
            if token == EOS {
                break;
            }
        }
        dict.add_word_ngrams(out, &hashes, self.ft.args().word_ngrams);
    }

    /// Predict the top-`k` labels for one line of text.
    ///
    /// `k = -1` returns all labels (subject to `threshold`; C++ semantics). Returns `(label_index, prob)` pairs
    /// sorted by decreasing probability.
    pub fn predict(&self, text: &str, k: i32, threshold: f32) -> Result<Vec<Pred>> {
        let mut ids = Vec::with_capacity(64);
        self.predict_with_buf(text, k, threshold, &mut ids)
    }

    fn predict_with_buf(
        &self,
        text: &str,
        k: i32,
        threshold: f32,
        ids: &mut Vec<i32>,
    ) -> Result<Vec<Pred>> {
        if !self.supervised {
            return Err(Error::NotSupervised);
        }
        if text.as_bytes().contains(&b'\n') {
            return Err(Error::Newline);
        }
        let k = self.check_k(k)?;
        self.line_ids(text, ids);
        Ok(self.predict_ids(ids, k, threshold))
    }

    /// C++ `Model::predict`: k == -1 means "as many as output-matrix rows" (= nlabels).
    fn check_k(&self, k: i32) -> Result<usize> {
        match k {
            -1 => Ok(self.labels.len()),
            k if k <= 0 => Err(Error::InvalidK),
            k => Ok(k as usize),
        }
    }

    /// Predict from input-matrix row ids (as produced by [`Model::line_ids`]).
    fn predict_ids(&self, ids: &[i32], k: usize, threshold: f32) -> Vec<Pred> {
        self.ft
            .predict_on_words_raw(ids, k, threshold)
            .into_iter()
            .map(|(log_prob, idx)| (idx as u32, log_prob.exp()))
            .collect()
    }

    /// Predict many lines in parallel on the rayon thread pool.
    pub fn predict_batch<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        k: i32,
        threshold: f32,
    ) -> Result<Vec<Vec<Pred>>> {
        if !self.supervised {
            return Err(Error::NotSupervised);
        }
        texts
            .par_iter()
            .map_init(
                || Vec::with_capacity(64),
                |ids, t| self.predict_with_buf(t.as_ref(), k, threshold, ids),
            )
            .collect()
    }

    /// Sum input rows (dense or quantized) for `ids` into `out`, C++ `addInputVector` order.
    fn add_rows(&self, ids: &[i32], out: &mut [f32]) {
        self.ft.add_input_rows(ids, out);
    }

    /// C++ `FastText::getWordVector`.
    pub fn word_vector(&self, word: &str) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dimension()];
        self.word_vector_into(word, &mut out);
        out
    }

    fn word_vector_into(&self, word: &str, out: &mut [f32]) {
        out.fill(0.0);
        let ids = self.ft.dict().get_subwords_for_string(word);
        if ids.is_empty() {
            return;
        }
        self.add_rows(&ids, out);
        let s = (1.0f64 / ids.len() as f64) as f32;
        out.iter_mut().for_each(|o| *o *= s);
    }

    /// C++ `FastText::getSentenceVector` (applied to `text + "\n"` like the Python binding).
    pub fn sentence_vector(&self, text: &str) -> Result<Vec<f32>> {
        if text.as_bytes().contains(&b'\n') {
            return Err(Error::Newline);
        }
        let dim = self.dimension();
        let mut out = vec![0.0f32; dim];
        if self.supervised {
            let mut ids = Vec::new();
            self.line_ids(text, &mut ids);
            self.add_rows(&ids, &mut out);
            if !ids.is_empty() {
                let s = (1.0f64 / ids.len() as f64) as f32;
                out.iter_mut().for_each(|o| *o *= s);
            }
        } else {
            let mut wv = vec![0.0f32; dim];
            let mut count = 0usize;
            for word in text.split(is_c_space).filter(|w| !w.is_empty()) {
                self.word_vector_into(word, &mut wv);
                let norm = wv.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 {
                    let s = (1.0f64 / norm as f64) as f32;
                    for (o, v) in out.iter_mut().zip(&wv) {
                        *o += v * s;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                let s = (1.0f64 / count as f64) as f32;
                out.iter_mut().for_each(|o| *o *= s);
            }
        }
        Ok(out)
    }

    /// C++ `FastText::getSubwordId`: input-matrix row of a character n-gram.
    pub fn subword_id(&self, subword: &str) -> Result<i32> {
        let bucket = self.ft.args().bucket;
        if bucket <= 0 {
            return Err(Error::Unsupported(
                "the model has no subword buckets (bucket = 0)",
            ));
        }
        Ok(self.nwords + (fnv1a(subword.as_bytes()) % bucket as u32) as i32)
    }

    /// C++ `FastText::getInputVector`: row `id` of the (dense or quantized) input matrix.
    pub fn input_vector(&self, id: i32) -> Result<Vec<f32>> {
        let rows = if self.is_quantized() {
            self.ft.quant_input().map_or(0, |q| q.rows())
        } else {
            self.ft.input_matrix().rows()
        };
        if id < 0 || id as i64 >= rows {
            return Err(Error::Unsupported("input vector index out of range"));
        }
        let mut out = vec![0.0f32; self.dimension()];
        self.add_rows(&[id], &mut out);
        Ok(out)
    }

    /// C++ `Dictionary::getSubwords(word, ngrams, substrings)`: the word itself (if in the
    /// vocabulary) and its character n-gram strings, plus the input-matrix ids. Like C++, the
    /// strings list every n-gram while the ids skip n-grams pruned from a quantized model.
    pub fn subwords(&self, word: &str) -> (Vec<String>, Vec<i32>) {
        let dict = self.ft.dict();
        let mut strings = Vec::new();
        let mut ids = Vec::new();
        if let Some(id) = dict.get_id(word) {
            strings.push(word.to_string());
            ids.push(id);
        }
        if word != EOS {
            let marked = format!("<{word}>");
            let (minn, maxn) = (self.ft.args().minn, self.ft.args().maxn);
            if maxn > 0 && self.ft.args().bucket > 0 {
                strings.extend(char_ngrams(&marked, minn, maxn));
            }
            dict.compute_subwords(&marked, &mut ids);
        }
        (strings, ids)
    }

    /// Nearest neighbours by cosine similarity: `(score, word)`. The normalised word-vector
    /// matrix is computed on first use and cached, like C++.
    pub fn nearest_neighbors(&self, word: &str, k: usize) -> Vec<(f32, String)> {
        let wv = self
            .word_vectors
            .get_or_init(|| self.ft.precompute_word_vectors());
        self.ft.get_nn_with_word_vectors(wv, word, k)
    }

    /// Analogies `A - B + C`: `(score, word)` (C++ `getAnalogies`).
    pub fn analogies(&self, a: &str, b: &str, c: &str, k: usize) -> Vec<(f32, String)> {
        let wv = self
            .word_vectors
            .get_or_init(|| self.ft.precompute_word_vectors());
        self.ft.get_analogies_with_word_vectors(wv, a, b, c, k)
    }

    /// C++ `getLine` as exposed by the Python binding on `text + "\n"`: `(words, labels)` up
    /// to and including the first `</s>` (which counts as a word). Labels are listed only if
    /// they are in the dictionary.
    pub fn tokens(&self, text: &str) -> (Vec<String>, Vec<String>) {
        let dict = self.ft.dict();
        let (mut words, mut labels) = (Vec::new(), Vec::new());
        let tokens = text.split(|c: char| c.is_ascii() && is_tok_ws(c as u8));
        for token in tokens.filter(|t| !t.is_empty()).chain(std::iter::once(EOS)) {
            let wid = dict.get_id(token).unwrap_or(-1);
            let ty = if wid < 0 {
                dict.get_type_from_str(token)
            } else {
                dict.get_type_by_id(wid)
            };
            if ty == EntryType::Word {
                words.push(token.to_string());
            } else if wid >= 0 {
                labels.push(token.to_string());
            }
            if token == EOS {
                break;
            }
        }
        (words, labels)
    }

    /// Dense input matrix as `(rows, cols, row-major data)`.
    pub fn input_matrix(&self) -> Result<(usize, usize, &[f32])> {
        if self.is_quantized() {
            return Err(Error::Unsupported(
                "Can't get quantized Matrix (input matrix of a quantized model)",
            ));
        }
        let m = self.ft.input_matrix();
        let cols = self.dimension();
        Ok((m.data().len() / cols.max(1), cols, m.data()))
    }

    /// Dense output matrix as `(rows, cols, row-major data)`.
    pub fn output_matrix(&self) -> Result<(usize, usize, &[f32])> {
        if self.is_quantized() && self.ft.quant_output().is_some() {
            return Err(Error::Unsupported(
                "Can't get quantized Matrix (output matrix of a qout model)",
            ));
        }
        let m = self.ft.output_matrix();
        let cols = self.dimension();
        Ok((m.data().len() / cols.max(1), cols, m.data()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn fnv1a_matches_cpp_reference_values() {
        // Values computed with the C++ `Dictionary::hash` (signed-byte FNV-1a).
        assert_eq!(fnv1a(b""), 2_166_136_261);
        assert_eq!(fnv1a(b"a"), 0xe40c292c);
        // Non-ASCII bytes are sign-extended (differs from textbook FNV-1a).
        let textbook = {
            let mut h: u32 = 2_166_136_261;
            for &b in "é".as_bytes() {
                h ^= b as u32;
                h = h.wrapping_mul(16_777_619);
            }
            h
        };
        assert_ne!(fnv1a("é".as_bytes()), textbook);
    }

    #[test]
    fn whitespace_sets() {
        for b in [b' ', b'\n', b'\r', b'\t', 0x0b, 0x0c, 0] {
            assert!(is_tok_ws(b));
        }
        assert!(!is_tok_ws(b'a'));
        assert!(!is_c_space('\0'));
        assert!(!is_c_space('\u{a0}'));
    }

    /// Model fixtures are optional (not checked in). They are looked up in `$FASTTEXT_NEW_DATA`
    /// (default: `data/` in the repository), then in `tests/data/`.
    fn fixture(name: &str) -> Option<PathBuf> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let data = std::env::var_os("FASTTEXT_NEW_DATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("data"));
        let dirs = [data, root.join("tests/data")];
        dirs.into_iter().map(|d| d.join(name)).find(|p| p.exists())
    }

    #[test]
    fn hash_agrees_with_crate_lookup() {
        let Some(p) = fixture("lid.176.ftz") else {
            eprintln!("skipping: lid.176.ftz not found");
            return;
        };
        let m = Model::load(p).unwrap();
        let (words, _) = m.words();
        let dict = m.inner().dict();
        for (i, w) in words.iter().enumerate().step_by(97) {
            assert_eq!(
                dict.get_id_with_hash(w, fnv1a(w.as_bytes())),
                Some(i as i32)
            );
        }
    }

    #[test]
    fn lid_ftz_predicts_languages() {
        let Some(p) = fixture("lid.176.ftz") else {
            eprintln!("skipping: lid.176.ftz not found");
            return;
        };
        let m = Model::load(p).unwrap();
        assert!(m.is_quantized());
        assert_eq!(m.dimension(), 16);
        let cases = [
            ("Bonjour tout le monde, comment allez-vous ?", "__label__fr"),
            ("The quick brown fox jumps over the lazy dog", "__label__en"),
            (
                "Der schnelle braune Fuchs springt über den faulen Hund",
                "__label__de",
            ),
            (
                "Быстрая коричневая лиса прыгает через ленивую собаку",
                "__label__ru",
            ),
            ("我们今天去公园散步，天气非常好。", "__label__zh"),
        ];
        for (text, want) in cases {
            let p = m.predict(text, 1, 0.0).unwrap();
            assert_eq!(m.labels()[p[0].0 as usize], want, "{text}");
            assert!(p[0].1 > 0.5);
        }
        // Empty text still predicts from </s> (C++ behaviour).
        assert_eq!(m.predict("", 2, 0.0).unwrap().len(), 2);
        // k = -1 returns everything above the (HS-pruned) threshold, sorted.
        let all = m.predict("hello world", -1, 0.0).unwrap();
        assert!(all.len() > 100);
        assert!(all.windows(2).all(|w| w[0].1 >= w[1].1));
        // threshold filters
        assert!(
            m.predict("hello world", -1, 0.05)
                .unwrap()
                .iter()
                .all(|p| p.1 >= 0.05)
        );
        assert!(matches!(m.predict("a\nb", 1, 0.0), Err(Error::Newline)));
        assert!(matches!(m.predict("a", 0, 0.0), Err(Error::InvalidK)));
        assert!(matches!(m.predict("a", -2, 0.0), Err(Error::InvalidK)));
        // batch == sequential
        let texts: Vec<&str> = cases.iter().map(|c| c.0).collect();
        let batch = m.predict_batch(&texts, 3, 0.0).unwrap();
        for (t, b) in texts.iter().zip(&batch) {
            assert_eq!(&m.predict(t, 3, 0.0).unwrap(), b);
        }
    }

    #[test]
    fn line_ids_append_eos_and_skip_labels() {
        let Some(p) = fixture("lid.176.ftz") else {
            eprintln!("skipping: lid.176.ftz not found");
            return;
        };
        let m = Model::load(p).unwrap();
        let eos = m.word_id(EOS);
        assert!(eos >= 0);
        let mut ids = Vec::new();
        m.line_ids("", &mut ids);
        assert_eq!(ids, vec![eos]);
        m.line_ids("  \t ", &mut ids);
        assert_eq!(ids, vec![eos]);
        // label tokens contribute nothing
        m.line_ids("__label__en", &mut ids);
        assert_eq!(ids, vec![eos]);
        // tokens after a literal </s> are ignored, like C++ getLine
        let mut a = Vec::new();
        m.line_ids("hello", &mut a);
        m.line_ids("hello </s> world", &mut ids);
        assert_eq!(a, ids);
        assert_eq!(*a.last().unwrap(), eos);
    }

    #[test]
    fn vectors_have_model_dimension() {
        let Some(p) = fixture("lid.176.ftz") else {
            eprintln!("skipping: lid.176.ftz not found");
            return;
        };
        let m = Model::load(p).unwrap();
        assert_eq!(m.word_vector("hello").len(), 16);
        assert_eq!(m.sentence_vector("hello world").unwrap().len(), 16);
        let (subs, ids) = m.subwords("hello");
        // pruned model: every n-gram string, but only the surviving ids (C++ behaviour)
        assert!(subs.len() >= ids.len());
        assert!(!ids.is_empty());
        assert!(m.input_matrix().is_err());
    }

    #[test]
    fn tokenize_matches_cpp_read_word() {
        assert_eq!(tokenize("a b\tc\n\nd"), ["a", "b", "c", EOS, EOS, "d"]);
        assert_eq!(tokenize("  x\x0by\0z\r\n"), ["x", "y", "z", EOS]);
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn char_ngrams_skip_boundary_unigrams() {
        assert_eq!(char_ngrams("<ab>", 1, 2), ["<a", "a", "ab", "b", "b>"]);
        // UTF-8 aware: "é" is one character
        assert_eq!(char_ngrams("<é>", 2, 2), ["<é", "é>"]);
    }

    #[test]
    fn meter_nan_semantics() {
        let c = Counts::default();
        assert!(c.precision().is_nan() && c.recall().is_nan() && c.f1().is_nan());
        let c = Counts {
            gold: 4,
            predicted: 2,
            predicted_gold: 1,
        };
        assert_eq!((c.precision(), c.recall()), (0.5, 0.25));
        assert!((c.f1() - 2.0 / 6.0).abs() < 1e-12);
    }

    /// Train -> test -> save -> load -> quantize on a tiny generated dataset (no fixtures).
    #[test]
    fn train_test_save_quantize_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ftnew-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let train = dir.join("train.txt");
        let mut text = String::new();
        for i in 0..600 {
            let (lab, words) = match i % 3 {
                0 => ("a", "apple apricot avocado"),
                1 => ("b", "banana blueberry blackberry"),
                _ => ("c", "cherry coconut cranberry"),
            };
            text += &format!("__label__{lab} {words} filler{} common words\n", i % 13);
        }
        std::fs::write(&train, &text).unwrap();
        let mut args = Args {
            input: train.clone(),
            dim: 10,
            epoch: 5,
            thread: 2,
            verbose: 0,
            ..Args::default()
        };
        args.apply_supervised_defaults();
        args.lr = 0.5;
        args.word_ngrams = 2;
        args.bucket = 10_000;
        let m = Model::train(args, Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(m.labels().len(), 3);
        let r = m.test(&train, 1, 0.0).unwrap();
        assert_eq!(r.nexamples, 600);
        assert!(r.total.precision() > 0.95, "{r:?}");
        assert_eq!(r.labels.iter().map(|c| c.gold).sum::<u64>(), 600);

        let path = dir.join("m.bin");
        m.save(&path).unwrap();
        let mut loaded = Model::load(&path).unwrap();
        for t in ["apple apricot", "banana", "cherry coconut", ""] {
            assert_eq!(
                loaded.predict(t, 2, 0.0).unwrap(),
                m.predict(t, 2, 0.0).unwrap()
            );
        }
        assert_eq!(loaded.test(&train, 1, 0.0).unwrap(), r);

        let qargs = Args {
            dsub: 2,
            ..Args::default()
        };
        loaded.quantize(&qargs).unwrap();
        assert!(loaded.is_quantized());
        assert!(loaded.quantize(&qargs).is_err());
        let qpath = dir.join("m.ftz");
        loaded.save(&qpath).unwrap();
        let q = Model::load(&qpath).unwrap();
        assert_eq!(
            q.predict("banana", 3, 0.0).unwrap(),
            loaded.predict("banana", 3, 0.0).unwrap()
        );
        assert!(q.test(&train, 1, 0.0).unwrap().total.precision() > 0.9);
        std::fs::remove_dir_all(&dir).ok();
    }
}
