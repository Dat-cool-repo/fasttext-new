//! PyO3 extension module `fasttext_new._fasttext_new`.
//!
//! Returns plain Python containers plus `bytearray`s of native-endian `float32` data; the thin
//! Python layer (`python/fasttext_new/FastText.py`) turns those into the exact numpy shapes the
//! original `fasttext` package returns. This keeps the extension abi3-compatible without a
//! compile-time numpy dependency.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{RecvTimeoutError, channel};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};
use std::time::Duration;

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyString, PyTuple};

use crate::model::{Args, Entry, Error, LossName, Model, ModelName, Pred};

fn to_py(e: Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn f32_bytes<'py>(py: Python<'py>, v: &[f32]) -> Bound<'py, PyByteArray> {
    // SAFETY: f32 is plain-old-data with no padding; viewing it as bytes is sound.
    let b =
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    PyByteArray::new(py, b)
}

/// A dictionary string for Python, like C++ `castToPythonString`: valid UTF-8 becomes a
/// `str`; an entry with invalid UTF-8 is decoded with the `on_unicode_error` handler
/// (`"strict"` raises `UnicodeDecodeError`, `"replace"`, `"ignore"`, ...).
fn entry_str<'py>(py: Python<'py>, e: &Entry, errors: &str) -> PyResult<Bound<'py, PyAny>> {
    match &e.raw {
        None => Ok(PyString::new(py, &e.word).into_any()),
        Some(b) => PyBytes::new(py, b).call_method1("decode", ("utf-8", errors)),
    }
}

/// A word returned by a nearest-neighbour / analogy query, decoded like [`entry_str`].
fn word_str<'py>(py: Python<'py>, m: &Model, w: &str, errors: &str) -> PyResult<Bound<'py, PyAny>> {
    match m.entry_for(w) {
        Some(e) if m.has_raw() => entry_str(py, e, errors),
        _ => Ok(PyString::new(py, w).into_any()),
    }
}

/// `(label, precision, recall, f1score)` rows of `test_label`.
type LabelMetrics<'py> = Vec<(Bound<'py, PyAny>, f64, f64, f64)>;

/// `predict(str)` releases the GIL only for texts at least this long (bytes).
const DETACH_MIN_BYTES: usize = 2048;

/// Cached rayon pools keyed by thread count (for the `threads=` argument).
static POOLS: Mutex<Vec<(usize, Arc<rayon::ThreadPool>)>> = Mutex::new(Vec::new());

fn pool(n: usize) -> PyResult<Arc<rayon::ThreadPool>> {
    let mut pools = POOLS
        .lock()
        .map_err(|_| PyRuntimeError::new_err("pool lock poisoned"))?;
    if let Some((_, p)) = pools.iter().find(|(k, _)| *k == n) {
        return Ok(Arc::clone(p));
    }
    let p = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    );
    pools.push((n, Arc::clone(&p)));
    Ok(p)
}

/// Required item of an argument dict built by the Python layer.
fn item<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    d.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(key.to_string()))?
        .extract()
        .map_err(Into::into)
}

fn loss_from_str(s: &str) -> PyResult<LossName> {
    Ok(match s {
        "softmax" => LossName::Softmax,
        "ns" => LossName::NegativeSampling,
        "hs" => LossName::HierarchicalSoftmax,
        "ova" => LossName::OneVsAll,
        _ => return Err(PyValueError::new_err("Unrecognized loss name")),
    })
}

fn loss_to_str(l: LossName) -> &'static str {
    match l {
        LossName::Softmax => "softmax",
        LossName::NegativeSampling => "ns",
        LossName::HierarchicalSoftmax => "hs",
        LossName::OneVsAll => "ova",
    }
}

fn model_from_str(s: &str) -> PyResult<ModelName> {
    Ok(match s {
        "supervised" => ModelName::Supervised,
        "skipgram" => ModelName::SkipGram,
        "cbow" => ModelName::Cbow,
        _ => return Err(PyValueError::new_err("Unrecognized model name")),
    })
}

fn model_to_str(m: ModelName) -> &'static str {
    match m {
        ModelName::Supervised => "supervised",
        ModelName::SkipGram => "skipgram",
        ModelName::Cbow => "cbow",
    }
}

/// Training arguments from the dict built by `fasttext_new.train_supervised` /
/// `train_unsupervised` (C++ argument names).
fn train_args(d: &Bound<'_, PyDict>) -> PyResult<Args> {
    let a = Args {
        input: PathBuf::from(item::<String>(d, "input")?),
        model: model_from_str(&item::<String>(d, "model")?)?,
        loss: loss_from_str(&item::<String>(d, "loss")?)?,
        lr: item(d, "lr")?,
        dim: item(d, "dim")?,
        ws: item(d, "ws")?,
        epoch: item(d, "epoch")?,
        min_count: item(d, "minCount")?,
        min_count_label: item(d, "minCountLabel")?,
        minn: item(d, "minn")?,
        maxn: item(d, "maxn")?,
        neg: item(d, "neg")?,
        word_ngrams: item(d, "wordNgrams")?,
        bucket: item(d, "bucket")?,
        thread: item(d, "thread")?,
        lr_update_rate: item(d, "lrUpdateRate")?,
        t: item(d, "t")?,
        label: item(d, "label")?,
        verbose: item(d, "verbose")?,
        pretrained_vectors: PathBuf::from(item::<String>(d, "pretrainedVectors")?),
        seed: item(d, "seed")?,
        autotune_validation_file: PathBuf::from(item::<String>(d, "autotuneValidationFile")?),
        autotune_metric: item(d, "autotuneMetric")?,
        autotune_predictions: item(d, "autotunePredictions")?,
        autotune_duration: item(d, "autotuneDuration")?,
        autotune_model_size: item(d, "autotuneModelSize")?,
        ..Args::default()
    };
    if a.dim <= 0 || a.epoch <= 0 || a.thread <= 0 || a.ws <= 0 || a.bucket < 0 {
        return Err(PyValueError::new_err(
            "dim, epoch, thread and ws must be positive and bucket non-negative",
        ));
    }
    Ok(a)
}

/// Run `work` on a background thread with the GIL released, checking for signals
/// (Ctrl-C) every 100 ms. On a signal, `abort` is set, the worker is awaited and the
/// signal's exception (e.g. `KeyboardInterrupt`) is raised.
fn run_interruptible<T: Send + 'static>(
    py: Python<'_>,
    abort: Arc<AtomicBool>,
    work: impl FnOnce() -> T + Send + 'static,
) -> PyResult<T> {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("fasttext-train".into())
        .spawn(move || {
            let _ = tx.send(work());
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // `Receiver` is not `Sync`; the mutex makes `&rx` sendable into `detach`.
    let rx = Mutex::new(rx);
    let recv = |timeout: Option<Duration>| {
        let rx = rx.lock().unwrap_or_else(|e| e.into_inner());
        match timeout {
            Some(t) => rx.recv_timeout(t),
            None => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
        }
    };
    loop {
        match py.detach(|| recv(Some(Duration::from_millis(100)))) {
            Ok(v) => return Ok(v),
            Err(RecvTimeoutError::Timeout) => {
                if let Err(e) = py.check_signals() {
                    abort.store(true, Ordering::Relaxed);
                    let _ = py.detach(|| recv(None));
                    return Err(e);
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(PyRuntimeError::new_err("training thread panicked"));
            }
        }
    }
}

/// Train a model. `args` is a dict with every C++ argument (built by the Python layer).
/// The GIL is released while training; Ctrl-C stops training and raises `KeyboardInterrupt`.
#[pyfunction]
fn train(py: Python<'_>, args: &Bound<'_, PyDict>) -> PyResult<PyModel> {
    let args = train_args(args)?;
    let abort = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&abort);
    let model = run_interruptible(py, abort, move || Model::train(args, flag))?.map_err(to_py)?;
    Ok(PyModel::new(py, model))
}

/// C++ `fasttext.tokenize`: split on fastText whitespace, newlines become `</s>`.
#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    crate::model::tokenize(text)
}

/// Low-level model handle. Use `fasttext_new.load_model()` / `train_supervised()` instead.
#[pyclass(frozen, name = "Model", module = "fasttext_new._fasttext_new")]
pub struct PyModel {
    /// Write-locked only by `quantize`.
    inner: RwLock<Model>,
    /// Interned label strings, so predictions don't allocate new Python strings.
    labels: Vec<Py<PyString>>,
    /// Raw bytes of labels that are not valid UTF-8 (empty if there are none).
    raw_labels: Vec<Option<Box<[u8]>>>,
}

impl PyModel {
    fn new(py: Python<'_>, inner: Model) -> Self {
        let labels = inner
            .labels()
            .iter()
            .map(|l| PyString::new(py, l).unbind())
            .collect();
        let entries = &inner.entries()[inner.nwords()..];
        let raw_labels = if entries.iter().any(|e| e.raw.is_some()) {
            entries.iter().map(|e| e.raw.clone()).collect()
        } else {
            Vec::new()
        };
        Self {
            inner: RwLock::new(inner),
            labels,
            raw_labels,
        }
    }

    /// Label `id` for Python (decoded with `errors` if it is not valid UTF-8).
    fn label<'py>(&self, py: Python<'py>, id: u32, errors: &str) -> PyResult<Bound<'py, PyAny>> {
        match self.raw_labels.get(id as usize) {
            Some(Some(b)) => PyBytes::new(py, b).call_method1("decode", ("utf-8", errors)),
            _ => Ok(self.labels[id as usize].bind(py).clone().into_any()),
        }
    }

    fn m(&self) -> PyResult<RwLockReadGuard<'_, Model>> {
        self.inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("model lock poisoned"))
    }

    fn label_list<'py>(
        &self,
        py: Python<'py>,
        preds: &[Pred],
        errors: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        if self.raw_labels.is_empty() {
            return PyList::new(py, preds.iter().map(|p| self.labels[p.0 as usize].bind(py)));
        }
        PyList::new(
            py,
            preds
                .iter()
                .map(|p| self.label(py, p.0, errors))
                .collect::<PyResult<Vec<_>>>()?,
        )
    }
}

impl PyModel {
    fn entry_list<'py>(
        &self,
        py: Python<'py>,
        m: &Model,
        entries: &[Entry],
        errors: &str,
    ) -> PyResult<(Bound<'py, PyList>, Vec<i64>)> {
        let counts = entries.iter().map(|e| e.count).collect();
        let list = if m.has_raw() {
            PyList::new(
                py,
                entries
                    .iter()
                    .map(|e| entry_str(py, e, errors))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
        } else {
            PyList::new(py, entries.iter().map(|e| e.word.as_str()))?
        };
        Ok((list, counts))
    }
}

#[pymethods]
impl PyModel {
    #[new]
    fn py_new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| Model::load(&path)).map_err(to_py)?;
        Ok(Self::new(py, inner))
    }

    /// Predict one line. Returns `(labels: tuple[str], probs: list[float])`.
    #[pyo3(signature = (text, k=1, threshold=0.0, on_unicode_error="strict"))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        text: String,
        k: i32,
        threshold: f32,
        on_unicode_error: &str,
    ) -> PyResult<(Bound<'py, PyTuple>, Vec<f64>)> {
        let m = self.m()?;
        let m = &*m;
        // Releasing the GIL costs more than predicting a short line and hurts throughput
        // when many Python threads call predict(); only release it for long documents.
        let preds = if text.len() >= DETACH_MIN_BYTES {
            py.detach(|| m.predict(&text, k, threshold))
        } else {
            m.predict(&text, k, threshold)
        }
        .map_err(to_py)?;
        let labels = if self.raw_labels.is_empty() {
            PyTuple::new(py, preds.iter().map(|p| self.labels[p.0 as usize].bind(py)))?
        } else {
            self.label_list(py, &preds, on_unicode_error)?.to_tuple()
        };
        Ok((labels, preds.iter().map(|p| p.1 as f64).collect()))
    }

    /// Predict many lines in parallel with the GIL released.
    ///
    /// Returns `(labels: list[list[str]], probs: bytearray[float32], counts: list[int])`, where
    /// `probs` is the concatenation of every row's probabilities and `counts[i]` is the number
    /// of predictions for row `i`. `threads=None` uses the global rayon pool
    /// (`RAYON_NUM_THREADS`, default: all cores).
    #[pyo3(signature = (texts, k=1, threshold=0.0, threads=None, on_unicode_error="strict"))]
    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        k: i32,
        threshold: f32,
        threads: Option<usize>,
        on_unicode_error: &str,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyByteArray>, Vec<usize>)> {
        let pool = threads.filter(|&n| n > 0).map(pool).transpose()?;
        let m = self.m()?;
        let m = &*m;
        let rows = py
            .detach(|| match pool {
                Some(p) => p.install(|| m.predict_batch(&texts, k, threshold)),
                None => m.predict_batch(&texts, k, threshold),
            })
            .map_err(to_py)?;
        let labels = PyList::new(
            py,
            rows.iter()
                .map(|r| self.label_list(py, r, on_unicode_error))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().map(|p| p.1)).collect();
        let counts = rows.iter().map(Vec::len).collect();
        Ok((labels, f32_bytes(py, &flat), counts))
    }

    fn get_word_vector<'py>(
        &self,
        py: Python<'py>,
        word: String,
    ) -> PyResult<Bound<'py, PyByteArray>> {
        Ok(f32_bytes(py, &self.m()?.word_vector(&word)))
    }

    fn get_sentence_vector<'py>(
        &self,
        py: Python<'py>,
        text: String,
    ) -> PyResult<Bound<'py, PyByteArray>> {
        let m = self.m()?;
        let m = &*m;
        let v = if text.len() >= DETACH_MIN_BYTES {
            py.detach(|| m.sentence_vector(&text))
        } else {
            m.sentence_vector(&text)
        }
        .map_err(to_py)?;
        Ok(f32_bytes(py, &v))
    }

    fn get_dimension(&self) -> PyResult<usize> {
        Ok(self.m()?.dimension())
    }

    fn is_quantized(&self) -> PyResult<bool> {
        Ok(self.m()?.is_quantized())
    }

    fn is_supervised(&self) -> PyResult<bool> {
        Ok(self.m()?.is_supervised())
    }

    /// `(model, loss, wordNgrams, minn, maxn, bucket)`
    fn args_summary(&self) -> PyResult<(String, String, i32, i32, i32, i32)> {
        Ok(self.m()?.args_summary())
    }

    /// Model arguments as a dict with the C++ names (`lr`, `dim`, `minCount`, ...).
    /// `lr`, `thread`, `verbose`, ... are not stored in model files; loaded models report the
    /// C++ defaults for them, like the original package.
    fn get_args<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = self.m()?;
        let a = m.args();
        let d = PyDict::new(py);
        d.set_item("model", model_to_str(a.model))?;
        d.set_item("loss", loss_to_str(a.loss))?;
        d.set_item("lr", a.lr)?;
        d.set_item("dim", a.dim)?;
        d.set_item("ws", a.ws)?;
        d.set_item("epoch", a.epoch)?;
        d.set_item("minCount", a.min_count)?;
        d.set_item("minCountLabel", a.min_count_label)?;
        d.set_item("minn", a.minn)?;
        d.set_item("maxn", a.maxn)?;
        d.set_item("neg", a.neg)?;
        d.set_item("wordNgrams", a.word_ngrams)?;
        d.set_item("bucket", a.bucket)?;
        d.set_item("thread", a.thread)?;
        d.set_item("lrUpdateRate", a.lr_update_rate)?;
        d.set_item("t", a.t)?;
        d.set_item("label", a.label.as_str())?;
        d.set_item("verbose", a.verbose)?;
        d.set_item("pretrainedVectors", a.pretrained_vectors.to_string_lossy())?;
        d.set_item("seed", a.seed)?;
        d.set_item("qout", a.qout)?;
        Ok(d)
    }

    /// `(words, counts)`; words that are not valid UTF-8 are decoded with `on_unicode_error`.
    #[pyo3(signature = (on_unicode_error="strict"))]
    fn get_words<'py>(
        &self,
        py: Python<'py>,
        on_unicode_error: &str,
    ) -> PyResult<(Bound<'py, PyList>, Vec<i64>)> {
        let m = self.m()?;
        let entries = &m.entries()[..m.nwords()];
        self.entry_list(py, &m, entries, on_unicode_error)
    }

    /// `(labels, counts)`; labels that are not valid UTF-8 are decoded with `on_unicode_error`.
    #[pyo3(signature = (on_unicode_error="strict"))]
    fn get_labels<'py>(
        &self,
        py: Python<'py>,
        on_unicode_error: &str,
    ) -> PyResult<(Bound<'py, PyList>, Vec<i64>)> {
        let m = self.m()?;
        let entries = &m.entries()[m.nwords()..];
        self.entry_list(py, &m, entries, on_unicode_error)
    }

    fn get_word_id(&self, word: &str) -> PyResult<i32> {
        Ok(self.m()?.word_id(word))
    }

    fn get_label_id(&self, label: &str) -> PyResult<i32> {
        Ok(self.m()?.label_id(label))
    }

    fn get_subwords(&self, word: &str) -> PyResult<(Vec<String>, Vec<i32>)> {
        Ok(self.m()?.subwords(word))
    }

    fn get_subword_id(&self, subword: &str) -> PyResult<i32> {
        self.m()?.subword_id(subword).map_err(to_py)
    }

    fn get_input_vector<'py>(
        &self,
        py: Python<'py>,
        ind: i32,
    ) -> PyResult<Bound<'py, PyByteArray>> {
        Ok(f32_bytes(py, &self.m()?.input_vector(ind).map_err(to_py)?))
    }

    /// C++ `getLine`: `(words, labels)` of one line.
    fn get_line(&self, text: &str) -> PyResult<(Vec<String>, Vec<String>)> {
        Ok(self.m()?.tokens(text))
    }

    #[pyo3(signature = (word, k=10, on_unicode_error="strict"))]
    fn get_nearest_neighbors<'py>(
        &self,
        py: Python<'py>,
        word: String,
        k: usize,
        on_unicode_error: &str,
    ) -> PyResult<Vec<(f32, Bound<'py, PyAny>)>> {
        let m = self.m()?;
        let m = &*m;
        let res = py.detach(|| m.nearest_neighbors(&word, k));
        res.into_iter()
            .map(|(s, w)| Ok((s, word_str(py, m, &w, on_unicode_error)?)))
            .collect()
    }

    #[pyo3(signature = (a, b, c, k=10, on_unicode_error="strict"))]
    fn get_analogies<'py>(
        &self,
        py: Python<'py>,
        a: String,
        b: String,
        c: String,
        k: usize,
        on_unicode_error: &str,
    ) -> PyResult<Vec<(f32, Bound<'py, PyAny>)>> {
        let m = self.m()?;
        let m = &*m;
        let res = py.detach(|| m.analogies(&a, &b, &c, k));
        res.into_iter()
            .map(|(s, w)| Ok((s, word_str(py, m, &w, on_unicode_error)?)))
            .collect()
    }

    /// `(rows, cols, bytearray[float32])`
    fn get_input_matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(usize, usize, Bound<'py, PyByteArray>)> {
        let m = self.m()?;
        let (r, c, d) = m.input_matrix().map_err(to_py)?;
        Ok((r, c, f32_bytes(py, d)))
    }

    /// `(rows, cols, bytearray[float32])`
    fn get_output_matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(usize, usize, Bound<'py, PyByteArray>)> {
        let m = self.m()?;
        let (r, c, d) = m.output_matrix().map_err(to_py)?;
        Ok((r, c, f32_bytes(py, d)))
    }

    /// Save the model (C++-compatible `.bin` / `.ftz`). The GIL is released.
    fn save_model(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let m = self.m()?;
        let m = &*m;
        py.detach(|| m.save(&path)).map_err(to_py)
    }

    /// Quantize in place. `qargs`: dict with `input`, `qout`, `cutoff`, `retrain`, `epoch`,
    /// `lr`, `thread`, `verbose`, `dsub`, `qnorm` (built by the Python layer).
    fn quantize(&self, py: Python<'_>, qargs: &Bound<'_, PyDict>) -> PyResult<()> {
        let q = Args {
            input: PathBuf::from(item::<String>(qargs, "input")?),
            qout: item(qargs, "qout")?,
            cutoff: item(qargs, "cutoff")?,
            retrain: item(qargs, "retrain")?,
            epoch: item(qargs, "epoch")?,
            lr: item(qargs, "lr")?,
            thread: item(qargs, "thread")?,
            verbose: item(qargs, "verbose")?,
            dsub: item(qargs, "dsub")?,
            qnorm: item(qargs, "qnorm")?,
            ..Args::default()
        };
        if q.dsub == 0 || q.thread <= 0 {
            return Err(PyValueError::new_err("dsub and thread must be positive"));
        }
        py.detach(|| {
            let mut m = self
                .inner
                .write()
                .map_err(|_| Error::Failed("model lock poisoned".into()))?;
            m.quantize(&q)
        })
        .map_err(to_py)
    }

    /// C++ `test`: `(number of examples, precision@k, recall@k)`.
    #[pyo3(signature = (path, k=1, threshold=0.0))]
    fn test(
        &self,
        py: Python<'_>,
        path: PathBuf,
        k: i32,
        threshold: f32,
    ) -> PyResult<(u64, f64, f64)> {
        let m = self.m()?;
        let m = &*m;
        let r = py.detach(|| m.test(&path, k, threshold)).map_err(to_py)?;
        Ok((r.nexamples, r.total.precision(), r.total.recall()))
    }

    /// C++ `testLabel`: `[(label, precision, recall, f1score)]` for every label.
    #[pyo3(signature = (path, k=1, threshold=0.0))]
    fn test_label<'py>(
        &self,
        py: Python<'py>,
        path: PathBuf,
        k: i32,
        threshold: f32,
    ) -> PyResult<LabelMetrics<'py>> {
        let m = self.m()?;
        let m = &*m;
        let r = py.detach(|| m.test(&path, k, threshold)).map_err(to_py)?;
        // C++ returns the labels as `std::string` keys, i.e. strictly decoded.
        r.labels
            .iter()
            .enumerate()
            .map(|(i, c)| {
                Ok((
                    self.label(py, i as u32, "strict")?,
                    c.precision(),
                    c.recall(),
                    c.f1(),
                ))
            })
            .collect()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", *self.m()?))
    }
}

#[pymodule(gil_used = false)]
fn _fasttext_new(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
