"""Self-consistency tests that need neither the C++ package nor downloaded data.

They run on every CI platform (Linux, macOS, Windows, free-threaded Python): train -> save ->
load -> predict round trips for every loss, quantization round trips, the Python API surface
(types, errors, arguments), malformed model files, and threading / GIL behaviour. Training
data is the synthetic ``tests/data/models/train_tiny.txt`` (``scripts/make_tiny_models.py``).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import fasttext_new

HERE = Path(__file__).resolve().parent
MODELS = HERE / "data" / "models"
TRAIN = MODELS / "train_tiny.txt"
TEXTS = [
    "apple banana cherry the of",
    "football tennis goal match",
    "guitar piano melody",
    "compiler function rust bug",
    "café crème brûlée",
    "東京 大阪 寿司",
    "",
    "   ",
    "unknownword anotherunknown",
    "x" * 500,
]
FAST = dict(dim=10, epoch=5, lr=0.5, wordNgrams=2, minn=2, maxn=4, bucket=2000, thread=2, verbose=0)


def exact(a, b):
    """Predictions from the same model file must be bit-identical."""
    (la, pa), (lb, pb) = a, b
    assert la == lb
    if isinstance(pa, np.ndarray):
        assert pa.tobytes() == np.asarray(pb, dtype=pa.dtype).tobytes()
    else:
        for x, y in zip(pa, pb):
            assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


@pytest.fixture(scope="module")
def softmax_model():
    return fasttext_new.train_supervised(input=str(TRAIN), **FAST)


# --- training quality and round trips ------------------------------------------------------


@pytest.mark.parametrize("loss", ["softmax", "hs", "ova", "ns"])
def test_train_save_load_predict(tmp_path, loss):
    m = fasttext_new.train_supervised(input=str(TRAIN), loss=loss, **FAST)
    n, p1, r1 = m.test(str(TRAIN))
    assert n == 600
    assert p1 > 0.9, (loss, p1)
    assert m.loss == loss and m.dim == 10 and m.epoch == 5 and m.wordNgrams == 2
    path = tmp_path / "m.bin"
    m.save_model(str(path))
    m2 = fasttext_new.load_model(str(path))
    for t in TEXTS:
        exact(m.predict(t, k=3), m2.predict(t, k=3))
        exact(m.predict(t, k=-1, threshold=0.01), m2.predict(t, k=-1, threshold=0.01))
        assert m.get_sentence_vector(t).tobytes() == m2.get_sentence_vector(t).tobytes()
    exact(m.predict(TEXTS, k=2), m2.predict(TEXTS, k=2))
    assert m.test(str(TRAIN), k=2) == m2.test(str(TRAIN), k=2)
    assert m.get_words() == m2.get_words() and m.get_labels() == m2.get_labels()
    assert np.array_equal(m.get_input_matrix(), m2.get_input_matrix())
    assert np.array_equal(m.get_output_matrix(), m2.get_output_matrix())


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(qnorm=True),
        dict(cutoff=300),
        dict(cutoff=300, retrain=True, epoch=2),
        dict(qout=True),
        dict(qout=True, qnorm=True, dsub=5),
    ],
    ids=["default", "qnorm", "cutoff", "retrain", "qout", "qout-qnorm-dsub5"],
)
def test_quantize_round_trip(tmp_path, kw):
    m = fasttext_new.train_supervised(input=str(TRAIN), **FAST)
    p_dense = m.test(str(TRAIN))[1]
    m.quantize(input=str(TRAIN), **kw)
    assert m.is_quantized()
    assert m.test(str(TRAIN))[1] > p_dense - 0.1
    with pytest.raises(ValueError):
        m.get_input_matrix()
    path = tmp_path / "m.ftz"
    m.save_model(str(path))
    q = fasttext_new.load_model(str(path))
    assert q.is_quantized()
    for t in TEXTS:
        exact(m.predict(t, k=3), q.predict(t, k=3))
        assert m.get_word_vector(t.split(" ")[0]).tobytes() == q.get_word_vector(t.split(" ")[0]).tobytes()
    assert m.test(str(TRAIN)) == q.test(str(TRAIN))
    with pytest.raises(ValueError):
        q.quantize(input=str(TRAIN))  # already quantized


@pytest.mark.parametrize("model", ["cbow", "skipgram"])
def test_unsupervised_round_trip(tmp_path, model):
    raw = tmp_path / "raw.txt"
    raw.write_text(
        "\n".join(" ".join(w for w in line.split() if not w.startswith("__label__"))
                  for line in TRAIN.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    m = fasttext_new.train_unsupervised(str(raw), model=model, dim=10, epoch=3, minCount=2,
                                        minn=2, maxn=4, bucket=2000, thread=2, verbose=0)
    assert m.get_labels() == m.get_words() or m.get_labels() == []
    with pytest.raises(ValueError):
        m.predict("apple")
    nn = m.get_nearest_neighbors("apple", k=5)
    assert len(nn) == 5 and all(isinstance(s, float) and isinstance(w, str) for s, w in nn)
    assert [s for s, _ in nn] == sorted((s for s, _ in nn), reverse=True)
    assert "apple" not in [w for _, w in nn]
    assert len(m.get_analogies("apple", "banana", "guitar", k=3)) == 3
    path = tmp_path / "u.bin"
    m.save_model(str(path))
    m2 = fasttext_new.load_model(str(path))
    for w in ["apple", "guitar", "oovword", "東京"]:
        assert m.get_word_vector(w).tobytes() == m2.get_word_vector(w).tobytes()
    assert m2.get_nearest_neighbors("apple", k=5) == nn


def test_committed_models_load_and_predict():
    for p in sorted(MODELS.glob("tiny_*")):
        m = fasttext_new.load_model(str(p))
        assert m.get_dimension() == 10
        assert m.is_quantized() == (p.suffix == ".ftz")
        if p.name != "tiny_cbow.bin":
            labels, probs = m.predict("apple banana cherry", k=2)
            assert len(labels) == 2 and probs[0] >= probs[1]
            if "qout" not in p.name:  # the qout models use 300 other labels
                assert m.test(str(TRAIN))[1] > 0.5, p.name


# --- API surface ---------------------------------------------------------------------------


def test_predict_types_and_shapes(softmax_model):
    m = softmax_model
    labels, probs = m.predict("apple banana")
    assert isinstance(labels, tuple) and isinstance(labels[0], str)
    assert isinstance(probs, np.ndarray) and probs.dtype == np.float64 and probs.shape == (1,)
    labels, probs = m.predict(["apple banana", "guitar"], k=2)
    assert isinstance(labels, list) and all(isinstance(x, list) for x in labels)
    assert all(isinstance(p, np.ndarray) and p.dtype == np.float32 and p.shape == (2,) for p in probs)
    all_labels, all_probs = m.predict("apple", k=-1)
    assert len(all_labels) == len(m.get_labels())
    assert abs(float(np.sum(all_probs)) - 1.0) < 1e-3  # softmax
    assert all(p >= 0.2 for p in m.predict("apple", k=-1, threshold=0.2)[1])


def test_predict_batch_matches_predict(softmax_model):
    m = softmax_model
    texts = TEXTS * 20
    for threads in (None, 1, 3):
        bl, bp = m.predict_batch(texts, k=3, threads=threads)
        for t, l, p in zip(texts, bl, bp):
            sl, sp = m.predict(t, k=3)
            assert tuple(l) == sl
            assert np.allclose(p, sp, rtol=0, atol=1e-7)
    assert m.predict_batch([]) == ([], [])
    assert m.predict_batch(iter(["apple"]))[0] == [list(m.predict("apple")[0])]


def test_errors(softmax_model, tmp_path):
    m = softmax_model
    with pytest.raises(ValueError):
        m.predict("two\nlines")
    with pytest.raises(ValueError):
        m.predict(["ok", "two\nlines"])
    with pytest.raises(ValueError):
        m.predict("apple", k=0)
    with pytest.raises(ValueError):
        m.get_sentence_vector("a\nb")
    with pytest.raises(ValueError):
        fasttext_new.load_model(str(tmp_path / "missing.bin"))
    with pytest.raises(ValueError):
        m.test(str(tmp_path / "missing.txt"))
    with pytest.raises(TypeError):
        fasttext_new.train_supervised(input=str(TRAIN), not_an_arg=1)
    with pytest.raises(TypeError):
        fasttext_new.train_supervised(str(TRAIN), input=str(TRAIN))
    with pytest.raises(ValueError):
        fasttext_new.train_supervised(input=str(TRAIN), loss="nope")
    with pytest.raises(ValueError):
        fasttext_new.train_supervised(input=str(tmp_path / "missing.txt"), verbose=0)
    with pytest.raises(ValueError):
        fasttext_new.train_supervised(input=str(TRAIN), dim=0, verbose=0)
    with pytest.raises(ValueError):
        m.quantize(retrain=True)
    with pytest.raises(NotImplementedError):
        m.set_matrices(None, None)
    with pytest.raises(Exception):
        fasttext_new.FastText.supervised()


def test_vocabulary_api(softmax_model):
    m = softmax_model
    words, freqs = m.get_words(include_freq=True)
    labels, lfreqs = m.get_labels(include_freq=True)
    assert freqs.dtype == np.int64 and len(words) == len(freqs)
    assert list(freqs) == sorted(freqs, reverse=True)
    assert len(labels) == 6 and all(lab.startswith("__label__") for lab in labels)
    assert sum(lfreqs) >= 600
    assert words[0] == "</s>" or "</s>" in words
    assert m.words == words and m.labels == labels
    assert "apple" in m and "notaword" not in m
    assert m.get_word_id("apple") == words.index("apple")
    assert m.get_word_id("notaword") == -1
    assert m.get_label_id(labels[2]) == 2
    assert m.get_label_id("__label__nope") == -1
    sub, ids = m.get_subwords("apple")
    assert sub[0] == "apple" and ids[0] == m.get_word_id("apple") and len(sub) == len(ids)
    assert m.get_subword_id("app") >= len(words)
    assert np.array_equal(m["apple"], m.get_word_vector("apple"))
    assert m.get_word_vector("apple").dtype == np.float32
    assert m.get_word_vector("apple").shape == (10,)
    rows = m.get_input_matrix().shape
    assert rows == (len(words) + 2000, 10)
    assert m.get_output_matrix().shape == (6, 10)
    assert np.array_equal(m.get_input_vector(3), m.get_input_matrix()[3])
    with pytest.raises(ValueError):
        m.get_input_vector(rows[0])
    assert m.get_line("apple __label__fruit banana") == (["apple", "banana", "</s>"], ["__label__fruit"])
    assert m.get_line(["a b", "c"]) == ([["a", "b", "</s>"], ["c", "</s>"]], [[], []])
    assert fasttext_new.tokenize("a b\nc") == ["a", "b", "</s>", "c"]
    assert "fasttext_new" in repr(m)
    assert isinstance(fasttext_new.__version__, str)


def test_test_label(softmax_model):
    r = softmax_model.test_label(str(TRAIN))
    assert set(r) == set(softmax_model.get_labels())
    for v in r.values():
        assert set(v) == {"precision", "recall", "f1score"}
        assert 0.0 <= v["precision"] <= 1.0


def test_on_unicode_error_and_install_shim():
    m = fasttext_new.load_model(str(MODELS / "tiny_softmax.bin"))
    assert m.get_words(on_unicode_error="replace") == m.get_words()
    saved = {k: sys.modules.get(k) for k in ("fasttext", "fasttext.FastText")}
    try:
        for k in saved:
            sys.modules.pop(k, None)
        fasttext_new.install_as_fasttext()
        import fasttext  # noqa: F401

        assert fasttext is fasttext_new
        assert fasttext.load_model(str(MODELS / "tiny_hs.bin")).predict("apple")[0]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_verbose_progress_output(capfd):
    fasttext_new.train_supervised(input=str(TRAIN), epoch=1, dim=4, thread=1, verbose=2)
    err = capfd.readouterr().err
    assert "Number of words:" in err and "Progress: 100.0%" in err


def test_pretrained_vectors(tmp_path):
    vec = tmp_path / "pre.vec"
    vec.write_text("2 10\napple " + " ".join(["0.5"] * 10) + "\nzzzunseen " + " ".join(["1"] * 10) + "\n",
                   encoding="utf-8")
    m = fasttext_new.train_supervised(input=str(TRAIN), pretrainedVectors=str(vec), dim=10, epoch=1,
                                      thread=1, verbose=0)
    assert "zzzunseen" in m.get_words()
    bad = tmp_path / "bad.vec"
    bad.write_text("1 7\napple 1 2 3 4 5 6 7\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fasttext_new.train_supervised(input=str(TRAIN), pretrainedVectors=str(bad), dim=10, verbose=0)


# --- threading and the GIL -----------------------------------------------------------------


def test_concurrent_predict_threads(softmax_model):
    m = softmax_model
    texts = TEXTS * 50
    want = [m.predict(t, k=2) for t in texts]
    errors = []

    def work():
        try:
            for t, w in zip(texts, want):
                got = m.predict(t, k=2)
                assert got[0] == w[0] and got[1].tobytes() == w[1].tobytes()
            bl, _ = m.predict_batch(texts, k=2, threads=2)
            assert bl == [list(w[0]) for w in want]
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_training_releases_the_gil():
    ticks = 0
    done = threading.Event()

    def train():
        fasttext_new.train_supervised(input=str(TRAIN), epoch=500, dim=20, thread=1, verbose=0)
        done.set()

    t = threading.Thread(target=train)
    t.start()
    start = time.perf_counter()
    while not done.is_set():  # pure-Python work: only runs if training released the GIL
        ticks += 1
    t.join()
    elapsed = time.perf_counter() - start
    assert elapsed > 0.05 and ticks > 10_000, (elapsed, ticks)


def test_predict_batch_releases_the_gil(softmax_model):
    texts = ["apple banana " * 200] * 4000
    counter = 0
    done = threading.Event()

    def score():
        softmax_model.predict_batch(texts, k=1, threads=1)
        done.set()

    t = threading.Thread(target=score)
    t.start()
    while not done.is_set():
        counter += 1
    t.join()
    assert counter > 10_000


def test_concurrent_quantize_and_predict(tmp_path):
    m = fasttext_new.train_supervised(input=str(TRAIN), **FAST)
    stop = threading.Event()
    errors = []

    def reader():
        while not stop.is_set():
            try:
                m.predict("apple banana", k=2)
            except Exception as e:  # pragma: no cover
                errors.append(e)

    rs = [threading.Thread(target=reader) for _ in range(3)]
    for r in rs:
        r.start()
    m.quantize(input=str(TRAIN), qnorm=True)
    stop.set()
    for r in rs:
        r.join()
    assert not errors and m.is_quantized()


@pytest.mark.skipif(not hasattr(sys, "_is_gil_enabled"), reason="needs Python 3.13+")
def test_free_threaded_build_keeps_gil_disabled():
    if sysconfig_free_threaded():
        assert not sys._is_gil_enabled()


def sysconfig_free_threaded() -> bool:
    import sysconfig

    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@pytest.mark.skipif(not sysconfig_free_threaded(), reason="free-threaded CPython only")
@pytest.mark.skipif((os.cpu_count() or 1) < 3, reason="needs 3+ CPUs")
def test_free_threaded_predict_scales_across_threads(softmax_model):
    """On free-threaded Python, predict(str) on short lines (GIL never released) runs in
    parallel across Python threads."""
    m = softmax_model
    texts = TEXTS[:6] * 400
    n = min(4, os.cpu_count() or 1)

    def work(chunk):
        for t in chunk:
            m.predict(t, k=2)

    def timed(nthreads):
        threads = [threading.Thread(target=work, args=(texts,)) for _ in range(nthreads)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return time.perf_counter() - t0

    best = 0.0
    for _ in range(3):  # shared CI machines are noisy: best of 3
        one = timed(1)
        many = timed(n)
        best = max(best, n * one / many)  # speed-up over running the n workloads serially
    assert best > 1.3, f"{n} threads only {best:.2f}x faster than serial"
