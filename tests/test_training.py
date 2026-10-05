"""Training, test(), save_model() and quantize() vs the original C++ `fasttext` package.

Training is stochastic (Hogwild! SGD on several threads), so trained models are compared by
quality, not bit by bit: P@1 / R@5 from ``model.test()`` on a held-out split (supervised) and
the language purity of word nearest neighbours (unsupervised) must land within a small
tolerance of the C++ package trained with the same arguments. Everything that *is*
deterministic is compared exactly: vocabularies, ``test()`` / ``test_label()`` on the same model
file, and predictions of models saved by one implementation and loaded by the other.

Datasets (``python scripts/make_train_data.py``): cooking.stackexchange (fastText tutorial,
735 tags) and the papluca 20-language identification splits.
"""

from __future__ import annotations

import _thread
import collections
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import fasttext_new

ref = pytest.importorskip("fasttext", reason="reference C++ package (fasttext-numpy2) not installed")
ref.FastText.eprint = lambda *a, **k: None

DATA = Path(os.environ.get("FASTTEXT_NEW_DATA", Path(__file__).resolve().parent.parent / "data"))
THREADS = 4


def data(name: str) -> str:
    p = DATA / name
    if not p.exists():
        pytest.skip(f"{name} missing (run scripts/make_train_data.py)")
    return str(p)


_CPP_JOB = """
import json, sys
import fasttext
job = json.loads(sys.argv[1])
if job["fn"] == "quantize":
    m = fasttext.load_model(job["src"])
    m.quantize(**job["kwargs"])
else:
    m = getattr(fasttext, job["fn"])(**job["kwargs"])
m.save_model(job["out"])
"""


def cpp(fn: str, out, src=None, **kwargs):
    """Train (``fn="train_supervised"`` / ``"train_unsupervised"``) or quantize (``src=``)
    with the C++ package and return the saved result loaded by the C++ package.

    Runs in a subprocess: the C++ trainer sporadically fails with "Encountered NaN", which is
    sometimes raised inside a worker thread and aborts the whole process. Retried."""
    job = json.dumps(dict(fn=fn, out=str(out), src=str(src), kwargs=kwargs))
    for _ in range(30):
        p = subprocess.run([sys.executable, "-c", _CPP_JOB, job], capture_output=True, text=True)
        if p.returncode == 0:
            return ref.load_model(str(out))
        if "NaN" not in p.stderr and p.returncode >= 0:
            raise RuntimeError(p.stderr[-2000:])
    pytest.skip("C++ reference kept failing with 'Encountered NaN'")


def vocab(m):
    w, f = m.get_words(include_freq=True)
    labels, lf = m.get_labels(include_freq=True)
    return dict(zip(w, f.tolist())), dict(zip(labels, lf.tolist()))


# --------------------------------------------------------------------------------------------
# supervised quality
# --------------------------------------------------------------------------------------------

COOK = dict(lr=1.0, epoch=25, wordNgrams=2, bucket=200_000)
LID = dict(lr=0.5, epoch=3, dim=16, minn=2, maxn=4, wordNgrams=2, bucket=200_000)
# (train, valid, args, max |dP@1| and |dR@5|). Measured over 3 runs each: |dP@1| <= 0.011,
# run-to-run sd <= 0.003 (C++ only initialises thread/10 of the input matrix, see README).
SUPERVISED = {
    "cooking-softmax": ("cooking.train", "cooking.valid", COOK, 0.02),
    "cooking-hs": ("cooking.train", "cooking.valid", dict(COOK, loss="hs"), 0.025),
    "cooking-ova": ("cooking.train", "cooking.valid", dict(COOK, lr=0.5, loss="ova"), 0.025),
    "cooking-ns": ("cooking.train", "cooking.valid", dict(COOK, lr=0.5, loss="ns"), 0.025),
    "lid-softmax": ("lid.train", "lid.valid", LID, 0.01),
    "lid-hs": ("lid.train", "lid.valid", dict(LID, loss="hs"), 0.01),
}


@pytest.mark.parametrize("name", list(SUPERVISED))
def test_supervised_quality_matches_cpp(name, tmp_path):
    train, valid, kw, tol = SUPERVISED[name]
    train, valid = data(train), data(valid)
    m = fasttext_new.train_supervised(input=train, **kw, thread=THREADS, verbose=0)
    r = cpp("train_supervised", tmp_path / "cpp.bin", input=train, **kw, thread=THREADS, verbose=0)
    n1, p1, _ = m.test(valid)
    rn1, rp1, _ = r.test(valid)
    _, _, r5 = m.test(valid, k=5)
    _, _, rr5 = r.test(valid, k=5)
    print(f"TRAIN {name}: P@1 rs={p1:.4f} cpp={rp1:.4f} (d={p1 - rp1:+.4f})  R@5 rs={r5:.4f} cpp={rr5:.4f}")
    assert n1 == rn1
    assert abs(p1 - rp1) <= tol
    assert abs(r5 - rr5) <= tol
    # Vocabulary building is deterministic: same words, labels and counts.
    assert vocab(m) == vocab(r)
    assert m.get_dimension() == r.get_dimension()


# --------------------------------------------------------------------------------------------
# unsupervised quality: language purity of word nearest neighbours
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def word_langs():
    langs = collections.defaultdict(collections.Counter)
    with open(data("lid.train"), encoding="utf-8") as f:
        for line in f:
            lab, _, text = line.partition(" ")
            for w in set(text.split()):
                langs[w][lab] += 1
    out = {}
    for w, c in langs.items():
        n = sum(c.values())
        lab, top = c.most_common(1)[0]
        if n >= 20 and top / n >= 0.9:  # frequent words specific to one language
            out[w] = lab
    return out


def nn_purity(m, word_langs, k=10):
    words = sorted(w for w in word_langs if m.get_word_id(w) >= 0)
    x = np.stack([m.get_word_vector(w) for w in words])
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    s = x @ x.T
    np.fill_diagonal(s, -np.inf)
    nn = np.argpartition(-s, k, axis=1)[:, :k]
    labs = np.array([word_langs[w] for w in words])
    return float((labs[nn] == labs[:, None]).mean())


@pytest.mark.parametrize("model", ["skipgram", "cbow"])
def test_unsupervised_quality_matches_cpp(model, word_langs, tmp_path):
    kw = dict(input=data("lid_raw.txt"), model=model, dim=32, epoch=3, minn=2, maxn=4, bucket=200_000)
    m = fasttext_new.train_unsupervised(**kw, thread=THREADS, verbose=0)
    r = cpp("train_unsupervised", tmp_path / "cpp.bin", **kw, thread=THREADS, verbose=0)
    pm, pr = nn_purity(m, word_langs), nn_purity(r, word_langs)
    print(f"TRAIN {model}: neighbour language purity rs={pm:.4f} cpp={pr:.4f}")
    assert abs(pm - pr) <= 0.02  # measured: |d| <= 0.0005, sd <= 0.0002
    assert vocab(m) == vocab(r)
    assert m.f.args_summary()[0] == ("sg" if model == "skipgram" else "cbow")


# --------------------------------------------------------------------------------------------
# test() / test_label(): exact on the same model file
# --------------------------------------------------------------------------------------------


def assert_same_test_label(a: dict, b: dict):
    assert a.keys() == b.keys()
    for lab in a:
        for key in ("precision", "recall", "f1score"):
            x, y = a[lab][key], b[lab][key]
            assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y, abs=1e-12), (lab, key, x, y)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A small cooking model trained by us and saved, loaded by both implementations."""
    path = tmp_path_factory.mktemp("models") / "cooking_rs.bin"
    m = fasttext_new.train_supervised(input=data("cooking.train"), **COOK, thread=THREADS, verbose=0)
    m.save_model(str(path))
    return m, path


@pytest.mark.parametrize("model_name", ["ours", "fx_softmax_ng2.bin", "fx_ova_many.bin", "fx_hs_ng3_q.ftz"])
@pytest.mark.parametrize("k,threshold", [(1, 0.0), (5, 0.0), (3, 0.05), (-1, 0.0)])
def test_test_and_test_label_match_cpp(trained, model_name, k, threshold):
    if model_name == "ours":
        path, valid = trained[1], data("cooking.valid")
    else:
        path, valid = DATA / model_name, data("train_many.txt" if "many" in model_name else "train.txt")
        if not path.exists():
            pytest.skip(f"{model_name} missing")
    m, r = fasttext_new.load_model(str(path)), ref.load_model(str(path))
    assert m.test(valid, k=k, threshold=threshold) == pytest.approx(r.test(valid, k=k, threshold=threshold), abs=1e-12)
    assert_same_test_label(m.test_label(valid, k=k, threshold=threshold), r.test_label(valid, k=k, threshold=threshold))


def test_test_edge_cases(tmp_path, trained):
    m, path = trained
    r = ref.load_model(str(path))
    f = tmp_path / "edge.txt"
    # no trailing newline, unknown labels, label-only and word-only lines, a literal </s>,
    # several labels, blank lines, CRLF
    f.write_bytes(
        b"__label__baking how to bake bread\n\n__label__unknown-tag soup\n__label__baking\n"
        b"just words\n__label__bread __label__baking banana bread </s> __label__eggs eggs\r\n"
        b"__label__eggs boil eggs"
    )
    for k in (1, 2, -1):
        assert m.test(str(f), k=k) == r.test(str(f), k=k)
        assert_same_test_label(m.test_label(str(f), k=k), r.test_label(str(f), k=k))
    with pytest.raises(ValueError):
        m.test(str(f), k=0)


# --------------------------------------------------------------------------------------------
# save_model: our files load in C++ and predict identically
# --------------------------------------------------------------------------------------------


def valid_lines(n=1000):
    with open(data("cooking.valid"), encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f][:n]


def assert_same_predictions(a, b, texts, k=3):
    for t in texts:
        la, pa = a.predict(t, k=k)
        lb, pb = b.predict(t, k=k)
        assert la == lb, t
        np.testing.assert_allclose(pa, pb, rtol=0, atol=1e-6, err_msg=t)


def test_saved_model_loads_in_cpp(trained):
    m, path = trained
    r = ref.load_model(str(path))
    texts = valid_lines()
    assert_same_predictions(m, r, texts, k=3)
    assert m.get_words(include_freq=True)[0] == r.get_words(include_freq=True)[0]
    assert m.get_labels() == r.get_labels()
    assert np.array_equal(m.get_input_matrix(), r.get_input_matrix())
    assert np.array_equal(m.get_output_matrix(), r.get_output_matrix())
    # and our own loader reads it back identically
    assert_same_predictions(m, fasttext_new.load_model(str(path)), texts[:200], k=3)


def test_saved_unsupervised_model_loads_in_cpp(tmp_path):
    m = fasttext_new.train_unsupervised(
        input=data("lid_raw.txt"), model="cbow", dim=16, epoch=1, minn=2, maxn=4, bucket=50_000,
        thread=THREADS, verbose=0,
    )
    path = tmp_path / "cbow.bin"
    m.save_model(str(path))
    r = ref.load_model(str(path))
    assert m.get_words() == r.get_words()
    for w in m.get_words()[:200:7] + ["unknownword"]:
        np.testing.assert_allclose(m.get_word_vector(w), r.get_word_vector(w), atol=1e-6)


# --------------------------------------------------------------------------------------------
# quantize: ours -> C++ and C++ -> ours
# --------------------------------------------------------------------------------------------

QUANT = {
    "default": dict(),
    "qnorm-cutoff-retrain": dict(qnorm=True, cutoff=20_000, retrain=True, epoch=5),
    "qout": dict(qout=True),
    "cutoff-dsub4": dict(cutoff=50_000, dsub=4),
}


@pytest.mark.parametrize("variant", list(QUANT))
def test_quantize_interop(trained, tmp_path, variant):
    _, path = trained
    kw = dict(QUANT[variant])
    if kw.get("retrain"):
        kw["input"] = data("cooking.train")
        kw["thread"] = THREADS
    valid, texts = data("cooking.valid"), valid_lines(500)

    # ours: quantize -> save -> C++ loads and predicts exactly like our in-memory model
    m = fasttext_new.load_model(str(path))
    p1_dense = m.test(valid)[1]
    m.quantize(**kw, verbose=0)
    assert m.is_quantized()
    out = tmp_path / "ours.ftz"
    m.save_model(str(out))
    assert out.stat().st_size < path.stat().st_size / 4
    r = ref.load_model(str(out))
    assert r.is_quantized()
    assert_same_predictions(m, r, texts)
    assert m.test(valid) == pytest.approx(r.test(valid), abs=1e-12)
    assert_same_predictions(m, fasttext_new.load_model(str(out)), texts[:200])
    p1_q = m.test(valid)[1]

    # C++: quantize the same .bin -> save -> we load it and predict exactly like C++
    out_cpp = tmp_path / "cpp.ftz"
    r = cpp("quantize", out_cpp, src=path, **kw, verbose=0)
    m2 = fasttext_new.load_model(str(out_cpp))
    assert_same_predictions(m2, r, texts)
    p1_q_cpp = r.test(valid)[1]
    print(f"QUANT {variant}: P@1 dense={p1_dense:.4f} ours={p1_q:.4f} cpp={p1_q_cpp:.4f}")
    # quantization quality: close to C++'s quantization of the same model, and not far below
    # the dense model (C++ itself loses 0.04-0.05 P@1 with cutoff=50000, dsub=4)
    assert abs(p1_q - p1_q_cpp) <= 0.03
    assert p1_q >= p1_dense - 0.07


def test_quantize_errors(trained):
    m = fasttext_new.load_model(str(trained[1]))
    with pytest.raises(ValueError):
        m.quantize(cutoff=100, retrain=True)  # needs input
    m.quantize(verbose=0)
    with pytest.raises(ValueError):
        m.quantize(verbose=0)  # already quantized
    cbow = fasttext_new.load_model(str(DATA / "fx_cbow.bin")) if (DATA / "fx_cbow.bin").exists() else None
    if cbow is not None:
        with pytest.raises(ValueError):
            cbow.quantize(verbose=0)


# --------------------------------------------------------------------------------------------
# API details: argument handling, attributes, verbose, GIL, Ctrl-C, pretrained vectors
# --------------------------------------------------------------------------------------------


def test_train_argument_handling():
    train = data("cooking.train")
    small = dict(epoch=1, thread=THREADS, verbose=0)
    m = fasttext_new.train_supervised(train, 0.5, 20, **small)  # positional: input, lr, dim
    assert m.get_dimension() == 20 and m.lr == 0.5 and m.dim == 20
    assert m.loss == "softmax" and m.wordNgrams == 1 and m.bucket == 0  # bucket=0 like C++
    assert m.minCount == 1 and m.minn == 0 and m.maxn == 0 and m.epoch == 1
    m = fasttext_new.train_supervised(input=train, word_ngrams=2, min_count=2, label_prefix="__label__", **small)
    assert m.wordNgrams == 2 and m.minCount == 2 and m.bucket == 2_000_000
    for bad in (dict(foo=1), dict(input=train, lr=0.1, wordNgrams=1, word_ngrams=1)):
        with pytest.raises(TypeError):
            fasttext_new.train_supervised(**{"input": train, **bad})
    with pytest.raises(TypeError):
        fasttext_new.train_supervised(train, input=train)
    with pytest.raises(ValueError):
        fasttext_new.train_supervised(input=train, loss="nope", **small)
    m = fasttext_new.train_unsupervised(data("cooking.train"), "cbow", dim=8, minCount=2, **small)
    assert m.f.args_summary()[0] == "cbow" and m.get_dimension() == 8 and m.loss == "ns"
    with pytest.raises(ValueError):
        fasttext_new.train_unsupervised(input=train, model="nope", **small)


def test_verbose_output(capfd):
    kw = dict(input=data("cooking.train"), epoch=2, thread=2)
    fasttext_new.train_supervised(**kw, verbose=0)
    assert capfd.readouterr().err == ""
    fasttext_new.train_supervised(**kw, verbose=2)
    err = capfd.readouterr().err
    assert "Number of words:  8952" in err and "Number of labels: 735" in err
    assert "Progress: 100.0%" in err


def test_training_releases_gil():
    done = threading.Event()
    result = {}

    def work():
        result["m"] = fasttext_new.train_supervised(input=data("cooking.train"), **COOK, thread=2, verbose=0)
        done.set()

    t = threading.Thread(target=work)
    start = time.perf_counter()
    t.start()
    ticks = 0
    while not done.is_set():
        ticks += 1  # pure-Python work that needs the GIL
    t.join()
    elapsed = time.perf_counter() - start
    assert elapsed > 0.5 and ticks > 100_000, (elapsed, ticks)
    assert result["m"].test(data("cooking.valid"))[1] > 0.5


def test_ctrl_c_stops_training():
    timer = threading.Timer(0.5, _thread.interrupt_main)
    timer.start()
    start = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        fasttext_new.train_supervised(input=data("cooking.train"), epoch=100_000, thread=2, verbose=0)
    assert time.perf_counter() - start < 10


def test_pretrained_vectors(tmp_path):
    # unsupervised vectors -> .vec -> supervised training initialised from them (both impls)
    u = fasttext_new.train_unsupervised(
        input=data("lid_raw.txt"), model="skipgram", dim=16, epoch=1, minCount=10, minn=2, maxn=4,
        bucket=50_000, thread=THREADS, verbose=0,
    )
    words = u.get_words()
    vec = tmp_path / "vectors.vec"
    with open(vec, "w", encoding="utf-8") as f:
        f.write(f"{len(words)} 16\n")
        for w in words:
            f.write(w + " " + " ".join(f"{x:.5f}" for x in u.get_word_vector(w)) + "\n")
    kw = dict(input=data("lid.train"), dim=16, epoch=2, lr=0.5, pretrainedVectors=str(vec), thread=THREADS, verbose=0)
    m = fasttext_new.train_supervised(**kw)
    r = cpp("train_supervised", tmp_path / "cpp.bin", **kw)
    # like C++, every pretrained word joins the vocabulary
    assert vocab(m) == vocab(r)
    p, rp = m.test(data("lid.valid"))[1], r.test(data("lid.valid"))[1]
    print(f"PRETRAINED: P@1 rs={p:.4f} cpp={rp:.4f}")
    assert abs(p - rp) <= 0.01
    with pytest.raises(ValueError):
        fasttext_new.train_supervised(**dict(kw, dim=8))  # dimension mismatch


def test_autotune_smoke():
    # (a 5 s budget barely gets past the default arguments; this only checks the plumbing)
    m = fasttext_new.train_supervised(
        input=data("cooking.train"), autotuneValidationFile=data("cooking.valid"), autotuneDuration=5,
        thread=THREADS, verbose=0,
    )
    assert len(m.get_labels()) == 735
    assert m.test(data("cooking.valid"))[1] > 0.1


# --------------------------------------------------------------------------------------------
# on_unicode_error: models whose dictionary contains invalid UTF-8
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dirty_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("dirty")
    lines = []
    for i in range(300):
        lab = [b"__label__clean", b"__label__caf\xe9", b"__label__\xff\xfe"][i % 3]
        lines.append(lab + b" word%d caf\xe9 \xc3( r\xe9sum\xe9 %d ok\n" % (i % 7, i % 11))
    train = d / "dirty.txt"
    train.write_bytes(b"".join(lines))
    path = d / "dirty.bin"
    cpp("train_supervised", path, input=str(train), epoch=3, minn=2, maxn=3, bucket=1000, wordNgrams=2, verbose=0)
    return path, d


def test_on_unicode_error(dirty_model):
    path, d = dirty_model
    r, m = ref.load_model(str(path)), fasttext_new.load_model(str(path))
    for f in (r, m):
        with pytest.raises(UnicodeDecodeError):
            f.get_words()
        with pytest.raises(UnicodeDecodeError):
            f.get_labels()
    for errors in ("replace", "ignore", "backslashreplace", "surrogateescape"):
        assert m.get_words(on_unicode_error=errors) == r.get_words(on_unicode_error=errors)
        assert m.get_labels(on_unicode_error=errors) == r.get_labels(on_unicode_error=errors)
        for t in ["word1 ok 3", "word2", "ok ok", ""]:
            a = r.predict(t, k=-1, on_unicode_error=errors)
            b = m.predict(t, k=-1, on_unicode_error=errors)
            assert a[0] == b[0] and np.allclose(a[1], b[1], atol=1e-6)
            assert r.predict([t], k=2, on_unicode_error=errors)[0] == m.predict([t], k=2, on_unicode_error=errors)[0]
    with pytest.raises(UnicodeDecodeError):
        m.predict("word3 ok", k=-1)
    # the raw bytes survive our save_model: C++ reads back the same dictionary
    out = d / "dirty_resaved.bin"
    m.save_model(str(out))
    r2 = ref.load_model(str(out))
    assert r2.get_words(on_unicode_error="surrogateescape") == r.get_words(on_unicode_error="surrogateescape")
    assert r2.predict("word4 ok", k=3, on_unicode_error="replace")[0] == m.predict("word4 ok", k=3, on_unicode_error="replace")[0]
