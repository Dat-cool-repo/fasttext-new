"""Parity tests: fasttext_new vs the original C++ `fasttext` package.

Models are looked up in $FASTTEXT_NEW_DATA (default: ./data in the repository) and
tests/data/. Missing models are skipped. Run `scripts/setup_env.sh` to download lid.176.* and
`python scripts/make_fixtures.py` to train the small per-loss fixtures.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fasttext_new

ref = pytest.importorskip("fasttext", reason="reference C++ package (fasttext-numpy2) not installed")
ref.FastText.eprint = lambda *a, **k: None  # silence load_model warning

HERE = Path(__file__).resolve().parent
DATA_DIRS = [Path(os.environ.get("FASTTEXT_NEW_DATA", HERE.parent / "data")), HERE / "data"]

# Probabilities / vectors must agree to this absolute tolerance (both sides are float32).
TOL = 1e-5


def find(name: str) -> Path | None:
    for d in DATA_DIRS:
        if (d / name).exists():
            return d / name
    return None


from golden_texts import EDGE_CASES  # noqa: E402  (shared with scripts/make_golden.py)


def load_corpus(limit: int | None = None) -> list[str]:
    lines = (HERE / "data" / "sentences.txt").read_text(encoding="utf-8").splitlines()
    lines += EDGE_CASES
    csv_path = find("papluca_test.csv")
    if csv_path is not None:
        with open(csv_path, encoding="utf-8") as f:
            lines += [" ".join(r["text"].split()) for r in csv.DictReader(f)]
    return lines[:limit] if limit else lines


CORPUS = load_corpus()
SMALL = CORPUS[:1500]

LID_MODELS = ["lid.176.ftz", "lid.176.bin"]
FIXTURES = [
    "fx_softmax_ng2.bin",
    "fx_hs_ng3.bin",
    "fx_ova_ng2.bin",
    "fx_ns_ng1.bin",
    "fx_softmax_nosub.bin",
    "fx_hs_many.bin",
    "fx_ova_many.bin",
    "fx_ns_many.bin",
    "fx_softmax_ng2_q.ftz",
    "fx_hs_ng3_q.ftz",
    "fx_hs_many_qout.ftz",
    "fx_ova_many_qout.ftz",
    "fx_ns_many_qout.ftz",
]
SUPERVISED = LID_MODELS + FIXTURES

_cache: dict = {}


def models(name: str):
    if name not in _cache:
        p = find(name)
        if p is None:
            pytest.skip(f"{name} not available")
        _cache[name] = (ref.load_model(str(p)), fasttext_new.load_model(str(p)))
    return _cache[name]


def corpus_for(name: str) -> list[str]:
    return CORPUS if name.startswith("lid") else SMALL


def assert_same_prediction(ref_labels, ref_probs, labels, probs, k, threshold, ctx):
    ref_probs = np.asarray(ref_probs, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    if list(ref_labels) == list(labels):
        np.testing.assert_allclose(probs, ref_probs, rtol=0, atol=TOL, err_msg=ctx)
        return
    # Differences are only acceptable between (near-)ties or at the k / threshold cut-off.
    r = dict(zip(ref_labels, ref_probs))
    o = dict(zip(labels, probs))
    for lab in set(r) & set(o):
        assert abs(r[lab] - o[lab]) <= TOL, ctx
    cut = min(list(ref_probs) + list(probs), default=threshold)
    for lab in set(r) ^ set(o):
        p = r.get(lab, o.get(lab))
        assert abs(p - cut) <= TOL or abs(p - threshold) <= TOL, f"{ctx}: {lab} {p}"
    np.testing.assert_allclose(np.sort(probs)[::-1][: len(ref_probs)], np.sort(ref_probs)[::-1][: len(probs)], atol=TOL, err_msg=ctx)


# --------------------------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SUPERVISED)
@pytest.mark.parametrize("k,threshold", [(1, 0.0), (5, 0.0), (-1, 0.0), (3, 0.1), (-1, -1.0), (2, -0.5)])
def test_predict_single(name, k, threshold):
    r, m = models(name)
    texts = corpus_for(name)
    if k == -1:
        # a negative threshold disables HS pruning: every label is returned
        texts = texts[:300] if threshold >= 0 else texts[:60]
    exact = 0
    max_diff = 0.0
    for t in texts:
        rl, rp = r.predict(t, k=k, threshold=threshold)
        ml, mp = m.predict(t, k=k, threshold=threshold)
        assert isinstance(ml, tuple) and isinstance(mp, np.ndarray)
        assert mp.dtype == rp.dtype == np.float64
        assert_same_prediction(rl, rp, ml, mp, k, threshold, f"{name} {t[:60]!r}")
        if list(rl) == list(ml):
            exact += 1
            if len(rp):
                max_diff = max(max_diff, float(np.abs(rp - mp).max()))
    # Report how many label lists are *exactly* identical (incl. order) and the max |dp|.
    print(f"PARITY {name} k={k} thr={threshold}: identical labels {exact}/{len(texts)}, max|dp|={max_diff:.2e}")


@pytest.mark.parametrize("name", SUPERVISED)
def test_predict_list_and_batch(name):
    r, m = models(name)
    texts = corpus_for(name)
    rl, rp = r.predict(texts, k=3)
    ml, mp = m.predict(texts, k=3)
    assert isinstance(ml, list) and isinstance(mp, list) and len(ml) == len(mp) == len(texts)
    assert all(isinstance(x, list) for x in ml)
    assert all(isinstance(x, np.ndarray) and x.dtype == np.float32 for x in mp)
    assert all(x.dtype == np.float32 for x in rp)
    for t, a, b, c, d in zip(texts, rl, rp, ml, mp):
        assert_same_prediction(a, b, c, d, 3, 0.0, f"{name} {t[:60]!r}")
    # predict_batch is the same computation, with an explicit thread count too
    for threads in (None, 1, 3):
        bl, bp = m.predict_batch(texts, k=3, threads=threads)
        assert bl == ml
        assert all(np.array_equal(x, y) for x, y in zip(bp, mp))


def test_lid_examples():
    r, m = models("lid.176.ftz")
    assert m.predict("Bonjour tout le monde")[0] == ("__label__fr",)
    labels, probs = m.predict_batch(["Hello world, how are you?", "Hola, ¿cómo estás?"], k=2)
    assert [x[0] for x in labels] == ["__label__en", "__label__es"]
    assert all(p.shape == (2,) for p in probs)


# --------------------------------------------------------------------------------------------
# vectors and dictionary
# --------------------------------------------------------------------------------------------

ALL_MODELS = SUPERVISED + ["fx_cbow.bin"]


def vocab_sample(r, n=300):
    words = r.get_words()
    return words[:: max(1, len(words) // n)] + ["xyzzyoov", "Überraschung", "日本語", "</s>", ""]


@pytest.mark.parametrize("name", ALL_MODELS)
def test_word_vectors(name):
    r, m = models(name)
    for w in vocab_sample(r):
        a, b = r.get_word_vector(w), m.get_word_vector(w)
        assert b.dtype == np.float32 and b.shape == (m.get_dimension(),)
        np.testing.assert_allclose(b, a, rtol=0, atol=TOL, err_msg=f"{name} {w!r}")


@pytest.mark.parametrize("name", ALL_MODELS)
def test_sentence_vectors(name):
    r, m = models(name)
    for t in SMALL[:400]:
        a, b = r.get_sentence_vector(t), m.get_sentence_vector(t)
        assert b.dtype == np.float32
        np.testing.assert_allclose(b, a, rtol=0, atol=TOL, err_msg=f"{name} {t[:60]!r}")


@pytest.mark.parametrize("name", ALL_MODELS)
def test_dictionary(name):
    r, m = models(name)
    assert m.get_dimension() == r.get_dimension()
    assert m.is_quantized() == r.is_quantized()
    assert m.get_words() == r.get_words()
    assert m.get_labels() == r.get_labels()
    w1, f1 = m.get_words(include_freq=True)
    w2, f2 = r.get_words(include_freq=True)
    assert w1 == w2 and np.array_equal(f1, f2) and f1.dtype == f2.dtype
    l1, g1 = m.get_labels(include_freq=True)
    l2, g2 = r.get_labels(include_freq=True)
    assert l1 == l2 and np.array_equal(g1, g2)
    assert m.words == r.words and m.labels == r.labels
    for w in vocab_sample(r, 50):
        assert m.get_word_id(w) == r.get_word_id(w)
        assert np.array_equal(m.get_subwords(w)[1], r.get_subwords(w)[1])
        # (for pruned .ftz models C++ lists every n-gram string but only the surviving ids)
        assert m.get_subwords(w)[0] == r.get_subwords(w)[0]
        if m.f.args_summary()[5] > 0:  # bucket
            for sub in m.get_subwords(w)[0][1:4]:
                assert m.get_subword_id(sub) == r.get_subword_id(sub)
    for lab in r.get_labels()[:20]:
        assert m.get_label_id(lab) == r.get_label_id(lab)


@pytest.mark.parametrize("name", ["fx_softmax_ng2.bin", "fx_hs_ng3.bin", "fx_cbow.bin", "fx_hs_ng3_q.ftz"])
def test_matrices(name):
    r, m = models(name)
    if r.is_quantized():
        for f in (r, m):
            with pytest.raises(ValueError):
                f.get_input_matrix()
            with pytest.raises(ValueError):
                f.get_output_matrix()
    else:
        assert np.array_equal(m.get_input_matrix(), r.get_input_matrix())
        assert np.array_equal(m.get_output_matrix(), r.get_output_matrix())


def test_nearest_neighbors():
    r, m = models("fx_cbow.bin")
    for w in r.get_words()[5:25]:
        a = r.get_nearest_neighbors(w, k=5)
        b = m.get_nearest_neighbors(w, k=5)
        np.testing.assert_allclose([s for s, _ in b], [s for s, _ in a], atol=1e-4)


# --------------------------------------------------------------------------------------------
# errors and compatibility
# --------------------------------------------------------------------------------------------


def test_errors():
    r, m = models("lid.176.ftz")
    for f in (r, m):
        with pytest.raises(ValueError):
            f.predict("two\nlines")
        with pytest.raises(ValueError):
            f.predict(["ok", "two\nlines"])
        with pytest.raises(ValueError):
            f.get_sentence_vector("two\nlines")
    with pytest.raises(ValueError):
        fasttext_new.load_model("/nonexistent/model.bin")
    for f in (ref, fasttext_new):
        with pytest.raises(ValueError):
            f.train_supervised(input="/nonexistent/train.txt", verbose=0)
        with pytest.raises(ValueError):
            f.load_model(str(find("lid.176.ftz"))).test("/nonexistent/valid.txt")


def test_unsupervised_predict_raises():
    r, m = models("fx_cbow.bin")
    with pytest.raises(ValueError):
        m.predict("hello")
    assert m.get_labels() == r.get_labels()


def test_k_zero_and_empty_list():
    r, m = models("lid.176.ftz")
    for f in (r, m):
        with pytest.raises(ValueError):
            f.predict("hello", k=0)
    assert m.predict([], k=1) == ([], [])


def test_install_as_fasttext_shim():
    p = find("lid.176.ftz")
    if p is None:
        pytest.skip("lid.176.ftz not available")
    code = (
        "import fasttext_new; fasttext_new.install_as_fasttext()\n"
        "import fasttext\n"
        "assert fasttext is fasttext_new\n"
        f"print(fasttext.load_model({str(p)!r}).predict('Guten Morgen, wie geht es dir?')[0][0])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "__label__de"


# --------------------------------------------------------------------------------------------
# smaller API surface: tokenize, get_line, analogies, input vectors
# --------------------------------------------------------------------------------------------


def test_tokenize():
    for t in SMALL[:300] + EDGE_CASES + ["a\nb  c\n\nd", "x\n", "\n"]:
        assert fasttext_new.tokenize(t) == ref.tokenize(t), repr(t[:60])


@pytest.mark.parametrize("name", ["fx_ova_ng2.bin", "lid.176.ftz"])
def test_get_line(name):
    r, m = models(name)
    labels = r.get_labels()[:3]
    texts = [f"{labels[i % 3]} {t}" for i, t in enumerate(SMALL[:300])] + EDGE_CASES
    for t in texts:
        assert m.get_line(t) == r.get_line(t), repr(t[:60])
    assert m.get_line(texts[:20]) == r.get_line(texts[:20])
    with pytest.raises(ValueError):
        m.get_line("a\nb")


def test_analogies_and_input_vectors():
    r, m = models("fx_cbow.bin")
    words = r.get_words()[10:40]
    for a, b, c in zip(words[0::3], words[1::3], words[2::3]):
        x, y = r.get_analogies(a, b, c, k=5), m.get_analogies(a, b, c, k=5)
        np.testing.assert_allclose([s for s, _ in y], [s for s, _ in x], atol=1e-4)
    for name in ("fx_cbow.bin", "fx_hs_ng3_q.ftz"):
        r, m = models(name)
        for i in (0, 5, len(r.get_words()) + 3):
            np.testing.assert_allclose(m.get_input_vector(i), r.get_input_vector(i), atol=TOL)
        w = r.get_words()[7]
        np.testing.assert_allclose(m[w], r[w], atol=TOL)
        assert (w in m) and ("xyzzy-not-a-word" not in m)
