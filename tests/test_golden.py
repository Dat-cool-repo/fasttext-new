"""Parity against recorded outputs of the C++ package (no C++ package needed).

Golden files are written by ``scripts/make_golden.py`` on a machine with ``fasttext-numpy2``:
``tests/data/golden/*.json`` (checked in) and ``$FASTTEXT_NEW_DATA/golden/*.json`` (local). Each
test is skipped when its model file is not available. This is what runs on Windows, macOS CI
and free-threaded Python, where the C++ package cannot be installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import fasttext_new

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FASTTEXT_NEW_DATA", HERE.parent / "data"))
TOL = 1e-5


def golden_files() -> dict[str, Path]:
    files = {}
    for d in (HERE / "data" / "golden", DATA / "golden"):  # local (bigger) files win
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                if p.name != "train_cooking.json":
                    files[p.name[: -len(".json")]] = p
    return files


GOLDEN = golden_files()
_models: dict = {}


def load(name: str):
    g = json.loads(GOLDEN[name].read_text(encoding="utf-8"))
    for d in (DATA, HERE / "data"):
        if (d / g["model"]).exists():
            if name not in _models:
                _models[name] = fasttext_new.load_model(str(d / g["model"]))
            return g, _models[name], d
    pytest.skip(f"model {g['model']} not available")


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_predictions(name):
    g, m, _ = load(name)
    if "predict" not in g:
        pytest.skip("unsupervised model")
    assert m.get_labels() == g["labels"]
    texts = [t for t, _, _ in g["predict"]]
    for t, labels, probs in g["predict"]:
        ml, mp = m.predict(t, k=3)
        assert list(ml) == labels, repr(t[:60])
        np.testing.assert_allclose(mp, probs, rtol=0, atol=TOL, err_msg=repr(t[:60]))
    # batch path (rayon, GIL released) gives the same answers
    bl, bp = m.predict(texts, k=3)
    assert bl == [labels for _, labels, _ in g["predict"]]
    for p, (_, _, probs) in zip(bp, g["predict"]):
        np.testing.assert_allclose(p, probs, rtol=0, atol=TOL)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_vectors_and_vocab(name):
    g, m, _ = load(name)
    assert m.get_dimension() == g["dimension"]
    assert m.is_quantized() == g["quantized"]
    words = m.get_words()
    assert len(words) == g["nwords"]
    assert words[:: max(1, len(words) // 50)][:50] == g["words_sample"]
    for w, v in g["word_vectors"].items():
        np.testing.assert_allclose(m.get_word_vector(w), v, rtol=0, atol=TOL, err_msg=w)
    texts = g.get("sentence_texts") or [t for t, _, _ in g.get("predict", [])][:30]
    for t, v in zip(texts, g["sentence_vectors"]):
        np.testing.assert_allclose(m.get_sentence_vector(t), v, rtol=0, atol=TOL)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_test_metrics(name):
    g, m, d = load(name)
    if "test" not in g or not (d / g["test"]["file"]).exists():
        pytest.skip("no test() reference")
    f = str(d / g["test"]["file"])
    assert list(m.test(f, k=1)) == pytest.approx(g["test"]["k1"], abs=1e-12)
    assert list(m.test(f, k=5)) == pytest.approx(g["test"]["k5"], abs=1e-12)


def test_training_quality_vs_recorded_cpp():
    ref_file = DATA / "golden" / "train_cooking.json"
    if not ref_file.exists() or not (DATA / "cooking.train").exists():
        pytest.skip("train_cooking.json / cooking data not available")
    g = json.loads(ref_file.read_text())
    m = fasttext_new.train_supervised(input=str(DATA / "cooking.train"), **g["args"], thread=4, verbose=0)
    p1 = m.test(str(DATA / "cooking.valid"))[1]
    print(f"TRAIN cooking: P@1 {p1:.4f} vs recorded C++ {g['p1_mean']:.4f}")
    assert abs(p1 - g["p1_mean"]) <= 0.02


def _need(g: dict, key: str) -> None:
    if key not in g:
        pytest.skip(f"no {key} reference in this golden file")


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_predict_all_labels(name):
    """k=-1 with a threshold: every label above it, in the same order."""
    g, m, _ = load(name)
    _need(g, "predict_all")
    for t, labels, probs in g["predict_all"]:
        ml, mp = m.predict(t, k=-1, threshold=0.05)
        assert list(ml) == labels, repr(t[:60])
        np.testing.assert_allclose(mp, probs, rtol=0, atol=TOL)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_test_label(name):
    g, m, d = load(name)
    _need(g, "test_label")
    got = m.test_label(str(d / g["test"]["file"]), k=1)
    assert sorted(got) == sorted(g["test_label"])
    for label, (p, r, f1) in g["test_label"].items():
        want = [p, r, f1]
        have = [got[label]["precision"], got[label]["recall"], got[label]["f1score"]]
        np.testing.assert_allclose(have, want, rtol=0, atol=1e-12, err_msg=label)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_subwords_and_ids(name):
    g, m, _ = load(name)
    _need(g, "subwords")
    for w, (strings, ids) in g["subwords"].items():
        s, i = m.get_subwords(w)
        assert list(s) == strings, w
        assert [int(x) for x in i] == ids, w
    for w, i in g["word_ids"].items():
        assert m.get_word_id(w) == i, w
    for w, i in g.get("subword_ids", {}).items():
        assert m.get_subword_id(w) == i, w


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_nearest_neighbors(name):
    g, m, _ = load(name)
    _need(g, "nn")
    for w, want in g["nn"].items():
        got = m.get_nearest_neighbors(w, k=5)
        assert [x for _, x in got] == [x for _, x in want], w
        np.testing.assert_allclose([s for s, _ in got], [s for s, _ in want], rtol=0, atol=TOL)
    got = m.get_analogies("apple", "banana", "guitar", k=5)
    assert [x for _, x in got] == [x for _, x in g["analogies"]]
    np.testing.assert_allclose([s for s, _ in got], [s for s, _ in g["analogies"]], rtol=0, atol=TOL)
