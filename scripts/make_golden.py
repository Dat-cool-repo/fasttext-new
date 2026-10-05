"""Record reference outputs of the original C++ `fasttext` package as JSON ("golden" files).

`tests/test_golden.py` compares fasttext_new against them on platforms where the C++ package
cannot be installed (Windows, macOS CI, free-threaded Python).

    python scripts/make_golden.py [DATA_DIR]

* ``tests/data/golden/lid.176.ftz.json`` (checked in, small): the checked-in sentences + edge
  cases.
* ``DATA_DIR/golden/<model>.json`` (local): more models and 1,000 corpus lines, ``test()``
  results, and the P@1 of C++ training runs on cooking.stackexchange.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from pathlib import Path

import fasttext as ref

ref.FastText.eprint = lambda *a, **k: None
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tests"))
from golden_texts import EDGE_CASES  # noqa: E402

MODELS = [
    "lid.176.ftz",
    "lid.176.bin",
    "fx_softmax_ng2.bin",
    "fx_hs_ng3.bin",
    "fx_ova_ng2.bin",
    "fx_ns_ng1.bin",
    "fx_softmax_ng2_q.ftz",
    "fx_hs_many_qout.ftz",
    "fx_cbow.bin",
]


def record(model_path: Path, texts: list[str], test_file: Path | None) -> dict:
    m = ref.load_model(str(model_path))
    out: dict = {"model": model_path.name, "dimension": m.get_dimension(), "quantized": m.is_quantized()}
    words = m.get_words()
    out["nwords"] = len(words)
    out["words_sample"] = words[:: max(1, len(words) // 50)][:50]
    out["word_vectors"] = {w: m.get_word_vector(w).tolist() for w in out["words_sample"][:20] + ["xyzzyoov", "日本語"]}
    out["sentence_vectors"] = [m.get_sentence_vector(t).tolist() for t in texts[:30]]
    if m.get_labels() != words:  # supervised
        out["labels"] = m.get_labels()
        out["predict"] = []
        for t in texts:
            labels, probs = m.predict(t, k=3)
            out["predict"].append([t, list(labels), [float(p) for p in probs]])
        if test_file is not None:
            out["test"] = {"file": test_file.name, "k1": list(m.test(str(test_file), k=1)), "k5": list(m.test(str(test_file), k=5))}
    return out


def main(data: Path) -> None:
    small = (HERE / "tests/data/sentences.txt").read_text(encoding="utf-8").splitlines() + EDGE_CASES
    big = list(small)
    if (data / "papluca_test.csv").exists():
        with open(data / "papluca_test.csv", encoding="utf-8") as f:
            big += [" ".join(r["text"].split()) for r in csv.DictReader(f)][:1000]

    checked_in = HERE / "tests/data/golden"
    checked_in.mkdir(parents=True, exist_ok=True)
    local = data / "golden"
    local.mkdir(parents=True, exist_ok=True)

    if (data / "lid.176.ftz").exists():
        g = record(data / "lid.176.ftz", small, None)
        (checked_in / "lid.176.ftz.json").write_text(json.dumps(g, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", checked_in / "lid.176.ftz.json")
    for name in MODELS:
        p = data / name
        if not p.exists():
            continue
        test_file = data / ("train_many.txt" if "many" in name else "train.txt")
        g = record(p, big, test_file if test_file.exists() and not name.startswith(("lid", "fx_cbow")) else None)
        (local / f"{name}.json").write_text(json.dumps(g, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", local / f"{name}.json")

    # C++ training quality on cooking (for the training check without the C++ package)
    if (data / "cooking.train").exists():
        p1 = []
        while len(p1) < 3:
            try:
                m = ref.train_supervised(input=str(data / "cooking.train"), lr=1.0, epoch=25, wordNgrams=2,
                                         bucket=200_000, thread=4, verbose=0)
            except RuntimeError:
                continue
            p1.append(m.test(str(data / "cooking.valid"))[1])
        g = {"args": dict(lr=1.0, epoch=25, wordNgrams=2, bucket=200_000), "p1": p1, "p1_mean": statistics.mean(p1)}
        (local / "train_cooking.json").write_text(json.dumps(g) + "\n", encoding="utf-8")
        print("wrote", local / "train_cooking.json", g["p1_mean"])


if __name__ == "__main__":
    default = os.environ.get("FASTTEXT_NEW_DATA", HERE / "data")
    main(Path(sys.argv[1] if len(sys.argv) > 1 else default))
