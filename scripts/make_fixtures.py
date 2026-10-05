"""Train small fastText models with the *original C++* package to use as parity fixtures.

Covers every output layer (softmax / hs / ova / ns), word n-grams > 1 (exercises the
`</s>` bigram), quantized models with and without `qout`, and an unsupervised cbow model.

Usage (inside the project venv, which has `fasttext-numpy2` installed):
    python scripts/make_fixtures.py [DATA_DIR]
Writes `fx_*.bin` / `fx_*.ftz` and `train.txt` to DATA_DIR (default $FASTTEXT_NEW_DATA or
./data in the repository).

The C++ trainer sporadically aborts with "Encountered NaN" on these tiny runs, so every model
is trained in a fresh subprocess and retried.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

SUPERVISED = {
    "fx_ova_ng2": dict(loss="ova", wordNgrams=2, lr=0.5),
    "fx_softmax_ng2": dict(loss="softmax", wordNgrams=2),
    "fx_hs_ng3": dict(loss="hs", wordNgrams=3),
    "fx_ns_ng1": dict(loss="ns", wordNgrams=1, neg=5),
    "fx_softmax_nosub": dict(loss="softmax", wordNgrams=1, minn=0, maxn=0, epoch=10),
    # >= 256 labels so the output matrix can be quantized (qout)
    "fx_hs_many": dict(loss="hs", wordNgrams=2, many=True),
    "fx_ova_many": dict(loss="ova", wordNgrams=2, lr=0.5, many=True),
    "fx_ns_many": dict(loss="ns", wordNgrams=1, neg=5, many=True),
}
QUANTIZED = {
    "fx_softmax_ng2_q": ("fx_softmax_ng2", dict(qnorm=True, cutoff=20_000, retrain=False)),
    "fx_hs_ng3_q": ("fx_hs_ng3", dict()),
    "fx_hs_many_qout": ("fx_hs_many", dict(qout=True, qnorm=True)),
    "fx_ova_many_qout": ("fx_ova_many", dict(qout=True)),
    "fx_ns_many_qout": ("fx_ns_many", dict(qout=True, cutoff=10_000)),
}


def corpus_lines(data: Path):
    csv_path = data / "papluca_test.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = " ".join(row["text"].split())
                if text:
                    yield row["labels"], text
    else:  # fall back to the small checked-in corpus with a fake 3-way label
        for i, line in enumerate((HERE / "tests/data/sentences.txt").read_text(encoding="utf-8").splitlines()):
            yield f"c{i % 3}", line


def worker(job: dict) -> None:
    """Runs in a subprocess: train / quantize one model."""
    import fasttext

    kind, out = job["kind"], job["out"]
    if kind == "sup":
        m = fasttext.train_supervised(**job["args"])
    elif kind == "cbow":
        m = fasttext.train_unsupervised(**job["args"])
    else:
        m = fasttext.load_model(job["base"])
        m.quantize(**job["args"])
    m.save_model(out)


def run(job: dict, attempts: int = 10) -> None:
    for i in range(attempts):
        p = subprocess.run([sys.executable, __file__, "--job", json.dumps(job)], capture_output=True, text=True)
        if p.returncode == 0:
            print("wrote", Path(job["out"]).name, f"(attempt {i + 1})")
            return
        err = (p.stderr.strip().splitlines() or ["?"])[-1]
        print(f"  {Path(job['out']).name}: {err}; retrying")
    raise RuntimeError(f"could not build {job['out']}")


def main(data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    train = data / "train.txt"
    lines = list(corpus_lines(data))
    with open(train, "w", encoding="utf-8") as f:
        for i, (lab, text) in enumerate(lines):
            extra = f" __label__x{i % 2}" if i % 5 == 0 else ""  # multi-label rows (for OVA)
            f.write(f"__label__{lab}{extra} {text}\n")
    many = data / "train_many.txt"
    with open(many, "w", encoding="utf-8") as f:
        for i, (lab, text) in enumerate(lines):
            f.write(f"__label__{lab}{i % 15} {text}\n")  # 20 languages x 15 = 300 labels
    raw = data / "raw.txt"
    raw.write_text("\n".join(t for _, t in lines) + "\n", encoding="utf-8")

    common = dict(input=str(train), dim=16, epoch=5, minn=2, maxn=4, bucket=100_000, thread=4, verbose=0)
    for name, kw in SUPERVISED.items():
        kw = dict(kw)
        src = many if kw.pop("many", False) else train
        out = data / f"{name}.bin"
        if not out.exists():
            run(dict(kind="sup", out=str(out), args={**common, **kw, "input": str(src)}))
    for name, (base, kw) in QUANTIZED.items():
        run(dict(kind="quant", base=str(data / f"{base}.bin"), out=str(data / f"{name}.ftz"), args=dict(input=str(train), **kw)))
    cbow = dict(input=str(raw), model="cbow", dim=16, epoch=2, minn=2, maxn=4, bucket=100_000, thread=4, verbose=0)
    run(dict(kind="cbow", out=str(data / "fx_cbow.bin"), args=cbow))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--job":
        worker(json.loads(sys.argv[2]))
    else:
        default = os.environ.get("FASTTEXT_NEW_DATA", HERE / "data")
        main(Path(sys.argv[1] if len(sys.argv) > 1 else default))
