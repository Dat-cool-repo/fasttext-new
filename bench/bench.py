"""Throughput benchmark: fasttext_new vs the original C++ `fasttext` package.

    python bench/bench.py [--data DIR] [--models lid.176.ftz lid.176.bin] [--threads 4] [--repeat 3]

Measures documents/second on the multilingual parity corpus (10k sentences) for:
  * single-thread `predict(str)` in a Python loop (both packages)
  * `predict(list)` (C++ multilinePredict, single-threaded) vs `predict_batch(threads=1)`
  * `predict_batch(threads=N)` (rayon, GIL released)
  * N Python threads calling `predict(str)` (shows whether the GIL is released)
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fasttext_new

try:
    import fasttext as ref

    ref.FastText.eprint = lambda *a, **k: None
except ImportError:  # pragma: no cover
    ref = None

HERE = Path(__file__).resolve().parent.parent


def corpus(data: Path) -> list[str]:
    p = data / "papluca_test.csv"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return [" ".join(r["text"].split()) for r in csv.DictReader(f)]
    lines = (HERE / "tests/data/sentences.txt").read_text(encoding="utf-8").splitlines()
    return (lines * (10_000 // len(lines) + 1))[:10_000]


def best_of(fn, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def run(model_path: Path, docs: list[str], threads: int, repeat: int) -> list[tuple[str, str, float]]:
    rows = []
    n = len(docs)

    def loop(m):
        return lambda: [m.predict(d, k=1) for d in docs]

    def py_threads(m):
        chunks = [docs[i::threads] for i in range(threads)]

        def work(chunk):
            for d in chunk:
                m.predict(d, k=1)

        def go():
            with ThreadPoolExecutor(threads) as ex:
                list(ex.map(work, chunks))

        return go

    impls = []
    if ref is not None:
        impls.append(("C++ fasttext", ref.load_model(str(model_path))))
    impls.append(("fasttext_new", fasttext_new.load_model(str(model_path))))

    for name, m in impls:
        loop(m)()  # warm-up
        rows.append((name, "predict(str) loop, 1 thread", n / best_of(loop(m), repeat)))
        if name.startswith("C++"):
            rows.append((name, "predict(list), 1 thread", n / best_of(lambda: m.predict(docs, k=1), repeat)))
        else:
            rows.append((name, "predict_batch, threads=1", n / best_of(lambda: m.predict_batch(docs, k=1, threads=1), repeat)))
            rows.append((name, f"predict_batch, threads={threads}", n / best_of(lambda: m.predict_batch(docs, k=1, threads=threads), repeat)))
        rows.append((name, f"predict(str), {threads} Python threads", n / best_of(py_threads(m), repeat)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("FASTTEXT_NEW_DATA", HERE / "data"))
    ap.add_argument("--models", nargs="+", default=["lid.176.ftz", "lid.176.bin"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--doc-chars", type=int, default=0, help="join sentences into docs of ~N chars (web-page-like inputs)")
    a = ap.parse_args()
    data = Path(a.data)
    docs = corpus(data)
    if a.doc_chars:
        joined, cur = [], []
        for s in docs:
            cur.append(s)
            if sum(map(len, cur)) >= a.doc_chars:
                joined.append(" ".join(cur))
                cur = []
        docs = joined
    avg_chars = sum(map(len, docs)) / len(docs)
    print(f"{len(docs)} docs, avg {avg_chars:.0f} chars; {platform.processor() or platform.machine()}, {len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()} usable CPUs\n")
    print("| model | implementation | mode | docs/s |")
    print("|---|---|---|--:|")
    for model in a.models:
        p = data / model
        if not p.exists():
            print(f"(skipping {model}: not found)")
            continue
        for impl, mode, dps in run(p, docs, a.threads, a.repeat):
            print(f"| {model} | {impl} | {mode} | {dps:,.0f} |")


if __name__ == "__main__":
    main()
