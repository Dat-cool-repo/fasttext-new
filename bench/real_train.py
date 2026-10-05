"""Train the same supervised classifier on real web text with fasttext_new and the C++ package.

Builds a 17-language identification task from FineWeb-2 / FineWeb parquet shards (the label is
the shard's language; the text is the first ``--chars`` characters of each document, newlines
replaced by spaces), splits it 90/10 and trains both implementations with identical arguments,
``--runs`` times each, reporting test P@1 / R@1, training time and model size.

    python bench/real_train.py --out-dir /tmp/realtrain PARQUET...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time

import pyarrow.parquet as pq

CPP_JOB = r"""
import json, sys, time, fasttext
job = json.loads(sys.argv[1])
t0 = time.perf_counter()
m = fasttext.train_supervised(**job["kwargs"])
dt = time.perf_counter() - t0
n, p, r = m.test(job["test"])
print(json.dumps(dict(seconds=dt, n=n, p1=p, r1=r)))
"""


def lang_of(path: str) -> str:
    m = re.search(r"fineweb2_([a-z]{3}_[A-Za-z]{4})_", os.path.basename(path))
    return m.group(1) if m else "eng_Latn"


def build(paths, per_lang: int, chars: int, out_dir: str, seed: int = 0):
    rng = random.Random(seed)
    rows = []
    for p in paths:
        lang = lang_of(p)
        n = 0
        for b in pq.ParquetFile(p).iter_batches(batch_size=4000, columns=["text"]):
            for t in b.column(0).to_pylist():
                t = " ".join(t[: chars * 2].split())[:chars]
                if t:
                    rows.append(f"__label__{lang} {t}")
                    n += 1
                if n >= per_lang:
                    break
            if n >= per_lang:
                break
    rng.shuffle(rows)
    cut = len(rows) // 10
    os.makedirs(out_dir, exist_ok=True)
    train, test = os.path.join(out_dir, "train.txt"), os.path.join(out_dir, "test.txt")
    with open(train, "w", encoding="utf-8") as f:
        f.write("\n".join(rows[cut:]) + "\n")
    with open(test, "w", encoding="utf-8") as f:
        f.write("\n".join(rows[:cut]) + "\n")
    return train, test, len(rows) - cut, cut


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-lang", type=int, default=20000)
    ap.add_argument("--chars", type=int, default=100)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("parquet", nargs="+")
    a = ap.parse_args()
    train, test, ntrain, ntest = build(a.parquet, a.per_lang, a.chars, a.out_dir)
    print(f"train {ntrain} lines, test {ntest} lines, {a.chars} chars each", flush=True)
    kwargs = dict(input=train, lr=0.5, epoch=5, dim=32, wordNgrams=2, minn=2, maxn=4, bucket=500_000,
                  thread=4, verbose=0)
    import fasttext_new

    res = {"rust": [], "cpp": []}
    for _ in range(a.runs):
        t0 = time.perf_counter()
        m = fasttext_new.train_supervised(**kwargs)
        dt = time.perf_counter() - t0
        n, p, r = m.test(test)
        res["rust"].append(dict(seconds=dt, n=n, p1=p, r1=r))
        print("rust", res["rust"][-1], flush=True)
        for _attempt in range(10):
            pr = subprocess.run([sys.executable, "-c", CPP_JOB, json.dumps(dict(kwargs=kwargs, test=test))],
                                capture_output=True, text=True)
            if pr.returncode == 0:
                res["cpp"].append(json.loads(pr.stdout.strip().splitlines()[-1]))
                print("cpp ", res["cpp"][-1], flush=True)
                break
            print("cpp failed:", pr.stderr.strip().splitlines()[-1:], flush=True)
    summary = {
        impl: dict(p1_mean=statistics.mean(x["p1"] for x in v), p1_runs=[round(x["p1"], 4) for x in v],
                   seconds_mean=round(statistics.mean(x["seconds"] for x in v), 2))
        for impl, v in res.items() if v
    }
    print(json.dumps(dict(train_lines=ntrain, test_lines=ntest, args=kwargs, summary=summary), indent=1))


if __name__ == "__main__":
    main()
