"""Language-ID agreement and throughput on real web text: fasttext_new vs the C++ package.

Streams parquet files (e.g. FineWeb / FineWeb-2 shards, column ``text``), replaces newlines by
spaces (as datatrove does before calling fastText) and predicts the top-1 label of every
document with one implementation per process, so that peak RSS is measured separately:

    python bench/real_agreement.py run  --impl rust --model lid.176.bin --out rust_bin.npz PARQUET...
    python bench/real_agreement.py run  --impl cpp  --model lid.176.bin --out cpp_bin.npz  PARQUET...
    python bench/real_agreement.py compare rust_bin.npz cpp_bin.npz PARQUET...

``--impl rust`` uses ``predict_batch`` (``--threads`` rayon threads); ``--impl cpp`` uses the
C++ package's ``predict(list)``. ``compare`` reports the exact agreement of labels and float32
probabilities and prints every mismatch.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import numpy as np
import pyarrow.parquet as pq


def batches(paths, batch_rows: int, limit_bytes: int | None):
    seen = 0
    for path in paths:
        f = pq.ParquetFile(path)
        for b in f.iter_batches(batch_size=batch_rows, columns=["text"]):
            texts = [t.replace("\n", " ") for t in b.column(0).to_pylist()]
            yield texts
            seen += sum(len(t.encode("utf-8")) for t in texts)
            if limit_bytes and seen >= limit_bytes:
                return


def run(args) -> None:
    if args.impl == "rust":
        import fasttext_new

        m = fasttext_new.load_model(args.model)
        predict = lambda texts: m.predict_batch(texts, k=1, threads=args.threads)  # noqa: E731
    else:
        import fasttext

        fasttext.FastText.eprint = lambda *a, **k: None
        m = fasttext.load_model(args.model)
        predict = lambda texts: m.predict(texts, k=1)  # noqa: E731
    labels = m.get_labels()
    index = {lab: i for i, lab in enumerate(labels)}
    lab_out, prob_out = [], []
    ndocs = nbytes = 0
    t_pred = 0.0
    for texts in batches(args.parquet, args.batch, args.limit_bytes):
        t0 = time.perf_counter()
        ls, ps = predict(texts)
        t_pred += time.perf_counter() - t0
        lab_out.append(np.array([index[x[0]] for x in ls], dtype=np.int16))
        prob_out.append(np.array([p[0] for p in ps], dtype=np.float32))
        ndocs += len(texts)
        nbytes += sum(len(t.encode("utf-8")) for t in texts)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    stats = dict(impl=args.impl, model=args.model, threads=args.threads, docs=ndocs, bytes=nbytes,
                 predict_seconds=round(t_pred, 2), docs_per_s=round(ndocs / t_pred),
                 mb_per_s=round(nbytes / t_pred / 1e6, 2), peak_rss_mb=round(rss_mb))
    np.savez(args.out, labels=np.concatenate(lab_out), probs=np.concatenate(prob_out),
             label_names=np.array(labels), stats=json.dumps(stats))
    print(json.dumps(stats))


def compare(args) -> None:
    a, b = np.load(args.a), np.load(args.b)
    assert list(a["label_names"]) == list(b["label_names"]), "label lists differ"
    la, lb, pa, pb = a["labels"], b["labels"], a["probs"], b["probs"]
    assert len(la) == len(lb)
    same_label = la == lb
    same_prob = pa.view(np.uint32) == pb.view(np.uint32)
    diff = np.abs(pa.astype(np.float64) - pb.astype(np.float64))
    print(json.dumps(dict(
        docs=int(len(la)),
        label_agreement=float(same_label.mean()),
        label_mismatches=int((~same_label).sum()),
        prob_bit_identical=float(same_prob.mean()),
        prob_mismatches=int((~same_prob).sum()),
        max_abs_prob_diff=float(diff.max()) if len(diff) else 0.0,
    )))
    bad = np.flatnonzero(~(same_label & same_prob))
    if len(bad):
        names = a["label_names"]
        want = set(bad[: args.show].tolist())
        i = 0
        for texts in batches(args.parquet, 2000, None):
            for t in texts:
                if i in want:
                    print(f"#{i}: {names[la[i]]} {pa[i]!r} vs {names[lb[i]]} {pb[i]!r}: {t[:300]!r}")
                i += 1
            if i > max(want):
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--impl", choices=["rust", "cpp"], required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--batch", type=int, default=2000)
    r.add_argument("--limit-bytes", type=int, default=None)
    r.add_argument("parquet", nargs="+")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--show", type=int, default=20)
    c.add_argument("parquet", nargs="*")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
