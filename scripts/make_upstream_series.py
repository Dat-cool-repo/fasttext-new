"""Split vendor/fasttext (patched crates.io `fasttext` 0.8.0) into an upstreamable patch series.

    python3 scripts/make_upstream_series.py [--test]

* Diffs the pristine crate (``$FASTTEXT_NEW_UPSTREAM_DIR/fasttext-0.8.0``, default
  ``~/target/fasttext-python-bindings/upstream``) against ``vendor/fasttext/src`` with zero
  context, assigns every hunk to one topic (fails if a hunk is unassigned), and builds the
  cumulative tree after each topic.
* Writes ``upstream/NNNN-<topic>.patch`` (``git am``-compatible, paths relative to the crate /
  repository root, normal 3-line context) and ``upstream/series``.
* Upstream copies drop our ``[fasttext-python-bindings patch]`` marker comments (and reword two
  comments that describe the old code); otherwise the series reproduces ``vendor/fasttext/src``
  byte for byte, which is checked.
* ``--test``: runs the crate's own test suite (``cargo test --release``) on every intermediate
  tree, so each patch builds and passes on its own.

Nothing is committed anywhere; the patches are plain files.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
VENDOR = PROJ / "vendor" / "fasttext"
BASE = Path(os.environ.get("FASTTEXT_NEW_UPSTREAM_DIR", Path.home() / "target/fasttext-python-bindings/upstream"))
PRISTINE = BASE / "fasttext-0.8.0"
WORK = BASE / "series-work"
OUT = PROJ / "upstream"
AUTHOR = os.environ.get("FASTTEXT_NEW_PATCH_AUTHOR", "Dat-cool-repo <Dat-cool-repo@users.noreply.github.com>")
TRAILER = (
    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
    "Claude-Session: https://claude.ai/code/session_01QeswAUonCzqFJEnwyzk7PA\n"
)

# --------------------------------------------------------------------------------------------
# Topics, in series order: (slug, subject, body)
# --------------------------------------------------------------------------------------------
TOPICS = [
    (
        "fix-quantized-hs-ns-predict",
        "Fix quantized (.ftz) prediction for hierarchical-softmax and negative-sampling models",
        """\
`predict_raw_quantized` always applied a softmax over the first `nlabels` rows of the output
matrix. That is only correct for softmax (and the sigmoid for one-vs-all). For hierarchical
softmax those rows are Huffman-tree nodes, and negative sampling is a binary logistic per label,
so every HS / NS `.ftz` model predicted garbage. `lid.176.ftz` (the official language-ID model,
HS) returned `__label__zh` for French text.

Like C++ `Model::predict`, delegate to the model's real loss whenever the output matrix is
dense (`qout=false`). For HS / NS with a quantized output matrix (`qout=true`), run the loss
over a dequantized copy of the output matrix, built lazily once (`qout_model`) and reset when
the model is re-quantized.

Verified against the C++ package: `lid.176.ftz` top-k labels and probabilities are identical
on 10,121 multilingual lines, as are HS/NS fixture models with and without `qout`.
""",
    ),
    (
        "bit-exact-scores",
        "Make prediction scores bit-identical to C++ fastText",
        """\
Several small numeric differences made probabilities differ from C++ in the last bits. They also
changed the order of tied predictions, which is common with the table-based sigmoid of OVA / NS
models:

* `std_log` is computed in f64 like C++ (`std::log(x + 1e-5)` with a double literal).
* top-k selection (`find_k_best`) and the HS DFS use a port of libstdc++'s
  `push_heap` / `pop_heap` / `sort_heap` with C++'s comparator, so ties come out in the same
  order as C++.
* The exact HS sigmoid divides in double (`1. / (1 + std::exp(-f))`).
* softmax divides by `z` (C++ `output[i] /= z`) instead of multiplying by `1/z`.
* The output-layer dot product accumulates sequentially like C++ `DenseMatrix::dotRow`. The
  SIMD/FMA version is kept behind a new `simd-dot` feature. The hidden-layer average dominates
  inference time, so the cost is small.

With these changes `lid.176.bin` / `lid.176.ftz` predictions (labels, order and
probabilities) are identical to the C++ package (max |dp| = 0) on 10,121 lines.

Open question for the maintainers: whether exact parity or the SIMD dot product should be the
default (this patch makes parity the default).
""",
    ),
    (
        "raw-predict-and-input-rows",
        "Add predict_on_words_raw() and add_input_rows() for allocation-free bindings",
        """\
* `FastText::predict_on_words_raw(word_ids, k, threshold) -> Vec<(log_prob, label_index)>` is
  `predict_on_words` without allocating a label `String` per prediction. Language bindings can
  intern the label strings once. The threshold is passed through unclamped like C++: a negative
  threshold filters nothing, and `std_log(threshold)` becomes NaN, which disables the HS pruning,
  so `k = -1` returns every label.
* `FastText::add_input_rows(ids, out)` adds input-matrix rows to `out` for dense and quantized
  models alike (C++ `addInputVector`). Callers can then compute sentence / word vectors exactly
  like C++, including for `.ftz` models.
""",
    ),
    (
        "faster-subword-hashing",
        "Speed up tokenization: incremental n-gram hashing, cheap hasher for pruneidx",
        """\
* `for_each_ngram` copied every character n-gram into a fresh `Vec` and hashed it from scratch.
  An n-gram starting at a position only grows, so its FNV-1a hash is now updated incrementally
  (same values as `utils::hash`) and the n-gram bytes are borrowed from the word.
* `pruneidx` (consulted once per n-gram of every OOV token in a quantized model) used SipHash.
  It now uses a cheap multiplicative hasher for its `i32` keys (`PruneIdx` type alias).

Tokenizing for `lid.176.ftz` gets about 3x faster, and its predict throughput goes from below
C++ to about 2x above it.
""",
    ),
    (
        "training-os-threads-and-progress",
        "Train on dedicated OS threads and print C++-style progress (verbose)",
        """\
* The Hogwild! workers were tasks on rayon's global pool. With fewer pool threads than
  `args.thread` (e.g. `RAYON_NUM_THREADS=4`, `thread=12`, or training started from inside a
  rayon task) they ran one after another. The later ones found the token budget exhausted,
  which changes the learning-rate schedule and the data partitioning. Like C++ `startThreads`,
  every worker now gets its own scoped OS thread. The same applies to the retraining in
  `quantize`, which also honours `verbose` now.
* `verbose` was ignored. Like C++, `verbose > 0` prints the dictionary summary ("Read NM words /
  Number of words / Number of labels") and a final progress line. `verbose > 1` also prints a
  `Progress: ... words/sec/thread ... lr ... avg.loss ... ETA` line every 100 ms from a monitor
  loop, using thread 0's running loss as C++ does.
""",
    ),
    (
        "parallel-matrix-init",
        "Initialise the input matrix in parallel",
        """\
`DenseMatrix::uniform` fills 10 blocks, each with its own `minstd_rand` seeded with
`block + seed`. The blocks are independent, so large matrices now fill them in parallel on the
rayon pool. The values are unchanged. For a 2M x 100 matrix (`wordNgrams > 1` with the default
bucket count) this removes about 0.5 s of single-threaded start-up from every training run.

(For reference: C++ only fills the first `thread` blocks when `thread < 10`, leaving the rest of
the matrix at zero. This crate fills all 10, which seems the better choice.)
""",
    ),
    (
        "pretrained-vectors-join-dictionary",
        "pretrainedVectors: add the .vec words to the dictionary like C++",
        """\
C++ `getInputMatrixFromFile` adds every word of the pretrained `.vec` file to the dictionary
(`threshold(1, 0)` + `init()`) before sizing the input matrix. Pretrained words that are missing
from the training data, or below `minCount`, therefore keep their vectors. The crate only
overwrote the rows of words already in the vocabulary, so vocabularies (and models) differed
from C++ whenever `pretrainedVectors` was used. The `.vec` file is now read first, its words
added, and the matrix built from the final dictionary. Verified: identical vocabularies and
counts to C++ on a 20-language dataset.
""",
    ),
    (
        "quantize-args-and-memory",
        "quantize(): keep the retrain arguments like C++, free the dense input matrix",
        """\
* With `retrain`, C++ overwrites `epoch`, `lr`, `thread` and `verbose` in the model's args
  (`epoch` is part of the saved file). The crate kept the old values, so a model quantized with
  `retrain=true, epoch=E` saved a different header than C++.
* After quantization the dense input matrix was kept alive, although the quantized prediction
  path never reads it, so a freshly quantized model kept the whole dense matrix (often hundreds
  of MB) next to the few-MB quantized one. It is now dropped, as C++ moves it into the
  `QuantMatrix`.
""",
    ),
    (
        "dictionary-non-utf8-words",
        "Load models whose dictionary contains words that are not valid UTF-8",
        """\
C++ fastText treats words as raw bytes, so models trained on web text often contain tokens that
are not valid UTF-8 (truncated multi-byte sequences, Latin-1 text, ...). The crate refused to
load such models ("Invalid UTF-8 in dictionary word").

`Entry` gains `raw: Option<Box<[u8]>>` with the exact bytes when they are not valid UTF-8
(`word` then holds a lossy copy), plus `Entry::bytes()`. Hashing, hash-table probing, character
n-grams and saving all use the exact bytes. Such models therefore load, predict and round-trip
through `save` exactly like in C++, and bindings can decode the words with the caller's error
policy (e.g. Python's `on_unicode_error`).

Not covered: training files with invalid UTF-8 are still tokenized lossily (U+FFFD), as before.
Making the training tokenizer byte-based is a larger change.
""",
    ),
]

# Hunk (file, old range of `diff -U0`) -> topic slug.
Q, X, R, P, TR, PI, PV, QA, U8 = (t[0] for t in TOPICS)
ASSIGN = {
    # dictionary.rs
    ("dictionary.rs", "-3,0"): P, ("dictionary.rs", "-88"): P, ("dictionary.rs", "-112"): P,
    ("dictionary.rs", "-560"): P, ("dictionary.rs", "-565"): P, ("dictionary.rs", "-568"): P,
    ("dictionary.rs", "-573,2"): P, ("dictionary.rs", "-997"): P, ("dictionary.rs", "-1004"): P,
    ("dictionary.rs", "-1122,2"): P,
    ("dictionary.rs", "-53"): U8, ("dictionary.rs", "-54,0"): U8, ("dictionary.rs", "-62,0"): U8,
    ("dictionary.rs", "-129,0"): U8, ("dictionary.rs", "-132"): U8, ("dictionary.rs", "-149,0"): U8,
    ("dictionary.rs", "-280"): U8, ("dictionary.rs", "-538"): U8, ("dictionary.rs", "-550"): U8,
    ("dictionary.rs", "-588,0"): U8, ("dictionary.rs", "-605"): U8, ("dictionary.rs", "-608"): U8,
    ("dictionary.rs", "-618"): U8, ("dictionary.rs", "-736"): U8, ("dictionary.rs", "-1037,3"): U8,
    ("dictionary.rs", "-1065"): U8, ("dictionary.rs", "-1097,3"): U8, ("dictionary.rs", "-1114,0"): U8,
    # fasttext/mod.rs
    ("fasttext/mod.rs", "-122,0"): TR, ("fasttext/mod.rs", "-169,0"): Q, ("fasttext/mod.rs", "-304,0"): Q,
    ("fasttext/mod.rs", "-410,0"): R,
    # fasttext/predict.rs
    ("fasttext/predict.rs", "-76,0"): R, ("fasttext/predict.rs", "-142,7"): Q,
    ("fasttext/predict.rs", "-150,5"): Q, ("fasttext/predict.rs", "-168"): Q, ("fasttext/predict.rs", "-174"): Q,
    # fasttext/quantize.rs
    ("fasttext/quantize.rs", "-1"): TR, ("fasttext/quantize.rs", "-4,2"): TR,
    ("fasttext/quantize.rs", "-80,0"): QA, ("fasttext/quantize.rs", "-123,0"): QA,
    ("fasttext/quantize.rs", "-144,0"): Q, ("fasttext/quantize.rs", "-150,0"): QA,
    ("fasttext/quantize.rs", "-152,0"): QA, ("fasttext/quantize.rs", "-176,0"): TR,
    ("fasttext/quantize.rs", "-187,0"): TR, ("fasttext/quantize.rs", "-189,23"): TR,
    ("fasttext/quantize.rs", "-215,0"): TR,
    # fasttext/train.rs
    ("fasttext/train.rs", "-3"): TR, ("fasttext/train.rs", "-5,2"): TR, ("fasttext/train.rs", "-73"): TR,
    ("fasttext/train.rs", "-131,0"): PV, ("fasttext/train.rs", "-140,2"): PV,
    ("fasttext/train.rs", "-189,0"): TR, ("fasttext/train.rs", "-191,23"): TR,
    ("fasttext/train.rs", "-217,0"): TR, ("fasttext/train.rs", "-221,0"): TR,
    ("fasttext/train.rs", "-235,2"): (TR, PV),  # verbose summary + pretrained (split below)
    ("fasttext/train.rs", "-262,0"): Q,
    ("fasttext/train.rs", "-329,9"): PV, ("fasttext/train.rs", "-363,7"): PV, ("fasttext/train.rs", "-372"): PV,
    ("fasttext/train.rs", "-377,6"): PV, ("fasttext/train.rs", "-396,0"): PV, ("fasttext/train.rs", "-404"): PV,
    ("fasttext/train.rs", "-407"): PV, ("fasttext/train.rs", "-617,0"): TR,
    # loss.rs, simd, utils
    **{("loss.rs", k): X for k in ("-22", "-43,0", "-46", "-176,11", "-189", "-192,3", "-197,8", "-570",
                                   "-576,6", "-583", "-588,0", "-591", "-662,2")},
    ("simd/mod.rs", "-0,0"): X, ("simd/mod.rs", "-64,0"): X, ("utils.rs", "-103"): X, ("utils.rs", "-105"): X,
    # matrix.rs
    ("matrix.rs", "-239,7"): PI, ("matrix.rs", "-248"): PI, ("matrix.rs", "-252,0"): PI,
}


# Marker comments are for the vendored copy only; upstream patches drop them.
MARKER = "[fasttext-python-bindings patch]"
REWORD = {
    "        // The original code below always applied a softmax\n"
    "        // (or OVA sigmoid) over the first `nlabels` output rows. That is wrong for\n"
    "        // hierarchical softmax (output rows are Huffman-tree nodes, e.g. lid.176.ftz) and for\n"
    "        // negative sampling (binary logistic). Like C++ `Model::predict`, delegate to the real\n":
        "        // A softmax (or the OVA sigmoid) over the first `nlabels` output rows is only right\n"
        "        // for those losses: for hierarchical softmax the rows are Huffman-tree nodes (e.g.\n"
        "        // lid.176.ftz), and negative sampling is a binary logistic. Like C++\n"
        "        // `Model::predict`, delegate to the real\n",
    "    /// Lazily-built inference model over a dequantized copy\n    /// of `quant_output`, used":
        "    /// Lazily-built inference model over a dequantized copy of `quant_output`, used",
    "        // The crate used to only overwrite rows of words already in the vocabulary.\n": "",
}


def clean(text: str) -> str:
    text = re.sub(r" // \[fasttext-python-bindings patch\] C\+\+ does this$", " // like C++", text, flags=re.M)
    text = re.sub(r" // \[fasttext-python-bindings patch\]$", "", text, flags=re.M)
    text = text.replace(MARKER + " ", "")
    for old, new in REWORD.items():
        text = text.replace(old, new)
    assert MARKER not in text
    return text


def split_verbose_pretrained(old: list[str], new: list[str], applied: set[str]) -> list[str]:
    """train_internal: the verbose dictionary summary (TR) and the pretrained-vector loading (PV)
    touch the same lines."""
    if PV in applied:
        return new
    if TR not in applied:
        return old
    end = next(i for i, ln in enumerate(new) if i > 0 and ln == "        }\n")
    return [new[0].replace("let mut dict", "let dict")] + new[1 : end + 1] + [old[1]]


def parse_u0(diff: str):
    hunks = []
    for m in re.finditer(r"(?ms)^diff -r -U0 \S+ (\S+)\n--- [^\n]*\n\+\+\+ [^\n]*\n(.*?)(?=^diff -r|\Z)", diff):
        path = m.group(1)
        rel = path.split("/src/", 1)[1] if "/src/" in path else path.split("src/", 1)[1]
        for h in re.finditer(r"(?ms)^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@[^\n]*\n(.*?)(?=^@@|\Z)", m.group(2)):
            start, count = int(h.group(1)), int(h.group(2) if h.group(2) is not None else 1)
            lines = h.group(3).splitlines(keepends=True)
            key = (rel, f"-{h.group(1)}" + (f",{h.group(2)}" if h.group(2) is not None else ""))
            hunks.append(dict(file=rel, key=key, start=start, count=count,
                              old=[ln[1:] for ln in lines if ln.startswith("-")],
                              new=[ln[1:] for ln in lines if ln.startswith("+")]))
    return hunks


def build_tree(hunks, applied: set[str], dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(PRISTINE / "src", dest / "src")
    cargo = (PRISTINE / "Cargo.toml.orig").read_text()
    if X in applied:
        cargo = cargo.replace(
            '[features]\ncli = ["clap"]\n',
            '[features]\ncli = ["clap"]\n# Use the SIMD/FMA output-layer dot product (faster for large `dim`, but not\n'
            "# bit-identical to C++ fastText).\nsimd-dot = []\n",
        )
    (dest / "Cargo.toml").write_text(cargo)
    by_file: dict[str, list] = {}
    for h in hunks:
        by_file.setdefault(h["file"], []).append(h)
    for rel, hs in by_file.items():
        p = dest / "src" / rel
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        for h in sorted(hs, key=lambda h: h["start"], reverse=True):
            topic = ASSIGN[h["key"]]
            if isinstance(topic, tuple):
                repl = split_verbose_pretrained(h["old"], h["new"], applied)
            elif topic in applied:
                repl = h["new"]
            else:
                continue
            lo = h["start"] - 1 if h["count"] > 0 else h["start"]
            assert lines[lo : lo + h["count"]] == h["old"], (h["key"], "context mismatch")
            lines[lo : lo + h["count"]] = repl
        p.write_text(clean("".join(lines)), encoding="utf-8", newline="\n")


def main(run_tests: bool) -> None:
    if not PRISTINE.exists():
        BASE.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"curl -sL https://crates.io/api/v1/crates/fasttext/0.8.0/download | tar xz -C '{BASE}'",
                       shell=True, check=True)
    diff = subprocess.run(["diff", "-r", "-U0", str(PRISTINE / "src"), str(VENDOR / "src")],
                          capture_output=True, text=True).stdout
    hunks = parse_u0(diff)
    missing = [h["key"] for h in hunks if h["key"] not in ASSIGN]
    assert not missing, f"unassigned hunks: {missing}"
    print(f"{len(hunks)} hunks in {len({h['file'] for h in hunks})} files")

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    if OUT.exists():
        for f in OUT.glob("*.patch"):
            f.unlink()
    OUT.mkdir(exist_ok=True)
    prev = WORK / "step-0"
    build_tree(hunks, set(), prev)
    names = []
    for i, (slug, subject, body) in enumerate(TOPICS, 1):
        cur = WORK / f"step-{i}"
        build_tree(hunks, {t[0] for t in TOPICS[:i]}, cur)
        shutil.copytree(prev, WORK / "a", dirs_exist_ok=False)
        shutil.copytree(cur, WORK / "b", dirs_exist_ok=False)
        d = subprocess.run(["diff", "-ruN", "a", "b"], cwd=WORK, capture_output=True, text=True).stdout
        shutil.rmtree(WORK / "a")
        shutil.rmtree(WORK / "b")
        d = re.sub(r"(?m)^(---|\+\+\+) ([ab]/\S+)\t.*$", r"\1 \2", d)
        d = re.sub(r"(?m)^diff -ruN a/(\S+) b/\S+$", r"diff --git a/\1 b/\1", d)
        stat = subprocess.run(["diffstat", "-p1"], input=d, capture_output=True, text=True).stdout \
            if shutil.which("diffstat") else ""
        name = f"{i:04d}-{slug}.patch"
        msg = (f"From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001\n"
               f"From: {AUTHOR}\nDate: Mon, 5 Oct 2026 00:00:00 +0000\n"
               f"Subject: [PATCH {i}/{len(TOPICS)}] {subject}\n\n{body}\n{TRAILER}---\n{stat}\n{d}-- \n2.43.0\n\n")
        (OUT / name).write_text(msg, encoding="utf-8", newline="\n")
        names.append(name)
        print(f"wrote upstream/{name}: {d.count(chr(10) + '+') - d.count(chr(10) + '+++')} +, "
              f"{d.count(chr(10) + '-') - d.count(chr(10) + '---')} -")
        prev = cur
    (OUT / "series").write_text("\n".join(names) + "\n", newline="\n")

    expected = WORK / "vendor-clean"
    shutil.copytree(VENDOR / "src", expected / "src")
    for f in (expected / "src").rglob("*.rs"):
        f.write_text(clean(f.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    final = subprocess.run(["diff", "-r", str(prev / "src"), str(expected / "src")], capture_output=True, text=True)
    assert final.returncode == 0, "series does not reproduce vendor/fasttext/src:\n" + final.stdout[:2000]
    print("OK: the series reproduces vendor/fasttext/src exactly (minus the marker comments)")

    if run_tests:
        published = (PRISTINE / "Cargo.toml").read_text()
        for i in range(1, len(TOPICS) + 1):
            t = WORK / f"test-{i}"
            if t.exists():
                shutil.rmtree(t)
            shutil.copytree(PRISTINE, t, ignore=shutil.ignore_patterns("src"))
            shutil.copytree(WORK / f"step-{i}" / "src", t / "src")
            (t / "Cargo.toml").write_text(published.replace('cli = ["clap"]', 'cli = ["clap"]\nsimd-dot = []'))
            r = subprocess.run(
                "cargo test --release -q -- --test-threads=4 2>&1 | grep -E '^test result|FAILED|panicked|^error' ",
                shell=True, cwd=t, capture_output=True, text=True,
                env=dict(os.environ, CARGO_TARGET_DIR=str(Path.home() / "target/ftpb-vendor-test")))
            results = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed", r.stdout)
            passed = sum(int(p) for _, p, _ in results)
            failed = sum(int(f) for _, _, f in results)
            ok = results and failed == 0 and "error" not in r.stdout
            print(f"step {i} ({TOPICS[i - 1][0]}): {'OK' if ok else 'FAIL'} - {passed} passed, {failed} failed")
            if not ok:
                print(r.stdout[-3000:])
                sys.exit(1)
            shutil.rmtree(t)


if __name__ == "__main__":
    main("--test" in sys.argv)
