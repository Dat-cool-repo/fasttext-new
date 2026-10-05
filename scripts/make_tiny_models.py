"""Train the tiny fixture models in ``tests/data/models/`` with the *original C++* package.

These models are small enough to commit (5 to 40 KB each), so the golden tests
(``tests/test_golden.py``) cover every loss, word n-grams, quantization (with and without
``qnorm`` / ``qout`` / pruning) and an unsupervised model on every CI platform, including the
ones where the C++ package cannot be installed. They are also the seed corpus of the
model-loading fuzz target (``fuzz/``).

The training text is synthetic (generated below with a fixed seed), so the models and the
golden outputs recorded from them are covered by this repository's MIT / Apache-2.0 license.

Why ``thread=10`` and ``dim=10``: the C++ package allocates the input matrix uninitialized
(``intgemm::AlignedVector``) and ``DenseMatrix::uniform`` fills only ``thread`` tenths of it
(blocks of ``rows * dim / 10`` floats). With fewer than 10 threads, or when ``rows * dim`` is
not a multiple of 10, part of the matrix keeps whatever was in that memory. Large matrices come
from fresh (zeroed) pages, small ones from recycled heap memory, so training tiny models failed
at random with "Encountered NaN" (``DenseMatrix::dotRow`` checks for NaN). ``thread=10`` with
``dim=10`` initializes every value. The runs are not deterministic (Hogwild! SGD), which does
not matter because the model files are committed and the golden outputs are recorded from them.

    python scripts/make_tiny_models.py            # needs fasttext-numpy2 (the C++ package)
"""

from __future__ import annotations

import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "tests" / "data" / "models"

TOPICS = {
    "fruit": "apple banana cherry mango pear plum grape lemon",
    "sport": "football tennis goal match player score team coach",
    "music": "guitar piano melody song concert drum violin chord",
    "code": "compiler function variable loop rust python bug commit",
    "food": "café crème brûlée pâté soupe fromage pain beurre",
    "asia": "東京 大阪 寿司 ラーメン 京都 北京 上海 서울",
}
FILLER = "the a of and to in is it with for on this that very really some".split()


def synthetic_lines(rng: random.Random, n: int, many: bool = False) -> list[str]:
    topics = list(TOPICS)
    lines = []
    for i in range(n):
        if many:  # 300 labels: a topic and a numbered subclass with its own marker word
            t = topics[i % len(topics)]
            sub = i % 300
            label = f"__label__{t}{sub}"
            words = rng.sample(TOPICS[t].split(), 3) + [f"k{sub}", f"k{sub}x"]
        else:
            t = topics[i % len(topics)]
            label = f"__label__{t}"
            if i % 7 == 0:  # multi-label lines (exercise one-vs-all)
                label += f" __label__{topics[(i + 1) % len(topics)]}"
            words = rng.sample(TOPICS[t].split(), 4)
        words += rng.sample(FILLER, 3)
        if i % 11 == 0:
            words.append(rng.choice(["😀", "naïve", "Zürich", "x" * rng.randint(1, 30)]))
        rng.shuffle(words)
        lines.append(f"{label} {' '.join(words)}")
    return lines


SUPERVISED = {
    "tiny_softmax.bin": dict(loss="softmax", wordNgrams=2),
    "tiny_hs.bin": dict(loss="hs", wordNgrams=3),
    "tiny_ova.bin": dict(loss="ova", wordNgrams=2),
    "tiny_ns.bin": dict(loss="ns", wordNgrams=1, neg=3),
    "tiny_nosub.bin": dict(loss="softmax", wordNgrams=1, minn=0, maxn=0),
}
# (output, base, quantize args); bases ending in "_many" are trained on 300 labels and not kept.
QUANTIZED = {
    "tiny_softmax_q.ftz": ("tiny_softmax.bin", dict(qnorm=True, cutoff=500)),
    "tiny_hs_q.ftz": ("tiny_hs.bin", dict()),
    "tiny_hs_qout.ftz": ("_many_hs", dict(qout=True, qnorm=True)),
    "tiny_ova_qout.ftz": ("_many_ova", dict(qout=True)),
}


def main() -> None:
    import fasttext

    fasttext.FastText.eprint = lambda *a, **k: None
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20251005)
    train = OUT / "train_tiny.txt"
    train.write_text("\n".join(synthetic_lines(rng, 600)) + "\n", encoding="utf-8")
    many = OUT.parent.parent / "_tmp_train_many.txt"  # not committed
    many.write_text("\n".join(synthetic_lines(rng, 1200, many=True)) + "\n", encoding="utf-8")
    raw = OUT.parent.parent / "_tmp_raw.txt"
    raw.write_text("\n".join(" ".join(l.split()[1:]) for l in train.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")

    common = dict(dim=10, epoch=10, lr=0.5, minn=2, maxn=4, bucket=400, thread=10, verbose=0)
    tmp = {}
    for name, kw in SUPERVISED.items():
        m = fasttext.train_supervised(input=str(train), **{**common, **kw})
        m.save_model(str(OUT / name))
        print("wrote", name, (OUT / name).stat().st_size)
    for base, loss in (("_many_hs", "hs"), ("_many_ova", "ova")):
        m = fasttext.train_supervised(input=str(many), **{**common, "loss": loss, "wordNgrams": 1})
        tmp[base] = OUT.parent.parent / f"{base}.bin"
        m.save_model(str(tmp[base]))
    for name, (base, kw) in QUANTIZED.items():
        src = tmp.get(base, OUT / base)
        m = fasttext.load_model(str(src))
        m.quantize(input=str(many if base.startswith("_many") else train), thread=10, **kw)
        m.save_model(str(OUT / name))
        print("wrote", name, (OUT / name).stat().st_size)
    m = fasttext.train_unsupervised(input=str(raw), model="cbow", dim=10, epoch=5, minCount=2, minn=2, maxn=4,
                                    bucket=400, thread=10, verbose=0)
    m.save_model(str(OUT / "tiny_cbow.bin"))
    print("wrote tiny_cbow.bin", (OUT / "tiny_cbow.bin").stat().st_size)
    for p in [many, raw, *tmp.values()]:
        p.unlink()


if __name__ == "__main__":
    main()
