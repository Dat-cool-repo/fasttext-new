"""fasttext_new: drop-in replacement for the archived ``fasttext`` Python package.

Backed by a pure-Rust fastText implementation (no C++ toolchain, NumPy 2 friendly, abi3,
free-threading ready). PyPI distribution: ``fasttext-new``.

    import fasttext_new as fasttext
    model = fasttext.load_model("lid.176.bin")
    model.predict("Bonjour tout le monde")      # (('__label__fr',), array([0.98...]))
    labels, probs = model.predict_batch(docs, k=1)   # parallel, GIL released

    model = fasttext.train_supervised(input="train.txt", epoch=25, wordNgrams=2)
    print(model.test("valid.txt"))              # (N, precision@1, recall@1)
    model.quantize(input="train.txt", retrain=True)
    model.save_model("model.ftz")               # loadable by the C++ package

Call :func:`install_as_fasttext` to make ``import fasttext`` resolve to this package.
"""

from __future__ import annotations

import sys

from . import FastText
from ._fasttext_new import __version__
from .FastText import (
    _FastText,
    cbow,
    load_model,
    skipgram,
    supervised,
    tokenize,
    train_supervised,
    train_unsupervised,
    unsupervised_default,
)

__all__ = [
    "load_model",
    "train_supervised",
    "train_unsupervised",
    "tokenize",
    "FastText",
    "install_as_fasttext",
    "__version__",
]


def install_as_fasttext(force: bool = False) -> None:
    """Register this package as ``fasttext`` in ``sys.modules`` (compatibility shim).

    After calling this, ``import fasttext`` and ``fasttext.load_model`` use fasttext_new.
    Does nothing if a real ``fasttext`` module is already imported, unless ``force=True``.
    """
    mod = sys.modules[__name__]
    if "fasttext" in sys.modules and sys.modules["fasttext"] is not mod and not force:
        return
    sys.modules["fasttext"] = mod
    sys.modules["fasttext.FastText"] = FastText
