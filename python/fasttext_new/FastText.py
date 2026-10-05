"""fastText-compatible model wrapper around the Rust extension.

Mirrors ``fasttext.FastText`` from the archived C++ package (0.9.x), including return shapes:

* ``predict(str)``  -> ``(tuple[str], np.ndarray[float64])``
* ``predict(list)`` -> ``(list[list[str]], list[np.ndarray[float32]])``
"""

from __future__ import annotations

import multiprocessing
from itertools import chain
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from . import _fasttext_new
from ._fasttext_new import Model as _RustModel

__all__ = [
    "_FastText",
    "load_model",
    "train_supervised",
    "train_unsupervised",
    "tokenize",
    "unsupervised_default",
]

# Same defaults as the C++ package (fasttext/FastText.py).
unsupervised_default = {
    "model": "skipgram",
    "lr": 0.05,
    "dim": 100,
    "ws": 5,
    "epoch": 5,
    "minCount": 5,
    "minCountLabel": 0,
    "minn": 3,
    "maxn": 6,
    "neg": 5,
    "wordNgrams": 1,
    "loss": "ns",
    "bucket": 2000000,
    "thread": max(1, multiprocessing.cpu_count() - 1),
    "lrUpdateRate": 100,
    "t": 1e-4,
    "label": "__label__",
    "verbose": 2,
    "pretrainedVectors": "",
    "seed": 0,
    "autotuneValidationFile": "",
    "autotuneMetric": "f1",
    "autotunePredictions": 1,
    "autotuneDuration": 60 * 5,  # 5 minutes
    "autotuneModelSize": "",
}

_ARG_ATTRS = [
    "lr", "dim", "ws", "epoch", "minCount", "minCountLabel", "minn", "maxn", "neg",
    "wordNgrams", "loss", "bucket", "thread", "lrUpdateRate", "t", "label", "verbose",
    "pretrainedVectors",
]  # fmt: skip


def eprint(*args, **kwargs):
    import sys

    print(*args, file=sys.stderr, **kwargs)


def _f32(buf) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32)


def _check_line(text: str, what: str = "predict") -> None:
    if text.find("\n") != -1:
        raise ValueError(f"{what} processes one line at a time (remove '\\n')")


class _FastText:
    """A fastText model (loaded ``.bin`` / quantized ``.ftz``, or freshly trained)."""

    def __init__(self, model_path=None, args=None, _rust=None):
        if _rust is not None:
            self.f = _rust
        elif model_path is not None:
            self.f = _RustModel(str(model_path))
        else:
            raise ValueError("fasttext_new._FastText needs a model_path; use train_supervised() to train")
        self._words = None
        self._labels = None
        self.set_args(args)

    def set_args(self, args=None):
        """Copy training arguments onto the model as attributes (like the C++ package)."""
        if args:
            for name in _ARG_ATTRS:
                setattr(self, name, args[name])

    def __repr__(self) -> str:
        return f"<fasttext_new model {self.f!r}>"

    def __getitem__(self, word: str) -> np.ndarray:
        return self.get_word_vector(word)

    def __contains__(self, word: str) -> bool:
        return word in self.words

    @property
    def words(self) -> List[str]:
        if self._words is None:
            self._words = self.get_words()
        return self._words

    @property
    def labels(self) -> List[str]:
        if self._labels is None:
            self._labels = self.get_labels()
        return self._labels

    # -- basic info --------------------------------------------------------------------------
    def get_dimension(self) -> int:
        """Get the dimension (size) of a lookup vector (hidden layer)."""
        return self.f.get_dimension()

    def is_quantized(self) -> bool:
        return self.f.is_quantized()

    def get_words(self, include_freq: bool = False, on_unicode_error: str = "strict"):
        """Words of the dictionary (optionally with an int64 numpy array of counts).

        Words that are not valid UTF-8 are decoded with ``on_unicode_error`` (any
        ``bytes.decode`` error handler; ``"strict"`` raises ``UnicodeDecodeError``)."""
        words, freq = self.f.get_words(on_unicode_error)
        if include_freq:
            return words, np.array(freq, dtype=np.int64)
        return words

    def get_labels(self, include_freq: bool = False, on_unicode_error: str = "strict"):
        """Labels of a supervised model (falls back to words for unsupervised models)."""
        if not self.f.is_supervised():
            return self.get_words(include_freq, on_unicode_error)
        labels, freq = self.f.get_labels(on_unicode_error)
        if include_freq:
            return labels, np.array(freq, dtype=np.int64)
        return labels

    def get_word_id(self, word: str) -> int:
        return self.f.get_word_id(word)

    def get_label_id(self, label: str) -> int:
        return self.f.get_label_id(label)

    def get_subword_id(self, subword: str) -> int:
        """Index of a character n-gram's row in the input matrix (``nwords + hash % bucket``)."""
        return self.f.get_subword_id(subword)

    def get_subwords(self, word: str, on_unicode_error: str = "strict"):
        """``(subword strings, np.ndarray[int32] ids)``. For pruned (``.ftz``) models only
        subwords that survived pruning are listed."""
        subwords, ids = self.f.get_subwords(word)
        return subwords, np.array(ids, dtype=np.int32)

    def get_line(self, text: Union[str, List[str]], on_unicode_error: str = "strict"):
        """Split a line into ``(words, labels)``, like the C++ tokenizer (labels must be in
        the dictionary)."""
        if type(text) is list:
            for t in text:
                _check_line(t, "get_line")
            pairs = [self.f.get_line(t) for t in text]
            return [p[0] for p in pairs], [p[1] for p in pairs]
        _check_line(text, "get_line")
        return self.f.get_line(text)

    # -- vectors -----------------------------------------------------------------------------
    def get_word_vector(self, word: str) -> np.ndarray:
        """Vector representation of a word (float32, length ``get_dimension()``)."""
        return _f32(self.f.get_word_vector(word))

    def get_sentence_vector(self, text: str) -> np.ndarray:
        """Vector representation of one line of text (float32)."""
        _check_line(text)
        return _f32(self.f.get_sentence_vector(text))

    def get_input_vector(self, ind: int) -> np.ndarray:
        """Row ``ind`` of the input matrix (dense or quantized)."""
        return _f32(self.f.get_input_vector(ind))

    def get_nearest_neighbors(self, word: str, k: int = 10, on_unicode_error: str = "strict"):
        return self.f.get_nearest_neighbors(word, k, on_unicode_error)

    def get_analogies(self, wordA: str, wordB: str, wordC: str, k: int = 10, on_unicode_error: str = "strict"):
        """Words closest to ``wordA - wordB + wordC``: list of ``(score, word)``."""
        return self.f.get_analogies(wordA, wordB, wordC, k, on_unicode_error)

    def get_input_matrix(self) -> np.ndarray:
        """Full input matrix (dense models only, like the C++ package)."""
        if self.f.is_quantized():
            raise ValueError("Can't get quantized Matrix")
        rows, cols, data = self.f.get_input_matrix()
        return _f32(data).reshape(rows, cols)

    def get_output_matrix(self) -> np.ndarray:
        """Full output matrix (dense models only, like the C++ package)."""
        if self.f.is_quantized():
            raise ValueError("Can't get quantized Matrix")
        rows, cols, data = self.f.get_output_matrix()
        return _f32(data).reshape(rows, cols)

    # -- prediction --------------------------------------------------------------------------
    def predict(
        self,
        text: Union[str, List[str]],
        k: int = 1,
        threshold: float = 0.0,
        on_unicode_error: str = "strict",
    ):
        """Predict the most likely labels.

        ``str`` input returns ``(labels: tuple[str], probs: np.ndarray[float64])``; ``list``
        input returns ``(list[list[str]], list[np.ndarray[float32]])`` exactly like the C++
        package. List input is scored in parallel with the GIL released.

        A negative ``threshold`` filters nothing and disables the hierarchical-softmax pruning
        (``k=-1`` then returns every label), as in C++. Labels that are not valid UTF-8 are
        decoded with ``on_unicode_error``.
        """
        # (newlines are rejected with ValueError by the extension, like the C++ package)
        if type(text) is list:
            return self.predict_batch(text, k, threshold, on_unicode_error=on_unicode_error)
        labels, probs = self.f.predict(text, k, threshold, on_unicode_error)
        return labels, np.asarray(probs)

    def predict_batch(
        self,
        texts: Sequence[str],
        k: int = 1,
        threshold: float = 0.0,
        threads: Optional[int] = None,
        on_unicode_error: str = "strict",
    ) -> Tuple[List[List[str]], List[np.ndarray]]:
        """Score many lines in parallel (rayon) with the GIL released.

        Returns ``(list[list[str]], list[np.ndarray[float32]])`` (same as ``predict(list)``).
        ``threads=None`` uses the global rayon pool (``RAYON_NUM_THREADS``, default all cores).
        """
        if not isinstance(texts, list):
            texts = list(texts)
        labels, flat, counts = self.f.predict_batch(texts, k, threshold, threads, on_unicode_error)
        arr = _f32(flat)
        probs = []
        pos = 0
        for c in counts:
            probs.append(arr[pos : pos + c])
            pos += c
        return labels, probs

    # -- evaluation --------------------------------------------------------------------------
    def test(self, path, k: int = 1, threshold: float = 0.0):
        """Evaluate on a labelled file: ``(number of examples, precision@k, recall@k)``."""
        return self.f.test(str(path), k, threshold)

    def test_label(self, path, k: int = 1, threshold: float = 0.0):
        """Per-label metrics: ``{label: {"precision": p, "recall": r, "f1score": f}}``.

        Labels that were never predicted / never gold get ``nan`` values, like C++."""
        return {
            label: {"precision": p, "recall": r, "f1score": f}
            for label, p, r, f in self.f.test_label(str(path), k, threshold)
        }

    # -- saving / compression ----------------------------------------------------------------
    def save_model(self, path):
        """Save the model to ``path`` (C++-compatible ``.bin``, or ``.ftz`` once quantized)."""
        self.f.save_model(str(path))

    def quantize(
        self,
        input=None,
        qout=False,
        cutoff=0,
        retrain=False,
        epoch=None,
        lr=None,
        thread=None,
        verbose=None,
        dsub=2,
        qnorm=False,
    ):
        """Quantize the model in place (product quantization), like the C++ package.

        ``cutoff`` keeps only the ``cutoff`` input rows with the largest norm; ``retrain``
        then fine-tunes them on ``input``. Unset ``epoch`` / ``lr`` / ``thread`` / ``verbose``
        default to the model's arguments."""
        a = self.f.get_args()
        if not epoch:
            epoch = a["epoch"]
        if not lr:
            lr = a["lr"]
        if not thread:
            thread = a["thread"]
        if not verbose:
            verbose = a["verbose"]
        if retrain and not input:
            raise ValueError("Need input file path if retraining")
        if input is None:
            input = ""
        self.f.quantize(
            dict(
                input=str(input),
                qout=bool(qout),
                cutoff=int(cutoff),
                retrain=bool(retrain),
                epoch=int(epoch),
                lr=float(lr),
                thread=int(thread),
                verbose=int(verbose),
                dsub=int(dsub),
                qnorm=bool(qnorm),
            )
        )
        self._words = None
        self._labels = None

    def set_matrices(self, input_matrix, output_matrix):
        raise NotImplementedError("set_matrices is not supported by fasttext_new")


def load_model(path) -> _FastText:
    """Load a model given a filepath and return a model object."""
    return _FastText(model_path=path)


def tokenize(text: str) -> List[str]:
    """Split text into tokens like the C++ tokenizer (each newline becomes ``</s>``)."""
    return _fasttext_new.tokenize(text)


def read_args(arg_list, arg_dict, arg_names, default_values):
    param_map = {
        "min_count": "minCount",
        "word_ngrams": "wordNgrams",
        "lr_update_rate": "lrUpdateRate",
        "label_prefix": "label",
        "pretrained_vectors": "pretrainedVectors",
    }
    ret = {}
    manually_set_args = set()
    for arg_name, arg_value in chain(zip(arg_names, arg_list), arg_dict.items()):
        if arg_name in param_map:
            arg_name = param_map[arg_name]
        if arg_name not in arg_names:
            raise TypeError("unexpected keyword argument '%s'" % arg_name)
        if arg_name in ret:
            raise TypeError("multiple values for argument '%s'" % arg_name)
        ret[arg_name] = arg_value
        manually_set_args.add(arg_name)
    for arg_name, arg_value in default_values.items():
        if arg_name not in ret:
            ret[arg_name] = arg_value
    return ret, manually_set_args


def _build_args(args):
    if args["model"] not in ("cbow", "skipgram", "supervised"):
        raise ValueError("Unrecognized model name")
    if args["loss"] not in ("ns", "hs", "softmax", "ova"):
        raise ValueError("Unrecognized loss name")
    a = dict(args)
    a["autotuneModelSize"] = str(a["autotuneModelSize"])
    a["input"] = str(a["input"])
    a["pretrainedVectors"] = str(a["pretrainedVectors"] or "")
    a["autotuneValidationFile"] = str(a["autotuneValidationFile"] or "")
    if a["wordNgrams"] <= 1 and a["maxn"] == 0:
        a["bucket"] = 0
    return a


def _train(args) -> _FastText:
    a = _build_args(args)
    ft = _FastText(_rust=_fasttext_new.train(a))
    ft.set_args(ft.f.get_args())
    return ft


def train_supervised(*kargs, **kwargs) -> _FastText:
    """Train a supervised model and return a model object.

    ``input`` must be a filepath. Each line of the input file contains one or more labels
    (prefixed with ``__label__`` by default) followed by the text. Accepts the same arguments
    as the C++ package (``lr``, ``dim``, ``ws``, ``epoch``, ``minCount``, ``minCountLabel``,
    ``minn``, ``maxn``, ``neg``, ``wordNgrams``, ``loss``, ``bucket``, ``thread``,
    ``lrUpdateRate``, ``t``, ``label``, ``verbose``, ``pretrainedVectors``, ``seed``,
    ``autotune*``). The GIL is released during training and Ctrl-C stops it.
    """
    supervised_default = unsupervised_default.copy()
    supervised_default.update(
        {"lr": 0.1, "minCount": 1, "minn": 0, "maxn": 0, "loss": "softmax", "model": "supervised"}
    )
    arg_names = [
        "input", "lr", "dim", "ws", "epoch", "minCount", "minCountLabel", "minn", "maxn",
        "neg", "wordNgrams", "loss", "bucket", "thread", "lrUpdateRate", "t", "label",
        "verbose", "pretrainedVectors", "seed", "autotuneValidationFile", "autotuneMetric",
        "autotunePredictions", "autotuneDuration", "autotuneModelSize",
    ]  # fmt: skip
    args, _ = read_args(kargs, kwargs, arg_names, supervised_default)
    return _train(args)


def train_unsupervised(*kargs, **kwargs) -> _FastText:
    """Train an unsupervised model (``model="skipgram"`` or ``"cbow"``) on a raw text file.

    Accepts the same arguments as the C++ package. The GIL is released during training.
    """
    arg_names = [
        "input", "model", "lr", "dim", "ws", "epoch", "minCount", "minCountLabel", "minn",
        "maxn", "neg", "wordNgrams", "loss", "bucket", "thread", "lrUpdateRate", "t",
        "label", "verbose", "pretrainedVectors", "seed",
    ]  # fmt: skip
    args, _ = read_args(kargs, kwargs, arg_names, unsupervised_default)
    return _train(args)


def _deprecated(name, replacement):
    def f(*args, **kwargs):
        raise Exception(
            "`{}` is not supported any more. Please use `{}` instead.".format(name, replacement)
        )

    f.__name__ = name
    return f


cbow = _deprecated("cbow", "train_unsupervised(model='cbow')")
skipgram = _deprecated("skipgram", "train_unsupervised(model='skipgram')")
supervised = _deprecated("supervised", "train_supervised")
