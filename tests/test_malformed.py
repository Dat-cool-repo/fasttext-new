"""Corrupt, truncated and malicious model files must raise ``ValueError``: never crash the
interpreter, hang, or allocate memory out of proportion to the file.

The inputs are mutations of the committed tiny models (``tests/data/models``) and the minimized
crash inputs found by fuzzing (``fuzz/regressions/``, see ``fuzz/README.md``).
"""

from __future__ import annotations

import random
import struct
from pathlib import Path

import pytest

import fasttext_new

HERE = Path(__file__).resolve().parent
MODELS = HERE / "data" / "models"
REGRESSIONS = HERE.parent / "fuzz" / "regressions"

# Offsets in a fastText model file (all little-endian).
DIM, WS, EPOCH, NEG, WORDNGRAMS, LOSS, MODEL, BUCKET, MINN, MAXN = 8, 12, 16, 24, 28, 32, 36, 40, 44, 48
DICT_SIZE, DICT_NWORDS, DICT_NLABELS, DICT_PRUNEIDX = 64, 68, 72, 84


def model_bytes(name: str) -> bytes:
    return (MODELS / name).read_bytes()


def load_or_value_error(tmp_path: Path, data: bytes, name: str = "m.bin"):
    p = tmp_path / name
    p.write_bytes(data)
    try:
        m = fasttext_new.load_model(str(p))
    except ValueError:
        return None
    # If it loads, it must be usable. (Corrupt words may not be valid UTF-8: decoding them
    # strictly raises UnicodeDecodeError, like the C++ package.)
    labels = m.get_labels(on_unicode_error="replace")
    if labels and labels != m.get_words(on_unicode_error="replace"):
        m.predict("apple banana", k=-1, on_unicode_error="replace")
    m.get_word_vector("apple")
    m.get_sentence_vector("apple banana")
    return m


def patch_i32(data: bytes, off: int, value: int) -> bytes:
    return data[:off] + struct.pack("<i", value) + data[off + 4 :]


def patch_i64(data: bytes, off: int, value: int) -> bytes:
    return data[:off] + struct.pack("<q", value) + data[off + 8 :]


def test_not_a_model(tmp_path):
    for data in [b"", b"\x00" * 3, b"hello world, not a model" * 10, bytes(range(256)) * 50]:
        assert load_or_value_error(tmp_path, data) is None


@pytest.mark.parametrize("name", ["tiny_softmax.bin", "tiny_hs_q.ftz", "tiny_hs_qout.ftz", "tiny_softmax_q.ftz"])
def test_truncated(tmp_path, name):
    data = model_bytes(name)
    cuts = sorted(set(list(range(0, 200, 3)) + list(range(0, len(data), max(1, len(data) // 150)))))
    for n in cuts:
        assert load_or_value_error(tmp_path, data[:n]) is None, n
    assert load_or_value_error(tmp_path, data) is not None


@pytest.mark.parametrize(
    "off,value",
    [
        (DIM, 0), (DIM, -1), (DIM, 1 << 30), (DIM, 2**31 - 1),
        (BUCKET, -1), (BUCKET, 2**31 - 1),
        (MAXN, 1 << 30), (WORDNGRAMS, 1 << 30), (LOSS, 7), (MODEL, 9),
        (DICT_SIZE, 2**31 - 1), (DICT_SIZE, -5), (DICT_NWORDS, 2**31 - 1), (DICT_NWORDS, 0),
        (DICT_NLABELS, 1 << 20), (DICT_NLABELS, 0),
    ],
)
@pytest.mark.parametrize("name", ["tiny_softmax.bin", "tiny_hs.bin", "tiny_hs_q.ftz"])
def test_bad_header_fields(tmp_path, name, off, value):
    assert load_or_value_error(tmp_path, patch_i32(model_bytes(name), off, value)) is None


@pytest.mark.parametrize("value", [2**62, 2**40, 5, -7])
def test_bad_pruneidx_size(tmp_path, value):
    load_or_value_error(tmp_path, patch_i64(model_bytes("tiny_softmax_q.ftz"), DICT_PRUNEIDX, value))


def _matrix_header_offset(data: bytes) -> int:
    """Offset of the input matrix header (just after the dictionary)."""
    size, = struct.unpack_from("<i", data, DICT_SIZE)
    pruneidx, = struct.unpack_from("<q", data, DICT_PRUNEIDX)
    pos = 92
    for _ in range(size):
        pos = data.index(b"\x00", pos) + 1 + 8 + 1
    return pos + max(pruneidx, 0) * 8


@pytest.mark.parametrize("name", ["tiny_softmax.bin", "tiny_nosub.bin", "tiny_hs_q.ftz", "tiny_hs_qout.ftz"])
@pytest.mark.parametrize("field", range(6))
@pytest.mark.parametrize("value", [0, 1, -1, 2**31 - 1, 2**40, 2**62])
def test_bad_matrix_sizes(tmp_path, name, field, value):
    """Dense: m (i64), n (i64). Quantized: qnorm (1), m, n, codesize (i32), PQ dim / nsubq ..."""
    data = model_bytes(name)
    pos = _matrix_header_offset(data) + 1  # skip the quant flag
    quant = name.endswith(".ftz")
    if quant:
        offsets = [(pos + 1, 8), (pos + 9, 8), (pos + 17, 4)]
        codesize, = struct.unpack_from("<i", data, pos + 17)
        pq = pos + 21 + codesize
        offsets += [(pq, 4), (pq + 4, 4), (pq + 8, 4)]
    else:
        offsets = [(pos, 8), (pos + 8, 8)] * 3
    off, width = offsets[field]
    if width == 4:
        if not -(2**31) <= value < 2**31:
            pytest.skip("value does not fit")
        data = patch_i32(data, off, value)
    else:
        data = patch_i64(data, off, value)
    load_or_value_error(tmp_path, data)


@pytest.mark.parametrize("seed", range(40))
def test_random_byte_flips(tmp_path, seed):
    rng = random.Random(seed)
    name = rng.choice(sorted(p.name for p in MODELS.glob("tiny_*")))
    data = bytearray(model_bytes(name))
    # Corrupt the header / dictionary / matrix headers more often than the float data.
    for _ in range(rng.randint(1, 8)):
        i = rng.randrange(min(len(data), 400)) if rng.random() < 0.7 else rng.randrange(len(data))
        data[i] = rng.randrange(256)
    load_or_value_error(tmp_path, bytes(data))


def test_degenerate_hierarchical_softmax_counts(tmp_path):
    """Label counts that turn the Huffman tree into a deep chain (recursion / quadratic paths)."""
    data = bytearray(model_bytes("tiny_hs_qout.ftz"))
    size, = struct.unpack_from("<i", data, DICT_SIZE)
    nwords, = struct.unpack_from("<i", data, DICT_NWORDS)
    pos = 92
    for i in range(size):
        end = data.index(b"\x00", pos)
        if i >= nwords:
            struct.pack_into("<q", data, end + 1, 0 if i % 2 else -1)
        pos = end + 1 + 8 + 1
    load_or_value_error(tmp_path, bytes(data))


@pytest.mark.parametrize(
    "path",
    sorted(REGRESSIONS.glob("load_model/*")) if REGRESSIONS.is_dir() else [],
    ids=lambda p: p.name,
)
def test_fuzz_regressions(tmp_path, path):
    load_or_value_error(tmp_path, path.read_bytes())
