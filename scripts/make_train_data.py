"""Download and prepare the small datasets used by the training-parity tests.

* cooking.stackexchange (the official fastText tutorial data, ~15k questions, 735 tags,
  multi-label), preprocessed exactly like the tutorial and split 12,404 / 3,000 lines into
  ``cooking.train`` / ``cooking.valid``.
* papluca/language-identification (20 languages): ``lid.train`` (70k lines) and
  ``lid.valid`` (10k lines) in fastText format, plus ``lid_raw.txt`` (the training text without
  labels) for unsupervised training.

Usage: python scripts/make_train_data.py [DATA_DIR]   (default $FASTTEXT_NEW_DATA, else ./data)
"""

from __future__ import annotations

import csv
import io
import os
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

COOKING_URL = "https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz"
PAPLUCA_URL = "https://huggingface.co/datasets/papluca/language-identification/resolve/main/{}.csv"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def cooking(data: Path) -> None:
    if (data / "cooking.train").exists() and (data / "cooking.valid").exists():
        return
    raw = data / "cooking.stackexchange.txt"
    if not raw.exists():
        with tarfile.open(fileobj=io.BytesIO(fetch(COOKING_URL)), mode="r:gz") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("cooking.stackexchange.txt"))
            raw.write_bytes(tf.extractfile(member).read())
    # Tutorial preprocessing: sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]"
    lines = raw.read_text(encoding="utf-8").splitlines()
    lines = [re.sub(r"([.!?,'/()])", r" \1 ", ln).lower() for ln in lines]
    (data / "cooking.train").write_text("\n".join(lines[:12404]) + "\n", encoding="utf-8")
    (data / "cooking.valid").write_text("\n".join(lines[-3000:]) + "\n", encoding="utf-8")


def papluca(data: Path) -> None:
    for split, name in (("train", "lid.train"), ("valid", "lid.valid")):
        out = data / name
        if out.exists():
            continue
        csv_path = data / f"papluca_{split}.csv"
        if not csv_path.exists():
            csv_path.write_bytes(fetch(PAPLUCA_URL.format(split)))
        with open(csv_path, encoding="utf-8") as f:
            rows = [(r["labels"], " ".join(r["text"].split())) for r in csv.DictReader(f)]
        out.write_text("".join(f"__label__{lab} {t}\n" for lab, t in rows if t), encoding="utf-8")
        if split == "train":
            (data / "lid_raw.txt").write_text("".join(f"{t}\n" for _, t in rows if t), encoding="utf-8")


def main(data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    cooking(data)
    papluca(data)
    for n in ("cooking.train", "cooking.valid", "lid.train", "lid.valid", "lid_raw.txt"):
        p = data / n
        print(f"{n}: {sum(1 for _ in open(p, encoding='utf-8'))} lines")


if __name__ == "__main__":
    default = os.environ.get("FASTTEXT_NEW_DATA", Path(__file__).resolve().parent.parent / "data")
    main(Path(sys.argv[1] if len(sys.argv) > 1 else default))
