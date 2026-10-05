"""Download about 1 GB of real multilingual web text for bench/real_agreement.py.

* FineWeb-2 (https://huggingface.co/datasets/HuggingFaceFW/fineweb-2, ODC-By): the test shard of
  16 languages (40-105 MB of parquet each).
* FineWeb (https://huggingface.co/datasets/HuggingFaceFW/fineweb, ODC-By): the first row groups
  of one English sample shard (about 200 MB of text), read with HTTP range requests.

    python bench/fetch_real_data.py OUT_DIR

Needs ``pyarrow`` and ``huggingface_hub``. Files that already exist are skipped.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

LANGS = [
    "rus_Cyrl", "cmn_Hani", "deu_Latn", "jpn_Jpan", "spa_Latn", "fra_Latn", "arb_Arab", "hin_Deva",
    "vie_Latn", "ita_Latn", "por_Latn", "pol_Latn", "tur_Latn", "kor_Hang", "ell_Grek", "tha_Thai",
]  # fmt: skip
FW2 = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/{}/test/000_00000.parquet"


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        dest = out / f"fineweb2_{lang}_test.parquet"
        if not dest.exists():
            urllib.request.urlretrieve(FW2.format(lang), str(dest) + ".part")
            os.replace(str(dest) + ".part", dest)
        print(dest.name, dest.stat().st_size >> 20, "MiB", flush=True)

    dest = out / "fineweb_eng_sample.parquet"
    if not dest.exists():
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        path = "datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet"
        with fs.open(path, "rb", block_size=8 << 20) as f:
            pf = pq.ParquetFile(f)
            writer, nbytes = None, 0
            for i in range(pf.num_row_groups):
                t = pf.read_row_group(i, columns=["text", "id", "language"])
                nbytes += sum(len(s.as_py().encode()) for s in t.column("text"))
                writer = writer or pq.ParquetWriter(str(dest) + ".part", t.schema)
                writer.write_table(t)
                if nbytes > 200 << 20:
                    break
            writer.close()
        os.replace(str(dest) + ".part", dest)
    print(dest.name, dest.stat().st_size >> 20, "MiB")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
