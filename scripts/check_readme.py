"""Run the Python examples of README.md ("Quick start" and the `import fasttext` shim) as written.

    python scripts/check_readme.py WORKDIR

WORKDIR must contain ``lid.176.ftz`` and ``lid.176.bin`` (the official models) and the files
the examples use: ``train.txt`` / ``valid.txt`` (labelled lines, e.g. cooking.stackexchange)
and ``corpus.txt`` (raw text). Each ```python block is executed in WORKDIR; the script fails
on the first block that raises.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def blocks(text: str, start: str, end: str) -> list[str]:
    section = text[text.index(start) : text.index(end)]
    return re.findall(r"```python\n(.*?)```", section, flags=re.S)


def main(workdir: Path) -> None:
    text = README.read_text(encoding="utf-8")
    code = blocks(text, "## Quick start", "## API compatibility")
    assert code, "no python blocks found"
    os.chdir(workdir)
    for i, src in enumerate(code, 1):
        print(f"--- README block {i} ---\n{src}", flush=True)
        exec(compile(src, f"README block {i}", "exec"), {"__name__": "__readme__"})
        print(f"--- block {i} OK", flush=True)
        for mod in [m for m in sys.modules if m == "fasttext" or m.startswith("fasttext.")]:
            del sys.modules[mod]  # the shim example registers `fasttext`; start clean


if __name__ == "__main__":
    main(Path(sys.argv[1]))
