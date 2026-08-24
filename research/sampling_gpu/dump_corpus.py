"""Write the census corpus as .stim files for the C++ benchmark driver.

Run:  uv run python research/sampling_gpu/dump_corpus.py [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from census import build_workloads  # noqa: E402


def main() -> None:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "corpus"
    outdir.mkdir(parents=True, exist_ok=True)
    for name, text in build_workloads():
        path = outdir / f"{name}.stim"
        path.write_text(text if text.endswith("\n") else text + "\n")
        print(path)


if __name__ == "__main__":
    main()
