"""Prepare the paper's documented T variants from a pinned local checkout.

Circuit bytes stay outside this repository. No noise is removed or rescaled.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

REVISION = "3e3be604595b2b12343a45dd4ba2b03a532b21a0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != REVISION:
        parser.error(f"checkout must be at {REVISION}")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = []
    for stem in ["d3a6", "d3a6f2", "d5a19", "d5a19f13"]:
        for suffix in ["", "_p1e-3"]:
            name = stem + "_inject+cultivate" + suffix + ".stim"
            source = "cliffordea/sim/stim_files/" + name
            # Read committed bytes, avoiding accidental use of local edits or
            # the explicitly uncorrected distance-five reference files.
            original = subprocess.check_output(
                ["git", "-C", str(args.checkout), "show", REVISION + ":" + source]
            )
            converted, replacements = re.subn(
                rb"(?m)^(\s*)S(_DAG)?(?=\s)",
                lambda match: match[1] + b"T" + (match[2] or b""),
                original,
            )
            if replacements != (5 if stem.startswith("d3") else 13):
                raise ValueError(f"unexpected S gate count in {source}")
            target = args.output / ("T_" + name)
            target.write_bytes(converted)
            provenance.append(
                {
                    "source": "https://github.com/timchan0/cliffordea/blob/"
                    + REVISION
                    + "/"
                    + source,
                    "original_sha256": hashlib.sha256(original).hexdigest(),
                    "derived_sha256": hashlib.sha256(converted).hexdigest(),
                    "replacements": replacements,
                    "path": str(target),
                }
            )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
