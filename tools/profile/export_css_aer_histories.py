"""Regenerate independent physical-gate witnesses for the native CSS executor."""

from pathlib import Path

import numpy as np
from msc_factor_sequence import Sequence
from msc_protocol import Program
from msc_reference import reference


def main():
    rows = ["# Aer elementary-gate reference: axis, Pauli fault bits, records, log probability."]
    for axis in "XYZ":
        sequence = Sequence(3, 0.1, axis)
        for seed in (145, 871):
            history = sequence.history(seed)
            result = reference(Program(sequence.circuit(history)), {}, seed + 1)
            faults = "".join(
                str(int(pauli == choice))
                for layer in history[0]
                for pauli in layer
                for choice in (1, 2, 3)
            )
            records = "".join(map(str, result["outcomes"]))
            log_probability = np.log(result["probabilities"]).sum()
            rows.append(f"{axis} {faults} {records} {log_probability:.17g}")
    path = Path(__file__).resolve().parents[2] / "tests/fixtures/css_aer_histories.txt"
    path.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
