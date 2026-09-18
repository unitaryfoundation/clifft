"""Recheck large research histories with CH states for production replay tests."""

import json
from pathlib import Path

import numpy as np
from msc_factor_sequence import Sequence
from msc_protocol import evaluate


def main():
    root = Path(__file__).resolve().parents[2]
    study = json.loads((root / "tools/profile/research/msc_factor_sequence_data.json").read_text())
    rows = ["# CH reference at 61 data wires: axis, fault bits, records, log probability."]
    large = next(case for case in study["circuits"] if case["distance_label"] == 9)
    for case in large["validation"]:
        if "coherent_relative_error" not in case:
            continue
        sample = case["sample"]
        sequence = Sequence(9, case["noise"], case["axis"])
        history = [list(map(int, row)) for row in sample["faults"]], list(map(int, sample["flips"]))
        program, bound = sequence.program_history(history)
        expected = evaluate(program, bound, list(map(int, sample["outcomes"])))
        faults = "".join(
            str(int(pauli == choice))
            for layer in history[0]
            for pauli in layer
            for choice in (1, 2, 3)
        )
        weight = np.log(expected["trajectory_probability"])
        rows.append(f"{case['axis']} {faults} {sample['outcomes']} {weight:.17g}")
    (root / "tests/fixtures/css_ch_histories.txt").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
