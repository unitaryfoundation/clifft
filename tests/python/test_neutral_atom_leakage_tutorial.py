from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "docs" / "guide" / "scripts" / "neutral_atom_leakage_tutorial.py"
MANIFEST = ROOT / "docs" / "guide" / "circuits" / "neutral_atom" / "manifest.json"

spec = importlib.util.spec_from_file_location("neutral_atom_leakage_tutorial", SCRIPT)
assert spec is not None and spec.loader is not None
tutorial = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tutorial
spec.loader.exec_module(tutorial)


def test_exported_circuit_hashes_match_manifest():
    manifest = json.loads(MANIFEST.read_text())
    for entry in manifest["translation"]["circuits"].values():
        circuit = MANIFEST.parent / entry["file"]
        assert hashlib.sha256(circuit.read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("circuit", tuple(tutorial.CIRCUIT_FILES))
def test_noiseless_circuit_decodes_to_ideal_support(circuit):
    text = (tutorial.CIRCUIT_DIR / tutorial.CIRCUIT_FILES[circuit]).read_text()
    noiseless = "\n".join(
        line
        for line in text.splitlines()
        if "_ERROR(" not in line and not line.startswith("LEVEL_TRANSITION")
    )
    model = tutorial.noncomp.Model(
        initial_state=(0, 1, 0, 0, 0),
        classifier=tutorial.noncomp.Classifier(
            [
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        ),
    )
    result = tutorial.noncomp.sample(noiseless, model, shots=512, seed=19)
    raw = tutorial.Counter("".join(str(symbol) for symbol in row) for row in result.symbols())
    decoded = tutorial.decode_counts(circuit, raw)

    assert set(decoded) <= set(tutorial.IDEAL_DISTRIBUTION)
    assert tutorial.total_variation_distance(decoded, tutorial.IDEAL_DISTRIBUTION) < 0.1


@pytest.mark.parametrize(
    ("model", "accepted", "heralded", "tvd"),
    (
        ("matched", 29, 13, 0.22413793103448276),
        ("exact", 36, 11, 0.1388888888888889),
    ),
)
def test_ldu_tutorial_regression(model, accepted, heralded, tvd):
    result = tutorial.run_experiment(
        "two_row_ldu",
        model,
        shots=128,
        seed=31,
    )

    assert result.accepted == accepted
    assert result.decoded_samples == result.accepted
    assert result.heralded == heralded
    assert result.tvd == pytest.approx(tvd)
