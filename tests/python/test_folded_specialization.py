"""Public compilation and sampling of structurally certified folded regions."""

import re
from pathlib import Path

import numpy as np
import pytest

import clifft

FIXTURES = Path(__file__).parents[1] / "fixtures" / "folded"


def source(distance=3, probability=0.0):
    text = (FIXTURES / f"f{distance}.stim").read_text()
    text = "\n".join(
        line for line in text.splitlines() if not line.startswith(("#", "QUBIT_COORDS"))
    )
    return re.sub(r"(X_ERROR|DEPOLARIZE[123])\([^)]*\)", rf"\g<1>({probability})", text)


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_standard_program_and_sampling(distance):
    text = source(distance)
    ordinary = clifft.compile(text)
    specialized = clifft.compile(text, specialize_folded=True)
    assert isinstance(specialized, clifft.Program)
    assert specialized.has_folded_regions and not ordinary.has_folded_regions
    assert "SAMPLE_FOLDED_REGION" in specialized.inspect()
    for name in (
        "num_measurements",
        "num_hidden_measurements",
        "num_detectors",
        "num_observables",
        "num_qubits",
    ):
        assert getattr(specialized, name) == getattr(ordinary, name)
    np.testing.assert_array_equal(
        specialized.noise_site_probabilities, ordinary.noise_site_probabilities
    )
    result = clifft.sample(specialized, shots=4, seed=417)
    assert result.measurements.shape == (4, ordinary.num_measurements)
    assert not result.detectors.any() and not result.observables.any()
    assert specialized.peak_active_width < ordinary.peak_active_width


@pytest.mark.parametrize("distance", [3, 5])
def test_noisy_distributions_match_ordinary_sampling(distance):
    text = source(distance, 0.02)
    ordinary = clifft.compile(text)
    specialized = clifft.compile(text, specialize_folded=True)
    np.testing.assert_allclose(
        specialized.noise_site_probabilities, ordinary.noise_site_probabilities, atol=0, rtol=0
    )
    shots = 4096 if distance == 3 else 512
    a = clifft.sample(ordinary, shots=shots, seed=1613, threads=2, batch_size=1)
    b = clifft.sample(specialized, shots=shots, seed=6821, threads=2)
    for left, right in (
        (a.measurements, b.measurements),
        (a.detectors, b.detectors),
        (a.observables, b.observables),
    ):
        p, q = left.mean(axis=0), right.mean(axis=0)
        tolerance = 6 * np.sqrt((p * (1 - p) + q * (1 - q)) / shots) + 2 / shots
        assert np.all(np.abs(p - q) <= tolerance)
    p = np.all(a.detectors == 0, axis=1).mean()
    q = np.all(b.detectors == 0, axis=1).mean()
    assert abs(p - q) <= 6 * np.sqrt((p * (1 - p) + q * (1 - q)) / shots) + 2 / shots


def test_seeded_sampling_is_independent_of_worker_count():
    program = clifft.compile(source(3, 0.025), specialize_folded=True)
    a = clifft.sample(program, shots=256, seed=761, threads=1)
    b = clifft.sample(program, shots=256, seed=761, threads=2)
    for name in ("measurements", "detectors", "observables"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_postselection_and_expected_parities_use_normal_outputs():
    text = source(3, 0.012)
    full = clifft.compile(text, specialize_folded=True)
    mask = [int(i % 3 == 0) for i in range(full.num_detectors)]
    expected = [int(i == 2) for i in range(full.num_detectors)]
    selected = clifft.compile(
        text,
        specialize_folded=True,
        postselection_mask=mask,
        expected_detectors=expected,
        expected_observables=[1],
    )
    result = clifft.sample_survivors(selected, shots=512, seed=1234, keep_records=True, threads=2)
    assert len(result.measurements) > 0
    assert not result.detectors[:, np.array(mask, dtype=bool)].any()
    unselected = clifft.sample(full, shots=512, seed=1234, threads=2)
    accepted = ~unselected.detectors[:, np.array(mask, dtype=bool)].any(axis=1)
    np.testing.assert_array_equal(result.measurements, unselected.measurements[accepted])
    np.testing.assert_array_equal(result.detectors, unselected.detectors[accepted] ^ expected)
    np.testing.assert_array_equal(result.observables, unselected.observables[accepted] ^ 1)
    # Reconstruct all user-facing outputs from their original absolute record references.
    nodes = clifft.parse(text).nodes
    detector = 0
    for node in nodes:
        if node.gate.name == "DETECTOR":
            actual = np.zeros(len(result.measurements), dtype=np.uint8)
            for t in node.targets:
                actual ^= result.measurements[:, t.value]
            np.testing.assert_array_equal(
                result.detectors[:, detector], actual ^ expected[detector]
            )
            detector += 1
    with pytest.raises(ValueError, match="sample_survivors"):
        clifft.sample(selected, shots=1)


def test_syndrome_normalization_and_observable_accumulation():
    text = source() + "\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-2]\n"
    program = clifft.compile(text, specialize_folded=True, normalize_syndromes=True)
    result = clifft.sample(program, shots=16, seed=84)
    assert program.has_folded_regions
    assert not result.detectors.any() and not result.observables.any()


def test_recognition_accepts_relabeling_and_omitted_noise():
    text = "\n".join(
        line for line in source().splitlines() if not line.startswith(("X_ERROR", "DEPOLARIZE"))
    )
    lines = []
    for line in text.splitlines():
        if line.startswith(("DETECTOR", "OBSERVABLE")):
            lines.append(line)
        else:
            parts = line.split()
            lines.append(parts[0] + " " + " ".join(str(31 - int(q)) for q in parts[1:]))
    program = clifft.compile("\n".join(lines), specialize_folded=True)
    assert program.has_folded_regions
    result = clifft.sample(program, shots=16, seed=914)
    assert not result.detectors.any() and not result.observables.any()


@pytest.mark.parametrize("mutation", ["unmatched", "boundary"])
def test_near_matches_fall_back_without_changing_results(mutation):
    text = source()
    # This is the first cat preparation of the terminal region in this fixture.
    marker = "R 13\nX_ERROR(0.0) 13\nR 14"
    assert marker in text
    if mutation == "unmatched":
        text = text.replace(marker, marker.replace("R 14", "X 0\nR 14"), 1)
    else:
        text = text.replace(marker, "H 0\n" + marker, 1)
    a = clifft.compile(text)
    b = clifft.compile(text, specialize_folded=True)
    assert not b.has_folded_regions
    assert "folded_specialization:" in b.inspect()
    left, right = clifft.sample(a, shots=16, seed=718), clifft.sample(b, shots=16, seed=718)
    np.testing.assert_array_equal(left.measurements, right.measurements)


def test_unsupported_sampling_modes_are_explicit():
    program = clifft.compile(source(), specialize_folded=True)
    with pytest.raises(ValueError, match="fixed-fault"):
        clifft.sample_k(program, shots=1, k=0)
    with pytest.raises(ValueError, match="scalar"):
        clifft.sample(program, shots=4, batch_size=4)
    with pytest.raises(ValueError, match="replay"):
        clifft.record_probabilities(program, ["0" * program.num_measurements])


def test_nontrivial_logical_input_against_aer():
    from qiskit_aer import AerSimulator
    from utils_qiskit import stim_to_qiskit

    text = "\n".join(
        line for line in source().splitlines() if not line.startswith(("X_ERROR", "DEPOLARIZE"))
    )
    text = text.replace("T 6", "R_Z(0.37) 6", 1)
    program = clifft.compile(text, specialize_folded=True)
    assert program.has_folded_regions
    qc = stim_to_qiskit(text)
    sim = AerSimulator(
        method="matrix_product_state",
        max_parallel_threads=1,
        matrix_product_state_truncation_threshold=1e-14,
    )
    memory = sim.run(qc, shots=1024, seed_simulator=381, memory=True).result().get_memory()
    reference = np.array([[int(c) for c in row[::-1]] for row in memory], dtype=np.uint8)
    actual = clifft.sample(program, shots=4096, seed=881).measurements
    p, q = reference.mean(axis=0), actual.mean(axis=0)
    tolerance = 6 * np.sqrt(p * (1 - p) / len(reference) + q * (1 - q) / len(actual)) + 2 / len(
        reference
    )
    assert np.all(np.abs(p - q) <= tolerance)


def test_custom_passes_apply_to_the_whole_circuit():
    passes = clifft.HirPassManager()
    passes.add(clifft.RemoveNoisePass())
    program = clifft.compile(source(3, 0.2), specialize_folded=True, hir_passes=passes)
    assert not program.has_folded_regions
    assert "custom HIR passes" in program.inspect()
    result = clifft.sample(program, shots=16, seed=123)
    assert not result.detectors.any() and not result.observables.any()


def test_numpy_options_and_no_passes():
    text = source()
    ordinary = clifft.compile(text)
    program = clifft.compile(
        text,
        specialize_folded=True,
        hir_passes=None,
        postselection_mask=np.ones(ordinary.num_detectors, dtype=np.uint8),  # type: ignore[arg-type]
        expected_detectors=np.zeros(ordinary.num_detectors, dtype=np.uint8),  # type: ignore[arg-type]
        expected_observables=np.zeros(ordinary.num_observables, dtype=np.uint8),  # type: ignore[arg-type]
    )
    assert program.has_folded_regions
    result = clifft.sample_survivors(program, shots=8, seed=901, keep_records=True)
    assert len(result.measurements) == 8


def test_misplaced_noise_declines_specialization():
    text = source().replace("R 13\nX_ERROR(0.0) 13\nR 14", "R 13\nR 14\nX_ERROR(0.1) 13", 1)
    program = clifft.compile(text, specialize_folded=True)
    assert not program.has_folded_regions
