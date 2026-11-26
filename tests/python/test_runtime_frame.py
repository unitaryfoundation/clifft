"""Tests for sampling programs planned with a runtime Clifford frame.

The planner option is internal. These tests reach it through the private
``_lower_runtime_frame`` binding and compare its programs with ordinary ones
and with Qiskit Aer through the public sampling functions.
"""

import numpy as np
import pytest
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error
from utils_qiskit import qiskit_statevector, stim_to_qiskit_noiseless

import clifft
from clifft._clifft_core import _lower_runtime_frame, _replay_record

# A Bell pair whose T and T_DAG cancel unless the fault flips one sign.
SIGNED_PAIR = (
    "H 0\n"
    "CX 0 1\n"
    "X_ERROR(0.3) 0\n"
    "T 0\n"
    "T_DAG 1\n"
    "MPP X0*X1\n"
    "MPP Y0*X1\n"
    "M 0 1\n"
    "DETECTOR rec[-4]\n"
    "OBSERVABLE_INCLUDE(0) rec[-3]\n"
)

# The same structure where the sign comes from a random stabilizer outcome.
RANDOM_OUTCOME = "H 0 1\nMPP Z0*Z1\nT 0\nT_DAG 1\nH 0 1\nM 0 1\n"

# Each case prepares a stabilizer state, measures a Z-product parity whose
# outcome is random, and applies diagonal gates whose rotation signs depend on
# that outcome: (qubits, preparation, parity qubits, gates after the parity).
PARITY_CASES = {
    "t-pair": (2, "H 0 1", (0, 1), "T 0\nT_DAG 1\nH 0 1"),
    "rz-pair": (2, "H 0 1", (0, 1), "R_Z(0.25) 0\nR_Z(1.75) 1\nH 0 1"),
    "ccz": (3, "H 0 1 2", (0, 1), "CCZ 0 1 2\nH 0 1 2"),
    "ccz-and-t": (4, "H 0 1 2 3\nCX 2 3", (0, 1), "CCZ 0 1 2\nT 3\nH 0 1 2 3"),
}


def _programs(text: str, postselection_mask: list[int] | None = None) -> tuple:
    hir = clifft.trace(clifft.parse(text), postselection_mask=postselection_mask or [])
    return clifft.lower(hir), _lower_runtime_frame(hir)


def _postselected_pairs(active_qubits: int) -> str:
    """Eight signed pairs with one postselected check each.

    Every further qubit carries a T gate between Hadamards and needs an active
    coordinate outside the frame.
    """
    pairs = [(2 * p, 2 * p + 1) for p in range(8)]
    others = range(16, 16 + active_qubits)
    lines = [f"H {a}\nCX {a} {b}\nX_ERROR(0.05) {a}" for a, b in pairs]
    lines += [f"H {q}" for q in others]
    lines += [f"T {a}\nT_DAG {b}" for a, b in pairs]
    lines += [f"T {q}" for q in others]
    lines += [f"MPP X{a}*X{b}\nDETECTOR rec[-1]" for a, b in pairs]
    lines += [f"H {q}\nM {q}" for q in others]
    return "\n".join(lines) + "\n"


def _pattern_counts(rows: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(rows.shape[1])
    return np.bincount(rows @ weights, minlength=1 << rows.shape[1])


def _assert_same_distribution(a: np.ndarray, b: np.ndarray) -> None:
    """Two-sample check at six standard deviations for every outcome pattern."""
    counts_a = _pattern_counts(a)
    counts_b = _pattern_counts(b)
    shots = a.shape[0]
    assert b.shape[0] == shots
    pooled = (counts_a + counts_b) / (2 * shots)
    sigma = np.sqrt(2 * pooled * (1 - pooled) / shots)
    assert np.all(np.abs(counts_a - counts_b) / shots <= 6 * sigma + 1e-12)


def _assert_same_survivor_counts(expected: object, actual: object, shots: int) -> None:
    """Two-sample check at six standard deviations for the survivor counts."""
    for attribute in ("passed_shots", "logical_errors"):
        a = getattr(expected, attribute)
        b = getattr(actual, attribute)
        pooled = (a + b) / (2 * shots)
        assert abs(a - b) / shots <= 6 * np.sqrt(2 * pooled * (1 - pooled) / shots)


def test_frame_plan_needs_no_active_coordinate() -> None:
    fixed, frame = _programs(SIGNED_PAIR)
    assert fixed.peak_active_width == 1
    assert frame.peak_active_width == 0
    assert "FRAME_CLIFFORD" in frame.inspect()
    assert "FRAME_CLIFFORD" not in fixed.inspect()


def test_sample_follows_the_ordinary_plan() -> None:
    fixed, frame = _programs(SIGNED_PAIR)
    shots = 40_000
    expected = clifft.sample(fixed, shots, seed=11)
    actual = clifft.sample(frame, shots, seed=12)
    _assert_same_distribution(expected.measurements, actual.measurements)
    _assert_same_distribution(
        np.hstack([expected.detectors, expected.observables]),
        np.hstack([actual.detectors, actual.observables]),
    )


def test_sample_is_repeatable_and_ignores_batch_size() -> None:
    _, frame = _programs(SIGNED_PAIR)
    first = clifft.sample(frame, 512, seed=5)
    again = clifft.sample(frame, 512, seed=5)
    np.testing.assert_array_equal(first.measurements, again.measurements)
    # Frame plans always run on the scalar executor.
    batched = clifft.sample(frame, 512, seed=5, batch_size=64)
    np.testing.assert_array_equal(first.measurements, batched.measurements)


@pytest.mark.parametrize("keep_records", [False, True])
def test_sample_survivors_follows_the_ordinary_plan(keep_records: bool) -> None:
    fixed, frame = _programs(SIGNED_PAIR, postselection_mask=[1])
    shots = 40_000
    expected = clifft.sample_survivors(fixed, shots, seed=21, keep_records=keep_records)
    actual = clifft.sample_survivors(frame, shots, seed=22, keep_records=keep_records)
    _assert_same_survivor_counts(expected, actual, shots)
    if keep_records:
        assert actual.measurements.shape == (actual.passed_shots, 4)
        assert not actual.detectors.any()


def test_record_probabilities_match_the_ordinary_plan() -> None:
    fixed, frame = _programs(RANDOM_OUTCOME)
    assert frame.peak_active_width < fixed.peak_active_width
    records = [f"{value:03b}" for value in range(8)]
    expected = clifft.record_probabilities(fixed, records)
    actual = clifft.record_probabilities(frame, records)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual.sum() == pytest.approx(1.0)


def test_reference_replay_matches_record_probabilities() -> None:
    """The later T gates promote frame coordinates, so the replay grows storage."""
    fixed, frame = _programs("H 0 1\nMPP Z0*Z1\nT 0\nT_DAG 1\nH 0 1\nT 0\nT 1\nH 0 1\nM 0 1\n")
    assert frame.peak_active_width == fixed.peak_active_width == 2
    records = [f"{value:03b}" for value in range(8)]
    expected = clifft.record_probabilities(frame, records)
    for record, probability in zip(records, expected):
        replay = _replay_record(frame, [int(bit) for bit in record])
        actual = np.exp(replay["log_probability"]) if replay["reachable"] else 0.0
        assert actual == pytest.approx(probability, abs=1e-12)


@pytest.mark.parametrize(
    "num_qubits,preparation,parity,gates", PARITY_CASES.values(), ids=PARITY_CASES.keys()
)
def test_record_probabilities_match_qiskit(
    num_qubits: int, preparation: str, parity: tuple[int, ...], gates: str
) -> None:
    """Aer runs the same circuit with the parity copied onto an ancilla."""
    qubits = " ".join(map(str, range(num_qubits)))
    product = "*".join(f"Z{q}" for q in parity)
    fixed, frame = _programs(f"{preparation}\nMPP {product}\n{gates}\nM {qubits}")
    assert frame.peak_active_width < fixed.peak_active_width

    ancilla = num_qubits
    copies = " ".join(f"{q} {ancilla}" for q in parity)
    amplitudes = qiskit_statevector(
        stim_to_qiskit_noiseless(f"{preparation}\nCX {copies}\n{gates}")
    )
    # Record 0 is the parity outcome and record 1 + q is the final measurement
    # of qubit q. Aer's basis index holds qubit q in bit q and the ancilla on top.
    records = [
        str(index >> num_qubits) + "".join(str((index >> q) & 1) for q in range(num_qubits))
        for index in range(2 << num_qubits)
    ]
    actual = clifft.record_probabilities(frame, records)
    np.testing.assert_allclose(actual, np.abs(amplitudes) ** 2, atol=1e-12)


def test_noisy_region_matches_qiskit_density_matrix() -> None:
    """A bit flip decides whether the CCZ acts as a CZ or as the identity."""
    preparation, gates, flip = "H 0 2\nCX 0 1", "CCZ 0 1 2\nH 0 1 2", 0.3
    fixed, frame = _programs(f"{preparation}\nX_ERROR({flip}) 1\n{gates}\nM 0 1 2")
    assert frame.peak_active_width < fixed.peak_active_width

    circuit = stim_to_qiskit_noiseless(preparation)
    circuit.append(pauli_error([("X", flip), ("I", 1 - flip)]), [1])
    circuit = circuit.compose(stim_to_qiskit_noiseless(gates))
    circuit.save_probabilities()
    result = AerSimulator(method="density_matrix").run(circuit, shots=1).result()
    expected = np.asarray(result.data()["probabilities"])

    shots = 200_000
    frequencies = _pattern_counts(clifft.sample(frame, shots, seed=7).measurements) / shots
    sigma = np.sqrt(expected * (1 - expected) / shots)
    assert np.all(np.abs(frequencies - expected) <= 6 * sigma)


def test_calibration_keeps_frame_plans_on_the_scalar_executor() -> None:
    _, frame = _programs(SIGNED_PAIR, postselection_mask=[1])
    result = clifft.sample_survivors(
        frame, 4096, seed=5, batch_size="tune", tuning_budget_seconds=0.05
    )
    report = result.batch_tuning
    assert report.batch_size == 1
    assert report.stop_reason == "single_candidate"
    pinned = clifft.sample_survivors(frame, 4096, seed=5)
    assert result.passed_shots == pinned.passed_shots


@pytest.mark.parametrize("active_qubits", [0, 1])
def test_sampling_selection_keeps_frame_plans_on_the_scalar_executor(active_qubits: int) -> None:
    _, frame = _programs(_postselected_pairs(active_qubits), postselection_mask=[1] * 8)
    assert frame.peak_active_width == active_qubits
    assert "FRAME_CLIFFORD" in frame.inspect()
    scalar = clifft.sample_survivors(frame, 16384, seed=5, threads=1, batch_size=1)
    automatic = clifft.sample_survivors(frame, 16384, seed=5, threads=1)
    assert automatic.execution.executor == "scalar"
    assert automatic.execution.sharing is False
    assert automatic.passed_shots == scalar.passed_shots
    for controls in ({"sharing": True}, {"noise_preparation": clifft.NoisePreparation.DENSE}):
        with pytest.raises(ValueError, match="runtime Clifford frame"):
            clifft.sample_survivors(frame, 64, seed=5, **controls)


def _rounds(count: int) -> str:
    """Rounds of a signed pair and a plain T qubit that reset every qubit."""
    one_round = "H 0 2\nCX 0 1\nX_ERROR(0.3) 0\nT 0 2\nT_DAG 1\nCX 0 1\nH 0 2\nMR 0 1 2\n"
    return one_round * count


def test_repeated_rounds_retire_the_frame() -> None:
    ordinary, frame = _programs(_rounds(3))
    # Every round ends with its qubits reset, so the frame ends there and the
    # next round is reduced like the first.
    assert "FRAME_RETIREMENT" in frame.original_plan.inspect()
    assert "FRAME_ROTATION" not in frame.original_plan.inspect()
    assert frame.original_plan.peak_active_width == 1
    assert ordinary.original_plan.peak_active_width == 2
    shots = 60_000
    expected = clifft.sample(ordinary, shots, seed=11).measurements
    actual = clifft.sample(frame, shots, seed=12).measurements
    _assert_same_distribution(expected, actual)


def _rounds_beside_data(count: int) -> str:
    """Rounds of a signed pair that reset it, beside a pair checked only at the end."""
    data = "H 4\nCX 4 5\nX_ERROR(0.3) 4\nT 4\nT_DAG 5\n"
    one_round = "H 0\nCX 0 1\nX_ERROR(0.3) 0\nT 0\nT_DAG 1\nCX 0 1\nH 0\nMR 0 1\n"
    return data + one_round * count + "MPP X4*X5\nM 4 5\n"


def test_rounds_beside_unmeasured_data_release_frame_coordinates() -> None:
    ordinary, frame = _programs(_rounds_beside_data(3))
    # The unmeasured pair keeps the frame live. Each round's reset fixes that
    # round's coordinate, which the plan hands back before the next round, so
    # every round is reduced like the first.
    listing = frame.original_plan.inspect()
    assert listing.count("FRAME_RELEASE") == 2
    assert "FRAME_ROTATION" not in listing
    assert frame.original_plan.peak_active_width == 0
    assert ordinary.original_plan.peak_active_width == 2
    shots = 60_000
    expected = clifft.sample(ordinary, shots, seed=41).measurements
    actual = clifft.sample(frame, shots, seed=42).measurements
    _assert_same_distribution(expected, actual)


@pytest.mark.parametrize("k", [0, 1])
def test_fixed_fault_sampling_follows_the_ordinary_plan(k: int) -> None:
    fixed, frame = _programs(SIGNED_PAIR)
    shots = 40_000
    expected = clifft.sample_k(fixed, shots, k, seed=31)
    actual = clifft.sample_k(frame, shots, k, seed=32)
    _assert_same_distribution(expected.measurements, actual.measurements)


@pytest.mark.parametrize("k", [0, 1])
def test_fixed_fault_survivor_sampling_follows_the_ordinary_plan(k: int) -> None:
    fixed, frame = _programs(SIGNED_PAIR, postselection_mask=[1])
    shots = 40_000
    expected = clifft.sample_k_survivors(fixed, shots, k, seed=41)
    actual = clifft.sample_k_survivors(frame, shots, k, seed=42)
    _assert_same_survivor_counts(expected, actual, shots)
