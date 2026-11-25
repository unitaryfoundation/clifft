"""Tests for sampling programs planned with a runtime Clifford frame.

These tests compare programs compiled with ``runtime_frame=True`` with
ordinary ones and with Qiskit Aer through the public sampling functions, on
the scalar executor and with ``sharing=True``.
"""

from typing import Literal

import numpy as np
import pytest
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error
from utils_qiskit import qiskit_statevector, stim_to_qiskit_noiseless

import clifft
from clifft._clifft_core import _replay_record

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
    """The ordinary and the runtime-frame program of one unoptimized circuit."""
    ordinary, frame = (
        clifft.compile(
            text,
            postselection_mask=postselection_mask,
            hir_passes=None,
            reconstruction=False,
            runtime_frame=runtime_frame,
        )
        for runtime_frame in (False, True)
    )
    return ordinary, frame


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
    lines += [f"H {q}\nM {q}\nOBSERVABLE_INCLUDE(0) rec[-1]" for q in others]
    return "\n".join(lines) + "\n"


def _signed_pairs(pairs: int, plain: int, probability: float = 0.05) -> str:
    """Signed pairs, each with one check, and plain qubits with a T gate.

    A pair needs an active qubit in the ordinary plan and none on a runtime
    frame. A plain qubit needs one either way.
    """
    wires = [(2 * p, 2 * p + 1) for p in range(pairs)]
    others = range(2 * pairs, 2 * pairs + plain)
    lines = [f"H {a}\nCX {a} {b}\nX_ERROR({probability}) {a}" for a, b in wires]
    lines += [f"H {q}" for q in others]
    lines += [f"T {a}\nT_DAG {b}" for a, b in wires]
    lines += [f"T {q}" for q in others]
    lines += [f"MPP X{a}*X{b}\nDETECTOR rec[-1]" for a, b in wires]
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


@pytest.mark.parametrize("sharing", [False, True])
def test_sample_follows_the_ordinary_plan(sharing: bool) -> None:
    fixed, frame = _programs(SIGNED_PAIR)
    shots = 40_000
    expected = clifft.sample(fixed, shots, seed=11)
    actual = clifft.sample(frame, shots, seed=12, sharing=sharing)
    assert actual.execution.executor == ("shared" if sharing else "scalar")
    _assert_same_distribution(expected.measurements, actual.measurements)
    _assert_same_distribution(
        np.hstack([expected.detectors, expected.observables]),
        np.hstack([actual.detectors, actual.observables]),
    )


def test_sample_is_repeatable_and_a_packed_capacity_runs_scalar() -> None:
    _, frame = _programs(SIGNED_PAIR)
    first = clifft.sample(frame, 512, seed=5)
    again = clifft.sample(frame, 512, seed=5)
    np.testing.assert_array_equal(first.measurements, again.measurements)
    # Packed execution cannot run a frame, so a packed capacity runs scalar.
    scalar = clifft.sample(frame, 512, seed=5, batch_size=1)
    batched = clifft.sample(frame, 512, seed=5, batch_size=64)
    assert batched.execution.executor == "scalar"
    np.testing.assert_array_equal(scalar.measurements, batched.measurements)


@pytest.mark.parametrize("keep_records", [False, True])
def test_sample_survivors_follows_the_ordinary_plan(keep_records: bool) -> None:
    fixed, frame = _programs(SIGNED_PAIR, postselection_mask=[1])
    shots = 40_000
    expected = clifft.sample_survivors(fixed, shots, seed=21, keep_records=keep_records)
    # The automatic choice shares a frame plan at this shot count.
    actual = clifft.sample_survivors(
        frame, shots, seed=22, keep_records=keep_records, sharing=False
    )
    assert actual.execution.executor == "scalar"
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
    expected = np.abs(amplitudes) ** 2
    np.testing.assert_allclose(clifft.record_probabilities(frame, records), expected, atol=1e-12)

    # Shared execution keeps the random parity outcome in each shot's sign bits.
    shots = 100_000
    rows = clifft.sample(frame, shots, seed=9, sharing=True).measurements
    weights = np.array([1 << num_qubits] + [1 << q for q in range(num_qubits)])
    frequencies = np.bincount(rows @ weights, minlength=2 << num_qubits) / shots
    sigma = np.sqrt(expected * (1 - expected) / shots)
    assert np.all(np.abs(frequencies - expected) <= 6 * sigma + 1e-12)


@pytest.mark.parametrize("sharing", [False, True])
def test_noisy_region_matches_qiskit_density_matrix(sharing: bool) -> None:
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
    rows = clifft.sample(frame, shots, seed=7, sharing=sharing).measurements
    frequencies = _pattern_counts(rows) / shots
    sigma = np.sqrt(expected * (1 - expected) / shots)
    assert np.all(np.abs(frequencies - expected) <= 6 * sigma)


def test_calibration_tunes_the_shared_capacity_of_a_frame_plan() -> None:
    _, frame = _programs(SIGNED_PAIR, postselection_mask=[1])
    result = clifft.sample_survivors(
        frame, 4096, seed=5, batch_size="tune", tuning_budget_seconds=0.05
    )
    report = result.batch_tuning
    assert report.sharing is True
    assert result.execution.executor == "shared"
    repeated = clifft.sample_survivors(
        frame, 4096, seed=5, sharing=True, batch_size=report.batch_size
    )
    assert result.passed_shots == repeated.passed_shots


@pytest.mark.parametrize("active_qubits", [0, 1])
def test_automatic_selection_shares_postselected_frame_plans(active_qubits: int) -> None:
    _, frame = _programs(_postselected_pairs(active_qubits), postselection_mask=[1] * 8)
    assert frame.peak_active_width == active_qubits
    assert frame.original_plan.uses_runtime_frame
    automatic = clifft.sample_survivors(frame, 16384, seed=5, threads=1)
    assert automatic.execution.executor == "shared"
    requested = clifft.sample_survivors(frame, 16384, seed=5, threads=1, sharing=True)
    assert automatic.passed_shots == requested.passed_shots
    # A short call, an explicit refusal and a packed capacity all run scalar.
    for controls in ({"shots": 63}, {"sharing": False}, {"batch_size": 96}):
        shots = controls.pop("shots", 16384)
        scalar = clifft.sample_survivors(frame, shots, seed=5, threads=1, **controls)
        assert scalar.execution.executor == "scalar"
        assert scalar.execution.sharing is False
    # Packed execution cannot run a frame, so without sharing there is no noise to prepare.
    with pytest.raises(ValueError, match="sharing=True"):
        clifft.sample_survivors(frame, 64, seed=5, noise_preparation=clifft.NoisePreparation.DENSE)


def test_automatic_selection_without_postselection_sizes_batches_by_divergence() -> None:
    # Rare faults leave most shots on one frame, so the batch shares it whole.
    rare = _signed_pairs(pairs=8, plain=1, probability=0.01)
    _, frame = _programs(rare)
    quiet = clifft.sample(frame, 4096, seed=1, threads=1)
    assert quiet.execution.executor == "shared"
    assert quiet.execution.batch_size == 4096
    # A fair coin per pair decides its signs, so nearly every shot has its own
    # frame. Shared execution still matches the scalar executor there, in
    # batches small enough to keep their tableaus in the cache.
    coins = "".join(
        f"H {a} {b}\nMPP Z{a}*Z{b}\nT {a}\nT_DAG {b}\nH {a} {b}\nM {a} {b}\n"
        for a, b in ((2 * p, 2 * p + 1) for p in range(13))
    )
    _, frame = _programs(coins)
    assert frame.original_plan.uses_runtime_frame
    divergent = clifft.sample(frame, 4096, seed=1, threads=1)
    assert divergent.execution.executor == "shared"
    assert divergent.execution.batch_size == 256
    # A caller's batch size is kept.
    chosen = clifft.sample(frame, 4096, seed=1, threads=1, sharing=True, batch_size=1024)
    assert chosen.execution.batch_size == 1024


def _rounds(count: int, detector: bool = False) -> str:
    """Rounds of a signed pair and a plain T qubit that reset every qubit.

    With ``detector``, each round checks the pair, which a fault flips.
    """
    one_round = "H 0 2\nCX 0 1\nX_ERROR(0.3) 0\nT 0 2\nT_DAG 1\nCX 0 1\nH 0 2\nMR 0 1 2\n"
    if detector:
        one_round += "DETECTOR rec[-3]\n"
    return one_round * count


@pytest.mark.parametrize(
    "controls",
    [{}, {"sharing": False}, {"sharing": True}, {"sharing": True, "group_by_noise": True}],
    ids=["automatic", "scalar", "shared", "noise-groups"],
)
def test_repeated_rounds_retire_the_frame(controls: dict) -> None:
    ordinary, frame = _programs(_rounds(3))
    # Every round ends with its qubits reset, so the frame ends there and the
    # next round is reduced like the first.
    assert "FRAME_RETIREMENT" in frame.original_plan.inspect()
    assert "FRAME_ROTATION" not in frame.original_plan.inspect()
    assert frame.original_plan.peak_active_width == 1
    assert ordinary.original_plan.peak_active_width == 2
    shots = 60_000
    expected = clifft.sample(ordinary, shots, seed=11).measurements
    actual = clifft.sample(frame, shots, seed=12, **controls).measurements
    _assert_same_distribution(expected, actual)


def test_postselected_rounds_share_through_retirement() -> None:
    ordinary, frame = _programs(_rounds(3, detector=True), postselection_mask=[1, 1, 1])
    shots = 60_000
    expected = clifft.sample_survivors(ordinary, shots, seed=31, keep_records=True)
    actual = clifft.sample_survivors(frame, shots, seed=32, keep_records=True)
    assert actual.execution.executor == "shared"
    assert 0 < actual.passed_shots < shots
    _assert_same_survivor_counts(expected, actual, shots)
    _assert_same_distribution(
        expected.measurements[: min(expected.passed_shots, actual.passed_shots)],
        actual.measurements[: min(expected.passed_shots, actual.passed_shots)],
    )


def test_runtime_frame_is_off_unless_requested() -> None:
    text = _signed_pairs(pairs=4, plain=0)
    default = clifft.compile(text)
    assert not default.original_plan.uses_runtime_frame
    assert default.runtime_frame_fallback_reason
    frame = clifft.compile(text, runtime_frame=True)
    assert frame.original_plan.uses_runtime_frame
    assert frame.runtime_frame_fallback_reason == ""
    assert frame.original_plan.peak_active_width == 0
    assert default.original_plan.peak_active_width > 0
    with pytest.raises(ValueError, match="runtime_frame"):
        clifft.compile(text, runtime_frame="always")  # type: ignore[arg-type]
    shots = 40_000
    expected = clifft.sample(default, shots, seed=3, keep_records="syndromes").detectors
    actual = clifft.sample(frame, shots, seed=4, keep_records="syndromes").detectors
    _assert_same_distribution(expected, actual)


def test_runtime_frame_true_keeps_the_ordinary_plan_where_no_sign_varies() -> None:
    text = "H 0 1\nT 0 1\nCX 0 1\nT 1\nH 0 1\nM 0 1\n"
    ordinary = clifft.compile(text)
    frame = clifft.compile(text, runtime_frame=True)
    assert not frame.original_plan.uses_runtime_frame
    assert frame.runtime_frame_fallback_reason
    assert frame.original_plan.inspect() == ordinary.original_plan.inspect()


@pytest.mark.parametrize(
    ("pairs", "plain", "selected"),
    [(16, 0, True), (15, 0, False), (4, 12, True), (3, 13, False)],
    ids=["wide", "narrow", "four-fewer", "three-fewer"],
)
def test_automatic_runtime_frame_needs_a_wide_plan_and_a_clear_reduction(
    pairs: int, plain: int, selected: bool
) -> None:
    # Without passes the ordinary plan holds every pair and plain qubit at once.
    text = _signed_pairs(pairs, plain)
    ordinary = clifft.compile(text, hir_passes=None, reconstruction=False)
    assert ordinary.original_plan.peak_active_width == pairs + plain
    forced = clifft.compile(text, hir_passes=None, reconstruction=False, runtime_frame=True)
    assert forced.original_plan.peak_active_width == plain
    automatic = clifft.compile(text, hir_passes=None, reconstruction=False, runtime_frame="auto")
    assert automatic.original_plan.uses_runtime_frame is selected
    assert (automatic.runtime_frame_fallback_reason == "") is selected
    chosen = forced if selected else ordinary
    assert automatic.original_plan.inspect() == chosen.original_plan.inspect()


def test_runtime_frame_plans_a_circuit_too_wide_for_the_ordinary_plan() -> None:
    text = _signed_pairs(pairs=70, plain=0)
    with pytest.raises(OverflowError):
        clifft.compile(text, hir_passes=None, reconstruction=False)
    choices: tuple[bool | Literal["auto"], ...] = (True, "auto")
    for runtime_frame in choices:
        program = clifft.compile(
            text, hir_passes=None, reconstruction=False, runtime_frame=runtime_frame
        )
        assert program.original_plan.uses_runtime_frame
        assert program.original_plan.peak_active_width == 0
    result = clifft.sample(program, 256, seed=2, keep_records="syndromes")
    assert result.detectors.shape == (256, 70)


def test_runtime_frame_is_planned_before_the_passes() -> None:
    text = _signed_pairs(pairs=4, plain=2)
    frame = clifft.compile(text, reconstruction=False, runtime_frame=True)
    unoptimized = clifft.compile(text, hir_passes=None, reconstruction=False, runtime_frame=True)
    assert frame.original_plan.inspect() == unoptimized.original_plan.inspect()
    # The passes reorder this circuit: the ordinary plans differ, the frame
    # plans do not.
    explicit = clifft.compile(
        text,
        hir_passes=clifft.default_hir_pass_manager(),
        reconstruction=False,
        runtime_frame=True,
    )
    assert explicit.original_plan.inspect() == frame.original_plan.inspect()
    assert (
        clifft.compile(text, reconstruction=False).original_plan.inspect()
        != clifft.compile(text, hir_passes=None, reconstruction=False).original_plan.inspect()
    )


def test_passes_that_change_the_distribution_apply_to_the_runtime_frame_plan() -> None:
    text = _signed_pairs(pairs=4, plain=0)
    assert clifft.compile(text, runtime_frame=True).original_plan.uses_runtime_frame
    # Without its noise the circuit has no sign that varies between shots.
    passes = clifft.HirPassManager()
    passes.add(clifft.RemoveNoisePass())
    noiseless = clifft.compile(text, hir_passes=passes, runtime_frame=True)
    assert not noiseless.original_plan.uses_runtime_frame


def test_runtime_frame_keeps_normalized_syndromes() -> None:
    # The pair check reads one without noise; normalization makes that zero.
    text = "X 0\n" + _signed_pairs(pairs=3, plain=0, probability=0.02)
    frame = clifft.compile(text, normalize_syndromes=True, runtime_frame=True)
    assert frame.original_plan.uses_runtime_frame
    ordinary = clifft.compile(text, normalize_syndromes=True)
    shots = 20_000
    expected = clifft.sample(ordinary, shots, seed=8, keep_records="syndromes").detectors
    actual = clifft.sample(frame, shots, seed=9, keep_records="syndromes").detectors
    assert actual.mean() < 0.2
    _assert_same_distribution(expected, actual)


@pytest.mark.parametrize("active_qubits", [0, 1])
@pytest.mark.parametrize(
    "controls",
    [
        {},
        {"keep_records": True},
        {"group_by_noise": True},
        {"batch_size": 96},
        *({"noise_preparation": mode} for mode in clifft.NoisePreparation),
        {"batch_size": "tune", "tuning_budget_seconds": 0.05},
    ],
    ids=[
        "default",
        "rows",
        "noise-groups",
        "capacity",
        *(f"noise-{mode.name.lower()}" for mode in clifft.NoisePreparation),
        "tuned",
    ],
)
def test_shared_survivor_sampling_follows_the_ordinary_plan(
    active_qubits: int, controls: dict
) -> None:
    fixed, frame = _programs(_postselected_pairs(active_qubits), postselection_mask=[1] * 8)
    shots = 40_000
    expected = clifft.sample_survivors(fixed, shots, seed=51)
    actual = clifft.sample_survivors(frame, shots, seed=52, sharing=True, **controls)
    assert actual.execution.executor == "shared"
    assert actual.execution.sharing is True
    _assert_same_survivor_counts(expected, actual, shots)
    if active_qubits:
        assert actual.logical_errors > 0
    if controls.get("keep_records"):
        assert actual.measurements.shape == (actual.passed_shots, 8 + active_qubits)
        assert not actual.detectors.any()
    # A seeded call repeats at the capacity it reports.
    replayed = {
        name: value
        for name, value in controls.items()
        if name not in ("batch_size", "tuning_budget_seconds")
    }
    again = clifft.sample_survivors(
        frame, shots, seed=52, sharing=True, batch_size=actual.execution.batch_size, **replayed
    )
    assert again.passed_shots == actual.passed_shots
    assert again.logical_errors == actual.logical_errors


def test_shared_frame_sampling_does_not_depend_on_the_worker_count() -> None:
    _, frame = _programs(_postselected_pairs(1), postselection_mask=[1] * 8)
    one, three = (
        clifft.sample_survivors(
            frame, 20_000, seed=9, sharing=True, batch_size=512, threads=threads, keep_records=True
        )
        for threads in (1, 3)
    )
    assert one.passed_shots == three.passed_shots
    np.testing.assert_array_equal(one.measurements, three.measurements)
    np.testing.assert_array_equal(one.observables, three.observables)


def test_shared_frame_sampling_needs_room_for_two_groups() -> None:
    # The frame needs no active coordinate, but each group still holds a tableau.
    _, frame = _programs(SIGNED_PAIR)
    assert frame.peak_active_width == 0
    with pytest.raises(ValueError, match="at least two shared states"):
        clifft.sample(frame, 64, seed=1, sharing=True, max_worker_state_bytes=4096)


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


def _parity_signed_pairs(pairs: int) -> str:
    """Random parities sign a T pair on each Bell pair; one padded record is checked."""
    qubits = " ".join(map(str, range(2 * pairs)))
    evens = " ".join(str(2 * p) for p in range(pairs))
    odds = " ".join(str(2 * p + 1) for p in range(pairs))
    parities = "".join(f"MPP Z{2 * p}*Z{2 * p + 1}\n" for p in range(pairs))
    return (
        f"H {qubits}\n{parities}T {evens}\nT_DAG {odds}\nH {qubits}\nM {qubits}\n"
        "MPAD 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    )


def test_normalized_reference_uses_the_frame_when_the_ordinary_plan_overflows() -> None:
    """With 65 pairs no dense state holds the ordinary plan."""
    text = _parity_signed_pairs(65)
    with pytest.raises(OverflowError):
        clifft.compile(text, normalize_syndromes=True, hir_passes=None, reconstruction=False)
    raw = clifft.compile(text, runtime_frame=True, hir_passes=None, reconstruction=False)
    normalized = clifft.compile(
        text, runtime_frame=True, normalize_syndromes=True, hir_passes=None, reconstruction=False
    )
    for program, expected in ((raw, 1), (normalized, 0)):
        result = clifft.sample(program, 64, seed=3)
        assert (result.detectors == expected).all()
        assert (result.observables == expected).all()


@pytest.mark.parametrize("pairs", [40, 65])
@pytest.mark.parametrize("runtime_frame", [True, "auto"])
def test_normalized_reference_is_planned_on_a_frame_with_default_settings(
    pairs: int, runtime_frame: bool | Literal["auto"]
) -> None:
    """At 40 pairs the ordinary reference plans but no state holds it.

    The default passes split the regions a frame reduces, so the reference is
    traced from the circuit again.
    """
    program = clifft.compile(
        _parity_signed_pairs(pairs), runtime_frame=runtime_frame, normalize_syndromes=True
    )
    # The frame plan needs no active qubit; "auto" may sample a reconstruction of width one.
    assert program.peak_active_width <= 1
    result = clifft.sample(program, 64, seed=3)
    assert not result.detectors.any()
    assert not result.observables.any()


def test_reduced_plan_without_a_frame_keeps_the_statevector() -> None:
    """T T reduces to S with no frame action left, and the reduced plan is kept."""
    text = "H 0 1\nCX 0 1\nT 0\nT 0\nT_DAG 1\nH 1\n"
    ordinary, reduced = _programs(text)
    assert not reduced.original_plan.uses_runtime_frame
    assert reduced.peak_active_width < ordinary.peak_active_width
    expected = clifft.get_statevector(ordinary)
    actual = clifft.get_statevector(reduced)
    assert abs(np.vdot(expected, actual)) == pytest.approx(1.0, abs=1e-12)


def test_bit_packed_detectors_match_unpacked_rows_on_a_frame_plan() -> None:
    _, frame = _programs(SIGNED_PAIR)
    assert frame.original_plan.uses_runtime_frame
    rows = clifft.sample_detectors(frame, 4096, seed=8)
    packed = clifft.sample_detectors(frame, 4096, seed=8, bit_packed=True)
    unpacked = np.unpackbits(packed, axis=1, bitorder="little")[:, : rows.shape[1]]
    np.testing.assert_array_equal(unpacked.astype(bool), rows)
    assert rows.any()


@pytest.mark.parametrize("group_by_noise", [False, True])
def test_noise_groups_follow_a_frame_measurement_that_signs_a_later_rotation(
    group_by_noise: bool,
) -> None:
    """The parity outcome applies an X between two rotations, which then cancel."""
    text = (
        "H 0\nCX 0 1\nX_ERROR(0.3) 0\nT 0\nT_DAG 1\nMPP X0*X1\n"
        "H 2\nR_Z(0.2) 2\nCX rec[-1] 2\nR_Z(0.2) 2\nH 2\nM 2\n"
    )
    _, frame = _programs(text)
    assert frame.original_plan.uses_runtime_frame
    result = clifft.sample(
        frame, 12_000, seed=1, sharing=True, batch_size=128, group_by_noise=group_by_noise
    )
    parity, last = result.measurements[:, 0], result.measurements[:, 1]
    assert parity.any() and last.any()
    assert not (parity & last).any()
