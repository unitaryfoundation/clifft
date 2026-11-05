"""Automatic packing accounts for the classical work of a fixed leak/loss plan."""

import numpy as np
import pytest
from utils_conformance import assert_joint_distribution, unitary_reference

from clifft import noncomp


def circuit(width, *, depth=0, inputs=256, probability=0.3, gate="T"):
    targets = " ".join(map(str, range(width)))
    text = f"H {targets}\n{gate} {targets}\nLOSS({probability}) {width}\n"
    if depth:
        text += f"REPEAT {depth} {{\nR_X(0.173) {targets}\nR_Z(0.291) {targets}\n}}\n"
    if inputs:
        text += f"REPEAT {inputs} {{\nX_ERROR(0.01) {width}\nM {width}\n}}\n"
    text += f"H {targets}\nM {targets}\nHERALD_LOSS_EVENT {width}\n"
    text += "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"
    return text


def model():
    return noncomp.Model(
        classifier=noncomp.Classifier([[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]])
    )


def same_rows(left, right):
    for name in ("measurements", "heralds", "detectors", "observables", "final_status"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)


@pytest.mark.parametrize("prepared", [False, True], ids=["convenience", "prepared"])
@pytest.mark.parametrize("width", [6, 8])
def test_auto_packs_wider_classical_work_and_replays_without_calibration(prepared, width):
    text = circuit(width)
    noise = model()
    program = noncomp.compile(text, noise, final_status=True) if prepared else None

    def sample(**kwargs):
        if program is not None:
            return program.sample(**kwargs)
        return noncomp.sample(text, noise, final_status=True, **kwargs)

    options = dict(shots=2049, seed=82, collect_stats=True)
    serial = sample(**options)
    parallel = sample(**options, threads=3)
    stats = parallel.stats
    assert stats.strategy == "fixed"
    assert stats.peak_active_width == width
    assert stats.packed_shots == stats.packed_classical_shots == 2049
    assert stats.scalar_shots == stats.continuation_compilations == 0
    assert stats.workers == 3
    assert stats.batch_size == serial.stats.batch_size > 1
    assert stats.packed_batches == (2049 + stats.batch_size - 1) // stats.batch_size
    assert 0 < stats.packed_worker_bytes <= 8 * 1024**2
    assert stats.packed_worker_bytes * stats.workers <= 64 * 1024**2
    assert serial.batch_tuning is parallel.batch_tuning is None
    same_rows(serial, parallel)
    same_rows(parallel, sample(**options, threads=3, batch_size=stats.batch_size))
    zero_budget = sample(**options, batch_size="tune", tuning_budget_seconds=0)
    assert zero_budget.batch_tuning.trial_shots == 0
    assert zero_budget.batch_tuning.baseline_batch_size == stats.batch_size
    same_rows(serial, zero_budget)
    forced = sample(**options, batch_size=1)
    assert forced.stats.scalar_shots == 2049
    assert forced.stats.packed_shots == 0


@pytest.mark.parametrize(
    "width,depth,inputs,probability",
    [(6, 128, 256, 0.3), (6, 0, 0, 0.3), (6, 0, 256, 0), (9, 0, 256, 0.3)],
    ids=["sustained-quantum-work", "little-classical-work", "static-inputs", "wider-state"],
)
def test_auto_preserves_conservative_fallbacks(width, depth, inputs, probability):
    program = noncomp.compile(
        circuit(width, depth=depth, inputs=inputs, probability=probability), model()
    )
    result = program.sample(129, seed=83, collect_stats=True)
    assert result.stats is not None
    assert result.stats.strategy == "fixed"
    assert result.stats.peak_active_width == width
    assert result.stats.batch_size == 1
    assert result.stats.scalar_shots == 129
    same_rows(result, program.sample(129, seed=83, batch_size=1))
    forced = program.sample(129, seed=83, batch_size=65, collect_stats=True)
    assert forced.stats is not None
    assert forced.stats.packed_shots == 129


@pytest.mark.parametrize("shots", [0, 1, 63, 64, 65])
def test_wider_auto_preserves_small_request_boundary(shots):
    program = noncomp.compile(circuit(6), model())
    result = program.sample(shots, seed=84, collect_stats=True)
    assert result.stats is not None
    assert result.stats.batch_size == (1 if shots < 64 else shots)
    assert result.stats.packed_shots == (0 if shots < 64 else shots)
    assert result.batch_tuning is None


@pytest.mark.parametrize("gate", ["T", "R_Z(0.173)"])
def test_wider_auto_matches_aer_and_independent_loss_status(gate):
    width = 6
    program = noncomp.compile(circuit(width, gate=gate), model(), final_status=True)
    result = program.sample(16000, seed=85, collect_stats=True)
    assert result.stats is not None
    assert result.final_status is not None
    assert result.stats.peak_active_width == width
    assert result.stats.packed_shots == result.stats.packed_classical_shots == 16000
    # The six product qubits have no interaction with the noisy, lost auxiliary.
    # Aer gives their quantum law; the final probe and status must agree per row.
    targets = " ".join(map(str, range(width)))
    expected = np.abs(unitary_reference(f"H {targets}\n{gate} {targets}\nH {targets}")) ** 2
    assert_joint_distribution(result.measurements[:, 256:], np.kron([0.7, 0.3], expected))
    loss = result.measurements[:, -1]
    np.testing.assert_array_equal(result.final_status[:, width] == noncomp.QubitStatus.LOST, loss)
    np.testing.assert_array_equal(result.detectors[:, 0], loss)
    np.testing.assert_array_equal(result.observables[:, 0], result.measurements[:, -2])
    assert not result.heralds.any()
