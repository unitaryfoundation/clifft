"""Independent distribution checks for scalar and packed fixed execution."""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import stim
import utils_noncomp_enumerator as dense
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error

from clifft import _clifft_core, noncomp

CLASSIFIER = [[1, 0, 0.7, 0.7, 0.4], [0, 1, 0.3, 0.3, 0.6]]
DEPOLARIZE = noncomp.PartnerEffect(pauli=(0.25, 0.25, 0.25))


def prepare(text, model, **kwargs):
    program = _clifft_core._prepare_noncomputational_fixed(
        _clifft_core.parse(text), model._handle, **kwargs
    )
    assert program is not None, "test must actually use fixed execution"
    return program


def sample(program, shots=16000, seed=43, threads=1, *, batch_size="auto"):
    values, stats = program.sample(shots, seed, threads, batch_size=batch_size)
    assert stats["scalar_shots"] + stats["packed_shots"] == shots
    if batch_size == 1 or shots <= 1:
        assert stats["scalar_shots"] == shots
        assert stats["packed_batches"] == 0
    elif isinstance(batch_size, int):
        assert stats["packed_shots"] == shots, "test must actually use packed execution"
        assert stats["batch_size"] == min(batch_size, shots, 2048)
        assert stats["packed_batches"] == (shots + stats["batch_size"] - 1) // stats["batch_size"]
    return SimpleNamespace(
        measurements=values[0],
        detectors=values[1],
        observables=values[2],
        final_status=values[3],
        heralds=values[4],
        stats=stats,
    )


@pytest.fixture(params=[1, 65], ids=["scalar", "packed"])
def fixed_sample(request):
    return partial(sample, batch_size=request.param)


@pytest.mark.parametrize(
    "text",
    [
        "H 0\nCX 0 1\nLEAKAGE(0.37) 0\nM 0 0 1",
        "H 0\nT 0\nH 0\nCX 0 1\nLEAKAGE(0.37) 0\nM 0 1",
        "H 0\nCX 0 1\nLEAKAGE(0.37) 0\nCX 0 1\nM 0 1",
        "H 1\nT 1\nCX 1 2\nLEAKAGE(0.31) 0\nCX 0 1\nH 1 2\nM 0 1 2",
        "H 0\nCX 0 1\nLOSS(0.41) 1\nCX 0 1\nT 0\nH 0\nM 0 1",
        "H 0\nLEAKAGE(0.43) 0\nMR 0\nH 0\nT 0\nH 0\nM 0",
        "H 0\nCX 0 1\nLEAKAGE(0.29) 0\nLOSS(0.31) 0\nR 0\nCX 0 1\nM 0 1",
        "H 0\nT 0\nCX 0 1\nLEAKAGE(0.34) 0\nT_DAG 0\nH 0\nR 0\nCX 0 1\nM 0 1",
    ],
)
def test_fixed_joint_distribution_matches_dense_physical_model(text, fixed_sample):
    model = noncomp.Model(
        classifier=noncomp.Classifier(CLASSIFIER),
        gate_partner_effects={"leaked": DEPOLARIZE, "lost": DEPOLARIZE},
    )
    expected = dense.enumerate_exact(
        text,
        initial=[1, 0, 0, 0, 0],
        transitions={},
        classifier=CLASSIFIER,
        depolarize_partner=True,
    )
    assert expected.dropped_mass < 1e-10
    program = prepare(text, model)
    before = program.metrics
    result = fixed_sample(program)
    empirical = dense.empirical_record_probs(result.measurements)
    assert dense.tvd(expected.record_probs, empirical) < 0.025
    assert program.metrics == before
    assert before["preparation_compilations"] == 1
    assert result.final_status is None


@pytest.mark.parametrize("rotation", ["T", "R_Z(0.173)"])
@pytest.mark.parametrize("loss_probability", [0.0, 0.4, 1.0])
def test_fixed_non_clifford_partner_channel_matches_aer(rotation, loss_probability, fixed_sample):
    # Physical loss branch: trace out q0, drop the following CX, and apply
    # full depolarization to its live partner. q2 carries
    # an entangled environment, so marginal-only agreement is insufficient.
    text = (
        f"H 0\nCX 0 2\nH 1\n{rotation} 1\nCX 1 2\nLOSS({loss_probability}) 0\n"
        f"CX 0 1\n{rotation} 2\nH 1 2\nM 1 2"
    )
    model = noncomp.Model(
        classifier=noncomp.Classifier(CLASSIFIER), gate_partner_effects={"lost": DEPOLARIZE}
    )
    reference = QuantumCircuit(3)
    reference.h(0)
    reference.cx(0, 2)
    reference.h(1)
    if rotation == "T":
        reference.t(1)
    else:
        reference.rz(np.pi * 0.173, 1)
    reference.cx(1, 2)
    no_loss = reference.copy()
    no_loss.cx(0, 1)
    reference.reset(0)
    reference.append(
        pauli_error([("I", 0.25), ("X", 0.25), ("Y", 0.25), ("Z", 0.25)]).to_instruction(), [1]
    )
    for branch in [reference, no_loss]:
        if rotation == "T":
            branch.t(2)
        else:
            branch.rz(np.pi * 0.173, 2)
        branch.h(1)
        branch.h(2)
        branch.save_probabilities([1, 2])
    oracle = (
        AerSimulator(method="density_matrix", max_parallel_threads=1)
        .run([reference, no_loss])
        .result()
    )
    probabilities = (
        loss_probability * oracle.data(0)["probabilities"]
        + (1 - loss_probability) * oracle.data(1)["probabilities"]
    )
    result = fixed_sample(prepare(text, model))
    actual = np.bincount(result.measurements @ np.array([1, 2]), minlength=4) / len(
        result.measurements
    )
    np.testing.assert_allclose(actual, probabilities, atol=0.02)


def test_fixed_clifford_status_conditioned_channel_matches_stim(fixed_sample):
    model = noncomp.Model(
        classifier=noncomp.Classifier(CLASSIFIER), gate_partner_effects={"leaked": DEPOLARIZE}
    )
    text = "H 1\nCX 1 2\nLEAKAGE(1) 0\nCX 0 1\nH 1 2\nM 1 2"
    # The physical gate drops; DEPOLARIZE1(0.75) is a uniform I/X/Y/Z channel.
    reference = stim.Circuit("H 1\nCX 1 2\nDEPOLARIZE1(0.75) 1\nH 1 2\nM 1 2")
    expected = reference.compile_sampler(seed=73).sample(16000)
    actual = fixed_sample(prepare(text, model)).measurements
    assert (
        dense.tvd(dense.empirical_record_probs(actual), dense.empirical_record_probs(expected))
        < 0.025
    )


def test_fixed_ternary_heralds_and_feedback_are_jointly_preserved(fixed_sample):
    model = noncomp.Model(
        classifier=noncomp.Classifier(
            [[1, 0, 0.2, 0.2, 0.2], [0, 1, 0.3, 0.3, 0.3], [0, 0, 0.5, 0.5, 0.5]]
        )
    )
    text = "LEAKAGE(1) 0\nM 0\nCX rec[-1] 1\nM 1\nDETECTOR rec[-1] rec[-2]"
    result = fixed_sample(prepare(text, model))
    np.testing.assert_array_equal(result.measurements[:, 0], result.measurements[:, 1])
    assert not result.detectors.any()
    counts = np.bincount(result.measurements[:, 0] + 2 * result.heralds[:, 0], minlength=4)
    np.testing.assert_allclose(counts / len(result.measurements), [0.2, 0.3, 0.25, 0.25], atol=0.02)


def test_fixed_unreachable_effects_and_erased_labels_do_not_block_preparation(fixed_sample):
    model = noncomp.Model(classifier=noncomp.Classifier([[1, 0, 1, 0, 0.5], [0, 1, 0, 1, 0.5]]))
    program = prepare("LEAKAGE(0) 0\nCX 0 1\nLEAKAGE(1) 0\nR 0\nM 0 1", model, final_status=True)
    result = fixed_sample(program, shots=17)
    assert not result.measurements.any()
    assert not result.final_status.any()


def test_fixed_rejections_leave_public_trajectory_sampling_available():
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    text = "LOSS(0.2) 0\nCX 0 1\nM 1"
    assert (
        _clifft_core._prepare_noncomputational_fixed(_clifft_core.parse(text), model._handle)
        is None
    )
    assert noncomp.sample(text, model, shots=16, seed=2).measurements.shape == (16, 1)


@pytest.mark.parametrize(("qubits", "cap"), [(5, 4), (64, 4), (64, None)])
def test_fixed_width_rejection_leaves_optimized_trajectories_available(
    qubits, cap, noncomp_sampling_api
):
    # Each raw pair promotes a coordinate; trajectory fusion combines it into
    # a Clifford. Include measurements so successful fallback checks a result.
    text = "\n".join(f"H {q}\nT {q}\nT {q}\nS_DAG {q}\nH {q}" for q in range(qubits))
    text += "\nM " + " ".join(map(str, range(qubits)))
    model = noncomp.Model()
    assert (
        _clifft_core._prepare_noncomputational_fixed(
            _clifft_core.parse(text), model._handle, max_active_width=cap
        )
        is None
    )
    result = noncomp_sampling_api(text, model, shots=32, seed=5, max_active_width=cap)
    np.testing.assert_array_equal(result.measurements, np.zeros((32, qubits), dtype=np.uint8))


@pytest.mark.parametrize("gate", ["SPP", "SPP_DAG", "TPP", "TPP_DAG", "R_PAULI(0.125)"])
def test_fixed_pauli_product_rejection_preserves_trajectory_result(gate, noncomp_sampling_api):
    text = f"LOSS(1) 0\n{gate} X0*X1\nM 1"
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    assert (
        _clifft_core._prepare_noncomputational_fixed(_clifft_core.parse(text), model._handle)
        is None
    )
    result = noncomp_sampling_api(text, model, shots=32, seed=8)
    assert not result.measurements.any()


@pytest.mark.parametrize(
    ("transition", "interaction"),
    [("LEAKAGE", "LEAKAGE_INTERACTION(0.1,0.2,0.3,0)"), ("LOSS", "LOSS_INTERACTION(0.1,0.2,0.3)")],
)
@pytest.mark.parametrize("probability", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("measurement", ["M", "MX"])
def test_fixed_explicit_partner_channel_matches_stim(
    transition, interaction, probability, measurement, fixed_sample
):
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    prefix = "H 1\nCX 1 2\n"
    suffix = f"{measurement} 1 2"
    text = f"{prefix}{transition}({probability}) 0\n{interaction} 0 1\n{suffix}"
    # q0 starts unentangled, so the status-conditioned channel has these
    # unconditional Pauli weights on the live half of the Bell pair.
    reference = stim.Circuit(
        f"{prefix}PAULI_CHANNEL_1("
        f"{0.1 * probability},{0.2 * probability},{0.3 * probability}) 1\n{suffix}"
    )
    actual = fixed_sample(prepare(text, model)).measurements
    expected = reference.compile_sampler(seed=13).sample(16000)
    assert (
        dense.tvd(dense.empirical_record_probs(actual), dense.empirical_record_probs(expected))
        < 0.025
    )


@pytest.mark.parametrize(
    ("transition", "interaction"),
    [("LEAKAGE", "LOSS_INTERACTION(1,0,0)"), ("LOSS", "LEAKAGE_INTERACTION(1,0,0,0)")],
)
def test_fixed_explicit_interaction_requires_matching_source_status(
    transition, interaction, fixed_sample
):
    text = f"{transition}(1) 0\n{interaction} 0 1\nM 1"
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    assert not fixed_sample(prepare(text, model), shots=32).measurements.any()


@pytest.mark.parametrize("status", ["leaked", "lost"])
@pytest.mark.parametrize("source_operand", [0, 1])
def test_fixed_gate_specific_partner_rule_matches_stim(status, source_operand, fixed_sample):
    model = noncomp.Model(
        classifier=noncomp.Classifier(CLASSIFIER),
        interactions=[
            noncomp.InteractionRule(
                gate="CZ", source_operand=source_operand, source_status=status, effect=DEPOLARIZE
            )
        ],
    )
    transition = "LEAKAGE" if status == "leaked" else "LOSS"
    operands = "0 1" if source_operand == 0 else "1 0"
    text = f"H 1\nCX 1 2\n{transition}(1) 0\nCZ {operands}\nMX 1 2"
    actual = fixed_sample(prepare(text, model)).measurements
    expected = (
        stim.Circuit("H 1\nCX 1 2\nDEPOLARIZE1(0.75) 1\nMX 1 2")
        .compile_sampler(seed=13)
        .sample(16000)
    )
    assert (
        dense.tvd(dense.empirical_record_probs(actual), dense.empirical_record_probs(expected))
        < 0.025
    )
    # The override must not make other physical gates eligible.
    assert (
        _clifft_core._prepare_noncomputational_fixed(
            _clifft_core.parse(text.replace("CZ", "CX")), model._handle
        )
        is None
    )


@pytest.mark.parametrize("dynamic", [False, True], ids=["static-inputs", "dynamic-inputs"])
def test_fixed_python_sampling_is_reproducible_across_worker_counts(dynamic, fixed_sample):
    model = noncomp.Model(
        classifier=noncomp.Classifier(
            [[1, 0, 0.2, 0.2, 0.2], [0, 1, 0.3, 0.3, 0.3], [0, 0, 0.5, 0.5, 0.5]]
        ),
        gate_partner_effects={"leaked": DEPOLARIZE},
    )
    probability = 0.4 if dynamic else 0
    text = (
        f"H 0\nT 0\nCX 0 1\nLEAKAGE({probability}) 0\nCX 0 1\nM 0\n"
        "HERALD_LEAKAGE_EVENT 0\nCX rec[-2] 2\nR 0\nM 1 2\n"
        "DETECTOR rec[-1] rec[-4]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    program = prepare(text, model, final_status=True)
    before = program.metrics
    assert (before["classical_workspace_bytes"] > 0) == dynamic
    expected = fixed_sample(program, shots=1025, seed=12)
    for workers in (2, 3):
        actual = fixed_sample(program, shots=1025, seed=12, threads=workers)
        assert actual.stats["workers"] == workers
        for field in ("measurements", "heralds", "detectors", "observables", "final_status"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert program.metrics == before


def test_fixed_later_readout_noise_consumes_reported_record(fixed_sample):
    model = noncomp.Model(classifier=noncomp.Classifier([[1, 0, 0, 0, 0], [0, 1, 1, 1, 1]]))
    text = "LEAKAGE(1) 0\nMR 0\nREADOUT_NOISE(1,0) rec[-1]\nCX rec[-1] 1\nM 0 1"
    result = fixed_sample(prepare(text, model, final_status=True), shots=32)
    np.testing.assert_array_equal(result.measurements, np.tile([1, 0, 1], (32, 1)))


@pytest.mark.parametrize("dynamic", [False, True], ids=["static-inputs", "dynamic-inputs"])
def test_fixed_channels_on_one_record_preserve_order_and_detector_snapshots(dynamic, fixed_sample):
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    text = (
        f"LEAKAGE({int(dynamic)}) 2\nM 0\nREADOUT_NOISE(1,0) rec[-1]\n"
        "DETECTOR rec[-1]\nREADOUT_NOISE(0,1) rec[-1]\nCX rec[-1] 1\nM 1\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]\nR 2"
    )
    result = fixed_sample(prepare(text, model, final_status=True), shots=32)
    assert not result.measurements.any()
    assert result.detectors.all()
    assert not result.observables.any()
    assert not result.final_status.any()


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("probability", [0, 1])
def test_fixed_noise_groups_bind_empty_and_live_sites_separately(width, probability, fixed_sample):
    probabilities = [str(probability)] + ["0"] * (4**width - 2)
    targets = " ".join(map(str, range(2 * width)))
    text = (
        f"LEAKAGE(1) 0\nPAULI_CHANNEL_{width}({','.join(probabilities)}) {targets}\n"
        f"M {width - 1} {2 * width - 1}"
    )
    model = noncomp.Model(classifier=noncomp.Classifier([[1, 0, 1, 1, 1], [0, 1, 0, 0, 0]]))
    result = fixed_sample(prepare(text, model), shots=32)
    # The first outcome is X on the last operand; only the second group is live.
    np.testing.assert_array_equal(result.measurements, np.tile([0, probability], (32, 1)))


@pytest.mark.parametrize(
    ("noise", "physical_noise"),
    [
        ("X_ERROR(0.3) 1", "X_ERROR(0.3) 1"),
        ("DEPOLARIZE1(0.4) 1", "DEPOLARIZE1(0.4) 1"),
        ("PAULI_CHANNEL_1(0.1,0.2,0.3) 1", "PAULI_CHANNEL_1(0.1,0.2,0.3) 1"),
        ("DEPOLARIZE2(0.7) 0 1", ""),
        ("DEPOLARIZE3(0.7) 0 1 2", ""),
        ("CORRELATED_ERROR(0.3) X0\nELSE_CORRELATED_ERROR(0.4) X1", "X_ERROR(0.28) 1"),
        (
            "CORRELATED_ERROR(0.3) X0 X1\nELSE_CORRELATED_ERROR(0.4) Z0 X2",
            "CORRELATED_ERROR(0.3) X1\nELSE_CORRELATED_ERROR(0.4) X2",
        ),
    ],
)
def test_fixed_noise_masking_and_correlated_branch_selection(noise, physical_noise, fixed_sample):
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    text = f"LEAKAGE(1) 0\n{noise}\nCX 1 2\nM 1 2"
    result = fixed_sample(prepare(text, model))
    reference = stim.Circuit(f"{physical_noise}\nCX 1 2\nM 1 2")
    expected = reference.compile_sampler(seed=13).sample(16000)
    assert (
        dense.tvd(
            dense.empirical_record_probs(result.measurements),
            dense.empirical_record_probs(expected),
        )
        < 0.025
    )


def test_fixed_zero_shots_preserves_the_requested_status_shape(fixed_sample):
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    requested = fixed_sample(prepare("M 0", model, final_status=True), shots=0)
    omitted = fixed_sample(prepare("M 0", model), shots=0)
    assert requested.final_status.shape == (0, 1)
    assert omitted.final_status is None


def test_fixed_static_inputs_keep_asymmetric_readout_and_reset_semantics(fixed_sample):
    model = noncomp.Model(
        classifier=noncomp.Classifier([[0.8, 0.4, 0.5, 0.5, 0.5], [0.2, 0.6, 0.5, 0.5, 0.5]])
    )
    program = prepare("LEAKAGE(0) 0\nX 0\nMR 0\nM 0", model, final_status=True)
    assert program.metrics["classical_workspace_bytes"] == 0
    result = fixed_sample(program)
    np.testing.assert_allclose(result.measurements.mean(axis=0), [0.6, 0.2], atol=0.02)
    assert not result.final_status.any()


@pytest.mark.parametrize(
    ("text", "restore_lost"),
    [
        pytest.param(
            "LEAKAGE(0.4) 0\nM 0\nCX rec[-1] 1\nM 1\nHERALD_LEAKAGE_EVENT 0\nR 0",
            False,
            id="herald-feedback-and-status-probe",
        ),
        pytest.param(
            "H 0\nT 0\nCX 0 2\nLOSS(0.35) 0\nCX 0 1\nMX 1\nM 2\nHERALD_LOSS_EVENT 0\nR 0",
            False,
            id="non-clifford-partner-and-persistent-loss",
        ),
        pytest.param(
            "H 0\nLOSS(0.4) 0\nMRX !0\nREADOUT_NOISE(0.25,0.1) rec[-1]\nCX rec[-1] 1\nM 0 1",
            True,
            id="inverted-reset-readout-and-feedback",
        ),
        pytest.param(
            "LEAKAGE(0.4) 0\nCORRELATED_ERROR(0.3) X0\nELSE_CORRELATED_ERROR(0.4) X1\n"
            "DEPOLARIZE2(0.7) 0 1\nM 1\nHERALD_LEAKAGE_EVENT 0\nR 0",
            False,
            id="correlated-branch-and-whole-operation-noise",
        ),
        pytest.param(
            "LOSS(0.4) 3\nDEPOLARIZE3(1) 0 1 2\nM 0 1 2\nHERALD_LOSS_EVENT 3",
            False,
            id="certain-three-qubit-noise",
        ),
        pytest.param(
            "LEAKAGE(0.4) 0\nM 0\nM 0\nMR 0",
            False,
            id="repeated-classified-readouts-and-restoration",
        ),
        pytest.param(
            "H 0\nLEAKAGE(0.4) 0\nSWAP 0 1\nM 0 1\nR 0 1",
            False,
            id="swap-with-an-inactive-source",
        ),
        pytest.param(
            "LOSS(0.4) 0\nLOSS_INTERACTION(0.1,0.2,0.3) 0 1\nLEAKAGE(0.3) 2\n"
            "LEAKAGE_INTERACTION(0.2,0.1,0.1,0) 2 1\nM 1\nHERALD_LOSS_EVENT 0\nR 2",
            False,
            id="explicit-partial-partner-effects",
        ),
        pytest.param(
            "H 0\nLEAKAGE(0.3) 0\nCX 0 1 0 2\nM 1 2\nR 0",
            False,
            id="reused-source-across-physical-pairs",
        ),
        pytest.param(
            "LEAKAGE(0) 0\nM 0\nREADOUT_NOISE(0.2,0.1) rec[-1]\nDETECTOR rec[-1]\n"
            "READOUT_NOISE(0.4,0.3) rec[-1]\nCX rec[-1] 1\nM 1",
            False,
            id="static-ordered-readout-channels",
        ),
    ],
)
def test_fixed_joint_output_contract_matches_trajectories(
    text, restore_lost, noncomp_sampling_api, fixed_sample
):
    model = noncomp.Model(
        classifier=noncomp.Classifier(
            [[1, 0, 0.2, 0.2, 0.2], [0, 1, 0.3, 0.3, 0.3], [0, 0, 0.5, 0.5, 0.5]]
        ),
        reset_restores_lost=restore_lost,
        gate_partner_effects={"leaked": DEPOLARIZE, "lost": DEPOLARIZE},
    )
    text += "\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    program = prepare(text, model, final_status=True)
    actual = fixed_sample(program)
    expected = noncomp_sampling_api(text, model, shots=16000, seed=71)

    def joint(result):
        return dense.empirical_record_probs(
            np.concatenate(
                [
                    result.measurements,
                    result.heralds,
                    result.detectors,
                    result.observables,
                    result.final_status,
                ],
                axis=1,
            )
        )

    # Compare the full joint law; matching marginals can hide broken feedback
    # or correlations between the reported record and the status sidecars.
    assert dense.tvd(joint(actual), joint(expected)) < 0.04


@pytest.mark.parametrize("rounds", [2, 40], ids=["ordinary-propagation", "optimized-carriers"])
def test_packed_fixed_composes_status_noise_readout_and_non_clifford_execution(rounds):
    model = noncomp.Model(
        classifier=noncomp.Classifier(
            [[1, 0, 0.2, 0.2, 0.2], [0, 1, 0.3, 0.3, 0.3], [0, 0, 0.5, 0.5, 0.5]]
        )
    )
    text = (
        f"REPEAT {rounds} {{\nR 0 1\nDEPOLARIZE1(0.3) 3\nM 3\nCX rec[-1] 1\n"
        "DEPOLARIZE1(0.3) 0\nLEAKAGE(0.4) 0\n"
        "M 0\nCX rec[-1] 1\nM 1\nHERALD_LEAKAGE_EVENT 0\nMR 0\nM 0\n}\n"
        "CX rec[-4] 2\nH 2\nT 2\nH 2\nM 2\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    program = prepare(text, model, final_status=True)
    before = program.metrics
    assert (before["packed_noise_carriers"] > 0) == (rounds == 40)
    assert before["peak_active_width"] > 0
    shots = 2051
    scalar = sample(program, shots=shots, seed=52, batch_size=1)
    for capacity in (64, 65, 129, "auto"):
        result = sample(program, shots=shots, seed=52, threads=3, batch_size=capacity)
        assert result.stats["packed_shots"] == shots
        # Until the final T experiment, all quantum outcomes are fixed by the
        # supplied Pauli inputs. The shared per-shot input stream must therefore
        # reproduce these records and sidecars exactly across batch boundaries.
        np.testing.assert_array_equal(result.measurements[:, :-1], scalar.measurements[:, :-1])
        np.testing.assert_array_equal(result.heralds, scalar.heralds)
        np.testing.assert_array_equal(result.final_status, scalar.final_status)
        records = result.measurements[:, :-1].reshape(shots, rounds, 6)
        np.testing.assert_array_equal(records[:, :, 0] ^ records[:, :, 1], records[:, :, 2])
        assert not records[:, :, 5].any()  # The reset follows the classified MR result.
        assert not result.final_status.any()
        leaked = records[:, :, 3].astype(bool)
        reported_one = records[:, :, 1].astype(bool)
        # Live X/Y errors flip with probability 0.2. A leaked readout reports 1
        # with probability 0.3 + 0.5/2, including the herald placeholder.
        assert abs((reported_one & ~leaked).mean() - 0.6 * 0.2) < 0.025
        assert abs((reported_one & leaked).mean() - 0.4 * 0.55) < 0.025
        flips = result.measurements[:, -1] != records[:, -1, 2]
        assert abs(flips.mean() - np.sin(np.pi / 8) ** 2) < 0.03
        np.testing.assert_array_equal(result.detectors[:, 0], result.measurements[:, -1])
        np.testing.assert_array_equal(result.observables[:, 0], result.measurements[:, -1])
    assert program.metrics == before
    assert before["preparation_compilations"] == 1


@pytest.mark.parametrize("dynamic", [False, True], ids=["static-inputs", "dynamic-inputs"])
@pytest.mark.parametrize("shots", [0, 1, 63, 64, 2049])
def test_fixed_automatic_batch_policy_reports_executed_work(dynamic, shots):
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    program = prepare(f"LOSS({0.4 if dynamic else 0}) 0\nM 0", model, final_status=True)
    expected_capacity = 1 if shots < 64 else min(shots, 2048)
    serial = sample(program, shots=shots, seed=14)
    parallel = sample(program, shots=shots, seed=14, threads=3)
    assert serial.stats["batch_size"] == parallel.stats["batch_size"] == expected_capacity
    assert parallel.stats["workers"] == min(3, (shots + expected_capacity - 1) // expected_capacity)
    assert parallel.stats["packed_shots"] == (shots if expected_capacity > 1 else 0)
    for field in ("measurements", "heralds", "detectors", "observables", "final_status"):
        np.testing.assert_array_equal(getattr(serial, field), getattr(parallel, field))
    if shots:
        assert (parallel.stats["packed_worker_bytes"] > 0) == (expected_capacity > 1)


def test_fixed_automatic_batching_falls_back_for_wider_non_clifford_states():
    text = "H 0 1 2 3 4 5\nT 0 1 2 3 4 5\nLOSS(0.3) 6\nM 0 1 2 3 4 5 6"
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    program = prepare(text, model, final_status=True)
    assert program.metrics["peak_active_width"] == 6
    result = sample(program, shots=129, threads=3)
    assert result.stats["scalar_shots"] == 129
    assert result.stats["batch_size"] == 1
    assert result.stats["workers"] == 3
    forced = sample(program, shots=129, threads=3, batch_size=65)
    assert forced.stats["packed_shots"] == 129
    np.testing.assert_array_equal(result.final_status, forced.final_status)


@pytest.mark.parametrize("batch_size", [0, -1, 2**32, "tune", "invalid"])
@pytest.mark.parametrize("shots", [0, 1])
def test_fixed_rejects_invalid_batch_options_before_sampling(batch_size, shots):
    program = prepare("M 0", noncomp.Model())
    with pytest.raises(ValueError, match="batch_size"):
        program.sample(shots, batch_size=batch_size)


def test_fixed_automatic_batching_counts_retained_external_buffers():
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    program = prepare("LOSS(0.1) 0\nREPEAT 6000 {\nM 0\n}", model)
    result = sample(program, shots=2049, threads=16)
    stats = result.stats
    assert stats["packed_shots"] == 2049
    assert 64 <= stats["batch_size"] < 2048
    assert 0 < stats["packed_worker_bytes"] <= 8 * 1024**2
    assert stats["packed_worker_bytes"] * stats["workers"] <= 64 * 1024**2
