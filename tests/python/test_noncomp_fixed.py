"""Independent distribution checks for the internal scalar fixed path."""

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


def sample(program, shots=16000, seed=43, threads=1):
    values = program.sample(shots, seed, threads)
    return SimpleNamespace(
        measurements=values[0],
        detectors=values[1],
        observables=values[2],
        final_status=values[3],
        heralds=values[4],
    )


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
def test_fixed_joint_distribution_matches_dense_physical_model(text):
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
    result = sample(program)
    empirical = dense.empirical_record_probs(result.measurements)
    assert dense.tvd(expected.record_probs, empirical) < 0.025
    assert program.metrics == before
    assert before["preparation_compilations"] == 1
    assert result.final_status is None


@pytest.mark.parametrize("rotation", ["T", "R_Z(0.173)"])
@pytest.mark.parametrize("loss_probability", [0.0, 0.4, 1.0])
def test_fixed_non_clifford_partner_channel_matches_aer(rotation, loss_probability):
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
    result = sample(prepare(text, model))
    actual = np.bincount(result.measurements @ np.array([1, 2]), minlength=4) / len(
        result.measurements
    )
    np.testing.assert_allclose(actual, probabilities, atol=0.02)


def test_fixed_clifford_status_conditioned_channel_matches_stim():
    model = noncomp.Model(
        classifier=noncomp.Classifier(CLASSIFIER), gate_partner_effects={"leaked": DEPOLARIZE}
    )
    text = "H 1\nCX 1 2\nLEAKAGE(1) 0\nCX 0 1\nH 1 2\nM 1 2"
    # The physical gate drops; DEPOLARIZE1(0.75) is a uniform I/X/Y/Z channel.
    reference = stim.Circuit("H 1\nCX 1 2\nDEPOLARIZE1(0.75) 1\nH 1 2\nM 1 2")
    expected = reference.compile_sampler(seed=73).sample(16000)
    actual = sample(prepare(text, model)).measurements
    assert (
        dense.tvd(dense.empirical_record_probs(actual), dense.empirical_record_probs(expected))
        < 0.025
    )


def test_fixed_ternary_heralds_and_feedback_are_jointly_preserved():
    model = noncomp.Model(
        classifier=noncomp.Classifier(
            [[1, 0, 0.2, 0.2, 0.2], [0, 1, 0.3, 0.3, 0.3], [0, 0, 0.5, 0.5, 0.5]]
        )
    )
    text = "LEAKAGE(1) 0\nM 0\nCX rec[-1] 1\nM 1\nDETECTOR rec[-1] rec[-2]"
    result = sample(prepare(text, model))
    np.testing.assert_array_equal(result.measurements[:, 0], result.measurements[:, 1])
    assert not result.detectors.any()
    counts = np.bincount(result.measurements[:, 0] + 2 * result.heralds[:, 0], minlength=4)
    np.testing.assert_allclose(counts / len(result.measurements), [0.2, 0.3, 0.25, 0.25], atol=0.02)


def test_fixed_unreachable_effects_and_erased_labels_do_not_block_preparation():
    model = noncomp.Model(classifier=noncomp.Classifier([[1, 0, 1, 0, 0.5], [0, 1, 0, 1, 0.5]]))
    program = prepare("LEAKAGE(0) 0\nCX 0 1\nLEAKAGE(1) 0\nR 0\nM 0 1", model, final_status=True)
    result = sample(program, shots=17)
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
    transition, interaction, probability, measurement
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
    actual = sample(prepare(text, model)).measurements
    expected = reference.compile_sampler(seed=13).sample(16000)
    assert (
        dense.tvd(dense.empirical_record_probs(actual), dense.empirical_record_probs(expected))
        < 0.025
    )


@pytest.mark.parametrize(
    ("transition", "interaction"),
    [("LEAKAGE", "LOSS_INTERACTION(1,0,0)"), ("LOSS", "LEAKAGE_INTERACTION(1,0,0,0)")],
)
def test_fixed_explicit_interaction_requires_matching_source_status(transition, interaction):
    text = f"{transition}(1) 0\n{interaction} 0 1\nM 1"
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    assert not sample(prepare(text, model), shots=32).measurements.any()


@pytest.mark.parametrize("status", ["leaked", "lost"])
@pytest.mark.parametrize("source_operand", [0, 1])
def test_fixed_gate_specific_partner_rule_matches_stim(status, source_operand):
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
    actual = sample(prepare(text, model)).measurements
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
def test_fixed_python_sampling_is_reproducible_across_worker_counts(dynamic):
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
    expected = sample(program, shots=1025, seed=12)
    for workers in (2, 3):
        actual = sample(program, shots=1025, seed=12, threads=workers)
        for field in ("measurements", "heralds", "detectors", "observables", "final_status"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert program.metrics == before


def test_fixed_later_readout_noise_consumes_reported_record():
    model = noncomp.Model(classifier=noncomp.Classifier([[1, 0, 0, 0, 0], [0, 1, 1, 1, 1]]))
    text = "LEAKAGE(1) 0\nMR 0\nREADOUT_NOISE(1,0) rec[-1]\nCX rec[-1] 1\nM 0 1"
    result = sample(prepare(text, model, final_status=True), shots=32)
    np.testing.assert_array_equal(result.measurements, np.tile([1, 0, 1], (32, 1)))


@pytest.mark.parametrize("dynamic", [False, True], ids=["static-inputs", "dynamic-inputs"])
def test_fixed_channels_on_one_record_preserve_order_and_detector_snapshots(dynamic):
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    text = (
        f"LEAKAGE({int(dynamic)}) 2\nM 0\nREADOUT_NOISE(1,0) rec[-1]\n"
        "DETECTOR rec[-1]\nREADOUT_NOISE(0,1) rec[-1]\nCX rec[-1] 1\nM 1\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]\nR 2"
    )
    result = sample(prepare(text, model, final_status=True), shots=32)
    assert not result.measurements.any()
    assert result.detectors.all()
    assert not result.observables.any()
    assert not result.final_status.any()


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("probability", [0, 1])
def test_fixed_noise_groups_bind_empty_and_live_sites_separately(width, probability):
    probabilities = [str(probability)] + ["0"] * (4**width - 2)
    targets = " ".join(map(str, range(2 * width)))
    text = (
        f"LEAKAGE(1) 0\nPAULI_CHANNEL_{width}({','.join(probabilities)}) {targets}\n"
        f"M {width - 1} {2 * width - 1}"
    )
    model = noncomp.Model(classifier=noncomp.Classifier([[1, 0, 1, 1, 1], [0, 1, 0, 0, 0]]))
    result = sample(prepare(text, model), shots=32)
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
def test_fixed_noise_masking_and_correlated_branch_selection(noise, physical_noise):
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    text = f"LEAKAGE(1) 0\n{noise}\nCX 1 2\nM 1 2"
    result = sample(prepare(text, model))
    reference = stim.Circuit(f"{physical_noise}\nCX 1 2\nM 1 2")
    expected = reference.compile_sampler(seed=13).sample(16000)
    assert (
        dense.tvd(
            dense.empirical_record_probs(result.measurements),
            dense.empirical_record_probs(expected),
        )
        < 0.025
    )


def test_fixed_zero_shots_preserves_the_requested_status_shape():
    model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER))
    requested = sample(prepare("M 0", model, final_status=True), shots=0)
    omitted = sample(prepare("M 0", model), shots=0)
    assert requested.final_status.shape == (0, 1)
    assert omitted.final_status is None


def test_fixed_static_inputs_keep_asymmetric_readout_and_reset_semantics():
    model = noncomp.Model(
        classifier=noncomp.Classifier([[0.8, 0.4, 0.5, 0.5, 0.5], [0.2, 0.6, 0.5, 0.5, 0.5]])
    )
    program = prepare("LEAKAGE(0) 0\nX 0\nMR 0\nM 0", model, final_status=True)
    assert program.metrics["classical_workspace_bytes"] == 0
    result = sample(program)
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
def test_fixed_joint_output_contract_matches_trajectories(text, restore_lost, noncomp_sampling_api):
    model = noncomp.Model(
        classifier=noncomp.Classifier(
            [[1, 0, 0.2, 0.2, 0.2], [0, 1, 0.3, 0.3, 0.3], [0, 0, 0.5, 0.5, 0.5]]
        ),
        reset_restores_lost=restore_lost,
        gate_partner_effects={"leaked": DEPOLARIZE, "lost": DEPOLARIZE},
    )
    text += "\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    program = prepare(text, model, final_status=True)
    actual = sample(program)
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
