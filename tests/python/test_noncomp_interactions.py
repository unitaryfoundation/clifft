"""Conditional partner effects and their model-level configuration."""

import numpy as np
import pytest

from clifft import noncomp


def classifier():
    return noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]])


def rule(gate="CX", source=0, status="leaked", effect=None):
    return noncomp.InteractionRule(
        gate=gate,
        source_operand=source,
        source_status=status,
        effect=noncomp.PartnerEffect() if effect is None else effect,
    )


def assert_same_result(a, b):
    for field in ("measurements", "detectors", "observables", "heralds", "final_status"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field), err_msg=field)


@pytest.mark.parametrize("pauli", [(1, 1, 0), (-0.1, 0, 0), (float("nan"), 0, 0), (0, 0)])
def test_partner_pauli_validation(pauli):
    with pytest.raises(ValueError):
        noncomp.PartnerEffect(pauli=pauli)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_partner_spreading_validation(probability):
    with pytest.raises(ValueError, match="spread_probability"):
        noncomp.PartnerEffect(spread_probability=probability)


def test_effect_copies_probabilities_and_model_snapshots_rules(noncomp_sampling_api):
    probabilities = [1, 0, 0]
    effect = noncomp.PartnerEffect(pauli=probabilities)
    probabilities[0] = 0
    assert effect.pauli == (1, 0, 0)
    defaults = {"leaked": effect}
    overrides = [rule()]
    model = noncomp.Model(
        classifier=classifier(), gate_partner_effects=defaults, interactions=overrides
    )
    defaults.clear()
    overrides.clear()
    result = noncomp_sampling_api("LEAKAGE(1) 0\nCX 0 1\nCZ 0 2\nM 1 2", model, shots=32, seed=1)
    np.testing.assert_array_equal(result.measurements, np.tile([0, 1], (32, 1)))
    assert "PartnerEffect" in repr(model)
    assert "InteractionRule" in repr(model)


@pytest.mark.parametrize("gate", ["H", "CH", "CCX", "II", "DEPOLARIZE2", "MXX", "typo"])
def test_rule_rejects_non_native_pair_gates(gate):
    with pytest.raises(ValueError, match="native two-qubit unitary"):
        noncomp.Model(interactions=[rule(gate=gate)])


def test_rule_and_default_validation():
    with pytest.raises(ValueError, match="duplicates"):
        noncomp.Model(interactions=[rule(), rule(gate="CNOT")])
    with pytest.raises(ValueError, match="source_operand"):
        rule(source=2)
    with pytest.raises(ValueError, match="source_status"):
        rule(status="missing")
    with pytest.raises(ValueError, match="source status"):
        noncomp.Model(gate_partner_effects={"missing": noncomp.PartnerEffect()})
    spreading = noncomp.PartnerEffect(spread_probability=0.1)
    with pytest.raises(ValueError, match="leaked source"):
        rule(status="lost", effect=spreading)
    with pytest.raises(ValueError, match="leaked source"):
        noncomp.Model(gate_partner_effects={"lost": spreading})


def test_model_defaults_and_overrides_match_explicit_expansion(noncomp_sampling_api):
    effect = noncomp.PartnerEffect(pauli=(0.1, 0.2, 0.3), spread_probability=0.25)
    lost_effect = noncomp.PartnerEffect(pauli=(0.2, 0.1, 0.1))
    model = noncomp.Model(
        classifier=classifier(),
        gate_partner_effects={"leaked": effect, "lost": lost_effect},
        interactions=[rule(gate="CNOT"), rule(source=1, status="lost")],
    )
    prefix = "H 0\nLEAKAGE(0.5) 0\nLOSS(0.3) 1\n"
    suffix = "HERALD_LEAKAGE_EVENT 0 1 2\nM 0 1 2\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"
    source = prefix + "CX 0 1\nR_ZZ(0.17) 1 2\n" + suffix
    expanded = (
        prefix
        + """
        CX 0 1
        LOSS_INTERACTION(0.2, 0.1, 0.1) 0 1
        LEAKAGE_INTERACTION(0.1, 0.2, 0.3, 0.25) 1 0
        R_ZZ(0.17) 1 2
        LEAKAGE_INTERACTION(0.1, 0.2, 0.3, 0.25) 1 2
        LOSS_INTERACTION(0.2, 0.1, 0.1) 1 2
        LEAKAGE_INTERACTION(0.1, 0.2, 0.3, 0.25) 2 1
        LOSS_INTERACTION(0.2, 0.1, 0.1) 2 1
    """
        + suffix
    )
    a = noncomp_sampling_api(source, model, shots=128, seed=2)
    b = noncomp_sampling_api(expanded, noncomp.Model(classifier=classifier()), shots=128, seed=2)
    assert_same_result(a, b)


def test_empty_effects_preserve_default_seeded_results(noncomp_sampling_api):
    source = "H 0\nCX 0 1\nLEAKAGE(0.3) 0\nCX 0 1\nLOSS(0.2) 1\nM 0 1"
    plain = noncomp.Model(classifier=classifier())
    empty = noncomp.Model(
        classifier=classifier(),
        gate_partner_effects={"leaked": noncomp.PartnerEffect(), "lost": noncomp.PartnerEffect()},
        interactions=[rule()],
    )
    assert_same_result(
        noncomp_sampling_api(source, plain, shots=128, seed=3),
        noncomp_sampling_api(source, empty, shots=128, seed=3),
    )


def test_generated_effects_precede_hooks_and_explicit_effects_follow_them(noncomp_sampling_api):
    transition = np.zeros((5, 5))
    transition[noncomp.Level.LEAK_G, noncomp.Level.G] = 1
    effect = noncomp.PartnerEffect(pauli=(1, 0, 0))
    generated = noncomp.Model(
        transitions={"CX": transition.tolist()},
        classifier=classifier(),
        gate_partner_effects={"leaked": effect},
    )
    hooked = noncomp.Model(transitions={"CX": transition.tolist()}, classifier=classifier())
    named = noncomp.Model(transitions={"after_cx": transition.tolist()}, classifier=classifier())
    prefix = "X 1\nCX 0 1\n"
    interaction = "LEAKAGE_INTERACTION(1, 0, 0, 0) 0 1\n"
    a = noncomp_sampling_api(prefix + "M 1", generated, shots=32, seed=4)
    b = noncomp_sampling_api(prefix + interaction + "M 1", hooked, shots=32, seed=4)
    c = noncomp_sampling_api(
        prefix + interaction + "LEVEL_TRANSITION[after_cx] 0 1\nM 1", named, shots=32, seed=4
    )
    d = noncomp_sampling_api(
        prefix + "LEVEL_TRANSITION[after_cx] 0 1\n" + interaction + "M 1", named, shots=32, seed=4
    )
    assert a.measurements.all()
    assert not b.measurements.any()
    assert_same_result(a, c)
    assert_same_result(b, d)


def test_explicit_and_generated_effects_compose_sequentially(noncomp_sampling_api):
    model = noncomp.Model(
        classifier=classifier(),
        gate_partner_effects={
            "leaked": noncomp.PartnerEffect(pauli=(0.2, 0, 0), spread_probability=0.1)
        },
    )
    result = noncomp_sampling_api(
        "LEAKAGE(1) 0\nCX 0 1\nLEAKAGE_INTERACTION(0.3, 0, 0, 0.2) 0 1\nM 1",
        model,
        shots=6000,
        seed=5,
    )
    status = result.final_status[:, 1]
    bit = result.measurements[:, 0]
    observed = np.array(
        [
            ((status == noncomp.QubitStatus.COMPUTATIONAL) & (bit == 0)).mean(),
            ((status == noncomp.QubitStatus.COMPUTATIONAL) & (bit == 1)).mean(),
            (status == noncomp.QubitStatus.LEAK_G).mean(),
            (status == noncomp.QubitStatus.LEAK_E).mean(),
        ]
    )
    # The second channel acts only on the 90% that survive the first leakage
    # attempt. Its net X rate with the first Pauli channel is 0.2 + 0.3 - 2*0.2*0.3.
    flip = 0.2 + 0.3 - 2 * 0.2 * 0.3
    expected = [
        0.9 * 0.8 * (1 - flip),
        0.9 * 0.8 * flip,
        0.1 * 0.8 + 0.9 * 0.2 * (1 - flip),
        0.1 * 0.2 + 0.9 * 0.2 * flip,
    ]
    np.testing.assert_allclose(observed, expected, atol=0.025, rtol=0)


@pytest.mark.parametrize("basis", ["M", "MX", "MY"])
def test_partner_pauli_channel_matches_stim_on_an_entangled_pair(basis, noncomp_sampling_api):
    import stim

    prefix = "H 1\nCX 1 2\n"
    suffix = f"{basis} 1 2"
    model = noncomp.Model(
        classifier=classifier(),
        gate_partner_effects={"leaked": noncomp.PartnerEffect(pauli=(0.1, 0.2, 0.3))},
    )
    result = noncomp_sampling_api(
        prefix + "LEAKAGE(1) 0\nCZ 0 1\n" + suffix, model, shots=6000, seed=6
    )
    reference = stim.Circuit(prefix + "PAULI_CHANNEL_1(0.1, 0.2, 0.3) 1\n" + suffix)
    expected = reference.compile_sampler(seed=7).sample(6000).astype(np.uint8)
    observed_hist = np.bincount(result.measurements @ [2, 1], minlength=4) / 6000
    expected_hist = np.bincount(expected @ [2, 1], minlength=4) / 6000
    np.testing.assert_allclose(observed_hist, expected_hist, atol=0.035, rtol=0)


def test_spreading_on_a_near_clifford_entangled_partner_matches_dense_oracle(noncomp_sampling_api):
    import utils_noncomp_oracle as oracle

    weights = {"I": 0.4, "X": 0.1, "Y": 0.2, "Z": 0.3}
    spread = 0.4
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    # The oracle's first tensor factor is the most significant qubit; Aer
    # numbers that factor as qubit 1.
    preparation = QuantumCircuit(2)
    preparation.h(1)
    preparation.t(1)
    preparation.cx(1, 0)
    preparation.h(1)
    preparation.save_statevector()
    reference = AerSimulator(method="statevector", max_parallel_threads=1)
    psi = np.asarray(reference.run(preparation).result().get_statevector())
    expected = np.zeros(12)
    for pauli, weight in weights.items():
        state = psi if pauli == "I" else oracle.apply_1q(psi, pauli, 0, 2)
        live = oracle.apply_1q(state, "H", 0, 2)
        live = oracle.apply_1q(oracle.apply_1q(live, "S", 1, 2), "H", 1, 2)
        expected[:4] += weight * (1 - spread) * np.abs(live) ** 2
        for source in (0, 1):
            born, collapsed = oracle.collapse(state, 0, source, 2)
            leaked = oracle.apply_1q(oracle.apply_1q(collapsed, "S", 1, 2), "H", 1, 2)
            offset = 4 * (1 + source)
            expected[offset : offset + 4] += weight * spread * born * np.abs(leaked) ** 2
    assert abs(expected.sum() - 1) < 1e-12
    model = noncomp.Model(
        classifier=classifier(),
        gate_partner_effects={
            "leaked": noncomp.PartnerEffect(pauli=(0.1, 0.2, 0.3), spread_probability=spread)
        },
    )
    result = noncomp_sampling_api(
        "H 1\nT 1\nCX 1 2\nH 1\nLEAKAGE(1) 0\nCX 0 1\nH 1\nS 2\nH 2\n"
        "M 1 2\nHERALD_LEAKAGE_EVENT 1",
        model,
        shots=6000,
        seed=8,
    )
    observed = (
        np.bincount(
            4 * result.final_status[:, 1] + result.measurements[:, :2] @ [2, 1], minlength=12
        )
        / 6000
    )
    np.testing.assert_allclose(observed, expected, atol=0.025, rtol=0)
    np.testing.assert_array_equal(
        result.measurements[:, 2], result.final_status[:, 1] != noncomp.QubitStatus.COMPUTATIONAL
    )


@pytest.mark.parametrize(
    "noise, expected",
    [
        ("DEPOLARIZE2(1) 0 1", [0, 0]),
        ("DEPOLARIZE3(1) 0 1 2", [0, 0]),
        ("PAULI_CHANNEL_2(" + ",".join(map(str, [0] * 4 + [1] + [0] * 10)) + ") 0 1", [0, 0]),
        ("PAULI_CHANNEL_3(" + ",".join(map(str, [0] * 20 + [1] + [0] * 42)) + ") 0 1 2", [0, 0]),
        ("E(1) X0 X1\nELSE_CORRELATED_ERROR(1) X2", [1, 0]),
        ("E(1) X0\nELSE_CORRELATED_ERROR(1) X1 X2", [0, 0]),
        ("X_ERROR(1) 0\nX_ERROR(1) 1", [1, 0]),
    ],
)
@pytest.mark.parametrize("source", ["LEAKAGE(1)", "LOSS(1)"])
def test_partner_configuration_preserves_existing_noise_policy(
    noise, expected, source, noncomp_sampling_api
):
    effect = noncomp.PartnerEffect(pauli=(1, 0, 0))
    model = noncomp.Model(
        classifier=classifier(), gate_partner_effects={"leaked": effect, "lost": effect}
    )
    result = noncomp_sampling_api(f"{source} 0\n{noise}\nM 1 2", model, shots=32, seed=9)
    np.testing.assert_array_equal(result.measurements, np.tile(expected, (32, 1)))


def test_partner_effects_exclude_record_controlled_feedback(noncomp_sampling_api):
    model = noncomp.Model(
        classifier=classifier(),
        gate_partner_effects={"leaked": noncomp.PartnerEffect(spread_probability=1)},
    )
    result = noncomp_sampling_api(
        "H 2\nLEAKAGE(1) 0\nHERALD_LEAKAGE_EVENT 0\nCX rec[-1] 1\nCZ rec[-1] 2\nM 1\nMX 2",
        model,
        shots=32,
        seed=10,
    )
    assert result.measurements.all()
    assert (result.final_status[:, 1:] == noncomp.QubitStatus.COMPUTATIONAL).all()


def test_multi_pair_interactions_propagate_in_order_and_restore_at_named_transitions(
    noncomp_sampling_api,
):
    transition = np.zeros((5, 5))
    transition[noncomp.Level.E, noncomp.Level.LEAK_G] = 1
    model = noncomp.Model(
        classifier=classifier(),
        transitions={"recover": transition.tolist()},
        gate_partner_effects={"leaked": noncomp.PartnerEffect(spread_probability=1)},
    )
    result = noncomp_sampling_api(
        "LEAKAGE(1) 0\nCX 0 1 1 2\nHERALD_LEAKAGE_EVENT 0 1 2\n"
        "LEVEL_TRANSITION[recover] 1\nCX 1 3\nM 1 3",
        model,
        shots=32,
        seed=11,
    )
    assert result.measurements.all()
    assert (result.final_status[:, 1] == noncomp.QubitStatus.COMPUTATIONAL).all()
    assert (result.final_status[:, 2] == noncomp.QubitStatus.LEAK_G).all()
    assert (result.final_status[:, 3] == noncomp.QubitStatus.COMPUTATIONAL).all()


def test_interaction_continuations_preserve_seeded_rows_across_worker_counts():
    model = noncomp.Model(
        classifier=classifier(),
        reset_restores_lost=True,
        gate_partner_effects={
            "leaked": noncomp.PartnerEffect(pauli=(0.1, 0.2, 0.1), spread_probability=0.3),
            "lost": noncomp.PartnerEffect(pauli=(0.2, 0.1, 0.2)),
        },
    )
    source = """
        REPEAT 3 {
            H 1
            T 1
            LEAKAGE(0.2) 0
            CX 0 1 1 2
            HERALD_LEAKAGE_EVENT 0 1 2
            R 1
            LOSS(0.1) 2
            CX 2 1
            R 2
        }
        M 0 1 2
    """
    assert_same_result(
        noncomp.sample(source, model, shots=256, seed=12, threads=1),
        noncomp.sample(source, model, shots=256, seed=12, threads=2),
    )


def test_inactive_and_zero_annotations_preserve_seeded_results(noncomp_sampling_api):
    circuit = "H 0\nT 0\nCX 0 1\nM 0 1"
    default = noncomp.Model()
    configured = noncomp.Model(
        gate_partner_effects={
            "leaked": noncomp.PartnerEffect(pauli=(0.25, 0.25, 0.25), spread_probability=1)
        },
    )
    inactive = circuit.replace("M 0 1", "LEAKAGE_INTERACTION(1, 0, 0, 1) 0 1\nM 0 1")
    baseline = noncomp_sampling_api(circuit, default, shots=128, seed=13)
    assert_same_result(baseline, noncomp_sampling_api(circuit, configured, shots=128, seed=13))
    assert_same_result(baseline, noncomp_sampling_api(inactive, default, shots=128, seed=13))

    circuit = "LEAKAGE(0.5) 0\nH 1\nM 1"
    model = noncomp.Model(classifier=classifier())
    explicit = circuit.replace(
        "H 1", "LEAKAGE_INTERACTION(0, 0, 0, 0) 0 1\nLOSS_INTERACTION(0, 0, 0) 0 1\nH 1"
    )
    assert_same_result(
        noncomp_sampling_api(circuit, model, shots=128, seed=14),
        noncomp_sampling_api(explicit, model, shots=128, seed=14),
    )
