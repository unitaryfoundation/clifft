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
