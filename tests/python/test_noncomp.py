"""Python-facing tests for noncomputational leakage and loss.

Covers model construction, end-to-end sampling, classifier/record semantics,
and record-layout invariants. The model uses the built-in five-level set;
matrices use the fixed clifft.noncomp.Level order.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import noncomp_classifier_matrix_with_column, noncomp_transition_matrix

import clifft
from clifft import noncomp

LEAK_G, LEAK_E, LOST = noncomp.Level.LEAK_G, noncomp.Level.LEAK_E, noncomp.Level.LOST
ALL_G = [1.0, 0.0, 0.0, 0.0, 0.0]
ALL_E = [0.0, 1.0, 0.0, 0.0, 0.0]
COMPUTATIONAL = noncomp.QubitStatus.COMPUTATIONAL
LOST_KIND = noncomp.QubitStatus.LOST


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def transition_to(level: int) -> list[list[float]]:
    """T[to][from]: g and e both jump to `level` (source-independent)."""
    return noncomp_transition_matrix({(level, noncomp.Level.G): 1.0, (level, noncomp.Level.E): 1.0})


def classifier_for(level: int, col: list[float]) -> noncomp.Classifier:
    """Classifier whose selected column is `col`; computational readout is faithful."""
    return noncomp.Classifier(noncomp_classifier_matrix_with_column(level, col))


def leak_model(classifier: noncomp.Classifier | None = None) -> noncomp.Model:
    return noncomp.Model(
        initial_state=ALL_G, transitions={"S": transition_to(LEAK_G)}, classifier=classifier
    )


def test_level_names_and_indices():
    assert (int(noncomp.Level.G), int(noncomp.Level.LEAK_G), int(noncomp.Level.LOST)) == (0, 2, 4)


def test_initial_state_wrong_sum_raises():
    with pytest.raises(ValueError):
        noncomp.Model(initial_state=[0.5, 0.0, 0.0, 0.0, 0.0])


def test_initial_state_wrong_length_raises():
    with pytest.raises(ValueError):
        noncomp.Model(initial_state=[1.0, 0.0])


def test_initial_state_out_of_range_probability_raises():
    with pytest.raises(ValueError):
        noncomp.Model(initial_state=[2.0, -1.0, 0.0, 0.0, 0.0])


def test_non_gate_key_is_a_named_transition(noncomp_sampling_api):
    """A key that names no gate is a named transition: referenceable from a
    LEVEL_TRANSITION[key] annotation, hooked on nothing."""
    model = noncomp.Model(initial_state=ALL_G, transitions={"my_leak": transition_to(LEAK_G)})
    r = noncomp_sampling_api("H 0\nS 0\nM 0", model, shots=8, seed=1)
    assert r.final_status.shape == (8, 1)  # no hook fired; plain sampling ran


def test_non_hookable_instruction_key_raises():
    with pytest.raises(ValueError, match="does not support transition hooks"):
        noncomp.Model(transitions={"DEPOLARIZE1": transition_to(LEAK_G)})


def test_local_annotations_run_end_to_end(noncomp_sampling_api):
    """Hand-written transition annotations drive the trajectory."""
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"leak": transition_to(LEAK_G)},
        classifier=classifier_for(LEAK_G, [0.0, 1.0]),
    )
    r = noncomp_sampling_api("H 0\nLEVEL_TRANSITION[leak] 0\nM 0", model, shots=32, seed=2)
    assert np.all(r.measurements[:, 0] == 1)  # leaked slot takes the classifier bit

    lossy = noncomp.Model(
        initial_state=ALL_G, transitions={}, classifier=classifier_for(LOST, [1.0, 0.0])
    )
    s = noncomp_sampling_api("H 0\nLOSS(1) 0\nM 0", lossy, shots=32, seed=3)
    assert np.all(s.measurements[:, 0] == 0)  # lost slot fixed by the classifier


def test_leakage_preserves_computational_source_level(noncomp_sampling_api):
    result = noncomp_sampling_api("X 1\nLEAKAGE(1) 0 1", noncomp.Model(), shots=32, seed=208)
    assert (result.final_status[:, 0] == noncomp.QubitStatus.LEAK_G).all()
    assert (result.final_status[:, 1] == noncomp.QubitStatus.LEAK_E).all()


@pytest.mark.parametrize(
    ("initial", "expected"),
    [
        ([0, 0, 1, 0, 0], noncomp.QubitStatus.LEAK_G),
        ([0, 0, 0, 1, 0], noncomp.QubitStatus.LEAK_E),
        ([0, 0, 0, 0, 1], noncomp.QubitStatus.LOST),
    ],
)
def test_leakage_leaves_noncomputational_sources_unchanged(initial, expected, noncomp_sampling_api):
    result = noncomp_sampling_api(
        "LEAKAGE(1) 0", noncomp.Model(initial_state=initial), shots=16, seed=209
    )
    assert (result.final_status[:, 0] == expected).all()


def test_multi_target_leakage_samples_targets_independently(noncomp_sampling_api):
    p = 0.3
    shots = 10_000
    result = noncomp_sampling_api(f"LEAKAGE({p}) 0 1", noncomp.Model(), shots=shots, seed=210)
    leaked = result.final_status != noncomp.QubitStatus.COMPUTATIONAL
    assert abs(float(leaked[:, 0].mean()) - p) < 0.025
    assert abs(float(leaked[:, 1].mean()) - p) < 0.025
    assert abs(float(np.logical_and(leaked[:, 0], leaked[:, 1]).mean()) - p * p) < 0.02


def test_sample_accepts_a_parsed_circuit(noncomp_sampling_api):
    circuit = clifft.parse("LOSS(1) 0\nM 0")
    model = noncomp.Model(classifier=classifier_for(LOST, [1.0, 0.0]))
    result = noncomp_sampling_api(circuit, model, shots=8, seed=4)
    assert (result.final_status == LOST_KIND).all()
    assert (result.measurements == 0).all()


def test_gate_hooks_expand_inside_repeat_blocks(noncomp_sampling_api):
    circuit = "REPEAT 2 {\nS 0\n}\nM 0"
    model = noncomp.Model(
        transitions={"S": transition_to(LOST)},
        classifier=classifier_for(LOST, [1.0, 0.0]),
    )
    result = noncomp_sampling_api(circuit, model, shots=8, seed=5)
    assert (result.final_status == LOST_KIND).all()


def test_transition_wrong_shape_raises():
    with pytest.raises(ValueError):
        noncomp.Model(initial_state=ALL_G, transitions={"S": _zeros(4, 4)})


def test_desugared_gate_transition_key_raises():
    t = transition_to(LEAK_G)
    with pytest.raises(ValueError, match="rewritten at parse time"):
        noncomp.Model(initial_state=ALL_G, transitions={"MXX": t})


def test_identity_noop_transition_key_raises():
    t = transition_to(LEAK_G)
    with pytest.raises(ValueError, match="identity no-ops"):
        noncomp.Model(initial_state=ALL_G, transitions={"I": t})


def test_whitespace_transition_key_raises():
    t = transition_to(LEAK_G)
    with pytest.raises(ValueError, match="LEVEL_TRANSITION tag"):
        noncomp.Model(initial_state=ALL_G, transitions={" padded": t})


def test_lossless_matches_plain_record_shape(noncomp_sampling_api):
    model = noncomp.Model(initial_state=ALL_G)
    r = noncomp_sampling_api("H 0\nM 0\n", model, shots=512, seed=2)
    assert r.measurements.shape == (512, 1)
    assert r.num_measurements == 1
    ones = int(r.measurements.sum())
    assert 200 < ones < 312  # H then M is ~50/50


def test_initial_one_population_prep_is_deterministic(noncomp_sampling_api):
    # initial all-e is computational |1>: X-prep makes M read 1 every shot.
    model = noncomp.Model(initial_state=ALL_E)
    r = noncomp_sampling_api("M 0\n", model, shots=64, seed=3)
    assert (r.measurements == 1).all()


@pytest.mark.parametrize(
    "level,status,probability,seed",
    [
        (LEAK_G, noncomp.QubitStatus.LEAK_G, 0.3, 31),
        (LOST, noncomp.QubitStatus.LOST, 0.5, 32),
    ],
)
def test_initial_noncomputational_population_sampling(
    level, status, probability, seed, noncomp_sampling_api
):
    initial = [0.0] * 5
    initial[noncomp.Level.G] = 1.0 - probability
    initial[level] = probability
    model = noncomp.Model(
        initial_state=initial,
        classifier=classifier_for(level, [0.0, 1.0]),
    )

    r = noncomp_sampling_api("M 0\n", model, shots=8000, seed=seed)
    assert abs(float((r.final_status[:, 0] == status).mean()) - probability) < 0.04
    assert abs(float(r.measurements[:, 0].mean()) - probability) < 0.04


def test_state_independent_loss_changes_final_status(noncomp_sampling_api):
    model = noncomp.Model(initial_state=ALL_G, transitions={"S": transition_to(LOST)})
    r = noncomp_sampling_api("H 0\nS 0\n", model, shots=16, seed=4)
    assert (r.final_status == LOST_KIND).all()


def test_known_source_dependent_transition_accepted(noncomp_sampling_api):
    # Source-dependent (g->leak_g, e->leak_e). At S entry the qubit sits
    # in |g> -- no scrambling yet -- so the fire collapses onto g and the
    # destination is fixed to leak_g.
    t = _zeros(5, 5)
    t[LEAK_G][0] = 1.0
    t[LEAK_E][1] = 1.0
    model = noncomp.Model(initial_state=ALL_G, transitions={"S": t})
    r = noncomp_sampling_api("S 0\n", model, shots=8, seed=5)
    assert (r.final_status == noncomp.QubitStatus.LEAK_G).all()


def test_reset_reload_policy_changes_lost_site(noncomp_sampling_api):
    # A reset on a lost qubit drops by default (the site stays lost); with
    # reset_restores_lost it reloads the qubit to a computational state.
    circuit = "H 0\nS 0\nR 0\n"
    dropped = noncomp.Model(initial_state=ALL_G, transitions={"S": transition_to(LOST)})
    r = noncomp_sampling_api(circuit, dropped, shots=8, seed=7)
    assert (r.final_status == LOST_KIND).all()  # reset dropped; the site stays lost

    restore = noncomp.Model(
        initial_state=ALL_G, transitions={"S": transition_to(LOST)}, reset_restores_lost=True
    )
    r = noncomp_sampling_api(circuit, restore, shots=8, seed=7)
    assert (r.final_status == COMPUTATIONAL).all()


def test_leaked_classifier_bit_in_measurements(noncomp_sampling_api):
    model = leak_model(classifier_for(LEAK_G, [0.0, 1.0]))
    r = noncomp_sampling_api("H 0\nS 0\nM 0\n", model, shots=64, seed=8)
    assert (r.measurements == 1).all()


def test_classifier_bit_feeds_detector(noncomp_sampling_api):
    model = leak_model(classifier_for(LEAK_G, [0.0, 1.0]))
    r = noncomp_sampling_api("H 0\nS 0\nM 0\nDETECTOR rec[-1]\n", model, shots=64, seed=9)
    assert r.num_detectors == 1
    assert (r.detectors == 1).all()


def test_classifier_bit_feeds_observable(noncomp_sampling_api):
    model = leak_model(classifier_for(LEAK_G, [0.0, 1.0]))
    r = noncomp_sampling_api(
        "H 0\nS 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n", model, shots=64, seed=10
    )
    assert r.num_observables == 1
    assert (r.observables == 1).all()


def test_inverted_classifier_bit_feeds_records_detectors_and_observables(noncomp_sampling_api):
    model = leak_model(classifier_for(LEAK_G, [1.0, 0.0]))
    r = noncomp_sampling_api(
        "H 0\nS 0\nM !0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        model,
        shots=64,
        seed=101,
    )
    assert (r.measurements[:, 0] == 1).all()
    assert (r.detectors[:, 0] == 1).all()
    assert (r.observables[:, 0] == 1).all()


def test_lost_measurement_classifier_bit(noncomp_sampling_api):
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=classifier_for(LOST, [0.0, 1.0]),
    )
    r = noncomp_sampling_api("H 0\nS 0\nM 0\n", model, shots=64, seed=11)
    assert (r.measurements == 1).all()


def test_measure_reset_on_leaked_preserves_slot_and_resets(noncomp_sampling_api):
    model = leak_model(classifier_for(LEAK_G, [0.0, 1.0]))
    r = noncomp_sampling_api("H 0\nS 0\nMR 0\nM 0\n", model, shots=64, seed=12)
    assert r.num_measurements == 2
    assert (r.measurements[:, 0] == 1).all()  # MR slot: classifier bit
    assert (r.measurements[:, 1] == 0).all()  # reset, then M reads 0


def test_inverted_measure_reset_classifier_preserves_slot_and_resets(noncomp_sampling_api):
    model = leak_model(classifier_for(LEAK_G, [1.0, 0.0]))
    r = noncomp_sampling_api("H 0\nS 0\nMR !0\nM 0\n", model, shots=64, seed=102)
    assert r.num_measurements == 2
    assert (r.measurements[:, 0] == 1).all()
    assert (r.measurements[:, 1] == 0).all()


def test_missing_classifier_on_leaked_measurement_raises(noncomp_sampling_api):
    model = leak_model(classifier=None)
    with pytest.raises(ValueError, match="classifier"):
        noncomp_sampling_api("H 0\nS 0\nM 0\n", model, shots=8, seed=13)


def test_substochastic_classifier_column_rejects_at_construction():
    # Column sums to 0.6: the reserved reject mass has no sampling path,
    # so the model refuses to build.
    with pytest.raises(ValueError, match="reject columns are not supported"):
        leak_model(classifier_for(LEAK_G, [0.3, 0.3]))


def test_four_symbol_classifier_rejects_at_construction():
    mat = _zeros(4, 5)
    for lvl in range(5):
        mat[0][lvl] = 1.0
    mat[0][LEAK_G], mat[1][LEAK_G], mat[2][LEAK_G], mat[3][LEAK_G] = 0.4, 0.3, 0.2, 0.1
    with pytest.raises(ValueError, match="two record symbols"):
        leak_model(noncomp.Classifier(mat))


def test_classifier_stores_matrix():
    mat = _zeros(2, 5)
    for lvl in range(5):
        mat[0][lvl] = 1.0
    mat[0][noncomp.Level.E] = 0.0
    mat[1][noncomp.Level.E] = 1.0
    c = noncomp.Classifier(mat)
    assert len(c.matrix) == 2
    assert len(c.matrix[0]) == 5


def test_ternary_herald_rides_the_sidecar(noncomp_sampling_api):
    """A lost qubit's measurement heralds; the visible record stays binary."""
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=classifier_for(LOST, [0.0, 0.0, 1.0]),
    )
    r = noncomp_sampling_api("H 0\nCX 0 1\nS 0\nM 0\nM 1\n", model, shots=4000, seed=21)
    assert r.heralds.shape == (4000, 2)
    assert np.all(r.heralds[:, 0] == 1)  # the lost qubit's slot heralds
    assert np.all(r.heralds[:, 1] == 0)  # the survivor's does not
    # The heralded slot's record bit is a uniform draw, not a fixed value.
    assert abs(r.measurements[:, 0].mean() - 0.5) < 0.04
    # symbols() folds the herald back in as a third value.
    sym = r.symbols()
    assert np.all(sym[:, 0] == 2)
    assert np.array_equal(sym[:, 1], r.measurements[:, 1])


def test_two_symbol_classifier_heralds_nothing(noncomp_sampling_api):
    model = leak_model(classifier_for(LEAK_G, [0.5, 0.5]))
    r = noncomp_sampling_api("H 0\nS 0\nM 0\n", model, shots=64, seed=22)
    assert r.heralds.shape == (64, 1)
    assert not r.heralds.any()
    assert np.array_equal(r.symbols(), r.measurements)


def test_herald_deterministic_in_seed(noncomp_sampling_api):
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=classifier_for(LOST, [0.2, 0.1, 0.7]),
    )
    a = noncomp_sampling_api("H 0\nS 0\nM 0\n", model, shots=128, seed=23)
    b = noncomp_sampling_api("H 0\nS 0\nM 0\n", model, shots=128, seed=23)
    assert np.array_equal(a.heralds, b.heralds)
    assert np.array_equal(a.measurements, b.measurements)
    assert abs(a.heralds[:, 0].mean() - 0.7) < 0.15


def test_ternary_herald_feeds_detector_and_observable(noncomp_sampling_api):
    """Heralded slots keep a binary record drawn uniformly; anything consuming
    the record (detectors, observables) must see that same patched bit, and
    the herald rides only in the sidecar."""
    # The S hook loses with certainty; every shot is classified.
    # Classifier for LOST: P(0)=0.2, P(1)=0.1, P(herald)=0.7.
    # Heralded shots draw a uniform binary bit for the record.
    # Non-heralded shots draw binary from the conditional: P(1|not herald)=0.1/0.3=1/3.
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=classifier_for(LOST, [0.2, 0.1, 0.7]),
    )
    circuit = "H 0\nS 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    r = noncomp_sampling_api(circuit, model, shots=4000, seed=24)

    # Detector and observable both read the classifier-substituted measurement bit.
    assert np.array_equal(r.detectors[:, 0], r.measurements[:, 0])
    assert np.array_equal(r.observables[:, 0], r.measurements[:, 0])

    heralded = r.heralds[:, 0].astype(bool)

    # Herald fraction.
    assert abs(heralded.mean() - 0.7) < 0.04

    # Among heralded shots, the binary record is uniform.
    meas = r.measurements[:, 0]
    assert abs(float(meas[heralded].mean()) - 0.5) < 0.05

    # Among non-heralded shots, the record is drawn from the binary conditional.
    assert abs(float(meas[~heralded].mean()) - 1 / 3) < 0.07

    # symbols() reports 2 exactly where the herald is set; elsewhere it matches
    # the binary measurement bit.
    sym = r.symbols()
    assert np.array_equal(sym[:, 0] == 2, heralded)
    assert np.array_equal(sym[~heralded, 0], meas[~heralded])


def test_hidden_trace_out_does_not_shift_visible_count(noncomp_sampling_api):
    plain = noncomp.Model(initial_state=ALL_G)
    rp = noncomp_sampling_api("H 0\nCX 0 1\nM 0\nM 1\n", plain, shots=8, seed=16)
    assert rp.num_measurements == 2

    # Losing q0 inserts a hidden trace-out R, but the visible record is unchanged.
    lossy = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=classifier_for(LOST, [1.0, 0.0]),
    )
    r = noncomp_sampling_api("H 0\nCX 0 1\nS 0\nM 0\nM 1\n", lossy, shots=8, seed=17)
    assert r.num_measurements == 2


def test_zero_shots_reports_widths_and_empty_arrays(noncomp_sampling_api):
    model = noncomp.Model(initial_state=ALL_G)
    r = noncomp_sampling_api("H 0\nM 0\nDETECTOR rec[-1]\n", model, shots=0, seed=1)
    assert r.shots == 0
    assert r.num_measurements == 1
    assert r.num_detectors == 1
    assert r.measurements.shape == (0, 1)
    assert r.detectors.shape == (0, 1)


def test_final_status_shape_is_shots_by_qubits(noncomp_sampling_api):
    model = noncomp.Model(initial_state=ALL_G)
    r = noncomp_sampling_api("H 0\nCX 0 1\nM 0\nM 1\n", model, shots=32, seed=18)
    assert r.final_status.shape == (32, 2)


def test_deterministic_in_seed(noncomp_sampling_api):
    model = leak_model(classifier_for(LEAK_G, [0.5, 0.5]))
    circuit = "H 0\nS 0\nM 0\nDETECTOR rec[-1]\n"
    a = noncomp_sampling_api(circuit, model, shots=128, seed=42)
    b = noncomp_sampling_api(circuit, model, shots=128, seed=42)
    assert np.array_equal(a.measurements, b.measurements)
    assert np.array_equal(a.detectors, b.detectors)
    assert np.array_equal(a.final_status, b.final_status)


def test_different_seeds_differ(noncomp_sampling_api):
    # The companion to the determinism check: a different seed must actually
    # change the draws, so a fixed-sequence regression cannot masquerade as
    # "deterministic". 128 coin-flip measurements collide with probability
    # 2**-128.
    model = leak_model(classifier_for(LEAK_G, [0.5, 0.5]))
    circuit = "H 0\nS 0\nM 0\nDETECTOR rec[-1]\n"
    a = noncomp_sampling_api(circuit, model, shots=128, seed=42)
    b = noncomp_sampling_api(circuit, model, shots=128, seed=43)
    assert not np.array_equal(a.measurements, b.measurements)


def test_deterministic_in_seed_with_noncomputational_initials(noncomp_sampling_api):
    """Shots that start leaked or lost select their own starting modules
    rather than the shared main line; the streams must still be fully
    seed-determined under both dampings, and a different seed must differ."""
    leak_from_e = _zeros(5, 5)
    leak_from_e[LEAK_E][noncomp.Level.E] = 0.3
    circuit = "H 0\nS 0\nM 0\nH 1\nM 1\n"
    for damping in ("exact", "neglect"):
        model = noncomp.Model(
            initial_state=[0.5, 0.0, 0.2, 0.0, 0.3],
            transitions={"S": leak_from_e},
            classifier=classifier_for(LEAK_E, [0.0, 1.0]),
            damping=damping,
        )
        a = noncomp_sampling_api(circuit, model, shots=64, seed=29)
        b = noncomp_sampling_api(circuit, model, shots=64, seed=29)
        assert np.array_equal(a.measurements, b.measurements)
        assert np.array_equal(a.final_status, b.final_status)
        assert np.array_equal(a.heralds, b.heralds)
        # Vacuity guard: the noncomputational-initial path really ran.
        assert (a.final_status != noncomp.QubitStatus.COMPUTATIONAL).any()
        c = noncomp_sampling_api(circuit, model, shots=64, seed=30)
        assert not np.array_equal(a.measurements, c.measurements) or not np.array_equal(
            a.final_status, c.final_status
        )


def test_policy_knob_strings_validate():
    noncomp.Model(initial_state=ALL_G, damping="neglect")
    with pytest.raises(ValueError, match="damping"):
        noncomp.Model(initial_state=ALL_G, damping="bogus")


def test_loss_only_exact_damping_stays_at_stabilizer_cost(noncomp_sampling_api):
    """Equal per-source rates (LOSS always qualifies) take the trap-form
    lowering under damping="exact", so a loss-only model compiles at the
    neglect width -- max_active_width=0 passes -- while the physics stays exact:
    survivors keep full interference and the loss rate is right."""
    circuit = "H 0\nLOSS(0.3) 0\nH 0\nM 0"
    cls = noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]])
    model = noncomp.Model(classifier=cls)  # damping="exact" default
    r = noncomp_sampling_api(circuit, model, shots=4000, seed=17, max_active_width=0)
    meas = r.measurements[:, 0]
    lost = r.final_status[:, 0] == noncomp.QubitStatus.LOST
    # Survivors: the H .. H sandwich returns |0> deterministically; a lost
    # qubit reads the classifier's lost column (1).
    assert (meas[~lost] == 0).all()
    assert (meas[lost] == 1).all()
    assert lost.any() and (~lost).any()
    assert abs(lost.mean() - 0.3) < 4 * (0.3 * 0.7 / 4000) ** 0.5


def test_loss_on_many_coherent_qubits_compiles_flat(noncomp_sampling_api):
    """Per-qubit LOSS sites do not accumulate active width under damping="exact"."""
    circuit = (
        "".join(f"H {i}\n" for i in range(5))
        + "".join(f"LOSS(0.01) {i}\n" for i in range(5))
        + "".join(f"H {i}\nM {i}\n" for i in range(5))
    )
    cls = noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]])
    r = noncomp_sampling_api(
        circuit, noncomp.Model(classifier=cls), shots=8, seed=3, max_active_width=0
    )
    assert r.num_measurements == 5


def test_max_active_width_rejects_over_budget_exact_but_neglect_fits(noncomp_sampling_api):
    # A source-dependent leak (only out of e) on a coherent dormant qubit is
    # genuinely non-Clifford: damping="exact" expands it into the amplitude
    # array, adding one to the compiled active width per site. Three H-prefixed sites
    # push the peak to 3, over a cap of 2, so exact rejects (naming the
    # offending line); damping="neglect" keeps the active width flat and fits.
    leak_from_e = _zeros(5, 5)
    leak_from_e[LEAK_E][noncomp.Level.E] = 0.2  # T[to][from]; nothing out of g
    circuit = (
        "H 0\nH 1\nH 2\n"
        "LEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 1\nLEVEL_TRANSITION[leak] 2\n"
        "M 0\nM 1\nM 2\n"
    )

    def model(damping: str) -> noncomp.Model:
        return noncomp.Model(
            initial_state=ALL_G,
            transitions={"leak": leak_from_e},
            classifier=classifier_for(LEAK_E, [0.0, 1.0]),
            damping=damping,
        )

    with pytest.raises(ValueError, match="max_active_width"):
        noncomp_sampling_api(circuit, model("exact"), shots=4, seed=1, max_active_width=2)

    r = noncomp_sampling_api(circuit, model("neglect"), shots=4, seed=1, max_active_width=2)
    assert r.num_measurements == 3


def test_max_active_width_ignores_the_unreachable_all_computational_module(noncomp_sampling_api):
    """A model whose initial state has zero computational mass never runs the
    no-event module; its active width must not be able to reject the run."""
    circuit = (
        "H 0\n"
        + "".join(f"CX 0 {i}\n" for i in range(1, 6))
        + "".join(f"T {i}\nH {i}\n" for i in range(6))
        + "".join(f"M {i}\n" for i in range(6))
    )
    cls = noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]])
    # Control: with computational initials the no-event module is real and
    # its active width exceeds the cap -- this is what makes the test discriminating.
    with pytest.raises(ValueError, match="max_active_width"):
        noncomp_sampling_api(
            circuit, noncomp.Model(classifier=cls), shots=4, seed=3, max_active_width=0
        )
    lost = noncomp.Model(initial_state=[0, 0, 0, 0, 1], classifier=cls)
    r = noncomp_sampling_api(circuit, lost, shots=16, seed=3, max_active_width=0)
    assert (r.final_status == noncomp.QubitStatus.LOST).all()
    # The lost column reads symbol 1 with certainty: a raw readout of the
    # dropped-everything |0> carriers would give 0, so all-1 records verify
    # that the classifier wrote them.
    assert (r.measurements == 1).all()


def test_a_multi_round_circuit_runs_through_loss(noncomp_sampling_api):
    """A lost data qubit drops its syndrome CXs (identity on the ancilla).

    Dropping an operation with no representable effect on a vacated site is
    the only op policy -- there is no reject mode to opt out of.
    """
    circuit = "H 0\nS 0\nCX 0 1\nMR 1\nCX 0 1\nMR 1\nM 0\n"
    transitions = {"S": transition_to(LOST)}
    classifier = classifier_for(LOST, [0.0, 1.0])

    model = noncomp.Model(initial_state=ALL_G, transitions=transitions, classifier=classifier)
    r = noncomp_sampling_api(circuit, model, shots=16, seed=2)
    assert r.num_measurements == 3
    assert np.array_equal(r.measurements, np.tile([0, 0, 1], (16, 1)))
    assert np.all(r.final_status[:, 0] == LOST_KIND)


def test_mx_measurement_of_a_lost_qubit_classifies(noncomp_sampling_api):
    """An X-basis measurement of a lost qubit reads the classifier bit.

    On a vacated carrier the readout basis is incidental; MX classifies
    identically to M.  The lost column [0.0, 1.0] is deterministic 1."""
    circuit = "S 0\nMX 0\n"
    transitions = {"S": transition_to(LOST)}
    classifier = classifier_for(LOST, [0.0, 1.0])
    model = noncomp.Model(initial_state=ALL_G, transitions=transitions, classifier=classifier)
    r = noncomp_sampling_api(circuit, model, shots=8, seed=2)
    assert r.num_measurements == 1
    assert (r.measurements[:, 0] == 1).all()
    assert (r.final_status[:, 0] == LOST_KIND).all()


def test_zero_fire_loss_before_firing_loss_samples_cleanly(noncomp_sampling_api):
    """A zero-rate LOSS elided by trace() must not shift later site ids."""
    classifier = noncomp.Classifier([[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 1.0]])
    model = noncomp.Model(initial_state=ALL_G, classifier=classifier)
    r = noncomp_sampling_api("LOSS(0) 0\nLOSS(0.5) 0\nM 0\n", model, shots=64, seed=90)
    assert r.shots == 64
    # Every shot ends Lost or Computational (no assertion failures).
    assert r.final_status.shape == (64, 1)
    # Lost shots read 1 (classifier's lost column), computational shots read 0.
    for i in range(64):
        if r.final_status[i, 0] == LOST_KIND:
            assert r.measurements[i, 0] == 1
        else:
            assert r.final_status[i, 0] == COMPUTATIONAL


def test_seepage_only_before_firing_site_samples_cleanly(noncomp_sampling_api):
    """A transition elided on computational sources must not shift later site ids."""
    seep = _zeros(5, 5)
    seep[LEAK_E][LEAK_E] = 1.0  # leak_e -> leak_e, only noncomp columns

    classifier = noncomp.Classifier([[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 1.0]])
    model = noncomp.Model(initial_state=ALL_G, transitions={"seep": seep}, classifier=classifier)
    r = noncomp_sampling_api("LEVEL_TRANSITION[seep] 0\nLOSS(1) 0\nM 0\n", model, shots=32, seed=91)
    assert r.shots == 32
    assert (r.final_status == LOST_KIND).all()
    assert (r.measurements[:, 0] == 1).all()


def test_computational_readout_confusion_misreports_the_record(noncomp_sampling_api):
    """The classifier's computational columns act as asymmetric readout
    confusion on Z measurements: the qubit collapses to its true state but
    the record bit is misreported at the column's off-diagonal rate."""
    m = _zeros(2, 5)
    m[0][noncomp.Level.G] = 0.7  # true 0 misread as 1 with probability 0.3
    m[1][noncomp.Level.G] = 0.3
    m[0][noncomp.Level.E] = 0.2  # true 1 misread as 0 with probability 0.2
    m[1][noncomp.Level.E] = 0.8
    for lvl in (LEAK_G, LEAK_E, LOST):
        m[0][lvl] = 1.0
    model = noncomp.Model(initial_state=ALL_G, transitions={}, classifier=noncomp.Classifier(m))

    zero = noncomp_sampling_api("M 0", model, shots=4000, seed=11)
    ones = int(zero.measurements[:, 0].sum())
    assert 1020 <= ones <= 1380  # expected 1200; ~6 sigma band

    one = noncomp_sampling_api("X 0\nM 0", model, shots=4000, seed=12)
    zeros = int((1 - one.measurements[:, 0]).sum())
    assert 650 <= zeros <= 950  # expected 800; ~6 sigma band


def _lose_model() -> noncomp.Model:
    """S loses its qubit with certainty; the classifier reads out
    faithfully on g/e and reads 0 on the lost column."""
    return noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=classifier_for(LOST, [1.0, 0.0]),
    )


def test_correlated_chain_on_lost_qubit_does_not_crash(noncomp_sampling_api):
    """A correlated-error chain whose head operand is lost must pass through
    the noncomp layers without an exception. Dropping the head would orphan
    the ELSE and produce a dangling-chain error at trace time."""
    model = _lose_model()
    circuit = "S 0\nE(0.5) X0 X1\nELSE_CORRELATED_ERROR(0.5) X1\nM 0 1\n"
    result = noncomp_sampling_api(circuit, model, shots=32, seed=83)
    # The loss really happened, and the lost column reads 0 every shot.
    assert (result.final_status[:, 0] == LOST_KIND).all()
    assert np.all(result.measurements[:, 0] == 0)


def test_fired_chain_head_on_lost_operand_blocks_else(noncomp_sampling_api):
    """Conditioning check: E(1) fires (the head always fires, operating on the
    vacated q0 carrier), so the ELSE must not fire. q1 records must all be 0
    because the ELSE's X1 was never applied; a dropped head would promote the
    ELSE to fire unconditionally and flip q1."""
    model = _lose_model()
    circuit = "S 0\nE(1) X0\nELSE_CORRELATED_ERROR(1) X1\nM 0 1\n"
    result = noncomp_sampling_api(circuit, model, shots=32, seed=89)
    assert (result.final_status[:, 0] == LOST_KIND).all()
    assert np.all(result.measurements[:, 1] == 0)


def test_chain_flip_on_parked_carrier_is_destroyed_by_restoring_reset(noncomp_sampling_api):
    """The passthrough is sound because a vacated carrier is unobservable;
    this verifies the restoration leg: a chain flip parked on a lost carrier is
    overwritten by the restoring reset, so the restored qubit reads a clean
    |0>. Had the qubit never been lost, the same X would flip a live qubit
    and the record would read 1, so a passing 0 proves the loss happened."""
    # A classifier is required when the model is capable and the circuit
    # measures; the identity columns carry no readout confusion here.
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=classifier_for(LOST, [1.0, 0.0]),
        reset_restores_lost=True,
    )
    result = noncomp_sampling_api("S 0\nE(1) X0\nR 0\nM 0", model, shots=16, seed=11)
    assert (result.final_status[:, 0] == COMPUTATIONAL).all()
    assert np.all(result.measurements[:, 0] == 0)


def test_chain_flip_on_parked_carrier_is_destroyed_by_recapture(noncomp_sampling_api):
    """The recapture leg of the same contract: a chain flip parked on a
    leaked carrier neither disturbs the classified record (first M reads the
    leak_g column, proving the leak happened) nor survives the recapture,
    which materializes the drawn destination exactly (second M reads 1)."""
    seep = _zeros(5, 5)
    seep[noncomp.Level.E][LEAK_G] = 1.0  # leak_g -> e, certainly
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LEAK_G), "seep": seep},
        classifier=classifier_for(LEAK_G, [1.0, 0.0]),
    )
    circuit = "S 0\nE(1) X0\nM 0\nLEVEL_TRANSITION[seep] 0\nM 0\n"
    result = noncomp_sampling_api(circuit, model, shots=16, seed=13)
    assert np.all(result.measurements[:, 0] == 0)  # classified while leaked
    assert np.all(result.measurements[:, 1] == 1)  # recaptured to e
    assert (result.final_status[:, 0] == COMPUTATIONAL).all()


def test_mx_classified_deterministic(noncomp_sampling_api):
    """MX on a certainly-lost qubit reads the classifier's lost column.

    The lost column [0.0, 1.0] is deterministic 1."""
    circuit = "S 0\nMX 0\n"
    transitions = {"S": transition_to(LOST)}
    classifier = classifier_for(LOST, [0.0, 1.0])
    model = noncomp.Model(initial_state=ALL_G, transitions=transitions, classifier=classifier)
    r = noncomp_sampling_api(circuit, model, shots=8, seed=201)
    assert r.num_measurements == 1
    assert (r.measurements[:, 0] == 1).all()
    assert (r.final_status[:, 0] == LOST_KIND).all()


def test_mx_inverted_complements_classifier_bit(noncomp_sampling_api):
    """MX !0 on a certainly-lost qubit produces the complement of the classifier bit."""
    circuit = "S 0\nMX !0\n"
    transitions = {"S": transition_to(LOST)}
    classifier = classifier_for(LOST, [0.0, 1.0])  # lost column: always 1 without inversion
    model = noncomp.Model(initial_state=ALL_G, transitions=transitions, classifier=classifier)
    r = noncomp_sampling_api(circuit, model, shots=8, seed=203)
    assert (r.measurements[:, 0] == 0).all()  # inverted: 1 -> 0


def test_mry_classifies_and_restores_a_leaked_site(noncomp_sampling_api):
    model = noncomp.Model(
        transitions={"S": transition_to(LEAK_G)},
        classifier=classifier_for(LEAK_G, [0.0, 1.0]),
    )
    result = noncomp_sampling_api("S 0\nMRY 0", model, shots=8, seed=204)
    assert (result.measurements == 1).all()
    assert (result.final_status == COMPUTATIONAL).all()


def test_capable_model_with_measurement_requires_classifier(noncomp_sampling_api):
    """A model that can lose qubits requires a classifier when the circuit measures."""
    transitions = {"S": transition_to(LOST)}
    model = noncomp.Model(initial_state=ALL_G, transitions=transitions)
    # The circuit uses the "S" hook so the annotated circuit has a
    # LEVEL_TRANSITION node that fires into the noncomp category.
    with pytest.raises(ValueError, match="classifier is required"):
        noncomp_sampling_api("H 0\nS 0\nM 0\n", model, shots=4, seed=4)


def test_mpad_does_not_require_classifier(noncomp_sampling_api):
    """MPAD writes classical literals and never reads a noncomputational carrier."""
    model = noncomp.Model(initial_state=ALL_G, transitions={"S": transition_to(LOST)})
    result = noncomp_sampling_api("S 0\nMPAD 0 1\n", model, shots=4, seed=4)

    assert (result.measurements == [0, 1]).all()
    assert (result.final_status[:, 0] == LOST_KIND).all()


def test_capable_model_rejects_mpp(noncomp_sampling_api):
    """MPP is not supported when the model can leak or lose qubits."""
    transitions = {"S": transition_to(LOST)}
    classifier = classifier_for(LOST, [0.5, 0.5])
    model = noncomp.Model(initial_state=ALL_G, transitions=transitions, classifier=classifier)
    with pytest.raises(ValueError, match="not supported"):
        noncomp_sampling_api("S 0\nMPP Z0*Z1\n", model, shots=4, seed=5)


def test_exp_val_probe_is_rejected(noncomp_sampling_api):
    with pytest.raises(ValueError, match="EXP_VAL probes are not supported"):
        noncomp_sampling_api("EXP_VAL Z0\n", noncomp.Model(), shots=1, seed=5)


def test_contract_validated_for_zero_shots(noncomp_sampling_api):
    """Validation is shot-count independent: shots=0 still rejects a
    leak-capable model that measures without a classifier."""
    model = noncomp.Model(initial_state=ALL_G, transitions={"S": transition_to(LOST)})
    with pytest.raises(ValueError, match="classifier is required"):
        noncomp_sampling_api("S 0\nM 0", model, shots=0, seed=1)


def test_malformed_transition_error_names_key():
    """A bad transition matrix error must include the offending key in the message."""
    bad = [[0.0] * 5 for _ in range(5)]
    bad[0][0] = 1.5  # column sum exceeds 1

    with pytest.raises(ValueError, match="'CZ'"):
        noncomp.Model(initial_state=ALL_G, transitions={"S": transition_to(LEAK_G), "CZ": bad})


@pytest.mark.parametrize("annotation", ["LEAKAGE(0.1) 0", "LOSS(0.1) 0"])
def test_compile_annotated_circuit_raises_invalid_argument_with_noncomp_sample_hint(annotation):
    """Ordinary compilation directs noncomputational annotations to noncomp.sample."""
    import clifft

    with pytest.raises(ValueError, match="noncomp.sample"):
        clifft.compile(f"{annotation}\nM 0")


def test_sample_type_hints_resolve_at_runtime():
    """The Circuit | str annotation must be introspectable: Circuit is a
    real module-level name, not a TYPE_CHECKING-only import."""
    from typing import get_type_hints

    hints = get_type_hints(noncomp.sample)
    assert "circuit" in hints


def test_noncomp_preserves_true_prefix_expressions_across_continuations():
    model = noncomp.Model(classifier=classifier_for(LEAK_G, [1.0, 0.0]))
    result = noncomp.sample("X_ERROR(1) 0\nLEAKAGE(1) 1\nM 0", model, shots=8, seed=280)
    assert np.array_equal(result.measurements, np.ones((8, 1), dtype=np.uint8))


def test_qubit_status_values():
    """QubitStatus integer values differ from Level values for the shared names."""
    assert int(noncomp.QubitStatus.COMPUTATIONAL) == 0
    assert int(noncomp.QubitStatus.LEAK_G) == 1
    assert int(noncomp.QubitStatus.LEAK_E) == 2
    assert int(noncomp.QubitStatus.LOST) == 3


def test_qubit_status_not_level_collision_guard():
    """QubitStatus.LOST != Level.LOST and QubitStatus.LEAK_G != Level.LEAK_G.

    The two enums share member names with different integer values; mixing
    them in a comparison would silently produce wrong results.
    """
    assert int(noncomp.QubitStatus.LOST) != int(noncomp.Level.LOST)
    assert int(noncomp.QubitStatus.LEAK_G) != int(noncomp.Level.LEAK_G)


def test_fine_grained_leak_e_status(noncomp_sampling_api):
    """A certain leak into leak_e yields final_status == QubitStatus.LEAK_E."""
    t = _zeros(5, 5)
    t[noncomp.Level.LEAK_E][noncomp.Level.G] = 1.0
    t[noncomp.Level.LEAK_E][noncomp.Level.E] = 1.0
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": t},
        classifier=classifier_for(noncomp.Level.LEAK_E, [0.0, 1.0]),
    )
    r = noncomp_sampling_api("S 0\n", model, shots=8, seed=77)
    assert (r.final_status == noncomp.QubitStatus.LEAK_E).all()


def test_model_default_initial_state_reads_zero(noncomp_sampling_api):
    r = noncomp_sampling_api("M 0", noncomp.Model(), shots=4, seed=1)
    assert np.all(r.measurements == 0)


def test_model_repr_contains_keys_and_damping():
    model = noncomp.Model(
        transitions={"S": transition_to(LEAK_G), "seep": transition_to(LEAK_E)},
        damping="neglect",
    )
    r = repr(model)
    assert "S" in r
    assert "seep" in r
    assert "neglect" in r


def test_ternary_herald_on_measure_reset(noncomp_sampling_api):
    """MR on a leaked qubit heralds the sidecar, resets the qubit, and the
    next M reads 0.

    Three-row classifier: always-herald column for leak_g ([[1,0,0,0,1],
    [0,1,0,1,0],[0,0,1,0,0]]). "leak" = {g->leak_g p=1} hooked on S.
    Circuit: S 0 / MR 0 / M 0.

    Expect:
    - heralds[:, 0] == 1 on every shot (MR slot heralded).
    - r.measurements[:, 0] carries an unbiased bit (both 0 and 1 occur over
      >= 256 shots): the heralded slot gets a uniformly drawn replacement.
    - r.measurements[:, 1] == 0 on every shot: MR restored the leaked qubit
      to |0>, so the second M reads 0.
    - final_status[:, 0] == COMPUTATIONAL: the MR reset the qubit level.
    """
    HERALD_SHOTS = 256

    # Three-row classifier: g/e faithful; leak_g always heralds (row 2);
    # e/leak_e/lost columns are standard symbol-0 or symbol-1 per level.
    clf_matrix = [
        [1.0, 0.0, 0.0, 0.0, 1.0],  # symbol 0
        [0.0, 1.0, 0.0, 1.0, 0.0],  # symbol 1
        [0.0, 0.0, 1.0, 0.0, 0.0],  # symbol 2 (herald)
    ]
    transitions_leak = _zeros(5, 5)
    transitions_leak[LEAK_G][noncomp.Level.G] = 1.0  # g -> leak_g, certainly
    transitions_leak[LEAK_G][noncomp.Level.E] = 1.0  # e -> leak_g, certainly

    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transitions_leak},
        classifier=noncomp.Classifier(clf_matrix),
    )
    r = noncomp_sampling_api("S 0\nMR 0\nM 0\n", model, shots=HERALD_SHOTS, seed=271)

    assert r.num_measurements == 2

    # MR slot heralds on every shot.
    assert np.all(r.heralds[:, 0] == 1), "MR slot did not herald on every shot"

    # The heralded MR record bit is a uniformly drawn replacement: both values occur.
    mr_bits = r.measurements[:, 0]
    assert mr_bits.any(), "MR slot never recorded 1 over 256 shots"
    assert not mr_bits.all(), "MR slot always recorded 1 over 256 shots"

    # The second M reads 0 on every shot: MR restored the leaked qubit.
    assert np.all(r.measurements[:, 1] == 0), "post-MR measurement was not 0"

    # Final status is COMPUTATIONAL: the MR reset cleared the leak level.
    assert np.all(r.final_status[:, 0] == COMPUTATIONAL), "final status was not COMPUTATIONAL"


def test_seed_none_smoke(noncomp_sampling_api):
    """sample() with no seed runs and returns the correct shapes.

    The entropy-default path (seed=None) is never exercised by the
    determinism or different-seed checks.  A shape-only smoke test is
    sufficient: correctness is covered by the seeded suite."""
    r = noncomp_sampling_api("H 0\nM 0", noncomp.Model(), shots=8)
    assert r.shots == 8
    assert r.num_measurements == 1
    assert r.measurements.shape == (8, 1)
    assert r.final_status.shape == (8, 1)
