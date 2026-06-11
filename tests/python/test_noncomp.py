"""Python-facing tests for the noncomputational (leakage/loss) API.

Covers model construction, end-to-end sampling, classifier/record semantics,
and record-layout invariants through the clifft.noncomp surface. The model uses
the default five-level set (g=0, e=1, leak_g=2, leak_e=3, lost=4); matrices are
positional over those levels, so no level ids appear in the API.
"""

from __future__ import annotations

import numpy as np
import pytest

from clifft import noncomp

LEAK_G, LEAK_E, LOST = noncomp.Level.LEAK_G, noncomp.Level.LEAK_E, noncomp.Level.LOST
ALL_G = [1.0, 0.0, 0.0, 0.0, 0.0]
ALL_E = [0.0, 1.0, 0.0, 0.0, 0.0]
COMPUTATIONAL = noncomp.QubitStatusKind.COMPUTATIONAL
LEAKED = noncomp.QubitStatusKind.LEAKED
LOST_KIND = noncomp.QubitStatusKind.LOST


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def transition_to(level: int) -> list[list[float]]:
    """T[to][from]: g and e both jump to `level` (source-independent)."""
    m = _zeros(5, 5)
    m[level][noncomp.Level.G] = 1.0
    m[level][noncomp.Level.E] = 1.0
    return m


def classifier_for(level: int, col: list[float]) -> noncomp.Classifier:
    """Binary classifier; `level`'s column is `col`, every other column is symbol 0."""
    m = _zeros(2, 5)
    for lvl in range(5):
        m[0][lvl] = 1.0
    m[0][level] = col[0]
    m[1][level] = col[1]
    return noncomp.Classifier(["0", "1"], m)


def leak_model(classifier: noncomp.Classifier | None = None) -> noncomp.Model:
    return noncomp.Model(
        initial_state=ALL_G, transitions={"S": transition_to(LEAK_G)}, classifier=classifier
    )


# --- 1. Model construction -------------------------------------------------


def test_level_names_and_indices():
    assert noncomp.LEVELS == ("g", "e", "leak_g", "leak_e", "lost")
    assert (int(noncomp.Level.G), int(noncomp.Level.LEAK_G), int(noncomp.Level.LOST)) == (0, 2, 4)


def test_build_model_needs_no_level_ids():
    # A full model is described by matrices + initial probabilities alone.
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LEAK_G)},
        classifier=classifier_for(LEAK_G, [0.0, 1.0]),
        reset_restores_lost=True,
    )
    assert isinstance(model, noncomp.Model)


def test_initial_state_wrong_sum_raises():
    with pytest.raises(ValueError):
        noncomp.Model(initial_state=[0.5, 0.0, 0.0, 0.0, 0.0])


def test_initial_state_wrong_length_raises():
    with pytest.raises(ValueError):
        noncomp.Model(initial_state=[1.0, 0.0])


def test_initial_state_out_of_range_probability_raises():
    with pytest.raises(ValueError):
        noncomp.Model(initial_state=[2.0, -1.0, 0.0, 0.0, 0.0])


def test_unknown_gate_key_raises():
    with pytest.raises(ValueError, match="transition key"):
        noncomp.Model(initial_state=ALL_G, transitions={"NOTAGATE": transition_to(LEAK_G)})


def test_transition_wrong_shape_raises():
    with pytest.raises(ValueError):
        noncomp.Model(initial_state=ALL_G, transitions={"S": _zeros(4, 4)})


# --- 2. End-to-end sampling ------------------------------------------------


def test_lossless_matches_plain_record_shape():
    model = noncomp.Model(initial_state=ALL_G)
    r = noncomp.sample("H 0\nM 0\n", model, shots=512, seed=2)
    assert r.measurements.shape == (512, 1)
    assert r.num_measurements == 1
    ones = int(r.measurements.sum())
    assert 200 < ones < 312  # H then M is ~50/50


def test_initial_one_population_prep_is_deterministic():
    # initial all-e is computational |1>: X-prep makes M read 1 every shot.
    model = noncomp.Model(initial_state=ALL_E)
    r = noncomp.sample("M 0\n", model, shots=64, seed=3)
    assert (r.measurements == 1).all()


def test_state_independent_loss_changes_final_status():
    model = noncomp.Model(initial_state=ALL_G, transitions={"S": transition_to(LOST)})
    r = noncomp.sample("H 0\nS 0\n", model, shots=16, seed=4)
    assert (r.final_status == LOST_KIND).all()


def test_known_source_dependent_transition_accepted():
    # Source-dependent (g->leak_g, e->leak_e). At S entry the qubit is
    # ComputationalKnown(g) -- no scrambling yet -- so the source is pinned.
    t = _zeros(5, 5)
    t[LEAK_G][0] = 1.0
    t[LEAK_E][1] = 1.0
    model = noncomp.Model(initial_state=ALL_G, transitions={"S": t})
    r = noncomp.sample("S 0\n", model, shots=8, seed=5)
    assert (r.final_status == LEAKED).all()


def test_unknown_source_dependent_transition_rejected():
    t = _zeros(5, 5)
    t[LEAK_G][0] = 1.0
    t[LEAK_E][1] = 1.0
    model = noncomp.Model(initial_state=ALL_G, transitions={"S": t})
    with pytest.raises(ValueError, match="source-dependent"):
        noncomp.sample("H 0\nS 0\n", model, shots=8, seed=6)


def test_reset_reload_policy_changes_lost_site():
    circuit = "H 0\nS 0\nR 0\n"
    reject = noncomp.Model(initial_state=ALL_G, transitions={"S": transition_to(LOST)})
    with pytest.raises(ValueError, match="not representable"):
        noncomp.sample(circuit, reject, shots=8, seed=7)

    restore = noncomp.Model(
        initial_state=ALL_G, transitions={"S": transition_to(LOST)}, reset_restores_lost=True
    )
    r = noncomp.sample(circuit, restore, shots=8, seed=7)
    assert (r.final_status == COMPUTATIONAL).all()


# --- 3. Classifier / record semantics --------------------------------------


def test_leaked_classifier_bit_in_measurements():
    model = leak_model(classifier_for(LEAK_G, [0.0, 1.0]))
    r = noncomp.sample("H 0\nS 0\nM 0\n", model, shots=64, seed=8)
    assert (r.measurements == 1).all()


def test_classifier_bit_feeds_detector():
    model = leak_model(classifier_for(LEAK_G, [0.0, 1.0]))
    r = noncomp.sample("H 0\nS 0\nM 0\nDETECTOR rec[-1]\n", model, shots=64, seed=9)
    assert r.num_detectors == 1
    assert (r.detectors == 1).all()


def test_classifier_bit_feeds_observable():
    model = leak_model(classifier_for(LEAK_G, [0.0, 1.0]))
    r = noncomp.sample("H 0\nS 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n", model, shots=64, seed=10)
    assert r.num_observables == 1
    assert (r.observables == 1).all()


def test_lost_measurement_classifier_bit():
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=classifier_for(LOST, [0.0, 1.0]),
    )
    r = noncomp.sample("H 0\nS 0\nM 0\n", model, shots=64, seed=11)
    assert (r.measurements == 1).all()


def test_measure_reset_on_leaked_preserves_slot_and_resets():
    model = leak_model(classifier_for(LEAK_G, [0.0, 1.0]))
    r = noncomp.sample("H 0\nS 0\nMR 0\nM 0\n", model, shots=64, seed=12)
    assert r.num_measurements == 2
    assert (r.measurements[:, 0] == 1).all()  # MR slot: classifier bit
    assert (r.measurements[:, 1] == 0).all()  # reset, then M reads 0


def test_missing_classifier_on_leaked_measurement_raises():
    model = leak_model(classifier=None)
    with pytest.raises(ValueError, match="classifier"):
        noncomp.sample("H 0\nS 0\nM 0\n", model, shots=8, seed=13)


def test_substochastic_classifier_column_unsupported():
    model = leak_model(classifier_for(LEAK_G, [0.3, 0.3]))  # column sums to 0.6
    with pytest.raises(ValueError, match="reject columns are not supported"):
        noncomp.sample("H 0\nS 0\nM 0\n", model, shots=8, seed=14)


def test_four_symbol_classifier_unsupported():
    mat = _zeros(4, 5)
    for lvl in range(5):
        mat[0][lvl] = 1.0
    mat[0][LEAK_G], mat[1][LEAK_G], mat[2][LEAK_G], mat[3][LEAK_G] = 0.4, 0.3, 0.2, 0.1
    model = leak_model(noncomp.Classifier(["0", "1", "2", "3"], mat))
    with pytest.raises(ValueError, match="two- or three-symbol"):
        noncomp.sample("H 0\nS 0\nM 0\n", model, shots=8, seed=15)


def _ternary_classifier(level: int, col: list[float]) -> noncomp.Classifier:
    """Three-symbol classifier; `level`'s column is `col`, others read symbol 0."""
    m = _zeros(3, 5)
    for lvl in range(5):
        m[0][lvl] = 1.0
    m[0][level], m[1][level], m[2][level] = col
    return noncomp.Classifier(["0", "1", "2"], m)


def test_ternary_herald_rides_the_sidecar():
    """A lost qubit's measurement heralds; the visible record stays binary."""
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=_ternary_classifier(LOST, [0.0, 0.0, 1.0]),
    )
    r = noncomp.sample("H 0\nCX 0 1\nS 0\nM 0\nM 1\n", model, shots=4000, seed=21)
    assert r.heralds.shape == (4000, 2)
    assert np.all(r.heralds[:, 0] == 1)  # the lost qubit's slot heralds
    assert np.all(r.heralds[:, 1] == 0)  # the survivor's does not
    # The heralded slot's record bit is a uniform draw, not a pinned value.
    assert abs(r.measurements[:, 0].mean() - 0.5) < 0.04
    # symbols() folds the herald back in as a third value.
    sym = r.symbols()
    assert np.all(sym[:, 0] == 2)
    assert np.array_equal(sym[:, 1], r.measurements[:, 1])


def test_two_symbol_classifier_heralds_nothing():
    model = leak_model(classifier_for(LEAK_G, [0.5, 0.5]))
    r = noncomp.sample("H 0\nS 0\nM 0\n", model, shots=64, seed=22)
    assert r.heralds.shape == (64, 1)
    assert not r.heralds.any()
    assert np.array_equal(r.symbols(), r.measurements)


def test_herald_deterministic_in_seed():
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=_ternary_classifier(LOST, [0.2, 0.1, 0.7]),
    )
    a = noncomp.sample("H 0\nS 0\nM 0\n", model, shots=128, seed=23)
    b = noncomp.sample("H 0\nS 0\nM 0\n", model, shots=128, seed=23)
    assert np.array_equal(a.heralds, b.heralds)
    assert np.array_equal(a.measurements, b.measurements)
    assert abs(a.heralds[:, 0].mean() - 0.7) < 0.15


# --- 4. Record layout invariants -------------------------------------------


def test_hidden_trace_out_does_not_shift_visible_count():
    plain = noncomp.Model(initial_state=ALL_G)
    rp = noncomp.sample("H 0\nCX 0 1\nM 0\nM 1\n", plain, shots=8, seed=16)
    assert rp.num_measurements == 2

    # Losing q0 inserts a hidden trace-out R, but the visible record is unchanged.
    lossy = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=classifier_for(LOST, [1.0, 0.0]),
    )
    r = noncomp.sample("H 0\nCX 0 1\nS 0\nM 0\nM 1\n", lossy, shots=8, seed=17)
    assert r.num_measurements == 2


def test_zero_shots_reports_widths_and_empty_arrays():
    model = noncomp.Model(initial_state=ALL_G)
    r = noncomp.sample("H 0\nM 0\nDETECTOR rec[-1]\n", model, shots=0, seed=1)
    assert r.shots == 0
    assert r.num_measurements == 1
    assert r.num_detectors == 1
    assert r.measurements.shape == (0, 1)
    assert r.detectors.shape == (0, 1)


def test_final_status_shape_is_shots_by_qubits():
    model = noncomp.Model(initial_state=ALL_G)
    r = noncomp.sample("H 0\nCX 0 1\nM 0\nM 1\n", model, shots=32, seed=18)
    assert r.final_status.shape == (32, 2)


def test_deterministic_in_seed():
    model = leak_model(classifier_for(LEAK_G, [0.5, 0.5]))
    circuit = "H 0\nS 0\nM 0\nDETECTOR rec[-1]\n"
    a = noncomp.sample(circuit, model, shots=128, seed=42)
    b = noncomp.sample(circuit, model, shots=128, seed=42)
    assert np.array_equal(a.measurements, b.measurements)
    assert np.array_equal(a.detectors, b.detectors)
    assert np.array_equal(a.final_status, b.final_status)


# --- 8. Policy knobs --------------------------------------------------------


def test_policy_knob_strings_validate():
    noncomp.Model(
        initial_state=ALL_G,
        unknown_source_policy="equalize_rates",
        lost_leaked_ops="drop",
    )
    with pytest.raises(ValueError, match="unknown_source_policy"):
        noncomp.Model(initial_state=ALL_G, unknown_source_policy="bogus")
    with pytest.raises(ValueError, match="lost_leaked_ops"):
        noncomp.Model(initial_state=ALL_G, lost_leaked_ops="bogus")


def test_equalize_rates_matches_the_analytic_plus_state_mixture():
    """|+> under a source-dependent leak (only g leaks, p = 0.4), equalized.

    Fires with probability p; on fire the uniform source draw either leaks
    (g column, classifier reads 1) or pseudo-jumps onto e (carrier collapses
    to |1>, reads 1); otherwise the |+> reads 1 half the time. So
    P(M=1) = p/2 + p/2 + (1-p)/2 = 0.5 + p/2 and P(leaked) = p/2.
    """
    leak_from_g_only = _zeros(5, 5)
    leak_from_g_only[LEAK_G][noncomp.Level.G] = 0.4
    circuit = "H 0\nS 0\nM 0\n"
    classifier = classifier_for(LEAK_G, [0.0, 1.0])

    reject_model = noncomp.Model(
        initial_state=ALL_G, transitions={"S": leak_from_g_only}, classifier=classifier
    )
    with pytest.raises(ValueError, match="source-dependent"):
        noncomp.sample(circuit, reject_model, shots=4, seed=1)

    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": leak_from_g_only},
        classifier=classifier,
        unknown_source_policy="equalize_rates",
    )
    r = noncomp.sample(circuit, model, shots=4000, seed=5)
    assert abs(r.measurements[:, 0].mean() - 0.7) < 0.04
    assert abs((r.final_status[:, 0] == LEAKED).mean() - 0.2) < 0.035


def test_drop_policy_runs_a_multi_round_circuit_through_loss():
    """A lost data qubit drops its syndrome CXs (identity on the ancilla)."""
    circuit = "H 0\nS 0\nCX 0 1\nMR 1\nCX 0 1\nMR 1\nM 0\n"
    transitions = {"S": transition_to(LOST)}
    classifier = classifier_for(LOST, [0.0, 1.0])

    reject_model = noncomp.Model(
        initial_state=ALL_G, transitions=transitions, classifier=classifier
    )
    with pytest.raises(ValueError, match="CX"):
        noncomp.sample(circuit, reject_model, shots=4, seed=2)

    model = noncomp.Model(
        initial_state=ALL_G,
        transitions=transitions,
        classifier=classifier,
        lost_leaked_ops="drop",
    )
    r = noncomp.sample(circuit, model, shots=16, seed=2)
    assert r.num_measurements == 3
    assert np.array_equal(r.measurements, np.tile([0, 0, 1], (16, 1)))
    assert np.all(r.final_status[:, 0] == LOST_KIND)
