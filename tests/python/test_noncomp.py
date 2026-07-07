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
    """Binary classifier; `level`'s column is `col`, computational levels read
    out faithfully (no readout confusion), other columns are symbol 0."""
    m = _zeros(2, 5)
    for lvl in range(5):
        m[0][lvl] = 1.0
    m[0][noncomp.Level.E] = 0.0
    m[1][noncomp.Level.E] = 1.0
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


def test_non_gate_key_is_a_named_transition():
    """A key that names no gate is a named transition: referenceable from a
    LEVEL_TRANSITION[key] annotation, hooked on nothing."""
    model = noncomp.Model(initial_state=ALL_G, transitions={"my_leak": transition_to(LEAK_G)})
    r = noncomp.sample("H 0\nS 0\nM 0", model, shots=8, seed=1)
    assert r.final_status.shape == (8, 1)  # no hook fired; plain sampling ran


def test_local_annotations_run_end_to_end():
    """Hand-written LOSS and LEVEL_TRANSITION annotations drive the trajectory."""
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"leak": transition_to(LEAK_G)},
        classifier=classifier_for(LEAK_G, [0.0, 1.0]),
    )
    r = noncomp.sample("H 0\nLEVEL_TRANSITION[leak] 0\nM 0", model, shots=32, seed=2)
    assert np.all(r.measurements[:, 0] == 1)  # leaked slot takes the classifier bit

    lossy = noncomp.Model(
        initial_state=ALL_G, transitions={}, classifier=classifier_for(LOST, [1.0, 0.0])
    )
    s = noncomp.sample("H 0\nLOSS(1) 0\nM 0", lossy, shots=32, seed=3)
    assert np.all(s.measurements[:, 0] == 0)  # lost slot pinned by the classifier


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
    # Source-dependent (g->leak_g, e->leak_e). At S entry the qubit sits
    # in |g> -- no scrambling yet -- so the fire collapses onto g and the
    # destination is pinned to leak_g.
    t = _zeros(5, 5)
    t[LEAK_G][0] = 1.0
    t[LEAK_E][1] = 1.0
    model = noncomp.Model(initial_state=ALL_G, transitions={"S": t})
    r = noncomp.sample("S 0\n", model, shots=8, seed=5)
    assert (r.final_status == LEAKED).all()


def test_reset_reload_policy_changes_lost_site():
    # A reset on a lost qubit drops by default (the site stays lost); with
    # reset_restores_lost it reloads the qubit to a computational state.
    circuit = "H 0\nS 0\nR 0\n"
    dropped = noncomp.Model(initial_state=ALL_G, transitions={"S": transition_to(LOST)})
    r = noncomp.sample(circuit, dropped, shots=8, seed=7)
    assert (r.final_status == LOST_KIND).all()  # reset dropped; the site stays lost

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


def test_inverted_classifier_bit_feeds_records_detectors_and_observables():
    model = leak_model(classifier_for(LEAK_G, [1.0, 0.0]))
    r = noncomp.sample(
        "H 0\nS 0\nM !0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        model,
        shots=64,
        seed=101,
    )
    assert (r.measurements[:, 0] == 1).all()
    assert (r.detectors[:, 0] == 1).all()
    assert (r.observables[:, 0] == 1).all()


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


def test_inverted_measure_reset_classifier_preserves_slot_and_resets():
    model = leak_model(classifier_for(LEAK_G, [1.0, 0.0]))
    r = noncomp.sample("H 0\nS 0\nMR !0\nM 0\n", model, shots=64, seed=102)
    assert r.num_measurements == 2
    assert (r.measurements[:, 0] == 1).all()
    assert (r.measurements[:, 1] == 0).all()


def test_missing_classifier_on_leaked_measurement_raises():
    model = leak_model(classifier=None)
    with pytest.raises(ValueError, match="classifier"):
        noncomp.sample("H 0\nS 0\nM 0\n", model, shots=8, seed=13)


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
        leak_model(noncomp.Classifier(["0", "1", "2", "3"], mat))


def test_duplicate_classifier_symbols_reject_in_python():
    mat = _zeros(2, 5)
    for lvl in range(5):
        mat[0][lvl] = 1.0
    mat[0][noncomp.Level.E] = 0.0
    mat[1][noncomp.Level.E] = 1.0
    with pytest.raises(ValueError, match="duplicate symbol"):
        noncomp.Classifier(["0", "0"], mat)


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


def test_different_seeds_differ():
    # The companion to the determinism pin: a different seed must actually
    # change the draws, so a fixed-sequence regression cannot masquerade as
    # "deterministic". 128 coin-flip measurements collide with probability
    # 2**-128.
    model = leak_model(classifier_for(LEAK_G, [0.5, 0.5]))
    circuit = "H 0\nS 0\nM 0\nDETECTOR rec[-1]\n"
    a = noncomp.sample(circuit, model, shots=128, seed=42)
    b = noncomp.sample(circuit, model, shots=128, seed=43)
    assert not np.array_equal(a.measurements, b.measurements)


# --- 8. Policy knobs --------------------------------------------------------


def test_policy_knob_strings_validate():
    noncomp.Model(initial_state=ALL_G, damping="neglect")
    with pytest.raises(ValueError, match="damping"):
        noncomp.Model(initial_state=ALL_G, damping="bogus")


def test_max_rank_rejects_over_budget_exact_but_neglect_fits():
    # A source-dependent leak (only out of e) on a coherent dormant qubit is
    # genuinely non-Clifford: damping="exact" expands it into the amplitude
    # array, adding one to the compiled rank per site. Three H-prefixed sites
    # push the peak to 3, over a cap of 2, so exact rejects (naming the
    # offending line); damping="neglect" keeps the rank flat and fits.
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

    with pytest.raises(ValueError, match="max_rank"):
        noncomp.sample(circuit, model("exact"), shots=4, seed=1, max_rank=2)

    r = noncomp.sample(circuit, model("neglect"), shots=4, seed=1, max_rank=2)
    assert r.num_measurements == 3


def test_a_multi_round_circuit_runs_through_loss():
    """A lost data qubit drops its syndrome CXs (identity on the ancilla).

    Dropping an operation with no representable effect on a vacated site is
    the only op policy -- there is no reject mode to opt out of.
    """
    circuit = "H 0\nS 0\nCX 0 1\nMR 1\nCX 0 1\nMR 1\nM 0\n"
    transitions = {"S": transition_to(LOST)}
    classifier = classifier_for(LOST, [0.0, 1.0])

    model = noncomp.Model(initial_state=ALL_G, transitions=transitions, classifier=classifier)
    r = noncomp.sample(circuit, model, shots=16, seed=2)
    assert r.num_measurements == 3
    assert np.array_equal(r.measurements, np.tile([0, 0, 1], (16, 1)))
    assert np.all(r.final_status[:, 0] == LOST_KIND)


def test_xy_basis_measurement_of_a_lost_qubit_raises():
    """An X/Y-basis or parity measurement of a leaked or lost qubit has no
    faithful single-bit form, so it raises -- a representability limit, not a
    policy the caller can turn off."""
    circuit = "S 0\nMX 0\n"
    transitions = {"S": transition_to(LOST)}
    classifier = classifier_for(LOST, [0.0, 1.0])
    model = noncomp.Model(initial_state=ALL_G, transitions=transitions, classifier=classifier)
    with pytest.raises(ValueError, match="representable"):
        noncomp.sample(circuit, model, shots=4, seed=2)


def test_zero_fire_loss_before_firing_loss_samples_cleanly():
    """LOSS(0) before LOSS(0.5) must not shift site ids or abort.

    Before the fix, LOSS(0) was incorrectly kept in the site table while
    trace() elided it, causing a site-id mismatch that aborted in Debug or
    segfaulted in Release.
    """
    classifier = noncomp.Classifier(
        ["0", "1"], [[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 1.0]]
    )
    model = noncomp.Model(initial_state=ALL_G, classifier=classifier)
    r = noncomp.sample("LOSS(0) 0\nLOSS(0.5) 0\nM 0\n", model, shots=64, seed=90)
    assert r.shots == 64
    # Every shot ends Lost or Computational (no assertion failures).
    assert r.final_status.shape == (64, 1)
    # Lost shots read 1 (classifier's lost column), computational shots read 0.
    for i in range(64):
        if r.final_status[i, 0] == LOST_KIND:
            assert r.measurements[i, 0] == 1
        else:
            assert r.final_status[i, 0] == COMPUTATIONAL


def test_seepage_only_before_firing_site_samples_cleanly():
    """A seepage-only LEVEL_TRANSITION before a firing site must not shift site ids.

    A transition with column_sum(G)=column_sum(E)=0 is elided by trace() but
    was previously kept in the site table, producing a site-id mismatch.
    """
    seep = _zeros(5, 5)
    seep[LEAK_E][LEAK_E] = 1.0  # leak_e -> leak_e, only noncomp columns

    classifier = noncomp.Classifier(
        ["0", "1"], [[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 1.0]]
    )
    model = noncomp.Model(initial_state=ALL_G, transitions={"seep": seep}, classifier=classifier)
    r = noncomp.sample("LEVEL_TRANSITION[seep] 0\nLOSS(1) 0\nM 0\n", model, shots=32, seed=91)
    assert r.shots == 32
    assert (r.final_status == LOST_KIND).all()
    assert (r.measurements[:, 0] == 1).all()


def test_computational_readout_confusion_misreports_the_record():
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
    model = noncomp.Model(
        initial_state=ALL_G, transitions={}, classifier=noncomp.Classifier(["0", "1"], m)
    )

    zero = noncomp.sample("M 0", model, shots=4000, seed=11)
    ones = int(zero.measurements[:, 0].sum())
    assert 1020 <= ones <= 1380  # expected 1200; ~6 sigma band

    one = noncomp.sample("X 0\nM 0", model, shots=4000, seed=12)
    zeros = int((1 - one.measurements[:, 0]).sum())
    assert 650 <= zeros <= 950  # expected 800; ~6 sigma band


# --- 9. Correlated-chain passthrough on noncomputational operands ----------


def _lose_model() -> noncomp.Model:
    """S loses its qubit with certainty; the classifier reads out
    faithfully on g/e and reads 0 on the lost column."""
    return noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        classifier=classifier_for(LOST, [1.0, 0.0]),
    )


def test_correlated_chain_on_lost_qubit_does_not_crash():
    """A correlated-error chain whose head operand is lost must pass through
    the noncomp layers without an exception. Dropping the head would orphan
    the ELSE and produce a dangling-chain error at trace time."""
    model = _lose_model()
    circuit = "S 0\nE(0.5) X0 X1\nELSE_CORRELATED_ERROR(0.5) X1\nM 0 1\n"
    result = noncomp.sample(circuit, model, shots=32, seed=83)
    # The loss really happened, and the lost column reads 0 every shot.
    assert (result.final_status[:, 0] == LOST_KIND).all()
    assert np.all(result.measurements[:, 0] == 0)


def test_fired_chain_head_on_lost_operand_blocks_else():
    """Conditioning pin: E(1) fires (the head always fires, operating on the
    vacated q0 carrier), so the ELSE must not fire. q1 records must all be 0
    because the ELSE's X1 was never applied; a dropped head would promote the
    ELSE to fire unconditionally and flip q1."""
    model = _lose_model()
    circuit = "S 0\nE(1) X0\nELSE_CORRELATED_ERROR(1) X1\nM 0 1\n"
    result = noncomp.sample(circuit, model, shots=32, seed=89)
    assert (result.final_status[:, 0] == LOST_KIND).all()
    assert np.all(result.measurements[:, 1] == 0)


def test_chain_flip_on_parked_carrier_is_destroyed_by_restoring_reset():
    """The passthrough is sound because a vacated carrier is unobservable;
    this pins the restoration leg: a chain flip parked on a lost carrier is
    overwritten by the restoring reset, so the restored qubit reads a clean
    |0>. Had the qubit never been lost, the same X would flip a live qubit
    and the record would read 1, so a passing 0 proves the loss happened."""
    model = noncomp.Model(
        initial_state=ALL_G,
        transitions={"S": transition_to(LOST)},
        reset_restores_lost=True,
    )
    result = noncomp.sample("S 0\nE(1) X0\nR 0\nM 0", model, shots=16, seed=11)
    assert (result.final_status[:, 0] == COMPUTATIONAL).all()
    assert np.all(result.measurements[:, 0] == 0)


def test_chain_flip_on_parked_carrier_is_destroyed_by_recapture():
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
    result = noncomp.sample(circuit, model, shots=16, seed=13)
    assert np.all(result.measurements[:, 0] == 0)  # classified while leaked
    assert np.all(result.measurements[:, 1] == 1)  # recaptured to e
    assert (result.final_status[:, 0] == COMPUTATIONAL).all()


# =========================================================================
# Static pre-sampling validation tests
# =========================================================================


def test_static_check_low_rate_mx_raises_deterministically():
    """A low-rate leak to leak_e makes MX not representable.

    Before the static check, this circuit sampled cleanly on most seeds
    because the 0.01 rate rarely fired on the first shot.  The static
    check rejects the pair deterministically before any shot is drawn.
    """
    m = _zeros(5, 5)
    m[LEAK_E][noncomp.Level.G] = 0.01
    m[LEAK_E][noncomp.Level.E] = 0.01
    transitions = {"S": m}
    classifier = classifier_for(LEAK_E, [0.0, 1.0])
    model = noncomp.Model(initial_state=ALL_G, transitions=transitions, classifier=classifier)
    with pytest.raises(ValueError, match="representable"):
        noncomp.sample("S 0\nMX 0", model, shots=1, seed=1)


def test_static_check_r_restores_leak_before_mx():
    """R always restores a leaked qubit; MX after R meets only a computational qubit.

    This is a false-positive guard: the static check must not reject a
    circuit where the leak is provably gone before the unsupported gate.
    """
    m = _zeros(5, 5)
    m[LEAK_E][noncomp.Level.G] = 0.01
    m[LEAK_E][noncomp.Level.E] = 0.01
    transitions = {"S": m}
    classifier = classifier_for(LEAK_E, [0.0, 1.0])
    model = noncomp.Model(initial_state=ALL_G, transitions=transitions, classifier=classifier)
    result = noncomp.sample("S 0\nR 0\nMX 0", model, shots=8, seed=1)
    assert result.shots == 8
    assert result.measurements.shape == (8, 1)


def test_static_check_certain_recapture_accepts_mx():
    """A qubit starting on leak_e whose S hook recaptures to g with
    certainty always meets MX computationally; the no-event branch of the
    leak_e source is unreachable, so the reachability walk must retire it."""
    m = _zeros(5, 5)
    m[LEAK_E][noncomp.Level.G] = 0.1
    m[LEAK_E][noncomp.Level.E] = 0.1
    m[noncomp.Level.G][LEAK_E] = 1.0
    model = noncomp.Model(initial_state=[0.0, 0.0, 0.0, 1.0, 0.0], transitions={"S": m})
    result = noncomp.sample("S 0\nMX 0", model, shots=4, seed=1)
    assert result.measurements.shape == (4, 1)
    assert (result.final_status[:, 0] == COMPUTATIONAL).all()


def test_static_check_re_leak_after_recapture_rejects():
    """After the certain recapture, a second S can leak again from the
    computational columns, so MX genuinely can meet a leaked qubit."""
    m = _zeros(5, 5)
    m[LEAK_E][noncomp.Level.G] = 0.1
    m[LEAK_E][noncomp.Level.E] = 0.1
    m[noncomp.Level.G][LEAK_E] = 1.0
    model = noncomp.Model(initial_state=[0.0, 0.0, 0.0, 1.0, 0.0], transitions={"S": m})
    with pytest.raises(ValueError, match="representable"):
        noncomp.sample("S 0\nS 0\nMX 0", model, shots=1, seed=1)
