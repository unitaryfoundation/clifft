"""Tests for compile() with configurable HIR optimization passes."""

import gc

import pytest
from conftest import assert_statevectors_equiv, random_dense_clifford_t_circuit

import clifft


def test_compile_default_matches_explicit_pipeline() -> None:
    text = "H 0\nT 0\nCNOT 0 1\nM 0 1"
    compiled = clifft.compile(text)

    hir = clifft.trace(clifft.parse(text))
    clifft.default_hir_pass_manager().run(hir)
    manual = clifft.lower(hir)

    assert compiled.num_actions == manual.num_actions
    assert compiled.peak_active_width == manual.peak_active_width


def test_compile_explicit_none_skips_optimization() -> None:
    text = "H 0\nT 0\nCNOT 0 1\nM 0 1"
    compiled = clifft.compile(text, hir_passes=None)
    manual = clifft.lower(clifft.trace(clifft.parse(text)))

    assert compiled.num_actions == manual.num_actions
    assert compiled.peak_active_width == manual.peak_active_width


def test_hir_passes_reduce_t_cancellation() -> None:
    text = "H 0\nT 0\nT_DAG 0\nM 0"
    unoptimized = clifft.compile(text, hir_passes=None)
    optimized = clifft.compile(text)
    assert optimized.peak_active_width <= unoptimized.peak_active_width


def test_compile_postselection_with_passes() -> None:
    program = clifft.compile(
        "H 0\nM 0\nDETECTOR rec[-1]",
        postselection_mask=[1],
        hir_passes=clifft.default_hir_pass_manager(),
    )
    assert program.num_detectors == 1
    assert program.has_postselection


@pytest.mark.parametrize(
    "num_qubits,depth,seed",
    [(3, 50, 5000), (4, 80, 5001), (5, 100, 5002), (6, 100, 5004)],
)
def test_statevector_equiv_with_passes(num_qubits: int, depth: int, seed: int) -> None:
    text = random_dense_clifford_t_circuit(num_qubits, depth, seed)
    baseline = clifft.get_statevector(clifft.compile(text, hir_passes=None))
    optimized = clifft.get_statevector(clifft.compile(text))
    assert_statevectors_equiv(optimized, baseline)


def test_custom_pass_manager_via_compile() -> None:
    manager = clifft.HirPassManager()
    manager.add(clifft.PeepholeFusionPass())
    program = clifft.compile("H 0\nT 0\nT_DAG 0\nM 0", hir_passes=manager)
    assert program.peak_active_width == 0


def test_default_overrides_replace_in_original_slot() -> None:
    phase = clifft.PhasePolynomialPass(max_variables=0)
    manager = clifft.default_hir_pass_manager(overrides=[phase])
    program = clifft.compile("H 0\nT 0\nT_DAG 0\nT 1\nH 1\nM 0 1", hir_passes=manager)

    # Peephole cancels the pair before this disabled phase pass sees T 1.
    # The retained rotation pass removes T 1 afterward.
    assert phase.input_t_count == phase.output_t_count == 1
    assert not phase.applied
    assert program.peak_active_width == 0


@pytest.mark.parametrize("empty", [[], ()])
def test_empty_overrides_preserve_defaults(empty: object) -> None:
    source = "H 0 1\nT 0\nT 1\nH 0 1\nM 0 1"
    expected = clifft.trace(clifft.parse(source))
    actual = clifft.trace(clifft.parse(source))
    clifft.default_hir_pass_manager().run(expected)
    clifft.default_hir_pass_manager(overrides=empty).run(actual)
    assert actual.as_dict() == expected.as_dict()


@pytest.mark.parametrize("seed", [5000, 5001, 5002, 5004])
def test_default_configured_override_matches_default_pipeline(seed: int) -> None:
    source = random_dense_clifford_t_circuit(4, 80, seed)
    expected = clifft.trace(clifft.parse(source))
    actual = clifft.trace(clifft.parse(source))
    clifft.default_hir_pass_manager().run(expected)
    clifft.default_hir_pass_manager(overrides=[clifft.PhasePolynomialPass()]).run(actual)
    assert actual.as_dict() == expected.as_dict()


def test_multiple_overrides_ignore_argument_order() -> None:
    phase = clifft.PhasePolynomialPass(max_variables=0)
    peephole = clifft.PeepholeFusionPass()
    manager = clifft.default_hir_pass_manager(overrides=(phase, peephole))
    source = "H 0 1\nT 0\nT_DAG 0\nT 0\nT 1\nH 0 1\nM 0 1"
    hir = clifft.trace(clifft.parse(source))
    manager.run(hir)

    assert peephole.cancellations == 1
    assert phase.input_t_count == 2
    assert clifft.lower(hir).peak_active_width == 1


def test_overrides_retain_each_instance_until_manager_is_deleted() -> None:
    deleted = []

    class TrackedPhase(clifft.PhasePolynomialPass):
        def __del__(self) -> None:
            deleted.append(True)

    overrides = [TrackedPhase(max_variables=0)]
    manager = clifft.default_hir_pass_manager(overrides=overrides)
    overrides.clear()
    gc.collect()
    assert deleted == []
    program = clifft.compile("T 0", hir_passes=manager)
    assert program.peak_active_width == 0
    del manager
    gc.collect()
    assert deleted == [True]


def test_duplicate_overrides_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate override for PhasePolynomialPass"):
        clifft.default_hir_pass_manager(
            overrides=[clifft.PhasePolynomialPass(), clifft.PhasePolynomialPass(max_variables=0)]
        )


@pytest.mark.parametrize(
    "pass_type", [clifft.ActiveWidthSchedulePass, clifft.RemoveNoisePass, clifft.DropNonUnitaryPass]
)
def test_nondefault_overrides_are_rejected(pass_type: type) -> None:
    with pytest.raises(ValueError, match="is not a default HIR pass"):
        clifft.default_hir_pass_manager(overrides=[pass_type()])


@pytest.mark.parametrize("invalid", [None, 1, "PhasePolynomialPass", clifft.PhasePolynomialPass])
def test_override_instances_are_required(invalid: object) -> None:
    with pytest.raises(TypeError, match="overrides must contain HIR pass instances"):
        clifft.default_hir_pass_manager(overrides=[invalid])


def test_overrides_require_a_sequence() -> None:
    with pytest.raises(TypeError):
        clifft.default_hir_pass_manager(overrides=None)
