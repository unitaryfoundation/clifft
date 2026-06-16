"""Python binding coverage for the exact phase-polynomial T-count pass."""

import clifft


def test_exact_phase_polynomial_pass_binding_and_stats() -> None:
    hir = clifft.trace(clifft.parse("T 0\nT_DAG 0"))
    exact = clifft.ExactPhasePolynomialTCountPass()

    pm = clifft.HirPassManager()
    pm.add(exact)
    pm.run(hir)

    assert hir.num_t_gates == 0
    assert len(hir) == 0
    assert exact.max_rank == 4
    assert exact.blocks_considered == 1
    assert exact.blocks_optimized == 1
    assert exact.t_removed == 2
    assert "ExactPhasePolynomialTCountPass" in repr(exact)


def test_exact_phase_polynomial_pass_rank_cap_is_clamped() -> None:
    assert clifft.ExactPhasePolynomialTCountPass(99).max_rank == 4


def test_t_gate_block_collection_pass_binding_and_stats() -> None:
    hir = clifft.trace(clifft.parse("T 0\nR_Z(0.125) 0\nT 0"))
    collect = clifft.TGateBlockCollectionPass()

    pm = clifft.HirPassManager()
    pm.add(collect)
    pm.run(hir)

    assert hir.num_t_gates == 2
    assert collect.max_scan == 64
    assert collect.blocks_collected == 1
    assert collect.t_gates_moved == 1
    assert collect.adjacent_swaps == 1
    assert "TGateBlockCollectionPass" in repr(collect)
