"""Tests for source map and active-k history propagation through Python bindings."""

import ucc


def test_hir_source_map_parallel_to_ops() -> None:
    """HirModule.source_map length matches num_ops."""
    hir = ucc.trace(ucc.parse("H 0\nT 0\nM 0"))
    assert len(hir.source_map) == hir.num_ops


def test_hir_source_map_contains_correct_lines() -> None:
    """T gate on line 2 maps to source_line 2."""
    hir = ucc.trace(ucc.parse("H 0\nT 0\nM 0"))
    # First HIR op is the T gate (H absorbed), should be line 2
    assert hir.source_map[0] == [2]


def test_compiled_source_map_parallel_to_bytecode() -> None:
    """CompiledModule source_map and active_k_history match bytecode length."""
    prog = ucc.lower(ucc.trace(ucc.parse("H 0\nT 0\nM 0")))
    assert len(prog.source_map) == prog.num_instructions
    assert len(prog.active_k_history) == prog.num_instructions


def test_active_k_history_shows_expansion_and_compaction() -> None:
    """k rises for T gate expansion and falls back after measurement."""
    prog = ucc.lower(ucc.trace(ucc.parse("H 0\nT 0\nM 0")))
    k_hist = list(prog.active_k_history)
    assert max(k_hist) >= 1, "T gate should expand k to at least 1"
    assert k_hist[-1] == 0, "Measurement should compact k back to 0"


def test_optimizer_preserves_source_map() -> None:
    """Peephole fusion keeps source_map in sync."""
    hir = ucc.trace(ucc.parse("H 0\nT 0\nT 0\nM 0"))
    pm = ucc.PassManager()
    pm.add(ucc.PeepholeFusionPass())
    pm.run(hir)
    assert len(hir.source_map) == hir.num_ops
    # The fused S gate should carry both T-gate source lines
    found = False
    for i, op in enumerate(hir):
        if op.as_dict()["op_type"] == "CLIFFORD_PHASE":
            lines = hir.source_map[i]
            assert 2 in lines and 3 in lines
            found = True
            break
    assert found, "Expected a fused CLIFFORD_PHASE op"


def test_empty_circuit_produces_empty_maps() -> None:
    """Empty circuit yields empty source_map and k_history."""
    prog = ucc.lower(ucc.trace(ucc.parse("")))
    assert len(prog.source_map) == 0
    assert len(prog.active_k_history) == 0
