"""Tests for read-only introspection bindings.

Verifies that HIR operations and their JSON-friendly
dictionary representations are accessible from Python without copying
entire vectors.
"""

import json
from typing import Any

import pytest

import clifft


class TestHirIntrospection:
    """HIR-level introspection: HeisenbergOp, OpType, iteration."""

    def test_trace_rotation_canonicalization_tolerance(self) -> None:
        inside = clifft.trace(clifft.parse("R_Z(0.5000000000005) 0"))
        outside = clifft.trace(clifft.parse("R_Z(0.500000000002) 0"))

        assert inside.num_ops == 0
        assert outside.num_ops == 1
        assert outside[0].op_type == clifft.OpType.PHASE_ROTATION
        assert outside[0].as_dict()["alpha"] == 0.500000000002

    def test_source_map_preserves_python_line_provenance(self) -> None:
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))

        assert len(hir.source_map) == hir.num_ops
        assert hir.source_map[0] == [2]
        assert clifft.trace(clifft.parse("")).source_map == []

    def test_hir_str_prints_ops(self) -> None:
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
        text = str(hir)
        assert "T +" in text or "T -" in text
        assert "MEASURE" in text

    def test_hir_repr(self) -> None:
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
        r = repr(hir)
        assert "HirModule" in r
        assert "1 T-gates" in r

    def test_hir_len(self) -> None:
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
        assert len(hir) == hir.num_ops

    def test_hir_getitem_positive(self) -> None:
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
        op = hir[0]
        assert isinstance(op, clifft.HeisenbergOp)
        assert op.op_type == clifft.OpType.T_GATE

    def test_hir_getitem_negative(self) -> None:
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
        last = hir[-1]
        assert last.op_type == clifft.OpType.MEASURE

    def test_hir_getitem_out_of_bounds(self) -> None:
        hir = clifft.trace(clifft.parse("M 0"))
        with pytest.raises(IndexError):
            _ = hir[999]

    def test_hir_iteration(self) -> None:
        hir = clifft.trace(clifft.parse("H 0\nT 0\nT 0\nM 0"))
        ops = list(hir)
        assert len(ops) == len(hir)
        for op in ops:
            assert isinstance(op, clifft.HeisenbergOp)

    def test_heisenberg_op_properties(self) -> None:
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
        t_op = hir[0]
        assert t_op.op_type == clifft.OpType.T_GATE
        assert isinstance(t_op.is_dagger, bool)
        assert isinstance(t_op.sign, bool)
        assert isinstance(t_op.pauli_string, str)
        assert "X" in t_op.pauli_string or "Z" in t_op.pauli_string

    def test_heisenberg_op_str_repr(self) -> None:
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
        op = hir[0]
        assert "T" in str(op)
        assert "HeisenbergOp" in repr(op)

    def test_heisenberg_op_as_dict(self) -> None:
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
        d: dict[str, Any] = hir[0].as_dict()
        assert d["op_type"] == "T_GATE"
        assert "pauli_string" in d
        assert isinstance(d["is_dagger"], bool)

    def test_measure_op_as_dict_has_meas_idx(self) -> None:
        hir = clifft.trace(clifft.parse("M 0"))
        d: dict[str, Any] = hir[-1].as_dict()
        assert d["op_type"] == "MEASURE"
        assert "meas_record_idx" in d

    def test_hir_as_dict_is_json_serializable(self) -> None:
        hir = clifft.trace(clifft.parse("H 0\nT 0\nCX 0 1\nM 0 1"))
        d: dict[str, Any] = hir.as_dict()
        text = json.dumps(d)
        assert len(text) > 0
        parsed: dict[str, Any] = json.loads(text)
        assert parsed["num_qubits"] == 2
        assert len(parsed["ops"]) == len(hir)

    def test_iter_wrapper_keeps_module_alive(self) -> None:
        """Regression: HeisenbergOp wrappers from __iter__ must keep the
        owning HirModule alive. If the module is collected first, accessing
        the wrapper's mask data would dereference freed arena memory.
        """
        import gc

        def make_op() -> Any:
            hir = clifft.trace(clifft.parse("H 0\nT 0"))
            it = iter(hir)
            op = next(it)
            # Drop local refs; the returned op must hold the HirModule
            # alive on its own.
            return op

        op = make_op()
        gc.collect()
        # Mask access must still work -- if the module were collected,
        # this would hit freed memory.
        assert "X" in op.pauli_string
        assert isinstance(op.sign, bool)
        assert isinstance(op.as_dict(), dict)

    def test_getitem_wrapper_keeps_module_alive(self) -> None:
        """Same regression as test_iter_wrapper_keeps_module_alive but
        through __getitem__ rather than __iter__.
        """
        import gc

        def make_op() -> Any:
            hir = clifft.trace(clifft.parse("H 0\nT 0"))
            return hir[0]

        op = make_op()
        gc.collect()
        assert "X" in op.pauli_string
        assert isinstance(op.as_dict(), dict)


class TestEnumBindings:
    """Verify OpType is accessible."""

    def test_optype_values(self) -> None:
        assert clifft.OpType.T_GATE is not None
        assert clifft.OpType.MEASURE is not None
        assert clifft.OpType.EXP_VAL is not None


class TestEnumBindingCompleteness:
    """Tripwire: detect new C++ enum values not bound in Python.

    The C++ side exposes _num_optypes(), derived from the sentinel enum value. If a new value
    is appended in C++ but not registered in the nanobind enum binding,
    the Python member count will be less than the C++ count.
    """

    def test_all_optypes_bound(self) -> None:
        from clifft._clifft_core import _num_optypes

        py_count = len(clifft.OpType.__members__)
        cpp_count: int = _num_optypes()
        assert py_count == cpp_count, (
            f"OpType mismatch: Python has {py_count} members but C++ has {cpp_count}. "
            "A new OpType was added in hir.h but not bound in bindings.cc."
        )

    def test_all_optypes_format_without_unknown(self) -> None:
        """Every bound OpType must have a real name in op_type_to_str."""
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
        for op in hir:
            d: dict[str, Any] = op.as_dict()
            assert d["op_type"] != "UNKNOWN", f"op_type_to_str returned UNKNOWN for {op}"

    def test_all_gate_types_bound(self) -> None:
        from clifft._clifft_core import _num_gate_types

        py_count = len(clifft.GateType.__members__)
        cpp_count: int = _num_gate_types()
        assert py_count == cpp_count, (
            f"GateType mismatch: Python has {py_count} members but C++ has {cpp_count}. "
            "A new GateType was added in gate_data.h but not bound in bindings.cc."
        )


class TestAstNodeAnnotations:
    """AstNode.tag and annotation GateType binding tests."""

    def test_loss_gate_type_is_bound(self) -> None:
        node = clifft.parse("LOSS(0.1) 0\nM 0").nodes[0]
        assert node.gate == clifft.GateType.LOSS

    def test_leakage_gate_type_is_bound(self) -> None:
        node = clifft.parse("LEAKAGE(0.1) 0\nM 0").nodes[0]
        assert node.gate == clifft.GateType.LEAKAGE

    def test_level_transition_gate_type_is_bound(self) -> None:
        node = clifft.parse("LEVEL_TRANSITION[cz_leak] 0\nM 0").nodes[0]
        assert node.gate == clifft.GateType.LEVEL_TRANSITION

    def test_level_transition_tag_property(self) -> None:
        node = clifft.parse("LEVEL_TRANSITION[cz_leak] 0\nM 0").nodes[0]
        assert node.tag == "cz_leak"

    def test_loss_tag_is_empty(self) -> None:
        node = clifft.parse("LOSS(0.1) 0\nM 0").nodes[0]
        assert node.tag == ""

    def test_leakage_tag_is_empty(self) -> None:
        node = clifft.parse("LEAKAGE(0.1) 0\nM 0").nodes[0]
        assert node.tag == ""

    def test_repr_works_for_annotation_nodes(self) -> None:
        circuit = clifft.parse("LOSS(0.1) 0\nM 0")
        assert repr(circuit.nodes[0]) == "LOSS 0"
        leakage = clifft.parse("LEAKAGE(0.1) 0\nM 0")
        assert repr(leakage.nodes[0]) == "LEAKAGE 0"
        circuit2 = clifft.parse("LEVEL_TRANSITION[cz_leak] 0\nM 0")
        assert repr(circuit2.nodes[0]) == "LEVEL_TRANSITION 0"
