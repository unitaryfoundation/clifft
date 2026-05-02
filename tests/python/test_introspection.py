"""Tests for read-only introspection bindings.

Verifies that HIR operations, VM instructions, and their JSON-friendly
dictionary representations are accessible from Python without copying
entire vectors.
"""

import json
from typing import Any

import pytest

import clifft


class TestHirIntrospection:
    """HIR-level introspection: HeisenbergOp, OpType, iteration."""

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


class TestProgramIntrospection:
    """Program-level introspection: Instruction, Opcode, iteration."""

    def test_program_str_prints_bytecode(self) -> None:
        prog = clifft.compile("H 0\nT 0\nM 0")
        text = str(prog)
        assert "OP_" in text

    def test_program_repr(self) -> None:
        prog = clifft.compile("H 0\nT 0\nM 0")
        r = repr(prog)
        assert "Program" in r
        assert "peak_rank" in r

    def test_program_len(self) -> None:
        prog = clifft.compile("H 0\nT 0\nM 0")
        assert len(prog) == prog.num_instructions

    def test_program_getitem_positive(self) -> None:
        prog = clifft.compile("H 0\nT 0\nM 0")
        inst = prog[0]
        assert isinstance(inst, clifft.Instruction)

    def test_program_getitem_negative(self) -> None:
        prog = clifft.compile("M 0")
        last = prog[-1]
        assert isinstance(last, clifft.Instruction)

    def test_program_getitem_out_of_bounds(self) -> None:
        prog = clifft.compile("M 0")
        with pytest.raises(IndexError):
            _ = prog[999]

    def test_program_iteration(self) -> None:
        prog = clifft.compile("H 0\nT 0\nM 0")
        insts = list(prog)
        assert len(insts) == len(prog)
        for inst in insts:
            assert isinstance(inst, clifft.Instruction)

    def test_instruction_properties(self) -> None:
        prog = clifft.compile("H 0\nT 0\nM 0")
        inst = prog[0]
        assert isinstance(inst.opcode, clifft.Opcode)
        assert isinstance(inst.flags, int)
        assert isinstance(inst.axis_1, int)
        assert isinstance(inst.axis_2, int)

    def test_instruction_str_repr(self) -> None:
        prog = clifft.compile("H 0\nT 0\nM 0")
        inst = prog[0]
        assert "OP_" in str(inst)
        assert "Instruction" in repr(inst)

    def test_instruction_as_dict(self) -> None:
        prog = clifft.compile("H 0\nT 0\nM 0")
        d: dict[str, Any] = prog[0].as_dict()
        assert "opcode" in d
        assert "description" in d
        assert isinstance(d["opcode"], str)

    def test_meas_instruction_as_dict_has_classical_fields(self) -> None:
        prog = clifft.compile("M 0")
        for inst in prog:
            d: dict[str, Any] = inst.as_dict()
            if "MEAS" in d["opcode"]:
                assert "classical_idx" in d
                break

    def test_program_as_dict_is_json_serializable(self) -> None:
        prog = clifft.compile("H 0\nT 0\nCX 0 1\nM 0 1")
        d: dict[str, Any] = prog.as_dict()
        text = json.dumps(d)
        assert len(text) > 0
        parsed: dict[str, Any] = json.loads(text)
        assert parsed["peak_rank"] >= 0
        assert len(parsed["bytecode"]) == len(prog)


class TestSvmBackend:
    """Verify svm_backend() returns a valid ISA string."""

    def test_svm_backend_returns_valid_isa(self) -> None:
        backend = clifft.svm_backend()
        assert backend in ("avx512", "avx2", "scalar")

    def test_svm_backend_respects_force_env(self) -> None:
        """CLIFFT_FORCE_ISA is read at first call; just verify the return is stable."""
        b1 = clifft.svm_backend()
        b2 = clifft.svm_backend()
        assert b1 == b2


class TestEnumBindings:
    """Verify OpType and Opcode enums are accessible."""

    def test_optype_values(self) -> None:
        assert clifft.OpType.T_GATE is not None
        assert clifft.OpType.MEASURE is not None
        assert clifft.OpType.EXP_VAL is not None

    def test_opcode_values(self) -> None:
        assert clifft.Opcode.OP_EXPAND is not None
        assert clifft.Opcode.OP_ARRAY_T is not None
        assert clifft.Opcode.OP_MEAS_ACTIVE_DIAGONAL is not None
        assert clifft.Opcode.OP_POSTSELECT is not None
        assert clifft.Opcode.OP_EXP_VAL is not None


class TestEnumBindingCompleteness:
    """Tripwire: detect new C++ enum values not bound in Python.

    The C++ side exposes _num_optypes() and _num_opcodes() which return
    the count derived from the last sentinel enum value. If a new value
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

    def test_all_opcodes_bound(self) -> None:
        from clifft._clifft_core import _num_opcodes

        py_count = len(clifft.Opcode.__members__)
        cpp_count: int = _num_opcodes()
        assert py_count == cpp_count, (
            f"Opcode mismatch: Python has {py_count} members but C++ has {cpp_count}. "
            "A new Opcode was added in backend.h but not bound in bindings.cc."
        )

    def test_all_optypes_format_without_unknown(self) -> None:
        """Every bound OpType must have a real name in op_type_to_str."""
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
        for op in hir:
            d: dict[str, Any] = op.as_dict()
            assert d["op_type"] != "UNKNOWN", f"op_type_to_str returned UNKNOWN for {op}"

    def test_all_opcodes_format_without_unknown(self) -> None:
        """Every instruction in a compiled program must format with a real name."""
        prog = clifft.compile("H 0\nT 0\nH 1\nCX 0 1\nM 0 1")
        for inst in prog:
            d: dict[str, Any] = inst.as_dict()
            assert d["opcode"] != "UNKNOWN", f"opcode_to_str returned UNKNOWN for {inst}"
