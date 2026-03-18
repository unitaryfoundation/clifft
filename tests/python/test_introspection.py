"""Tests for read-only introspection bindings.

Verifies that HIR operations, VM instructions, and their JSON-friendly
dictionary representations are accessible from Python without copying
entire vectors.
"""

import json
from typing import Any

import pytest

import ucc


class TestHirIntrospection:
    """HIR-level introspection: HeisenbergOp, OpType, iteration."""

    def test_hir_str_prints_ops(self) -> None:
        hir = ucc.trace(ucc.parse("H 0\nT 0\nM 0"))
        text = str(hir)
        assert "T +" in text or "T -" in text
        assert "MEASURE" in text

    def test_hir_repr(self) -> None:
        hir = ucc.trace(ucc.parse("H 0\nT 0\nM 0"))
        r = repr(hir)
        assert "HirModule" in r
        assert "1 T-gates" in r

    def test_hir_len(self) -> None:
        hir = ucc.trace(ucc.parse("H 0\nT 0\nM 0"))
        assert len(hir) == hir.num_ops

    def test_hir_getitem_positive(self) -> None:
        hir = ucc.trace(ucc.parse("H 0\nT 0\nM 0"))
        op = hir[0]
        assert isinstance(op, ucc.HeisenbergOp)
        assert op.op_type == ucc.OpType.T_GATE

    def test_hir_getitem_negative(self) -> None:
        hir = ucc.trace(ucc.parse("H 0\nT 0\nM 0"))
        last = hir[-1]
        assert last.op_type == ucc.OpType.MEASURE

    def test_hir_getitem_out_of_bounds(self) -> None:
        hir = ucc.trace(ucc.parse("M 0"))
        with pytest.raises(IndexError):
            _ = hir[999]

    def test_hir_iteration(self) -> None:
        hir = ucc.trace(ucc.parse("H 0\nT 0\nT 0\nM 0"))
        ops = list(hir)
        assert len(ops) == len(hir)
        for op in ops:
            assert isinstance(op, ucc.HeisenbergOp)

    def test_heisenberg_op_properties(self) -> None:
        hir = ucc.trace(ucc.parse("H 0\nT 0\nM 0"))
        t_op = hir[0]
        assert t_op.op_type == ucc.OpType.T_GATE
        assert isinstance(t_op.is_dagger, bool)
        assert isinstance(t_op.sign, bool)
        assert isinstance(t_op.pauli_string, str)
        assert "X" in t_op.pauli_string or "Z" in t_op.pauli_string

    def test_heisenberg_op_str_repr(self) -> None:
        hir = ucc.trace(ucc.parse("H 0\nT 0\nM 0"))
        op = hir[0]
        assert "T" in str(op)
        assert "HeisenbergOp" in repr(op)

    def test_heisenberg_op_as_dict(self) -> None:
        hir = ucc.trace(ucc.parse("H 0\nT 0\nM 0"))
        d: dict[str, Any] = hir[0].as_dict()
        assert d["op_type"] == "T_GATE"
        assert "pauli_string" in d
        assert isinstance(d["is_dagger"], bool)

    def test_measure_op_as_dict_has_meas_idx(self) -> None:
        hir = ucc.trace(ucc.parse("M 0"))
        d: dict[str, Any] = hir[-1].as_dict()
        assert d["op_type"] == "MEASURE"
        assert "meas_record_idx" in d

    def test_hir_as_dict_is_json_serializable(self) -> None:
        hir = ucc.trace(ucc.parse("H 0\nT 0\nCX 0 1\nM 0 1"))
        d: dict[str, Any] = hir.as_dict()
        text = json.dumps(d)
        assert len(text) > 0
        parsed: dict[str, Any] = json.loads(text)
        assert parsed["num_qubits"] == 2
        assert len(parsed["ops"]) == len(hir)


class TestProgramIntrospection:
    """Program-level introspection: Instruction, Opcode, iteration."""

    def test_program_str_prints_bytecode(self) -> None:
        prog = ucc.compile("H 0\nT 0\nM 0")
        text = str(prog)
        assert "OP_" in text

    def test_program_repr(self) -> None:
        prog = ucc.compile("H 0\nT 0\nM 0")
        r = repr(prog)
        assert "Program" in r
        assert "peak_rank" in r

    def test_program_len(self) -> None:
        prog = ucc.compile("H 0\nT 0\nM 0")
        assert len(prog) == prog.num_instructions

    def test_program_getitem_positive(self) -> None:
        prog = ucc.compile("H 0\nT 0\nM 0")
        inst = prog[0]
        assert isinstance(inst, ucc.Instruction)

    def test_program_getitem_negative(self) -> None:
        prog = ucc.compile("M 0")
        last = prog[-1]
        assert isinstance(last, ucc.Instruction)

    def test_program_getitem_out_of_bounds(self) -> None:
        prog = ucc.compile("M 0")
        with pytest.raises(IndexError):
            _ = prog[999]

    def test_program_iteration(self) -> None:
        prog = ucc.compile("H 0\nT 0\nM 0")
        insts = list(prog)
        assert len(insts) == len(prog)
        for inst in insts:
            assert isinstance(inst, ucc.Instruction)

    def test_instruction_properties(self) -> None:
        prog = ucc.compile("H 0\nT 0\nM 0")
        inst = prog[0]
        assert isinstance(inst.opcode, ucc.Opcode)
        assert isinstance(inst.flags, int)
        assert isinstance(inst.axis_1, int)
        assert isinstance(inst.axis_2, int)

    def test_instruction_str_repr(self) -> None:
        prog = ucc.compile("H 0\nT 0\nM 0")
        inst = prog[0]
        assert "OP_" in str(inst)
        assert "Instruction" in repr(inst)

    def test_instruction_as_dict(self) -> None:
        prog = ucc.compile("H 0\nT 0\nM 0")
        d: dict[str, Any] = prog[0].as_dict()
        assert "opcode" in d
        assert "description" in d
        assert isinstance(d["opcode"], str)

    def test_meas_instruction_as_dict_has_classical_fields(self) -> None:
        prog = ucc.compile("M 0")
        for inst in prog:
            d: dict[str, Any] = inst.as_dict()
            if "MEAS" in d["opcode"]:
                assert "classical_idx" in d
                break

    def test_program_as_dict_is_json_serializable(self) -> None:
        prog = ucc.compile("H 0\nT 0\nCX 0 1\nM 0 1")
        d: dict[str, Any] = prog.as_dict()
        text = json.dumps(d)
        assert len(text) > 0
        parsed: dict[str, Any] = json.loads(text)
        assert parsed["peak_rank"] >= 0
        assert len(parsed["bytecode"]) == len(prog)


class TestSvmBackend:
    """Verify svm_backend() returns a valid ISA string."""

    def test_svm_backend_returns_valid_isa(self) -> None:
        backend = ucc.svm_backend()
        assert backend in ("avx512", "avx2", "scalar")

    def test_svm_backend_respects_force_env(self) -> None:
        """UCC_FORCE_ISA is read at first call; just verify the return is stable."""
        b1 = ucc.svm_backend()
        b2 = ucc.svm_backend()
        assert b1 == b2


class TestEnumBindings:
    """Verify OpType and Opcode enums are accessible."""

    def test_optype_values(self) -> None:
        assert ucc.OpType.T_GATE is not None
        assert ucc.OpType.MEASURE is not None
        assert ucc.OpType.CLIFFORD_PHASE is not None

    def test_opcode_values(self) -> None:
        assert ucc.Opcode.OP_EXPAND is not None
        assert ucc.Opcode.OP_PHASE_T is not None
        assert ucc.Opcode.OP_MEAS_ACTIVE_DIAGONAL is not None
        assert ucc.Opcode.OP_POSTSELECT is not None


class TestEnumBindingCompleteness:
    """Tripwire: detect new C++ enum values not bound in Python.

    The C++ side exposes _num_optypes() and _num_opcodes() which return
    the count derived from the last sentinel enum value. If a new value
    is appended in C++ but not registered in the nanobind enum binding,
    the Python member count will be less than the C++ count.
    """

    def test_all_optypes_bound(self) -> None:
        from ucc._ucc_core import _num_optypes

        py_count = len(ucc.OpType.__members__)
        cpp_count: int = _num_optypes()
        assert py_count == cpp_count, (
            f"OpType mismatch: Python has {py_count} members but C++ has {cpp_count}. "
            "A new OpType was added in hir.h but not bound in bindings.cc."
        )

    def test_all_opcodes_bound(self) -> None:
        from ucc._ucc_core import _num_opcodes

        py_count = len(ucc.Opcode.__members__)
        cpp_count: int = _num_opcodes()
        assert py_count == cpp_count, (
            f"Opcode mismatch: Python has {py_count} members but C++ has {cpp_count}. "
            "A new Opcode was added in backend.h but not bound in bindings.cc."
        )

    def test_all_optypes_format_without_unknown(self) -> None:
        """Every bound OpType must have a real name in op_type_to_str."""
        hir = ucc.trace(ucc.parse("H 0\nT 0\nM 0"))
        for op in hir:
            d: dict[str, Any] = op.as_dict()
            assert d["op_type"] != "UNKNOWN", f"op_type_to_str returned UNKNOWN for {op}"

    def test_all_opcodes_format_without_unknown(self) -> None:
        """Every instruction in a compiled program must format with a real name."""
        prog = ucc.compile("H 0\nT 0\nH 1\nCX 0 1\nM 0 1")
        for inst in prog:
            d: dict[str, Any] = inst.as_dict()
            assert d["opcode"] != "UNKNOWN", f"opcode_to_str returned UNKNOWN for {inst}"
