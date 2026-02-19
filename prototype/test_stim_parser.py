"""Tests for stim_parser.py — .stim text → op-list conversion."""

import pytest
from stim_parser import stim_circuit_to_ops, stim_file_to_ops, ops_summary
from pathlib import Path


class TestBasicParsing:
    def test_single_qubit_gates(self):
        ops = stim_circuit_to_ops("H 0 1 2")
        assert ops == [("H", [0]), ("H", [1]), ("H", [2])]

    def test_two_qubit_gates(self):
        ops = stim_circuit_to_ops("CX 0 1 2 3")
        assert ops == [("CX", [0, 1]), ("CX", [2, 3])]

    def test_t_gates(self):
        ops = stim_circuit_to_ops("T 0 1\nT_DAG 2")
        assert ops == [("T", [0]), ("T", [1]), ("T_DAG", [2])]

    def test_measurements(self):
        ops = stim_circuit_to_ops("M 0 1\nMX 2 3")
        assert ops == [("M", [0]), ("M", [1]), ("MX", [2]), ("MX", [3])]

    def test_measurement_with_noise(self):
        ops = stim_circuit_to_ops("M(0.01) 0 1")
        assert ops == [("M", [0]), ("M", [1])]

    def test_reset_gates(self):
        ops = stim_circuit_to_ops("R 0\nRX 1 2")
        assert ops == [("R", [0]), ("RX", [1]), ("RX", [2])]

    def test_cnot_alias(self):
        ops = stim_circuit_to_ops("CNOT 0 1")
        assert ops == [("CX", [0, 1])]


class TestAnnotations:
    def test_skip_annotations_default(self):
        ops = stim_circuit_to_ops("H 0\nTICK\nDETECTOR rec[-1]\nH 1")
        assert ops == [("H", [0]), ("H", [1])]

    def test_keep_annotations(self):
        ops = stim_circuit_to_ops("H 0\nTICK\nH 1", skip_annotations=False)
        names = [name for name, _ in ops]
        assert "TICK" in names

    def test_qubit_coords_skipped(self):
        ops = stim_circuit_to_ops("QUBIT_COORDS(0, 0) 0\nH 0")
        assert ops == [("H", [0])]

    def test_shift_coords_skipped(self):
        ops = stim_circuit_to_ops("SHIFT_COORDS(0, 0, 1)\nH 0")
        assert ops == [("H", [0])]

    def test_observable_include_skipped(self):
        ops = stim_circuit_to_ops("OBSERVABLE_INCLUDE(0) rec[-1]\nH 0")
        assert ops == [("H", [0])]


class TestNoise:
    def test_noise_included_by_default(self):
        ops = stim_circuit_to_ops("DEPOLARIZE1(0.01) 0 1")
        assert len(ops) == 2
        assert ops[0] == ("DEPOLARIZE1", [0, 0.01])
        assert ops[1] == ("DEPOLARIZE1", [1, 0.01])

    def test_noise_skipped(self):
        ops = stim_circuit_to_ops("DEPOLARIZE1(0.01) 0 1", skip_noise=True)
        assert ops == []

    def test_depolarize2_pairs(self):
        ops = stim_circuit_to_ops("DEPOLARIZE2(0.01) 0 1 2 3")
        assert ops == [
            ("DEPOLARIZE2", [0, 1, 0.01]),
            ("DEPOLARIZE2", [2, 3, 0.01]),
        ]

    def test_x_error(self):
        ops = stim_circuit_to_ops("X_ERROR(0.001) 0", skip_noise=False)
        assert ops == [("X_ERROR", [0, 0.001])]

    def test_z_error(self):
        ops = stim_circuit_to_ops("Z_ERROR(0.001) 5", skip_noise=False)
        assert ops == [("Z_ERROR", [5, 0.001])]


class TestMPP:
    def test_single_product(self):
        ops = stim_circuit_to_ops("MPP X0*X1*X2")
        assert len(ops) == 1
        assert ops[0] == ("MPP", [[("X", 0), ("X", 1), ("X", 2)]])

    def test_multiple_products(self):
        ops = stim_circuit_to_ops("MPP X0*X1 Z2*Z3")
        assert len(ops) == 2
        assert ops[0] == ("MPP", [[("X", 0), ("X", 1)]])
        assert ops[1] == ("MPP", [[("Z", 2), ("Z", 3)]])

    def test_mixed_pauli_product(self):
        ops = stim_circuit_to_ops("MPP Y0*Y3*Y7*Y9")
        assert ops[0][0] == "MPP"
        product = ops[0][1][0]
        assert all(p == "Y" for p, _ in product)
        assert [q for _, q in product] == [0, 3, 7, 9]

    def test_mpp_with_noise(self):
        ops = stim_circuit_to_ops("MPP(0.001) X0*Z1")
        assert len(ops) == 1
        assert ops[0] == ("MPP", [[("X", 0), ("Z", 1)]])


class TestRepeat:
    def test_repeat_block(self):
        text = """REPEAT 3 {
            H 0
            M 0
        }"""
        ops = stim_circuit_to_ops(text)
        assert len(ops) == 6  # 3 * (H + M)
        assert ops == [("H", [0]), ("M", [0])] * 3

    def test_nested_repeat(self):
        text = """REPEAT 2 {
            REPEAT 3 {
                H 0
            }
        }"""
        ops = stim_circuit_to_ops(text)
        assert len(ops) == 6
        assert all(op == ("H", [0]) for op in ops)


class TestTargetCircuit:
    """Integration test with the actual target circuit."""

    @pytest.fixture
    def circuit_path(self):
        p = Path(__file__).resolve().parent.parent / "circuits" / "target.stim"
        if not p.exists():
            pytest.skip("target circuit not found")
        return str(p)

    def test_parses_without_error(self, circuit_path):
        n, ops = stim_file_to_ops(circuit_path, skip_noise=True)
        assert n == 15
        assert len(ops) > 100

    def test_gate_coverage(self, circuit_path):
        n, ops = stim_file_to_ops(circuit_path, skip_noise=False)
        counts = ops_summary(ops)
        assert "T" in counts
        assert "T_DAG" in counts
        assert "MPP" in counts
        assert "MX" in counts
        assert "RX" in counts
        assert "CX" in counts

    def test_no_noise_reduces_ops(self, circuit_path):
        _, ops_noisy = stim_file_to_ops(circuit_path, skip_noise=False)
        _, ops_clean = stim_file_to_ops(circuit_path, skip_noise=True)
        assert len(ops_clean) < len(ops_noisy)
