"""Independent phase-sensitive checks of the coherent Clifford reduction."""

import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import stim


@pytest.fixture
def gadget():
    pytest.importorskip("cirq")
    import clifford_gadget

    return clifford_gadget


def matrix(atoms, width):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(width)
    for atom in atoms:
        method = {"T_DAG": "tdg", "S_DAG": "sdg", "S": "s"}.get(atom.name, atom.name.lower())
        getattr(qc, method)(*atom.targets)
    return Operator(qc).data


def pauli_matrix(p):
    from qiskit.quantum_info import Pauli

    label = "".join("IXYZ"[axis] for axis in reversed(list(p)))
    return p.sign * Pauli(label).to_matrix()


def test_noisy_measurement_reset_operator_identity(gadget):
    pytest.importorskip("qiskit")
    atom = gadget.Atom
    width, q = 3, 1
    forward = [atom("T", (0,)), atom("T_DAG", (1,)), atom("T", (2,))]
    backward = [atom("T_DAG", (0,)), atom("T", (1,)), atom("T_DAG", (2,))]
    before = [atom("CX", (0, 1)), atom("CX", (2, 1)), atom("CX", (1, 0))]
    after = list(reversed(before))
    x = stim.PauliString(width)
    x[q] = "X"
    z = stim.PauliString(width)
    z[q] = "Z"
    u = matrix(forward, width)
    identity = np.eye(1 << width)
    # All Pauli choices on the measured/reset wire cover the subtle hidden
    # reset correction. Extra faults on other wires retain correlated effects.
    for pre, middle, post in itertools.product("XYZ", repeat=3):
        lead = before + [atom(pre, (q,)), atom("X", (2,))]
        mid = [atom(middle, (q,)), atom("Y", (0,))]
        tail = [atom(post, (q,)), atom("Z", (2,))] + after
        block = gadget.Gadget(
            width, forward, lead + [atom("MX", (q,))] + mid + [atom("RX", (q,))] + tail, backward
        )
        for bit in (0, 1):
            operators, reset = block.terms(bit)
            measured = (identity + (-1) ** bit * pauli_matrix(x)) / 2
            reset_projector = (identity + (-1) ** reset * pauli_matrix(x)) / 2
            reset_correction = pauli_matrix(z) if reset else identity
            expected = (
                u.conj().T
                @ matrix(tail, width)
                @ reset_correction
                @ reset_projector
                @ matrix(mid, width)
                @ measured
                @ matrix(lead, width)
                @ u
            )
            actual = sum(c * (u.conj().T @ pauli_matrix(p) @ u) for c, p in operators)
            np.testing.assert_allclose(actual, expected, atol=2e-14)


def test_conjugated_pauli_phases_against_aer(gadget):
    aer = pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit

    simulator = aer.AerSimulator(method="unitary", max_parallel_threads=1)
    for axes in itertools.product("IXYZ", repeat=2):
        qc = QuantumCircuit(2)
        qc.t(0)
        qc.tdg(1)
        for q, axis in enumerate(axes):
            if axis != "I":
                getattr(qc, axis.lower())(q)
        qc.tdg(0)
        qc.t(1)
        qc.save_unitary()
        expected = np.asarray(simulator.run(qc).result().data()["unitary"])
        p = stim.PauliString("".join(axes))
        for basis in range(4):
            term = gadget.Term(2)
            for q in range(2):
                if basis >> q & 1:
                    term.gate("X", (q,))
            term.conjugated_pauli(p, {0: 1, 1: -1})
            actual = (
                np.array(
                    [
                        term.ch.inner_product_of_state_and_x(((j & 1) << 1) | (j >> 1))
                        for j in range(4)
                    ]
                )
                * term.coefficient
            )
            np.testing.assert_allclose(actual, expected[:, basis], atol=2e-14)


def test_complete_records_with_readout_and_reset_faults(gadget, tmp_path):
    native = Path("build-study/replay_cultivation").resolve()
    if not native.is_file():
        pytest.skip("build replay_cultivation")
    text = """RX 0 1
T 0
T_DAG 1
CX 0 1
Y 0
MX(1) 1
Y 1
RX 1
Z 0
CX 0 1
T_DAG 0
T 1
MPP Y0*Y1
MX 0 1
"""
    path = tmp_path / "physical.stim"
    path.write_text(gadget.native_replay_text(text))
    width, visible, hidden, atoms = gadget.atoms_from_text(text)
    program = gadget.compile_gadgets(width, atoms)
    assert sum(isinstance(x, gadget.Gadget) for x in program) == 1
    for seed in range(4):
        native_result = json.loads(subprocess.check_output([str(native), str(path), str(seed)]))
        assert native_result["reachable"]
        actual = gadget.evaluate(program, width, list(map(int, native_result["records"])))
        assert actual["log_probability"] == pytest.approx(
            native_result["log_probability"], abs=1e-11
        )
        sampled = gadget.sample(program, width, visible + hidden, seed)
        records = tmp_path / "records.txt"
        records.write_text("".join(map(str, sampled["records"])))
        replay = json.loads(
            subprocess.check_output([str(native), str(path), str(seed), str(records)])
        )
        assert replay["reachable"]
        assert sampled["log_probability"] == pytest.approx(replay["log_probability"], abs=1e-11)


def test_noninverse_clifford_network_declines(gadget):
    atom = gadget.Atom
    with pytest.raises(ValueError, match="not inverses"):
        gadget.Gadget(
            2,
            [atom("T", (0,)), atom("T_DAG", (1,))],
            [atom("CX", (0, 1)), atom("MX", (1,)), atom("RX", (1,))],
            [atom("T_DAG", (0,)), atom("T", (1,))],
        )


def test_boundary_compression_preserves_coherent_phase(gadget):
    state = gadget.CoherentState(3)
    state.terms = []
    for gates, coefficient in [([], 0.2j), (["X"], 0.3), (["H"], 0.4 - 0.1j), (["H", "Z"], -0.2)]:
        term = gadget.Term(3)
        for name in gates:
            term.gate(name, (2,))
        term.coefficient = coefficient
        state.terms.append(term)

    def vector():
        return sum(
            t.coefficient * np.array([t.ch.inner_product_of_state_and_x(i) for i in range(8)])
            for t in state.terms
        )

    expected = vector()
    state.compress_qubit()
    assert len(state.terms) == 2
    np.testing.assert_allclose(vector(), expected, atol=1e-13)


def test_duplicate_inverse_layer_declines(gadget):
    atom = gadget.Atom
    with pytest.raises(ValueError, match="distinct inverses"):
        gadget.Gadget(
            2,
            [atom("T", (0,)), atom("T_DAG", (1,))],
            [atom("MX", (1,)), atom("RX", (1,))],
            [atom("T_DAG", (0,)), atom("T", (1,)), atom("T", (1,))],
        )


def test_genuine_s_instruction_cannot_be_mistaken_for_t(gadget):
    with pytest.raises(ValueError, match="reserves the S proxy"):
        gadget.atoms_from_text("RX 0\nS 0\nMX 0")
