"""Independent channel-level oracles for the standalone native compiler study."""

import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import clifft

_ROOT = Path(__file__).resolve().parents[2]
_BINARY = Path(
    os.environ.get(
        "CLIFFT_CHANNEL_COVARIANCE_STUDY",
        str(_ROOT / "build/channel-covariance-study/channel_covariance_study"),
    )
)
pytestmark = pytest.mark.skipif(not _BINARY.is_file(), reason="standalone native study not built")


def _aer_joint(source: str) -> np.ndarray:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import pauli_error

    n = clifft.parse(source).num_qubits
    count = sum(
        len(line.split()) - 1 for line in source.splitlines() if line.startswith(("M ", "MX "))
    )
    circuit = QuantumCircuit(n + count)
    record = n
    for line in source.splitlines():
        name, *targets = line.split()
        gate = name.split("(")[0]
        args = list(map(float, name.split("(")[1][:-1].split(","))) if "(" in name else []
        if gate in ("H", "S", "S_DAG", "T", "T_DAG"):
            method = {"S_DAG": "sdg", "T_DAG": "tdg"}.get(gate, gate.lower())
            getattr(circuit, method)(list(map(int, targets)))
        elif gate in ("CX", "CZ"):
            for a, b in zip(targets[::2], targets[1::2]):
                control = record + int(a[4:-1]) if a.startswith("rec[") else int(a)
                getattr(circuit, gate.lower())(control, int(b))
        elif gate == "R_Z":
            for target in targets:
                circuit.rz(np.pi * args[0], int(target))
        elif gate in ("M", "MX"):
            for target in targets:
                q = int(target)
                if gate == "MX":
                    circuit.h(q)
                circuit.cx(q, record)
                if gate == "MX":
                    circuit.h(q)
                record += 1
        elif gate in ("DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1"):
            arity = 2 if gate == "DEPOLARIZE2" else 1
            paulis = ["".join(term) for term in itertools.product("IXYZ", repeat=arity)][1:]
            weights = args if gate == "PAULI_CHANNEL_1" else [args[0] / len(paulis)] * len(paulis)
            instruction = pauli_error(
                list(zip(paulis, weights)) + [("I" * arity, 1 - sum(weights))]
            ).to_instruction()
            for index in range(0, len(targets), arity):
                circuit.append(instruction, list(map(int, targets[index : index + arity])))
        else:
            raise AssertionError(gate)
    assert record == n + count
    circuit.save_probabilities(list(range(n, record)), label="joint")
    return np.asarray(AerSimulator(method="density_matrix").run(circuit).result().data(0)["joint"])


def _run(tmp_path: Path, source: str) -> dict[str, Any]:
    path = tmp_path / "circuit.stim"
    path.write_text(source)
    process = subprocess.run(
        [str(_BINARY), str(path), "0", "--joint"], check=True, capture_output=True, text=True
    )
    result: dict[str, Any] = json.loads(process.stdout)
    return result


def test_native_channel_pairing_matches_transfer_eigenvalues() -> None:
    result = subprocess.run(
        [str(_BINARY), "--self-test"], check=True, capture_output=True, text=True
    )
    checks = json.loads(result.stdout)
    assert checks["transfer_checks"] == 5120
    assert checks["multiword_checks"] == 1


@pytest.mark.parametrize(
    "source,covariant",
    [
        ("H 0\nT 0\nDEPOLARIZE1(0.2) 0\nT_DAG 0\nMX 0", True),
        ("H 0\nT 0\nDEPOLARIZE1(0.2) 0\nT 0\nMX 0", True),
        ("H 0 1\nCX 0 1\nT 1\nDEPOLARIZE1(0.2) 1\nT_DAG 1\nCX 0 1\nMX 0\nM 1", True),
        ("H 0 1\nT 0\nDEPOLARIZE2(0.2) 0 1\nT_DAG 0\nMX 0 1", True),
        ("H 0\nT 0\nPAULI_CHANNEL_1(0.07,0.07,0.02) 0\nT_DAG 0\nMX 0", True),
        ("H 0\nT 0\nPAULI_CHANNEL_1(0.03,0.06,0.09) 0\nT_DAG 0\nMX 0", False),
        ("H 0\nR_Z(0.137) 0\nDEPOLARIZE1(0.2) 0\nR_Z(-0.137) 0\nMX 0", True),
        ("H 0 1\nT 0\nCX 1 0\nDEPOLARIZE1(0.2) 1\nCX 1 0\nT_DAG 0\nMX 0 1", False),
        (
            "H 0\nT 0\nDEPOLARIZE1(0.2) 0\nT_DAG 0\nMX 0\n"
            "T 0\nDEPOLARIZE1(0.1) 0\nT_DAG 0\nMX 0",
            True,
        ),
        (
            "H 0\nT 0\nDEPOLARIZE1(0.2) 0\nT_DAG 0\nMX 0\nCX rec[-1] 1\n"
            "T 1\nDEPOLARIZE1(0.1) 1\nT_DAG 1\nM 0 1",
            True,
        ),
    ],
)
def test_channel_covariance_preserves_joint_records_and_fault_count(
    tmp_path: Path, source: str, covariant: bool
) -> None:
    result = _run(tmp_path, source)
    reference = _aer_joint(source)
    by_count = np.asarray(result["joint_by_k"]).reshape(9, -1, len(reference))
    np.testing.assert_allclose(
        by_count[1:], np.broadcast_to(by_count[0], by_count[1:].shape), atol=2e-14, rtol=0
    )
    np.testing.assert_allclose(
        by_count.sum(axis=1), np.broadcast_to(reference, (9, len(reference))), atol=2e-14, rtol=0
    )
    crossings = sum(result["covariance_crossings"][:2])
    assert crossings > 0 if covariant else crossings == 0
    labelled = result["maximum_labelled_fault_difference"][2:6]
    labelled.append(result["maximum_labelled_fault_difference"][7])
    assert max(labelled) < 2e-14


def test_channel_covariance_changes_original_labelled_fault_paths(tmp_path: Path) -> None:
    result = _run(tmp_path, "H 0\nT 0\nDEPOLARIZE1(0.2) 0\nT_DAG 0\nMX 0")
    assert max(result["maximum_labelled_fault_difference"]) > 0.49
