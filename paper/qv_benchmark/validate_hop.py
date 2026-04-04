"""Validate UCC Quantum Volume via Heavy Output Probability and statevector fidelity.

Computes the Heavy Output Probability (HOP) from ideal statevector simulation
and cross-checks UCC statevectors against Qiskit-Aer reference values.

For noiseless simulation of random QV circuits the HOP converges to
(ln(2) + 1) / 2 ~ 0.846 as circuit width grows.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Allow sibling imports when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def compute_hop(statevector: np.ndarray) -> float:
    """Compute the Heavy Output Probability from a dense statevector.

    Steps
    -----
    1. Compute outcome probabilities  p_i = |sv_i|^2  for all 2^N bitstrings.
    2. Find the median probability.
    3. Sum the probabilities of *heavy* outputs (those with p_i > median).

    Parameters
    ----------
    statevector : np.ndarray
        Complex statevector of length 2^N.

    Returns
    -------
    float
        Heavy Output Probability in [0, 1].
    """
    probs: np.ndarray = np.abs(statevector) ** 2
    median: float = float(np.median(probs))
    hop: float = float(np.sum(probs[probs > median]))
    return hop


def validate_ucc(num_qubits: int, seed: int = 42) -> Dict[str, object]:
    """Run a single QV validation round for *num_qubits* qubits.

    Generates a random (unmeasured) QV circuit, simulates it with both UCC and
    Qiskit, then checks statevector fidelity and HOP.

    Parameters
    ----------
    num_qubits : int
        Circuit width (and depth) of the QV circuit.
    seed : int
        PRNG seed handed to the QV circuit generator.

    Returns
    -------
    dict
        Keys: ``num_qubits``, ``seed``, ``fidelity``, ``hop``, ``pass``.
        ``pass`` is *True* when fidelity > 0.999 **and** HOP > 0.70.
    """
    # -- generate QASM --------------------------------------------------
    from generator import generate_qv_qasm_unmeasured

    qasm: str = generate_qv_qasm_unmeasured(num_qubits, seed=seed)

    # -- UCC statevector ------------------------------------------------
    from qasm_adapter import to_ucc_stim

    import ucc

    stim: str = to_ucc_stim(qasm)
    prog = ucc.compile(stim)
    state = ucc.State(
        prog.peak_rank,
        prog.num_measurements,
        prog.num_detectors,
        prog.num_observables,
        seed=42,
    )
    ucc.execute(prog, state)
    sv_ucc: np.ndarray = ucc.get_statevector(prog, state)

    # -- Qiskit reference statevector -----------------------------------
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qkc: QuantumCircuit = QuantumCircuit.from_qasm_str(qasm)
    sv_qk: np.ndarray = Statevector.from_instruction(qkc).data

    # -- fidelity: |<sv_ucc|sv_qk>|^2 ----------------------------------
    fidelity: float = float(np.abs(np.vdot(sv_ucc, sv_qk)) ** 2)

    # -- HOP from UCC statevector ---------------------------------------
    hop: float = compute_hop(sv_ucc)

    passed: bool = fidelity > 0.999 and hop > 0.70

    return {
        "num_qubits": num_qubits,
        "seed": seed,
        "fidelity": fidelity,
        "hop": hop,
        "pass": passed,
    }


def _main() -> None:
    """Run validation for several qubit counts and report results."""
    qubit_counts: List[int] = [4, 6, 8]
    results: List[Dict[str, object]] = []

    for n in qubit_counts:
        print(f"Validating N={n} ... ", end="", flush=True)
        res = validate_ucc(n, seed=42)
        results.append(res)
        status = "PASS" if res["pass"] else "FAIL"
        print(f"{status}  fidelity={res['fidelity']:.6f}  HOP={res['hop']:.4f}")

    any_failed: bool = any(not r["pass"] for r in results)
    if any_failed:
        print("\nSome validations FAILED.")
        sys.exit(1)
    else:
        print("\nAll validations passed.")


if __name__ == "__main__":
    _main()
