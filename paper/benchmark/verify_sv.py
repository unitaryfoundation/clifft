"""Statevector equivalence verification: UCC vs Qiskit-Aer.

Verifies that UCC's factored state expansion matches Qiskit's dense
statevector up to global phase for the benchmark generator circuits.
This is Task 1.2 from the performance comparison plan.

Usage:
    python verify_sv.py           # default: N in [4, 6, 8, 10]
    python verify_sv.py --max-n 8 # limit max qubit count
"""

import argparse
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from generator import generate_boundary_circuit

FIDELITY_THRESHOLD = 0.9999


def ucc_statevector(stim_text: str) -> npt.NDArray[np.complex128]:
    """Compile and execute a stim circuit in UCC, return dense statevector."""
    import ucc

    # Strip measurement lines -- we only want the unitary part
    lines = [line for line in stim_text.strip().split("\n") if not line.startswith("M ")]
    prog = ucc.compile("\n".join(lines))
    state = ucc.State(
        prog.peak_rank,
        prog.num_measurements,
        prog.num_detectors,
        prog.num_observables,
        seed=42,
    )
    ucc.execute(prog, state)
    sv: npt.NDArray[np.complex128] = ucc.get_statevector(prog, state)
    return sv


def qiskit_statevector(qasm_path: str) -> npt.NDArray[np.complex128]:
    """Run a QASM circuit in Qiskit-Aer, return dense statevector."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit.from_qasm_file(qasm_path)
    qc.remove_final_measurements()
    qc.save_statevector()
    sim = AerSimulator(method="statevector", max_parallel_threads=1)
    result = sim.run(qc).result()
    sv: npt.NDArray[np.complex128] = np.array(result.get_statevector(qc))
    return sv


def state_fidelity(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> float:
    """Compute |<a|b>| fidelity (accounts for global phase difference)."""
    return float(np.abs(np.vdot(a, b)))


def verify(
    max_n: int = 10,
    t_values: list[int] | None = None,
    k_values: list[int] | None = None,
) -> bool:
    """Run statevector equivalence checks across a parameter grid."""
    if t_values is None:
        t_values = [2, 4, 8]
    if k_values is None:
        k_values = [2, 3, 4]

    n_values = list(range(4, max_n + 1, 2))
    total = 0
    passed = 0
    failed_cases: list[str] = []

    print(f"Verifying UCC vs Qiskit statevectors (max N={max_n})")
    print(f"N values: {n_values}")
    print(f"t values: {t_values}")
    print(f"k values: {k_values}")
    print(f"Fidelity threshold: {FIDELITY_THRESHOLD}")
    print("-" * 60)

    for n_val in n_values:
        for t_val in t_values:
            for k_val in k_values:
                actual_k = min(k_val, n_val)
                label = f"N={n_val:2d} t={t_val:2d} k={actual_k:2d}"
                total += 1

                base = f"/tmp/verify_N{n_val}_t{t_val}_k{actual_k}"
                stim_f, qasm_f = generate_boundary_circuit(n_val, t_val, actual_k, base)

                with open(stim_f) as f:
                    stim_text = f.read()

                try:
                    ucc_sv = ucc_statevector(stim_text)
                    qiskit_sv = qiskit_statevector(qasm_f)
                    f_val = state_fidelity(ucc_sv, qiskit_sv)

                    if f_val >= FIDELITY_THRESHOLD:
                        passed += 1
                        print(f"  PASS  {label}: fidelity={f_val:.8f}")
                    else:
                        failed_cases.append(f"{label}: fidelity={f_val:.8f}")
                        print(f"  FAIL  {label}: fidelity={f_val:.8f}")
                except Exception as e:
                    failed_cases.append(f"{label}: {e}")
                    print(f"  ERROR {label}: {e}")

    print("-" * 60)
    print(f"Results: {passed}/{total} passed")

    if failed_cases:
        print("\nFailed cases:")
        for case in failed_cases:
            print(f"  - {case}")
        return False

    print("All statevector equivalence checks passed.")
    return True


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Verify UCC vs Qiskit statevectors")
    parser.add_argument(
        "--max-n",
        type=int,
        default=10,
        help="Maximum qubit count to test (default: 10, hard limit: 10)",
    )
    args = parser.parse_args()

    max_n = min(args.max_n, 10)  # get_statevector limited to 10 qubits
    success = verify(max_n=max_n)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
