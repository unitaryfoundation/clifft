import os


def generate_boundary_circuit(N: int, t: int, k_target: int, base_filename: str) -> tuple[str, str]:
    """Generate a circuit that mathematically isolates N, t, and k.

    Outputs both .stim and .qasm files. Returns the paths to both.
    """
    stim_lines: list[str] = []
    qasm_lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{N}];",
        f"creg c[{N}];",
    ]

    if N < 1 or (t > 0 and k_target < 1):
        raise ValueError(f"Invalid params: N={N}, t={t}, k_target={k_target}")

    actual_k = min(k_target, N)

    # Hadamard the active core to create superposition on qubits 0..k-1
    for i in range(actual_k):
        stim_lines.append(f"H {i}")
        qasm_lines.append(f"h q[{i}];")

    # Inject t T-gates confined to the active core, interleaved with
    # Cliffords to prevent trivial ZX-calculus cancellations by tsim.
    for i in range(t):
        target_q = i % actual_k
        stim_lines.append(f"T {target_q}")
        qasm_lines.append(f"t q[{target_q}];")

        stim_lines.append(f"H {target_q}")
        qasm_lines.append(f"h q[{target_q}];")

        if actual_k > 1:
            next_q = (target_q + 1) % actual_k
            stim_lines.append(f"CX {target_q} {next_q}")
            qasm_lines.append(f"cx q[{target_q}], q[{next_q}];")

    # Global Clifford padding forces Qiskit to track 2^N, but UCC absorbs
    # this into its offline frame so the active array stays at 2^k.
    for i in range(actual_k, N):
        stim_lines.append(f"H {i}")
        qasm_lines.append(f"h q[{i}];")

    for i in range(N - 1):
        stim_lines.append(f"CX {i} {i + 1}")
        qasm_lines.append(f"cx q[{i}], q[{i + 1}];")

    # Measure all qubits
    for i in range(N):
        stim_lines.append(f"M {i}")
        qasm_lines.append(f"measure q[{i}] -> c[{i}];")

    dirname = os.path.dirname(base_filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    stim_path = f"{base_filename}.stim"
    qasm_path = f"{base_filename}.qasm"

    with open(stim_path, "w") as f:
        f.write("\n".join(stim_lines) + "\n")

    with open(qasm_path, "w") as f:
        f.write("\n".join(qasm_lines) + "\n")

    return stim_path, qasm_path


if __name__ == "__main__":
    generate_boundary_circuit(N=10, t=5, k_target=3, base_filename="circuits/test_circuit")
    print("Test circuit generated in circuits/")
