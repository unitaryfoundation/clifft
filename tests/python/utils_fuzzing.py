"""Memory-bounded procedural circuit generators for optimizer invariant testing.

Each generator produces circuits with specific topological patterns that
trigger particular optimizer passes, while strictly bounding the active
dimension (k_max) by restricting T-gate spread to a small qubit subset.
"""

import numpy as np

# Maximum number of qubits that receive T gates in any generator.
# This bounds k_max to keep memory usage trivial.
_T_QUBIT_LIMIT = 4


def generate_commutation_gauntlet(num_qubits: int, depth: int, seed: int) -> str:
    """Generate a circuit with deep commutation/anti-commutation chains.

    Interleaves T/T_DAG gates (restricted to qubits 0..3) with deep chains
    of commuting ops (Z, CZ, M) and anti-commuting ops (X, H) across the
    full register. Tests the HIR pass manager's Pauli commutation tracking.

    Memory bound: k_max <= _T_QUBIT_LIMIT since only qubits 0..3 get T gates.

    Args:
        num_qubits: Total number of qubits in the circuit.
        depth: Number of gate layers to generate.
        seed: PRNG seed for reproducibility.

    Returns:
        Circuit string in .stim format.
    """
    rng = np.random.default_rng(seed)
    lines: list[str] = []
    t_qubits = min(num_qubits, _T_QUBIT_LIMIT)

    for _ in range(depth):
        r = rng.random()

        if r < 0.15:
            # T or T_DAG on a T-eligible qubit
            q = int(rng.integers(0, t_qubits))
            gate = "T" if rng.random() < 0.5 else "T_DAG"
            lines.append(f"{gate} {q}")

        elif r < 0.35:
            # Commuting diagonal: Z on random qubit
            q = int(rng.integers(0, num_qubits))
            lines.append(f"Z {q}")

        elif r < 0.50 and num_qubits > 1:
            # Commuting two-qubit diagonal: CZ
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            lines.append(f"CZ {q1} {q2}")

        elif r < 0.65:
            # Anti-commuting: X or H on random qubit
            q = int(rng.integers(0, num_qubits))
            gate = "X" if rng.random() < 0.5 else "H"
            lines.append(f"{gate} {q}")

        elif r < 0.80 and num_qubits > 1:
            # CX to spread entanglement
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            lines.append(f"CX {q1} {q2}")

        else:
            # Measure + reset a non-T qubit to flush active state
            if num_qubits > t_qubits:
                q = int(rng.integers(t_qubits, num_qubits))
                lines.append(f"M {q}")
                lines.append(f"R {q}")

    # Final measurement of all qubits
    all_qubits = " ".join(str(i) for i in range(num_qubits))
    lines.append(f"M {all_qubits}")

    return "\n".join(lines)


def generate_star_graph_honeypot(num_qubits: int, depth: int, seed: int) -> str:
    """Generate circuits with star-graph CX/CZ patterns to trigger MultiGatePass.

    Forces contiguous CX gates sharing a target and contiguous CZ gates
    sharing a control, which are the exact patterns MultiGatePass fuses.
    Sprinkles DEPOLARIZE1 blocks to test NoiseBlockPass batching.

    Memory bound: T gates restricted to qubits 0..3, plus periodic measurement
    flushes to keep k_max <= 10.

    Args:
        num_qubits: Total number of qubits in the circuit.
        depth: Number of gate layers to generate.
        seed: PRNG seed for reproducibility.

    Returns:
        Circuit string in .stim format.
    """
    rng = np.random.default_rng(seed)
    lines: list[str] = []
    t_qubits = min(num_qubits, _T_QUBIT_LIMIT)
    layers_since_flush = 0

    for _ in range(depth):
        r = rng.random()

        if r < 0.10:
            # T gate on eligible qubit
            q = int(rng.integers(0, t_qubits))
            lines.append(f"T {q}")

        elif r < 0.35 and num_qubits > 2:
            # Star-graph CX burst: multiple controls -> same target
            target = int(rng.integers(0, num_qubits))
            controls = [i for i in range(num_qubits) if i != target]
            rng.shuffle(controls)
            fan_in = min(int(rng.integers(2, min(6, len(controls) + 1))), len(controls))
            for c in controls[:fan_in]:
                lines.append(f"CX {c} {target}")

        elif r < 0.55 and num_qubits > 2:
            # Star-graph CZ burst: same control -> multiple targets
            control = int(rng.integers(0, num_qubits))
            targets = [i for i in range(num_qubits) if i != control]
            rng.shuffle(targets)
            fan_out = min(int(rng.integers(2, min(6, len(targets) + 1))), len(targets))
            for t in targets[:fan_out]:
                lines.append(f"CZ {control} {t}")

        elif r < 0.70:
            # DEPOLARIZE1 block on a batch of qubits
            batch_size = min(int(rng.integers(2, min(8, num_qubits + 1))), num_qubits)
            batch = rng.choice(num_qubits, size=batch_size, replace=False)
            targets_str = " ".join(str(int(q)) for q in sorted(batch))
            p = float(rng.uniform(0.001, 0.05))
            lines.append(f"DEPOLARIZE1({p:.4f}) {targets_str}")

        elif r < 0.85:
            # H or S gate
            q = int(rng.integers(0, num_qubits))
            gate = "H" if rng.random() < 0.5 else "S"
            lines.append(f"{gate} {q}")

        else:
            # Single-qubit Clifford
            q = int(rng.integers(0, num_qubits))
            lines.append(f"X {q}")

        layers_since_flush += 1

        # Periodic flush: measure and reset all qubits to bound k_max
        if layers_since_flush >= depth // 5 + 10:
            all_q = " ".join(str(i) for i in range(num_qubits))
            lines.append(f"M {all_q}")
            lines.append(f"R {all_q}")
            layers_since_flush = 0

    # Final measurement
    all_qubits = " ".join(str(i) for i in range(num_qubits))
    lines.append(f"M {all_qubits}")

    return "\n".join(lines)


# Maximum total T-gates in the uncomputation ladder's forward pass.
# Each unique T-gate can expand active_k by 1 in the unoptimized baseline,
# so this directly bounds peak_rank.
_LADDER_MAX_T_GATES = 10


def generate_uncomputation_ladder(
    num_qubits: int, depth: int, seed: int, *, noise_prob: float = 0.01
) -> str:
    """Generate a Clifford+T forward pass followed by its exact inverse.

    Builds an entangling Clifford+T sequence with bounded T-count, then
    appends the analytical inverse (reversed order, daggers applied).
    Injects noise gates at random points to create stochastic trajectories.

    Without noise, all final measurements must deterministically yield 0
    since U * U_dag = I. With noise, both optimized and unoptimized paths
    must produce identical stochastic records.

    Memory bound: T gates restricted to qubits 0..3 AND total T-count
    capped at _LADDER_MAX_T_GATES to keep peak_rank bounded.

    Args:
        num_qubits: Total number of qubits in the circuit.
        depth: Number of gate layers in the forward pass.
        seed: PRNG seed for reproducibility.
        noise_prob: Probability of DEPOLARIZE1 noise per injection point.

    Returns:
        Circuit string in .stim format.
    """
    rng = np.random.default_rng(seed)
    t_qubits = min(num_qubits, _T_QUBIT_LIMIT)
    t_gates_emitted = 0

    # Dagger map for inverting gates
    dagger_map: dict[str, str] = {
        "H": "H",
        "X": "X",
        "Y": "Y",
        "Z": "Z",
        "S": "S_DAG",
        "S_DAG": "S",
        "T": "T_DAG",
        "T_DAG": "T",
    }

    # Build forward pass, recording gates for inversion
    forward_gates: list[str] = []
    noise_lines: list[tuple[int, str]] = []  # (position, noise_line)

    for _ in range(depth):
        r = rng.random()

        if r < 0.15 and t_gates_emitted < _LADDER_MAX_T_GATES:
            # T/T_DAG on eligible qubit
            q = int(rng.integers(0, t_qubits))
            gate = "T" if rng.random() < 0.5 else "T_DAG"
            forward_gates.append(f"{gate} {q}")
            t_gates_emitted += 1

        elif r < 0.35 and num_qubits > 1:
            # CX
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            forward_gates.append(f"CX {q1} {q2}")

        elif r < 0.50 and num_qubits > 1:
            # CZ (self-inverse)
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            forward_gates.append(f"CZ {q1} {q2}")

        elif r < 0.70:
            # H (self-inverse)
            q = int(rng.integers(0, num_qubits))
            forward_gates.append(f"H {q}")

        elif r < 0.85:
            # S / S_DAG
            q = int(rng.integers(0, num_qubits))
            gate = "S" if rng.random() < 0.5 else "S_DAG"
            forward_gates.append(f"{gate} {q}")

        else:
            # X (self-inverse)
            q = int(rng.integers(0, num_qubits))
            forward_gates.append(f"X {q}")

        # Possibly inject noise at this position
        if rng.random() < 0.1:
            nq = int(rng.integers(0, num_qubits))
            noise_lines.append((len(forward_gates), f"DEPOLARIZE1({noise_prob}) {nq}"))

    # Build reverse pass by inverting each gate
    reverse_gates: list[str] = []
    for gate_line in reversed(forward_gates):
        parts = gate_line.split()
        gate_name = parts[0]
        qubits = parts[1:]

        if gate_name in ("CX", "CY", "CZ"):
            # Two-qubit Cliffords: CX^dag = CX, CZ^dag = CZ, CY^dag = CY
            reverse_gates.append(gate_line)
        elif gate_name in dagger_map:
            reverse_gates.append(f"{dagger_map[gate_name]} {' '.join(qubits)}")
        else:
            reverse_gates.append(gate_line)

    # Assemble: forward + noise + reverse + final measurements
    lines: list[str] = []

    # Emit forward gates with noise injected at recorded positions
    noise_idx = 0
    for i, gate_line in enumerate(forward_gates):
        lines.append(gate_line)
        while noise_idx < len(noise_lines) and noise_lines[noise_idx][0] == i + 1:
            lines.append(noise_lines[noise_idx][1])
            noise_idx += 1

    # Emit remaining noise
    while noise_idx < len(noise_lines):
        lines.append(noise_lines[noise_idx][1])
        noise_idx += 1

    # Reverse pass
    lines.extend(reverse_gates)

    # Final measurement of all qubits
    all_qubits = " ".join(str(i) for i in range(num_qubits))
    lines.append(f"M {all_qubits}")

    return "\n".join(lines)
