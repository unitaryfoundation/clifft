"""Clifford Proxy Pass: compute upper bound on |v| by analyzing the circuit.

Uses the GF(2) shift-vector rank to bound the number of non-zero entries
in the generalized stabilizer coefficient vector. This is a fast
polynomial-time pre-pass that enables precise buffer pre-allocation.

Theory: Each T gate on qubit q produces a "shift vector" β(Z_q) ∈ GF(2)^n
that describes how the T gate branches the coefficient vector. The shift
vector has β_i = 1 iff the i-th stabilizer anticommutes with Z_q (i.e.,
has an X or Y component on qubit q). If β = 0, Z_q is a stabilizer and
the T gate applies only a scalar phase (no branching).

The key insight: when multiple T gates produce linearly dependent shift
vectors (over GF(2)), their branches overlap. The total number of distinct
branch indices is 2^rank, where rank is the GF(2) rank of the accumulated
shift vector matrix, NOT 2^(count of T gates).

Measurements project the state, which can reduce |v|. When measuring Z_q,
if β(Z_q) is in the span of the accumulated shift vectors, the measurement
fixes one bit of the branch index, reducing the effective rank by 1.

The shift vectors are extracted from stim's inverse tableau:
T^{-1}(Z_q) has X-bits that give exactly the destabilizer components β.
These are directly comparable across Clifford gates (no basis change issue)
because they represent shifts in the fixed v-index space.
"""

import numpy as np
import stim


def _gf2_rank(matrix: list[np.ndarray]) -> int:
    """Compute GF(2) rank of a list of binary vectors."""
    if not matrix:
        return 0
    m = np.array(matrix, dtype=int) % 2
    rows, cols = m.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if m[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for row in range(rows):
            if row != rank and m[row, col] == 1:
                m[row] = (m[row] + m[rank]) % 2
        rank += 1
    return rank


def _gf2_reduce(matrix: list[np.ndarray]) -> list[np.ndarray]:
    """Compute reduced row echelon form over GF(2), removing zero rows."""
    if not matrix:
        return []
    m = np.array(matrix, dtype=int) % 2
    rows, cols = m.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if m[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for row in range(rows):
            if row != rank and m[row, col] == 1:
                m[row] = (m[row] + m[rank]) % 2
        rank += 1
    return [m[i].copy() for i in range(rank)]


def _is_in_gf2_span(matrix: list[np.ndarray], vector: np.ndarray) -> bool:
    """Check if vector is in the GF(2) span of the matrix rows."""
    if not matrix:
        return np.all(vector == 0)
    extended = matrix + [vector]
    return _gf2_rank(extended) == _gf2_rank(matrix)


def _pauli_x_bits(pauli_string: stim.PauliString, n: int) -> np.ndarray:
    """Extract X-component bits from a PauliString.

    Returns a binary vector where bit i is 1 iff the Pauli at position i
    has an X or Y component (i.e., pauli index ∈ {1=X, 2=Y}).
    """
    return np.array([1 if pauli_string[i] in (1, 2) else 0 for i in range(n)], dtype=int)


def _get_shift_vector(sim: stim.TableauSimulator, qubit: int, n: int) -> np.ndarray:
    """Extract shift vector β(Z_q) from stim's inverse tableau.

    The inverse tableau maps Z_q to its decomposition in the Heisenberg
    picture. The X-bits of T^{-1}(Z_q) give the destabilizer components,
    which are exactly the shift vector β.
    """
    inv_tab = sim.current_inverse_tableau()
    z_out = inv_tab.z_output(qubit)
    return _pauli_x_bits(z_out, n)


def _get_shift_vector_for_observable(
    sim: stim.TableauSimulator, observable: stim.PauliString, n: int
) -> np.ndarray:
    """Extract shift vector β(P) for a general Pauli observable P.

    Rewinding P through the inverse tableau gives T^{-1}(P) whose X-bits
    are the shift vector. For a product P = P_0 ⊗ P_1 ⊗ ... this is
    computed by multiplying the individual rewound Paulis.
    """
    inv_tab = sim.current_inverse_tableau()
    # Rewind each qubit's Pauli component through the inverse tableau
    # and multiply them together.
    result = None
    for q in range(n):
        p = observable[q]  # 0=I, 1=X, 2=Y, 3=Z
        if p == 0:
            continue
        if p == 1:  # X
            rewound = inv_tab.x_output(q)
        elif p == 2:  # Y
            # Y = iXZ, so T^{-1}(Y_q) = T^{-1}(X_q) * T^{-1}(Z_q)
            rewound = inv_tab.x_output(q) * inv_tab.z_output(q)
        elif p == 3:  # Z
            rewound = inv_tab.z_output(q)
        else:
            raise ValueError(f"Unexpected Pauli index: {p}")

        if result is None:
            result = rewound
        else:
            result = result * rewound

    if result is None:
        return np.zeros(n, dtype=int)
    return _pauli_x_bits(result, n)


def compute_v_bound_trace(n: int, ops: list[tuple[str, list[int]]]) -> list[int]:
    """Compute the theoretical |v| upper bound at every step of the circuit.

    The algorithm tracks the GF(2) rank of accumulated shift vectors:
    1. At each T/T† gate, extract β(Z_q) from the inverse tableau.
       If β is nonzero and linearly independent, rank increases.
    2. Clifford gates update the simulator (affecting future β vectors).
    3. At measurements, if β(Z_q) is in the span of accumulated shifts,
       the measurement projects out one dimension (rank decreases by 1).

    The bound at each step is 2^rank, where rank is the current GF(2) rank
    of the shift space after accounting for measurement reductions.

    Args:
        n: number of qubits
        ops: list of (gate_name, qubit_list) tuples

    Returns:
        A list of bounds, one per operation.
    """
    if n == 0:
        return []

    # Map gate names to stim.TableauSimulator method names
    clifford_gate_map = {
        "H": "h",
        "S": "s",
        "S_DAG": "s_dag",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "CX": "cx",
        "CZ": "cz",
        "CY": "cy",
        "SWAP": "swap",
        "ISWAP": "iswap",
        "ISWAP_DAG": "iswap_dag",
        "SQRT_X": "sqrt_x",
        "SQRT_X_DAG": "sqrt_x_dag",
        "SQRT_Y": "sqrt_y",
        "SQRT_Y_DAG": "sqrt_y_dag",
    }

    # Noise and annotation gates that don't affect the tableau or rank
    skip_gates = {
        "TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR",
        "OBSERVABLE_INCLUDE", "MPAD",
        "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR",
        "Y_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2",
        "E", "ELSE_CORRELATED_ERROR",
    }

    sim = stim.TableauSimulator()
    sim.set_num_qubits(n)

    shift_rows: list[np.ndarray] = []  # maintained in GF(2) reduced form
    bounds: list[int] = []

    def _try_reduce_rank(beta_meas: np.ndarray):
        """If beta_meas is in the shift span, remove one dimension."""
        nonlocal shift_rows
        if (np.any(beta_meas != 0) and shift_rows
                and _is_in_gf2_span(shift_rows, beta_meas)):
            extended = shift_rows + [beta_meas]
            reduced = _gf2_reduce(extended)
            shift_rows = reduced[:-1] if len(reduced) >= len(shift_rows) else reduced

    for gate_name, args in ops:
        if gate_name in ("T", "T_DAG"):
            beta = _get_shift_vector(sim, args[0], n)
            if np.any(beta != 0):
                if not _is_in_gf2_span(shift_rows, beta):
                    shift_rows.append(beta)
                    shift_rows = _gf2_reduce(shift_rows)

        elif gate_name in ("M", "MR", "MZ"):
            q = args[0]
            beta_meas = _get_shift_vector(sim, q, n)
            _try_reduce_rank(beta_meas)
            sim.measure(q)
            if gate_name == "MR":
                sim.reset(q)

        elif gate_name in ("MX", "MRX"):
            q = args[0]
            # MX measures X_q. Shift vector for X_q comes from
            # the X-bits of T^{-1}(X_q) = inv_tab.x_output(q).
            inv_tab = sim.current_inverse_tableau()
            x_out = inv_tab.x_output(q)
            beta_meas = _pauli_x_bits(x_out, n)
            _try_reduce_rank(beta_meas)
            # Use do_circuit to apply MX (stim has no direct method)
            sim.do_circuit(stim.Circuit(f"MX {q}"))
            if gate_name == "MRX":
                sim.reset_x(q)

        elif gate_name in ("MY", "MRY"):
            q = args[0]
            inv_tab = sim.current_inverse_tableau()
            # Y = iXZ, rewound Y is x_output * z_output
            y_rewound = inv_tab.x_output(q) * inv_tab.z_output(q)
            beta_meas = _pauli_x_bits(y_rewound, n)
            _try_reduce_rank(beta_meas)
            sim.do_circuit(stim.Circuit(f"MY {q}"))
            if gate_name == "MRY":
                sim.reset_y(q)

        elif gate_name == "MPP":
            # args = [[(pauli_char, qubit_idx), ...]]
            product = args[0]
            # Build a stim.PauliString for the product observable
            obs = stim.PauliString(n)
            for pauli_char, qubit_idx in product:
                p_idx = {"X": 1, "Y": 2, "Z": 3}[pauli_char]
                obs[qubit_idx] = p_idx

            beta_meas = _get_shift_vector_for_observable(sim, obs, n)
            _try_reduce_rank(beta_meas)
            sim.measure_observable(obs)

        elif gate_name == "R":
            sim.reset(args[0])

        elif gate_name == "RX":
            sim.reset_x(args[0])

        elif gate_name == "RY":
            sim.reset_y(args[0])

        elif gate_name in clifford_gate_map:
            getattr(sim, clifford_gate_map[gate_name])(*args)

        elif gate_name in skip_gates:
            pass  # noise/annotations don't affect rank

        else:
            raise ValueError(f"Unknown gate: {gate_name}")

        bounds.append(2 ** len(shift_rows))

    return bounds


def compute_peak_v_bound(n: int, ops: list[tuple[str, list[int]]]) -> int:
    """Compute the peak |v| upper bound for the entire circuit.

    This is the value to use for pre-allocating the coefficient vector buffer.
    """
    bounds = compute_v_bound_trace(n, ops)
    return max(bounds) if bounds else 1
