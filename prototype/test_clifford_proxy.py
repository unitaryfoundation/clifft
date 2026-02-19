"""Validate the Clifford proxy bound against actual |v| from the prototype.

For random Clifford+T circuits, we run both:
  1. The actual generalized stabilizer simulation (tracking |v| at each step)
  2. The Clifford proxy pass (computing the bound at each step)

and verify that the bound is always >= the actual |v|.
"""

import numpy as np
import pytest

from state import GeneralizedStabilizerState
from clifford_proxy import compute_v_bound_trace, compute_peak_v_bound


def build_random_circuit(n: int, depth: int, rng: np.random.Generator, t_frac: float = 0.3):
    """Generate a random Clifford+T circuit as an ops list."""
    ops = []
    single_gates = ["H", "S", "S_DAG", "X", "Y", "Z"]
    t_gates = ["T", "T_DAG"]

    for _ in range(depth):
        r = rng.random()
        if r < t_frac:
            gate = rng.choice(t_gates)
            q = int(rng.integers(0, n))
            ops.append((gate, [q]))
        elif r < t_frac + 0.15 and n >= 2:
            ctrl, tgt = rng.choice(n, size=2, replace=False)
            ops.append(("CX", [int(ctrl), int(tgt)]))
        else:
            gate = rng.choice(single_gates)
            q = int(rng.integers(0, n))
            ops.append((gate, [q]))

    return ops


def run_prototype_tracking_v(n: int, ops: list) -> list[int]:
    """Run the prototype and record |v| after each operation."""
    gate_map = {
        "H": "apply_h",
        "S": "apply_s",
        "S_DAG": "apply_sdg",
        "X": "apply_x",
        "Y": "apply_y",
        "Z": "apply_z",
        "T": "apply_t",
        "T_DAG": "apply_tdg",
        "CX": "apply_cnot",
    }
    state = GeneralizedStabilizerState(n)
    sizes = []

    for gate_name, qubits in ops:
        method_name = gate_map[gate_name]
        getattr(state, method_name)(*qubits)
        sizes.append(state.num_entries)

    return sizes


def run_prototype_with_measurements(n, ops, seed=42):
    """Run prototype with measurements, tracking |v|."""
    gate_map = {
        "H": "apply_h", "S": "apply_s", "S_DAG": "apply_sdg",
        "X": "apply_x", "Y": "apply_y", "Z": "apply_z",
        "T": "apply_t", "T_DAG": "apply_tdg", "CX": "apply_cnot",
    }
    state = GeneralizedStabilizerState(n)
    rng = np.random.default_rng(seed)
    sizes = []
    for gate_name, qubits in ops:
        if gate_name == "M":
            state.measure(qubits[0], rng=rng)
        elif gate_name == "R":
            state.reset(qubits[0])
        elif gate_name == "MR":
            state.measure(qubits[0], rng=rng)
            state.reset(qubits[0])
        elif gate_name in gate_map:
            getattr(state, gate_map[gate_name])(*qubits)
        sizes.append(state.num_entries)
    return sizes


class TestCliffordProxyBound:
    """Test that the Clifford proxy bound is always >= the actual |v|."""

    def test_single_t_on_zero(self):
        """T|0>: Z_0 is stabilizer, β=0, no branching."""
        ops = [("T", [0])]
        bounds = compute_v_bound_trace(1, ops)
        assert bounds == [1]

    def test_t_on_plus(self):
        """H then T: β≠0, rank goes to 1, bound=2."""
        ops = [("H", [0]), ("T", [0])]
        bounds = compute_v_bound_trace(1, ops)
        assert bounds[0] == 1
        assert bounds[1] == 2

    def test_two_qubits_both_t(self):
        """H on both qubits, T on both: independent shifts, rank=2, bound=4."""
        ops = [("H", [0]), ("H", [1]), ("T", [0]), ("T", [1])]
        bounds = compute_v_bound_trace(2, ops)
        assert bounds[-1] == 4

    def test_pure_clifford_stays_1(self):
        """Pure Clifford circuit: no T gates, bound=1 throughout."""
        ops = [("H", [0]), ("CX", [0, 1]), ("S", [1]), ("H", [1])]
        bounds = compute_v_bound_trace(2, ops)
        for b in bounds:
            assert b == 1

    def test_t_gate_on_z_eigenstate_no_growth(self):
        """T on qubits in |0⟩: Z is stabilizer, β=0, no branching."""
        ops = [("T", [0]), ("T", [1])]
        bounds = compute_v_bound_trace(2, ops)
        assert bounds[0] == 1
        assert bounds[1] == 1

    def test_entangled_then_t(self):
        """Bell state + T: tests entangled stabilizers."""
        ops = [("H", [0]), ("CX", [0, 1]), ("T", [0])]
        bounds = compute_v_bound_trace(2, ops)
        assert bounds[-1] == 2

        state = GeneralizedStabilizerState(2)
        state.apply_h(0)
        state.apply_cnot(0, 1)
        state.apply_t(0)
        assert state.num_entries <= bounds[-1]

    def test_ghz_t_gates_rank_collapse(self):
        """GHZ state + T on all qubits: shift vectors are all identical.

        This is the key test for the rank optimization. Per-gate doubling
        would predict 2^3=8, but the shift vectors are all [1,0,0] (all
        equal), so rank=1 and bound=2.
        """
        ops = [
            ("H", [0]), ("CX", [0, 1]), ("CX", [0, 2]),
            ("T", [0]), ("T", [1]), ("T", [2]),
        ]
        bounds = compute_v_bound_trace(3, ops)
        # All 3 T gates have the same β vector → rank = 1
        assert bounds[-1] == 2

        # Verify actual |v| is also 2
        state = GeneralizedStabilizerState(3)
        state.apply_h(0)
        state.apply_cnot(0, 1)
        state.apply_cnot(0, 2)
        for q in range(3):
            state.apply_t(q)
        assert state.num_entries == 2

    def test_repeated_t_same_qubit_rank_1(self):
        """Repeated T on the same qubit: all identical β, rank=1, bound=2."""
        ops = [("H", [0]), ("T", [0]), ("T", [0]), ("T", [0]), ("T", [0])]
        bounds = compute_v_bound_trace(1, ops)
        # All T gates on same qubit have identical β
        assert bounds[-1] == 2

    def test_measurement_reduces_rank(self):
        """Measurement of a qubit whose Z is in the shift span reduces rank."""
        ops = [
            ("H", [0]), ("H", [1]), ("H", [2]),
            ("T", [0]), ("T", [1]), ("T", [2]),
            ("MR", [0]),
        ]
        bounds = compute_v_bound_trace(3, ops)
        # After 3 T gates on independent |+⟩ qubits: rank=3, bound=8
        assert bounds[5] == 8
        # After measuring q0, Z_0's β is in the span → rank drops to 2
        assert bounds[6] == 4

    def test_measurement_of_unrelated_qubit_no_reduction(self):
        """Measuring a qubit NOT in the shift span doesn't reduce rank."""
        ops = [
            ("H", [0]), ("T", [0]),
            ("MR", [1]),  # q1 is in |0⟩, Z_1 is stabilizer, β=0
        ]
        bounds = compute_v_bound_trace(2, ops)
        assert bounds[1] == 2  # After T(0)
        assert bounds[2] == 2  # After MR(1) — no reduction

    def test_peak_bound(self):
        """Peak bound should be the max of the trace."""
        ops = [("H", [0]), ("T", [0]), ("H", [1]), ("T", [1])]
        peak = compute_peak_v_bound(2, ops)
        trace = compute_v_bound_trace(2, ops)
        assert peak == max(trace)

    def test_bound_is_tight_independent_case(self):
        """For H+T on independent qubits, bound should be exactly tight."""
        ops = [
            ("H", [0]), ("H", [1]), ("H", [2]),
            ("T", [0]), ("T", [1]), ("T", [2]),
        ]
        bounds = compute_v_bound_trace(3, ops)
        actual = run_prototype_tracking_v(3, ops)
        assert bounds[-1] == 8
        assert actual[-1] == 8
        assert bounds[-1] == actual[-1]

    def test_empty_circuit(self):
        """Empty circuit should return empty bounds."""
        bounds = compute_v_bound_trace(2, [])
        assert bounds == []

    def test_bound_never_below_1(self):
        """Bound should always be >= 1."""
        rng = np.random.default_rng(777)
        for _ in range(20):
            n = int(rng.integers(2, 5))
            depth = int(rng.integers(10, 30))
            ops = build_random_circuit(n, depth, rng, t_frac=0.3)
            bounds = compute_v_bound_trace(n, ops)
            for b in bounds:
                assert b >= 1

    def test_random_circuits_bound_holds(self):
        """Fuzz test: for many random circuits, bound >= actual |v| at every step."""
        rng = np.random.default_rng(42)
        n_trials = 200
        violations = []

        for trial in range(n_trials):
            n = int(rng.integers(2, 6))
            depth = int(rng.integers(5, 25))
            t_frac = rng.uniform(0.1, 0.5)

            ops = build_random_circuit(n, depth, rng, t_frac)
            bounds = compute_v_bound_trace(n, ops)
            actual_sizes = run_prototype_tracking_v(n, ops)

            assert len(bounds) == len(actual_sizes)

            for step, (bound, actual) in enumerate(zip(bounds, actual_sizes)):
                if actual > bound:
                    violations.append({
                        "trial": trial, "step": step, "n": n,
                        "bound": bound, "actual": actual, "gate": ops[step],
                    })

        assert not violations, (
            f"Bound violated in {len(violations)} cases!\n"
            + "\n".join(
                f"  trial={v['trial']} step={v['step']} n={v['n']} "
                f"gate={v['gate']} bound={v['bound']} actual={v['actual']}"
                for v in violations[:10]
            )
        )

    def test_random_circuits_with_measurements_bound_holds(self):
        """Fuzz test with measurements: bound should still hold.

        Tests multiple measurement outcome seeds to verify the bound
        is valid regardless of which outcomes occur.
        """
        rng = np.random.default_rng(999)
        n_trials = 100
        violations = []

        for trial in range(n_trials):
            n = int(rng.integers(2, 5))
            depth = int(rng.integers(8, 20))

            ops = []
            single_gates = ["H", "S", "S_DAG"]
            for _ in range(depth):
                r = rng.random()
                if r < 0.2:
                    gate = rng.choice(["T", "T_DAG"])
                    q = int(rng.integers(0, n))
                    ops.append((gate, [q]))
                elif r < 0.35 and n >= 2:
                    ctrl, tgt = rng.choice(n, size=2, replace=False)
                    ops.append(("CX", [int(ctrl), int(tgt)]))
                elif r < 0.45:
                    q = int(rng.integers(0, n))
                    ops.append(("MR", [q]))
                else:
                    gate = rng.choice(single_gates)
                    q = int(rng.integers(0, n))
                    ops.append((gate, [q]))

            bounds = compute_v_bound_trace(n, ops)

            for seed in range(5):
                actual_sizes = run_prototype_with_measurements(
                    n, ops, seed=trial * 100 + seed
                )

                min_len = min(len(bounds), len(actual_sizes))
                for step in range(min_len):
                    if actual_sizes[step] > bounds[step]:
                        violations.append({
                            "trial": trial, "seed": seed, "step": step,
                            "n": n, "bound": bounds[step],
                            "actual": actual_sizes[step],
                        })

        assert not violations, (
            f"Bound violated in {len(violations)} cases with measurements!\n"
            + "\n".join(
                f"  trial={v['trial']} seed={v['seed']} step={v['step']} "
                f"n={v['n']} bound={v['bound']} actual={v['actual']}"
                for v in violations[:10]
            )
        )

    def test_bound_tightness_statistics(self):
        """Report tightness across random circuits.

        With the rank-based approach, we expect high tightness (~90%+)
        for random circuits. For structured circuits (e.g., QEC), the
        rank approach is dramatically tighter than per-gate doubling.
        """
        rng = np.random.default_rng(54321)
        tight_count = 0
        total_steps = 0
        max_ratio = 1.0

        for _ in range(100):
            n = int(rng.integers(2, 5))
            depth = int(rng.integers(8, 20))
            ops = build_random_circuit(n, depth, rng, t_frac=0.3)
            bounds = compute_v_bound_trace(n, ops)
            actual = run_prototype_tracking_v(n, ops)

            for b, a in zip(bounds, actual):
                total_steps += 1
                if b == a:
                    tight_count += 1
                if a > 0:
                    ratio = b / a
                    max_ratio = max(max_ratio, ratio)

        tight_pct = 100 * tight_count / total_steps if total_steps else 0
        print(f"\n  Tightness: {tight_count}/{total_steps} steps exactly tight ({tight_pct:.1f}%)")
        print(f"  Max bound/actual ratio: {max_ratio:.1f}x")
        assert tight_pct > 80, f"Expected >80% tightness, got {tight_pct:.1f}%"

    def test_rank_tighter_than_doubling_on_structured_circuit(self):
        """Verify rank bound is tighter than naive doubling on correlated circuits.

        For a GHZ state with T on all qubits, per-gate doubling gives 2^n
        while rank correctly gives 2.
        """
        for n in [3, 4, 5]:
            ops = [("H", [0])]
            for q in range(1, n):
                ops.append(("CX", [0, q]))
            for q in range(n):
                ops.append(("T", [q]))

            bounds = compute_v_bound_trace(n, ops)
            peak = compute_peak_v_bound(n, ops)

            # Rank bound: all β vectors identical → rank=1 → bound=2
            assert peak == 2, f"n={n}: rank bound should be 2, got {peak}"
            # Per-gate doubling would give 2^n
            assert peak < 2**n, f"n={n}: rank bound should be < 2^n={2**n}"
