"""Demonstrations of the generalized stabilizer simulation.

Each example shows how the state representation evolves step by step,
making visible how Clifford gates, T gates, and measurements affect
the (v, B(S,D)) representation.
"""

import numpy as np
from state import GeneralizedStabilizerState


def example_1_single_qubit_t():
    """Example 1: T gate on a single qubit.

    Shows that T|0⟩ doesn't increase |v| (Z_0 stabilizes |0⟩),
    but T|+⟩ doubles it.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single-qubit T gate")
    print("=" * 70)

    s = GeneralizedStabilizerState(1)
    s.print_state("Step 0: Initial |0⟩")

    print("\n  ➤ Applying T gate to |0⟩...")
    print("    T = e^{iπ/8}[cos(π/8)I - i·sin(π/8)Z]")
    print("    Since Z|0⟩ = +|0⟩, the Z term doesn't create a new basis entry.")
    print("    T|0⟩ = e^{iπ/8}[cos(π/8) - i·sin(π/8)]|0⟩ = |0⟩ (up to phase)")
    s.apply_t(0)
    s.print_state("Step 1: T|0⟩  (|v| stays at 1!)")

    print("\n  Now let's try T|+⟩ instead:")
    s2 = GeneralizedStabilizerState(1)
    s2.apply_h(0)
    s2.print_state("Step 0: H|0⟩ = |+⟩")

    print("\n  ➤ Applying T gate to |+⟩...")
    print("    Now s_0 = +X, d_0 = +Z.")
    print("    Z anticommutes with X, so β(Z)=[1] — the Z term shifts the index.")
    print("    T|+⟩ = cos(π/8)|b_0⟩ - i·sin(π/8)|b_1⟩")
    print("    The coefficient vector goes from 1 entry to 2!")
    s2.apply_t(0)
    s2.print_state("Step 1: T|+⟩  (|v| doubled to 2)")


def example_2_t_gate_depth():
    """Example 2: How |v| grows with T-depth.

    Each T gate on a qubit in superposition can double |v|.
    But T†T = I cancels, and TT = S is Clifford.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: T-gate depth and |v| growth")
    print("=" * 70)

    s = GeneralizedStabilizerState(1)
    s.apply_h(0)
    s.print_state("Start: |+⟩  (|v|=1)")

    s.apply_t(0)
    s.print_state("After T   (|v|=2)")

    print("\n  ➤ Now apply T† (should undo T, back to |+⟩)...")
    s.apply_tdg(0)
    s.print_state("After T†T = I  (|v| back to 1!)")

    print("\n  ➤ Now try T² = S (a Clifford gate):")
    s2 = GeneralizedStabilizerState(1)
    s2.apply_h(0)
    s2.apply_t(0)
    s2.apply_t(0)
    s2.print_state("After T² = S on |+⟩  (|v|=2 — entries don't auto-merge)")


def example_3_bell_plus_t():
    """Example 3: Bell state + T gate.

    Shows how entanglement interacts with the T-gate decomposition.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Bell state + T gate")
    print("=" * 70)

    s = GeneralizedStabilizerState(2)
    s.print_state("Step 0: |00⟩")

    s.apply_h(0)
    s.print_state("Step 1: H⊗I |00⟩")

    s.apply_cnot(0, 1)
    s.print_state("Step 2: CNOT → Bell state (Φ+)")

    print("\n  ➤ Applying T to qubit 0 of the Bell state...")
    s.apply_t(0)
    s.print_state("Step 3: (T⊗I)|Bell⟩  (|v|=2)")


def example_4_measurement():
    """Example 4: Measurement collapses entries.

    After T creates multiple entries, measurement can reduce them.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: T gate then measurement")
    print("=" * 70)

    rng = np.random.default_rng(42)

    s = GeneralizedStabilizerState(2)
    s.apply_h(0)
    s.apply_h(1)
    s.apply_t(0)
    s.apply_t(1)
    s.print_state("After H,H,T,T on 2 qubits (|v| up to 4)")

    print(f"\n  ➤ Measuring qubit 0...")
    outcome = s.measure(0, rng=rng)
    print(f"    Outcome: {outcome}")
    s.print_state(f"After measuring q0={outcome}  (|v| reduced!)")

    print(f"\n  ➤ Measuring qubit 1...")
    outcome = s.measure(1, rng=rng)
    print(f"    Outcome: {outcome}")
    s.print_state(f"After measuring q1={outcome}  (|v| = 1, fully collapsed)")


def example_5_clifford_circuit():
    """Example 5: Pure Clifford circuit (|v| stays 1 throughout).

    Inspired by Stim's getting-started: repetition code syndrome extraction.
    3 data qubits + 2 ancilla qubits, CNOT-based parity checks.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Clifford circuit (repetition code)")
    print("=" * 70)
    print("  A [5,1,3] repetition code syndrome extraction.")
    print("  5 qubits: data={0,1,2}, ancilla={3,4}")
    print("  All Clifford → |v| stays at 1 the whole time.")

    s = GeneralizedStabilizerState(5)
    rng = np.random.default_rng(123)

    # Prepare logical |+⟩
    s.apply_h(0)
    s.apply_cnot(0, 1)
    s.apply_cnot(0, 2)
    s.print_state("Logical |+⟩_L encoded: |000⟩+|111⟩")

    # Syndrome extraction round
    # Ancilla 3 checks parity of qubits 0,1
    s.apply_cnot(0, 3)
    s.apply_cnot(1, 3)
    # Ancilla 4 checks parity of qubits 1,2
    s.apply_cnot(1, 4)
    s.apply_cnot(2, 4)

    s.print_state("After syndrome CNOTs (|v| still 1)")

    # Measure ancillas
    m3 = s.measure(3, rng=rng)
    m4 = s.measure(4, rng=rng)
    print(f"\n  Syndrome measurements: ancilla3={m3}, ancilla4={m4}")
    print(f"  (Both 0 = no error detected)")
    s.print_state(f"After measurement (|v| still 1)")


def example_6_why_stabilizer_z_saves():
    """Example 6: Why Z-stabilizers prevent T-gate blowup.

    This is the key insight from the SOFT paper (Proposition 1):
    If Z_q is in the stabilizer group, then T_q doesn't increase |v|.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Z-stabilizers prevent T-gate growth")
    print("=" * 70)
    print("  Key insight from the SOFT paper: if Z_q is already a")
    print("  stabilizer (or product of stabilizers), T_q on qubit q")
    print("  does NOT increase |v|. This is why magic state cultivation")
    print("  with a color code keeps |v| bounded.")

    s = GeneralizedStabilizerState(2)
    print("\n  Start with |00⟩. Stabilizers are +Z_0 and +Z_1.")
    s.print_state("|00⟩")

    print("\n  ➤ Apply T to qubit 0. Z_0 stabilizes the state, so no growth:")
    s.apply_t(0)
    s.print_state("T_0|00⟩ (|v|=1, Z_0 was a stabilizer!)")

    print("\n  ➤ Now put qubit 0 in superposition:")
    s2 = GeneralizedStabilizerState(2)
    s2.apply_h(0)  # Now stabilizer for q0 is X, not Z
    s2.print_state("|+0⟩ (s_0 is now +X, not +Z)")

    print("\n  ➤ T on qubit 0 now DOES increase |v|:")
    s2.apply_t(0)
    s2.print_state("T_0|+0⟩ (|v|=2, Z_0 was NOT a stabilizer)")

    print("\n  ➤ But T on qubit 1 still doesn't grow (Z_1 is still a stabilizer):")
    s2.apply_t(1)
    s2.print_state("T_1 T_0|+0⟩ (|v| still 2!)")


if __name__ == "__main__":
    example_1_single_qubit_t()
    example_2_t_gate_depth()
    example_3_bell_plus_t()
    example_4_measurement()
    example_5_clifford_circuit()
    example_6_why_stabilizer_z_saves()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  The generalized stabilizer representation (v, B(S,D)) tracks:
    1. A stabilizer-destabilizer tableau (S,D) — updated by Cliffords
    2. A sparse coefficient vector v over the basis B(S,D)

  Key properties:
    - Clifford gates:  update tableau only, |v| unchanged
    - T/T† gates:      |v| can double (at most), but ONLY if Z_q
                       is not already in the stabilizer group
    - Measurements:    |v| can only decrease (projection/filtering)

  This is why quantum error correction circuits are tractable:
    - The code's Z-stabilizers protect against T-gate blowup
    - Measurements between T-layers restore stabilizer constraints
    - For the d=5 color code: |v| ≤ 2^(19-9) = 1024 always
""")
