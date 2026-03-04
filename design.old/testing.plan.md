# **UCC Implementation Plan: Invariant-Based Fuzzing & Exact Trajectories**

## **Mission Context for the AI Agent**

You are an expert C++ and Python systems engineer working on the UCC compiler. We need to implement robust, invariant-based fuzz testing to mathematically prove our state collapse and amplitude interference logic is flawless, specifically focusing on the interaction between non-Clifford $T$-gates and mid-circuit measurements.

**Crucial Constraint:** Because UCC tracks continuous probability amplitudes with exact relative phases, we cannot use standard stabilizer projection algorithms (which randomize global phase). Do **not** modify the C++ get_statevector() implementation. Instead, we will upgrade our exact pure-Python density matrix oracle to support projective measurements, allowing us to mathematically replay specific quantum trajectories sampled by UCC and verify the resulting exact statevectors.

In the below phases, we have some statements of "testing" all gate types. In the code examples they are a hard coded list. In the implementation you should pull the list from the front-end to ensure we always test all known gates.

### ---

**Phase 1: Upgrade CliffordTOracle for Projective Measurements**

**File:** tests/python/clifford_t_oracle.py

Currently, CliffordTOracle skips measurements. Add exact matrix projections to simulate state collapse.

1. Add the measure_and_project method to the CliffordTOracle class:

Python

    def measure_and_project(self, qubit: int, forced_outcome: int) -> float:
        """Project the state onto a forced measurement outcome, returning the probability.

        Args:
            qubit: The target qubit to measure.
            forced_outcome: The expected classical bit (0 or 1).

        Returns:
            The analytical probability of observing this outcome.
        """
        # Create mask of indices where the bit at 'qubit' matches 'forced_outcome'
        indices = np.arange(self.dim)
        bit_mask = (indices >> qubit) & 1
        keep_mask = (bit_mask == forced_outcome)

        # Calculate exact probability of this outcome (squared L2 norm)
        prob = float(np.sum(np.abs(self.sv[keep_mask])**2))

        if prob < 1e-12:
            self.sv.fill(0.0)
            return 0.0

        # Project and re-normalize
        self.sv[~keep_mask] = 0.0
        self.sv /= np.sqrt(prob)
        return prob

2. Add a new standalone function simulate_trajectory to the bottom of the file (do not modify simulate_circuit as it is used for pure unitary tests):

Python

def simulate_trajectory(circuit_str: str, trajectory: list[int]) -> tuple[np.ndarray, list[float]]:
    """Simulate a circuit, forcing specific measurement outcomes from a trajectory.

    Args:
        circuit_str: The Stim circuit string.
        trajectory: A list of 0s and 1s representing the forced measurement outcomes.

    Returns:
        Tuple of (final_normalized_statevector, list_of_probabilities_for_each_measurement).
    """
    max_qubit = 0
    ops: list[tuple[str, list[int]]] = []

    for line in circuit_str.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        gate = parts[0].upper()

        # Ignore annotations
        if gate in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue

        # Extract numeric targets (skip rec[-k] for this oracle)
        qubits = [int(p) for p in parts[1:] if p.lstrip('!').isdigit()]
        ops.append((gate, qubits))
        if qubits:
            max_qubit = max(max_qubit, max(qubits))

    sim = CliffordTOracle(max_qubit + 1)
    probs: list[float] = []
    meas_idx = 0

    for gate, qubits in ops:
        if gate == "M":
            for q in qubits:
                if meas_idx >= len(trajectory):
                    raise ValueError("Trajectory array too short for circuit measurements.")
                prob = sim.measure_and_project(q, trajectory[meas_idx])
                probs.append(prob)
                meas_idx += 1
        elif gate == "H": sim.h(qubits[0])
        elif gate == "S": sim.s(qubits[0])
        elif gate in ("S_DAG", "SDAG", "SDG"): sim.s_dag(qubits[0])
        elif gate == "X": sim.x(qubits[0])
        elif gate == "Y": sim.y(qubits[0])
        elif gate == "Z": sim.z(qubits[0])
        elif gate == "T": sim.t(qubits[0])
        elif gate in ("T_DAG", "TDAG", "TDG"): sim.t_dag(qubits[0])
        elif gate in ("CX", "CNOT"): sim.cx(qubits[0], qubits[1])
        elif gate == "CY": sim.cy(qubits[0], qubits[1])
        elif gate == "CZ": sim.cz(qubits[0], qubits[1])
        elif gate in ("MX", "MY", "R", "MR", "RX", "MRX"):
            raise NotImplementedError(f"Gate {gate} not supported in trajectory oracle.")
        else:
            raise ValueError(f"Unknown gate: {gate}")

    if meas_idx != len(trajectory):
        raise ValueError(f"Used {meas_idx} outcomes but {len(trajectory)} were provided.")

    return sim.statevector(), probs

### ---

**Phase 2: The Exact Trajectory Fuzzer**

**Files:** tests/python/conftest.py and tests/python/test_trajectory_oracle.py

1. In tests/python/conftest.py, add a circuit generator that mixes in measurements:

Python

def random_clifford_t_measure_circuit(num_qubits: int, depth: int, seed: int) -> str:
    """Generate a random universal circuit with mid-circuit measurements."""
    rng = np.random.default_rng(seed)
    gates_1q = ["H", "S", "S_DAG", "X", "Y", "Z", "T", "T_DAG"]
    lines: list[str] = []

    for _ in range(depth):
        action = rng.random()
        if num_qubits > 1 and action < 0.20:
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            lines.append(f"CX {q1} {q2}")
        elif action < 0.35:  # ~15% chance of a measurement
            q = rng.integers(0, num_qubits)
            lines.append(f"M {q}")
        else:
            gate = rng.choice(gates_1q)
            q = rng.integers(0, num_qubits)
            lines.append(f"{gate} {q}")
    return "\n".join(lines)

2. In tests/python/test_trajectory_oracle.py, add the fuzz test. It samples 1 shot from UCC, then forces the Oracle to replay that exact universe:

Python

import pytest
import numpy as np

class TestExactTrajectoryFuzzer:
    """Fuzz tests combining non-Clifford gates with mid-circuit measurements."""

    @pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
    @pytest.mark.parametrize("seed", range(5))
    def test_trajectory_fuzzing(self, num_qubits: int, seed: int) -> None:
        """UCC correctly computes post-measurement pure states and sampling probabilities."""
        import ucc
        from clifford_t_oracle import simulate_trajectory
        from conftest import random_clifford_t_measure_circuit, assert_statevectors_equal

        circuit_str = random_clifford_t_measure_circuit(num_qubits, depth=25, seed=seed)
        prog = ucc.compile(circuit_str)

        if prog.num_measurements == 0:
            pytest.skip("Random circuit generated no measurements.")

        # 1. Run exactly 1 shot in UCC to get a physically valid measurement trajectory
        state = ucc.State(prog.peak_rank, prog.num_measurements, seed=seed)
        ucc.execute(prog, state)
        ucc_sv = ucc.get_statevector(prog, state)
        sampled_trajectory = list(state.meas_record)

        # 2. Force the Dense Oracle to follow the exact same trajectory
        oracle_sv, probabilities = simulate_trajectory(circuit_str, sampled_trajectory)

        # 3. Assertions
        for i, p in enumerate(probabilities):
            assert p > 1e-8, f"UCC sampled an impossible outcome at index {i} (Prob = {p})."

        assert_statevectors_equal(
            ucc_sv, oracle_sv,
            msg=f"Statevectors diverged.\nTrajectory: {sampled_trajectory}\nCircuit:\n{circuit_str}"
        )

### ---

**Phase 3: Scrambled Basis Fuzzer (Max Rank Stress)**

**File:** tests/python/test_trajectory_oracle.py

Our current fuzzers start from $|0\dots0\rangle$. We must ensure that T-gates and measurements work flawlessly when dropped into a maximally entangled, chaotic GF(2) basis to stress OP_COLLIDE and OP_MEASURE_MERGE.

1. Add the following test to the TestExactTrajectoryFuzzer class:

Python

    @pytest.mark.parametrize("num_qubits", [3, 4, 5])
    @pytest.mark.parametrize("seed", range(5))
    def test_scrambled_basis_fuzzing(self, num_qubits: int, seed: int) -> None:
        """Stress-test OP_COLLIDE by applying T-gates to a dense random stabilizer state."""
        import stim
        import ucc
        from clifford_t_oracle import simulate_trajectory
        from conftest import random_clifford_t_measure_circuit, assert_statevectors_equal

        # 1. Generate a maximally dense random Clifford state
        tableau = stim.Tableau.random(num_qubits, seed=seed)
        clifford_prefix = str(stim.Circuit.generated_by_clifford_tableau(tableau))

        # 2. Append T-gates and measurements to force heavy collision math
        tail = random_clifford_t_measure_circuit(num_qubits, depth=15, seed=seed+100)
        circuit_str = clifford_prefix + "\n" + tail

        # 3. Evaluate in UCC (1 shot)
        prog = ucc.compile(circuit_str)
        if prog.num_measurements == 0:
            pytest.skip("No measurements")

        state = ucc.State(prog.peak_rank, prog.num_measurements, seed=seed)
        ucc.execute(prog, state)
        ucc_sv = ucc.get_statevector(prog, state)
        trajectory = list(state.meas_record)

        # 4. Evaluate in Oracle
        oracle_sv, probs = simulate_trajectory(circuit_str, trajectory)

        for i, p in enumerate(probs):
            assert p > 1e-8, f"Impossible outcome on scrambled basis. Prob = {p}"

        assert_statevectors_equal(ucc_sv, oracle_sv, msg="Scrambled basis interference failed.")

### ---

**Phase 4: Exhaustive C++ Gate-by-Gate Verification**

**File:** tests/test_frontend.cc

**Task:** Prove in C++ that the Front-End's Heisenberg rewinding perfectly aligns with Stim's mathematical definition of $U_{clifford}^\dagger Z U_{clifford}$.

Add this test to the bottom of the file:

C++

TEST_CASE("Frontend: Exhaustive 1-qubit T-gate rewinding", "[frontend][exhaustive]") {
    // We test that for every supported single-qubit Clifford, the rewound Z
    // operator computed by our Front-End matches the exact mathematical definition:
    // P_rewound = U_dag Z U.

    std::vector<std::string> cliffords = {
        "H", "S", "S_DAG", "X", "Y", "Z"
    };

    for (const auto& gate_name : cliffords) {
        // 1. Compile through our Front-End
        std::string circuit_text = gate_name + " 0\nT 0";
        auto circuit = parse(circuit_text);
        auto hir = trace(circuit);

        REQUIRE(hir.num_ops() == 1);
        REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);

        // 2. Compute ground truth using pure Stim Tableau Simulator
        stim::TableauSimulator<64> sim(std::mt19937_64(0), 1);
        stim::Circuit stim_circ;
        stim_circ.safe_append_u(gate_name, {0});
        sim.safe_do_circuit(stim_circ);

        // T gate rewinds the Z observable. Stim's inv_state stores exactly U_dag Z_q U
        auto expected = sim.inv_state.zs[0];

        // 3. Assert our Front-End emitted the exact same Pauli geometry
        REQUIRE(hir.ops[0].destab_mask() == expected.xs.u64[0]);
        REQUIRE(hir.ops[0].stab_mask() == expected.zs.u64[0]);
        REQUIRE(hir.ops[0].sign() == expected.sign);
    }
}
