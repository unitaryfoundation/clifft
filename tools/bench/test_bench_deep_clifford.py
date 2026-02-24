"""Benchmarks for deep Clifford circuits.

Circuit: 50 qubits, 5000 random Clifford gates (H, S, CX), 50 measurements.

Run with: just bench
"""

import random
from typing import Any

import pytest
import stim

import ucc

_NUM_QUBITS = 50
_CLIFFORD_DEPTH = 5000
_BENCHMARK_SHOTS = 100_000
_SEED = 42


def generate_deep_clifford_circuit(
    num_qubits: int = _NUM_QUBITS,
    depth: int = _CLIFFORD_DEPTH,
    seed: int = _SEED,
) -> str:
    """Generate a random deep Clifford circuit."""
    rng = random.Random(seed)
    lines = []

    for _ in range(depth):
        gate_type = rng.randint(0, 2)

        if gate_type == 0:
            q = rng.randrange(num_qubits)
            lines.append(f"H {q}")
        elif gate_type == 1:
            q = rng.randrange(num_qubits)
            lines.append(f"S {q}")
        else:
            q1 = rng.randrange(num_qubits)
            q2 = rng.randrange(num_qubits)
            while q2 == q1:
                q2 = rng.randrange(num_qubits)
            lines.append(f"CX {q1} {q2}")

    lines.append(f"M {' '.join(str(i) for i in range(num_qubits))}")

    return "\n".join(lines)


_DEEP_CLIFFORD_CIRCUIT = generate_deep_clifford_circuit()


def test_00_deep_clifford_info(record_property: Any) -> None:
    """Report deep Clifford circuit parameters and compilation results."""
    circuit = stim.Circuit(_DEEP_CLIFFORD_CIRCUIT)

    h_count = 0
    s_count = 0
    cx_count = 0
    m_count = 0

    for instr in circuit:
        name = instr.name
        targets = len(instr.targets_copy())
        if name == "H":
            h_count += targets
        elif name == "S":
            s_count += targets
        elif name == "CX":
            cx_count += targets // 2
        elif name == "M":
            m_count += targets

    total_cliffords = h_count + s_count + cx_count
    program = ucc.compile(_DEEP_CLIFFORD_CIRCUIT)

    record_property("num_qubits", _NUM_QUBITS)
    record_property("clifford_depth", _CLIFFORD_DEPTH)
    record_property("h_gates", h_count)
    record_property("s_gates", s_count)
    record_property("cx_gates", cx_count)
    record_property("total_cliffords", total_cliffords)
    record_property("measurements", m_count)
    record_property("ucc_instructions", program.num_instructions)
    record_property("ucc_peak_rank", program.peak_rank)

    compression = total_cliffords / program.num_instructions if program.num_instructions else 0

    print(f"\n  Circuit: {_NUM_QUBITS} qubits, {_CLIFFORD_DEPTH} gates")
    print(f"  H: {h_count}, S: {s_count}, CX: {cx_count}, M: {m_count}")
    print(f"  UCC: {program.num_instructions} instructions, {compression:.0f}x compression")


@pytest.fixture(scope="module")
def stim_circuit() -> stim.Circuit:
    """Pre-parsed Stim circuit."""
    return stim.Circuit(_DEEP_CLIFFORD_CIRCUIT)


@pytest.fixture(scope="module")
def stim_sampler(stim_circuit: stim.Circuit) -> stim.CompiledMeasurementSampler:
    """Pre-compiled Stim measurement sampler."""
    return stim_circuit.compile_sampler()


@pytest.fixture(scope="module")
def ucc_program() -> ucc.Program:
    """Pre-compiled UCC program."""
    return ucc.compile(_DEEP_CLIFFORD_CIRCUIT)


def test_compile_stim_deep(benchmark: Any) -> None:
    """Measure Stim circuit parsing and sampler compilation time."""

    def compile_stim() -> stim.CompiledMeasurementSampler:
        circuit = stim.Circuit(_DEEP_CLIFFORD_CIRCUIT)
        return circuit.compile_sampler()

    benchmark(compile_stim)


def test_compile_ucc_deep(benchmark: Any) -> None:
    """Measure UCC circuit parsing and compilation time."""

    def compile_ucc() -> ucc.Program:
        return ucc.compile(_DEEP_CLIFFORD_CIRCUIT)

    benchmark(compile_ucc)


def test_sample_stim_deep(benchmark: Any, stim_sampler: stim.CompiledMeasurementSampler) -> None:
    """Measure Stim execution time for 100k shots."""

    def sample_stim() -> object:
        return stim_sampler.sample(_BENCHMARK_SHOTS)

    benchmark(sample_stim)


def test_sample_ucc_deep(benchmark: Any, ucc_program: ucc.Program) -> None:
    """Measure UCC execution time for 100k shots."""

    def sample_ucc() -> object:
        return ucc.sample(ucc_program, _BENCHMARK_SHOTS, seed=0)

    benchmark(sample_ucc)


def test_clifford_absorption_scaling(record_property: Any) -> None:
    """Verify UCC instruction count is O(measurements), not O(gates)."""
    import time

    num_qubits = 20
    depths = [100, 500, 1000, 2000, 5000]

    print("\n  Clifford Absorption Scaling")
    print(f"  {'Depth':>8}  {'Compile (ms)':>12}  {'Instructions':>12}")

    for depth in depths:
        circuit_text = generate_deep_clifford_circuit(num_qubits=num_qubits, depth=depth, seed=42)

        t0 = time.perf_counter()
        program = ucc.compile(circuit_text)
        compile_time_ms = (time.perf_counter() - t0) * 1000

        print(f"  {depth:>8}  {compile_time_ms:>12.2f}  {program.num_instructions:>12}")

        record_property(f"depth_{depth}_instructions", program.num_instructions)

        assert program.num_instructions == num_qubits
