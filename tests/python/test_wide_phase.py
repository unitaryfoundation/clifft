"""Multiword phase rewrites checked against exact independent joint probabilities."""

import re
from functools import lru_cache

import numpy as np
import pytest
from utils_conformance import SamplingMode, assert_joint_distribution

import clifft


def _parity(mask: int) -> list[str]:
    bits = [q for q in range(mask.bit_length()) if mask >> q & 1]
    target = bits[-1]
    compute = [f"CX {q} {target}" for q in bits[:-1]]
    return compute + [f"T {target}"] + compute[::-1]


def _source(width: int, seed: int, noise: float, shuffle: bool = True, readout: float = 0) -> str:
    # The sum of all nonempty parities on four bits vanishes modulo eight.
    # Overlapping identities force wide collection while the exact oracle
    # needs only the two surviving data qubits and explicit record ancillas.
    lines = [
        "H 0",
        f"CX 0 {width}",
        f"M({readout}) {width}",
        "CX rec[-1] 0",
        "H 0",
        "H " + " ".join(map(str, range(1, width))),
    ]
    if noise:
        lines.append(f"Z_ERROR({noise}) 1")
    if seed & 1:
        lines += ["S 0 1", "H 0 1"]
    phases = []
    for q in range(3, width):
        positions = [0, 1, 2, q]
        for mask in range(1, 16):
            phases.append(_parity(sum(1 << positions[j] for j in range(4) if mask >> j & 1)))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(phases)
    lines.extend(line for phase in phases for line in phase)
    lines.append("T 0")
    if seed & 1:
        lines += ["H 0 1", "S_DAG 0 1"]
    lines += ["MPP X0*Z1", "CX rec[-1] 1", "T_DAG 0", "MX 0 1"]
    permutation = rng.permutation(width + 1)
    return "\n".join(
        " ".join(
            [
                words[0],
                *[
                    word
                    if word.startswith("rec[")
                    else re.sub(r"\d+", lambda match: str(permutation[int(match[0])]), word)
                    for word in words[1:]
                ],
            ]
        )
        for words in (line.split() for line in lines)
    )


@lru_cache(maxsize=None)
def _reference(noise: float, readout: float = 0, rotated: bool = False) -> np.ndarray:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import pauli_error

    circuit = QuantumCircuit(7)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.cx(2, 3)
    if readout:
        circuit.append(pauli_error([("X", readout), ("I", 1 - readout)]).to_instruction(), [3])
    circuit.cx(3, 0)
    circuit.h(0)
    circuit.h(1)
    if noise:
        circuit.append(pauli_error([("Z", noise), ("I", 1 - noise)]).to_instruction(), [1])
    if rotated:
        circuit.s([0, 1])
        circuit.h([0, 1])
    circuit.t(0)
    if rotated:
        circuit.h([0, 1])
        circuit.sdg([0, 1])
    circuit.h(0)
    circuit.cx(0, 4)
    circuit.h(0)
    circuit.cx(1, 4)
    circuit.cx(4, 1)
    circuit.tdg(0)
    circuit.h(0)
    circuit.cx(0, 5)
    circuit.h(1)
    circuit.cx(1, 6)
    circuit.save_probabilities([3, 4, 5, 6])
    result = AerSimulator(method="density_matrix", max_parallel_threads=1).run(circuit).result()
    return np.asarray(result.data()["probabilities"])


@pytest.mark.parametrize("width", [33, 64, 65])
def test_default_phase_expands_within_one_word(width: int) -> None:
    targets = " ".join(map(str, range(width)))
    hir = clifft.trace(clifft.parse(f"H {targets}\nT {targets}\nT {targets}\n"))
    phase = clifft.PhasePolynomialPass()
    manager = clifft.HirPassManager()
    manager.add(phase)
    manager.run(hir)
    assert hir.num_t_gates == (0 if width <= 64 else 2 * width)
    assert phase.wide_attempts == 0
    assert phase.wide_work == 0


@pytest.mark.parametrize("width,seed", [(65, 0), (87, 7), (128, 2), (129, 3)])
def test_wide_phase_preserves_exact_joint_records(width: int, seed: int) -> None:
    noise = 0
    source = _source(width, seed, noise, shuffle=width <= 128)
    phase = clifft.PhasePolynomialPass(max_variables=128)
    manager = clifft.HirPassManager()
    for pass_ in (clifft.PeepholeFusionPass(), phase, clifft.StatevectorSqueezePass()):
        manager.add(pass_)
    program = clifft.compile(source, hir_passes=manager)
    assert phase.wide_attempts > 0
    assert bool(phase.wide_rewrites) == (width <= 128)
    assert program.peak_active_width <= 4
    records = ["".join(str(key >> j & 1) for j in range(4)) for key in range(16)]
    np.testing.assert_allclose(
        clifft.record_probabilities(program, records),
        _reference(noise, rotated=bool(seed & 1)),
        atol=2e-12,
    )


@pytest.mark.parametrize(
    "budget", [{"max_wide_work": 0}, {"max_wide_work": 1}, {"max_wide_terms": 1}]
)
def test_wide_phase_budget_fallback_preserves_joint_records(budget: dict[str, int]) -> None:
    source = _source(65, 4, 0, shuffle=False)
    phase = clifft.PhasePolynomialPass(max_variables=128, **budget)
    manager = clifft.HirPassManager()
    for pass_ in (clifft.PeepholeFusionPass(), phase, clifft.StatevectorSqueezePass()):
        manager.add(pass_)
    program = clifft.compile(source, hir_passes=manager)
    assert not phase.wide_rewrites
    assert phase.wide_budget_exhausted or budget.get("max_wide_work") == 0
    assert program.peak_active_width <= 4
    records = ["".join(str(key >> j & 1) for j in range(4)) for key in range(16)]
    np.testing.assert_allclose(
        clifft.record_probabilities(program, records), _reference(0), atol=2e-12
    )


def test_wide_phase_cost_fallback_preserves_joint_records() -> None:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    measured = " ".join(map(str, range(1, 65)))
    padding = " ".join(map(str, range(4, 65)))
    phases = [
        "TPP " + "*".join(f"Z{q + 1}" for q in range(3) if mask >> q & 1) for mask in range(1, 8)
    ]
    source = "\n".join(
        [
            f"MX {measured}",
            f"M {measured}",
            f"T {padding}",
            *phases,
            "TPP X0",
            f"T {padding}",
            "M 0 1 2 3",
            "R_Z(0.137) 0",
            "T 65",
            "T 65",
        ]
    )
    phase = clifft.PhasePolynomialPass(max_variables=128)
    manager = clifft.HirPassManager()
    manager.add(phase)
    program = clifft.compile(source, hir_passes=manager)
    assert phase.wide_attempts == 1
    assert phase.wide_rewrites == 0
    assert not phase.wide_budget_exhausted
    assert phase.input_t_count == 132
    assert phase.output_t_count == 129
    assert program.peak_active_width == 1

    # Each flag has two independent random measurements followed by diagonal
    # phases. Its last Z readout repeats its second record. Only X0 affects
    # the remaining data qubit, giving an independent one-qubit Aer oracle.
    reference = QuantumCircuit(1)
    reference.h(0)
    reference.t(0)
    reference.h(0)
    reference.save_probabilities()
    result = AerSimulator(method="statevector", max_parallel_threads=1).run(reference).result()
    probabilities = np.asarray(result.data()["probabilities"])
    prefixes = np.random.default_rng(552).integers(0, 2, size=(4, 128), dtype=np.uint8)
    records = np.array(
        [
            np.concatenate((prefix, [outcome], prefix[64:67]))
            for prefix in prefixes
            for outcome in (0, 1)
        ],
        dtype=np.uint8,
    )
    np.testing.assert_allclose(
        clifft.record_probabilities(program, records, return_log=True),
        np.tile(np.log(probabilities) - 128 * np.log(2), len(prefixes)),
        rtol=0,
        atol=2e-12,
    )
    inconsistent = records[:1].copy()
    inconsistent[0, 129] ^= 1
    assert np.isneginf(clifft.record_probabilities(program, inconsistent, return_log=True)[0])


@pytest.mark.parametrize("readout", [0, 0.17])
def test_wide_phase_preserves_noisy_feedback_records(
    sampling_mode: SamplingMode, readout: float
) -> None:
    phase = clifft.PhasePolynomialPass(max_variables=128)
    manager = clifft.HirPassManager()
    for pass_ in (clifft.PeepholeFusionPass(), phase, clifft.StatevectorSqueezePass()):
        manager.add(pass_)
    program = sampling_mode.compile(_source(87, 9, 0.13, readout=readout), hir_passes=manager)
    assert phase.wide_rewrites == 1
    result = sampling_mode.sample(program, shots=16384, seed=319)
    assert_joint_distribution(result.measurements, _reference(0.13, readout, rotated=True))


@pytest.mark.expensive
def test_wide_phase_reduces_direct_readout_preparation() -> None:
    from pathlib import Path

    source = (Path(__file__).parents[1] / "fixtures/merlin_bt81_direct_x.stim").read_text()
    phase = clifft.PhasePolynomialPass(max_variables=128)
    manager = clifft.default_hir_pass_manager(overrides=[phase])
    program = clifft.compile(source, hir_passes=manager)
    assert phase.wide_rewrites == 1
    assert phase.output_t_count == 23
    assert program.peak_active_width == 6
    assert program.num_measurements == 405
    lines = source.splitlines()
    last_t = max(i for i, line in enumerate(lines) if line.startswith(("T ", "T_DAG ")))
    first_readout = next(i for i in range(last_t + 1, len(lines)) if lines[i].startswith("M"))
    preparation = clifft.compile("\n".join(lines[:first_readout]), hir_passes=manager)
    assert preparation.num_measurements == 162
    assert preparation.peak_active_width == 9
