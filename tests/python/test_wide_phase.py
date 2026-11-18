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
    phase = clifft.PhasePolynomialPass(max_variables=64)
    manager = clifft.HirPassManager()
    manager.add(phase)
    manager.run(hir)
    assert hir.num_t_gates == (0 if width <= 64 else 2 * width)
    assert phase.wide_attempts == 0
    assert phase.wide_work == 0

    default_hir = clifft.trace(clifft.parse(f"H {targets}\nT {targets}\nT {targets}\n"))
    default_phase = clifft.PhasePolynomialPass()
    default_manager = clifft.HirPassManager()
    default_manager.add(default_phase)
    default_manager.run(default_hir)
    assert default_hir.num_t_gates == 0
    assert bool(default_phase.wide_rewrites) == (width > 64)


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
    assert phase.wide_rewrites > 0
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


def _cube_rotation(start: int, mask: int) -> str:
    return "TPP " + "*".join(f"Z{start + bit}" for bit in range(4) if mask & (1 << bit))


@pytest.mark.parametrize("limit", ["variables", "work", None])
def test_wide_observer_extension_keeps_reducible_prefix(limit: str | None) -> None:
    # The 32 four-qubit parity cubes are identity. The remaining prefix
    # prepares S|+> on qubit 0 and T|+> on qubit 1.
    cubes = [_cube_rotation(q, mask) for mask in range(1, 16) for q in range(0, 128, 4)]
    preparation = "H " + " ".join(map(str, range(129)))
    prefix = "\n".join([preparation, "T 0", "TPP Z0*Z129", "EXP_VAL Y0", *cubes, "T 1"])
    boundary = prefix + "\nEXP_VAL Y0"
    options = {}
    if limit == "work":
        # Give the complete prefix its measured allowance. The suffix here
        # stays within 128 variables, so only the work limit can stop it.
        baseline = clifft.PhasePolynomialPass()
        clifft.compile(boundary, hir_passes=clifft.default_hir_pass_manager(overrides=[baseline]))
        assert baseline.wide_rewrites == 1
        assert not baseline.wide_budget_exhausted
        options["max_wide_work"] = baseline.wide_work
        parity = "*".join(f"Z{q}" for q in range(2, 82))
        suffix = f"TPP {parity}\nEXP_VAL X2"
    elif limit == "variables":
        suffix = "T 128\nEXP_VAL X128"
    else:
        # With room to extend, the T gates on either side of the observer cancel.
        suffix = "T_DAG 1\nEXP_VAL X2"

    phase = clifft.PhasePolynomialPass(**options)
    program = clifft.compile(
        boundary + "\n" + suffix + "\nEXP_VAL X1",
        hir_passes=clifft.default_hir_pass_manager(overrides=[phase]),
    )
    assert phase.wide_rewrites == 1
    assert phase.wide_budget_exhausted == (limit == "work")
    expected_t_count = 0 if limit is None else 2
    assert phase.output_t_count == expected_t_count
    assert program.peak_active_width <= expected_t_count
    expected_x = 1 if limit is None else np.sqrt(0.5)
    np.testing.assert_allclose(
        clifft.sample(program, 1, seed=7).exp_vals[0],
        [1, 1, expected_x, expected_x],
        rtol=0,
        atol=2e-12,
    )


def test_interleaved_measurements_leave_work_for_a_second_wide_region() -> None:
    # The first region reduces to 30 T gates. The two T layers in the
    # second region reduce to Clifford rotations.
    lines = ["H " + " ".join(map(str, range(118)))]
    rotations = [_cube_rotation(q, mask) for q in range(0, 88, 4) for mask in range(1, 16)]
    rotations.extend(f"T {q}" for q in range(88, 118))
    for rotation in rotations:
        lines.extend([rotation, "M 118"])
    layer = "T " + " ".join(map(str, range(70)))
    lines.extend(["R_X(0.137) 0", layer, layer])

    hir = clifft.trace(clifft.parse("\n".join(lines)))
    phase = clifft.PhasePolynomialPass()
    manager = clifft.HirPassManager()
    manager.add(phase)
    manager.run(hir)
    assert phase.input_t_count == 500
    assert phase.wide_rewrites == 2
    assert not phase.wide_budget_exhausted
    assert hir.num_t_gates == 30


@pytest.mark.parametrize("observer", [False, True])
def test_wide_phase_finishes_the_prefix_at_the_variable_limit(observer: bool) -> None:
    targets = " ".join(map(str, range(128)))
    lines = [f"H {targets} 128", f"T {targets}", f"T {targets}"]
    if observer:
        lines.append("EXP_VAL Z129")
    lines.extend(["T 128", "EXP_VAL X128"])
    hir = clifft.trace(clifft.parse("\n".join(lines)))
    phase = clifft.PhasePolynomialPass()
    manager = clifft.HirPassManager()
    manager.add(phase)
    manager.run(hir)
    assert phase.input_t_count == 257
    assert phase.wide_rewrites == 1
    assert hir.num_t_gates == 1


@pytest.mark.parametrize("width", [68, 116])
def test_two_word_core_search_leaves_work_for_a_second_region(width: int) -> None:
    axes = (0, 1, width - 2, width - 1)
    cube = [
        "TPP " + "*".join(f"Z{axis}" for bit, axis in enumerate(axes) if subset & (1 << bit))
        for subset in range(1, 16)
    ]
    targets = " ".join(map(str, range(width)))
    ends = f"TPP Z0*Z{width - 1}"
    # The T layer separates the two halves of the identity cube. At width 68
    # TOHPE can recover the whole layer. At 116 its cost exceeds the default
    # allowance, and projection leaves seven additional odd columns.
    lines = [f"H {targets}", *cube[:7], f"T {targets}", *cube[7:], ends, ends]
    # A second region of paired T gates reduces to Cliffords in both cases.
    layer = "T " + " ".join(map(str, range(68)))
    source = "\n".join([*lines, "R_X(0.137) 0", layer, layer, "R_X(0.137) 0"])
    hir = clifft.trace(clifft.parse(source))
    phase = clifft.PhasePolynomialPass()
    manager = clifft.HirPassManager()
    manager.add(phase)
    manager.run(hir)
    assert phase.input_t_count == width + 17 + 2 * 68
    assert phase.wide_rewrites == 2
    assert not phase.wide_budget_exhausted
    assert hir.num_t_gates == (68 if width == 68 else 123)


def test_wide_phase_keeps_a_better_prefix_when_the_full_core_fits() -> None:
    # The prefix is identity. After 48 more T gates the complete core fits,
    # but its 132 odd columns exceed the TOHPE search limit.
    lines = ["H " + " ".join(map(str, range(68)))]
    lines.extend(_cube_rotation(q, mask) for mask in range(1, 16) for q in range(0, 68, 4))
    lines.extend(["EXP_VAL Z68", "T " + " ".join(map(str, range(48))), "R_X(0.137) 0"])
    hir = clifft.trace(clifft.parse("\n".join(lines)))
    phase = clifft.PhasePolynomialPass()
    manager = clifft.HirPassManager()
    manager.add(phase)
    manager.run(hir)
    assert phase.input_t_count == 303
    assert phase.wide_rewrites == 1
    assert not phase.wide_budget_exhausted
    assert hir.num_t_gates == 48


@pytest.mark.parametrize("limit", ["terms", "variables", None])
def test_wide_phase_extends_a_reduced_prefix_within_its_limits(limit: str | None) -> None:
    targets = " ".join(map(str, range(128)))
    lines = [f"H {targets} 128", f"T {targets}", "EXP_VAL Z129"]
    # The first probe sees 128 independent T gates. Completing the parity
    # cubes cancels them all before the second probe.
    lines.extend(
        _cube_rotation(q, mask)
        for mask in range(1, 16)
        if mask.bit_count() > 1
        for q in range(0, 128, 4)
    )
    dense_parity = "*".join(f"Z{q}" for q in range(2, 128))
    options = {}
    if limit == "terms":
        # The table fits all 15 parities of each cube. The dense suffix adds one more.
        options["max_wide_terms"] = 32 * 15
        lines.extend(["T 1", "EXP_VAL X0", f"TPP {dense_parity}", "EXP_VAL Y1", "EXP_VAL X2"])
        expected = [1, 1, np.sqrt(0.5), np.sqrt(0.5)]
    elif limit == "variables":
        lines.extend(["TPP_DAG Z1*Z2", "EXP_VAL X0", "T 128", "EXP_VAL Y1*Z2", "EXP_VAL X128"])
        expected = [1, 1, -np.sqrt(0.5), np.sqrt(0.5)]
    else:
        # A 126-bit parity extends the nonlinear prefix as one more rotation.
        lines.extend(
            ["TPP_DAG Z1*Z2", "EXP_VAL X0", f"TPP {dense_parity}", "EXP_VAL Y1*Z2", "EXP_VAL X2"]
        )
        expected = [1, 1, -np.sqrt(0.5), 0.5]
    phase = clifft.PhasePolynomialPass(**options)
    manager = clifft.HirPassManager()
    manager.add(phase)
    program = clifft.compile("\n".join(lines), hir_passes=manager)
    assert phase.wide_budget_exhausted == (limit == "terms")
    assert phase.wide_rewrites == 1
    assert phase.output_t_count == 2
    assert program.peak_active_width <= 2
    np.testing.assert_allclose(
        clifft.sample(program, 1, seed=7).exp_vals[0], expected, rtol=0, atol=2e-12
    )
