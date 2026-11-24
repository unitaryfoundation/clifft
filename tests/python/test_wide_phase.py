"""Multiword phase rewrites checked against exact independent joint probabilities."""

import re
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stim
from utils_conformance import (
    SamplingMode,
    assert_joint_distribution,
    assert_two_sample_close,
    unpack_bits,
)

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
    assert preparation.original_plan.peak_active_width == 9

    shots = 8192
    result = clifft.sample(program, shots, seed=17)
    assert not result.detectors.any()
    block_law = [11 / 32] + [3 / 32] * 7
    for block in range(3):
        assert_joint_distribution(result.observables[:, 3 * block : 3 * (block + 1)], block_law)

    # Verify the full scheduled-and-squeezed program matches an unscheduled
    # wide-phase-only baseline and both scalar and shared execution paths.
    phase_only = clifft.PhasePolynomialPass(max_variables=128)
    phase_only_mgr = clifft.HirPassManager()
    phase_only_mgr.add(phase_only)
    baseline = clifft.compile(source, hir_passes=phase_only_mgr)
    assert phase_only.wide_rewrites == 1
    assert baseline.peak_active_width == 9

    projections = np.random.default_rng(173).integers(0, 2, size=(9, 24), dtype=np.uint8)
    ref = clifft.sample(baseline, shots, seed=19, batch_size=1, sharing=False)
    assert ref.execution is not None and ref.execution.executor == "scalar"
    assert not ref.detectors.any()
    ref_features = np.concatenate((ref.observables, (ref.observables @ projections) % 2), axis=1)

    for candidate in (
        result,
        clifft.sample(program, shots, seed=23, sharing=True),
    ):
        assert not candidate.detectors.any()
        cand_features = np.concatenate(
            (candidate.observables, (candidate.observables @ projections) % 2), axis=1
        )
        assert_two_sample_close(ref_features.sum(axis=0), shots, cand_features.sum(axis=0), shots)

    idx_mx = len(lines) - 18
    exp_source = "\n".join(
        lines[:idx_mx] + [f"EXP_VAL X{q}" for q in (0, 1, 2, 81, 82, 83, 162, 163, 164)]
    )
    exp_phase = clifft.PhasePolynomialPass(max_variables=128)
    exp_prog = clifft.compile(
        exp_source, hir_passes=clifft.default_hir_pass_manager(overrides=[exp_phase])
    )
    assert exp_phase.wide_rewrites == 1
    np.testing.assert_allclose(
        clifft.sample(exp_prog, 1, seed=7).exp_vals[0], np.full(9, 0.25), rtol=0, atol=1e-12
    )


def _cube_rotation(start: int, mask: int) -> str:
    return "TPP " + "*".join(f"Z{start + bit}" for bit in range(4) if mask & (1 << bit))


def _run_phase(
    source: str, *, postselection_mask: tuple[int, ...] = (), **limits: int
) -> clifft.PhasePolynomialPass:
    hir = clifft.trace(clifft.parse(source), postselection_mask=postselection_mask)
    phase = clifft.PhasePolynomialPass(**limits)
    manager = clifft.HirPassManager()
    manager.add(phase)
    manager.run(hir)
    return phase


def test_canceling_rotations_meet_after_a_wide_identity_is_removed() -> None:
    qs = " ".join(map(str, range(65)))
    source = f"H {qs}\nTPP Y0\nT {qs}\nT_DAG {qs}\nTPP_DAG Y0\nR_X(0.137) 0"
    phase = _run_phase(source)
    assert phase.wide_rewrites == 1
    assert phase.output_t_count == 0


@pytest.mark.parametrize("selected", [False, True])
def test_wide_region_stops_at_a_rotation_that_breaks_a_lookahead_constraint(selected: bool) -> None:
    qs = " ".join(map(str, range(1, 66)))
    source = (
        f"H 0 {qs}\nDEPOLARIZE1(0.15) 0\nT {qs}\nT_DAG {qs}\n"
        "M 0\nDETECTOR rec[-1]\nTPP X0\nTPP X0\nR_X(0.137) 1"
    )
    # Postselection supplies Z0 at entry. The wide rewrite must end before
    # the X rotations, leaving M in place. Without that constraint M can move.
    phase = _run_phase(source, postselection_mask=(int(selected),))
    assert phase.wide_rewrites == 1
    assert phase.pauli_pullbacks == int(not selected)
    assert phase.output_t_count == 0


@pytest.mark.parametrize("earlier_measurement", [False, True])
def test_wide_final_measurement_can_elide_phases_below_the_synthesis_budget(
    earlier_measurement: bool,
) -> None:
    qs = " ".join(map(str, range(68)))
    earlier = "M 68\n" if earlier_measurement else ""
    source = f"H {qs}\nT {qs}\n{earlier}T_DAG {qs}\nTPP Z1*Z2\nM 68"
    # A subsequent rotation requires materializing the remaining parity.
    materialized = _run_phase(source + "\nR_X(0.137) 0")
    assert materialized.output_t_count == 1
    phase = _run_phase(source, max_wide_work=materialized.wide_work - 1)
    assert phase.wide_rewrites == 1
    assert phase.pauli_pullbacks == 1 + int(earlier_measurement)
    assert phase.output_t_count == 0
    assert not phase.wide_budget_exhausted


def test_wide_final_measurement_elides_a_linear_phase_with_distinct_parities() -> None:
    qs = " ".join(map(str, range(80)))
    lines = [f"H {qs}", f"T {qs}"]
    # Each row and column contributes the inverse of its four single-qubit T gates.
    # Each qubit belongs to both, leaving one T_DAG per qubit across 520 distinct parities.
    for base in range(0, 80, 16):
        rows = [range(q, q + 4) for q in range(base, base + 16, 4)]
        for group in (*rows, *zip(*rows, strict=True)):
            for size in (2, 3, 4):
                for subset in combinations(group, size):
                    lines.append("TPP " + "*".join(f"Z{q}" for q in subset))
    lines.extend([f"H {qs}", "M 80"])
    phase = clifft.PhasePolynomialPass()
    program = clifft.compile(
        "\n".join(lines), hir_passes=clifft.default_hir_pass_manager(overrides=[phase])
    )
    assert phase.input_t_count == 520
    assert phase.wide_rewrites == 1
    assert phase.output_t_count == 0
    assert program.peak_active_width == 0


def test_wide_final_measurement_elides_a_reduction_found_by_synthesis() -> None:
    qs = " ".join(map(str, range(68)))
    # All 84 parities are distinct and the core has full rank. The fifteen
    # parities spanned by these four pairs form an identity that TOHPE removes.
    cube = [
        "TPP "
        + "*".join(f"Z{q}" for bit in range(4) if subset & (1 << bit) for q in (2 + bit, 64 + bit))
        for subset in range(1, 16)
    ]
    prefix = "\n".join([f"H {qs}", f"T {qs}", "TPP Z0*Z1*Z2", *cube])
    assert _run_phase(prefix + "\nR_X(0.137) 0").output_t_count == 69
    phase = _run_phase(prefix + "\nM 68")
    assert phase.wide_rewrites == 1
    assert phase.pauli_pullbacks == 1
    assert phase.output_t_count == 0


def test_a_full_parity_table_leaves_work_for_the_next_region() -> None:
    qs = " ".join(map(str, range(66)))
    layer = f"T {qs}\nT {qs}"
    source = f"H {qs} 66\n{layer}\nT 66\nR_X(0.137) 0\n{layer}\nR_X(0.137) 0"
    phase = _run_phase(source, max_wide_terms=66)
    assert phase.wide_budget_exhausted
    assert phase.wide_rewrites == 2
    assert phase.output_t_count == 1


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
        lines.extend(
            ["TPP Z1*Z2", "EXP_VAL X0", f"TPP {dense_parity}", "EXP_VAL Y1*Z2", "EXP_VAL X2"]
        )
        expected = [1, 1, np.sqrt(0.5), 0.5]
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


def _assert_wide_program_matches_stim(
    program: clifft.Program,
    stim_source: str,
    *,
    shots: int,
    seed: int,
    postselection_mask: list[int] | None = None,
) -> None:
    circuit = stim.Circuit(stim_source)
    stim_meas = circuit.compile_sampler(seed=seed).sample(shots).astype(np.uint8)
    stim_det, stim_obs = circuit.compile_m2d_converter().convert(
        measurements=stim_meas.astype(bool), separate_observables=True
    )
    stim_det = stim_det.astype(np.uint8)
    stim_obs = stim_obs.astype(np.uint8)

    if postselection_mask is not None:
        keep = ~stim_det[:, np.asarray(postselection_mask, dtype=bool)].any(axis=1)
        stim_meas = stim_meas[keep]
        stim_det = stim_det[keep]
        stim_obs = stim_obs[keep]
    stim_passed = stim_det.shape[0]
    assert stim_passed > 0

    projections = np.random.default_rng(seed + 101).integers(
        0,
        2,
        size=(program.num_measurements + program.num_detectors + program.num_observables, 32),
        dtype=np.uint8,
    )
    syn_projections = projections[program.num_measurements :]

    record_modes: tuple[Any, ...] = (True, "syndromes", False)
    for keep_records in record_modes:
        if postselection_mask is None and keep_records is False:
            continue
        unpacked = (
            clifft.sample_survivors(program, shots, seed=seed + 1, keep_records=keep_records)
            if postselection_mask is not None
            else clifft.sample(program, shots, seed=seed + 1, keep_records=keep_records)
        )
        packed = (
            clifft.sample_survivors(
                program, shots, seed=seed + 1, keep_records=keep_records, bit_packed=True
            )
            if postselection_mask is not None
            else clifft.sample(
                program, shots, seed=seed + 1, keep_records=keep_records, bit_packed=True
            )
        )

        if postselection_mask is not None:
            assert unpacked.total_shots == shots
            assert unpacked.passed_shots is not None
            assert unpacked.logical_errors is not None
            assert unpacked.observable_ones is not None
            assert_two_sample_close(
                np.array([unpacked.passed_shots, unpacked.logical_errors]),
                shots,
                np.array([stim_passed, int(stim_obs.any(axis=1).sum())]),
                shots,
            )
            np.testing.assert_array_equal(packed.observable_ones, unpacked.observable_ones)
            assert_two_sample_close(
                unpacked.observable_ones,
                unpacked.passed_shots,
                stim_obs.sum(axis=0),
                stim_passed,
            )

        for field, cols, should_have_rows in (
            ("measurements", program.num_measurements, keep_records is True),
            ("detectors", program.num_detectors, keep_records is not False),
            ("observables", program.num_observables, keep_records is not False),
        ):
            u_arr = getattr(unpacked, field)
            p_arr = getattr(packed, field)
            expected_rows = (
                (unpacked.passed_shots if postselection_mask is not None else shots)
                if should_have_rows
                else 0
            )
            assert u_arr.shape == (expected_rows, cols)
            assert p_arr.shape == (expected_rows, (cols + 7) // 8)
            if should_have_rows:
                np.testing.assert_array_equal(unpack_bits(p_arr, cols), u_arr)

        if keep_records is True:
            clifft_full = np.concatenate(
                (unpacked.measurements, unpacked.detectors, unpacked.observables), axis=1
            )
            stim_full = np.concatenate((stim_meas, stim_det, stim_obs), axis=1)
            clifft_feat = np.concatenate((clifft_full, (clifft_full @ projections) % 2), axis=1)
            stim_feat = np.concatenate((stim_full, (stim_full @ projections) % 2), axis=1)
            assert_two_sample_close(
                clifft_feat.sum(axis=0),
                clifft_full.shape[0],
                stim_feat.sum(axis=0),
                stim_full.shape[0],
            )
        elif keep_records == "syndromes":
            clifft_syn = np.concatenate((unpacked.detectors, unpacked.observables), axis=1)
            stim_syn = np.concatenate((stim_det, stim_obs), axis=1)
            clifft_feat = np.concatenate((clifft_syn, (clifft_syn @ syn_projections) % 2), axis=1)
            stim_feat = np.concatenate((stim_syn, (stim_syn @ syn_projections) % 2), axis=1)
            assert_two_sample_close(
                clifft_feat.sum(axis=0),
                clifft_syn.shape[0],
                stim_feat.sum(axis=0),
                stim_syn.shape[0],
            )


def test_wide_clifford_substitution_matches_stim() -> None:
    fixtures = Path(__file__).parents[1] / "fixtures"

    coherent_raw = (fixtures / "coherent_d7_r3.stim").read_text()
    coherent_cliff = re.sub(r"(?m)^(\s*)R_Z\([^)]*\)", r"\1S", coherent_raw).replace(
        "X_ERROR(0.001)", "X_ERROR(0.02)"
    )
    for mask in (None, [1] * 24 + [0] * 120):
        prog = clifft.compile(coherent_cliff, postselection_mask=mask)
        assert prog.num_qubits > 64
        assert prog.num_measurements > 64
        assert prog.num_detectors > 64
        assert prog.peak_active_width == 0
        _assert_wide_program_matches_stim(
            prog, coherent_cliff, shots=4096, seed=611, postselection_mask=mask
        )

    bt81_lines = (fixtures / "merlin_bt81_direct_x.stim").read_text().splitlines()
    for idx, line in enumerate(bt81_lines):
        if line.startswith("T "):
            bt81_lines[idx] = "S " + line[2:]
        elif line.startswith("T_DAG "):
            bt81_lines[idx] = "S_DAG " + line[6:]
    rx_indices = [i for i, line in enumerate(bt81_lines) if line.startswith("RX ")]
    for idx in rx_indices[::10]:
        qubit = bt81_lines[idx].split()[1]
        bt81_lines[idx] = f"RX {qubit}\nZ_ERROR(0.03) {qubit}\nX_ERROR(0.02) {qubit}"
    m_indices = [i for i, line in enumerate(bt81_lines) if line.startswith(("M ", "MX "))]
    for idx in m_indices[::10]:
        op, qubit = bt81_lines[idx].split()
        bt81_lines[idx] = f"{op}(0.02) {qubit}"
    bt81_noisy_cliff = "\n".join(bt81_lines)
    for mask in (None, [1] * 48 + [0] * 186):
        bt81_prog = clifft.compile(bt81_noisy_cliff, postselection_mask=mask)
        assert bt81_prog.num_qubits > 64
        assert bt81_prog.num_measurements > 64
        assert bt81_prog.num_detectors > 64
        assert bt81_prog.peak_active_width == 0
        _assert_wide_program_matches_stim(
            bt81_prog, bt81_noisy_cliff, shots=4096, seed=617, postselection_mask=mask
        )


def test_wide_postselected_pure_clifford_suffix_matches_stim() -> None:
    lines = ["H 0 65", "T 0 65", "T_DAG 0 65", "H 0 65"]
    for _ in range(2):
        for i in range(36):
            ancilla = 36 + i
            lines.extend(
                [
                    f"H {i}",
                    f"CX {i} {ancilla}",
                    f"DEPOLARIZE2(0.008) {i} {ancilla}",
                    f"CX {i} {ancilla}",
                    f"H {i}",
                    f"DEPOLARIZE1(0.01) {i} {ancilla} 63 64 71",
                    f"M(0.008) {ancilla}",
                    "DETECTOR rec[-1]",
                    f"R {ancilla}",
                ]
            )
    lines.extend(
        [
            "CX 0 64",
            "CZ 63 71",
            "DEPOLARIZE1(0.08) 0 64 71",
            "M(0.02) 0 64 65 71",
            "DETECTOR rec[-4]",
            "DETECTOR rec[-1]",
            "OBSERVABLE_INCLUDE(0) rec[-3] rec[-2] rec[-1]",
        ]
    )
    source = "\n".join(lines)
    num_detectors = sum(
        node.gate == clifft.GateType.DETECTOR for node in clifft.parse(source).nodes
    )
    mask = [int(i in (0, 1, 2, 3, 63, 64, 65, 72, 73)) for i in range(num_detectors)]

    # Keep the canceling prefix unoptimized so global reconstruction dilates it
    # and hands the trailing 72-qubit noisy suffix to SuffixTracker.
    program = clifft.compile(source, hir_passes=None, reconstruction=True, postselection_mask=mask)
    assert program.num_qubits == 72 > 64
    assert program.num_measurements > 64
    assert program.num_detectors > 64
    assert program.original_plan.peak_active_width == 2
    assert program.peak_active_width == 0
    for keep_records in (True, False):
        plan = program.sampling_plan(keep_records=keep_records)
        assert plan.reconstruction_selected
        assert plan.peak_active_width == 0

    stim_source = source.replace("T ", "S ").replace("T_DAG ", "S_DAG ")
    _assert_wide_program_matches_stim(
        program, stim_source, shots=4096, seed=629, postselection_mask=mask
    )
