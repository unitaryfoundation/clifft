"""Exact fault-selector algebra and bounded stochastic branch-bank checks."""

import itertools

import numpy as np
import pytest
import stim
from test_phase_core_study import _aer_branches, _single_core_source
from utils_conformance import assert_joint_distribution

import clifft
from tools.prototypes.phase_core_study import compress_phase, trace
from tools.prototypes.phase_polynomial import _add
from tools.prototypes.symbolic_fault_study import (
    BranchBank,
    FaultSite,
    SymbolicFrame,
    compile_frame,
)


def _small_model() -> SymbolicFrame:
    source = _single_core_source()
    lines = source.splitlines()
    first_t = next(j for j, line in enumerate(lines) if line.startswith("T "))
    second_check = [j for j, line in enumerate(lines) if line.startswith("MPP ")][1]
    sites = [
        FaultSite(first_t - 1, "X", 0, 0.17),
        FaultSite(first_t, "Y", 2, 0.13),
        FaultSite(first_t + 4, "X", 3, 0.19),
        FaultSite(second_check, "Z", 1, 0.11),
    ]
    return compile_frame(source, sites)


def _mixture(frame: SymbolicFrame) -> np.ndarray:
    distribution = np.zeros(256)
    for pattern in range(1 << len(frame.sites)):
        weight = np.prod(
            [
                site.probability if pattern & (1 << j) else 1 - site.probability
                for j, site in enumerate(frame.sites)
            ]
        )
        for record, state in _aer_branches(frame.original(pattern), 5).items():
            key = sum(bit << j for j, bit in enumerate(record))
            distribution[key] += weight * np.vdot(state, state).real
    return distribution


def test_symbolic_forms_match_all_fault_paths_and_aer_branches() -> None:
    frame = _small_model()
    ideal = trace(frame.source)
    ideal_magic, ideal_clifford, ideal_encoding = compress_phase(ideal.phase, ideal.width)
    assert frame.core_width == 1
    for pattern in range(1 << len(frame.sites)):
        faulty = trace(frame.original(pattern))
        magic, clifford, encoding = compress_phase(faulty.phase, faulty.width)
        assert magic == ideal_magic and encoding == ideal_encoding
        for mask, coefficient in ideal_clifford.items():
            _add(clifford, mask, -coefficient)
        assert frame.correction_phase(pattern) == clifford
        actual = _aer_branches(frame.render(pattern), 5)
        expected = _aer_branches(frame.original(pattern), 5)
        assert actual.keys() == expected.keys()
        for record, state in expected.items():
            np.testing.assert_allclose(
                np.outer(actual[record], actual[record].conj()),
                np.outer(state, state.conj()),
                atol=1e-12,
            )
        assert clifft.compile(frame.render(pattern)).peak_active_width <= 1


def test_branch_bank_preserves_stochastic_joint_records_and_postselection() -> None:
    frame = _small_model()
    expected = _mixture(frame)
    bank = BranchBank(frame, [1, 1, 1])
    unselected = BranchBank(frame, [0, 0, 0])
    assert_joint_distribution(unselected.sample_measurements(16384, seed=3817), expected)
    passed = sum(p for key, p in enumerate(expected) if not key & 7)
    logical_one = sum(p for key, p in enumerate(expected) if not key & 7 and key & 128)
    actual = bank.sample_survivors(16384, seed=1829)
    assert actual["total_shots"] == 16384
    np.testing.assert_allclose(
        [actual["passed_shots"] / 16384, actual["observable_ones"][0] / 16384],
        [passed, logical_one],
        atol=0.025,
    )
    baseline = clifft.compile(frame.original(), postselection_mask=[1, 1, 1])
    result = clifft.sample_survivors(baseline, shots=16384, seed=1831, batch_size=1, threads=1)
    np.testing.assert_allclose(
        [result.passed_shots / 16384, result.observable_ones[0] / 16384],
        [passed, logical_one],
        atol=0.025,
    )


def test_stochastic_clifford_bank_matches_stim() -> None:
    source = "H 0\nCX 0 1\nMPP X0*X1\nDETECTOR rec[-1]\nMX 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]"
    frame = compile_frame(source, [FaultSite(1, "Z", 0, 0.2), FaultSite(1, "X", 1, 0.1)])
    assert frame.core_width == 0 and frame.geometry_rank() == 0
    bank = BranchBank(frame, [0])
    expected = np.zeros(8)
    records = ["".join(map(str, record)) for record in itertools.product((0, 1), repeat=3)]
    for pattern in range(4):
        probabilities = clifft.record_probabilities(
            clifft.compile(
                "\n".join(
                    line
                    for line in frame.render(pattern).splitlines()
                    if not line.startswith(("DETECTOR", "OBSERVABLE_INCLUDE"))
                )
            ),
            records,
        )
        weight = np.prod(
            [
                site.probability if pattern & (1 << j) else 1 - site.probability
                for j, site in enumerate(frame.sites)
            ]
        )
        for record, probability in zip(records, probabilities):
            key = sum(int(bit) << j for j, bit in enumerate(record))
            expected[key] += weight * probability
    assert_joint_distribution(bank.sample_measurements(8192, seed=4813), expected)
    reference = stim.Circuit(frame.original()).compile_sampler(seed=4817).sample(8192)
    assert_joint_distribution(reference.astype(np.uint8), expected)


def _encoded_parities() -> tuple[str, int, int]:
    lines = ["H 0 1 2 3"]
    for mask in range(1, 16):
        target = mask + 3
        lines.extend(f"CX {q} {target}" for q in range(4) if mask & (1 << q))
    before = len(lines) - 1
    lines.extend(
        ["T " + " ".join(map(str, range(4, 19))), "T 18", "MX " + " ".join(map(str, range(19)))]
    )
    return "\n".join(lines), before, before + 2


def test_one_magic_qubit_does_not_bound_clifford_geometry_count() -> None:
    source, before, after = _encoded_parities()
    pre = compile_frame(source, [FaultSite(before, "X", q, 0.01) for q in range(4, 19)])
    post = compile_frame(source, [FaultSite(after, "Y", q, 0.01) for q in range(4, 19)])
    assert pre.core_width == post.core_width == 1
    assert pre.geometry_rank() == 10
    assert post.geometry_rank() == 0
    with pytest.raises(ValueError, match="leaf cap"):
        BranchBank(pre, [])


@pytest.mark.parametrize(
    "site",
    [
        FaultSite(-1, "X", 0, 0.1),
        FaultSite(0, "A", 0, 0.1),
        FaultSite(0, "X", 9, 0.1),
        FaultSite(0, "X", 0, float("nan")),
    ],
)
def test_invalid_fault_models_are_rejected(site: FaultSite) -> None:
    with pytest.raises(ValueError):
        compile_frame("H 0\nT 0\nMX 0", [site])


def test_affine_sampler_matches_all_noise_patterns_and_stochastic_aer_mixture() -> None:
    from tools.prototypes.symbolic_fault_study import AffineRecordSampler, affine_record_flips

    source = _single_core_source()
    lines = source.splitlines()
    last_t = max(j for j, line in enumerate(lines) if line.startswith("T "))
    frame = compile_frame(
        source, [FaultSite(last_t, "Y", 0, 0.17), FaultSite(last_t + 1, "Z", 3, 0.21)]
    )
    flips = affine_record_flips(frame)
    assert flips is not None
    ideal = _aer_branches(frame.render(0), 5)
    distribution = np.zeros(256)
    for pattern in range(4):
        expected = _aer_branches(frame.original(pattern), 5)
        changed = tuple((mask & pattern).bit_count() % 2 for mask in flips)
        relabeled = {
            tuple(bit ^ flip for bit, flip in zip(record, changed)): np.vdot(state, state).real
            for record, state in ideal.items()
        }
        assert expected.keys() == relabeled.keys()
        weight = np.prod(
            [
                site.probability if pattern & (1 << j) else 1 - site.probability
                for j, site in enumerate(frame.sites)
            ]
        )
        for record, state in expected.items():
            probability = np.vdot(state, state).real
            np.testing.assert_allclose(relabeled[record], probability, atol=1e-12)
            distribution[sum(bit << j for j, bit in enumerate(record))] += weight * probability
    sampler = AffineRecordSampler(frame, [1, 1, 1])
    assert sampler.program.peak_active_width <= 1
    assert_joint_distribution(sampler.sample_measurements(8192, seed=29417), distribution)
    passed = sum(p for key, p in enumerate(distribution) if not key & 7)
    result = sampler.sample_survivors(8192, seed=481)
    assert abs(result["passed_shots"] / 8192 - passed) < 0.025


def test_quadratic_fault_signs_and_geometry_changes_are_not_treated_as_affine() -> None:
    from tools.prototypes.symbolic_fault_study import Mod4Form, _half_affine, affine_record_flips

    assert _half_affine(Mod4Form(2, ((1, 2), (2, 2))), 2) == (1, 3)
    assert _half_affine(Mod4Form(0, ((1, 1), (2, 1), (3, 1))), 2) is None
    source, before, _ = _encoded_parities()
    frame = compile_frame(source, [FaultSite(before, "X", q, 0.01) for q in range(4, 19)])
    assert affine_record_flips(frame) is None


def test_core_only_signed_t_noise_has_an_affine_pauli_orbit() -> None:
    from tools.prototypes.symbolic_fault_study import AffineRecordSampler, affine_record_flips

    source = "H 0 1\nT 0 1\nCX 0 2\nMPP X0*X2\nMX 0 1 2"
    sites = [FaultSite(0, "X", 0, 0.1), FaultSite(0, "Z", 1, 0.2)]
    frame = compile_frame(source, sites)
    flips = affine_record_flips(frame)
    assert flips is not None
    sampler = AffineRecordSampler(frame, [])
    assert sampler.program.peak_active_width <= 2
    ideal = _aer_branches(frame.render(0), 3)
    for pattern in range(4):
        expected = _aer_branches(frame.original(pattern), 3)
        changed = tuple((mask & pattern).bit_count() % 2 for mask in flips)
        relabeled = {
            tuple(bit ^ flip for bit, flip in zip(record, changed)): np.vdot(state, state).real
            for record, state in ideal.items()
        }
        assert expected.keys() == relabeled.keys()
        for record, state in expected.items():
            np.testing.assert_allclose(relabeled[record], np.vdot(state, state).real, atol=1e-12)


def test_offline_reverse_geometry_rank_matches_explicit_fault_models() -> None:
    from tools.prototypes.symbolic_fault_study import (
        gate_noise_geometry_rank,
        gate_noise_geometry_stats,
    )

    source = "H 0 1 2\nCX 0 1\nT 0\nCX 1 2\nT_DAG 2\nCZ 0 2\nMX 0 1 2"
    sites = [
        FaultSite(j, "X", int(q), 0.01)
        for j, line in enumerate(source.splitlines())
        if not line.startswith("MX")
        for q in line.split()[1:]
    ]
    frame = compile_frame(source, sites)
    assert gate_noise_geometry_rank(source) == frame.geometry_rank()
    assert gate_noise_geometry_stats(source)["fixed_frame_prepared_core_width"] == (
        frame.fixed_frame_core_width()
    )


def test_common_stabilizer_bound_matches_exhaustive_aer_states() -> None:
    from qiskit.quantum_info import Pauli
    from test_phase_core_study import _aer_unitary

    frame = _small_model()
    states = []
    for pattern in range(1 << len(frame.sites)):
        source = "\n".join(
            line
            for line in frame.original(pattern).splitlines()
            if not line.startswith(("MPP", "MX", "DETECTOR", "OBSERVABLE_INCLUDE"))
        )
        states.append(_aer_unitary(source, 5)[:, 0])
    common = 0
    for axes in itertools.product("IXYZ", repeat=5):
        pauli = Pauli("".join(axes)).to_matrix()
        if all(abs(abs(np.vdot(state, pauli @ state)) - 1) < 1e-12 for state in states):
            common += 1
    assert frame.fixed_frame_core_width() == 5 - round(np.log2(common))
    source, before, after = _encoded_parities()
    pre = compile_frame(source, [FaultSite(before, "X", q, 0.01) for q in range(4, 19)])
    post = compile_frame(source, [FaultSite(after, "Y", q, 0.01) for q in range(4, 19)])
    assert pre.fixed_frame_core_width() == 4
    assert post.fixed_frame_core_width() == 1


def test_dephasing_throughout_the_fragment_has_fixed_record_corrections() -> None:
    import random

    from tools.prototypes.symbolic_fault_study import AffineRecordSampler, affine_record_flips

    source = _single_core_source()
    sites = [
        FaultSite(j, "Z", int(q), 0.001)
        for j, line in enumerate(source.splitlines())
        if line.split()[0] in ("H", "CX", "T", "CZ", "S")
        for q in line.split()[1:]
    ]
    frame = compile_frame(source, sites)
    flips = affine_record_flips(frame)
    assert flips is not None and frame.geometry_rank() == 0
    assert frame.fixed_frame_core_width() == frame.core_width == 1
    assert AffineRecordSampler(frame, [1, 1, 1]).program.peak_active_width <= 1
    ideal = _aer_branches(frame.render(0), 5)
    rng = random.Random(31391)
    for pattern in [0, (1 << len(sites)) - 1] + [rng.getrandbits(len(sites)) for _ in range(8)]:
        changed = tuple((mask & pattern).bit_count() % 2 for mask in flips)
        relabeled = {
            tuple(bit ^ flip for bit, flip in zip(record, changed)): np.vdot(state, state).real
            for record, state in ideal.items()
        }
        expected = _aer_branches(frame.original(pattern), 5)
        assert expected.keys() == relabeled.keys()
        for record, state in expected.items():
            np.testing.assert_allclose(relabeled[record], np.vdot(state, state).real, atol=1e-12)
