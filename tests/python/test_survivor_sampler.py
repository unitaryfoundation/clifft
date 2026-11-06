"""Reusable aggregate sampling, ownership, and fixed-configuration replay."""

import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import clifft

SOURCE = (
    "H 0 1\nT 0 1\nDEPOLARIZE1(0.04) 0 1\nH 0 1\nM(0.13) 0 1\n"
    "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
)


def counts(result):
    return (
        result.total_shots,
        result.passed_shots,
        result.logical_errors,
        tuple(result.observable_ones),
    )


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("early", [False, True])
def test_worker_count_and_storage_growth_preserve_seeded_counts(sharing, early):
    program = clifft.compile(SOURCE, postselection_mask=[1])
    reference = clifft.SurvivorSampler(program, batch_size=65, initial_states=65).sample(
        1301, seed=7
    )
    sampler = clifft.SurvivorSampler(
        program,
        batch_size=65,
        initial_states=1,
        threads=3,
        sharing=sharing,
        early_postselection=early,
    )
    for shots in (1301, 0, 1, 63, 65, 1301):
        result = sampler.sample(shots, seed=7)
        assert result.total_shots == shots
        assert result.measurements.shape == (0, program.num_measurements)
        assert result.detectors.shape == (0, program.num_detectors)
        assert result.observables.shape == (0, program.num_observables)
        assert result.exp_vals.shape == (0, 0)
        if shots == 1301:
            assert counts(result) == counts(reference)
        if shots == 0:
            assert result.passed_shots == result.logical_errors == result.discards == 0
            np.testing.assert_array_equal(result.observable_ones, [0])


def test_noise_layouts_preserve_their_corresponding_seeded_stream():
    program = clifft.compile(SOURCE, postselection_mask=[1])
    results = {
        mode: clifft.SurvivorSampler(program, batch_size=65, noise_preparation=mode).sample(
            1301, seed=9
        )
        for mode in clifft.NoisePreparation
    }
    assert counts(results[clifft.NoisePreparation.DENSE]) == counts(
        results[clifft.NoisePreparation.SPARSE]
    )
    assert counts(results[clifft.NoisePreparation.SITE_SPARSE]) == counts(
        results[clifft.NoisePreparation.SITE_ROWS]
    )


def test_program_ownership_and_concurrent_calls():
    sampler = clifft.SurvivorSampler(
        clifft.compile(SOURCE), batch_size=65, threads=2, initial_states=1
    )
    gc.collect()
    expected = counts(sampler.sample(1301, seed=17))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: counts(sampler.sample(1301, seed=17)), range(6)))
    assert results == [expected] * 6


def test_no_observables_and_empty_action_stream():
    sampler = clifft.SurvivorSampler(clifft.compile(""), batch_size=65)
    for shots in (0, 1, 131):
        result = sampler.sample(shots, seed=9)
        assert result.passed_shots == shots
        assert result.logical_errors == 0
        assert result.observable_ones.shape == (0,)
        assert result.measurements.shape == (0, 0)


@pytest.mark.parametrize("mode", [clifft.NoisePreparation.DENSE, clifft.NoisePreparation.SPARSE])
def test_width_zero_reuses_packed_execution_with_exact_counts(mode):
    program = clifft.compile(
        "H 0\nCX 0 1\nDEPOLARIZE2(0.07) 0 1\nM(0.13) 0 1\n"
        "DETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        postselection_mask=[1],
        expected_detectors=[1],
    )
    assert program.sampling_plan(keep_records=False).peak_active_width == 0
    sampler = clifft.SurvivorSampler(program, batch_size=65, noise_preparation=mode)
    # Both interfaces retain capacity 65, including the final partial batch.
    for shots in (65, 130, 193, 65):
        expected = clifft.sample_survivors(
            program, shots, seed=19, batch_size=65, keep_records=False
        )
        assert counts(sampler.sample(shots, seed=19)) == counts(expected)


@pytest.mark.parametrize("options", [{"batch_size": 0}, {"initial_states": 0}, {"threads": 0}])
def test_invalid_configuration_is_rejected_before_sampling(options):
    with pytest.raises(ValueError):
        clifft.SurvivorSampler(clifft.compile(SOURCE), **options)


@pytest.mark.parametrize("reconstruction", [False, True])
def test_detector_snapshot_precedes_asymmetric_readout(reconstruction):
    program = clifft.compile(
        "H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n"
        "READOUT_NOISE(0.2, 0.3) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        reconstruction=reconstruction,
        postselection_mask=[1],
        expected_detectors=[1],
        expected_observables=[1],
    )
    result = clifft.SurvivorSampler(program, batch_size=257, initial_states=1).sample(
        32768, seed=24
    )
    probability = np.sin(np.pi / 8) ** 2
    for count, mass in [
        (result.passed_shots, probability),
        (result.logical_errors, probability * 0.3),
    ]:
        tolerance = 7 * np.sqrt(32768 * mass * (1 - mass)) + 10
        assert abs(count - 32768 * mass) <= tolerance
    np.testing.assert_array_equal(result.observable_ones, [result.logical_errors])
