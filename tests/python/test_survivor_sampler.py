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


@pytest.mark.parametrize(
    "strategy",
    [
        {},
        {"compact_outputs": True},
        {"group_by_noise": True},
        {"compact_outputs": True, "group_by_noise": True},
    ],
)
@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("early", [False, True])
def test_worker_count_and_storage_growth_preserve_seeded_counts(sharing, early, strategy):
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
        **strategy,
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
        clifft.compile(SOURCE),
        batch_size=65,
        threads=2,
        initial_states=1,
        compact_outputs=True,
        group_by_noise=True,
    )
    gc.collect()
    expected = counts(sampler.sample(1301, seed=17))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: counts(sampler.sample(1301, seed=17)), range(6)))
    assert results == [expected] * 6


@pytest.mark.parametrize("compact", [False, True])
def test_no_observables_and_empty_action_stream(compact):
    sampler = clifft.SurvivorSampler(
        clifft.compile(""), batch_size=65, compact_outputs=compact, group_by_noise=True
    )
    for shots in (0, 1, 131):
        result = sampler.sample(shots, seed=9)
        assert result.passed_shots == shots
        assert result.logical_errors == 0
        assert result.observable_ones.shape == (0,)
        assert result.measurements.shape == (0, 0)


@pytest.mark.parametrize("mode", [clifft.NoisePreparation.DENSE, clifft.NoisePreparation.SPARSE])
@pytest.mark.parametrize("compact", [False, True])
def test_width_zero_reuses_packed_execution_with_exact_counts(mode, compact):
    program = clifft.compile(
        "H 0\nCX 0 1\nDEPOLARIZE2(0.07) 0 1\nM(0.13) 0 1\n"
        "DETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        postselection_mask=[1],
        expected_detectors=[1],
    )
    assert program.sampling_plan(keep_records=False).peak_active_width == 0
    sampler = clifft.SurvivorSampler(
        program, batch_size=65, noise_preparation=mode, compact_outputs=compact
    )
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
@pytest.mark.parametrize("compact", [False, True])
def test_detector_snapshot_precedes_asymmetric_readout(reconstruction, compact):
    program = clifft.compile(
        "H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n"
        "READOUT_NOISE(0.2, 0.3) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        reconstruction=reconstruction,
        postselection_mask=[1],
        expected_detectors=[1],
        expected_observables=[1],
    )
    result = clifft.SurvivorSampler(
        program, batch_size=257, initial_states=1, compact_outputs=compact, group_by_noise=True
    ).sample(32768, seed=24)
    probability = np.sin(np.pi / 8) ** 2
    for count, mass in [
        (result.passed_shots, probability),
        (result.logical_errors, probability * 0.3),
    ]:
        tolerance = 7 * np.sqrt(32768 * mass * (1 - mass)) + 10
        assert abs(count - 32768 * mass) <= tolerance
    np.testing.assert_array_equal(result.observable_ones, [result.logical_errors])


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": -1},
        {"batch_size": 2**32},
        {"batch_size": "unknown"},
        {"initial_states": -1},
        {"initial_states": 2**32},
        {"threads": -1},
        {"max_state_bytes": 0},
        {"max_state_bytes": -1},
        {"max_state_bytes": 1},
        {"tuning_budget_seconds": 0},
        {"batch_size": "tune", "tuning_budget_seconds": -1},
        {"batch_size": "tune", "tuning_budget_seconds": float("nan")},
    ],
)
def test_configuration_uses_existing_option_validation(options):
    with pytest.raises(ValueError):
        clifft.SurvivorSampler(clifft.compile(SOURCE), **options)


def test_state_budget_is_shared_across_workers_and_failure_allows_reuse():
    program = clifft.compile(SOURCE)
    # This narrow plan needs one 4096-byte allocation per shared state.
    with pytest.raises(ValueError):
        clifft.SurvivorSampler(program, threads=2, max_state_bytes=4096)
    sampler = clifft.SurvivorSampler(
        program, batch_size=65, threads=2, max_state_bytes=8192, sharing=False
    )
    with pytest.raises(RuntimeError):
        sampler.sample(130, seed=23)
    assert sampler.sample(1, seed=23).passed_shots == 1
    auto = clifft.SurvivorSampler(program, threads=2, max_state_bytes=8192)
    explicit = clifft.SurvivorSampler(program, batch_size=1, threads=2, max_state_bytes=8192)
    assert counts(auto.sample(131, seed=23)) == counts(explicit.sample(131, seed=23))


def test_tuning_is_reused_and_replays_at_the_selected_numeric_capacity():
    program = clifft.compile(SOURCE)
    sampler = clifft.SurvivorSampler(program, batch_size="tune", tuning_budget_seconds=0.15)
    assert sampler.sample(0, seed=25).batch_tuning is None
    result = sampler.sample(1301, seed=25)
    report = result.batch_tuning
    assert report is not None
    assert report.trial_shots > 0
    assert len(report.trials) >= 2
    assert result.total_shots == 1301
    explicit = clifft.SurvivorSampler(program, batch_size=report.batch_size)
    assert counts(result) == counts(explicit.sample(1301, seed=25))
    repeated = sampler.sample(1301, seed=25)
    assert counts(repeated) == counts(result)
    assert repeated.batch_tuning.trial_shots == report.trial_shots


def test_zero_budget_tuning_uses_auto_and_the_existing_warning():
    program = clifft.compile(SOURCE)
    sampler = clifft.SurvivorSampler(program, batch_size="tune", tuning_budget_seconds=0)
    with pytest.warns(RuntimeWarning):
        result = sampler.sample(1301, seed=25)
    assert result.batch_tuning.trial_shots == 0
    assert not result.batch_tuning.sufficient_measurements
    assert counts(result) == counts(clifft.SurvivorSampler(program).sample(1301, seed=25))
