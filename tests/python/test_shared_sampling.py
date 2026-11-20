"""Integrated shared-trajectory execution, overrides, validation, and inspection."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from utils_conformance import assert_joint_distribution, unpack_bits

import clifft

SOURCE = (
    "H 0 1\nT 0 1\nDEPOLARIZE1(0.04) 0 1\nH 0 1\nM(0.13) 0 1\n"
    "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
)

UNPOSTSELECTED_SOURCE = (
    "H 0 1 2\nT 0 1 2\nDEPOLARIZE1(0.04) 0 1 2\nH 0 1 2\nM(0.11) 0 1 2\n"
    "DETECTOR rec[-1] rec[-2]\nDETECTOR rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-3]\n"
)

WIDTH_ZERO_SOURCE = (
    "H 0\nCX 0 1\nDEPOLARIZE2(0.07) 0 1\nM(0.13) 0 1\n"
    "DETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
)


def counts(result: clifft.SampleResult) -> tuple[int, int, int, tuple[int, ...]]:
    assert result.total_shots is not None
    assert result.passed_shots is not None
    assert result.logical_errors is not None
    assert result.observable_ones is not None
    return (
        result.total_shots,
        result.passed_shots,
        result.logical_errors,
        tuple(int(x) for x in result.observable_ones),
    )


@pytest.mark.parametrize("group_by_noise", [False, True])
@pytest.mark.parametrize("readout", [0.001, 0.13])
def test_worker_count_and_storage_growth_preserve_seeded_counts_and_rows(
    group_by_noise: bool, readout: float
) -> None:
    program = clifft.compile(SOURCE.replace("M(0.13)", f"M({readout})"), postselection_mask=[1])
    reference = clifft.sample_survivors(
        program,
        1301,
        seed=7,
        keep_records=True,
        batch_size=65,
        threads=1,
        sharing=True,
        group_by_noise=group_by_noise,
    )
    assert reference.execution is not None
    assert reference.execution.executor == "shared"
    assert reference.execution.group_by_noise is group_by_noise

    for shots in (1301, 0, 1, 63, 65, 1301):
        count_only = clifft.sample_survivors(
            program,
            shots,
            seed=7,
            batch_size=65,
            threads=3,
            sharing=True,
            group_by_noise=group_by_noise,
        )
        assert count_only.total_shots == shots
        assert count_only.measurements.shape == (0, program.num_measurements)
        assert count_only.detectors.shape == (0, program.num_detectors)
        assert count_only.observables.shape == (0, program.num_observables)
        assert count_only.exp_vals.shape == (0, 0)
        if shots == 1301:
            assert counts(count_only) == counts(reference)
            retained = clifft.sample_survivors(
                program,
                shots,
                seed=7,
                keep_records=True,
                batch_size=65,
                threads=3,
                sharing=True,
                group_by_noise=group_by_noise,
            )
            assert counts(retained) == counts(reference)
            np.testing.assert_array_equal(retained.measurements, reference.measurements)
            np.testing.assert_array_equal(retained.detectors, reference.detectors)
            np.testing.assert_array_equal(retained.observables, reference.observables)
        if shots == 0:
            assert count_only.passed_shots == count_only.logical_errors == count_only.discards == 0
            assert count_only.execution.shot_workers == 0
            np.testing.assert_array_equal(count_only.observable_ones, [0])


def test_noise_preparations_preserve_seeded_streams_and_ordinary_compatibility() -> None:
    survivor_program = clifft.compile(SOURCE, postselection_mask=[1])
    survivor_results = {
        mode: clifft.sample_survivors(
            survivor_program, 1301, seed=9, batch_size=65, sharing=True, noise_preparation=mode
        )
        for mode in clifft.NoisePreparation
    }
    assert counts(survivor_results[clifft.NoisePreparation.DENSE]) == counts(
        survivor_results[clifft.NoisePreparation.SPARSE]
    )
    assert counts(survivor_results[clifft.NoisePreparation.SITE_SPARSE]) == counts(
        survivor_results[clifft.NoisePreparation.SITE_ROWS]
    )
    for mode, result in survivor_results.items():
        assert result.execution.noise_preparation == mode

    fixed_program = clifft.compile(UNPOSTSELECTED_SOURCE)
    sparse = clifft.sample(
        fixed_program,
        513,
        seed=11,
        batch_size=65,
        sharing=True,
        noise_preparation=clifft.NoisePreparation.SITE_SPARSE,
    )
    rows = clifft.sample(
        fixed_program,
        513,
        seed=11,
        batch_size=65,
        sharing=True,
        noise_preparation=clifft.NoisePreparation.SITE_ROWS,
    )
    np.testing.assert_array_equal(sparse.measurements, rows.measurements)
    np.testing.assert_array_equal(sparse.detectors, rows.detectors)
    np.testing.assert_array_equal(sparse.observables, rows.observables)

    # With sharing='auto', an explicit batch_size preserves ordinary packed
    # execution, its 2048 clamp, and Dense noise preparation.
    ordinary_default = clifft.sample(fixed_program, 4096, seed=13, batch_size=4096)
    ordinary_explicit = clifft.sample(
        fixed_program,
        4096,
        seed=13,
        batch_size=2048,
        sharing=False,
        noise_preparation=clifft.NoisePreparation.DENSE,
    )
    assert ordinary_default.execution.executor == "packed"
    assert ordinary_default.execution.batch_size == 2048
    assert ordinary_default.execution.noise_preparation == clifft.NoisePreparation.DENSE
    assert ordinary_default.execution.max_worker_state_bytes is None
    assert ordinary_default.execution == ordinary_explicit.execution
    np.testing.assert_array_equal(ordinary_default.measurements, ordinary_explicit.measurements)
    ordinary_sparse = clifft.sample(
        fixed_program,
        4096,
        seed=13,
        batch_size=2048,
        noise_preparation=clifft.NoisePreparation.SPARSE,
    )
    assert ordinary_sparse.execution.noise_preparation == clifft.NoisePreparation.SPARSE
    np.testing.assert_array_equal(ordinary_default.measurements, ordinary_sparse.measurements)

    shared_large = clifft.sample(fixed_program, 4096, seed=13, batch_size=4096, sharing=True)
    assert shared_large.execution.executor == "shared"
    assert shared_large.execution.batch_size == 4096
    assert shared_large.execution.noise_preparation == clifft.NoisePreparation.SITE_SPARSE


def test_preparation_reuse_and_concurrent_first_use_preserve_plans_and_streams() -> None:
    def compiled() -> clifft.Program:
        return clifft.compile(SOURCE, postselection_mask=[1])

    def shared_counts(
        program: clifft.Program, **options: object
    ) -> tuple[int, int, int, tuple[int, ...]]:
        return counts(
            clifft.sample_survivors(program, 1301, seed=7, batch_size=65, sharing=True, **options)
        )

    cold = shared_counts(compiled())
    program = compiled()
    untouched = compiled()
    inspected = program.sampling_plan(keep_records=False).inspect()

    shared_counts(program, noise_preparation=clifft.NoisePreparation.SITE_ROWS, group_by_noise=True)
    shared_counts(program, noise_preparation=clifft.NoisePreparation.DENSE, group_by_noise=False)
    assert shared_counts(program) == cold
    assert 0 < cold[1] < 1301
    assert program.sampling_plan(keep_records=False).inspect() == inspected
    assert program.num_actions == untouched.num_actions

    for keep_records in (False, True):
        expected = clifft.sample_survivors(
            untouched, 1301, seed=5, keep_records=keep_records, sharing=False
        )
        actual = clifft.sample_survivors(
            program, 1301, seed=5, keep_records=keep_records, sharing=False
        )
        assert counts(actual) == counts(expected)
        np.testing.assert_array_equal(actual.measurements, expected.measurements)
        np.testing.assert_array_equal(actual.detectors, expected.detectors)
        np.testing.assert_array_equal(actual.observables, expected.observables)

    concurrent_program = compiled()
    expected_concurrent = shared_counts(compiled(), threads=2, group_by_noise=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: shared_counts(concurrent_program, threads=2, group_by_noise=True),
                range(16),
            )
        )
    assert results == [expected_concurrent] * 16


def test_no_observables_and_empty_action_stream() -> None:
    program = clifft.compile("")
    for shots in (0, 1, 131):
        result = clifft.sample_survivors(
            program, shots, seed=9, batch_size=65, sharing=True, group_by_noise=True
        )
        assert result.passed_shots == shots
        assert result.logical_errors == 0
        assert result.observable_ones.shape == (0,)
        assert result.measurements.shape == (0, 0)
        assert result.execution.executor == "packed"
        assert result.execution.sharing is True
        assert result.execution.batch_size >= 2
        assert result.execution.max_worker_state_bytes is None
        assert result.execution.group_by_noise is False


@pytest.mark.parametrize("mode", list(clifft.NoisePreparation))
def test_width_zero_forced_sharing_uses_packed_batches_under_shared_policy(
    mode: clifft.NoisePreparation,
) -> None:
    program = clifft.compile(
        WIDTH_ZERO_SOURCE,
        postselection_mask=[1],
        expected_detectors=[1],
    )
    assert program.sampling_plan(keep_records=False).peak_active_width == 0
    for shots in (1, 65, 130, 193, 4096):
        shared = clifft.sample_survivors(
            program,
            shots,
            seed=19,
            batch_size=shots if shots == 4096 else 65,
            sharing=True,
            group_by_noise=True,
            noise_preparation=mode,
            max_worker_state_bytes=1,
        )
        assert shared.execution.executor == "packed"
        assert shared.execution.sharing is True
        assert shared.execution.batch_size >= 2
        assert shared.execution.noise_preparation == mode
        assert shared.execution.group_by_noise is False
        assert shared.execution.max_worker_state_bytes is None
        if shots == 4096:
            assert shared.execution.batch_size == 4096
        elif (
            mode in (clifft.NoisePreparation.DENSE, clifft.NoisePreparation.SPARSE) and shots >= 65
        ):
            expected = clifft.sample_survivors(
                program, shots, seed=19, batch_size=65, keep_records=False
            )
            assert expected.execution.sharing is False
            assert counts(shared) == counts(expected)

    if mode == clifft.NoisePreparation.SITE_SPARSE:
        auto_wide = clifft.sample_survivors(program, 20_000, seed=19, sharing=True)
        assert auto_wide.execution.executor == "packed"
        assert auto_wide.execution.sharing is True
        assert auto_wide.execution.batch_size == 8192
        replayed_wide = clifft.sample_survivors(
            program,
            20_000,
            seed=19,
            batch_size=auto_wide.execution.batch_size,
            sharing=auto_wide.execution.sharing,
            group_by_noise=auto_wide.execution.group_by_noise,
            noise_preparation=auto_wide.execution.noise_preparation,
        )
        assert replayed_wide.execution == auto_wide.execution
        assert counts(replayed_wide) == counts(auto_wide)

    # Circuits reconstructed to width zero when measurement rows are omitted
    # also succeed under forced sharing and group_by_noise=True.
    reconstructed = clifft.compile(
        "H 0\nT 0\nH 0\nDEPOLARIZE1(0.03) 0\nM 0\nH 1\nM(0.05) 1\n"
        "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        reconstruction=True,
        postselection_mask=[1],
        expected_detectors=[0],
    )
    assert reconstructed.sampling_plan(keep_records=True).peak_active_width > 0
    assert reconstructed.sampling_plan(keep_records=False).peak_active_width == 0
    result = clifft.sample_survivors(reconstructed, 257, seed=21, sharing=True, group_by_noise=True)
    assert result.execution.executor == "packed"
    assert result.execution.sharing is True
    assert result.execution.max_worker_state_bytes is None


@pytest.mark.parametrize("reconstruction", [False, True])
def test_detector_snapshot_precedes_asymmetric_readout(reconstruction: bool) -> None:
    program = clifft.compile(
        "H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n"
        "READOUT_NOISE(0.2, 0.3) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        reconstruction=reconstruction,
        postselection_mask=[1],
        expected_detectors=[1],
        expected_observables=[1],
    )
    result = clifft.sample_survivors(
        program, 32768, seed=24, batch_size=257, sharing=True, group_by_noise=True
    )
    probability = np.sin(np.pi / 8) ** 2
    for count, mass in [
        (result.passed_shots, probability),
        (result.logical_errors, probability * 0.3),
    ]:
        tolerance = 7 * np.sqrt(32768 * mass * (1 - mass)) + 10
        assert abs(count - 32768 * mass) <= tolerance
    np.testing.assert_array_equal(result.observable_ones, [result.logical_errors])


@pytest.mark.parametrize("keep_records", [True, "all", "syndromes", "measurements"])
@pytest.mark.parametrize("bit_packed", [False, True])
def test_retained_outputs_and_sample_detectors_forward_shared_controls(
    keep_records: bool | str, bit_packed: bool
) -> None:
    fixed = clifft.compile(UNPOSTSELECTED_SOURCE)
    survivor = clifft.compile(SOURCE, postselection_mask=[1])

    fixed_result = clifft.sample(
        fixed,
        257,
        seed=31,
        batch_size=65,
        threads=2,
        keep_records=keep_records,
        bit_packed=bit_packed,
        sharing=True,
        group_by_noise=True,
        noise_preparation=clifft.NoisePreparation.SITE_ROWS,
        max_worker_state_bytes=8 * 1024 * 1024,
    )
    expected_keep = "all" if keep_records is True else keep_records
    assert fixed_result.execution.executor == "shared"
    assert fixed_result.execution.keep_records == expected_keep
    assert fixed_result.execution.bit_packed is bit_packed
    assert fixed_result.execution.group_by_noise is True
    assert fixed_result.execution.noise_preparation == clifft.NoisePreparation.SITE_ROWS
    assert fixed_result.execution.max_worker_state_bytes == 8 * 1024 * 1024

    if bit_packed:
        unpacked = clifft.sample(
            fixed,
            257,
            seed=31,
            batch_size=65,
            threads=2,
            keep_records=keep_records,
            bit_packed=False,
            sharing=True,
            group_by_noise=True,
            noise_preparation=clifft.NoisePreparation.SITE_ROWS,
            max_worker_state_bytes=8 * 1024 * 1024,
        )
        if expected_keep in ("all", "measurements"):
            np.testing.assert_array_equal(
                unpack_bits(fixed_result.measurements, fixed.num_measurements),
                unpacked.measurements,
            )
        if expected_keep in ("all", "syndromes"):
            np.testing.assert_array_equal(
                unpack_bits(fixed_result.detectors, fixed.num_detectors),
                unpacked.detectors,
            )
            np.testing.assert_array_equal(
                unpack_bits(fixed_result.observables, fixed.num_observables),
                unpacked.observables,
            )

    survivor_result = clifft.sample_survivors(
        survivor,
        257,
        seed=37,
        keep_records=keep_records,
        batch_size=65,
        threads=2,
        bit_packed=bit_packed,
        sharing=True,
        group_by_noise=True,
    )
    assert survivor_result.execution.executor == "shared"
    assert survivor_result.execution.keep_records == expected_keep
    assert survivor_result.execution.bit_packed is bit_packed
    if expected_keep in ("all", "measurements"):
        assert survivor_result.measurements.shape[0] == survivor_result.passed_shots
    else:
        assert survivor_result.measurements.shape[0] == 0
    if expected_keep in ("all", "syndromes"):
        assert survivor_result.detectors.shape[0] == survivor_result.passed_shots
        assert survivor_result.observables.shape[0] == survivor_result.passed_shots
    else:
        assert survivor_result.detectors.shape[0] == 0
        assert survivor_result.observables.shape[0] == 0

    if keep_records == "syndromes":
        detectors, observables = clifft.sample_detectors(
            fixed,
            257,
            seed=31,
            batch_size=65,
            threads=2,
            separate_observables=True,
            bit_packed=bit_packed,
            sharing=True,
            group_by_noise=True,
            noise_preparation=clifft.NoisePreparation.SITE_ROWS,
            max_worker_state_bytes=8 * 1024 * 1024,
        )
        np.testing.assert_array_equal(detectors, fixed_result.detectors)
        np.testing.assert_array_equal(observables, fixed_result.observables)


def test_execution_summary_is_read_only_and_covers_all_cpu_entrypoints() -> None:
    program = clifft.compile(UNPOSTSELECTED_SOURCE)
    scalar = clifft.sample(program, 5, seed=3, batch_size=1, keep_records="syndromes")
    assert scalar.execution.executor == "scalar"
    assert scalar.execution.sharing is False
    assert scalar.execution.batch_size == 1
    assert scalar.execution.shot_workers == 1
    assert scalar.execution.intra_shot_workers == 1
    assert scalar.execution.keep_records == "syndromes"
    assert not scalar.execution.bit_packed
    assert scalar.execution.noise_preparation is None
    assert not scalar.execution.group_by_noise
    assert scalar.execution.max_worker_state_bytes is None
    assert "sharing=False" in repr(scalar.execution)
    with pytest.raises(AttributeError):
        scalar.execution.batch_size = 2
    with pytest.raises(AttributeError):
        scalar.execution.sharing = True

    k_fixed = clifft.sample_k(program, 16, 1, seed=3, batch_size=1)
    assert k_fixed.execution.executor == "scalar"
    assert k_fixed.execution.sharing is False
    assert k_fixed.execution.keep_records == "all"

    k_survivors = clifft.sample_k_survivors(
        clifft.compile(SOURCE, postselection_mask=[1]), 16, 1, seed=3, keep_records=False
    )
    assert k_survivors.execution is not None
    assert k_survivors.execution.sharing is False
    assert k_survivors.execution.keep_records is False


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": 2**32},
        {"batch_size": "unknown"},
        {"threads": 0},
        {"threads": -1},
        {"max_worker_state_bytes": 0},
        {"max_worker_state_bytes": -1},
        {"sharing": "unknown"},
        {"group_by_noise": "unknown"},
        {"noise_preparation": "unknown"},
        {"noise_preparation": "sparse"},
        {"tuning_budget_seconds": 0},
        {"batch_size": "tune", "tuning_budget_seconds": -1},
        {"batch_size": "tune", "tuning_budget_seconds": float("nan")},
        {"batch_size": 1, "sharing": True},
        {"batch_size": 1, "group_by_noise": True},
        {"batch_size": 1, "noise_preparation": clifft.NoisePreparation.DENSE},
        {"group_by_noise": True},
        {"sharing": False, "group_by_noise": True},
    ],
)
@pytest.mark.parametrize("shots", [0, 65])
def test_invalid_options_and_incompatible_combinations_fail_before_dispatch(
    options: dict[str, object], shots: int
) -> None:
    quantum = clifft.compile(UNPOSTSELECTED_SOURCE)
    classical = clifft.compile(WIDTH_ZERO_SOURCE)
    for program in (quantum, classical):
        with pytest.raises(ValueError):
            clifft.sample(program, shots, **options)
        with pytest.raises(ValueError):
            clifft.sample_survivors(program, shots, **options)


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 1.5},
        {"max_worker_state_bytes": "auto"},
        {"sharing": 1},
        {"group_by_noise": 1},
        {"noise_preparation": 0},
        {"noise_preparation": 1},
        {"noise_preparation": False},
    ],
)
def test_wrong_option_types_raise_type_error(options: dict[str, object]) -> None:
    program = clifft.compile(UNPOSTSELECTED_SOURCE)
    with pytest.raises(TypeError):
        clifft.sample(program, 8, **options)
    with pytest.raises(TypeError):
        clifft.sample_survivors(program, 8, **options)


def test_expectation_probes_intra_shot_layouts_and_harmless_disables() -> None:
    probe_program = clifft.compile("H 0\nT 0\nEXP_VAL Z0\nM 0\nDETECTOR rec[-1]")
    for shots in (0, 64):
        with pytest.raises(ValueError, match="EXP_VAL"):
            clifft.sample(probe_program, shots, sharing=True, keep_records=True)
        with pytest.raises(ValueError, match="EXP_VAL"):
            clifft.sample_survivors(probe_program, shots, sharing=True, keep_records=True)
    with pytest.raises(ValueError, match="batch_size=1"):
        clifft.sample(probe_program, 64, batch_size=1, sharing=True, keep_records="syndromes")
    # Omitting EXP_VAL outputs allows shared execution on the same compiled program.
    assert (
        clifft.sample(
            probe_program, 64, seed=5, sharing=True, keep_records="syndromes"
        ).execution.executor
        == "shared"
    )
    assert (
        clifft.sample_survivors(
            probe_program, 64, seed=5, sharing=True, keep_records=False
        ).execution.executor
        == "shared"
    )

    # Harmless disables succeed across scalar, ordinary packed, and width-zero plans.
    for program in (clifft.compile(UNPOSTSELECTED_SOURCE), clifft.compile(WIDTH_ZERO_SOURCE)):
        res = clifft.sample(program, 65, seed=5, batch_size=1, sharing=False, group_by_noise=False)
        assert res.execution.executor == "scalar"
        packed = clifft.sample(
            program,
            65,
            seed=5,
            batch_size=65,
            sharing=False,
            group_by_noise=False,
            noise_preparation=clifft.NoisePreparation.SITE_SPARSE,
        )
        assert packed.execution.executor == "packed"
        assert packed.execution.noise_preparation == clifft.NoisePreparation.SITE_SPARSE

    # If OpenMP intra-shot execution is compiled in, explicit intra-shot layouts
    # reject sharing=True and explicit noise_preparation deterministically.
    wide = clifft.compile(
        "".join(f"H {q}\nT {q}\n" for q in range(18)) + "M " + " ".join(str(q) for q in range(18))
    )
    try:
        probe = clifft.sample(wide, 1, seed=1, thread_layout=(1, 2))
    except ValueError:
        probe = None
    if probe is not None and probe.execution.intra_shot_workers > 1:
        for shots in (0, 4):
            with pytest.raises(ValueError, match="intra-shot"):
                clifft.sample(wide, shots, thread_layout=(1, 2), sharing=True)
            with pytest.raises(ValueError, match="intra-shot"):
                clifft.sample(
                    wide,
                    shots,
                    thread_layout=(1, 2),
                    noise_preparation=clifft.NoisePreparation.DENSE,
                )


def test_state_budget_applies_per_worker_and_split_overflow_leaves_program_reusable() -> None:
    program = clifft.compile(SOURCE)
    # This narrow plan needs one 4096-byte allocation per shared state (at least two states).
    for shots in (0, 65):
        with pytest.raises(ValueError, match="two shared states"):
            clifft.sample_survivors(
                program, shots, threads=2, sharing=True, max_worker_state_bytes=8191
            )
    # With sharing='auto', an insufficient state budget falls back cleanly.
    fallback = clifft.sample_survivors(program, 65, seed=23, max_worker_state_bytes=8191)
    assert fallback.execution.executor != "shared"
    assert fallback.execution.sharing is False

    with pytest.raises(RuntimeError):
        clifft.sample_survivors(
            program,
            130,
            seed=23,
            batch_size=65,
            threads=2,
            sharing=True,
            max_worker_state_bytes=8192,
        )
    recovered = clifft.sample_survivors(
        program, 1, seed=23, batch_size=65, threads=2, sharing=True, max_worker_state_bytes=8192
    )
    assert recovered.passed_shots == 1

    # Two states fit each worker's budget regardless of worker count.
    for threads in (1, 2, 4):
        auto = clifft.sample_survivors(
            program,
            131,
            seed=23,
            batch_size="auto",
            threads=threads,
            sharing=True,
            max_worker_state_bytes=8192,
        )
        explicit = clifft.sample_survivors(
            program,
            131,
            seed=23,
            batch_size=2,
            threads=threads,
            sharing=True,
            max_worker_state_bytes=8192,
        )
        assert auto.execution.batch_size == 2
        assert counts(auto) == counts(explicit)


def test_auto_shared_capacity_follows_each_call_and_ignores_call_history() -> None:
    program = clifft.compile(SOURCE)
    for shots in (1301, 257, 1301):
        auto = clifft.sample_survivors(
            program, shots, seed=29, batch_size="auto", threads=2, sharing=True
        )
        explicit = clifft.sample_survivors(
            program,
            shots,
            seed=29,
            batch_size=auto.execution.batch_size,
            threads=2,
            sharing=True,
        )
        assert counts(auto) == counts(explicit)
    auto_even = clifft.sample_survivors(
        program, 4096, seed=29, batch_size="auto", threads=2, sharing=True
    )
    explicit_even = clifft.sample_survivors(
        program, 4096, seed=29, batch_size=2048, threads=2, sharing=True
    )
    assert auto_even.execution.batch_size == 2048
    assert counts(auto_even) == counts(explicit_even)


def test_shared_tuning_respects_worker_budget_and_replays_selected_execution() -> None:
    program = clifft.compile(SOURCE)
    bounded = clifft.sample_survivors(
        program,
        1301,
        seed=25,
        batch_size="tune",
        tuning_budget_seconds=0.15,
        threads=2,
        sharing=True,
        max_worker_state_bytes=100 * 4096,
    )
    report = bounded.batch_tuning
    assert report is not None
    assert report.baseline_batch_size == 100
    assert {trial.batch_size for trial in report.trials} <= {64, 100}
    assert report.batch_size <= 100
    assert bounded.execution.batch_size == report.batch_size
    assert bounded.execution.shot_workers == report.shot_workers

    replayed = clifft.sample_survivors(
        program,
        1301,
        seed=25,
        batch_size=bounded.execution.batch_size,
        threads=2,
        sharing=bounded.execution.sharing,
        max_worker_state_bytes=100 * 4096,
    )
    assert counts(bounded) == counts(replayed)

    zero_shots = clifft.sample_survivors(
        program, 0, seed=25, batch_size="tune", tuning_budget_seconds=0.15, sharing=True
    )
    assert zero_shots.batch_tuning is not None
    assert zero_shots.batch_tuning.stop_reason == "zero_shots"
    assert zero_shots.execution.sharing is True
    assert zero_shots.execution.batch_size == 2
    assert zero_shots.execution.shot_workers == 0
    zero_replay = clifft.sample_survivors(
        program,
        0,
        seed=25,
        batch_size=zero_shots.execution.batch_size,
        sharing=zero_shots.execution.sharing,
    )
    assert zero_replay.execution == zero_shots.execution

    zero_budget = clifft.sample_survivors(
        program, 1301, seed=25, batch_size="tune", tuning_budget_seconds=0, sharing=True
    )
    assert zero_budget.batch_tuning is not None
    assert zero_budget.batch_tuning.stop_reason == "zero_budget"
    assert zero_budget.batch_tuning.trial_shots == 0
    auto = clifft.sample_survivors(program, 1301, seed=25, batch_size="auto", sharing=True)
    assert counts(zero_budget) == counts(auto)

    with pytest.warns(RuntimeWarning, match="using auto"):
        tiny_budget = clifft.sample_survivors(
            program, 1301, seed=25, batch_size="tune", tuning_budget_seconds=1e-300, sharing=True
        )
    assert tiny_budget.batch_tuning is not None
    assert tiny_budget.batch_tuning.stop_reason == "budget_exhausted"

    high_workers = clifft.sample_survivors(
        clifft.compile("H 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]"),
        1_000_000,
        seed=42,
        batch_size="tune",
        tuning_budget_seconds=0,
        threads=16,
        sharing=True,
    )
    assert high_workers.total_shots == high_workers.passed_shots == 1_000_000
    assert high_workers.batch_tuning.baseline_batch_size == 8192
    assert high_workers.batch_tuning.shot_workers == 16
    assert high_workers.batch_tuning.stop_reason == "zero_budget"

    single = clifft.sample_survivors(
        program, 1, seed=42, batch_size="tune", tuning_budget_seconds=0.1, sharing=True
    )
    assert single.total_shots == 1
    assert single.batch_tuning.stop_reason == "single_candidate"
    assert single.batch_tuning.trial_shots == 0
    assert single.batch_tuning.batch_size == 2
    assert single.execution.sharing is True
    assert single.execution.batch_size == 2
    single_replay = clifft.sample_survivors(
        program,
        1,
        seed=42,
        batch_size=single.execution.batch_size,
        sharing=single.execution.sharing,
    )
    assert single_replay.execution == single.execution
    assert counts(single_replay) == counts(single)


def test_shared_worker_reuse_across_repeated_and_concurrent_calls() -> None:
    def make_program() -> clifft.Program:
        return clifft.compile(UNPOSTSELECTED_SOURCE)

    reference_ungrouped = clifft.sample(
        make_program(),
        4096,
        seed=31,
        threads=4,
        batch_size=512,
        sharing=True,
        keep_records="syndromes",
        bit_packed=True,
    )
    reference_grouped = clifft.sample(
        make_program(),
        4096,
        seed=31,
        threads=4,
        batch_size=256,
        sharing=True,
        group_by_noise=True,
        keep_records="syndromes",
        bit_packed=True,
    )

    program = make_program()
    for _ in range(4):
        repeated = clifft.sample(
            program,
            4096,
            seed=31,
            threads=4,
            batch_size=512,
            sharing=True,
            keep_records="syndromes",
            bit_packed=True,
        )
        np.testing.assert_array_equal(repeated.detectors, reference_ungrouped.detectors)
        np.testing.assert_array_equal(repeated.observables, reference_ungrouped.observables)

    def run_task(index: int) -> tuple[bool, clifft.SampleResult]:
        grouped = index % 2 == 1
        return grouped, clifft.sample(
            program,
            4096,
            seed=31,
            threads=2,
            batch_size=256 if grouped else 512,
            sharing=True,
            group_by_noise=grouped,
            keep_records="syndromes",
            bit_packed=True,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(run_task, range(16)))
    for grouped, result in outcomes:
        expected = reference_grouped if grouped else reference_ungrouped
        np.testing.assert_array_equal(result.detectors, expected.detectors)
        np.testing.assert_array_equal(result.observables, expected.observables)


def test_bare_default_width_eight_shared_sampling_matches_exact_records() -> None:
    meas_lines = [
        *[f"U3(0.{31 + q},0.{17 + q},-0.{23 + q}) {q}" for q in range(8)],
        "CX 0 1 2 3 4 5 6 7",
        *[f"U3(0.{41 + q},-0.{19 + q},0.{29 + q}) {q}" for q in range(8)],
        "CX 1 2 3 4 5 6 7 0",
        *[f"U3(0.{53 + q},0.{11 + q},-0.{37 + q}) {q}" for q in range(8)],
        "CX 0 4 1 5 2 6 3 7",
        *[f"U3(0.{27 + q},-0.{31 + q},0.{19 + q}) {q}" for q in range(8)],
        "CX 0 2 1 3 4 6 5 7",
        *[f"U3(0.{37 + q},0.{23 + q},-0.{13 + q}) {q}" for q in range(8)],
        "M 0 1",
        "CX rec[-2] 2",
        "CZ rec[-1] 3",
        "R_Z(0.17) 2 3",
        "H 2 3",
        "M 2 3 4 5 6 7",
    ]
    syn_lines = meas_lines + [
        "DETECTOR rec[-8] rec[-6] rec[-4] rec[-2]",
        "DETECTOR rec[-7] rec[-5] rec[-3] rec[-1]",
        "OBSERVABLE_INCLUDE(0) rec[-8] rec[-7] rec[-2] rec[-1]",
    ]

    meas_prog = clifft.compile("\n".join(meas_lines))
    meas_base = clifft.compile("\n".join(meas_lines), hir_passes=None, reconstruction=False)
    records = ["".join(str((k >> i) & 1) for i in range(8)) for k in range(256)]
    expected = clifft.record_probabilities(meas_base, records)
    np.testing.assert_allclose(
        clifft.record_probabilities(meas_prog, records), expected, atol=1e-12, rtol=0
    )

    program = clifft.compile("\n".join(syn_lines))
    assert program.peak_active_width == 8

    shots = 16384
    serial = clifft.sample(program, shots, seed=41)
    assert serial.execution is not None
    assert serial.execution.executor == "shared"
    assert serial.execution.sharing is True

    threaded = clifft.sample(program, shots, seed=43, threads=2)
    assert threaded.execution is not None
    assert threaded.execution.executor == "shared"
    assert threaded.execution.sharing is True

    scalar = clifft.sample(program, shots, seed=47, batch_size=1, sharing=False)
    assert scalar.execution is not None
    assert scalar.execution.executor == "scalar"

    packed = clifft.sample(program, shots, seed=41, keep_records="syndromes", bit_packed=True)
    assert packed.execution is not None
    assert packed.execution.executor == "shared"
    assert packed.execution.sharing is True

    for result in (serial, threaded, scalar):
        assert_joint_distribution(result.measurements, expected)
        m = result.measurements
        np.testing.assert_array_equal(result.detectors[:, 0], m[:, 0] ^ m[:, 2] ^ m[:, 4] ^ m[:, 6])
        np.testing.assert_array_equal(result.detectors[:, 1], m[:, 1] ^ m[:, 3] ^ m[:, 5] ^ m[:, 7])
        np.testing.assert_array_equal(
            result.observables[:, 0], m[:, 0] ^ m[:, 1] ^ m[:, 6] ^ m[:, 7]
        )

    np.testing.assert_array_equal(unpack_bits(packed.detectors, 2), serial.detectors)
    np.testing.assert_array_equal(unpack_bits(packed.observables, 1), serial.observables)
