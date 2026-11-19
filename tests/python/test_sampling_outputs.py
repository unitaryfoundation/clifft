"""Selective and b8 sampling preserve retained records and survivor summaries."""

from typing import Any

import numpy as np
import pytest
import stim

import clifft


def feedback_circuit() -> str:
    # Nine outputs cross a byte boundary. Feedback follows a noisy measurement,
    # and repeated measurements retain correlations with the earlier records.
    return "\n".join(
        f"""
        H 0
        T 0
        EXP_VAL X0
        DEPOLARIZE1(0.1) 0
        H 0
        M(0.2) 0
        CX rec[-1] 1
        M 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE({i}) rec[-1]
        R 0 1
        """
        for i in range(9)
    )


@pytest.mark.parametrize("api", ["sample", "sample_k", "sample_survivors", "sample_k_survivors"])
@pytest.mark.parametrize("batch_size,threads", [(1, 1), (65, 2), ("auto", 1)])
@pytest.mark.parametrize("keep_records", [True, "all", "syndromes", "measurements"])
def test_selected_and_packed_rows(
    api: str, batch_size: int | str, threads: int, keep_records: Any
) -> None:
    survivor = api.endswith("survivors")
    program = clifft.compile(
        feedback_circuit(), postselection_mask=[1] + [0] * 8 if survivor else []
    )
    sample = getattr(clifft, api)
    options: dict[str, Any] = dict(seed=79, batch_size=batch_size, threads=threads)
    if "sample_k" in api:
        options["k"] = 2
    reference = sample(program, 131, keep_records=True, **options)
    unpacked = sample(program, 131, keep_records=keep_records, **options)
    packed = sample(program, 131, keep_records=keep_records, bit_packed=True, **options)
    assert not reference.bit_packed
    assert packed.bit_packed
    assert unpacked.retained_shots == packed.retained_shots == reference.retained_shots
    assert reference.retained_shots == (reference.passed_shots if survivor else 131)
    for field in ("measurements", "detectors", "observables"):
        expected = getattr(reference, field)
        retained = keep_records in (True, "all") or (
            keep_records == "measurements"
            if field == "measurements"
            else keep_records == "syndromes"
        )
        actual = getattr(unpacked, field)
        compact = getattr(packed, field)
        assert compact.dtype == np.uint8
        assert compact.shape == (expected.shape[0] if retained else 0, (expected.shape[1] + 7) // 8)
        if retained:
            # This small feedback circuit preserves individual rows at fixed capacity.
            if batch_size != "auto":
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(compact, np.packbits(actual, axis=1, bitorder="little"))
        else:
            assert actual.shape == (0, expected.shape[1])
    expected_probes = reference.exp_vals if keep_records in (True, "all") else np.empty((0, 9))
    np.testing.assert_array_equal(unpacked.exp_vals, expected_probes)
    np.testing.assert_array_equal(packed.exp_vals, expected_probes)
    if survivor:
        assert packed.total_shots == unpacked.total_shots == reference.total_shots
        assert packed.passed_shots == unpacked.passed_shots
        assert packed.logical_errors == unpacked.logical_errors
        np.testing.assert_array_equal(packed.observable_ones, unpacked.observable_ones)
        if batch_size != "auto":
            assert unpacked.passed_shots == reference.passed_shots
            assert unpacked.logical_errors == reference.logical_errors
            np.testing.assert_array_equal(unpacked.observable_ones, reference.observable_ones)


@pytest.mark.parametrize("api", ["sample", "sample_k", "sample_survivors", "sample_k_survivors"])
@pytest.mark.parametrize("batch_size", [1, 65])
def test_surface_code_selected_rows_cross_word_boundaries(api: str, batch_size: int) -> None:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=5,
        rounds=5,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
    )
    assert circuit.num_detectors > 64
    program = clifft.compile(str(circuit), normalize_syndromes=True)
    sample = getattr(clifft, api)
    options: dict[str, Any] = dict(seed=73, batch_size=batch_size)
    if "sample_k" in api:
        options["k"] = 3
    reference = sample(program, 131, keep_records=True, **options)
    # No postselection removes lanes, so the selected plans preserve each row.
    assert reference.detectors[:, 64:].any()
    for selection, fields in (
        ("syndromes", ("detectors", "observables")),
        ("measurements", ("measurements",)),
    ):
        for bit_packed in (False, True):
            actual = sample(program, 131, keep_records=selection, bit_packed=bit_packed, **options)
            assert actual.retained_shots == 131
            for field in fields:
                expected = getattr(reference, field)
                if bit_packed:
                    expected = np.packbits(expected, axis=1, bitorder="little")
                np.testing.assert_array_equal(getattr(actual, field), expected)


@pytest.mark.parametrize("api", ["sample", "sample_k", "sample_survivors", "sample_k_survivors"])
@pytest.mark.parametrize("keep_records", [False, "all", "syndromes", "measurements"])
@pytest.mark.parametrize("shots,source", [(0, "M 0\nDETECTOR rec[-1]"), (3, ""), (3, "M 0")])
@pytest.mark.parametrize("bit_packed", [False, True])
def test_empty_output_shapes(
    api: str, keep_records: Any, shots: int, source: str, bit_packed: bool
) -> None:
    sample = getattr(clifft, api)
    options: dict[str, Any] = dict(keep_records=keep_records, bit_packed=bit_packed)
    if "sample_k" in api:
        options["k"] = 0
    program = clifft.compile(source)
    if keep_records is False and not api.endswith("survivors"):
        with pytest.raises(ValueError):
            sample(program, shots, **options)
        return
    result = sample(program, shots, **options)
    rows = shots if keep_records is not False else 0
    assert result.retained_shots == rows
    for field, columns in (
        ("measurements", program.num_measurements),
        ("detectors", program.num_detectors),
        ("observables", program.num_observables),
    ):
        retained = keep_records == "all" or (
            keep_records == "measurements"
            if field == "measurements"
            else keep_records == "syndromes"
        )
        assert getattr(result, field).shape == (
            rows if retained or columns == 0 else 0,
            (columns + 7) // 8 if bit_packed else columns,
        )
    assert f"shots={rows}" in repr(result)


@pytest.mark.parametrize("batch_size", [1, 65])
@pytest.mark.parametrize("keep_records", [False, "all", "syndromes", "measurements"])
def test_all_discarded_and_unselected_survivor_counts(batch_size: int, keep_records: Any) -> None:
    source = "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    rejected = clifft.sample_survivors(
        clifft.compile(source, postselection_mask=[1]),
        67,
        keep_records=keep_records,
        bit_packed=True,
        batch_size=batch_size,
    )
    assert rejected.passed_shots == rejected.retained_shots == 0
    accepted = clifft.sample_survivors(
        clifft.compile(source),
        67,
        keep_records=keep_records,
        bit_packed=True,
        batch_size=batch_size,
    )
    assert accepted.passed_shots == accepted.logical_errors == 67
    np.testing.assert_array_equal(accepted.observable_ones, [67])


@pytest.mark.parametrize("batch_size", [1, 65])
@pytest.mark.parametrize("bit_packed", [False, True])
def test_detector_convenience_matches_stim(batch_size: int, bit_packed: bool) -> None:
    source = "\n".join(
        f"X_ERROR({int(i % 3 != 0)}) {i}\nM {i}\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE({i}) rec[-1]"
        for i in range(9)
    )
    program = clifft.compile(source)
    expected = (
        stim.Circuit(source)
        .compile_detector_sampler()
        .sample(67, separate_observables=True, bit_packed=bit_packed)
    )
    # Both samplers report detector events relative to the noiseless reference.
    normalized = clifft.compile(source, normalize_syndromes=True)
    actual = clifft.sample_detectors(
        normalized, 67, separate_observables=True, bit_packed=bit_packed, batch_size=batch_size
    )
    for left, right in zip(actual, expected):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    raw = clifft.sample(
        program, 67, keep_records="syndromes", bit_packed=bit_packed, batch_size=batch_size
    )
    direct = clifft.sample_detectors(program, 67, bit_packed=bit_packed, batch_size=batch_size)
    np.testing.assert_array_equal(direct, raw.detectors)


def test_detector_convenience_shares_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    program = clifft.compile("M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    result = clifft.sample(program, 3, keep_records="syndromes")
    monkeypatch.setattr(clifft, "sample", lambda *args, **kwargs: result)
    detectors, observables = clifft.sample_detectors(program, 3, separate_observables=True)
    assert np.shares_memory(detectors, result.detectors)
    assert np.shares_memory(observables, result.observables)
    packed_detectors, packed_observables = clifft.sample_detectors(
        program, 3, separate_observables=True, bit_packed=True
    )
    assert packed_detectors is result.detectors
    assert packed_observables is result.observables


@pytest.mark.parametrize("api", ["sample", "sample_k", "sample_survivors", "sample_k_survivors"])
def test_invalid_record_choice(api: str) -> None:
    options = {"k": 0} if "sample_k" in api else {}
    with pytest.raises(ValueError):
        getattr(clifft, api)(clifft.compile(""), 0, keep_records="unknown", **options)


@pytest.mark.parametrize("batch_size", [1, 65])
@pytest.mark.parametrize("keep_records", [False, "all", "syndromes", "measurements"])
def test_asymmetric_readout_preserves_snapshots(batch_size: int, keep_records: Any) -> None:
    source = """
        H 0
        T 0
        H 0
        M 0
        READOUT_NOISE(0.2,0.7) rec[-1]
        DETECTOR rec[-1]
        READOUT_NOISE(0.4,0.1) rec[-1]
        CX rec[-1] 1
        M 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """
    program = clifft.compile(source, postselection_mask=[0, 1], hir_passes=None)
    reference = clifft.sample_survivors(
        program, 131, seed=31, batch_size=batch_size, keep_records=True
    )
    result = clifft.sample_survivors(
        program, 131, seed=31, batch_size=batch_size, keep_records=keep_records, bit_packed=True
    )
    assert result.passed_shots == reference.passed_shots == 131
    assert result.logical_errors == reference.logical_errors
    np.testing.assert_array_equal(result.observable_ones, reference.observable_ones)
    for field in ("measurements", "detectors", "observables"):
        rows = getattr(result, field)
        if rows.shape[0]:
            np.testing.assert_array_equal(
                rows, np.packbits(getattr(reference, field), axis=1, bitorder="little")
            )


@pytest.mark.parametrize("api", ["sample", "sample_k", "sample_survivors", "sample_k_survivors"])
@pytest.mark.parametrize("keep_records", ["all", "syndromes", "measurements"])
def test_selective_reconstruction_and_fixed_fault_plans(api: str, keep_records: str) -> None:
    source = "H 0 1\nT 0 1\nH 0 1\nM 0 1\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"
    program = clifft.compile(source, hir_passes=None, reconstruction=True)
    assert program.sampling_plan(keep_records=keep_records).reconstruction_selected
    sample = getattr(clifft, api)
    options: dict[str, Any] = dict(seed=73, batch_size=65, keep_records=keep_records)
    if "sample_k" in api:
        options["k"] = 0
    unpacked = sample(program, 131, **options)
    packed = sample(program, 131, bit_packed=True, **options)
    assert packed.retained_shots == unpacked.retained_shots == 131
    for field in ("measurements", "detectors", "observables"):
        np.testing.assert_array_equal(
            getattr(packed, field), np.packbits(getattr(unpacked, field), axis=1, bitorder="little")
        )
    if "sample_k" in api:
        original = clifft.compile(source, hir_passes=None)
        expected = sample(original, 131, bit_packed=True, **options)
        for field in ("measurements", "detectors", "observables"):
            np.testing.assert_array_equal(getattr(packed, field), getattr(expected, field))


@pytest.mark.parametrize("api", ["sample", "sample_k", "sample_survivors", "sample_k_survivors"])
def test_packed_selective_tuning(api: str) -> None:
    program = clifft.compile("M 0\nDETECTOR rec[-1]")
    options = {"k": 0} if "sample_k" in api else {}
    result = getattr(clifft, api)(
        program,
        17,
        keep_records="syndromes",
        bit_packed=True,
        batch_size="tune",
        tuning_budget_seconds=0,
        **options,
    )
    assert result.retained_shots == 17
    assert result.measurements.shape == (0, 1)
    assert result.detectors.shape == (17, 1)
    assert result.batch_tuning.stop_reason == "zero_budget"


def _unpack_little(packed: np.ndarray, width: int) -> np.ndarray:
    return np.unpackbits(packed, axis=1, bitorder="little")[:, :width]


def _assert_two_sample_columns_close(
    counts_a: np.ndarray, shots_a: int, counts_b: np.ndarray, shots_b: int
) -> None:
    p_a = np.asarray(counts_a, dtype=np.float64) / shots_a
    p_b = np.asarray(counts_b, dtype=np.float64) / shots_b
    pooled = (np.asarray(counts_a, dtype=np.float64) + np.asarray(counts_b, dtype=np.float64)) / (
        shots_a + shots_b
    )
    se = np.sqrt(pooled * np.maximum(1.0 - pooled, 0.0) * (1.0 / shots_a + 1.0 / shots_b))
    tol = 7.0 * se + 12.0 / min(shots_a, shots_b)
    np.testing.assert_array_less(np.abs(p_a - p_b), tol)


def _width_zero_postselected_circuit(asymmetric_readout: bool) -> tuple[str, list[int]]:
    lines = ["H 0 2 4", "CX 0 1 2 3 4 5"]
    for round_idx in range(35):
        if round_idx % 4 == 0:
            lines.append("X_ERROR(0.02) 0")
            lines.append("CORRELATED_ERROR(0.015) X1 Z2")
        if round_idx % 4 == 2:
            lines.append("DEPOLARIZE1(0.025) 3")
        lines.append("CX 0 1 2 3")
        if round_idx == 33 and asymmetric_readout:
            lines.append("MR 1")
            lines.append("CX rec[-1] 2")
            lines.append("MR 3")
            lines.append("CZ rec[-1] 4")
            lines.append("DETECTOR rec[-2]")
            lines.append("DETECTOR rec[-1]")
            lines.append("READOUT_NOISE(0.15, 0.30) rec[-1]")
        else:
            lines.append("MR(0.02) 1")
            lines.append("CX rec[-1] 2")
            lines.append("MR(0.04) 3" if round_idx == 33 else "MR(0.015) 3")
            lines.append("CZ rec[-1] 4")
            lines.append("DETECTOR rec[-2]")
            lines.append("DETECTOR rec[-1]")
    lines.append("OBSERVABLE_INCLUDE(0) rec[-21] rec[-31] rec[-41] rec[-69]")
    lines.append("OBSERVABLE_INCLUDE(1) rec[-3] rec[-19] rec[-35]")
    # Postselect detectors in both words (0 and 4..13 in word 0; 64, 65, 68, 69 in word 1)
    # while leaving the round-33 snapshot detectors (66, 67) and middle detectors unpostselected.
    mask = [1 if (d == 0 or 4 <= d < 14 or d in (64, 65, 68, 69)) else 0 for d in range(70)]
    return "\n".join(lines), mask


def test_width_zero_postselected_multiword_matches_scalar() -> None:
    source, mask = _width_zero_postselected_circuit(asymmetric_readout=True)
    program = clifft.compile(source, postselection_mask=mask)
    assert program.peak_active_width == 0
    assert program.num_measurements > 64
    assert program.num_detectors > 64
    assert program.num_observables == 2

    shots = 8192
    scalar = clifft.sample_survivors(
        program, shots, seed=51, batch_size=1, sharing=False, keep_records=True
    )
    assert scalar.execution.executor == "scalar"
    assert 0 < scalar.passed_shots < shots

    mask_bool = np.asarray(mask, dtype=bool)
    assert not scalar.detectors[:, mask_bool].any()
    scalar_det_counts = scalar.detectors[:, ~mask_bool].sum(axis=0)
    scalar_summary = np.array(
        [scalar.passed_shots, scalar.logical_errors, *scalar.observable_ones], dtype=np.int64
    )
    assert np.all(scalar.observable_ones > 0.05 * scalar.passed_shots)

    configs: list[tuple[bool | str, bool | str, clifft.NoisePreparation, bool]] = [
        (False, True, clifft.NoisePreparation.DENSE, False),
        ("auto", False, clifft.NoisePreparation.SITE_SPARSE, False),
        (False, "syndromes", clifft.NoisePreparation.SITE_ROWS, True),
        (True, True, clifft.NoisePreparation.DENSE, True),
        (True, "syndromes", clifft.NoisePreparation.SITE_SPARSE, True),
        (True, False, clifft.NoisePreparation.SITE_ROWS, False),
    ]
    for offset, (sharing, keep_records, prep, bit_packed) in enumerate(configs):
        result = clifft.sample_survivors(
            program,
            shots,
            seed=101 + offset * 7,
            keep_records=keep_records,
            bit_packed=bit_packed,
            sharing=sharing,
            noise_preparation=prep,
        )
        assert result.execution.executor == "packed"
        assert result.execution.sharing is (sharing is True)
        if sharing is not True:
            assert result.execution.batch_size == 2048
        assert result.execution.noise_preparation == prep
        assert result.execution.bit_packed is bit_packed

        if bit_packed:
            same_seed_unpacked = clifft.sample_survivors(
                program,
                shots,
                seed=101 + offset * 7,
                keep_records=keep_records,
                bit_packed=False,
                sharing=sharing,
                noise_preparation=prep,
            )
            if keep_records is True:
                np.testing.assert_array_equal(
                    _unpack_little(result.measurements, program.num_measurements),
                    same_seed_unpacked.measurements,
                )
            detectors = _unpack_little(result.detectors, program.num_detectors)
            observables = _unpack_little(result.observables, program.num_observables)
            np.testing.assert_array_equal(detectors, same_seed_unpacked.detectors)
            np.testing.assert_array_equal(observables, same_seed_unpacked.observables)
        else:
            detectors = result.detectors
            observables = result.observables

        if keep_records in (True, "all", "syndromes"):
            assert not detectors[:, mask_bool].any()
            _assert_two_sample_columns_close(
                detectors[:, ~mask_bool].sum(axis=0), shots, scalar_det_counts, shots
            )
            np.testing.assert_array_equal(observables.sum(axis=0), result.observable_ones)

        summary = np.array(
            [result.passed_shots, result.logical_errors, *result.observable_ones], dtype=np.int64
        )
        _assert_two_sample_columns_close(summary, shots, scalar_summary, shots)


def test_width_zero_postselected_multiword_matches_stim() -> None:
    source, mask = _width_zero_postselected_circuit(asymmetric_readout=False)
    program = clifft.compile(source, postselection_mask=mask)
    assert program.peak_active_width == 0
    assert program.num_measurements > 64
    assert program.num_detectors > 64

    shots = 16384
    mask_bool = np.asarray(mask, dtype=bool)
    stim_dets, stim_obs = (
        stim.Circuit(source)
        .compile_detector_sampler(seed=61)
        .sample(shots, separate_observables=True)
    )
    surviving = ~stim_dets[:, mask_bool].any(axis=1)
    stim_det_counts = stim_dets[surviving][:, ~mask_bool].sum(axis=0)
    stim_obs_surv = stim_obs[surviving]
    stim_summary = np.array(
        [
            surviving.sum(),
            stim_obs_surv.any(axis=1).sum(),
            *stim_obs_surv.sum(axis=0),
        ],
        dtype=np.int64,
    )

    for sharing_opt, expected_sharing, expected_batch in (
        ("auto", False, 2048),
        (True, True, 8192),
    ):
        result = clifft.sample_survivors(
            program,
            shots,
            seed=67 if not expected_sharing else 71,
            keep_records="syndromes",
            sharing=sharing_opt,
        )
        assert result.execution.executor == "packed"
        assert result.execution.sharing is expected_sharing
        assert result.execution.batch_size == expected_batch
        assert not result.detectors[:, mask_bool].any()

        summary = np.array(
            [result.passed_shots, result.logical_errors, *result.observable_ones], dtype=np.int64
        )
        _assert_two_sample_columns_close(summary, shots, stim_summary, shots)
        _assert_two_sample_columns_close(
            result.detectors[:, ~mask_bool].sum(axis=0), shots, stim_det_counts, shots
        )
