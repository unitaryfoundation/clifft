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
        for result in (unpacked, packed):
            assert result.total_shots == reference.total_shots
            assert result.passed_shots == reference.passed_shots
            assert result.logical_errors == reference.logical_errors
            np.testing.assert_array_equal(result.observable_ones, reference.observable_ones)


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
    detectors, observables = clifft.sample_detectors(
        program, 3, separate_observables=True, bit_packed=True
    )
    assert detectors is result.detectors
    assert observables is result.observables


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
