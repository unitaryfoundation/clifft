"""Compatibility tests for the retained Stim-shaped sampling API."""

from pathlib import Path

import numpy as np
import pytest
import stim

import clifft

MEASUREMENT_CIRCUIT = """
    X 0 2 8
    M 0 1 2 3 4 5 6 7 8 9
"""

DETECTOR_CIRCUIT = """
    X_ERROR(1) 0
    M 0 1
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
"""


def test_measurement_sampler_matches_stim_shapes_dtypes_and_bit_order() -> None:
    shots = 5
    actual_sampler = clifft.compile_sampler(MEASUREMENT_CIRCUIT, seed=12, batch_size=65)
    expected_sampler = stim.Circuit(MEASUREMENT_CIRCUIT).compile_sampler(seed=12)

    actual = actual_sampler.sample(shots)
    expected = expected_sampler.sample(shots)
    assert actual.dtype == expected.dtype == np.dtype(np.bool_)
    assert actual.shape == expected.shape == (shots, 10)
    np.testing.assert_array_equal(actual, expected)

    actual_packed = actual_sampler.sample(shots, bit_packed=True)
    expected_packed = expected_sampler.sample(shots, bit_packed=True)
    assert actual_packed.dtype == expected_packed.dtype == np.dtype(np.uint8)
    assert actual_packed.shape == expected_packed.shape == (shots, 2)
    np.testing.assert_array_equal(actual_packed, expected_packed)


def test_detector_sampler_matches_stim_observable_placement() -> None:
    shots = 4
    actual_sampler = clifft.compile_detector_sampler(DETECTOR_CIRCUIT, seed=13, batch_size=65)
    expected_sampler = stim.Circuit(DETECTOR_CIRCUIT).compile_detector_sampler(seed=13)

    for prepend, append in [(False, False), (True, False), (False, True), (True, True)]:
        actual = actual_sampler.sample(
            shots,
            prepend_observables=prepend,
            append_observables=append,
        )
        expected = expected_sampler.sample(
            shots,
            prepend_observables=prepend,
            append_observables=append,
        )
        np.testing.assert_array_equal(actual, expected)

    actual_packed = actual_sampler.sample(
        shots, prepend_observables=True, append_observables=True, bit_packed=True
    )
    expected_packed = expected_sampler.sample(
        shots, prepend_observables=True, append_observables=True, bit_packed=True
    )
    np.testing.assert_array_equal(actual_packed, expected_packed)
    np.testing.assert_array_equal(actual_packed, np.full((shots, 1), 0b1011, dtype=np.uint8))


def test_detector_sampler_returns_separate_outputs_and_fills_caller_arrays() -> None:
    sampler = clifft.compile_detector_sampler(DETECTOR_CIRCUIT, seed=14, batch_size=65)
    detectors, observables = sampler.sample(3, separate_observables=True)
    np.testing.assert_array_equal(detectors, [[True, False]] * 3)
    np.testing.assert_array_equal(observables, [[True]] * 3)

    dets_out = np.empty((3, 2), dtype=np.bool_)
    obs_out = np.empty((3, 1), dtype=np.bool_)
    returned = sampler.sample(3, dets_out=dets_out, obs_out=obs_out)
    assert returned is dets_out
    np.testing.assert_array_equal(dets_out, detectors)
    np.testing.assert_array_equal(obs_out, observables)

    packed_dets_out = np.empty((3, 1), dtype=np.uint8)
    packed_obs_out = np.empty((3, 1), dtype=np.uint8)
    returned_pair = sampler.sample(
        3,
        separate_observables=True,
        bit_packed=True,
        dets_out=packed_dets_out,
        obs_out=packed_obs_out,
    )
    assert returned_pair[0] is packed_dets_out
    assert returned_pair[1] is packed_obs_out
    np.testing.assert_array_equal(packed_dets_out, [[1]] * 3)
    np.testing.assert_array_equal(packed_obs_out, [[1]] * 3)


def test_detector_sampler_uses_stim_reference_normalization() -> None:
    circuit = """
        X 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    actual = clifft.compile_detector_sampler(circuit, seed=15).sample(3, separate_observables=True)
    expected = (
        stim.Circuit(circuit).compile_detector_sampler(seed=15).sample(3, separate_observables=True)
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert not np.any(actual[0])
    assert not np.any(actual[1])


def test_compiled_sampler_advances_and_replays_seeded_calls() -> None:
    circuit = "X_ERROR(0.5) 0\nM 0"
    first = clifft.compile_sampler(circuit, seed=1601, batch_size=65)
    replay = clifft.compile_sampler(circuit, seed=1601, batch_size=65)

    first_call = first.sample(257)
    second_call = first.sample(257)
    np.testing.assert_array_equal(first_call, replay.sample(257))
    np.testing.assert_array_equal(second_call, replay.sample(257))
    assert not np.array_equal(first_call, second_call)


@pytest.mark.parametrize("source_kind", ["text", "path", "stim", "clifft"])
def test_compile_sampler_accepts_common_circuit_inputs(source_kind: str, tmp_path: Path) -> None:
    source: object
    if source_kind == "text":
        source = MEASUREMENT_CIRCUIT
    elif source_kind == "path":
        source = tmp_path / "circuit.stim"
        source.write_text(MEASUREMENT_CIRCUIT, encoding="utf-8")
    elif source_kind == "stim":
        source = stim.Circuit(MEASUREMENT_CIRCUIT)
    else:
        source = clifft.parse(MEASUREMENT_CIRCUIT)
    actual = clifft.compile_sampler(source, seed=17, hir_passes=None).sample(2)
    expected = stim.Circuit(MEASUREMENT_CIRCUIT).compile_sampler(seed=17).sample(2)
    np.testing.assert_array_equal(actual, expected)


def test_compiled_samplers_handle_empty_output_shapes() -> None:
    measurement_sampler = clifft.compile_sampler("M 0", seed=18)
    assert measurement_sampler.sample(0).shape == (0, 1)
    assert measurement_sampler.sample(0, bit_packed=True).shape == (0, 1)

    detector_sampler = clifft.compile_detector_sampler("M 0", seed=18)
    detectors, observables = detector_sampler.sample(0, separate_observables=True)
    assert detectors.shape == (0, 0)
    assert observables.shape == (0, 0)


def test_sample_write_matches_matrix_sampling(tmp_path: Path) -> None:
    measurement_matrix = clifft.compile_sampler(MEASUREMENT_CIRCUIT, seed=19, batch_size=65).sample(
        5
    )
    measurement_path = tmp_path / "measurements.01"
    clifft.compile_sampler(MEASUREMENT_CIRCUIT, seed=19, batch_size=65).sample_write(
        5, filepath=measurement_path, format="01"
    )
    expected_text = b"".join(
        bytes(row.astype(np.uint8) + ord("0")) + b"\n" for row in measurement_matrix
    )
    assert measurement_path.read_bytes() == expected_text

    detector_matrix, observable_matrix = clifft.compile_detector_sampler(
        DETECTOR_CIRCUIT, seed=20, batch_size=65
    ).sample(5, separate_observables=True, bit_packed=True)
    detector_path = tmp_path / "detectors.b8"
    observable_path = tmp_path / "observables.b8"
    clifft.compile_detector_sampler(DETECTOR_CIRCUIT, seed=20, batch_size=65).sample_write(
        5,
        filepath=detector_path,
        format="b8",
        obs_out_filepath=observable_path,
        obs_out_format="b8",
    )
    assert detector_path.read_bytes() == detector_matrix.tobytes()
    assert observable_path.read_bytes() == observable_matrix.tobytes()


def test_detector_sampler_rejects_incompatible_output_requests() -> None:
    sampler = clifft.compile_detector_sampler(DETECTOR_CIRCUIT, seed=21)
    with pytest.raises(ValueError, match="separate_observables"):
        sampler.sample(1, separate_observables=True, append_observables=True)
    with pytest.raises(TypeError, match="dtype"):
        sampler.sample(2, dets_out=np.empty((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        sampler.sample(2, dets_out=np.empty((2, 3), dtype=np.bool_))
    with pytest.raises((TypeError, ValueError), match="contiguous|incompatible"):
        sampler.sample(2, dets_out=np.empty((2, 4), dtype=np.bool_)[:, ::2])


def test_compiled_detector_distribution_matches_stim() -> None:
    circuit = """
        X_ERROR(0.2) 0
        X_ERROR(0.35) 1
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
    """
    shots = 20_000
    actual = clifft.compile_detector_sampler(circuit, seed=22, batch_size=257).sample(shots)
    expected = stim.Circuit(circuit).compile_detector_sampler(seed=23).sample(shots)
    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual.mean(axis=0), expected.mean(axis=0), atol=0.025, rtol=0)
