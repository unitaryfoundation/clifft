"""Tests for the UCC Sinter adapter (paper.magic.ucc_soft_sampler)."""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import sinter
import stim

from paper.magic.ucc_soft_sampler import UccSoftSampler


def _build_mask_from_coords(circuit: stim.Circuit) -> np.ndarray:
    """Build bit-packed postselection mask for detectors with coord[4]==-9."""
    num_dets = circuit.num_detectors
    mask = bytearray(math.ceil(num_dets / 8))
    for k, coord in circuit.get_detector_coordinates().items():
        if len(coord) >= 5 and coord[4] == -9:
            mask[k // 8] |= 1 << (k % 8)
    return np.array(mask, dtype=np.uint8)


class TestUccSoftSampler:
    """Integration tests for the Sinter adapter."""

    def test_simple_postselection_circuit(self) -> None:
        """Sampler correctly reports shots, discards, errors."""
        circuit = stim.Circuit(
            """
            H 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        )
        # Postselect on detector 0: survivors have meas[0]==0, obs==0
        mask = np.array([0x01], dtype=np.uint8)
        task = sinter.Task(
            circuit=circuit,
            decoder="ucc",
            postselection_mask=mask,
            skip_validation=True,
        )

        sampler = UccSoftSampler()
        compiled = sampler.compiled_sampler_for_task(task)
        stats = compiled.sample(1000)

        assert stats.shots == 1000
        assert stats.discards > 0
        assert stats.discards < 1000
        # Survivors always have obs==0, so errors==0
        assert stats.errors == 0
        assert stats.shots == stats.discards + (stats.shots - stats.discards)

    def test_no_postselection(self) -> None:
        """Without mask, all shots pass and observable errors are counted."""
        circuit = stim.Circuit(
            """
            H 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        )
        task = sinter.Task(circuit=circuit, decoder="ucc", skip_validation=True)

        sampler = UccSoftSampler()
        compiled = sampler.compiled_sampler_for_task(task)
        stats = compiled.sample(10000)

        assert stats.shots == 10000
        assert stats.discards == 0
        # ~50% of shots should have obs[0]==1 (random H measurement)
        assert 4000 < stats.errors < 6000

    def test_target_qec_via_sinter_collect(self) -> None:
        """End-to-end: run target_qec.stim through sinter.collect()."""
        circuit_path = pathlib.Path("tools/bench/target_qec.stim")
        if not circuit_path.exists():
            pytest.skip("target_qec.stim not found")

        circuit = stim.Circuit.from_file(str(circuit_path))
        mask = _build_mask_from_coords(circuit)

        task = sinter.Task(
            circuit=circuit,
            decoder="ucc",
            postselection_mask=mask,
            json_metadata={"d": 3, "p": 0.001},
        )

        results = sinter.collect(
            num_workers=1,
            tasks=[task],
            custom_decoders={"ucc": UccSoftSampler()},
            max_shots=5000,
            print_progress=False,
        )

        assert len(results) == 1
        stat = results[0]
        assert stat.shots == 5000
        assert stat.discards > 0
        # The d=3 circuit has noise, so some errors are expected
        passed = stat.shots - stat.discards
        assert passed > 0
        discard_rate = stat.discards / stat.shots
        # Sanity: discard rate should be non-trivial but not 100%
        assert 0.01 < discard_rate < 0.99
