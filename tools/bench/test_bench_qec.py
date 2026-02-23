"""Performance benchmarks comparing UCC and Stim.

This module measures compilation latency and per-shot execution overhead
for both engines on a representative QEC circuit.

Run with: just bench
"""

from pathlib import Path
from typing import Any

import pytest
import stim

import ucc

# Load target circuit from file
_CIRCUIT_PATH = Path(__file__).parent / "target_qec.stim"
_CIRCUIT_TEXT = _CIRCUIT_PATH.read_text()

# Number of shots for sampling benchmarks
_BENCHMARK_SHOTS = 100_000


def get_stim_simd_info() -> dict[str, Any]:
    """Detect Stim's SIMD backend and batch size.

    Stim's frame simulator uses SIMD to sample multiple shots in parallel.
    Each bit in the SIMD register represents one shot's Pauli frame.

    Returns:
        Dictionary with 'backend', 'simd_bits', and 'shots_per_batch' keys.
    """
    module = stim.Circuit.__module__

    if "avx2" in module:
        return {"backend": "avx2", "simd_bits": 256, "shots_per_batch": 256}
    elif "sse2" in module:
        return {"backend": "sse2", "simd_bits": 128, "shots_per_batch": 128}
    elif "polyfill" in module:
        return {"backend": "polyfill", "simd_bits": 64, "shots_per_batch": 64}
    else:
        return {"backend": "unknown", "simd_bits": 0, "shots_per_batch": 1}


# =============================================================================
# SIMD Info Test - Reports Stim's vectorization level
# =============================================================================


def test_00_stim_simd_info(record_property: Any) -> None:
    """Report Stim's SIMD backend (runs first due to naming).

    This test captures the SIMD vectorization level being used by Stim,
    which directly affects sampling throughput. Stim processes N shots
    per circuit pass, where N = SIMD width in bits.

    - SSE2: 128 bits = 128 shots/pass
    - AVX2: 256 bits = 256 shots/pass
    - AVX-512: 512 bits = 512 shots/pass (not in PyPI wheels)
    """
    info = get_stim_simd_info()

    # Record as pytest properties (visible in JUnit XML output)
    record_property("stim_backend", info["backend"])
    record_property("stim_simd_bits", info["simd_bits"])
    record_property("stim_shots_per_batch", info["shots_per_batch"])

    # Also print for console visibility
    print(f"\n  Stim SIMD backend: {info['backend']}")
    print(f"  Stim SIMD width: {info['simd_bits']} bits")
    print(f"  Stim shots per batch: {info['shots_per_batch']}")


# =============================================================================
# Fixtures - Pre-compile objects so compilation doesn't affect runtime metrics
# =============================================================================


@pytest.fixture(scope="module")
def stim_circuit() -> stim.Circuit:
    """Pre-parsed Stim circuit."""
    return stim.Circuit(_CIRCUIT_TEXT)


@pytest.fixture(scope="module")
def stim_sampler(stim_circuit: stim.Circuit) -> stim.CompiledDetectorSampler:
    """Pre-compiled Stim detector sampler."""
    return stim_circuit.compile_detector_sampler()


@pytest.fixture(scope="module")
def ucc_program() -> ucc.Program:
    """Pre-compiled UCC program."""
    return ucc.compile(_CIRCUIT_TEXT)


# =============================================================================
# Compilation Benchmarks
# =============================================================================


def test_compile_stim(benchmark: Any) -> None:
    """Measure Stim circuit parsing and sampler compilation time."""

    def compile_stim() -> stim.CompiledDetectorSampler:
        circuit = stim.Circuit(_CIRCUIT_TEXT)
        return circuit.compile_detector_sampler()

    benchmark(compile_stim)


def test_compile_ucc(benchmark: Any) -> None:
    """Measure UCC circuit parsing and compilation time."""

    def compile_ucc() -> ucc.Program:
        return ucc.compile(_CIRCUIT_TEXT)

    benchmark(compile_ucc)


# =============================================================================
# Sampling Benchmarks
# =============================================================================


def test_sample_stim(benchmark: Any, stim_sampler: stim.CompiledDetectorSampler) -> None:
    """Measure Stim execution time for 100k shots."""

    def sample_stim() -> object:
        return stim_sampler.sample(_BENCHMARK_SHOTS, separate_observables=True)

    benchmark(sample_stim)


def test_sample_ucc(benchmark: Any, ucc_program: ucc.Program) -> None:
    """Measure UCC execution time for 100k shots."""

    def sample_ucc() -> object:
        return ucc.sample(ucc_program, _BENCHMARK_SHOTS, seed=0)

    benchmark(sample_ucc)
