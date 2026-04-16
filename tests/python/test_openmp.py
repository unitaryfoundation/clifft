"""Tests for the OpenMP threading Python API.

The high-rank correctness and determinism tests (active_k >= 18) have
been ported to C++ (tests/test_openmp.cc) where they run under Thread
Sanitizer in the nightly CI.  The C++ tests use a QV-20 fixture that
sustains k >= 18 for most of the execution, providing better OpenMP
path coverage.

This file retains only the Python-specific API tests for
set_num_threads / get_num_threads.
"""

import clifft


class TestThreadingAPI:
    """Tests for set_num_threads / get_num_threads Python API."""

    def test_get_num_threads_returns_positive(self) -> None:
        n = clifft.get_num_threads()
        assert isinstance(n, int)
        assert n >= 1

    def test_set_and_get_roundtrip(self) -> None:
        original = clifft.get_num_threads()
        if original == 1:
            # No OpenMP: set_num_threads is a documented no-op, skip roundtrip.
            clifft.set_num_threads(4)
            assert clifft.get_num_threads() == 1
            return
        try:
            clifft.set_num_threads(1)
            assert clifft.get_num_threads() == 1
            clifft.set_num_threads(2)
            assert clifft.get_num_threads() == 2
        finally:
            clifft.set_num_threads(original)
