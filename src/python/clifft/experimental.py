"""Compatibility aliases for APIs that now use the symbolic backend by default."""

from clifft import (
    Circuit as Circuit,
)
from clifft import (
    MeasurementRecords,
    Program,
    basis_probabilities,
    compile,
    get_statevector,
    record_probabilities,
    sample,
    sample_k,
    sample_k_survivors,
    sample_survivors,
)
from clifft import noncomp as noncomp
from clifft._sample_result import SampleResult as SampleResult
from clifft.noncomp import sample as sample_noncomputational

__all__ = [
    "MeasurementRecords",
    "Program",
    "basis_probabilities",
    "compile",
    "get_statevector",
    "record_probabilities",
    "sample",
    "sample_k",
    "sample_k_survivors",
    "sample_noncomputational",
    "sample_survivors",
]
