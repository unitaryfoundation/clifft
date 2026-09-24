"""Developer-facing access to Clifft's optional AMD HIP sampling backend."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, cast

from clifft import (
    _DEFAULT_PASSES,
    HirModule,
    HirPassManager,
    SampleResult,
    _DefaultPasses,
    _prepare_hir_for_lowering,
    default_hir_pass_manager,
)

Precision = Literal["fp64", "fp32"]

_NATIVE_NAME = "clifft._clifft_hip"
_NATIVE_SPEC = importlib.util.find_spec(_NATIVE_NAME)
_native: ModuleType | None = None
_native_error: ImportError | None = None
if _NATIVE_SPEC is not None:
    try:
        _native = importlib.import_module(_NATIVE_NAME)
    except ImportError as error:
        _native_error = error


def is_built() -> bool:
    """Return whether this Clifft installation contains the HIP extension."""
    return _NATIVE_SPEC is not None


def is_available() -> bool:
    """Return whether the extension loaded and can see an AMD GPU."""
    return _native is not None and bool(_native.is_available())


def backend_info() -> str:
    """Describe the optional extension and any devices visible to HIP."""
    if _NATIVE_SPEC is None:
        return "HIP backend not built; rebuild Clifft with CLIFFT_ENABLE_HIP=ON"
    if _native is None:
        return f"HIP extension failed to load: {_native_error}"
    return cast(str, _native.backend_info())


def _require_native() -> ModuleType:
    if _native is None:
        raise RuntimeError(backend_info())
    return _native


def _precision_value(precision: Precision) -> Any:
    native = _require_native()
    if precision == "fp64":
        return native.CoefficientPrecision.FP64
    if precision == "fp32":
        return native.CoefficientPrecision.FP32
    raise ValueError("precision must be 'fp64' or 'fp32'")


class Program:
    """An immutable, backend-private lowering of a SamplingPlan."""

    __slots__ = ("_native",)

    def __init__(self, native: Any) -> None:
        self._native = native

    @property
    def peak_active_width(self) -> int:
        return cast(int, self._native.peak_active_width)

    @property
    def num_actions(self) -> int:
        return cast(int, self._native.num_actions)

    @property
    def num_measurements(self) -> int:
        return cast(int, self._native.num_measurements)

    @property
    def num_records(self) -> int:
        """Return the visible plus hidden record width required by replay."""
        return cast(int, self._native.num_records)

    @property
    def num_detectors(self) -> int:
        return cast(int, self._native.num_detectors)

    @property
    def num_observables(self) -> int:
        return cast(int, self._native.num_observables)

    @property
    def num_exp_vals(self) -> int:
        return cast(int, self._native.num_exp_vals)

    @property
    def has_postselection(self) -> bool:
        return cast(bool, self._native.has_postselection)

    @property
    def packed_bytes(self) -> int:
        return cast(int, self._native.packed_bytes)

    def inspect(self) -> str:
        """Return diagnostic text for the packed executable."""
        return cast(str, self._native.inspect())

    def __repr__(self) -> str:
        return (
            f"Program({self.num_actions} actions, " f"peak_active_width={self.peak_active_width})"
        )


def lower(
    hir: HirModule,
    postselection_mask: list[int] | None = None,
    expected_detectors: list[int] | None = None,
    expected_observables: list[int] | None = None,
) -> Program:
    """Lower optimized HIR into the experimental HIP executable format."""
    native = _require_native()
    return Program(
        native.lower(
            hir,
            postselection_mask if postselection_mask is not None else [],
            expected_detectors if expected_detectors is not None else [],
            expected_observables if expected_observables is not None else [],
        )
    )


def compile(
    stim_text: str,
    postselection_mask: list[int] | None = None,
    expected_detectors: list[int] | None = None,
    expected_observables: list[int] | None = None,
    normalize_syndromes: bool = False,
    hir_passes: HirPassManager | None | _DefaultPasses = _DEFAULT_PASSES,
) -> Program:
    """Compile Stim text through the shared HIR and SamplingPlan pipeline."""
    if isinstance(hir_passes, _DefaultPasses):
        hir_passes = default_hir_pass_manager()
    prepared = _prepare_hir_for_lowering(
        stim_text,
        expected_detectors if expected_detectors is not None else [],
        expected_observables if expected_observables is not None else [],
        normalize_syndromes,
        hir_passes,
    )
    hir = cast(HirModule, prepared[0])
    detectors = cast(list[int], prepared[1])
    observables = cast(list[int], prepared[2])
    return lower(hir, postselection_mask, detectors, observables)


@dataclass(frozen=True)
class ReplayResult:
    """Result of one forced-record path through the HIP interpreter."""

    reachable: bool
    survived: bool
    log_probability: float
    outputs: SampleResult


Tier = Literal["auto", "thread_per_shot", "block_shared", "block_global"]

_TIER_NAMES: dict[str, str] = {
    "auto": "Auto",
    "thread_per_shot": "ThreadPerShot",
    "block_shared": "BlockShared",
    "block_global": "BlockGlobal",
}


def _tier_value(tier: Tier) -> object:
    native = _require_native()
    try:
        return getattr(native.ExecutionTier, _TIER_NAMES[tier])
    except KeyError:
        raise ValueError(f"tier must be one of {sorted(_TIER_NAMES)}; got {tier!r}") from None


def _tier_name(value: object) -> Tier:
    native = _require_native()
    for name, attribute in _TIER_NAMES.items():
        if value == getattr(native.ExecutionTier, attribute):
            return cast(Tier, name)
    raise ValueError(f"unrecognised execution tier {value!r}")


def selected_tier(program: Program, precision: Precision = "fp64") -> Tier:
    """Return the automatic tier for the current device and precision."""
    native = _require_native()
    return _tier_name(native.selected_tier(program._native, _precision_value(precision)))


class Sampler:
    """A synchronous sampler with one uploaded program and retained workspace.

    Calls on one instance must not overlap. Use a separate sampler per caller.
    """

    __slots__ = ("_native", "program")

    def __init__(
        self,
        program: Program,
        *,
        precision: Precision = "fp64",
        max_batch_shots: int | None = None,
        tier: Tier = "auto",
    ) -> None:
        native = _require_native()
        self.program = program
        native_precision = _precision_value(precision)
        native_tier = _tier_value(tier)
        if max_batch_shots is None:
            self._native = native.Sampler(program._native, native_precision, tier=native_tier)
        else:
            self._native = native.Sampler(
                program._native, native_precision, max_batch_shots, native_tier
            )

    @property
    def precision(self) -> Precision:
        native = _require_native()
        if self._native.coefficient_precision == native.CoefficientPrecision.FP32:
            return "fp32"
        return "fp64"

    @property
    def tier(self) -> Tier:
        """Return the resolved execution tier; never ``"auto"``."""
        return _tier_name(self._native.execution_tier)

    @property
    def max_batch_shots(self) -> int:
        return cast(int, self._native.max_batch_shots)

    @property
    def allocated_device_bytes(self) -> int:
        return cast(int, self._native.allocated_device_bytes)

    def sample(
        self,
        shots: int,
        *,
        seed: int | None = None,
        block_size: int | None = None,
    ) -> SampleResult:
        """Sample fixed rows while reusing the retained device workspace."""
        if block_size is None:
            return cast(SampleResult, self._native.sample(shots, seed))
        return cast(SampleResult, self._native.sample(shots, seed, block_size))

    def sample_survivors(
        self,
        shots: int,
        *,
        keep_records: bool = False,
        seed: int | None = None,
        block_size: int | None = None,
    ) -> SampleResult:
        """Sample and retain only shots that pass postselection."""
        if block_size is None:
            return cast(
                SampleResult,
                self._native.sample_survivors(shots, keep_records, seed),
            )
        return cast(
            SampleResult,
            self._native.sample_survivors(shots, keep_records, seed, block_size),
        )

    def replay_shot(self, forced_records: list[int]) -> ReplayResult:
        """Force every record value to probe one measurement branch exactly."""
        result = self._native.replay_shot(forced_records)
        return ReplayResult(
            reachable=cast(bool, result["reachable"]),
            survived=cast(bool, result["survived"]),
            log_probability=cast(float, result["log_probability"]),
            outputs=cast(SampleResult, result["outputs"]),
        )


__all__ = [
    "Precision",
    "Program",
    "ReplayResult",
    "Sampler",
    "Tier",
    "backend_info",
    "compile",
    "is_available",
    "is_built",
    "lower",
    "selected_tier",
]
