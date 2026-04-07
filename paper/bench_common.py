"""Shared benchmark infrastructure for paper benchmarks.

Provides runner classes (StimRunner, UccRunner, TsimRunner) and a
generic timing loop that individual benchmarks can reuse.
"""

from __future__ import annotations

import os
import time
from typing import Any, TypeAlias

# Must be set before JAX is imported (tsim depends on JAX).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import stim

# Type alias: circuits may be stim.Circuit objects or raw program text
# (e.g. for non-Clifford circuits that stim cannot parse).
CircuitLike: TypeAlias = stim.Circuit | str

# ---------------------------------------------------------------------------
# Per-simulator runners
# ---------------------------------------------------------------------------

_WARMUP_SHOTS = 64


class StimRunner:
    """Stim detector-sampler benchmark runner."""

    def compile(self, circuit: CircuitLike, shots: int) -> None:
        if isinstance(circuit, str):
            circuit = stim.Circuit(circuit)
        self._sampler = circuit.compile_detector_sampler()
        # Warmup: populate CPU caches / branch predictors.
        self._sampler.sample(min(_WARMUP_SHOTS, shots), separate_observables=True)

    def compile_metadata(self) -> dict[str, object]:
        return {}

    def sample(self, shots: int) -> None:
        self._sampler.sample(shots, separate_observables=True)


class UccRunner:
    """UCC benchmark runner.

    NOTE: ucc.sample() always materializes measurements, detectors, and
    observables, whereas Stim/tsim compile_detector_sampler() only produces
    detectors + observables.  For d=7 this is 385 vs 337 values per shot
    (~6% difference in Stim's own timing), so UCC's numbers are slightly
    conservative relative to the other backends.
    """

    def compile(self, circuit: CircuitLike, shots: int) -> None:
        import ucc

        self._prog = ucc.compile(
            circuit if isinstance(circuit, str) else str(circuit),
            hir_passes=ucc.default_hir_pass_manager(),
            bytecode_passes=ucc.default_bytecode_pass_manager(),
        )
        # Warmup: populate CPU caches / branch predictors.
        ucc.sample(self._prog, min(_WARMUP_SHOTS, shots))
        self._ucc = ucc

    def compile_metadata(self) -> dict[str, object]:
        k_hist = list(self._prog.active_k_history)
        return {"peak_active_k": max(k_hist) if k_hist else 0}

    def sample(self, shots: int) -> None:
        self._ucc.sample(self._prog, shots)


def _autotune_tsim_batch_size(sampler: Any, shots: int, separate_observables: bool = True) -> int:
    """Find the batch size that maximizes tsim throughput.

    Starts from a small batch and doubles until throughput stops improving
    or the batch no longer fits in device memory, mirroring the strategy
    described in the tsim paper.  As a side-effect, the winning batch size
    also serves as the JIT warmup.

    Raises on the first probe if the sampler fails for any reason (including
    OOM), since there is no valid batch size to fall back to.
    """
    batch = min(64, shots)
    best_throughput = 0.0
    best_batch = batch

    first = True
    while batch <= shots:
        try:
            t0 = time.perf_counter()
            sampler.sample(batch, separate_observables=separate_observables)
            elapsed = time.perf_counter() - t0
        except Exception:
            if first:
                # No successful batch yet — propagate so the caller sees
                # the real error (OOM, API mismatch, backend bug, etc.).
                raise
            # Later iterations may hit device memory limits; keep the
            # previous best.
            break

        first = False
        throughput = batch / elapsed
        if throughput > best_throughput:
            best_throughput = throughput
            best_batch = batch
        else:
            # Throughput stopped improving — done.
            break
        batch *= 2

    return best_batch


class TsimRunner:
    """Tsim benchmark runner with batch-size autotuning."""

    def __init__(
        self,
        batch_size: int | None = None,
    ) -> None:
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    def compile(self, circuit: CircuitLike, shots: int) -> None:
        # CPU vs GPU is controlled by setting JAX_PLATFORMS=cpu before
        # launching the script.  JAX reads this env var once at import time,
        # so it cannot be toggled within a single process.
        import tsim

        tc = tsim.Circuit(circuit if isinstance(circuit, str) else str(circuit))
        self._sampler = tc.compile_detector_sampler()

        if self._batch_size is not None:
            # User-specified batch size — just warm up the JIT.
            self._sampler.sample(min(self._batch_size, shots), separate_observables=True)
        else:
            # Autotune batch size (also warms up the JIT).
            self._batch_size = _autotune_tsim_batch_size(self._sampler, shots)

    def compile_metadata(self) -> dict[str, object]:
        total = sum(
            csg.num_graphs
            for comp in self._sampler._program.components
            for csg in comp.compiled_scalar_graphs
        )
        return {"tsim_num_graphs": total}

    def sample(self, shots: int) -> None:
        self._sampler.sample(shots, batch_size=self._batch_size, separate_observables=True)


RUNNERS: dict[str, type] = {
    "stim": StimRunner,
    "ucc": UccRunner,
    "tsim": TsimRunner,
}


# ---------------------------------------------------------------------------
# Generic benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark_loop(
    *,
    circuits: list[tuple[dict[str, object], CircuitLike]],
    simulators: list[str],
    shots: int,
    repeats: int,
    batch_size: int | None = None,
    label_key: str = "circuit",
) -> list[dict[str, object]]:
    """Run the compile-once / sample-many timing loop.

    Parameters
    ----------
    circuits:
        List of ``(metadata, circuit)`` pairs.  *metadata* is a dict of
        columns to include in every result row for that circuit (e.g.
        ``{"distance": 7, "rounds": 7, "phys_error_rate": 1e-3}``).
        The value of *label_key* in the metadata dict is used for
        progress output.
    simulators:
        Simulator names (keys of :data:`RUNNERS`).
    shots:
        Number of shots per timed sample call.
    repeats:
        How many timed sample repetitions per (circuit, simulator).
    batch_size:
        Optional fixed tsim batch size (skip autotuning).
    label_key:
        Which key in *metadata* to use for human-readable progress
        messages.  Defaults to ``"circuit"``.

    Returns
    -------
    List of result dicts suitable for ``pd.DataFrame(results)``.
    """
    results: list[dict[str, object]] = []
    total = len(circuits) * len(simulators) * repeats
    done = 0

    for metadata, circuit in circuits:
        label = str(metadata.get(label_key, ""))
        for sim in simulators:
            factory = RUNNERS.get(sim)
            if factory is None:
                print(f"  Unknown simulator '{sim}', skipping.")
                continue

            runner = factory(batch_size=batch_size) if factory is TsimRunner else factory()
            header = f"{label} {sim}"
            print(f"  {header}: compiling ...", end="", flush=True)
            try:
                t0 = time.perf_counter()
                runner.compile(circuit, shots)
                compile_s = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                print(f" ERROR ({type(exc).__name__}: {exc})")
                for rep in range(repeats):
                    done += 1
                    results.append(
                        {
                            **metadata,
                            "shots": shots,
                            "simulator": sim,
                            "repeat": rep,
                            "status": "ERROR",
                            "compile_s": "",
                            "sample_s": "",
                        }
                    )
                continue

            compile_meta = runner.compile_metadata()
            batch_info = ""
            if isinstance(runner, TsimRunner) and runner.batch_size is not None:
                batch_info = f", batch_size={runner.batch_size}"
            meta_info = "".join(f", {k}={v}" for k, v in compile_meta.items())
            print(f" {compile_s * 1e3:.1f}ms{batch_info}{meta_info}")

            for rep in range(repeats):
                done += 1
                tag = f"[{done}/{total}] {header} (rep {rep + 1}/{repeats})"
                print(f"  {tag} -> ", end="", flush=True)

                row: dict[str, object] = {
                    **metadata,
                    **compile_meta,
                    "shots": shots,
                    "simulator": sim,
                    "repeat": rep,
                }

                try:
                    t1 = time.perf_counter()
                    runner.sample(shots)
                    sample_s = time.perf_counter() - t1

                    row["status"] = "SUCCESS"
                    row["compile_s"] = round(compile_s, 6)
                    row["sample_s"] = round(sample_s, 6)
                    print(f"SUCCESS ({sample_s:.3f}s)")
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "ERROR"
                    row["compile_s"] = round(compile_s, 6)
                    row["sample_s"] = ""
                    print(f"ERROR ({type(exc).__name__}: {exc})")

                results.append(row)

    return results
