"""Chain regional reductions while their quantum exits remain Clifford.

This is a bounded research host. Only the initial analysis is shared across
shots. Later analyses use the actual conditional state and run before entering
the unchanged Clifft executor.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import stim
from automatic_specialization import FaultModel, History
from regional_phase_specialization import RegionalPhase
from shared_phase_specialization import SharedPhase


@dataclass
class ChainResult:
    source: str
    steps: list[dict[str, Any]]
    stop_reason: str
    rejection: str | None = None


def t_count(source: str) -> int:
    return sum(line.startswith(("T ", "T_DAG ")) for line in source.splitlines())


class ChainedPhase:
    def __init__(self, source: str, *, max_regions: int = 8):
        if max_regions < 1:
            raise ValueError("The region budget must be positive")
        self.model = FaultModel(source)
        self.max_regions = max_regions
        self.initial_rejection = None
        self.first: RegionalPhase | None
        try:
            self.first = RegionalPhase(source)
        except ValueError as error:
            self.first = None
            self.initial_rejection = str(error)

    def rewrite(
        self, history: History, choose_prefix: Callable[[SharedPhase, int], int]
    ) -> ChainResult:
        if self.first is None:
            return ChainResult(
                self.model.render(history), [], "initial_fallback", self.initial_rejection
            )
        region = self.first
        pending_history = history
        steps: list[dict[str, Any]] = []
        old_records = old_values = 0
        analysis_seconds = 0.0
        while True:
            prefix, tail = region.split_history(pending_history)
            ideal_bits = choose_prefix(region.shared, len(steps))
            controls = region.shared.controls(prefix, ideal_bits)
            records = region.shared.prefix_records
            values = (controls >> 1) & ((1 << records) - 1)
            if records < old_records or values & ((1 << old_records) - 1) != old_values:
                raise AssertionError("Chaining changed an already observed preparation record")
            state = region.shared.render_state(controls)
            source = state + "\n".join(tail) + "\n"
            clifford_exit = not region.shared.nonclifford
            input_t = t_count("\n".join(region.model.lines[: region.boundary_line]))
            if input_t < 1 or (clifford_exit and t_count(state) != 0):
                raise AssertionError("Invalid constructive Clifford-exit certificate")
            if clifford_exit and t_count(source) >= t_count("\n".join(region.model.lines)):
                raise AssertionError("A certified chain step did not remove a phase region")
            steps.append(
                {
                    "input_t": input_t,
                    "residual_t": t_count(state),
                    "clifford_exit": clifford_exit,
                    "boundary_gate": region.boundary_gate,
                    "prefix_records": records,
                    "new_prefix_records": records - old_records,
                    "record_values": values,
                    "support_variables": region.shared.variables,
                    "entry_basis_sha256": hashlib.sha256(
                        "\n".join(map(str, region.shared.probe_rows)).encode()
                    ).hexdigest(),
                    "analysis_seconds": analysis_seconds,
                }
            )
            if not clifford_exit:
                return ChainResult(source, steps, "nonclifford_exit")
            if t_count(source) == 0:
                # The untouched continuation may contain a non-T rotation.
                # Absence of T syntax alone cannot certify Clifford completion.
                try:
                    stim.Circuit(source)
                except ValueError:
                    pass
                else:
                    return ChainResult(source, steps, "clifford_completion")
            if len(steps) >= self.max_regions:
                return ChainResult(source, steps, "region_budget")
            old_records, old_values = records, values
            started = perf_counter()
            try:
                region = RegionalPhase(source)
            except ValueError as error:
                return ChainResult(source, steps, "unsupported_next_region", str(error))
            analysis_seconds = perf_counter() - started
            # The entire fault history was drawn once at original locations.
            # The composed continuation already contains its realized Paulis.
            if region.model.sites:
                raise AssertionError("A later region would draw the physical noise twice")
            pending_history = ()

    def sample_history(self, history: History, rng: random.Random) -> ChainResult:
        def choose_prefix(shared: SharedPhase, _: int) -> int:
            # A later SharedPhase is rebuilt per shot. Reusing its constructor's
            # default sampler seed would repeat new outcomes and bias the law.
            sampler = shared.prefix.compile_sampler(seed=rng.getrandbits(64))
            sample = sampler.sample(1, bit_packed=True)[0]
            return int.from_bytes(sample.tobytes(), "little")

        return self.rewrite(history, choose_prefix)
