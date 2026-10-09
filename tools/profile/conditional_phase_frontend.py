"""Route raw circuits through the existing bounded conditional reductions.

This research entry point discovers regions from circuit dependencies. It
shares the first analysis, chains certified Clifford exits, and optionally
transports one surviving rotation. All work precedes native execution.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import stim
from automatic_specialization import FaultModel, History
from chained_phase_specialization import t_count
from deferred_phase_specialization import DeferredPhase
from one_core_phase_specialization import transport
from shared_phase_specialization import SharedPhase


@dataclass
class ConditionalResult:
    source: str
    steps: list[dict[str, Any]]
    stop_reason: str
    carrier: dict[str, Any] | None = None
    rejection: str | None = None


class ConditionalPhase:
    def __init__(self, source: str, *, max_regions: int = 8):
        if max_regions < 1:
            raise ValueError("The region budget must be positive")
        self.model = FaultModel(source)
        self.max_regions = max_regions
        self.rejection = None
        self.first: DeferredPhase | None
        try:
            self.first = DeferredPhase(source)
        except ValueError as error:
            self.first = None
            self.rejection = str(error)

    def rewrite(
        self,
        history: History,
        seed: int,
        *,
        choose_prefix: Callable[[SharedPhase, int], int] | None = None,
        decisions: tuple[int, ...] | None = None,
    ) -> ConditionalResult:
        if self.first is None:
            return ConditionalResult(
                self.model.render(history), [], "initial_fallback", rejection=self.rejection
            )
        rng = random.Random(seed)

        def sample_prefix(shared: SharedPhase, _: int) -> int:
            sampler = shared.prefix.compile_sampler(seed=rng.getrandbits(64))
            return int.from_bytes(sampler.sample(1, bit_packed=True)[0].tobytes(), "little")

        choose = choose_prefix or sample_prefix
        scheduled = self.first
        pending_history = history
        steps: list[dict[str, Any]] = []
        old_records = old_values = 0
        while True:
            region = scheduled.region
            prefix, tail = region.split_history(scheduled.map_history(pending_history))
            ideal_bits = choose(region.shared, len(steps))
            controls = region.shared.controls(prefix, ideal_bits)
            records = region.shared.prefix_records
            values = (controls >> 1) & ((1 << records) - 1)
            if records < old_records or values & ((1 << old_records) - 1) != old_values:
                raise AssertionError("A later reduction changed observed records")
            state = region.shared.render_state(controls)
            source = state + "\n".join(tail) + "\n"
            residual = t_count(state)
            steps.append(
                {
                    "residual_t": residual,
                    "support_variables": region.shared.variables,
                    "boundary_gate": region.boundary_gate,
                    "prefix_records": records,
                    "record_values": values,
                    "moved_operations": scheduled.audit["moved_operations"],
                }
            )
            if residual == 1:
                # Keep the complete continuation for the existing optimizer
                # without expanding it through another polynomial synthesis.
                carried = transport(source, rng.getrandbits(64), decisions=decisions)
                return ConditionalResult(
                    source=carried.source,
                    steps=steps,
                    stop_reason="one_rotation",
                    carrier=carried.metadata,
                )
            if residual:
                return ConditionalResult(source, steps, "residual_magic")
            if t_count(source) == 0:
                try:
                    stim.Circuit(source)
                except ValueError:
                    return ConditionalResult(source, steps, "unsupported_continuation")
                return ConditionalResult(source, steps, "clifford_completion")
            if len(steps) >= self.max_regions:
                return ConditionalResult(source, steps, "region_budget")
            old_records, old_values = records, values
            try:
                scheduled = DeferredPhase(source)
            except ValueError as error:
                # Sampling has already happened. Retain the conditional state
                # instead of returning an unconditioned original circuit.
                return ConditionalResult(
                    source, steps, "next_region_fallback", rejection=str(error)
                )
            if scheduled.model.sites:
                raise AssertionError("A later region would redraw physical faults")
            pending_history = ()
