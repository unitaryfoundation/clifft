"""Expose a larger phase region by commuting disjoint work before measurements.

Measurements remain real quantum operations with their original record order.
The scheduler uses wire and classical dependencies, not protocol annotations.
All scheduling and specialization occur in the research host before execution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from automatic_specialization import (
    ANNOTATIONS,
    MEASUREMENTS,
    FaultModel,
    History,
    instruction_text,
    record_count,
)
from regional_phase_specialization import PAULI_NOISE, PHASE_GATES, RegionalPhase

import clifft


@dataclass
class Block:
    node: Any
    index: int
    sites: tuple[int, ...]
    readout_probability: float | None = None

    @property
    def wires(self) -> set[int]:
        if self.node.gate.name in ANNOTATIONS | {"MPAD"}:
            return set()
        return {t.value for t in self.node.targets if not t.is_rec}


class DeferredPhase:
    def __init__(self, source: str):
        self.model = FaultModel(source)
        blocks: list[Block] = []
        site = 0
        for index, node in enumerate(clifft.parse(source).nodes):
            gate = node.gate.name
            if gate == "READOUT_NOISE":
                blocks[-1].readout_probability = node.arg
                blocks[-1].sites += (site,)
                site += 1
                continue
            count = (
                len(node.targets) // (2 if gate.endswith("2") else 1) if gate in PAULI_NOISE else 0
            )
            blocks.append(Block(node, index, tuple(range(site, site + count))))
            site += count
        if site != len(self.model.sites):
            raise AssertionError("Scheduler lost a categorical fault location")

        first = next(
            (i for i, block in enumerate(blocks) if block.node.gate.name in {"T", "T_DAG"}),
            len(blocks),
        )
        active = blocks[:first]
        deferred: list[Block] = []
        measured: set[int] = set()
        available_records = sum(record_count(block.node) for block in active)
        moved = moved_t = crossings = 0
        stop = len(blocks)
        for i in range(first, len(blocks)):
            block = blocks[i]
            gate = block.node.gate.name
            if gate in MEASUREMENTS:
                deferred.append(block)
                measured |= block.wires
            elif gate in ANNOTATIONS:
                (deferred if deferred else active).append(block)
            elif gate not in PHASE_GATES | PAULI_NOISE:
                stop = i
                break
            elif block.wires & measured or any(
                t.is_rec and t.value >= available_records for t in block.node.targets
            ):
                stop = i
                break
            else:
                active.append(block)
                if deferred:
                    moved += 1
                    moved_t += gate in {"T", "T_DAG"}
                    crossings += sum(record_count(b.node) for b in deferred)
        ordered = active + deferred + blocks[stop:]
        if sorted(b.index for b in ordered) != [b.index for b in blocks]:
            raise AssertionError("Scheduling duplicated or dropped an instruction")

        records = 0
        lines = []
        site_order: list[int] = []
        for block in ordered:
            # Targets in the parsed AST are absolute record indices. Re-render
            # against the new position so earlier-prefix feedback stays valid.
            line = instruction_text(block.node, records)
            if block.readout_probability is not None:
                gate, targets = line.split(" ", 1)
                line = f"{gate}({block.readout_probability:.17g}) {targets}"
            lines.append(line)
            records += record_count(block.node)
            site_order.extend(block.sites)
        self.scheduled_source = "\n".join(lines) + "\n"
        self.site_map = {old: new for new, old in enumerate(site_order)}
        self.region = RegionalPhase(self.scheduled_source)
        for new, old in enumerate(site_order):
            a, b = self.model.sites[old], self.region.model.sites[new]
            if (a.gate, a.probabilities, a.replacements) != (
                b.gate,
                b.probabilities,
                b.replacements,
            ):
                raise AssertionError("Scheduling changed the fault law or its physical action")
        if records != self.model.num_records or len(self.site_map) != len(self.model.sites):
            raise AssertionError("Scheduling changed the record or fault count")
        self.audit = {
            "moved_operations": moved,
            "moved_t": moved_t,
            "measurement_crossings": crossings,
            "deferred_records": sum(record_count(b.node) for b in deferred),
            "stop_gate": blocks[stop].node.gate.name if stop < len(blocks) else "END",
            "permutation_sha256": hashlib.sha256(
                str([b.index for b in ordered]).encode()
            ).hexdigest(),
            "scheduled_source_sha256": hashlib.sha256(self.scheduled_source.encode()).hexdigest(),
        }

    def map_history(self, history: History) -> History:
        # Validate against the original locations before applying the bijection.
        self.model.render(history)
        return tuple(sorted((self.site_map[site], outcome) for site, outcome in history))

    def compose(self, history: History, prefix_bits: int | None = None) -> str:
        return self.region.compose(self.map_history(history), prefix_bits)

    def metadata(self) -> dict[str, Any]:
        return {**self.region.metadata(), **self.audit}
