"""Compose a conditional phase reduction with an ordinary quantum continuation.

The first region starts at the first T gate after a Clifford preparation and
ends before an operation outside the phase representation. All construction
and specialization occur in the research host, outside the existing executor.
"""

from __future__ import annotations

from typing import Any

import stim
from automatic_specialization import (
    ANNOTATIONS,
    FaultModel,
    History,
    instruction_text,
    record_count,
)
from shared_phase_specialization import SharedPhase

import clifft

PHASE_GATES = {"CX", "CZ", "T", "T_DAG", "S", "S_DAG", "X", "Y", "Z", "I"}
PAULI_NOISE = {
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "PAULI_CHANNEL_1",
    "PAULI_CHANNEL_2",
}


def noisy_instruction_lines(source: str) -> list[str]:
    """Retain one entry per parsed node, merging split readout noise in place."""
    lines: list[str] = []
    records = 0
    for node in clifft.parse(source).nodes:
        if node.gate.name == "READOUT_NOISE":
            gate, targets = lines[-1].split(" ", 1)
            lines[-1] = f"{gate}({node.arg:.17g}) {targets}"
            lines.append("")
        else:
            lines.append(instruction_text(node, records))
        records += record_count(node)
    return lines


class RegionalPhase:
    def __init__(self, source: str):
        self.model = FaultModel(source)
        parsed = clifft.parse(source)
        first = None
        boundary = len(parsed.nodes)
        for index, node in enumerate(parsed.nodes):
            gate = node.gate.name
            if first is None:
                if gate == "READOUT_NOISE":
                    continue
                try:
                    stim.gate_data(gate)
                except IndexError:
                    if gate not in {"T", "T_DAG"}:
                        raise ValueError("Initial non-Clifford gate is outside the phase model")
                    first = index
            elif gate not in PHASE_GATES | PAULI_NOISE | ANNOTATIONS:
                boundary = index
                break
        if first is None:
            raise ValueError("No non-Clifford region")
        lines = noisy_instruction_lines(source)
        self.region_source = "\n".join(lines[:boundary]) + "\n"
        region_model = FaultModel(self.region_source)
        self.boundary_line = len(region_model.lines)
        self.region_sites = len(region_model.sites)
        # The continuation may use wires untouched by the selected prefix.
        # They are part of the zero-input interface and must remain available.
        self.shared = SharedPhase(
            self.region_source + f"I {self.model.num_qubits - 1}\n", preserve_state=True
        )
        if self.shared.model.sites != self.model.sites[: self.region_sites]:
            raise AssertionError("Splitting a region changed categorical fault locations")
        self.boundary_gate = (
            parsed.nodes[boundary].gate.name if boundary < len(parsed.nodes) else "END"
        )

    def split_history(self, history: History) -> tuple[History, list[str]]:
        prefix = []
        tail = self.model.lines[self.boundary_line :].copy()
        seen = set()
        for site_id, outcome in history:
            if site_id in seen or not 0 <= site_id < len(self.model.sites):
                raise ValueError("Invalid or repeated fault location")
            seen.add(site_id)
            site = self.model.sites[site_id]
            if not 0 < outcome < len(site.replacements):
                raise ValueError("Invalid categorical fault outcome")
            if site_id < self.region_sites:
                prefix.append((site_id, outcome))
            else:
                tail[site.line - self.boundary_line] = site.replacements[outcome]
        return tuple(prefix), tail

    def compose(self, history: History, prefix_bits: int | None = None) -> str:
        prefix, tail = self.split_history(history)
        controls = self.shared.controls(prefix, prefix_bits)
        return self.shared.render_state(controls) + "\n".join(tail) + "\n"

    def metadata(self) -> dict[str, Any]:
        return {
            **self.shared.metadata(),
            "boundary_gate": self.boundary_gate,
            "boundary_line": self.boundary_line,
            "region_noise_sites": self.region_sites,
            "continuation_noise_sites": len(self.model.sites) - self.region_sites,
            "continuation_records": self.model.num_records - self.shared.prefix_records,
        }
