"""Reuse an optimized fixed preparation before trajectory-specific Cliffords."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import stim
from automatic_specialization import History
from conditional_phase_frontend import ConditionalPhase, ConditionalResult
from one_core_phase_specialization import rotation_lines
from shared_phase_specialization import SharedPhase

import clifft


def optimize_preparation(source: str, binary: Path) -> tuple[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="clifft-prefix-") as directory:
        path = Path(directory) / "prefix.stim"
        path.write_text(source)
        exported = json.loads(
            subprocess.check_output([str(binary.resolve()), str(path)], text=True)
        )
    frame = stim.Tableau.from_conjugated_generators(
        xs=[stim.PauliString(p) for p in exported["xs"]],
        zs=[stim.PauliString(p) for p in exported["zs"]],
    )
    lines = [f"I {exported['qubits'] - 1}"]
    for rotation in exported["rotations"]:
        gate, terms = rotation.split()
        axis = stim.PauliString(exported["qubits"])
        for term in terms.split("*"):
            axis[int(term[1:])] = term[0]
        # Primitive Clifford+T syntax also lets the independent small-state
        # oracle consume the result without a new Pauli-rotation converter.
        lines += rotation_lines(axis, -1 if gate == "TPP_DAG" else 1)
    lines += str(frame.to_circuit()).splitlines()
    return "\n".join(lines) + "\n", {
        "input_t": exported["input_t"],
        "output_t": exported["output_t"],
        "qubits": exported["qubits"],
        "exported_lines": len(lines),
    }


class ReusablePhase(ConditionalPhase):
    def __init__(self, source: str, binary: Path):
        super().__init__(source)
        self.raw_prefix = self.optimized_prefix = ""
        self.reuse_info: dict[str, Any] = {"eligible": False}
        if self.first is None:
            self.reuse_info["reason"] = "no_initial_region"
            return
        region = self.first.region
        shared = region.shared
        if shared.metadata()["residual_t"] <= 1:
            self.reuse_info["reason"] = "retain_existing_clifford_or_carrier_route"
            return
        # Conditional corrections, encoder and Pauli faults are Clifford. A
        # Clifford-only continuation cannot introduce a new phase region that
        # would require the omitted full-circuit phase pass.
        clifford_tail = True
        try:
            stim.Circuit("\n".join(region.model.lines[region.boundary_line :]))
        except ValueError:
            clifford_tail = False
        physical = {compact: q for q, compact in shared.quantum_map.items()}
        lines = [f"I {shared.model.num_qubits - 1}"]
        for line in shared.preparation + shared.nonclifford:
            gate, *targets = line.split()
            lines.append(gate + " " + " ".join(str(physical[int(q)]) for q in targets))
        self.raw_prefix = "\n".join(lines) + "\n"
        start = perf_counter()
        self.optimized_prefix, metadata = optimize_preparation(self.raw_prefix, binary)
        self.reuse_info.update(
            eligible=True,
            clifford_tail=clifford_tail,
            preparation=metadata,
            export_seconds=perf_counter() - start,
        )

    @property
    def phase_required(self) -> bool:
        return not (self.reuse_info["eligible"] and self.reuse_info["clifford_tail"])

    def rewrite(
        self,
        history: History,
        seed: int,
        *,
        choose_prefix: Callable[[SharedPhase, int], int] | None = None,
        decisions: tuple[int, ...] | None = None,
    ) -> ConditionalResult:
        branch = super().rewrite(history, seed, choose_prefix=choose_prefix, decisions=decisions)
        if self.reuse_info["eligible"]:
            if branch.stop_reason != "residual_magic" or not branch.source.startswith(
                self.raw_prefix
            ):
                raise AssertionError(
                    "Reusable preparation does not match the conditional interface"
                )
            branch.source = self.optimized_prefix + branch.source[len(self.raw_prefix) :]
        return branch


def compile_stages(source: str, *, phase: bool = True) -> tuple[Any, dict[str, Any]]:
    start = perf_counter()
    circuit = clifft.parse(source)
    parsed = perf_counter()
    hir = clifft.trace(circuit)
    traced = perf_counter()
    times = {"parse": parsed - start, "trace": traced - parsed}
    counts = {"traced": hir.num_t_gates}
    passes: list[
        clifft.PeepholeFusionPass
        | clifft.PhasePolynomialPass
        | clifft.RotationSimplificationPass
        | clifft.StatevectorSqueezePass
    ] = [clifft.PeepholeFusionPass()]
    if phase:
        passes.append(clifft.PhasePolynomialPass())
    passes += [clifft.RotationSimplificationPass(), clifft.StatevectorSqueezePass()]
    for pass_ in passes:
        manager = clifft.HirPassManager()
        manager.add(pass_)
        before = perf_counter()
        manager.run(hir)
        times[type(pass_).__name__] = perf_counter() - before
        counts[type(pass_).__name__] = hir.num_t_gates
    before = perf_counter()
    width = clifft.active_width_trace(hir)["peak"]
    times["width"] = perf_counter() - before
    return hir, {"seconds": times, "t_counts": counts, "width": width}
