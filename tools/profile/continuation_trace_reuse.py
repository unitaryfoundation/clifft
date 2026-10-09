"""Precompute fixed-continuation Pauli responses before native execution."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from automatic_specialization import History, record_count
from prefix_trace_reuse import PreparedPhase, TraceWorker
from shared_phase_specialization import SharedPhase, bits, parity

import clifft


@dataclass(frozen=True)
class ContinuationShot:
    correction: str
    active: tuple[int, ...]
    flips: tuple[int, ...]


class ContinuationPhase(PreparedPhase):
    def __init__(self, source: str, binary: Path):
        super().__init__(source, binary)
        self.continuation_info: dict[str, Any] = {"eligible": False}
        if not self.reuse_info["eligible"] or not self.reuse_info["clifford_tail"]:
            self.continuation_info["reason"] = self.reuse_info.get(
                "reason", "nonclifford_continuation"
            )
            return
        assert self.first is not None
        region, shared = self.first.region, self.first.region.shared
        lines = [f"I {self.model.num_qubits - 1}"]
        for line in shared.encoder:
            gate, *targets = line.split()
            lines.append(gate + " " + " ".join(str(self.physical[int(q)]) for q in targets))
        self.generator_count = 0

        def generator(axis: str, q: int) -> int:
            result = self.generator_count
            self.generator_count += 1
            # These setup-only probes expose the native rewound Pauli axes.
            # They are removed from HIR before any program can be executed.
            lines.append(f"E(.5) {axis}{q}")
            return result

        self.offset_controls = [
            (offset, generator("X", q)) for q, offset in enumerate(shared.exit_offsets) if offset
        ]
        self.prefix_outputs = []
        records = 0
        for output in shared.output:
            if isinstance(output, str):
                lines.append(output)
            elif isinstance(output, int):
                self.prefix_outputs.append((output, records))
                records += 1
                lines.append("MPAD 0")
            else:
                raise AssertionError("State-preserving output contains a deferred readout")
        assert records == shared.prefix_records
        line_records = {}
        records = 0
        for node in clifft.parse("\n".join(region.model.lines) + "\n").nodes:
            if record_count(node):
                if record_count(node) != 1:
                    raise ValueError("Continuation requires individual record-producing nodes")
                line_records[node.source_line - 1] = records
            records += record_count(node)
        by_line = {s.line: i for i, s in enumerate(region.model.sites)}
        self.tail_controls: dict[int, tuple[int, ...]] = {}
        self.readout_records: dict[int, int] = {}
        for line_index in range(region.boundary_line, len(region.model.lines)):
            site_id = by_line.get(line_index)
            line = region.model.lines[line_index]
            if site_id is None:
                lines.append(line)
                continue
            site = region.model.sites[site_id]
            if site.gate == "READOUT_NOISE":
                self.readout_records[site_id] = line_records[line_index]
                lines.append(line)
                continue
            elementary = set()
            for replacement in site.replacements:
                for term in replacement.splitlines():
                    axis, target = term.split()
                    if axis not in {"X", "Y", "Z"}:
                        raise ValueError("Continuation replacement must be a Pauli product")
                    for component in "XZ" if axis == "Y" else axis:
                        elementary.add((int(target), component))
            ids = {(q, axis): generator(axis, q) for q, axis in sorted(elementary)}
            outcomes = []
            for replacement in site.replacements:
                mask = 0
                for term in replacement.splitlines():
                    axis, target = term.split()
                    for component in "XZ" if axis == "Y" else axis:
                        mask ^= 1 << ids[int(target), component]
                outcomes.append(mask)
            self.tail_controls[site_id] = tuple(outcomes)
        self.template_source = "\n".join(lines) + "\n"
        self.continuation_info.update(
            eligible=True,
            generators=self.generator_count,
            tail_fault_sites=len(self.tail_controls),
            readout_sites=len(self.readout_records),
            template_lines=len(lines),
            prefix_records=shared.prefix_records,
        )

    def payload(
        self,
        history: History,
        seed: int,
        *,
        choose_prefix: Callable[[SharedPhase, int], int] | None = None,
    ) -> ContinuationShot:
        if not self.continuation_info["eligible"]:
            raise ValueError("This circuit retains the preceding frontend route")
        assert self.first is not None
        region, shared = self.first.region, self.first.region.shared
        prefix = []
        active = flipped = 0
        seen = set()
        for site, outcome in history:
            if site in seen or not 0 <= site < len(self.model.sites):
                raise ValueError("Invalid or repeated categorical fault site")
            seen.add(site)
            if not 0 < outcome < len(self.model.sites[site].replacements):
                raise ValueError("Invalid categorical fault outcome")
            mapped = self.first.site_map[site]
            if mapped < region.region_sites:
                prefix.append((mapped, outcome))
            elif mapped in self.readout_records:
                flipped ^= 1 << self.readout_records[mapped]
            else:
                active ^= self.tail_controls[mapped][outcome]
        if choose_prefix is None:
            sampler = shared.prefix.compile_sampler(seed=random.Random(seed).getrandbits(64))
            ideal_bits = int.from_bytes(sampler.sample(1, bit_packed=True)[0].tobytes(), "little")
        else:
            ideal_bits = choose_prefix(shared, 0)
        controls = shared.controls(tuple(prefix), ideal_bits)
        for mask, index in self.offset_controls:
            if parity(mask, controls):
                active ^= 1 << index
        for output, index in self.prefix_outputs:
            if (controls >> (1 + output)) & 1:
                flipped ^= 1 << index
        end = -len(shared.encoder) if shared.encoder else None
        lines = []
        for line in shared.quantum_lines(controls)[self.fixed_gate_count : end]:
            gate, *targets = line.split()
            lines.append(gate + " " + " ".join(str(self.physical[int(q)]) for q in targets))
        return ContinuationShot("\n".join(lines), tuple(bits(active)), tuple(bits(flipped)))


class ContinuationWorker(TraceWorker):
    def instantiate(
        self,
        shot: ContinuationShot,
        seed: int,
        *,
        reference: str | None = None,
        mode: str = "continuation",
    ) -> dict[str, Any]:
        if mode not in {
            "continuation",
            "diagonal",
            "audit",
            "squeeze",
            "coordinate-audit",
            "coordinate-native",
            "coordinate-identity",
            "coordinate-columns",
            "coordinate-inverse32",
        }:
            raise ValueError("Unknown continuation construction mode")
        start = perf_counter()
        lines = shot.correction.splitlines()
        request = f"{mode} {seed} {len(lines)} {int(reference is not None)}\n"
        request += "".join(line + "\n" for line in lines)
        request += " ".join(map(str, shot.active)) + "\n"
        request += " ".join(map(str, shot.flips)) + "\n"
        if reference is not None:
            original = reference.splitlines()
            request += str(len(original)) + "\n" + "".join(line + "\n" for line in original)
        assert self.process.stdin is not None
        self.process.stdin.write(request)
        self.process.stdin.flush()
        result = self.read()
        result["roundtrip_seconds"] = perf_counter() - start
        result["request_bytes"] = len(request.encode())
        return result
